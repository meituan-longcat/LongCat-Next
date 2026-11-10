# Copyright (c) 2026 LongCat dNaViT Team.
# SPDX-License-Identifier: Apache-2.0

"""Residual modulation module for the visual encoder.

Architecture (config-controlled):

  taps from layer_idx_list (e.g. [1, 8, 16, 24])
      |                            (layer 1 = post-patch-embed feature;
      v                             layer N = output of trainable block N-1)
  [LayerNorm each tap]
      |
  concat along channel
      |
  Linear(len(taps) * hidden, hidden)        # fuser
      |
  N transformer blocks (PlainAttentionFlash + GeGluMlp)
      |
  LayerNorm
      |
  Linear(hidden, hidden * 2)                # sft_proj, ZERO INITIALIZED
      |
  split -> (scale, shift)
      |
  SFT: trainable_out * (1 + scale) + shift  -> returned to caller

The zero-initialized sft_proj makes the modulation an exact identity at
init, so an existing tokenizer checkpoint can be warm-started with ResMod
enabled without immediate divergence; the modulation strength is learned
gradually as sft_proj moves off zero.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

def _flash_attn_available() -> bool:
    try:
        from flash_attn import flash_attn_varlen_func  # noqa: F401
        return True
    except ImportError:
        return False

_FLASH_ATTN_AVAILABLE = _flash_attn_available()



class GeGluMlp(nn.Module):
    """LayerNorm -> GeGLU FFN -> residual-shaped output. Same channel dim in/out."""
    def __init__(self, in_features, hidden_features, drop=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(in_features, eps=1e-6)
        self.act = nn.GELU(approximate='tanh')
        self.w0 = nn.Linear(in_features, hidden_features)
        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        x = self.norm(x)
        x = self.act(self.w0(x)) * self.w1(x)
        return self.w2(x)


class PlainAttentionFlash(nn.Module):
    """Packed variable-length attention (adapted from ``strong_vq.py``'s
    ``PlainAttention``).

    Operates only on packed ``(seq_total, in_dim)`` tokens with an explicit
    ``cu_seqlens=(B+1,)`` int32 — fixed resolution must be packed by the caller
    (zero padding). Uses ``flash_attn_varlen_func`` when available, else falls
    back to ``F.scaled_dot_product_attention`` with per-image splits.

    Dim-mismatch handling mirrors the original ``PlainAttention``:
      - in_dim > out_dim: qkv lives in ``in_dim``; per-head output is averaged
        across heads (then optional adaptive_pool to ``out_dim``).
      - in_dim <= out_dim: qkv lives in ``out_dim``; standard reshape to
        ``(seq, num_heads*head_dim) = (seq, out_dim)``.
    """

    def __init__(self, in_dim, out_dim, num_heads):
        super().__init__()
        if in_dim > out_dim:
            qkv_dim = in_dim
        else:
            qkv_dim = out_dim
        if qkv_dim % num_heads != 0:
            raise ValueError(
                f"PlainAttentionFlash: qkv_dim={qkv_dim} not divisible by num_heads={num_heads}"
            )
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.qkv_dim = qkv_dim
        self.head_dim = qkv_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Qwen3ViT-style: single qkv Linear without bias + separate q/v biases
        # (k bias forced to zero). Lets us swap in flash_attn_varlen_func 1:1.
        self.qkv = nn.Linear(in_dim, qkv_dim * 3, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(qkv_dim))
        self.v_bias = nn.Parameter(torch.zeros(qkv_dim))
        self.register_buffer("zero_k_bias", torch.zeros(qkv_dim), persistent=False)

        self.proj = nn.Linear(out_dim, out_dim)
        self._use_flash = _FLASH_ATTN_AVAILABLE

    def _attn_varlen_sdpa(self, q, k, v, cu_seqlens):
        # SDPA fallback for variable-length: split per image, run SDPA, concat.
        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        outs = []
        offset = 0
        for L_i in lengths:
            q_i = q[offset:offset + L_i].transpose(0, 1).unsqueeze(0)    # (1, h, L_i, hd)
            k_i = k[offset:offset + L_i].transpose(0, 1).unsqueeze(0)
            v_i = v[offset:offset + L_i].transpose(0, 1).unsqueeze(0)
            out = F.scaled_dot_product_attention(q_i, k_i, v_i)
            outs.append(out.squeeze(0).transpose(0, 1))                  # (L_i, h, hd)
            offset += L_i
        return torch.cat(outs, dim=0)                                    # (seq_total, h, hd)

    def forward(self, x, cu_seqlens=None):
        """Packed varlen attention. ``x``: ``(seq_total, in_dim)``; ``cu_seqlens``:
        ``(B+1,)`` int32 (required). Fixed resolution must be pre-packed."""
        assert x.dim() == 2, \
            f"PlainAttentionFlash expects packed (seq_total, in_dim); got {x.shape}"
        assert cu_seqlens is not None, \
            "PlainAttentionFlash requires cu_seqlens for packed input."
        seq_total = x.shape[0]
        max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())

        qkv_bias = torch.cat((self.q_bias, self.zero_k_bias, self.v_bias))
        qkv = F.linear(x, self.qkv.weight, qkv_bias)
        qkv = qkv.reshape(seq_total, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(1, 0, 2, 3).unbind(0)  # each (seq_total, heads, head_dim)

        if self._use_flash and x.is_cuda:
            from flash_attn import flash_attn_varlen_func
            attn_out = flash_attn_varlen_func(
                q, k, v,
                cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
                softmax_scale=self.scale, causal=False,
            )  # (seq_total, heads, head_dim)
        else:
            attn_out = self._attn_varlen_sdpa(q, k, v, cu_seqlens)

        # Per-head → final out_dim (matches original PlainAttention behavior).
        if self.in_dim > self.out_dim:
            head_avg = attn_out.mean(dim=1)                              # (seq_total, head_dim)
            if self.head_dim != self.out_dim:
                head_avg = F.adaptive_avg_pool1d(head_avg, self.out_dim) # (seq_total, out_dim)
            flat = head_avg
        else:
            flat = attn_out.reshape(seq_total, self.num_heads * self.head_dim)

        return self.proj(flat)                                          # (seq_total, out_dim)



class _ResModLayer(nn.Module):
    """One pre-norm transformer block, varlen-only.

    Uses ``PlainAttentionFlash`` for attention, so it operates on packed
    ``(seq_total, D)`` tokens + ``cu_seqlens``. Pre-norm + optional LayerScale + skip.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        init_values: Optional[float] = None,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = PlainAttentionFlash(in_dim=dim, out_dim=dim, num_heads=num_heads)
        if init_values is not None and init_values > 0:
            self.ls1 = nn.Parameter(init_values * torch.ones(dim))
            self.ls2 = nn.Parameter(init_values * torch.ones(dim))
        else:
            self.ls1 = None
            self.ls2 = None
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = GeGluMlp(in_features=dim, hidden_features=int(dim * mlp_ratio))

    def forward(self, x, cu_seqlens=None):
        h = self.attn(self.norm1(x), cu_seqlens=cu_seqlens)
        if self.ls1 is not None:
            h = h * self.ls1
        x = x + h
        h = self.mlp(self.norm2(x))
        if self.ls2 is not None:
            h = h * self.ls2
        return x + h


class ResMod(nn.Module):
    """Residual modulation: tap-fuse-transform-modulate.

    Args:
        hidden_size: feature dim (1152 for qwen3vit, 1152 for siglip2-so400m).
        layer_idx_list: which encoder layers to tap. Convention:
            ``1`` = features right after the (frozen) patch_embed + pos_embed,
            ``i`` (i>=2) = output of trainable block ``i - 1``.
            Order is preserved (so ``[1, 8, 16, 24]`` keeps the early-to-late
            ordering in the channel-concat fuser, deterministically).
        depth: number of transformer blocks after the fuser. Default 2.
        num_heads: attention head count.
        mlp_ratio: FFN expansion ratio.
        init_values: optional LayerScale init (``None`` = no LayerScale).
    """

    def __init__(
        self,
        hidden_size: int,
        layer_idx_list: Sequence[int],
        depth: int = 2,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        init_values: Optional[float] = None,
        output_dim: Optional[int] = None,
        input_merge_size: int = 1,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        # Optional "internal merger" between the transformer blocks and the
        # final SFT head. Activated by ``merger_after_quant=True``: the blocks
        # still run at hidden_size on the original T tokens (full-resolution
        # pixel reasoning), then the merger groups every ``input_merge_size``
        # (= ms**2) consecutive tokens and projects to ``output_dim`` so the SFT
        # lives in the Qwen merger-out space. Default (input_merge_size=1,
        # output_dim=hidden_size) -> nn.Identity(), exactly the original ResMod.
        self.output_dim = int(output_dim) if output_dim is not None else self.hidden_size
        self.input_merge_size = int(input_merge_size)
        if self.input_merge_size < 1:
            raise ValueError("input_merge_size must be >= 1.")
        # Preserve user-specified ordering but de-duplicate.
        seen = set()
        kept: List[int] = []
        for idx in layer_idx_list:
            i = int(idx)
            if i in seen:
                continue
            seen.add(i)
            kept.append(i)
        self.layer_idx_list = kept
        n_taps = len(kept)
        if n_taps == 0:
            raise ValueError("ResMod requires at least one tap layer.")

        # Per-tap LayerNorm so each level enters the fuser on a comparable scale.
        self.tap_lns = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(n_taps)])
        # Channel-concat fuser: (n_taps * hidden) -> hidden.
        self.fuser = nn.Linear(n_taps * hidden_size, hidden_size)

        self.blocks = nn.ModuleList([
            _ResModLayer(
                dim=hidden_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                init_values=init_values,
            )
            for _ in range(depth)
        ])
        # Internal merger (post-blocks, pre-final_ln). Bridges the blocks'
        # (hidden_size, T tokens) to the SFT's (output_dim, T/ms2 tokens) when
        # merger_after_quant is on; nn.Identity() otherwise.
        if self.input_merge_size > 1 or self.output_dim != self.hidden_size:
            in_dim = self.input_merge_size * self.hidden_size
            self.internal_merger = nn.Linear(in_dim, self.output_dim)
        else:
            self.internal_merger = nn.Identity()

        self.final_ln = nn.LayerNorm(self.output_dim)
        # Zero-initialised SFT head: scale=0, shift=0 -> identity at step 0.
        self.sft_proj = nn.Linear(self.output_dim, self.output_dim * 2)

        self._init_weights()

    @property
    def has_internal_merger(self) -> bool:
        return not isinstance(self.internal_merger, nn.Identity)

    def _init_weights(self) -> None:
        # Zero the SFT projection so the encoder is functionally unchanged at
        # init. After this the parent model's apply(_init_weights) may overwrite
        # these — callers should invoke reset_sft_proj() after model
        # construction to be safe.
        nn.init.zeros_(self.sft_proj.weight)
        nn.init.zeros_(self.sft_proj.bias)
        # Modest gain on the fuser linear to keep bf16 + flash_attn stable.
        nn.init.xavier_uniform_(self.fuser.weight, gain=1.0 / (2 ** 0.5))
        nn.init.zeros_(self.fuser.bias)

    def reset_sft_proj(self) -> None:
        """Re-zero the SFT head. Call after any global ``apply(_init_weights)``
        or after loading a checkpoint that does not contain ``sft_proj``."""
        nn.init.zeros_(self.sft_proj.weight)
        nn.init.zeros_(self.sft_proj.bias)

    @property
    def n_taps(self) -> int:
        return len(self.layer_idx_list)

    def forward(
        self,
        taps: List[torch.Tensor],
        target: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Modulate ``target`` with features fused from ``taps``.

        Args:
            taps: one tensor per entry of ``layer_idx_list``, all packed in the
                same layout as the encoder backbone hidden (``(seq_total, hidden_size)``).
            target: the feature to modulate. Shape MUST match the SFT's
                post-merger axes: ``(seq_total // input_merge_size, output_dim)``
                when the internal merger is active, else ``(seq_total, hidden_size)``.
            cu_seqlens: ``(B+1,)`` int32 segment boundaries on the BLOCKS' token
                count (pre-merger). Required; fixed resolution is packed into
                equal-length segments by the caller.

        Returns:
            Modulated tensor with the same shape as ``target``.
        """
        if len(taps) != self.n_taps:
            raise ValueError(
                f"ResMod expected {self.n_taps} taps for layer_idx_list="
                f"{self.layer_idx_list}, got {len(taps)}.")

        normalized = [ln(t) for ln, t in zip(self.tap_lns, taps)]
        x = torch.cat(normalized, dim=-1)        # (..., n_taps * hidden)
        x = self.fuser(x)                        # (..., hidden)
        for blk in self.blocks:
            x = blk(x, cu_seqlens=cu_seqlens)

        # Internal merger: group every ``input_merge_size`` consecutive tokens
        # (Qwen raster-scan within each ms x ms cell) and project to output_dim.
        if self.has_internal_merger:
            ms2 = self.input_merge_size
            if x.dim() == 2:
                T_in = x.shape[0]
                if T_in % ms2 != 0:
                    raise ValueError(
                        f"ResMod.internal_merger needs seq divisible by {ms2}; got {T_in}.")
                x = x.view(T_in // ms2, ms2 * self.hidden_size)
            else:
                B, L_in, _ = x.shape
                if L_in % ms2 != 0:
                    raise ValueError(
                        f"ResMod.internal_merger needs L divisible by {ms2}; got {L_in}.")
                x = x.view(B, L_in // ms2, ms2 * self.hidden_size)
            x = self.internal_merger(x)          # (..., output_dim)

        x = self.final_ln(x)
        stats = self.sft_proj(x)                 # (..., 2 * output_dim)
        scale, shift = stats.chunk(2, dim=-1)
        # At init scale = shift = 0 -> exact identity pass-through.
        return target * (1.0 + scale) + shift
