# Copyright (c) 2026 LongCat dNaViT Team.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from collections import OrderedDict
from timm.layers import Mlp
from einops import rearrange
from einops.layers.torch import Rearrange
from transformers import AutoModel
from .res_mod import ResMod


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)

    @torch.amp.autocast(device_type='cuda', enabled=False)
    def forward(self, x):
        input_dtype = x.dtype
        x = x.float()
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        if self.elementwise_affine:
            output = output * self.weight
        return output.to(input_dtype)


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features = None,
        out_features = None,
        drop = 0.0,
        bias = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        hidden = self.drop(hidden)  # Apply dropout after hidden and before w3
        return self.w3(hidden)

# weight init
def init_weights(module):
    if (isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d) or
     isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d)):
        module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
    elif isinstance(module, (nn.LayerNorm, RMSNorm)):
        if hasattr(module, 'bias') and module.bias is not None:
            module.bias.data.zero_()
        if module.weight is not None:
            module.weight.data.fill_(1.0)

# attention layer with KV cache supported
class Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer = nn.LayerNorm,
            rope: nn.Module = None,
            target_aware_rope: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = True

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.kv_cache = False
        self.k_cache = None
        self.v_cache = None
        self.rope = rope
        self.target_aware_rope = target_aware_rope
        if target_aware_rope:
            assert num_heads % 2 == 0, 'num_heads must be even for target_aware_rope'

    def reset_kv_cache(self):
        self.k_cache = None
        self.v_cache = None

    def forward(self, x: torch.Tensor, attn_mask=None, rope_order=None,
                target_rope_order=None, cu_seqlens=None) -> torch.Tensor:
        # Dynamic-resolution (varlen) path: x is (seq_total, C), tokens packed
        # across images of varying lengths, with cu_seqlens=(B+1,) int32.
        # RoPE / kv_cache are not supported on this path; they're not needed
        # for the dNaViT varlen decoder path.
        if cu_seqlens is not None:
            return self._forward_varlen(x, cu_seqlens)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        # Apply RoPE if provided
        # rope_order contains the position indices for each token
        if self.rope is not None:
            # q and k shape: (B, num_heads, N, head_dim)
            if self.target_aware_rope and target_rope_order is not None:
                # Split Q heads: first half uses current_location, second half uses target_location
                half_heads = self.num_heads // 2
                q_current = q[:, :half_heads]  # First half of heads
                q_target = q[:, half_heads:]   # Second half of heads

                # Apply RoPE with current location to first half
                q_current = self.rope(q_current, rope_order=rope_order)
                # Apply RoPE with target location to second half
                q_target = self.rope(q_target, rope_order=target_rope_order)

                # Concatenate back
                q = torch.cat([q_current, q_target], dim=1)

                # K always uses current location
                k = self.rope(k, rope_order=rope_order)
            else:
                # Standard RoPE: apply same rope_order to both q and k
                q = self.rope(q, rope_order=rope_order)
                k = self.rope(k, rope_order=rope_order)

        if self.kv_cache:
            if self.k_cache is None and self.v_cache is None:
                k_cache = k
                v_cache = v
            else:
                assert N in [1, 2], f"x.shape {x.shape}"
                k_cache = torch.cat([self.k_cache, k], dim=-2)
                v_cache = torch.cat([self.v_cache, v], dim=-2)

            self.k_cache = k_cache
            self.v_cache = v_cache

            k = k_cache
            v = v_cache

        # Use PyTorch SDPA (scaled_dot_product_attention)
        # Let PyTorch choose the best backend automatically
        x = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
        )
        x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def _forward_varlen(self, x, cu_seqlens):
        """Variable-length attention for packed (seq_total, C) input.

        Uses ``flash_attn_varlen_func`` when available; falls back to SDPA
        with per-image splits otherwise. Skips RoPE / kv_cache (the dNaViT
        decoder, which is the only caller, uses neither).
        """
        assert x.dim() == 2, f"_forward_varlen expects (seq_total, C); got {x.shape}"
        seq_total, C = x.shape
        qkv = self.qkv(x).reshape(seq_total, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(1, 0, 2, 3).unbind(0)
        q = self.q_norm(q)
        k = self.k_norm(k)

        try:
            from flash_attn import flash_attn_varlen_func
            _have_flash = True
        except ImportError:
            _have_flash = False

        if _have_flash and x.is_cuda:
            max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())
            attn_out = flash_attn_varlen_func(
                q, k, v,
                cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
                softmax_scale=self.scale, causal=False,
            )  # (seq_total, heads, head_dim)
        else:
            lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
            outs = []
            offset = 0
            for L_i in lengths:
                q_i = q[offset:offset + L_i].transpose(0, 1).unsqueeze(0)  # (1,h,L_i,hd)
                k_i = k[offset:offset + L_i].transpose(0, 1).unsqueeze(0)
                v_i = v[offset:offset + L_i].transpose(0, 1).unsqueeze(0)
                o_i = F.scaled_dot_product_attention(q_i, k_i, v_i, is_causal=False)
                outs.append(o_i.squeeze(0).transpose(0, 1))             # (L_i, h, hd)
                offset += L_i
            attn_out = torch.cat(outs, dim=0)                            # (seq_total, h, hd)

        attn_out = attn_out.reshape(seq_total, C)
        return self.proj_drop(self.proj(attn_out))

def modulate(x, shift, scale):
    return x * (1 + scale) + shift

class FinalLayer(nn.Module):
    def __init__(self, dim, norm_layer=RMSNorm):
        super().__init__()
        self.norm_final = norm_layer(dim)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 2*dim)
        )
    
    def forward(self, x, c):
        scale, shift = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        return x


# basic transformer block
class Block(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            act_layer: nn.Module = nn.GELU,
            use_swiglu: bool = False,
            use_adaln: bool = False,
            norm_layer = RMSNorm,
            rope: nn.Module = None,
            target_aware_rope: bool = False,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)

        self.attn = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
            rope=rope,
            target_aware_rope=target_aware_rope,
        )

        self.norm2 = norm_layer(dim)

        if use_swiglu:
            self.mlp = SwiGLUFFN(
                in_features=dim,
                hidden_features=int(2/3 * int(dim * mlp_ratio)),
                drop=proj_drop,
            )
        else:
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=int(dim * mlp_ratio),
                act_layer=act_layer,
                drop=proj_drop,
            )

        self.use_adaln = use_adaln
        if self.use_adaln:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(dim, 6 * dim, bias=True)
            )


    def forward(self, x: torch.Tensor, attn_mask=None, c=None, rope_order=None,
                target_rope_order=None, cu_seqlens=None) -> torch.Tensor:
        if self.use_adaln:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
            x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), attn_mask=attn_mask, rope_order=rope_order, target_rope_order=target_rope_order, cu_seqlens=cu_seqlens)
            x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        else:
            x = x + self.attn(self.norm1(x), attn_mask=attn_mask, rope_order=rope_order, target_rope_order=target_rope_order, cu_seqlens=cu_seqlens)
            x = x + self.mlp(self.norm2(x))
        return x


def _expand_token(token, batch_size: int):
    return token.unsqueeze(0).expand(batch_size, -1, -1)


def varlen_attention(q, k, v, cu_seqlens, max_seqlen):
    """Variable-length attention over packed tokens (zero padding overhead).

    q/k/v: ``(total_tokens, num_heads, head_dim)``; ``cu_seqlens=(B+1,)`` int32.
    Uses ``flash_attn_varlen_func`` on CUDA + fp16/bf16, else per-segment SDPA.
    Returns ``(total_tokens, num_heads, head_dim)``.
    """
    if q.is_cuda and q.dtype in (torch.float16, torch.bfloat16):
        try:
            from flash_attn import flash_attn_varlen_func
            return flash_attn_varlen_func(
                q, k, v, cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
                dropout_p=0.0, causal=False)
        except ImportError:
            pass
    lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    outs, off = [], 0
    for L_i in lengths:
        qi = q[off:off + L_i].transpose(0, 1).unsqueeze(0)
        ki = k[off:off + L_i].transpose(0, 1).unsqueeze(0)
        vi = v[off:off + L_i].transpose(0, 1).unsqueeze(0)
        oi = F.scaled_dot_product_attention(qi, ki, vi, is_causal=False)
        outs.append(oi.squeeze(0).transpose(0, 1))
        off += L_i
    return torch.cat(outs, dim=0)


def _require_encoder_path(config, key: str) -> str:
    """Read the local checkpoint dir of the frozen vision backbone from config."""
    path = config.model.vq_model.get(key, None)
    if not path:
        raise ValueError(
            f"model.vq_model.{key} is not set. Point it at the local pretrained "
            f"checkpoint directory of the vision encoder.")
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"model.vq_model.{key}='{path}' is not an existing directory.")
    return path


class UnifiedEncoder(nn.Module):
    """
    Vision encoder with a dual-branch (frozen + trainable) backbone.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.original_image_size = config.dataset.preprocessing.crop_size
        self.image_size = self.original_image_size
        self.patch_size = 16
        self.grid_size = self.image_size // self.patch_size

        self.backbone_type = config.model.vq_model.get("encoder_type", "siglip2")
        # Optional Qwen patch merger (set by _init_qwen3vit when enabled).
        self.qwen_merger = None
        self.qwen_merge_factor = 1
        # Merger out-dim (out_hidden_size), exposed for merger_after_quant /
        # qwen_use_merger. None on backbones without a patch-merger (siglip2).
        self.qwen_merger_out_dim = None
        # Post-block LayerNorm for the Qwen 16x path (no merger). Raw Qwen block
        # outputs are large-norm and un-normalized; feeding them straight to the
        # unit-scale SimVQ codebook collapses it. Set by _init_qwen3vit when the
        # merger is OFF; the merger's own LayerNorm handles the 32x path.
        self.qwen_post_layernorm = None
        # FROZEN teacher twin of qwen_post_layernorm (stage-C symmetric teacher-
        # distill, 16x path). Built when use_teacher_res_mod is on so the distill
        # target lives in the SAME resmod+LN normalized space as the student
        # (target = LN_t(resmod_t(x))). Weights copied from the student LN in
        # warm-start; kept frozen. None otherwise.
        self.teacher_qwen_post_layernorm = None
        # The pretrained vision backbone is never bundled with the repository:
        # the config must point at a local checkpoint directory.
        if self.backbone_type == "siglip2":
            self._init_siglip2(_require_encoder_path(config, "siglip2_path"))
        elif self.backbone_type == "qwen3vit":
            self._init_qwen3vit(_require_encoder_path(config, "qwen3vit_path"))
        else:
            raise ValueError(f"Unknown encoder_type: {self.backbone_type!r}. Expected 'siglip2' or 'qwen3vit'.")


        # ---- Residual Modulation (ResMod) -----
        vq_cfg = config.model.vq_model
        if bool(vq_cfg.get("use_res_mod", False)):
            # ResMod runs on pre-merge block tokens (hidden_size). It is
            # incompatible with the end-of-forward merger (qwen_use_merger, where
            # self.width = out_hidden_size), but it IS the mechanism that carries
            # the merger under merger_after_quant (the merger lives in ResMod's
            # internal_merger, blocks still run at hidden_size).
            self.merger_after_quant = bool(vq_cfg.get("merger_after_quant", False))
            if self.qwen_merger is not None and self.qwen_merge_factor > 1:
                raise ValueError(
                    "use_res_mod is incompatible with qwen_use_merger: ResMod runs on "
                    "pre-merge block tokens (hidden_size) but self.width is the merger's "
                    "out_hidden_size. Disable one of them.")
            res_cfg = vq_cfg.get("res_mod", None)
            if res_cfg is None or res_cfg.get("layer_idx_list", None) is None:
                raise ValueError("use_res_mod=True requires model.vq_model.res_mod.layer_idx_list to be set (e.g. [1, 8, 16, 24]).")
            layer_idx_list = list(res_cfg.layer_idx_list)

            # Under merger_after_quant the encoder's final output lives in the
            # Qwen merger-out space (T/ms2 tokens at qwen_merger_out_dim). The
            # blocks still see the original token count (full-resolution pixel
            # modulation); the SFT runs in merger-out space via ResMod's
            # internal merger (output_dim / input_merge_size knobs).
            if self.merger_after_quant:
                if self.qwen_merger is None or self.qwen_merger_out_dim is None:
                    raise ValueError(
                        "merger_after_quant=True requires the qwen3vit backbone "
                        "(no patch-merger on siglip2).")
                resmod_output_dim = int(self.qwen_merger_out_dim)
                resmod_merge_size = int(self.qwen_spatial_merge_size) ** 2
            else:
                resmod_output_dim = self.width
                resmod_merge_size = 1

            self.res_mod = ResMod(
                hidden_size=self.width,
                layer_idx_list=layer_idx_list,
                depth=int(res_cfg.get("depth", 2)),
                num_heads=int(res_cfg.get("num_heads", 16)),
                mlp_ratio=float(res_cfg.get("mlp_ratio", 4.0)),
                init_values=res_cfg.get("init_values", None),
                output_dim=resmod_output_dim,
                input_merge_size=resmod_merge_size,
            )
            # Deterministic set lookup for tap collection inside the forward.
            self._res_mod_layer_set = set(self.res_mod.layer_idx_list)
            # Validate layer indices against the backbone depth. Convention:
            #   idx == 1               -> post-patch-embed feature (pre any block)
            #   idx in [2, depth + 1]  -> output of block (idx - 1) in 0-indexed
            #                             == "(idx-1)-th block" in 1-indexed
            backbone_depth = self._trainable_block_count()
            for idx in self.res_mod.layer_idx_list:
                if idx < 1 or idx > backbone_depth + 1:
                    raise ValueError(
                        f"res_mod.layer_idx_list contains {idx}, valid range is "
                        f"[1, {backbone_depth + 1}] for this backbone "
                        f"({self.backbone_type}, depth={backbone_depth}).")
            # Block-output taps (idx >= 2) tested inside the encoder forward.
            self._res_mod_block_set = set(self._res_mod_layer_set) - {1}

            # Optional FROZEN teacher ResMod (stage-C joint-ft): a structural
            # twin of res_mod, warm-started from the SAME stage-B weights and
            # kept frozen. It produces a stable post-ResMod target z_teacher
            # for the aux feature-distillation, so the trainable (student)
            # res_mod's output can flow into the quantizer WITHOUT being
            # detached. Built identically; weights copied in warm-start.
            self.use_teacher_res_mod = bool(vq_cfg.get("use_teacher_res_mod", False))
            if self.use_teacher_res_mod:
                self.teacher_res_mod = ResMod(
                    hidden_size=self.width,
                    layer_idx_list=layer_idx_list,
                    depth=int(res_cfg.get("depth", 2)),
                    num_heads=int(res_cfg.get("num_heads", 16)),
                    mlp_ratio=float(res_cfg.get("mlp_ratio", 4.0)),
                    init_values=res_cfg.get("init_values", None),
                    output_dim=resmod_output_dim,
                    input_merge_size=resmod_merge_size,
                )
                self.teacher_res_mod.eval()
                self.teacher_res_mod.requires_grad_(False)
            else:
                self.teacher_res_mod = None
        else:
            self.res_mod = None
            self.teacher_res_mod = None
            self.use_teacher_res_mod = False
            self.merger_after_quant = False
            self._res_mod_layer_set = set()
            self._res_mod_block_set = set()

    # ---- SigLIP2 backbone ---------------------------------------------------
    def _init_siglip2(self, path):
        # Load pretrained SigLIP2 vision model (frozen)
        full_model = AutoModel.from_pretrained(path)
        self.siglip = full_model.vision_model
        # Freeze the SigLIP2 model
        self.siglip.eval()
        self.siglip.requires_grad_(False)
        # The pooling head is never used (we only consume embeddings + encoder +
        # post_layernorm) — drop it so its parameters don't sit in memory.
        self.siglip.head = nn.Identity()
        self.siglip.use_head = False
        del full_model
        self.width = self.siglip.config.hidden_size

        # Geometry for the packed varlen (flash) forward: explicit head shape +
        # the abs pos-embed grid side used for per-sample interpolation.
        self.siglip_num_heads = self.siglip.config.num_attention_heads
        self.siglip_head_dim = self.width // self.siglip_num_heads
        self.siglip_pos_grid = int(self.siglip.config.num_patches ** 0.5)

        # Frozen encoder (alias) + trainable deep-copy.
        self.frozen_encoder = self.siglip.encoder
        self.frozen_encoder.eval()
        self.frozen_encoder.requires_grad_(False)

        self.trainable_encoder = copy.deepcopy(self.siglip.encoder)
        self.trainable_encoder.train()
        self.trainable_encoder.requires_grad_(True)

        # Post layer norms — frozen alias + trainable deep-copy.
        self.frozen_post_layernorm = self.siglip.post_layernorm
        self.frozen_post_layernorm.eval()
        self.frozen_post_layernorm.requires_grad_(False)

        self.trainable_post_layernorm = copy.deepcopy(self.siglip.post_layernorm)
        self.trainable_post_layernorm.train()
        self.trainable_post_layernorm.requires_grad_(True)

    # ---- Qwen3ViT backbone --------------------------------------------------
    def _init_qwen3vit(self, path):
        """Load the Qwen3-VL ViT the SigLIP way: as a transformers-native vision
        model, sourced from the full Qwen3-VL VLM checkpoint directory.
        """
        import os
        import json
        from safetensors import safe_open
        from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

        cfg_all = json.load(open(os.path.join(path, "config.json")))
        vcfg = dict(cfg_all["vision_config"])
        vcfg.pop("model_type", None)  # 'qwen3_5' isn't a registered vision model_type here
        vision_config = Qwen3VLVisionConfig(**vcfg)
        # Force FlashAttention-2 (default direct-construction path picks SDPA).
        vision_config._attn_implementation = "flash_attention_2"
        vit = Qwen3VLVisionModel(vision_config)

        # Read only model.visual.* (skip the LLM shards entirely).
        idx_path = os.path.join(path, "model.safetensors.index.json")
        if os.path.exists(idx_path):
            weight_map = json.load(open(idx_path))["weight_map"]
            by_shard = {}
            for k in weight_map:
                if k.startswith("model.visual."):
                    by_shard.setdefault(weight_map[k], []).append(k)
            shards = list(by_shard.items())
        else:
            shards = [("model.safetensors", None)]
        state = {}
        for shard, keys in shards:
            with safe_open(os.path.join(path, shard), framework="pt", device="cpu") as f:
                ks = keys if keys is not None else [k for k in f.keys() if k.startswith("model.visual.")]
                for k in ks:
                    state[k[len("model.visual."):]] = f.get_tensor(k)
        missing, unexpected = vit.load_state_dict(state, strict=False)
        del state
        print(f"[qwen3vit] native Qwen3VLVisionModel loaded from "
              f"{os.path.basename(path.rstrip('/'))} "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")

        if vit.patch_size != self.patch_size:
            raise ValueError(
                f"Qwen3ViT patch_size={vit.patch_size} != Encoder.patch_size={self.patch_size}"
            )

        # Optionally keep Qwen's native patch merger (2x2 spatial merge -> one
        # out_hidden_size token, i.e. 32x downsampling at patch16). It is kept
        # FROZEN with its pretrained weights. Two distinct modes use it:
        #   * ``qwen_use_merger`` (end-of-forward 32x): applied after the
        #     student/teacher blocks in _forward_qwen3vit; encoder.width becomes
        #     out_hidden_size and token count shrinks by the merge factor.
        #   * ``merger_after_quant``: the merger does NOT run at forward-end.
        #     ResMod's internal merger collapses tokens to merger-out space
        #     before the quantizer; the frozen merger is exposed for the
        #     tokenizer's teacher-target / sem-loss alignment only. encoder.width
        #     stays hidden_size (ResMod blocks run on full-grid tokens).
        # When neither is set we drop the merger (16x, hidden_size tokens). The
        # deepstack mergers are always dropped (LLM-only).
        use_merger = bool(self.config.model.vq_model.get("qwen_use_merger", False))
        merger_after_quant = bool(self.config.model.vq_model.get("merger_after_quant", False))
        self.qwen_merger_out_dim = None
        if use_merger:
            self.qwen_merger = vit.merger          # pretrained Qwen3VLVisionPatchMerger
            self.qwen_merger.eval()
            self.qwen_merger.requires_grad_(False)
            self.qwen_merge_factor = int(vit.spatial_merge_size) ** 2
            self.qwen_merger_out_dim = int(vision_config.out_hidden_size)
        elif merger_after_quant:
            # Keep the frozen merger available (consumed by ResMod / tokenizer)
            # but DON'T apply it at forward-end (qwen_merge_factor stays 1).
            self.qwen_merger = vit.merger
            self.qwen_merger.eval()
            self.qwen_merger.requires_grad_(False)
            self.qwen_merger_out_dim = int(vision_config.out_hidden_size)
        elif bool(self.config.model.vq_model.get("qwen_post_layernorm", True)):
            # 16x path: standardize per-token features before the quantizer so
            # they match the unit-scale SimVQ codebook (mirrors SigLIP's
            # trainable_post_layernorm). Warm-start from the merger's pretrained
            # LayerNorm (ln_q over hidden_size) when present, else default init;
            # kept TRAINABLE so it can adapt the recon target's scale. Disable
            # with ``qwen_post_layernorm: false`` to feed raw block features
            # straight to the quantizer (recon target = un-normalized z).
            hidden_size = int(vision_config.hidden_size)
            post_ln = None
            for sub in vit.merger.modules():
                if isinstance(sub, nn.LayerNorm) and tuple(sub.normalized_shape) == (hidden_size,):
                    post_ln = copy.deepcopy(sub)
                    break
            if post_ln is None:
                post_ln = nn.LayerNorm(hidden_size)
            post_ln.train()
            post_ln.requires_grad_(True)
            self.qwen_post_layernorm = post_ln
            # Symmetric teacher-distill (16x): a FROZEN twin LayerNorm so the
            # distill target is LN_t(resmod_t(x)) — same normalized space as the
            # student's LN_s(resmod_s(x)). Structural copy here; weights are
            # copied from the (warm-started) student LN in warm-start, then this
            # stays frozen. Gated by use_teacher_res_mod (same switch that builds
            # teacher_res_mod).
            if bool(self.config.model.vq_model.get("use_teacher_res_mod", False)):
                teacher_ln = copy.deepcopy(post_ln)
                teacher_ln.eval()
                teacher_ln.requires_grad_(False)
                self.teacher_qwen_post_layernorm = teacher_ln
        vit.merger = nn.Identity()
        vit.deepstack_merger_list = nn.ModuleList()
        self.frozen_blocks = vit.blocks            # frozen target branch (clip_gt)
        vit.blocks = nn.ModuleList()               # qwen_vit now only holds patch_embed / pos_embed / rotary
        vit.eval()
        vit.requires_grad_(False)

        # ``qwen_vit`` keeps the SHARED frozen input stages (patch_embed /
        # pos_embed / rotary_pos_emb) and the native helper methods
        # (rot_pos_emb / fast_pos_embed_interpolate).
        self.qwen_vit = vit
        for p in self.frozen_blocks.parameters():
            p.requires_grad_(False)

        # Trainable deep copy of the blocks (the branch that feeds the decoder).
        self.trainable_blocks = copy.deepcopy(self.frozen_blocks)
        self.trainable_blocks.train()
        self.trainable_blocks.requires_grad_(True)

        self.qwen_spatial_merge_size = int(vit.spatial_merge_size)
        self.qwen_temporal_patch_size = int(vision_config.temporal_patch_size)
        # With the end-of-forward merger on (qwen_use_merger), encoder tokens
        # live in out_hidden_size. Under merger_after_quant the merger runs
        # INSIDE ResMod/after-quant, so the encoder backbone width stays
        # hidden_size. Otherwise (16x) it's the raw block hidden_size.
        self.width = (int(vision_config.out_hidden_size)
                      if (self.qwen_merger is not None and self.qwen_merge_factor > 1)
                      else int(vision_config.hidden_size))

    # ---- helpers ------------------------------------------------------------
    def _trainable_block_count(self) -> int:
        """Number of trainable transformer blocks in the backbone."""
        if self.backbone_type == "siglip2":
            return len(self.trainable_encoder.layers)
        return len(self.trainable_blocks)

    def set_trainable_backbone_grad(self, flag: bool) -> None:
        """ For an encoder-freeze warmup """
        if self.backbone_type == "siglip2":
            self.trainable_encoder.requires_grad_(flag)
            self.trainable_post_layernorm.requires_grad_(flag)
        else:  # qwen3vit
            self.trainable_blocks.requires_grad_(flag)
            if self.qwen_post_layernorm is not None:
                # Symmetric teacher-distill: the student LN is a trainable
                # ADAPTER (paired with the frozen teacher LN twin), so it trains
                # regardless of the backbone-freeze — same treatment as res_mod.
                # Only couple it to the backbone freeze when there is no teacher
                # twin (legacy single-LN behaviour).
                if self.teacher_qwen_post_layernorm is None:
                    self.qwen_post_layernorm.requires_grad_(flag)

    # ---- forward ------------------------------------------------------------
    def forward(self, pixel_values, return_clip_gt=True, image_grid_thw=None):
        """Encode a batch through the dual-branch (student + teacher) backbone.

        Both backbones run a single packed varlen core — no padding tokens pass
        through any projection/attention — and fixed resolution is just the
        equal-length special case of variable resolution.

        Inputs:
          - ``(B, 3, H, W)`` dense + ``image_grid_thw=None``           (fixed-res), or
          - a list of ``(3, H_i, W_i)`` / a pre-patchified packed tensor, together
            with ``image_grid_thw=(B, 3) [t, h, w]``                  (variable-res).

        Returns (``trainable`` = student, ``frozen`` = clip_gt teacher):
          - fixed-res dense input -> ``(B, L, width)`` tensors;
          - otherwise             -> packed ``(seq_total, width)`` tensors plus a
            trailing ``cu_seqlens=(B+1,)`` int32.
        ``frozen`` is dropped when ``return_clip_gt=False``.
        """
        if self.backbone_type == "siglip2":
            return self._forward_siglip2(pixel_values, return_clip_gt, image_grid_thw)
        return self._forward_qwen3vit(pixel_values, return_clip_gt, image_grid_thw)

    @staticmethod
    def _normalize_grid_thw(image_grid_thw, device):
        if not torch.is_tensor(image_grid_thw):
            image_grid_thw = torch.tensor(image_grid_thw, device=device, dtype=torch.long)
        return image_grid_thw.to(device=device, dtype=torch.long)

    def _finalize(self, trainable, frozen, cu_seqlens, layout, return_clip_gt):
        """Shared output adapter for both backbones: reshape packed tokens back
        to ``(B, L, D)`` for fixed-res dense input; otherwise keep them packed
        and append ``cu_seqlens``."""
        if layout[0] == "dense":
            _, B, L = layout
            trainable = trainable.view(B, L, -1)
            if return_clip_gt:
                return trainable, frozen.view(B, L, -1)
            return trainable
        if return_clip_gt:
            return trainable, frozen, cu_seqlens
        return trainable, cu_seqlens

    def _finalize_merged(self, trainable, frozen, cu_trainable, layout, return_clip_gt):
        """``_finalize`` variant for merger_after_quant: ``trainable`` already
        lives in merged space (T/ms2 tokens at merger_out_dim), while ``frozen``
        stays in pre-merger hidden_size space (the tokenizer applies the frozen
        merger to it on demand). For dense input we convert layout's L (the
        pre-merger token count) into the merged token count for ``trainable``."""
        ms2 = int(self.qwen_spatial_merge_size) ** 2
        if layout[0] == "dense":
            _, B, L = layout
            if L % ms2 != 0:
                raise ValueError(
                    f"merger_after_quant dense path needs L divisible by {ms2}; got {L}.")
            trainable = trainable.view(B, L // ms2, -1)
            if return_clip_gt:
                return trainable, frozen.view(B, L, -1)
            return trainable
        if return_clip_gt:
            return trainable, frozen, cu_trainable
        return trainable, cu_trainable

    # ---- SigLIP2 ------------------------------------------------------------
    def _siglip_interp_pos(self, hp, wp, dtype, device):
        """Bilinear-interpolate the learned abs pos-embed to a (hp, wp) grid -> (hp*wp, D)."""
        g = self.siglip_pos_grid
        pe = self.siglip.embeddings.position_embedding.weight.float()
        pe = pe.reshape(g, g, -1).permute(2, 0, 1).unsqueeze(0)          # (1, D, g, g)
        if hp != g or wp != g:
            pe = F.interpolate(pe, size=(hp, wp), mode='bilinear',
                               align_corners=False, antialias=True)
        pe = pe.squeeze(0).permute(1, 2, 0).reshape(hp * wp, -1)
        return pe.to(dtype=dtype, device=device)

    def _siglip_pack_embed(self, pixel_values, image_grid_thw):
        """Any input layout -> packed token embeddings ``(total, D)`` + ``cu_seqlens``
        + layout tag. Patch order is ``(p1, p2, c)`` (the Linear patch_embedding's
        layout). Collects per-image grids, patch-embeds the valid tokens, then adds
        per-sample interpolated abs pos-embed."""
        ps = self.patch_size
        if image_grid_thw is None:
            B, C, H, W = pixel_values.shape
            if H % ps or W % ps:
                raise ValueError(f"SigLIP2 needs H,W divisible by {ps}; got {H}x{W}.")
            gh, gw = H // ps, W // ps
            patches = rearrange(
                pixel_values, 'b c (h p1) (w p2) -> (b h w) (p1 p2 c)', p1=ps, p2=ps)
            shapes = [(gh, gw)] * B
            layout = ("dense", B, gh * gw)
        else:
            device = (pixel_values[0].device if isinstance(pixel_values, (list, tuple))
                      else pixel_values.device)
            grid = self._normalize_grid_thw(image_grid_thw, device)
            if isinstance(pixel_values, (list, tuple)):
                chunks, shapes = [], []
                for img, row in zip(pixel_values, grid):
                    hi, wi = int(row[1]), int(row[2])
                    chunks.append(rearrange(
                        img, 'c (h p1) (w p2) -> (h w) (p1 p2 c)', p1=ps, p2=ps))
                    shapes.append((hi, wi))
                patches = torch.cat(chunks, dim=0)
            else:
                patches = pixel_values                              # pre-patchified (seq_total, ps*ps*3)
                shapes = [(int(r[1]), int(r[2])) for r in grid]
            layout = ("packed",)

        emb = self.siglip.embeddings
        hidden = emb.patch_embedding(patches.to(emb.patch_embedding.weight.dtype))
        hidden = hidden + torch.cat(
            [self._siglip_interp_pos(h, w, hidden.dtype, hidden.device) for (h, w) in shapes], 0)
        seqlens = [h * w for (h, w) in shapes]
        cu = torch.zeros(len(seqlens) + 1, dtype=torch.int32, device=hidden.device)
        cu[1:] = torch.tensor(seqlens, dtype=torch.int32, device=hidden.device).cumsum(0)
        return hidden, cu, layout

    def _run_siglip_blocks(self, layers, hidden, cu_seqlens, max_seqlen, collect_taps=False):
        """Run SigLIP2 encoder layers over packed ``(total, D)`` tokens via varlen
        attention. Shared by student/teacher. Taps: layer-1 = post-embed, layer
        (i+2) = output of block i (1-indexed)."""
        nh, hd = self.siglip_num_heads, self.siglip_head_dim
        use_rm = collect_taps and self.res_mod is not None
        taps = [hidden] if (use_rm and 1 in self._res_mod_layer_set) else []
        for i, layer in enumerate(layers):
            normed = layer.layer_norm1(hidden)
            attn, T = layer.self_attn, normed.shape[0]
            q = attn.q_proj(normed).view(T, nh, hd)
            k = attn.k_proj(normed).view(T, nh, hd)
            v = attn.v_proj(normed).view(T, nh, hd)
            o = varlen_attention(q, k, v, cu_seqlens, max_seqlen).reshape(T, self.width)
            hidden = hidden + attn.out_proj(o)
            hidden = hidden + layer.mlp(layer.layer_norm2(hidden))
            if use_rm and (i + 2) in self._res_mod_block_set:
                taps.append(hidden)
        return hidden, taps

    def _forward_siglip2(self, pixel_values, return_clip_gt=True, image_grid_thw=None):
        self.siglip.eval()
        self.siglip.requires_grad_(False)
        hidden, cu, layout = self._siglip_pack_embed(pixel_values, image_grid_thw)
        max_seqlen = int((cu[1:] - cu[:-1]).max().item())

        trainable, taps = self._run_siglip_blocks(
            self.trainable_encoder.layers, hidden, cu, max_seqlen, collect_taps=True)
        trainable = self.trainable_post_layernorm(trainable)
        if self.res_mod is not None:
            trainable = self.res_mod(taps, trainable, cu_seqlens=cu)

        frozen = None
        if return_clip_gt:
            with torch.no_grad():
                frozen, _ = self._run_siglip_blocks(
                    self.frozen_encoder.layers, hidden, cu, max_seqlen)
                frozen = self.frozen_post_layernorm(frozen)
        return self._finalize(trainable, frozen, cu, layout, return_clip_gt)

    # ---- Qwen3ViT -----------------------------------------------------------
    def _qwen_patchify(self, img_chw):
        """(C, H, W) -> ((gh*gw, C*T*ps*ps), (gh, gw)) in the Conv3d (C, T, ph, pw)
        order, with temporal duplication + spatial-merge token grouping (matches
        the official Qwen vision model)."""
        ps, T, ms = self.patch_size, self.qwen_temporal_patch_size, self.qwen_spatial_merge_size
        C, H, W = img_chw.shape
        if H % (ps * ms) or W % (ps * ms):
            raise ValueError(f"Qwen3ViT needs H,W divisible by {ps*ms}; got {H}x{W}.")
        gh, gw = H // ps, W // ps
        x = img_chw.unsqueeze(0).unsqueeze(2).expand(1, C, T, H, W).contiguous()
        x = x.view(1, C, T, gh // ms, ms, ps, gw // ms, ms, ps)
        x = x.permute(0, 3, 6, 4, 7, 1, 2, 5, 8).contiguous()
        return x.view(gh * gw, C * T * ps * ps), (gh, gw)

    def _qwen_pack_embed(self, pixel_values, image_grid_thw):
        """Any input layout -> packed patch_embeds ``(total, D)`` + ``cu_seqlens`` +
        rotary ``position_embeddings`` + layout. The patch-embed / pos-embed / rope
        use the native Qwen vision helpers (identical math to the official model)."""
        if image_grid_thw is None:
            B = pixel_values.shape[0]
            flats, grids = zip(*(self._qwen_patchify(img) for img in pixel_values))
            flat = torch.cat(flats, dim=0)
            gh, gw = grids[0]
            grid_thw = torch.tensor([[1, gh, gw]] * B, device=flat.device, dtype=torch.long)
            layout = ("dense", B, gh * gw)
        else:
            device = (pixel_values[0].device if isinstance(pixel_values, (list, tuple))
                      else pixel_values.device)
            grid_thw = self._normalize_grid_thw(image_grid_thw, device)
            if isinstance(pixel_values, (list, tuple)):
                flats = []
                for img, row in zip(pixel_values, grid_thw):
                    fi, (gh_i, gw_i) = self._qwen_patchify(img)
                    if int(row[1]) != gh_i or int(row[2]) != gw_i:
                        raise ValueError("image_grid_thw row mismatches image patch grid.")
                    flats.append(fi)
                flat = torch.cat(flats, dim=0)
            else:
                flat = pixel_values                                 # pre-patchified (seq_total, C*T*ps*ps)
            layout = ("packed",)

        vit = self.qwen_vit
        grid_thw = grid_thw.to(flat.device)
        hidden = vit.patch_embed(flat)
        hidden = hidden + vit.fast_pos_embed_interpolate(grid_thw).to(
            device=hidden.device, dtype=hidden.dtype)
        rotary = vit.rot_pos_emb(grid_thw).to(hidden.device)
        emb = torch.cat((rotary, rotary), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())
        cu = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(dim=0, dtype=torch.int32)
        cu = F.pad(cu, (1, 0), value=0)
        return hidden, cu, position_embeddings, layout

    def _run_qwen_blocks(self, blocks, hidden, cu_seqlens, position_embeddings, collect_taps=False):
        """Run native Qwen vision blocks over packed ``(total, D)`` tokens. Shared
        by student/teacher. Same tap convention as ``_run_siglip_blocks``."""
        use_rm = collect_taps and self.res_mod is not None
        taps = [hidden] if (use_rm and 1 in self._res_mod_layer_set) else []
        for i, blk in enumerate(blocks):
            hidden = blk(hidden, cu_seqlens=cu_seqlens, position_embeddings=position_embeddings)
            if use_rm and (i + 2) in self._res_mod_block_set:
                taps.append(hidden)
        return hidden, taps

    def _forward_qwen3vit(self, pixel_values, return_clip_gt=True, image_grid_thw=None):
        hidden, cu, position_embeddings, layout = self._qwen_pack_embed(
            pixel_values, image_grid_thw)

        # Side-channel for the aux feature-distillation target: the PRE-ResMod
        # frozen-merger feature (only set on the merger_after_quant path below;
        # read synchronously by the tokenizer right after this forward). None on
        # every other path so a stale value can never leak.
        self._last_merge_target = None
        self._last_teacher_z = None
        self._last_student_z_preln = None

        trainable, taps = self._run_qwen_blocks(
            self.trainable_blocks, hidden, cu, position_embeddings, collect_taps=True)

        frozen_pre = None
        if return_clip_gt:
            with torch.no_grad():
                frozen_pre, _ = self._run_qwen_blocks(
                    self.frozen_blocks, hidden, cu, position_embeddings)

        if getattr(self, "merger_after_quant", False):
            # ResMod's internal merger collapses ms2 tokens + lifts to
            # merger-out space; its SFT base target is frozen_merger(trainable)
            # (no grad on the merger). The encoder output ``z`` then lives in
            # (T/ms2, merger_out_dim). ``frozen`` (clip_gt) stays in pre-merger
            # hidden_size space — the tokenizer applies the frozen merger to it
            # on demand for sem-loss / teacher alignment.
            with torch.no_grad():
                merge_target = self.qwen_merger(trainable)
            # Stash the PRE-ResMod frozen-merger feature: this (NOT the ResMod
            # output ``z``) is the stable semantic target for the aux feature
            # distillation — it matches what the stage-1 FeatureDecoder learned to
            # reconstruct and is a fixed target (ResMod's SFT is excluded).
            self._last_merge_target = merge_target
            trainable = self.res_mod(taps, merge_target, cu_seqlens=cu)
            # FROZEN teacher ResMod twin: same taps + same pre-ResMod
            # merge_target, but its weights are fixed (warm-started from
            # stage-B). z_teacher is the stable post-ResMod feature-distill
            # target; stashed as a side-channel read synchronously by the
            # tokenizer right after this forward. None when unused.
            self._last_teacher_z = None
            if self.teacher_res_mod is not None:
                with torch.no_grad():
                    self._last_teacher_z = self.teacher_res_mod(
                        taps, merge_target, cu_seqlens=cu)
            cu = self._maybe_merge_cu(cu)
            # Dense layout: _finalize_merged reshapes `trainable` to
            # (B, L//ms2, out_dim), so the side-channel caches must match to
            # avoid broadcast errors in feature_distill_loss /
            # resmod_preserve_mse_loss (consumers compare against 3D z /
            # feature_recon_pred). Varlen keeps 2D packed (offline scripts
            # depend on it).
            if layout[0] == "dense":
                _, B, L = layout
                ms2 = int(self.qwen_spatial_merge_size) ** 2
                L_merged = L // ms2
                self._last_merge_target = self._last_merge_target.view(
                    B, L_merged, -1)
                if self._last_teacher_z is not None:
                    self._last_teacher_z = self._last_teacher_z.view(
                        B, L_merged, -1)
            return self._finalize_merged(trainable, frozen_pre, cu, layout, return_clip_gt)

        if self.res_mod is not None:
            # Stash the PRE-ResMod feature as the stable semantic anchor (the 16x
            # analog of the merger_after_quant path's frozen-merger merge_target),
            # so the continuous-AE pre_quant_sem leash can supervise
            # proj(post-ResMod z) against the pre-ResMod ViT output.
            self._last_merge_target = trainable
            # FROZEN teacher twin (stage-C symmetric teacher-distill, 16x path):
            # same taps + same pre-ResMod anchor as the student, but fixed
            # weights. The distill target is LN_t(resmod_t(x)) — the teacher's
            # post-ResMod z pushed through the FROZEN teacher LayerNorm twin, so
            # it lives in the SAME normalized space as the student side
            # (LN_s(resmod_s(x))). Stashed as a side-channel read synchronously by
            # the tokenizer right after this forward. None when unused. Computed
            # BEFORE the student overwrites ``trainable`` so both see the
            # identical pre-ResMod input.
            self._last_teacher_z = None
            if self.teacher_res_mod is not None:
                with torch.no_grad():
                    teacher_z = self.teacher_res_mod(
                        taps, trainable, cu_seqlens=cu)
                    if self.teacher_qwen_post_layernorm is not None:
                        teacher_z = self.teacher_qwen_post_layernorm(teacher_z)
                self._last_teacher_z = teacher_z
            trainable = self.res_mod(taps, trainable, cu_seqlens=cu)
            # PRE-LN student post-ResMod z: the resmod-preserve leash pulls THIS
            # (before the student LN) toward the pre-ResMod anchor, keeping the
            # leash in the same un-normalized space as the anchor so the LN is
            # excluded from the preserve objective.
            self._last_student_z_preln = trainable
        frozen = frozen_pre

        # 16x path post-norm: standardize per-token features so they match the
        # SimVQ codebook scale (the merger path normalizes internally instead).
        # With a teacher twin the student LN is a TRAINABLE adapter paired with
        # the frozen teacher LN (target = LN_t(resmod_t(x))).
        if self.qwen_post_layernorm is not None:
            trainable = self.qwen_post_layernorm(trainable)
            if frozen is not None:
                with torch.no_grad():
                    frozen = self.qwen_post_layernorm(frozen)

        # Frozen Qwen patch merger (qwen_use_merger, end-of-forward 32x): each
        # spatial_merge_size^2 group of consecutive tokens (the patchify already
        # emits them in merge-grouped order) collapses into one out_hidden_size
        # token. Token count and cu_seqlens shrink by the merge factor; each
        # per-image seqlen is divisible by it since gh, gw are each multiples of
        # spatial_merge_size.
        if self.qwen_merger is not None and self.qwen_merge_factor > 1:
            mf = self.qwen_merge_factor
            trainable = self.qwen_merger(trainable)
            if frozen is not None:
                frozen = self.qwen_merger(frozen)
            cu = (cu // mf).to(cu.dtype)
            if layout[0] == "dense":
                layout = ("dense", layout[1], layout[2] // mf)
        return self._finalize(trainable, frozen, cu, layout, return_clip_gt)

    def _maybe_merge_cu(self, cu):
        """Shrink cu_seqlens by ms2 for the merger_after_quant trainable branch
        (the internal merger collapses every ms2 consecutive tokens into one).
        Each entry of ``cu`` is a multiple of ms2 because the encoder grid is
        built as multiples of ``patch_size * spatial_merge_size`` upstream."""
        ms2 = int(self.qwen_spatial_merge_size) ** 2
        if (cu % ms2).any():
            raise ValueError(
                f"merger_after_quant: cu_seqlens entries must all be divisible by {ms2}; "
                f"got {cu.tolist()}.")
        return (cu // ms2).to(cu.dtype)


class UnifiedDecoder(nn.Module):
    """
    Randomly-initialized ViT decoder that consumes 2D tokens (grid_size^2)
    from the UnifiedEncoder and reconstructs the RGB image.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_size = config.dataset.preprocessing.crop_size
        self.patch_size = config.model.vq_model.vit_dec_patch_size
        self.grid_size = self.image_size // self.patch_size

        # ViT-L architecture parameters. ``width`` defaults to 1024 (ViT-L) but
        # is overridable via ``model.vq_model.vit_dec_width`` so an identity-
        # bottleneck tokenizer (AETokenizer) can match the encoder width
        # (e.g. 1152 for the Qwen3.6 ViT) and feed features straight through.
        self.width = config.model.vq_model.get("vit_dec_width", 1024)
        self.num_layers = 24
        self.num_heads = 16
        self.mlp_ratio = 4.0

        # Always use LayerNorm and standard MLP
        norm_layer = nn.LayerNorm

        scale = self.width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(1, self.width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn(self.grid_size ** 2 + 1, self.width))

        self.ln_pre = norm_layer(self.width)
        self.transformer = nn.ModuleList()

        for i in range(self.num_layers):
            self.transformer.append(Block(
                dim=self.width,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=True,
                qk_norm=False,
                proj_drop=0.,
                attn_drop=0.,
                norm_layer=norm_layer,
                use_swiglu=False,
                use_adaln=False,  # No AdaLN for decoder
            ))
        self.ln_post = norm_layer(self.width)

        # CLIP target width (always 1152 for so400m)
        target_width = 1152

        # RGB reconstruction
        self.ffn = nn.Sequential(
            nn.Conv2d(self.width, self.patch_size * self.patch_size * 3, 1, padding=0, bias=True),
            Rearrange('b (p1 p2 c) h w -> b c (h p1) (w p2)',
                p1=self.patch_size, p2=self.patch_size),
        )
        self.conv_out = nn.Conv2d(3, 3, 3, padding=1, bias=True)

        # CLIP alignment
        loss_config = config.get("losses", {"clip_loss_weight": 0.0})
        self.clip_align = loss_config.get("clip_loss_weight", 0.0) > 0
        self.clip_align_layer_id = config.model.vq_model.get("clip_align_layer_id", 2)
        self.clip_projector = nn.Sequential(
            nn.Linear(self.width, 2048),
            nn.SiLU(),
            nn.Linear(2048, 2048),
            nn.SiLU(),
            nn.Linear(2048, target_width),
        ) if self.clip_align else None

        # Potentially use checkpointing
        self.use_checkpoint = config.model.vq_model.get("use_checkpoint", False)
        self.attn_mask = None

    def _interp_pos_embed(self, grid_h, grid_w, dtype):
        """Bicubic-interpolate ``self.positional_embedding`` to fit a different
        spatial grid for dynamic-resolution decoding.

        ``self.positional_embedding`` has shape ``(self.grid_size**2 + 1, width)``
        with row 0 = cls token, rows 1: laid out row-major over the original
        (grid_size, grid_size) grid. We split, interpolate the spatial part to
        ``(grid_h, grid_w)``, then re-prepend cls.

        Returns ``(grid_h*grid_w + 1, width)`` tensor in ``dtype``.
        """
        if grid_h == self.grid_size and grid_w == self.grid_size:
            return self.positional_embedding.to(dtype)
        cls_pos = self.positional_embedding[:1]
        spatial = self.positional_embedding[1:].reshape(
            self.grid_size, self.grid_size, self.width)
        spatial = spatial.permute(2, 0, 1).unsqueeze(0)  # (1, D, gs, gs)
        interp = F.interpolate(
            spatial, size=(grid_h, grid_w),
            mode='bicubic', align_corners=False,
        ).squeeze(0).permute(1, 2, 0).reshape(grid_h * grid_w, self.width)
        return torch.cat([cls_pos, interp], dim=0).to(dtype)

    @staticmethod
    def _normalize_grid_thw(image_grid_thw, device):
        if not torch.is_tensor(image_grid_thw):
            image_grid_thw = torch.tensor(image_grid_thw, device=device, dtype=torch.long)
        return image_grid_thw.to(device=device, dtype=torch.long)

    def forward(self, z_quantized, image_grid_thw=None):
        """Decode quantized tokens to RGB through a single packed varlen
        transformer; fixed resolution is the equal-length special case.

        Static (``image_grid_thw=None``): ``z=(B, grid_size**2, C)`` -> image
        ``(B, 3, H, W)`` and ``clip_pred (B, grid_size**2, target_width)``.

        Dynamic: ``z=(seq_total, C)`` + ``image_grid_thw=(B, 3) [t, h, w]`` ->
        ``recon_list [list of (3, H_i, W_i)]`` and ``clip_pred (seq_total, target_width)``.

        Either way the transformer runs over a packed (per-image cls + interp
        pos-embed) sequence with varlen attention (``cu_seqlens``) — no padding.
        """
        tokens, cu_seqlens, boundaries, layout = self._pack(z_quantized, image_grid_thw)
        x = self.ln_pre(tokens)

        clip_pred = None
        for i in range(self.num_layers):
            if self.use_checkpoint:
                x = torch.utils.checkpoint.checkpoint(
                    lambda xx, _blk=self.transformer[i], _cu=cu_seqlens:
                        _blk(xx, cu_seqlens=_cu),
                    x, use_reentrant=False)
            else:
                x = self.transformer[i](x, cu_seqlens=cu_seqlens)
            if self.clip_align and i == self.clip_align_layer_id:
                clip_pred = self._drop_cls_packed(self.clip_projector(x), boundaries)

        x = self.ln_post(x)
        return self._unpack(x, clip_pred, boundaries, layout)

    def _pack(self, z_quantized, image_grid_thw):
        """Build a single packed ``(seq_total + B, D)`` sequence: per image,
        prepend cls and add the (interpolated) pos-embed. Returns
        ``(tokens, cu_seqlens, boundaries, layout)`` where ``boundaries`` is a
        list of ``(h_i, w_i, L_with_cls_i)``. Fixed-res input is split into B
        equal ``grid_size**2`` blocks; variable-res is split by image_grid_thw.
        """
        dtype = z_quantized.dtype
        cls = self.class_embedding.to(dtype)                       # (1, D)
        if image_grid_thw is None:
            B, L, _ = z_quantized.shape
            gs = self.grid_size
            if L != gs * gs:
                raise ValueError(f"Decoder static expects L=grid_size^2={gs*gs}; got {L}.")
            blocks = [(z_quantized[b], gs, gs) for b in range(B)]
            layout = ("dense", B, gs)
        else:
            grid_thw = self._normalize_grid_thw(image_grid_thw, z_quantized.device)
            blocks, off = [], 0
            for row in grid_thw:
                h_i, w_i = int(row[1]), int(row[2])
                L_i = h_i * w_i
                blocks.append((z_quantized[off:off + L_i], h_i, w_i))
                off += L_i
            layout = ("packed",)

        chunks, cu, boundaries = [], [0], []
        for img, h_i, w_i in blocks:
            pos_i = self._interp_pos_embed(h_i, w_i, dtype)        # (h*w + 1, D)
            chunks.append(torch.cat([cls, img], dim=0) + pos_i)
            cu.append(cu[-1] + h_i * w_i + 1)
            boundaries.append((h_i, w_i, h_i * w_i + 1))
        tokens = torch.cat(chunks, dim=0)                          # (seq_total + B, D)
        cu_seqlens = torch.tensor(cu, dtype=torch.int32, device=z_quantized.device)
        return tokens, cu_seqlens, boundaries, layout

    def _unpack(self, x, clip_pred, boundaries, layout):
        """Drop cls + reconstruct RGB from the post-transformer tokens. Fixed-res
        does a single batched conv (all images grid_size**2) and reshapes
        clip_pred to ``(B, gs**2, tw)``; variable-res runs a per-image conv and
        keeps clip_pred packed."""
        if layout[0] == "dense":
            _, B, gs = layout
            L = gs * gs
            feats = x.view(B, L + 1, self.width)[:, 1:1 + L]       # drop cls -> (B, L, D)
            feats = feats.permute(0, 2, 1).reshape(B, self.width, gs, gs)
            image = self.conv_out(self.ffn(feats.contiguous()))    # (B, 3, H, W)
            if clip_pred is not None:
                clip_pred = clip_pred.view(B, L, -1)
            return image, clip_pred

        # variable-res: per-image conv (sizes differ -> list output).
        recon, pos = [], 0
        for (h_i, w_i, L_with_cls_i) in boundaries:
            img_x = x[pos + 1: pos + L_with_cls_i]                 # (h*w, D)
            spatial = img_x.permute(1, 0).reshape(1, self.width, h_i, w_i)
            recon.append(self.conv_out(self.ffn(spatial.contiguous())).squeeze(0))
            pos += L_with_cls_i
        return recon, clip_pred

    @staticmethod
    def _drop_cls_packed(projected, boundaries):
        """Drop the first (cls) token of each image segment in a packed tensor.

        ``projected``: ``(seq_total + B, D)``. Returns ``(seq_total, D)``.
        """
        parts = []
        pos = 0
        for (_, _, L_with_cls_i) in boundaries:
            parts.append(projected[pos + 1: pos + L_with_cls_i])
            pos += L_with_cls_i
        return torch.cat(parts, dim=0)


class FeatureDecoder(nn.Module):
    """Transformer feature decoder for ``feature_recon_only`` mode.

    Consumes post-quant tokens (``feat_post_quant_conv`` output) and reconstructs
    the frozen encoder feature ``z``. Mirrors ``UnifiedDecoder``'s packed-varlen
    machinery (learned positional_embedding interpolated per image + ``cu_seqlens``
    attention) but drops the cls token and the pixel conv head, so the output is
    token-aligned 1:1 with the recon target. Used uniformly for the 16x and 32x
    (merger) encoders — only the grid resolution differs (handled by the caller).
    """
    def __init__(self, config, out_channels):
        super().__init__()
        self.config = config
        vq_cfg = config.model.vq_model
        self.width = vq_cfg.get("vit_dec_width", 1024)
        self.out_channels = out_channels
        self.num_layers = vq_cfg.get("feature_decoder_depth", 3)
        self.num_heads = 16
        self.mlp_ratio = 4.0

        # Reference grid for the learnable positional embedding; interpolated to
        # the actual per-image grid at runtime. 256/16 = 16 by default.
        image_size = config.dataset.preprocessing.crop_size
        patch = vq_cfg.get("vit_dec_patch_size", 16)
        self.base_grid = image_size // patch

        norm_layer = nn.LayerNorm
        scale = self.width ** -0.5
        self.positional_embedding = nn.Parameter(
            scale * torch.randn(self.base_grid ** 2, self.width))

        self.ln_pre = norm_layer(self.width)
        self.transformer = nn.ModuleList([
            Block(
                dim=self.width,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=True,
                qk_norm=False,
                proj_drop=0.,
                attn_drop=0.,
                norm_layer=norm_layer,
                use_swiglu=False,
                use_adaln=False,
            )
            for _ in range(self.num_layers)
        ])
        self.ln_post = norm_layer(self.width)

        # Decode head: RMSNorm -> Linear -> GELU -> Linear(-> z dim).
        self.decode_task_layer = nn.Sequential(
            RMSNorm(self.width),
            nn.Linear(self.width, self.width),
            nn.GELU(),
            nn.Linear(self.width, self.out_channels),
        )

        self.use_checkpoint = vq_cfg.get("use_checkpoint", False)

    def _interp_pos_embed(self, grid_h, grid_w, dtype):
        """Bicubic-interpolate ``positional_embedding`` (no cls) to ``(grid_h, grid_w)``."""
        if grid_h == self.base_grid and grid_w == self.base_grid:
            return self.positional_embedding.to(dtype)
        spatial = self.positional_embedding.reshape(
            self.base_grid, self.base_grid, self.width)
        spatial = spatial.permute(2, 0, 1).unsqueeze(0)
        interp = F.interpolate(
            spatial, size=(grid_h, grid_w),
            mode='bicubic', align_corners=False,
        ).squeeze(0).permute(1, 2, 0).reshape(grid_h * grid_w, self.width)
        return interp.to(dtype)

    @staticmethod
    def _normalize_grid_thw(image_grid_thw, device):
        if not torch.is_tensor(image_grid_thw):
            image_grid_thw = torch.tensor(image_grid_thw, device=device, dtype=torch.long)
        return image_grid_thw.to(device=device, dtype=torch.long)

    def _pack(self, z_quantized, image_grid_thw):
        """Build packed ``(seq_total, D)`` tokens with per-image interpolated
        pos-embed and ``cu_seqlens``. Returns ``(tokens, cu_seqlens, boundaries, layout)``."""
        dtype = z_quantized.dtype
        if image_grid_thw is None:
            B, L, _ = z_quantized.shape
            gs = int(round(L ** 0.5))
            if gs * gs != L:
                raise ValueError(f"FeatureDecoder static expects square L; got L={L}.")
            blocks = [(z_quantized[b], gs, gs) for b in range(B)]
            layout = ("dense", B, gs)
        else:
            grid_thw = self._normalize_grid_thw(image_grid_thw, z_quantized.device)
            blocks, off = [], 0
            for row in grid_thw:
                h_i, w_i = int(row[1]), int(row[2])
                L_i = h_i * w_i
                blocks.append((z_quantized[off:off + L_i], h_i, w_i))
                off += L_i
            layout = ("packed",)

        chunks, cu, boundaries = [], [0], []
        for img, h_i, w_i in blocks:
            pos_i = self._interp_pos_embed(h_i, w_i, dtype)
            chunks.append(img + pos_i)
            cu.append(cu[-1] + h_i * w_i)
            boundaries.append((h_i, w_i, h_i * w_i))
        tokens = torch.cat(chunks, dim=0)
        cu_seqlens = torch.tensor(cu, dtype=torch.int32, device=z_quantized.device)
        return tokens, cu_seqlens, boundaries, layout

    def forward(self, z_quantized, image_grid_thw=None):
        """Decode post-quant tokens to reconstructed features.

        Static (``image_grid_thw=None``): ``z=(B, L, D)`` -> ``(B, L, out_channels)``.
        Dynamic: ``z=(seq_total, D)`` + ``image_grid_thw=(B, 3)`` -> ``(seq_total, out_channels)``.
        """
        tokens, cu_seqlens, boundaries, layout = self._pack(z_quantized, image_grid_thw)
        x = self.ln_pre(tokens)
        for i in range(self.num_layers):
            if self.use_checkpoint:
                x = torch.utils.checkpoint.checkpoint(
                    lambda xx, _blk=self.transformer[i], _cu=cu_seqlens:
                        _blk(xx, cu_seqlens=_cu),
                    x, use_reentrant=False)
            else:
                x = self.transformer[i](x, cu_seqlens=cu_seqlens)
        x = self.ln_post(x)
        out = self.decode_task_layer(x)
        if layout[0] == "dense":
            _, B, gs = layout
            out = out.view(B, gs * gs, self.out_channels)
        return out


class LinearFeatureDecoder(nn.Module):
    """Single-Linear feature decoder (``feature_decoder_type == 'linear'``).

    A token-wise projection from post-quant tokens (``vit_dec_width``) directly to
    the recon-target dim (``out_channels``), with NO token mixing / attention /
    positional embedding / MLP head. Drop-in, call-compatible replacement for the
    transformer ``FeatureDecoder``: ``image_grid_thw`` is accepted for API parity
    but ignored (the op is pointwise, so the token layout is irrelevant). Works
    for both packed ``(seq_total, D)`` and dense ``(B, L, D)`` inputs since
    ``nn.Linear`` only touches the last dim.
    """
    def __init__(self, config, out_channels):
        super().__init__()
        vq_cfg = config.model.vq_model
        # When out_proj is Identity (force_identity), the post-quant tensor
        # stays at code_dim (no reduction), so the feature decoder's input
        # must be code_dim, not vit_dec_width. Otherwise (rmsnorm_mlp etc.)
        # out_proj already reduced to vit_dec_width, so input is vit_dec_width.
        # This mirrors TokenUnmerger's isinstance(out_proj, nn.Identity) check
        # in tokenizer.py — keeps the two consumers of post-quant features
        # consistent.
        if vq_cfg.get("out_proj_type", "") == "force_identity":
            self.width = vq_cfg.get("code_dim", 4096)
        else:
            self.width = vq_cfg.get("vit_dec_width", 1024)
        self.out_channels = out_channels
        self.proj = nn.Linear(self.width, out_channels)

    def forward(self, z_quantized, image_grid_thw=None):
        return self.proj(z_quantized)
