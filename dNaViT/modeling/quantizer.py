# Copyright (c) 2026 LongCat dNaViT Team.
# SPDX-License-Identifier: Apache-2.0

import random
from math import ceil

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from einops import rearrange
from torch.amp import autocast
import numpy as np
from typing import Iterable

from .modules.blocks import RMSNorm


# ---- distributed / dropout helpers -----------------------------------------

def is_distributed():
    return dist.is_initialized() and dist.get_world_size() > 1


def get_maybe_sync_seed(device, max_size=10_000):
    """Draw a random int that is identical across ranks (so quantize-dropout
    picks the same active depth on every rank)."""
    rand_int = torch.randint(0, max_size, (), device=device)
    if is_distributed():
        dist.all_reduce(rand_int)
    return int(rand_int.item())


def round_up_multiple(num, mult):
    return ceil(num / mult) * mult


# ---- rotation-trick STE (Fifty et al. https://arxiv.org/abs/2410.06424) -----

def _rot_l2norm(t, dim=-1, eps=1e-6):
    return F.normalize(t, p=2, dim=dim, eps=eps)


def _rot_safe_div(num, den, eps=1e-6):
    return num / den.clamp(min=eps)


def _rotation_trick_transform(u, q, e):
    """u, q: unit vectors (N, D); e: source (N, D). Householder-style rotation
    that maps u -> q while letting gradients flow into e only (u, q detached)."""
    e = e.unsqueeze(1)                                   # (N, 1, D)
    w = _rot_l2norm(u + q, dim=-1).detach()              # (N, D)
    out = (
        e
        - 2 * (e @ w.unsqueeze(-1) @ w.unsqueeze(1))
        + 2 * (e @ u.unsqueeze(-1).detach() @ q.unsqueeze(1).detach())
    )
    return out.squeeze(1)


def rotate_to(src, tgt):
    """Rotation-trick straight-through: forward value ~= tgt (rotated src),
    gradient path flows to ``src``. Computed in fp32 for numerical stability,
    then cast back to ``src``'s dtype. Accepts any (..., D) leading shape."""
    orig_dtype = src.dtype
    shape = src.shape
    src = src.reshape(-1, shape[-1]).float()
    tgt = tgt.reshape(-1, shape[-1]).float()
    norm_src = src.norm(dim=-1, keepdim=True)
    norm_tgt = tgt.norm(dim=-1, keepdim=True)
    rotated_tgt = _rotation_trick_transform(
        _rot_safe_div(src, norm_src),
        _rot_safe_div(tgt, norm_tgt),
        src,
    )
    rotated = rotated_tgt * _rot_safe_div(norm_tgt, norm_src).detach()
    return rotated.reshape(shape).to(orig_dtype)


# ---- VQ Class -----

class BaseVQ(nn.Module):
    """Base class for all Vector Quantization modules.

    All VQ variants (VQ, RVQ, FSQ) must inherit from this class and implement:
    - forward(z): quantize continuous features -> (z_quantized, result_dict)
    - get_codebook_entry(indices): decode discrete indices -> continuous features
    - codebook_size: property for the effective codebook size
    """
    def __init__(self):
        super().__init__()

    @property
    def codebook_size(self):
        raise NotImplementedError("Subclasses must implement codebook_size")

    def forward(self, z, condition=None):
        raise NotImplementedError("Subclasses must implement forward")

    def get_codebook_entry(self, indices):
        raise NotImplementedError("Subclasses must implement get_codebook_entry")


def _compute_distances_flat(z_flat, codebook):
    """Squared L2 distances between rows of z_flat and rows of codebook.

    z_flat:   (N, D)
    codebook: (K, D)
    returns:  (N, K)
    """
    codebook_t = codebook.t()
    z_norm_sq = z_flat.pow(2).sum(dim=1, keepdim=True)
    c_norm_sq = codebook_t.pow(2).sum(dim=0, keepdim=True)
    return torch.addmm(z_norm_sq + c_norm_sq, z_flat, codebook_t, alpha=-2.0)


def _build_proj(in_dim, out_dim, proj_type="attn_proj", mlp_ratio=4.0, force=True,
                num_heads=16, attn_mlp_ratio=2.0):
    """Build the in/out projection layer for VQ / RVQ."""
    if in_dim is None or out_dim is None:
        return nn.Identity()
    if not force and in_dim == out_dim:
        return nn.Identity()
    if proj_type == "force_identity":
        return nn.Identity()
    if proj_type == "linear":
        return nn.Linear(in_dim, out_dim)
    if proj_type == "mlp":
        hidden = int(mlp_ratio * max(in_dim, out_dim))
        return nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )
    if proj_type == "rmsnorm_mlp":
        # quant_conv / feat_post_quant_conv: pre-norm MLP
        # (RMSNorm -> Linear -> GELU -> Linear). Pointwise (no token mixing);
        # token mixing happens in the FeatureDecoder instead.
        hidden = int(mlp_ratio * max(in_dim, out_dim))
        return nn.Sequential(
            RMSNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )
    if proj_type in ("attn_proj", "attn"):
        return AttnProjection(
            in_dim, out_dim,
            num_heads=num_heads,
            mlp_ratio=attn_mlp_ratio,
        )
    raise ValueError(
        f"Unsupported proj_type: {proj_type!r} "
        "(expected 'linear', 'mlp', 'rmsnorm_mlp', or 'attn_proj')"
    )


class VQ(BaseVQ):
    """Unified Vector Quantization atom.

    Supports three update modes via the (update_mode, enable_simvq) pair:
      - gradient: learnable nn.Embedding codebook, trained by codebook loss.
      - ema:      frozen-as-param codebook updated by EMA over batch assignments.
      - simvq:    frozen N(0,1) buffer + learnable Linear projection.
                  enable_simvq=True forces this mode and ignores ``update_mode``.

    Always active features:
      - EMA usage tracking (used for dead-code restart and perplexity diagnostic).
      - Dead-code restart (threshold = restart_threshold_ratio / n_embed,
        capped at restart_max_per_step per step).
      - Entropy regularization loss (weight controlled by ``entropy_loss_weight``).
      - Diagnostic metrics in result_dict.

    Can also be used as a bare codebook atom inside RVQ via ``quantize_step``,
    which skips in/out projection, STE and loss aggregation.
    """

    def __init__(self,
                 in_channel=None,
                 out_channel=None,
                 n_embed=4096,
                 token_size=32,
                 update_mode="gradient",
                 enable_simvq=False,
                 commitment_cost=0.25,
                 use_l2_norm=False,
                 ema_decay=0.99,
                 restart_unused_codes=True,
                 restart_threshold_ratio=0.03,
                 restart_max_per_step=None,
                 entropy_loss_weight=0.1,
                 sim_weight_norm_codebook=True,
                 sim_sync_buffer=True,
                 sim_codebook_transform="linear",
                 sim_transform_mlp_ratio=2.0,
                 frozen_codebook_dim=None,
                 rotation_trick=False,
                 force_in_out_proj=True,
                 in_proj_type="attn_proj",
                 out_proj_type="attn_proj",
                 proj_mlp_ratio=4.0,
                 proj_num_heads=16,
                 attn_proj_mlp_ratio=2.0,
                 config=None):
        super().__init__()
        self.config = config
        self.n_embed = n_embed
        self.token_size = token_size
        self.commitment_cost = commitment_cost
        self.enable_simvq = enable_simvq

        # ``codebook_l2_norm: true`` maps to weight-normalizing the SimVQ projection
        # (sim_weight_norm_codebook), NOT to L2-normalizing codebook rows / inputs.
        self.use_l2_norm = use_l2_norm
        self.ema_decay = ema_decay
        self.restart_unused_codes = restart_unused_codes
        self.restart_threshold_ratio = restart_threshold_ratio
        self.restart_max_per_step = restart_max_per_step
        self.entropy_loss_weight = entropy_loss_weight
        self.enable_simvq = enable_simvq
        self.sim_weight_norm_codebook = sim_weight_norm_codebook
        self.sim_sync_buffer = sim_sync_buffer
        self.sim_codebook_transform = sim_codebook_transform
        self.rotation_trick = rotation_trick
        self.eps = 1e-5

        # enable_simvq=True forces the simvq update path and ignores update_mode.
        self._update_mode = "simvq" if enable_simvq else update_mode
        if self._update_mode not in ("gradient", "ema", "simvq"):
            raise ValueError(f"Unsupported update_mode: {update_mode}")

        # --- codebook storage ----------------------------------------------------
        if self._update_mode == "simvq":
            # Frozen N(0,1) buffer + learnable projection. N(0,1) init is required
            # so that after the std=1/sqrt(D) projection the projected codes have
            # norm matching the input scale (avoids collapse to single code).
            #
            # ``frozen_codebook_dim`` lets the frozen buffer live in a different
            # (usually smaller) dim than ``token_size`` (decouples codebook
            # capacity from the projection output), à la SimVQ's ``frozen_codebook_dim``.
            # ``sim_codebook_transform`` selects the projection:
            #   "linear" : single nn.Linear (default; supports sim_weight_norm).
            #   "mlp"    : Linear->GELU->Linear (more expressive "larger" SimVQ
            #              weight; sim_weight_norm is a no-op here).
            frozen_dim = frozen_codebook_dim if frozen_codebook_dim is not None else token_size
            self.sim_frozen_dim = frozen_dim
            embed = torch.randn(n_embed, frozen_dim, dtype=torch.float32)
            self.register_buffer("embed", embed)
            if sim_codebook_transform == "mlp":
                hidden = int(sim_transform_mlp_ratio * max(frozen_dim, token_size))
                self.embedding_proj = nn.Sequential(
                    nn.Linear(frozen_dim, hidden),
                    nn.GELU(),
                    nn.Linear(hidden, token_size),
                )
                nn.init.normal_(self.embedding_proj[-1].weight, std=token_size ** -0.5)
                nn.init.zeros_(self.embedding_proj[-1].bias)
            elif sim_codebook_transform == "linear":
                self.embedding_proj = nn.Linear(frozen_dim, token_size)
                nn.init.normal_(self.embedding_proj.weight, std=token_size ** -0.5)
                nn.init.zeros_(self.embedding_proj.bias)
            else:
                raise ValueError(
                    f"Unsupported sim_codebook_transform: {sim_codebook_transform!r} "
                    "(expected 'linear' or 'mlp')")
        else:
            self.embedding = nn.Embedding(n_embed, token_size)
            self.embedding.weight.data.uniform_(-1.0 / n_embed, 1.0 / n_embed)
            if self._update_mode == "ema":
                self.embedding.weight.requires_grad_(False)
                self.register_buffer("cluster_size_ema", torch.zeros(n_embed))
                self.register_buffer("embed_ema", self.embedding.weight.detach().clone())

        # Always-on EMA usage tracker (frequency, sums to 1).
        ema_usage = torch.ones(n_embed) / n_embed
        self.register_buffer("ema_usage", ema_usage)

        # In/out projection layers. ``force_in_out_proj=True`` (default) builds
        # a real projection even when in_channel == token_size; pass False to
        # restore the legacy "skip when dims match" behaviour. When VQ is used
        # as a bare codebook atom inside RVQ, in_channel / out_channel are
        # None and _build_proj returns Identity regardless.
        self.in_proj = _build_proj(
            in_channel, token_size,
            proj_type=in_proj_type, mlp_ratio=proj_mlp_ratio,
            force=force_in_out_proj,
            num_heads=proj_num_heads, attn_mlp_ratio=attn_proj_mlp_ratio)
        self.out_proj = _build_proj(
            token_size, out_channel,
            proj_type=out_proj_type, mlp_ratio=proj_mlp_ratio,
            force=force_in_out_proj,
            num_heads=proj_num_heads, attn_mlp_ratio=attn_proj_mlp_ratio)

    @property
    def codebook_size(self):
        return self.n_embed

    @property
    def update_mode(self):
        return self._update_mode

    @property
    def post_quant_dim(self):
        """Channel dim of post-quant feature exposed in result_dict
        (input dim for a PostQuantSemLoss projector)."""
        return self.token_size

    # ---- codebook accessors -------------------------------------------------

    def _get_codebook(self):
        """Return the effective codebook tensor of shape (n_embed, token_size)."""
        if self._update_mode == "simvq":
            # sim_weight_norm only meaningful for the single-Linear transform
            # (codebook_l2_norm semantics). For the MLP transform we run it
            # directly — its final-layer init already scales the output.
            if self.sim_weight_norm_codebook and isinstance(self.embedding_proj, nn.Linear):
                W = self.embedding_proj.weight
                W_n = W / (W.norm() + self.eps)
                return F.linear(self.embed, W_n, self.embedding_proj.bias)
            return self.embedding_proj(self.embed)
        return self.embedding.weight

    @torch.no_grad()
    def sync_buffer(self):
        """Broadcast SimVQ buffers from rank 0 to all ranks (training only)."""
        if self._update_mode != "simvq" or not self.sim_sync_buffer:
            return
        if not (self.training and dist.is_initialized()):
            return
        for buf in (self.embed, self.ema_usage):
            if dist.get_rank() == 0:
                tmp = buf.clone().detach()
            else:
                tmp = torch.empty_like(buf)
            dist.broadcast(tmp, src=0)
            buf.data.copy_(tmp)
        dist.barrier()

    # ---- EMA / restart helpers ----------------------------------------------

    @torch.no_grad()
    def _update_usage(self, indices):
        if not self.training:
            return
        idxs = indices.reshape(-1)
        n_vec = idxs.numel()
        encodings = F.one_hot(idxs, self.n_embed).to(self.ema_usage.dtype)
        batch_usage = encodings.sum(dim=0) / max(n_vec, 1)
        if dist.is_initialized():
            world = dist.get_world_size()
            dist.all_reduce(batch_usage, op=dist.ReduceOp.SUM)
            batch_usage.div_(world)
        self.ema_usage.data.mul_(self.ema_decay).add_(batch_usage, alpha=1 - self.ema_decay)

    @torch.no_grad()
    def _ema_update_codebook(self, z_flat, indices):
        """EMA codebook update: cluster_size_ema + embed_ema -> embedding.weight."""
        assert self._update_mode == "ema"
        idxs = indices.reshape(-1)
        n_vec = z_flat.shape[0]
        one_hot = z_flat.new_zeros(self.n_embed, n_vec)
        one_hot.scatter_(0, idxs.unsqueeze(0), z_flat.new_ones(1, n_vec))
        cluster_size = one_hot.sum(dim=1)
        vec_sum = one_hot @ z_flat
        if dist.is_initialized():
            dist.all_reduce(vec_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(cluster_size, op=dist.ReduceOp.SUM)
        self.cluster_size_ema.data.mul_(self.ema_decay).add_(cluster_size, alpha=1 - self.ema_decay)
        self.embed_ema.data.mul_(self.ema_decay).add_(vec_sum, alpha=1 - self.ema_decay)
        n = self.cluster_size_ema.sum()
        norm = (n * (self.cluster_size_ema + self.eps) / (n + self.n_embed * self.eps))
        self.embedding.weight.data.copy_(self.embed_ema / norm.reshape(-1, 1))

    @torch.no_grad()
    def _tile_with_noise(self, x, target_n):
        B, D = x.shape
        n_repeats = (target_n + B - 1) // B
        std = x.new_ones(D) * 0.01 / np.sqrt(D)
        x = x.repeat(n_repeats, 1)
        x = x + torch.rand_like(x) * std
        return x

    @torch.no_grad()
    def _restart_dead_codes(self, z_flat):
        if not (self.training and self.restart_unused_codes):
            return
        threshold = self.restart_threshold_ratio / self.n_embed
        dead_mask = self.ema_usage < threshold
        if not dead_mask.any():
            return
        dead_idx = torch.nonzero(dead_mask, as_tuple=False).squeeze(-1)
        if self.restart_max_per_step is None:
            n_reset = dead_idx.numel()
        else:
            n_reset = min(self.restart_max_per_step, dead_idx.numel())
        if n_reset == 0:
            return

        # Pick which dead codes to reset (rank-0 decides, broadcast to others).
        perm = torch.randperm(dead_idx.numel(), device=dead_idx.device)[:n_reset]
        reset_idx = dead_idx[perm]

        # Pick replacement vectors from current batch.
        N = z_flat.shape[0]
        if self._update_mode == "ema":
            # EMA restart: tile with noise to ensure enough samples, then sample.
            src = z_flat if N >= self.n_embed else self._tile_with_noise(z_flat, self.n_embed)
            src_perm = torch.randperm(src.shape[0], device=src.device)[:n_reset]
            new_codes = src[src_perm].detach()
        else:
            if N >= n_reset:
                src_perm = torch.randperm(N, device=z_flat.device)[:n_reset]
                new_codes = z_flat[src_perm].detach()
            else:
                rand = torch.randint(0, N, (n_reset,), device=z_flat.device)
                new_codes = z_flat[rand].detach()

        if dist.is_initialized():
            dist.broadcast(reset_idx, 0)
            dist.broadcast(new_codes, 0)

        if self._update_mode == "simvq":
            # Replacement vectors come from the (token_size-dim) encoder residual.
            # When the frozen buffer shares that dim we copy them in directly;
            # for a decoupled frozen_codebook_dim we instead re-seed the dead
            # rows with fresh N(0,1) (matching the buffer's init distribution),
            # broadcast so all ranks stay in sync.
            if new_codes.shape[-1] == self.embed.shape[-1]:
                self.embed.data[reset_idx] = new_codes
            else:
                repl = torch.randn(
                    reset_idx.numel(), self.embed.shape[-1],
                    device=self.embed.device, dtype=self.embed.dtype)
                if dist.is_initialized():
                    dist.broadcast(repl, 0)
                self.embed.data[reset_idx] = repl
        elif self._update_mode == "ema":
            self.embed_ema.data[reset_idx] = new_codes
            self.cluster_size_ema.data[reset_idx] = 1.0
            self.embedding.weight.data[reset_idx] = new_codes
        else:  # gradient
            self.embedding.weight.data[reset_idx] = new_codes

        self.ema_usage.data[reset_idx] = 1.0 / self.n_embed

    # ---- bare codebook lookup (used by RVQ) ---------------------------------

    def quantize_step(self, z, return_distances=False):
        """Codebook lookup without in/out proj, STE or loss aggregation.

        Args:
            z: (..., token_size) feature tensor (any leading shape).
            return_distances: if True, also return distances (..., n_embed) used
                              for entropy/soft-code computation.

        Returns:
            quant:   (..., token_size). Grad flows through the codebook
                     (gradient mode) or projection (simvq mode); detached in EMA
                     mode after the EMA update.
            indices: (...) long.
            distances (optional): (..., n_embed) with grad through codebook.
        """
        orig_shape = z.shape
        orig_dtype = z.dtype
        z_flat = z.reshape(-1, orig_shape[-1])

        if self._update_mode == "simvq":
            self.sync_buffer()
            z_flat = z_flat.to(torch.float32)
        elif self._update_mode == "ema":
            # EMA codebook statistics must accumulate in fp32: under bf16 the
            # per-step vec_sum (one_hot @ z_flat) and its all-reduce inject a
            # persistent rounding floor that fp32 buffers can't recover.
            z_flat = z_flat.to(torch.float32)

        if self.use_l2_norm:
            z_used = F.normalize(z_flat, dim=-1)
            codebook = F.normalize(self._get_codebook(), dim=-1)
        else:
            z_used = z_flat
            codebook = self._get_codebook()

        # Distances with grad (for entropy + codebook loss when applicable).
        distances = _compute_distances_flat(z_used, codebook)

        with torch.no_grad():
            indices = distances.argmin(dim=-1)

        # EMA update happens before fetching the quant so we use the freshly
        # updated codebook.
        if self._update_mode == "ema" and self.training:
            self._ema_update_codebook(z_used.detach(), indices)
            codebook = F.normalize(self.embedding.weight, dim=-1) if self.use_l2_norm else self.embedding.weight

        quant = F.embedding(indices, codebook)

        if self.training:
            self._update_usage(indices)
            self._restart_dead_codes(z_used.detach())

        quant = quant.reshape(orig_shape).to(orig_dtype)
        indices = indices.reshape(orig_shape[:-1])
        if return_distances:
            distances = distances.reshape(*orig_shape[:-1], self.n_embed)
            return quant, indices, distances
        return quant, indices

    # ---- full forward (standalone usage) ------------------------------------

    @staticmethod
    def _apply_proj(proj, x, cu_seqlens):
        """Forward through a projection module, routing cu_seqlens to
        AttnProjection / PlainAttentionFlash when applicable."""
        if cu_seqlens is None:
            return proj(x)
        if isinstance(proj, (AttnProjection, PlainAttentionFlash)):
            return proj(x, cu_seqlens=cu_seqlens)
        return proj(x)

    @autocast(device_type='cuda', enabled=False)
    def forward(self, z, condition=None, cu_seqlens=None):
        """Quantize z to nearest codebook entry, returning rich result_dict.

        Two operating modes:

        1) Static (default): ``z`` is ``(B, L, C)`` or ``(B, C, H, W)``;
           result keys laid out per-batch (e.g. ``min_encoding_indices`` is
           ``(B, L)``, ``quant_at_depth_with_grad[0]`` is ``(B, L, C)``).

        2) Dynamic varlen: ``z`` is ``(seq_total, C)`` and ``cu_seqlens=(B+1,)``
           int32 marks per-image boundaries; ``in_proj`` and ``out_proj`` get
           ``cu_seqlens`` (AttnProjection uses flash_attn_varlen). Result keys
           collapse the batch dim: ``min_encoding_indices`` is ``(seq_total,)``
           and ``quant_at_depth_with_grad[0]`` is ``(seq_total, C)``.
        """
        z = z.float()

        # ----- Dynamic / varlen branch ------------------------------------
        if cu_seqlens is not None:
            if z.dim() != 2:
                raise ValueError(
                    f"VQ varlen forward expects z of shape (seq_total, C); got {z.shape}"
                )
            z_proj = self._apply_proj(self.in_proj, z, cu_seqlens)        # (seq_total, token_size)
            seq_total, C = z_proj.shape

            # quantize_step handles SimVQ sync, EMA update, usage, restart.
            quant_flat, indices, distances = self.quantize_step(
                z_proj, return_distances=True)
            z_used = F.normalize(z_proj, dim=-1) if self.use_l2_norm else z_proj

            commitment_loss = self.commitment_cost * (quant_flat.detach() - z_used).pow(2).mean()
            codebook_loss = (quant_flat - z_used.detach()).pow(2).mean()

            if self.entropy_loss_weight > 0:
                probs = F.softmax(-distances, dim=-1)
                avg_probs = probs.mean(dim=0)
                if dist.is_initialized():
                    world = dist.get_world_size()
                    dist.all_reduce(avg_probs, op=dist.ReduceOp.SUM)
                    avg_probs = avg_probs / world
                entropy = -(avg_probs * torch.log(avg_probs + 1e-10)).sum()
                entropy_loss = -entropy * self.entropy_loss_weight
            else:
                entropy_loss = z_proj.new_zeros(())

            quantizer_loss = commitment_loss + codebook_loss + entropy_loss
            if self.rotation_trick:
                quant_flat_ste = rotate_to(z_used, quant_flat)
            else:
                quant_flat_ste = z_used + (quant_flat - z_used).detach()
            z_quantized = self._apply_proj(self.out_proj, quant_flat_ste, cu_seqlens)

            with torch.no_grad():
                probs_ema = self.ema_usage + 1e-10
                probs_ema = probs_ema / probs_ema.sum()
                perplexity = torch.exp(-(probs_ema * torch.log(probs_ema)).sum())
                active_thresh = 0.1 / self.n_embed
                active_codes = (self.ema_usage > active_thresh).sum()
                utilization = active_codes.float() / self.n_embed
                codebook_full = self._get_codebook()
                codebook_norm = codebook_full.norm(dim=-1).mean()
                codebook_std = codebook_full.std()

            result_dict = dict(
                quantizer_loss=quantizer_loss,
                commitment_loss=commitment_loss,
                codebook_loss=codebook_loss,
                entropy_loss=entropy_loss,
                perplexity=perplexity,
                utilization=utilization,
                active_codes=active_codes,
                codebook_norm=codebook_norm,
                codebook_std=codebook_std,
                min_encoding_indices=indices,                     # (seq_total,)
                quant_at_depth_with_grad=[quant_flat],            # (seq_total, C)
            )
            return z_quantized, result_dict

        # ----- Static branch ------------------------------
        z = self.in_proj(z)

        original_shape = z.shape
        if len(original_shape) == 4:
            z_seq = rearrange(z, 'b c h w -> b (h w) c')
        elif len(original_shape) == 3:
            z_seq = z
        else:
            raise ValueError(f"Unsupported input shape: {original_shape}")
        B, L, C = z_seq.shape

        z_flat = z_seq.reshape(-1, C)

        # quantize_step handles SimVQ sync, EMA update, usage and restart.
        quant_flat, indices, distances = self.quantize_step(z_flat, return_distances=True)

        # Use the same z-representation as quantize_step for losses.
        z_used = F.normalize(z_flat, dim=-1) if self.use_l2_norm else z_flat

        # Losses
        commitment_loss = self.commitment_cost * (quant_flat.detach() - z_used).pow(2).mean()
        codebook_loss = (quant_flat - z_used.detach()).pow(2).mean()

        if self.entropy_loss_weight > 0:
            probs = F.softmax(-distances, dim=-1)
            avg_probs = probs.mean(dim=0)
            if dist.is_initialized():
                world = dist.get_world_size()
                dist.all_reduce(avg_probs, op=dist.ReduceOp.SUM)
                avg_probs = avg_probs / world
            entropy = -(avg_probs * torch.log(avg_probs + 1e-10)).sum()
            entropy_loss = -entropy * self.entropy_loss_weight
        else:
            entropy_loss = z_flat.new_zeros(())

        quantizer_loss = commitment_loss + codebook_loss + entropy_loss

        # Straight-through estimator.
        if self.rotation_trick:
            quant_flat_ste = rotate_to(z_used, quant_flat)
        else:
            quant_flat_ste = z_used + (quant_flat - z_used).detach()

        # Reshape back to original layout.
        z_quantized = quant_flat_ste.view(B, L, C)
        if len(original_shape) == 4:
            z_quantized = rearrange(z_quantized, 'b (h w) c -> b c h w',
                                    h=original_shape[2], w=original_shape[3])

        z_quantized = self.out_proj(z_quantized)

        # Diagnostics.
        with torch.no_grad():
            probs_ema = self.ema_usage + 1e-10
            probs_ema = probs_ema / probs_ema.sum()
            perplexity = torch.exp(-(probs_ema * torch.log(probs_ema)).sum())
            active_thresh = 0.1 / self.n_embed
            active_codes = (self.ema_usage > active_thresh).sum()
            utilization = active_codes.float() / self.n_embed
            codebook_full = self._get_codebook()
            codebook_norm = codebook_full.norm(dim=-1).mean()
            codebook_std = codebook_full.std()

        result_dict = dict(
            quantizer_loss=quantizer_loss,
            commitment_loss=commitment_loss,
            codebook_loss=codebook_loss,
            entropy_loss=entropy_loss,
            perplexity=perplexity,
            utilization=utilization,
            active_codes=active_codes,
            codebook_norm=codebook_norm,
            codebook_std=codebook_std,
            min_encoding_indices=indices.view(B, L),
            # Pre-STE codebook lookup (with grad through codebook / SimVQ proj),
            # exposed as a length-1 list for symmetry with RVQ's per-depth output.
            quant_at_depth_with_grad=[quant_flat.view(B, L, C)],
        )
        return z_quantized, result_dict

    # ---- decoding -----------------------------------------------------------

    def get_codebook_entry(self, indices):
        """Decode discrete indices to continuous features.

        Args:
            indices: arbitrary-shape long tensor; returns one row per index.

        Returns:
            Tensor of shape (*indices.shape, token_size).
        """
        if not torch.is_tensor(indices):
            indices = torch.tensor(indices)
        if indices.dtype not in (torch.int64, torch.int32):
            indices = indices.long()
        codebook = self._get_codebook()
        if self.use_l2_norm:
            codebook = F.normalize(codebook, dim=-1)
        return F.embedding(indices, codebook)

    # ---- diagnostics --------------------------------------------------------

    @torch.no_grad()
    def get_codebook_usage(self):
        probs = self.ema_usage + 1e-10
        probs = probs / probs.sum()
        perplexity = torch.exp(-(probs * torch.log(probs)).sum())
        threshold = 0.1 / self.n_embed
        active_codes = int((self.ema_usage > threshold).sum().item())
        return {
            "perplexity": perplexity.item(),
            "utilization": active_codes / self.n_embed,
            "active_codes": active_codes,
        }


class IBQ(BaseVQ):
    """Index Backpropagation Quantization (single-level).

    Port of SEED-Voken's ``IndexPropagationQuantize`` into this repo's
    in/out-proj + varlen + result_dict conventions (mirrors ``VQ.forward``).

    Core difference vs ``VQ`` (which is L2-argmin with grad only to the selected
    code / EMA / SimVQ projection):
      - Assignment uses DOT-PRODUCT logits ``<z, codebook>`` (or cosine when
        ``use_l2_norm``), not squared-L2 distance.
      - A soft/hard one-hot straight-through (``hard - soft.detach() + soft``)
        routes the gradient through the softmax to the WHOLE codebook every
        step -- the "index backprop" that trains a plain learnable
        ``nn.Embedding`` codebook (no EMA, no SimVQ projection).
      - Loss = index-backprop recon ``||z_q_soft - z||^2`` + codebook
        ``||sg(z_q_hard) - z||^2`` + commitment ``beta*||z_q_hard - sg(z)||^2``,
        plus the entropy regularizer (sample-min + batch-max) that keeps
        codebook utilization high. An optional VQ-style dead-code restart
        (``restart_unused_codes``) is a second, orthogonal anti-collapse lever:
        it rewrites unused embedding rows without disturbing index-backprop.

    The forward value fed downstream is the hard-quantized vector (STE), so the
    out_proj / FeatureDecoder see the same discrete features as any other VQ.
    """

    def __init__(self,
                 in_channel=None,
                 out_channel=None,
                 n_embed=16384,
                 token_size=1152,
                 commitment_cost=0.25,
                 use_l2_norm=False,
                 entropy_loss_weight=0.1,
                 entropy_temperature=0.01,
                 sample_minimization_weight=1.0,
                 batch_maximization_weight=1.0,
                 restart_unused_codes=True,
                 restart_threshold_ratio=0.03,
                 restart_max_per_step=None,
                 usage_decay=0.99,
                 force_in_out_proj=True,
                 in_proj_type="rmsnorm_mlp",
                 out_proj_type="rmsnorm_mlp",
                 proj_mlp_ratio=4.0,
                 proj_num_heads=16,
                 attn_proj_mlp_ratio=2.0,
                 config=None):
        super().__init__()
        self.config = config
        self.n_embed = n_embed
        self.token_size = token_size
        self.commitment_cost = commitment_cost
        self.use_l2_norm = use_l2_norm
        self.entropy_loss_weight = entropy_loss_weight
        self.entropy_temperature = entropy_temperature
        self.sample_minimization_weight = sample_minimization_weight
        self.batch_maximization_weight = batch_maximization_weight
        # Dead-code restart: identical detection/replacement scheme as VQ's
        # gradient-mode path (revive codes whose EMA usage frequency drops below
        # restart_threshold_ratio / n_embed by overwriting embedding.weight rows
        # with fresh encoder vectors). Orthogonal to index-backprop -- the codebook
        # stays a plain gradient-trained nn.Embedding; restart only rewrites rows
        # that no assignment has touched, so the softmax grad still trains everyone.
        self.restart_unused_codes = restart_unused_codes
        self.restart_threshold_ratio = restart_threshold_ratio
        self.restart_max_per_step = restart_max_per_step
        self.usage_decay = usage_decay
        self.eps = 1e-5

        # Learnable codebook trained by index backprop (dot-product softmax STE).
        # Explicit small-scale init: the tokenizer's _init_weights would re-init
        # nn.Embedding to trunc_normal(std=0.02) anyway, but set it here too so
        # the quantizer is self-consistent when used standalone.
        self.embedding = nn.Embedding(n_embed, token_size)
        nn.init.trunc_normal_(self.embedding.weight, mean=0.0, std=0.02)

        # Rolling buffer of recently-seen codes for the utilization diagnostic
        # (windowed-unique, same口径 as RVQ/SimVQ). Separately, ``ema_usage`` is a
        # per-code EMA of assignment frequency used ONLY for dead-code detection
        # (mirrors VQ.ema_usage); the two never interfere.
        self._usage_buf_len = n_embed * 8
        self.register_buffer(
            "codebook_used", torch.zeros(self._usage_buf_len, dtype=torch.int64))
        self.register_buffer("ema_usage", torch.zeros(n_embed))

        self.in_proj = _build_proj(
            in_channel, token_size,
            proj_type=in_proj_type, mlp_ratio=proj_mlp_ratio,
            force=force_in_out_proj,
            num_heads=proj_num_heads, attn_mlp_ratio=attn_proj_mlp_ratio)
        self.out_proj = _build_proj(
            token_size, out_channel,
            proj_type=out_proj_type, mlp_ratio=proj_mlp_ratio,
            force=force_in_out_proj,
            num_heads=proj_num_heads, attn_mlp_ratio=attn_proj_mlp_ratio)

    @property
    def codebook_size(self):
        return self.n_embed

    @property
    def post_quant_dim(self):
        return self.token_size

    @property
    def code_depth(self):
        # Single-level quantizer; exposed so RVQTokenizer's depth-aware bookkeeping
        # (post_quant_align_depth default = full depth) treats IBQ as depth-1.
        return 1

    @staticmethod
    def _apply_proj(proj, x, cu_seqlens):
        if cu_seqlens is None:
            return proj(x)
        if isinstance(proj, (AttnProjection, PlainAttentionFlash)):
            return proj(x, cu_seqlens=cu_seqlens)
        return proj(x)

    @torch.no_grad()
    def _update_usage(self, indices):
        """Update both usage trackers while training.

        1) Roll the recent-code buffer (windowed unique -> utilization diagnostic,
           same口径 as RVQ/ProductVQ).
        2) EMA the per-code assignment frequency into ``ema_usage`` (all-reduced
           across ranks) so dead-code detection sees a global, smoothed usage.
        Eval forwards mutate neither buffer.
        """
        flat = indices.reshape(-1)
        # (1) windowed-unique buffer.
        buf = self.codebook_used
        cur = flat.shape[0]
        buf_size = buf.shape[0]
        if cur >= buf_size:
            buf.copy_(flat[-buf_size:].to(buf.dtype))
        else:
            buf[:-cur] = buf[cur:].clone()
            buf[-cur:] = flat.to(buf.dtype)
        # (2) EMA usage frequency for dead-code restart.
        n_vec = flat.numel()
        encodings = F.one_hot(flat, self.n_embed).to(self.ema_usage.dtype)
        batch_usage = encodings.sum(dim=0) / max(n_vec, 1)
        if dist.is_initialized():
            dist.all_reduce(batch_usage, op=dist.ReduceOp.SUM)
            batch_usage.div_(dist.get_world_size())
        self.ema_usage.data.mul_(self.usage_decay).add_(batch_usage, alpha=1 - self.usage_decay)

    def _tile_with_noise(self, x, target_n):
        B, D = x.shape
        n_repeats = (target_n + B - 1) // B
        std = x.new_ones(D) * 0.01 / np.sqrt(D)
        x = x.repeat(n_repeats, 1)
        x = x + torch.rand_like(x) * std
        return x

    @torch.no_grad()
    def _restart_dead_codes(self, z_flat):
        """Revive codes whose EMA usage fell below threshold by overwriting their
        embedding rows with fresh in-proj vectors (VQ gradient-mode scheme).

        ``z_flat``: (N, token_size) in-proj features (the same space the codebook
        lives in). Only rewrites dead rows, so the softmax gradient keeps training
        every code; reset codes get ``ema_usage = 1/n_embed`` so they aren't
        immediately re-killed. Rank-0 chooses the reset and broadcasts.
        """
        if not (self.training and self.restart_unused_codes):
            return
        threshold = self.restart_threshold_ratio / self.n_embed
        dead_mask = self.ema_usage < threshold
        if not dead_mask.any():
            return
        dead_idx = torch.nonzero(dead_mask, as_tuple=False).squeeze(-1)
        if self.restart_max_per_step is None:
            n_reset = dead_idx.numel()
        else:
            n_reset = min(self.restart_max_per_step, dead_idx.numel())
        if n_reset == 0:
            return

        perm = torch.randperm(dead_idx.numel(), device=dead_idx.device)[:n_reset]
        reset_idx = dead_idx[perm]

        N = z_flat.shape[0]
        if N >= n_reset:
            src_perm = torch.randperm(N, device=z_flat.device)[:n_reset]
            new_codes = z_flat[src_perm].detach()
        else:
            rand = torch.randint(0, N, (n_reset,), device=z_flat.device)
            new_codes = z_flat[rand].detach()

        if dist.is_initialized():
            dist.broadcast(reset_idx, 0)
            dist.broadcast(new_codes, 0)

        self.embedding.weight.data[reset_idx] = new_codes.to(self.embedding.weight.dtype)
        self.ema_usage.data[reset_idx] = 1.0 / self.n_embed

    def _ib_quantize(self, z_flat):
        """Index-backprop quantization on a (N, token_size) tensor.

        Returns z_q_soft (STE quant, hard forward + whole-codebook grad),
        z_q_hard (pure hard quant), indices, logits (for entropy).
        """
        codebook = self.embedding.weight                     # (K, D)
        if self.use_l2_norm:
            z_used = F.normalize(z_flat, dim=-1)
            cb = F.normalize(codebook, dim=-1)
            logits = z_used @ cb.t()                          # cosine
        else:
            logits = z_flat @ codebook.t()                   # (N, K) dot-product

        soft = F.softmax(logits, dim=-1)
        indices = logits.argmax(dim=-1)                      # (N,)
        hard = F.one_hot(indices, self.n_embed).type_as(soft)
        onehot_ste = hard - soft.detach() + soft             # hard fwd, soft bwd

        z_q_soft = onehot_ste @ codebook                     # grad -> whole codebook
        z_q_hard = hard @ codebook                           # = embedding(indices)
        return z_q_soft, z_q_hard, indices, logits

    def _entropy_loss(self, logits):
        """Non-negative entropy regularizer: sample_min * H(p) +
        batch_max * (logK - H(mean p)).

        ``logits``: (N, K). Minimizing per-sample entropy H(p) sharpens each
        assignment; minimizing (logK - H(mean p)) == maximizing the batch-mean
        entropy spreads usage across codes and is the anti-collapse force. The
        ``logK`` constant makes the batch term >= 0 (since H(mean p) <= logK),
        so the whole entropy loss is >= 0 and never drags the total negative.
        The constant does not depend on parameters, so gradients are IDENTICAL
        to the original ``H(p) - H(mean p)`` form. The batch term is all-reduced
        so the mean distribution is global across ranks.
        """
        probs = F.softmax(logits / self.entropy_temperature, dim=-1)
        log_probs = F.log_softmax(logits / self.entropy_temperature + self.eps, dim=-1)

        avg_probs = probs.mean(dim=0)
        if dist.is_initialized():
            dist.all_reduce(avg_probs, op=dist.ReduceOp.SUM)
            avg_probs = avg_probs / dist.get_world_size()
        avg_entropy = -(avg_probs * torch.log(avg_probs + self.eps)).sum()

        sample_entropy = -(probs * log_probs).sum(dim=-1).mean()
        log_K = float(np.log(self.n_embed))                  # max of H(mean p)
        loss = (self.sample_minimization_weight * sample_entropy
                + self.batch_maximization_weight * (log_K - avg_entropy))
        return loss

    @torch.no_grad()
    def _diagnostics(self):
        """Utilization / codebook-scale diagnostics for the training log.

        Keys are prefixed (``codebook_usage/``, ``codebook_std/``,
        ``codebook_norm/``) so the loss module forwards them and the logger
        renders ``CB-util[...]`` — same contract and windowed-unique口径 as
        RVQ/SimVQ. Single codebook -> a single ``level0`` label.
        """
        cb = self.embedding.weight
        utilization = len(torch.unique(self.codebook_used)) / self.n_embed
        return {
            "codebook_usage/level0": utilization,
            "codebook_std/level0": cb.std(),
            "codebook_norm/level0": cb.norm(dim=-1).mean(),
        }

    def _losses(self, z_q_soft, z_q_hard, z_flat, logits):
        ib_recon = (z_q_soft - z_flat).pow(2).mean()
        codebook_loss = (z_q_hard.detach() - z_flat).pow(2).mean()
        commitment_loss = self.commitment_cost * (z_q_hard - z_flat.detach()).pow(2).mean()
        if self.entropy_loss_weight > 0:
            entropy_loss = self.entropy_loss_weight * self._entropy_loss(logits)
        else:
            entropy_loss = z_flat.new_zeros(())
        quantizer_loss = ib_recon + codebook_loss + commitment_loss + entropy_loss
        return quantizer_loss, commitment_loss, codebook_loss, ib_recon, entropy_loss

    @autocast(device_type='cuda', enabled=False)
    def forward(self, z, condition=None, cu_seqlens=None):
        z = z.float()

        # ----- Dynamic / varlen branch ------------------------------------
        if cu_seqlens is not None:
            if z.dim() != 2:
                raise ValueError(
                    f"IBQ varlen forward expects z of shape (seq_total, C); got {z.shape}")
            z_proj = self._apply_proj(self.in_proj, z, cu_seqlens)   # (seq_total, D)
            z_q_soft, z_q_hard, indices, logits = self._ib_quantize(z_proj)
            (quantizer_loss, commitment_loss, codebook_loss,
             ib_recon_loss, entropy_loss) = self._losses(z_q_soft, z_q_hard, z_proj, logits)
            if self.training:
                self._update_usage(indices)
                self._restart_dead_codes(z_proj.detach())
            z_quantized = self._apply_proj(self.out_proj, z_q_soft, cu_seqlens)

            diag = self._diagnostics()
            result_dict = dict(
                quantizer_loss=quantizer_loss,
                commitment_loss=commitment_loss,
                codebook_loss=codebook_loss,
                ib_recon_loss=ib_recon_loss,
                entropy_loss=entropy_loss,
                min_encoding_indices=indices,                # (seq_total,)
                quant_at_depth_with_grad=[z_q_soft],         # (seq_total, D)
                **diag,
            )
            return z_quantized, result_dict

        # ----- Static branch ----------------------------------------------
        z = self.in_proj(z)
        original_shape = z.shape
        if len(original_shape) == 4:
            z_seq = rearrange(z, 'b c h w -> b (h w) c')
        elif len(original_shape) == 3:
            z_seq = z
        else:
            raise ValueError(f"Unsupported input shape: {original_shape}")
        B, L, C = z_seq.shape
        z_flat = z_seq.reshape(-1, C)

        z_q_soft, z_q_hard, indices, logits = self._ib_quantize(z_flat)
        (quantizer_loss, commitment_loss, codebook_loss,
         ib_recon_loss, entropy_loss) = self._losses(z_q_soft, z_q_hard, z_flat, logits)
        if self.training:
            self._update_usage(indices)
            self._restart_dead_codes(z_flat.detach())

        z_quantized = z_q_soft.view(B, L, C)
        if len(original_shape) == 4:
            z_quantized = rearrange(z_quantized, 'b (h w) c -> b c h w',
                                    h=original_shape[2], w=original_shape[3])
        z_quantized = self.out_proj(z_quantized)

        diag = self._diagnostics()
        result_dict = dict(
            quantizer_loss=quantizer_loss,
            commitment_loss=commitment_loss,
            codebook_loss=codebook_loss,
            ib_recon_loss=ib_recon_loss,
            entropy_loss=entropy_loss,
            min_encoding_indices=indices.view(B, L),
            quant_at_depth_with_grad=[z_q_soft.view(B, L, C)],
            **diag,
        )
        return z_quantized, result_dict

    def get_codebook_entry(self, indices):
        if not torch.is_tensor(indices):
            indices = torch.tensor(indices)
        if indices.dtype not in (torch.int64, torch.int32):
            indices = indices.long()
        return F.embedding(indices, self.embedding.weight)

    @torch.no_grad()
    def get_codebook_usage(self):
        utilization = len(torch.unique(self.codebook_used)) / self.n_embed
        return {
            "utilization": utilization,
            "active_codes": int(len(torch.unique(self.codebook_used))),
        }


class RVQ(BaseVQ):
    """Residual Vector Quantization.

    Multi-layer residual quantization where each depth uses a ``VQ`` atom as
    its codebook. Supports independent ``code_shape`` / ``latent_shape``
    (spatial packing à la SimRQBottleneck), per-step rolling utilization
    metrics and rich decoding APIs (per-depth / partial / soft codes).
    """

    def __init__(self,
                 in_channel,
                 out_channel,
                 n_embed,
                 code_shape,
                 latent_shape=None,
                 codebook_update_mode="ema",
                 enable_simvq=False,
                 shared_codebook=False,
                 shared_group_num=1,
                 sem_group_size=None,
                 commitment_loss_type="cumsum",
                 decay=0.99,
                 restart_unused_codes=True,
                 restart_threshold_ratio=0.03,
                 restart_max_per_step=None,
                 commitment_weight=0.25,
                 entropy_loss_weight=0.0,
                 use_l2_norm=False,
                 sim_weight_norm_codebook=True,
                 sim_sync_buffer=True,
                 sim_codebook_transform="linear",
                 sim_transform_mlp_ratio=2.0,
                 frozen_codebook_dim=None,
                 rotation_trick=False,
                 quantize_dropout=False,
                 quantize_dropout_cutoff_index=0,
                 quantize_dropout_multiple_of=1,
                 force_in_out_proj=True,
                 in_proj_type="attn_proj",
                 out_proj_type="attn_proj",
                 proj_mlp_ratio=4.0,
                 proj_num_heads=16,
                 attn_proj_mlp_ratio=2.0,
                 use_whiten=False,
                 whiten_momentum=0.01,
                 whiten_eps=1e-5,
                 use_center=False,
                 center_momentum=0.01,
                 config=None):
        super().__init__()
        self.config = config
        self.enable_simvq = enable_simvq

        # ``codebook_l2_norm: true`` maps to weight-normalizing the SimVQ
        # projection (sim_weight_norm_codebook), NOT to L2-normalizing the
        # codebook rows / inputs.
        # So we do NOT auto-force use_l2_norm here; sim_weight_norm_codebook defaults
        # to True instead.
        self.use_l2_norm = use_l2_norm

        self.code_shape = tuple(code_shape)
        if latent_shape is None:
            latent_shape = self.code_shape
        self.latent_shape = tuple(latent_shape)
        if len(self.code_shape) != 3 or len(self.latent_shape) != 3:
            raise ValueError("code_shape and latent_shape must both be length-3 (h, w, d).")
        if any(self.latent_shape[i] % self.code_shape[i] != 0 for i in range(2)):
            raise ValueError("latent_shape (h, w) must be divisible by code_shape (h, w).")

        self.shape_divisor = tuple(self.latent_shape[i] // self.code_shape[i] for i in range(3))
        self._reshape_spatial = (self.shape_divisor[0] != 1 or self.shape_divisor[1] != 1)
        # token_size per depth, after spatial packing.
        token_size = (self.shape_divisor[0] * self.shape_divisor[1] * self.latent_shape[2])

        self.shared_codebook = shared_codebook
        self.commitment_loss_type = commitment_loss_type
        if commitment_loss_type not in ("cumsum", "last"):
            raise ValueError(f"Unsupported commitment_loss_type: {commitment_loss_type}")
        self.commitment_weight = commitment_weight
        self._code_depth = self.code_shape[-1]

        # Rotation-trick STE (applied on the aggregate quant -> z_proj at the
        # output) and EnCodec-style structured quantize-dropout (random active
        # depth per step so coarse-only reconstructions stay in-distribution).
        self.rotation_trick = rotation_trick
        self.quantize_dropout = quantize_dropout
        self.quantize_dropout_cutoff_index = quantize_dropout_cutoff_index
        self.quantize_dropout_multiple_of = quantize_dropout_multiple_of

        # in/out proj on the OUTER (un-packed) embedding dim. After in_proj,
        # the channel matches latent_shape[-1] which then gets spatially packed
        # to ``token_size`` for the inner VQ.  ``force_in_out_proj=True`` (the
        # default) builds a real projection even when in_channel ==
        # latent_shape[-1].
        self.in_proj = _build_proj(
            in_channel, self.latent_shape[-1],
            proj_type=in_proj_type, mlp_ratio=proj_mlp_ratio,
            force=force_in_out_proj,
            num_heads=proj_num_heads, attn_mlp_ratio=attn_proj_mlp_ratio)
        self.out_proj = _build_proj(
            self.latent_shape[-1], out_channel,
            proj_type=out_proj_type, mlp_ratio=proj_mlp_ratio,
            force=force_in_out_proj,
            num_heads=proj_num_heads, attn_mlp_ratio=attn_proj_mlp_ratio)

        # Per-channel whitening of the in_proj output (RVQ L2 lookup space).
        # When on, z_proj is standardized to zero-mean/unit-std per channel
        # BEFORE the residual loop so the L2 nearest-neighbour assignment is
        # isotropic and low-variance semantic directions are not starved. The
        # codebook then lives in whitened space; out_proj (trainable) absorbs
        # the inverse. Running stats are EMA-tracked in train, frozen in eval.
        self.use_whiten = use_whiten
        self.whiten_momentum = whiten_momentum
        self.whiten_eps = whiten_eps
        if self.use_whiten:
            d_white = self.latent_shape[-1]
            self.register_buffer("whiten_mean", torch.zeros(d_white))
            self.register_buffer("whiten_var", torch.ones(d_white))
            self.register_buffer("whiten_count", torch.zeros((), dtype=torch.long))

        # Center-only (mean-subtract) variant of whitening: subtract an EMA-tracked
        # per-channel mean from z_proj BEFORE the residual loop, WITHOUT variance
        # normalization. A large shared DC offset in z_proj otherwise dominates the
        # level-0 L2 assignment and collapses the d0 codebook; removing just the
        # mean lets d0 spread while leaving the per-channel scale intact. The
        # codebook lives in centered space; out_proj (trainable) absorbs the
        # constant. Running mean is EMA-tracked in train, frozen in eval.
        self.use_center = use_center
        self.center_momentum = center_momentum
        if self.use_center:
            d_c = self.latent_shape[-1]
            self.register_buffer("center_mean", torch.zeros(d_c))
            self.register_buffer("center_count", torch.zeros((), dtype=torch.long))

        # ---- Codebook grouping ----------------------------------------------
        # ``shared_codebook`` is the on/off switch; ``shared_group_num`` is the
        # granularity within "shared" mode:
        #   shared_codebook=False                       : code_depth independent
        #                                                 codebooks (shared_group_num ignored).
        #   shared_codebook=True, shared_group_num=1    : all depths share 1 codebook.
        #   shared_codebook=True, shared_group_num=G    : depths split into G contiguous
        #                                                 groups (requires code_depth % G == 0),
        #                                                 each group shares one codebook.
        # ``sem_group_size`` overrides the equal split for the semantic/pixel
        # ablation: requires shared_group_num=2, makes group-0 (semantic) the
        # first ``sem_group_size`` depths and group-1 (pixel) the remaining
        # ``code_depth - sem_group_size`` depths. This lifts the equal-divisibility
        # constraint so the semantic:pixel depth ratio is freely tunable.
        if shared_codebook:
            if shared_group_num is None or shared_group_num < 1:
                raise ValueError(
                    f"shared_codebook=True requires shared_group_num >= 1, "
                    f"got {shared_group_num}."
                )
            if sem_group_size is not None:
                if int(shared_group_num) != 2:
                    raise ValueError(
                        f"sem_group_size is only valid with shared_group_num=2 "
                        f"(semantic + pixel), got shared_group_num={shared_group_num}."
                    )
                sem = int(sem_group_size)
                if not (1 <= sem < self._code_depth):
                    raise ValueError(
                        f"sem_group_size must be in [1, code_depth-1]="
                        f"[1, {self._code_depth - 1}], got {sem}."
                    )
                num_groups = 2
                depth_to_group = [0 if d < sem else 1 for d in range(self._code_depth)]
            else:
                if self._code_depth % shared_group_num != 0:
                    raise ValueError(
                        f"code_depth ({self._code_depth}) must be divisible by "
                        f"shared_group_num ({shared_group_num})."
                    )
                num_groups = int(shared_group_num)
                per_group = self._code_depth // num_groups
                depth_to_group = [d // per_group for d in range(self._code_depth)]
        else:
            if sem_group_size is not None:
                raise ValueError(
                    "sem_group_size requires shared_codebook=True (with shared_group_num=2)."
                )
            num_groups = self._code_depth
            depth_to_group = list(range(self._code_depth))
        self.shared_group_num = num_groups if shared_codebook else None
        self._num_groups = num_groups
        self._depth_to_group = depth_to_group
        self._depths_per_group = [
            [d for d in range(self._code_depth) if depth_to_group[d] == g]
            for g in range(num_groups)
        ]
        # Depth count of the semantic (first) group. Equals sem_group_size under
        # the explicit split, the size of group-0 under equal groups, and 1 with
        # no sharing. The tokenizer reads this to align post-quant semantic
        # distillation to the semantic-group boundary.
        self.sem_group_size = len(self._depths_per_group[0])

        # n_embed / decay broadcast: scalar => same across groups; list =>
        # length must match the number of distinct codebooks (== num_groups).
        # For backward compat, we ALSO expose self.n_embed / self.decay as
        # per-DEPTH aliased lists (length = code_depth) so existing call sites
        # like ``self.n_embed[d]`` keep working transparently.
        if isinstance(n_embed, Iterable):
            n_embed_per_group = list(n_embed)
            if len(n_embed_per_group) != num_groups:
                raise ValueError(
                    f"n_embed list length {len(n_embed_per_group)} must equal "
                    f"num_groups={num_groups} (depths sharing a codebook must "
                    "share its size)."
                )
        else:
            n_embed_per_group = [n_embed] * num_groups

        if isinstance(decay, Iterable):
            decay_per_group = list(decay)
            if len(decay_per_group) != num_groups:
                raise ValueError(
                    f"decay list length {len(decay_per_group)} must equal "
                    f"num_groups={num_groups}."
                )
        else:
            decay_per_group = [decay] * num_groups

        self._n_embed_per_group = n_embed_per_group
        self._decay_per_group = decay_per_group
        # Per-depth aliased view (length = code_depth) for backward compat.
        self.n_embed = [n_embed_per_group[depth_to_group[d]] for d in range(self._code_depth)]
        self.decay = [decay_per_group[depth_to_group[d]] for d in range(self._code_depth)]

        vq_kwargs = dict(
            in_channel=None, out_channel=None,
            token_size=token_size,
            update_mode=codebook_update_mode,
            enable_simvq=enable_simvq,
            commitment_cost=0.0,            # RVQ aggregates its own commitment loss
            use_l2_norm=self.use_l2_norm,   # Pass L2 norm setting to inner VQ atoms
            restart_unused_codes=restart_unused_codes,
            restart_threshold_ratio=restart_threshold_ratio,
            restart_max_per_step=restart_max_per_step,
            entropy_loss_weight=entropy_loss_weight,
            sim_weight_norm_codebook=sim_weight_norm_codebook,
            sim_sync_buffer=sim_sync_buffer,
            sim_codebook_transform=sim_codebook_transform,
            sim_transform_mlp_ratio=sim_transform_mlp_ratio,
            frozen_codebook_dim=frozen_codebook_dim,
        )

        # Build ``num_groups`` distinct VQ atoms; alias them into a per-depth
        # ModuleList so downstream code that uses ``self.codebooks[d]`` keeps
        # working without changes (depths in the same group point at the same
        # underlying nn.Module — PyTorch dedupes shared params automatically).
        groups = [
            VQ(n_embed=n_embed_per_group[g], ema_decay=decay_per_group[g], **vq_kwargs)
            for g in range(num_groups)
        ]
        self.codebooks = nn.ModuleList(
            [groups[depth_to_group[d]] for d in range(self._code_depth)]
        )

        # Rolling buffer of recently-seen codes for utilization diagnostic.
        # One buffer slot per UNIQUE codebook (i.e., per group).
        buf_len = n_embed_per_group[0] * 8
        if num_groups == 1:
            # Keep the old (buf_len,) shape for the shared_codebook case so
            # checkpoints from earlier runs still load.
            self.register_buffer("codebook_used", torch.zeros(buf_len, dtype=torch.int64))
        else:
            self.register_buffer(
                "codebook_used", torch.zeros(num_groups, buf_len, dtype=torch.int64))

    @property
    def codebook_size(self):
        return self.n_embed[0] if isinstance(self.n_embed, list) else self.n_embed

    @property
    def code_depth(self):
        return self._code_depth

    @property
    def post_quant_dim(self):
        """Channel dim of per-depth post-quant features exposed in
        result_dict['quant_at_depth_with_grad'] (input dim for a
        PostQuantSemLoss projector)."""
        return self.latent_shape[-1]

    # ---- spatial packing ----------------------------------------------------

    def _to_code_shape(self, x):
        """(B, H, W, D) -> (B, h, w, D')"""
        if not self._reshape_spatial:
            return x
        B, H, W, D = x.shape
        rH, rW, _ = self.shape_divisor
        x = x.reshape(B, H // rH, rH, W // rW, rW, D)
        x = x.permute(0, 1, 3, 2, 4, 5)
        return x.reshape(B, H // rH, W // rW, -1)

    def _to_latent_shape(self, x):
        """(B, h, w, D') -> (B, H, W, D)"""
        if not self._reshape_spatial:
            return x
        B, h, w, _ = x.shape
        D = self.latent_shape[-1]
        rH, rW, _ = self.shape_divisor
        x = x.reshape(B, h, w, rH, rW, D)
        x = x.permute(0, 1, 3, 2, 4, 5)
        return x.reshape(B, h * rH, w * rW, D)

    # ---- core quantize loop -------------------------------------------------

    @staticmethod
    def _apply_proj(proj, x, cu_seqlens):
        """Call a projection module forwarding cu_seqlens iff it accepts it
        (AttnProjection / PlainAttentionFlash do). Linear / MLP / Identity
        ignore the kwarg silently via this dispatch."""
        if cu_seqlens is None:
            return proj(x)
        if isinstance(proj, (AttnProjection, PlainAttentionFlash)):
            return proj(x, cu_seqlens=cu_seqlens)
        return proj(x)

    def _compute_active_depth(self, device):
        """Number of RVQ depths to actually use this forward.

        Training + ``quantize_dropout``: draw a rank-synced random cutoff in
        ``[cutoff_index, code_depth)`` so depths beyond it are dropped from the
        decoder output and the commitment/vq loss (EnCodec structured dropout).
        ``cutoff_index`` guarantees a minimum number of always-kept coarse
        depths (set it >= post_quant_align_depth-1 to keep semantic alignment
        consistent). All depths are still *computed* (codes / usage / dead-code
        restart stay full-depth); only the loss + decoder-facing aggregate are
        truncated. Eval / dropout-off always returns full depth.
        """
        if not (self.training and self.quantize_dropout and self._code_depth > 1):
            return self._code_depth
        cutoff = max(0, min(self.quantize_dropout_cutoff_index, self._code_depth - 1))
        seed = get_maybe_sync_seed(device)
        drop_index = random.Random(seed).randrange(cutoff, self._code_depth)
        if self.quantize_dropout_multiple_of != 1:
            drop_index = round_up_multiple(
                drop_index + 1, self.quantize_dropout_multiple_of) - 1
            drop_index = min(drop_index, self._code_depth - 1)
        return drop_index + 1

    def _quantize_residual(self, x):
        """Run the residual loop.

        Args:
            x: (B, h, w, token_size) or (B, L, token_size) post-packing.

        Returns:
            quant_list: list of cumulative-quant (B, ..., token_size), detached,
                        one entry per depth (last entry == full reconstruction).
            codes: (B, ..., D) long.
            aggregated_with_grad_list: list of cumulative-quant (B, ..., token_size)
                        WITH grad through codebooks, one entry per depth. The last
                        entry is the full grad-carrying sum used for vq_loss; any
                        earlier index k gives the post-quant feature you would get
                        if you only used the first (k+1) RVQ depths.
        """
        residual = x.detach().clone()
        quant_list = []
        code_list = []
        aggregated_with_grad_list = []
        aggregated_detached = torch.zeros_like(x)
        aggregated_with_grad = torch.zeros_like(x)

        for i in range(self._code_depth):
            quant, code = self.codebooks[i].quantize_step(residual)
            quant_detached = quant.detach()
            residual = residual - quant_detached
            aggregated_detached = aggregated_detached + quant_detached
            aggregated_with_grad = aggregated_with_grad + quant
            quant_list.append(aggregated_detached.clone())
            aggregated_with_grad_list.append(aggregated_with_grad)
            code_list.append(code.unsqueeze(-1))

        codes = torch.cat(code_list, dim=-1)
        return quant_list, codes, aggregated_with_grad_list

    def _apply_whiten(self, z_proj):
        """Per-channel standardize z_proj (seq_total, d) in fp32.

        Train: update running mean/var with a debiased EMA over the batch's
        per-channel stats, then normalize with the UPDATED stats. Eval: use the
        frozen running stats. Buffers are fp32; output cast back to z_proj dtype.
        """
        orig_dtype = z_proj.dtype
        zf = z_proj.float()
        if self.training:
            with torch.no_grad():
                batch_mean = zf.mean(dim=0)
                batch_var = zf.var(dim=0, unbiased=False)
                if self.whiten_count == 0:
                    self.whiten_mean.copy_(batch_mean)
                    self.whiten_var.copy_(batch_var)
                else:
                    m = self.whiten_momentum
                    self.whiten_mean.mul_(1 - m).add_(batch_mean, alpha=m)
                    self.whiten_var.mul_(1 - m).add_(batch_var, alpha=m)
                self.whiten_count += 1
        z_white = (zf - self.whiten_mean) / torch.sqrt(self.whiten_var + self.whiten_eps)
        return z_white.to(orig_dtype)

    def _apply_center(self, z_proj):
        """Subtract an EMA-tracked per-channel mean from z_proj (seq_total, d).

        Mean-only variant of ``_apply_whiten`` (no variance division). Train:
        debiased-EMA update of the running mean over the batch, then subtract the
        UPDATED mean. Eval: subtract the frozen running mean. Buffers are fp32;
        output cast back to z_proj dtype.
        """
        orig_dtype = z_proj.dtype
        zf = z_proj.float()
        if self.training:
            with torch.no_grad():
                batch_mean = zf.mean(dim=0)
                if self.center_count == 0:
                    self.center_mean.copy_(batch_mean)
                else:
                    m = self.center_momentum
                    self.center_mean.mul_(1 - m).add_(batch_mean, alpha=m)
                self.center_count += 1
        return (zf - self.center_mean).to(orig_dtype)

    def forward(self, z, condition=None, cu_seqlens=None):
        """Quantize z. Returns (z_quantized, result_dict).

        Single varlen-only code path; dense inputs are auto-converted.

        Input layouts accepted:
          - ``(B, C, H, W)`` dense, ``cu_seqlens=None``: synthesized to
            ``cu_seqlens = arange(B+1) * (H*W)``; output restored to
            ``(B, C, H, W)``.
          - ``(B, L, C)`` dense, ``cu_seqlens=None``: synthesized to
            ``cu_seqlens = arange(B+1) * L``; output restored to ``(B, L, C)``.
          - ``(seq_total, C)`` packed + explicit ``cu_seqlens=(B+1,)``:
            varlen straight through; output is ``(seq_total, out_channel)``.

        ``result_dict['quant_at_depth_with_grad']`` and
        ``result_dict['min_encoding_indices']`` follow the input layout (dense
        → ``(B, ..., D)``; packed → ``(seq_total, depth)``).

        Spatial-packing yamls (``shape_divisor != (1,1,1)``) still hit the
        legacy static branch — varlen is incompatible with spatial packing.

        Dtype: inherits the surrounding autocast context (bf16 under the
        yaml's ``mixed_precision: bf16``) so flash_attn inside ``AttnProjection``
        receives a compatible dtype. The previous ``@autocast(enabled=False)``
        + ``z.float()`` forced fp32 across the whole quantizer and broke flash.
        """

        # Legacy static path for spatial-packing yamls.
        if cu_seqlens is None and self._reshape_spatial:
            return self._forward_static_spatial_packing(z)

        # ---- Dense -> packed conversion (when caller didn't pre-pack) ----
        if cu_seqlens is None:
            if z.dim() == 4:
                B, C, H, W = z.shape
                L = H * W
                z_packed = z.permute(0, 2, 3, 1).reshape(B * L, C)
                dense_layout = ("bchw", B, H, W)
            elif z.dim() == 3:
                B, L, C = z.shape
                z_packed = z.reshape(B * L, C)
                dense_layout = ("blc", B, L)
            else:
                raise ValueError(f"Unsupported input shape: {z.shape}")
            cu_seqlens = torch.arange(
                B + 1, device=z.device, dtype=torch.int32) * L
        else:
            z_packed = z
            dense_layout = None

        # ---- Unified varlen body -----------------------------------------
        # in_proj takes (seq_total, in_channel) + cu_seqlens for AttnProjection;
        # fallback (Linear/MLP/Identity) ignores cu_seqlens.
        z_proj = self._apply_proj(self.in_proj, z_packed, cu_seqlens)  # (seq_total, latent_dim)
        # Quantization math (residual loop, distances, EMA) runs in fp32 while
        # in_proj/out_proj stay under autocast (bf16) so flash_attn projections
        # are unaffected. At large ||z_proj|| bf16 residual accumulation drops
        # low-order bits between depths; fp32 keeps them. Everything leaving the
        # quantizer is cast back to ``quant_dtype`` so downstream is unchanged.
        quant_dtype = z_proj.dtype
        z_proj = z_proj.float()
        if self.use_whiten:
            z_proj = self._apply_whiten(z_proj)
        if self.use_center:
            z_proj = self._apply_center(z_proj)
        quant_list, codes, agg_with_grad_list = self._quantize_residual(z_proj)

        # Structured quantize-dropout: only the first ``n_active`` depths feed
        # the decoder output and the loss this step (codes stay full-depth).
        n_active = self._compute_active_depth(z_proj.device)
        active_idx = n_active - 1
        quants_with_grad = agg_with_grad_list[active_idx]

        if self.commitment_loss_type == "cumsum":
            loss_terms = [(z_proj - q.detach()).pow(2).mean() for q in quant_list]
            commitment_loss = torch.mean(torch.stack(loss_terms[:n_active])) * self.commitment_weight
        else:
            loss_terms = [(z_proj - quant_list[active_idx].detach()).pow(2).mean()]
            commitment_loss = loss_terms[0] * self.commitment_weight
        vq_loss = (quants_with_grad - z_proj.detach()).pow(2).mean()
        quantizer_loss = commitment_loss + vq_loss

        # Per-depth + per-group commit / vq loss for diagnostic logging. For
        # cumsum mode, ``loss_terms[d]`` is the commit loss using the depth-d
        # cumulative quant; for "last" mode there's only one term (last
        # depth's residual). Per-group: aggregate the per-depth terms over
        # the depths assigned to each group via ``self._depths_per_group``
        # (matches the labelling used by codebook_usage / std / norm).
        per_depth_commit = {
            f"commit_per_depth/d{i}": t.detach()
            for i, t in enumerate(loss_terms)
        }
        per_group_commit = {}
        per_group_vq = {}
        for g, depths_g in enumerate(self._depths_per_group):
            lbl = self._diag_label(g)
            # commitment: average of per-depth terms within this group.
            if self.commitment_loss_type == "cumsum":
                grp_terms = [loss_terms[d] for d in depths_g]
                per_group_commit[f"commit_per_group/{lbl}"] = (
                    torch.stack(grp_terms).mean().detach())
            else:
                # "last" mode only has 1 term; show it under the last group
                # that contains the final depth.
                if (self._code_depth - 1) in depths_g:
                    per_group_commit[f"commit_per_group/{lbl}"] = loss_terms[0].detach()
            # vq: MSE between the per-depth aggregated-quant (cumulative up
            # to the LAST depth in this group) and z_proj. This shows how
            # well each group's cumulative quantization reconstructs z.
            last_d_in_g = depths_g[-1]
            agg_g = agg_with_grad_list[last_d_in_g]
            per_group_vq[f"vq_per_group/{lbl}"] = (
                (agg_g - z_proj.detach()).pow(2).mean().detach())

        # STE on the final aggregated quant (still (seq_total, latent_dim)).
        if self.rotation_trick:
            quants_trunc = rotate_to(z_proj, quant_list[active_idx])
        else:
            quants_trunc = z_proj + (quant_list[active_idx] - z_proj).detach()
        # Back to autocast dtype for out_proj (flash-safe); quant math was fp32.
        quants_trunc = quants_trunc.to(quant_dtype)
        z_quantized = self._apply_proj(self.out_proj, quants_trunc, cu_seqlens)

        # Per-depth cumulative-with-grad tensors stay as (seq_total, latent_dim);
        # cast back to autocast dtype so downstream consumers are unchanged.
        quant_at_depth_with_grad = [q.to(quant_dtype) for q in agg_with_grad_list]

        # ---- Restore dense layout if caller passed dense -----------------
        if dense_layout is not None:
            kind = dense_layout[0]
            out_dim = z_quantized.shape[-1]
            ld = quant_at_depth_with_grad[0].shape[-1]
            cd = codes.shape[-1]
            if kind == "bchw":
                _, B, H, W = dense_layout
                z_quantized = z_quantized.view(B, H, W, out_dim).permute(0, 3, 1, 2).contiguous()
                quant_at_depth_with_grad = [
                    q.view(B, H * W, ld) for q in quant_at_depth_with_grad]
                codes = codes.view(B, H, W, cd)
            else:  # "blc"
                _, B, L = dense_layout
                z_quantized = z_quantized.view(B, L, out_dim)
                quant_at_depth_with_grad = [
                    q.view(B, L, ld) for q in quant_at_depth_with_grad]
                codes = codes.view(B, L, cd)

        usage_dict = self._update_and_get_usage(codes)
        result_dict = dict(
            quantizer_loss=quantizer_loss,
            commitment_loss=commitment_loss,
            vq_loss=vq_loss,
            min_encoding_indices=codes,
            quant_at_depth_with_grad=quant_at_depth_with_grad,
            **usage_dict,
            **per_depth_commit,
            **per_group_commit,
            **per_group_vq,
        )
        return z_quantized, result_dict

    def _forward_static_spatial_packing(self, z):
        """Legacy static path for spatial-packing yamls (shape_divisor != (1,1,1)).

        Kept verbatim from the pre-unification implementation; varlen is
        incompatible with spatial packing because the per-position residual
        loop operates on packed-channel tokens, which requires a known H/W
        to invert the packing.
        """
        # Normalize to channel-last (B, L, C_in) BEFORE in_proj so the
        # projection always sees the channel on the last dim. AttnProjection's
        # attention only accepts (B, L, D) / (seq_total, D), and even nn.Linear
        # would otherwise project the W axis of a (B, C, H, W) tensor — so the
        # spatial dims must be flattened here first.
        original_shape = z.shape
        if len(original_shape) == 4:
            B, C_in, H, W = original_shape
            z_blc = z.permute(0, 2, 3, 1).reshape(B, H * W, C_in)
        elif len(original_shape) == 3:
            B, L, C_in = original_shape
            # Spatial-packing mode: latent_shape[:2] must match the encoder grid.
            H, W = self.latent_shape[0], self.latent_shape[1]
            if L != H * W:
                raise ValueError(
                    f"RVQ forward: L={L} does not match latent_shape H*W={H*W}.")
            z_blc = z
        else:
            raise ValueError(f"Unsupported input shape: {original_shape}")

        z_blc = self.in_proj(z_blc)                       # (B, L, C_latent)
        C = z_blc.shape[-1]
        z_bhwc = z_blc.view(B, H, W, C)

        z_packed = self._to_code_shape(z_bhwc)
        quant_list, codes, aggregated_with_grad_list = self._quantize_residual(z_packed)

        n_active = self._compute_active_depth(z_packed.device)
        active_idx = n_active - 1
        quants_with_grad = aggregated_with_grad_list[active_idx]

        if self.commitment_loss_type == "cumsum":
            loss_terms = [(z_packed - q.detach()).pow(2).mean() for q in quant_list]
            commitment_loss = torch.mean(torch.stack(loss_terms[:n_active])) * self.commitment_weight
        else:
            commitment_loss = (z_packed - quant_list[active_idx].detach()).pow(2).mean() * self.commitment_weight

        vq_loss = (quants_with_grad - z_packed.detach()).pow(2).mean()
        quantizer_loss = commitment_loss + vq_loss

        quants_trunc = self._to_latent_shape(quant_list[active_idx])
        if self.rotation_trick:
            quants_trunc = rotate_to(z_bhwc, quants_trunc)
        else:
            quants_trunc = z_bhwc + (quants_trunc - z_bhwc).detach()

        # out_proj on channel-last (B, L, C_latent); restore caller's layout.
        quants_blc = quants_trunc.reshape(B, H * W, C)
        z_quantized = self.out_proj(quants_blc)           # (B, L, C_out)
        if len(original_shape) == 4:
            out_dim = z_quantized.shape[-1]
            z_quantized = z_quantized.view(B, H, W, out_dim).permute(0, 3, 1, 2).contiguous()
        # 3D input: keep (B, L, C_out)

        quant_at_depth_with_grad = []
        for agg in aggregated_with_grad_list:
            unpacked = self._to_latent_shape(agg)
            quant_at_depth_with_grad.append(
                unpacked.reshape(unpacked.shape[0], -1, unpacked.shape[-1]))

        usage_dict = self._update_and_get_usage(codes)
        result_dict = dict(
            quantizer_loss=quantizer_loss,
            commitment_loss=commitment_loss,
            vq_loss=vq_loss,
            min_encoding_indices=codes,
            quant_at_depth_with_grad=quant_at_depth_with_grad,
            **usage_dict,
        )
        return z_quantized, result_dict

    # ---- diagnostics --------------------------------------------------------

    def _diag_label(self, g: int) -> str:
        """Per-group diagnostic key suffix. ``shared`` when all depths share a
        single codebook, ``d{g}`` when each depth has its own (g indexes the
        depth), ``g{g}`` otherwise (true group-sharing mode).
        """
        if self._num_groups == 1:
            return "shared"
        if self._num_groups == self._code_depth:
            return f"d{g}"
        return f"g{g}"

    @torch.no_grad()
    def _update_and_get_usage(self, codes):
        """Update the rolling code buffer and return per-group diagnostics.

        ``codes`` shape: ``(B, h, w, code_depth)`` (static) or
        ``(seq_total, code_depth)`` (varlen). We split codes by depth-to-group
        mapping and roll one buffer per unique codebook.

        Mutates ``self.codebook_used`` only when ``self.training`` so the
        diagnostic doesn't pick up discriminator-step (eval+no_grad) forwards
        — those would otherwise pollute utilization metrics and leak into
        checkpoints.

        Key naming convention:
          - num_groups == 1   (shared_codebook)        : keys end with ``shared``
          - num_groups == code_depth (default)          : keys end with ``d{d}``
          - otherwise (group_shared)                    : keys end with ``g{g}``
        """
        out = {}
        n_embed_pg = self._n_embed_per_group

        # Code tensor's last dim is code_depth; split by depths_per_group.
        for g in range(self._num_groups):
            depths_g = self._depths_per_group[g]
            # Gather all codes assigned to group g across selected depths.
            flat = codes[..., depths_g].reshape(-1)
            buf = self.codebook_used if self._num_groups == 1 else self.codebook_used[g]
            cur = flat.shape[0]
            buf_size = buf.shape[0]
            if self.training:
                if cur >= buf_size:
                    buf.copy_(flat[-buf_size:].to(buf.dtype))
                else:
                    buf[:-cur] = buf[cur:].clone()
                    buf[-cur:] = flat.to(buf.dtype)

            lbl = self._diag_label(g)
            out[f"codebook_usage/{lbl}"] = len(torch.unique(buf)) / n_embed_pg[g]
            cb = self.codebooks[depths_g[0]]._get_codebook()
            out[f"codebook_std/{lbl}"] = cb.std()
            out[f"codebook_norm/{lbl}"] = cb.norm(dim=-1).mean()
        return out

    # ---- decoding APIs ------------------------------------------------------

    def _stack_depth_embeds(self, code, to_latent_shape):
        """code: (B, h, w, D) or (B, L, D). Returns embeds list of (B, ..., token_size)."""
        code_slices = torch.chunk(code, chunks=code.shape[-1], dim=-1)
        embeds = []
        for i, cs in enumerate(code_slices):
            cb = self.codebooks[0] if self.shared_codebook else self.codebooks[i]
            codebook = cb._get_codebook()
            if cb.use_l2_norm:
                codebook = F.normalize(codebook, dim=-1)
            e = F.embedding(cs.squeeze(-1), codebook)
            embeds.append(e)
        if to_latent_shape and self._reshape_spatial:
            embeds = [self._to_latent_shape(e) for e in embeds]
        return embeds

    @torch.no_grad()
    def get_codebook_entry(self, indices):
        """Sum-over-depth decoding + spatial unpack.

        Args:
            indices: (B, h, w, D) or (B, L, D) long tensor.

        Returns:
            Tensor matching the un-packed spatial shape (channel last).
        """
        # Accept both spatial (B, h, w, D) and flattened (B, L, D) formats.
        if indices.dim() == 4:
            embeds = self._stack_depth_embeds(indices, to_latent_shape=False)
            agg = torch.stack(embeds, dim=-2).sum(-2)
            agg = self._to_latent_shape(agg)
            return self.out_proj(agg)
        elif indices.dim() == 3:
            B, L, D = indices.shape
            H, W = self.code_shape[0], self.code_shape[1]
            if L == H * W and self._reshape_spatial:
                idx_4d = indices.view(B, H, W, D)
                embeds = self._stack_depth_embeds(idx_4d, to_latent_shape=False)
                agg = torch.stack(embeds, dim=-2).sum(-2)
                agg = self._to_latent_shape(agg)
                # Return as (B, L', C) to match the RVQTokenizer.decode_tokens contract.
                agg = agg.reshape(B, -1, agg.shape[-1])
                return self.out_proj(agg)
            embeds = self._stack_depth_embeds(indices, to_latent_shape=False)
            agg = torch.stack(embeds, dim=-2).sum(-2)
            return self.out_proj(agg)
        else:
            raise ValueError(f"Unsupported indices shape: {indices.shape}")

    @torch.no_grad()
    def embed_code_with_depth(self, code, to_latent_shape=False):
        """Per-depth embeddings without summing. Returns (embeds, None).

        embeds shape: (..., D, token_size) (D = code_depth).
        """
        embeds = self._stack_depth_embeds(code, to_latent_shape=to_latent_shape)
        return torch.stack(embeds, dim=-2), None

    @torch.no_grad()
    def embed_partial_code(self, code, code_idx, mode="select"):
        """Decode only the first ``code_idx + 1`` depths (coarse-to-fine).

        Args:
            code: (B, h, w, D) or (B, L, D).
            code_idx: depth index.
            mode: 'select' (only that depth) or 'add' (sum over depths 0..k).
        """
        assert 0 <= code_idx < code.shape[-1]
        embeds = self._stack_depth_embeds(code, to_latent_shape=False)
        if mode == "select":
            out = embeds[code_idx]
        elif mode == "add":
            out = torch.stack(embeds[:code_idx + 1], dim=-2).sum(-2)
        else:
            raise NotImplementedError(f"Unsupported partial-decode mode: {mode}")
        if self._reshape_spatial and out.dim() == 4:
            out = self._to_latent_shape(out)
        return self.out_proj(out)

    @torch.no_grad()
    def get_soft_codes(self, z, temp=1.0, stochastic=False):
        """Soft assignment over depths (optionally with sampling).

        Args:
            z: (B, L, C) or (B, C, H, W) latent (pre-in_proj already done).
            temp: softmax temperature.
            stochastic: if True, sample from the softmax instead of argmin.

        Returns:
            soft_code: (..., D, n_embed).
            code: (..., D) long.
        """
        z = z.float()
        z = self.in_proj(z)
        if z.dim() == 4:
            z_bhwc = z.permute(0, 2, 3, 1).contiguous()
        else:
            B, L, C = z.shape
            if self._reshape_spatial:
                H, W = self.latent_shape[0], self.latent_shape[1]
                z_bhwc = z.view(B, H, W, C)
            else:
                z_bhwc = z.view(B, L, 1, C)
        z_packed = self._to_code_shape(z_bhwc)

        residual = z_packed.detach().clone()
        soft_list, code_list = [], []
        for i in range(self._code_depth):
            cb = self.codebooks[0] if self.shared_codebook else self.codebooks[i]
            codebook = cb._get_codebook()
            if cb.use_l2_norm:
                codebook = F.normalize(codebook, dim=-1)
            flat = residual.reshape(-1, residual.shape[-1])
            distances = _compute_distances_flat(flat, codebook)
            soft = F.softmax(-distances / temp, dim=-1)
            if stochastic:
                idx = torch.multinomial(soft, 1).squeeze(-1)
            else:
                idx = distances.argmin(dim=-1)
            quant = F.embedding(idx, codebook).reshape(residual.shape)
            residual = residual - quant
            soft_list.append(soft.reshape(*residual.shape[:-1], -1).unsqueeze(-2))
            code_list.append(idx.reshape(residual.shape[:-1]).unsqueeze(-1))

        soft_code = torch.cat(soft_list, dim=-2)
        code = torch.cat(code_list, dim=-1)
        return soft_code, code


class ProductVQ(BaseVQ):
    """Product Quantization (channel-split multi-codebook).

    Splits the post-in_proj latent ``code_dim`` into ``num_groups`` contiguous,
    equal-width sub-vectors and quantizes each independently with its own ``VQ``
    atom (parallel, NOT residual). One patch token therefore maps to
    ``num_groups`` discrete codes. Contrast with ``RVQ`` (full-dim, sequential
    residual refinement over ``rvq_depth`` depths).

    Drop-in for the feature-recon path: same ``forward(z, cu_seqlens=...) ->
    (z_quantized, result_dict)`` contract and the same result_dict keys
    (``quantizer_loss`` / ``commitment_loss`` / ``vq_loss`` /
    ``min_encoding_indices`` / ``quant_at_depth_with_grad`` + per-group
    diagnostics) so FeatureDecoder + losses need no changes. ``code_depth``
    reports ``num_groups`` so one patch -> ``num_groups`` tokens downstream.

    Codebook update mode is inherited by each atom (``ema`` / ``gradient`` /
    ``simvq``); this codebase defaults to EMA for parity with the RVQ baseline.
    """

    def __init__(self,
                 in_channel,
                 out_channel,
                 code_dim,
                 num_groups,
                 n_embed,
                 group_dim=None,
                 codebook_update_mode="ema",
                 enable_simvq=False,
                 commitment_loss_type="cumsum",
                 commitment_weight=0.25,
                 decay=0.99,
                 restart_unused_codes=True,
                 restart_threshold_ratio=0.03,
                 restart_max_per_step=None,
                 use_l2_norm=False,
                 entropy_loss_weight=0.0,
                 sim_weight_norm_codebook=True,
                 sim_sync_buffer=True,
                 sim_codebook_transform="linear",
                 sim_transform_mlp_ratio=2.0,
                 frozen_codebook_dim=None,
                 force_in_out_proj=True,
                 in_proj_type="attn_proj",
                 out_proj_type="attn_proj",
                 proj_mlp_ratio=4.0,
                 proj_num_heads=16,
                 attn_proj_mlp_ratio=2.0,
                 pre_pq_l2_norm=False,
                 config=None):
        super().__init__()
        self.config = config
        self.use_l2_norm = use_l2_norm
        self.pre_pq_l2_norm = bool(pre_pq_l2_norm)
        self.commitment_weight = commitment_weight
        self.entropy_loss_weight = entropy_loss_weight

        if code_dim % num_groups != 0:
            raise ValueError(
                f"code_dim ({code_dim}) must be divisible by num_groups ({num_groups}).")
        self.code_dim = int(code_dim)
        self._num_groups = int(num_groups)
        self.sub_dim = self.code_dim // self._num_groups
        # ``group_dim`` is the per-atom codebook width. Default = sub_dim (the
        # non-overlapping channel-split). Overlapping / factorized subclasses set
        # it to a wider value (each dim covered by >1 group) and provide their own
        # ``_analyze`` / ``_synthesize`` mapping code_dim <-> num_groups x group_dim.
        self.group_dim = int(group_dim) if group_dim is not None else self.sub_dim
        self._n_embed = int(n_embed)

        # in/out proj on the OUTER embedding dim: encoder.width <-> code_dim.
        # Quantization happens on the code_dim latent, channel-split into groups.
        self.in_proj = _build_proj(
            in_channel, self.code_dim,
            proj_type=in_proj_type, mlp_ratio=proj_mlp_ratio,
            force=force_in_out_proj,
            num_heads=proj_num_heads, attn_mlp_ratio=attn_proj_mlp_ratio)
        self.out_proj = _build_proj(
            self.code_dim, out_channel,
            proj_type=out_proj_type, mlp_ratio=proj_mlp_ratio,
            force=force_in_out_proj,
            num_heads=proj_num_heads, attn_mlp_ratio=attn_proj_mlp_ratio)

        # One VQ atom per group; each owns a (n_embed, sub_dim) codebook updated
        # via the configured mode (EMA / gradient / simvq) inside quantize_step.
        vq_kwargs = dict(
            in_channel=None, out_channel=None,
            token_size=self.group_dim,
            update_mode=codebook_update_mode,
            enable_simvq=enable_simvq,
            commitment_cost=0.0,            # ProductVQ aggregates its own losses.
            use_l2_norm=self.use_l2_norm,
            ema_decay=decay,
            restart_unused_codes=restart_unused_codes,
            restart_threshold_ratio=restart_threshold_ratio,
            restart_max_per_step=restart_max_per_step,
            entropy_loss_weight=entropy_loss_weight,
            sim_weight_norm_codebook=sim_weight_norm_codebook,
            sim_sync_buffer=sim_sync_buffer,
            sim_codebook_transform=sim_codebook_transform,
            sim_transform_mlp_ratio=sim_transform_mlp_ratio,
            frozen_codebook_dim=frozen_codebook_dim,
        )
        self.codebooks = nn.ModuleList(
            [VQ(n_embed=self._n_embed, **vq_kwargs) for _ in range(self._num_groups)])

        # Rolling buffer of recently-seen codes for the utilization diagnostic,
        # one row per group.
        buf_len = self._n_embed * 8
        self.register_buffer(
            "codebook_used", torch.zeros(self._num_groups, buf_len, dtype=torch.int64))

    # ---- interface properties (mirror RVQ) ----------------------------------
    @property
    def codebook_size(self):
        return self._n_embed

    @property
    def code_depth(self):
        # One patch -> num_groups parallel tokens.
        return self._num_groups

    @property
    def post_quant_dim(self):
        return self.code_dim

    @staticmethod
    def _apply_proj(proj, x, cu_seqlens):
        if cu_seqlens is None:
            return proj(x)
        if isinstance(proj, (AttnProjection, PlainAttentionFlash)):
            return proj(x, cu_seqlens=cu_seqlens)
        return proj(x)

    def _analyze(self, z_proj):
        """Map the (seq, code_dim) latent to ``num_groups`` per-group sub-vectors,
        each (seq, group_dim). Base ProductVQ = disjoint channel split. Overlapping
        / factorized subclasses override this."""
        return list(z_proj.split(self.sub_dim, dim=-1))

    def _synthesize(self, quant_list):
        """Inverse of ``_analyze``: combine ``num_groups`` per-group quantized
        sub-vectors back into a single (seq, code_dim) latent. Base = concat."""
        return torch.cat(quant_list, dim=-1)

    def forward(self, z, condition=None, cu_seqlens=None):
        """Quantize z. Returns (z_quantized, result_dict).

        Accepts the same layouts as ``RVQ.forward``: packed ``(seq_total, C)``
        with explicit ``cu_seqlens`` (the feature-recon t2i path) or dense
        ``(B, L, C)`` / ``(B, C, H, W)`` which are auto-packed and restored.
        """
        z = z.float()

        # ---- Dense -> packed conversion (when caller didn't pre-pack) ----
        if cu_seqlens is None:
            if z.dim() == 4:
                B, C, H, W = z.shape
                L = H * W
                z_packed = z.permute(0, 2, 3, 1).reshape(B * L, C)
                dense_layout = ("bchw", B, H, W)
            elif z.dim() == 3:
                B, L, C = z.shape
                z_packed = z.reshape(B * L, C)
                dense_layout = ("blc", B, L)
            else:
                raise ValueError(f"Unsupported input shape: {z.shape}")
            cu_seqlens = torch.arange(
                B + 1, device=z.device, dtype=torch.int32) * L
        else:
            z_packed = z
            dense_layout = None

        # ---- in_proj -> channel-split -> per-group quantize ----
        z_proj = self._apply_proj(self.in_proj, z_packed, cu_seqlens)   # (seq, code_dim)
        chunks = self._analyze(z_proj)                                 # num_groups x (seq, group_dim)

        # Optional: L2-normalize each group's sub-vector before quantization
        # (pre-PQ normalization). The "feature" being quantized is then the
        # unit-norm direction of each chunk; commitment / VQ / STE all operate
        # on this normalized latent so the loss landscape is consistent.
        if self.pre_pq_l2_norm:
            chunks = [F.normalize(c, dim=-1) for c in chunks]
            z_proj = self._synthesize(chunks)

        quant_chunks = []
        code_list = []
        per_group_commit = {}
        per_group_vq = {}
        for g, atom in enumerate(self.codebooks):
            zc = chunks[g]
            quant_g, idx_g = atom.quantize_step(zc)
            quant_chunks.append(quant_g)
            code_list.append(idx_g.unsqueeze(-1))
            lbl = f"g{g}"
            per_group_commit[f"commit_per_group/{lbl}"] = (
                (zc - quant_g.detach()).pow(2).mean().detach())
            per_group_vq[f"vq_per_group/{lbl}"] = (
                (quant_g - zc.detach()).pow(2).mean().detach())

        quant = self._synthesize(quant_chunks)         # (seq, code_dim); detached in EMA mode
        codes = torch.cat(code_list, dim=-1)           # (seq, num_groups)

        # ---- losses on the full concatenated latent ----
        commitment_loss = self.commitment_weight * (z_proj - quant.detach()).pow(2).mean()
        vq_loss = (quant - z_proj.detach()).pow(2).mean()
        quantizer_loss = commitment_loss + vq_loss

        # ---- STE: grad flows z_proj -> in_proj; quant carried straight-through ----
        quants_ste = z_proj + (quant - z_proj).detach()
        z_quantized = self._apply_proj(self.out_proj, quants_ste, cu_seqlens)

        # Interface parity with RVQ: list indexable up to code_depth-1. PQ has no
        # residual cumulation, so every entry is the full grad-carrying post-quant
        # (only consumed when a post-quant semantic projector exists; off here).
        quant_at_depth_with_grad = [quants_ste for _ in range(self._num_groups)]

        # ---- Restore dense layout if caller passed dense ----
        if dense_layout is not None:
            kind = dense_layout[0]
            out_dim = z_quantized.shape[-1]
            ld = quants_ste.shape[-1]
            cd = codes.shape[-1]
            if kind == "bchw":
                _, B, H, W = dense_layout
                z_quantized = z_quantized.view(B, H, W, out_dim).permute(0, 3, 1, 2).contiguous()
                quant_at_depth_with_grad = [q.view(B, H * W, ld) for q in quant_at_depth_with_grad]
                codes = codes.view(B, H, W, cd)
            else:  # "blc"
                _, B, L = dense_layout
                z_quantized = z_quantized.view(B, L, out_dim)
                quant_at_depth_with_grad = [q.view(B, L, ld) for q in quant_at_depth_with_grad]
                codes = codes.view(B, L, cd)

        usage_dict = self._update_and_get_usage(codes)
        result_dict = dict(
            quantizer_loss=quantizer_loss,
            commitment_loss=commitment_loss,
            vq_loss=vq_loss,
            min_encoding_indices=codes,
            quant_at_depth_with_grad=quant_at_depth_with_grad,
            **usage_dict,
            **per_group_commit,
            **per_group_vq,
        )
        return z_quantized, result_dict

    @torch.no_grad()
    def _update_and_get_usage(self, codes):
        """Roll the per-group code buffer and return utilization diagnostics.

        ``codes`` last dim is ``num_groups``. Mutates the buffer only while
        training so eval/discriminator forwards don't pollute the metric.
        """
        out = {}
        for g in range(self._num_groups):
            flat = codes[..., g].reshape(-1)
            buf = self.codebook_used[g]
            cur = flat.shape[0]
            buf_size = buf.shape[0]
            if self.training:
                if cur >= buf_size:
                    buf.copy_(flat[-buf_size:].to(buf.dtype))
                else:
                    buf[:-cur] = buf[cur:].clone()
                    buf[-cur:] = flat.to(buf.dtype)
            lbl = f"g{g}"
            out[f"codebook_usage/{lbl}"] = len(torch.unique(buf)) / self._n_embed
            cb = self.codebooks[g]._get_codebook()
            out[f"codebook_std/{lbl}"] = cb.std()
            out[f"codebook_norm/{lbl}"] = cb.norm(dim=-1).mean()
        return out

    @torch.no_grad()
    def get_codebook_entry(self, indices):
        """Decode PQ codes -> latent -> out_proj.

        Args:
            indices: (..., num_groups) long tensor.

        Returns:
            Tensor (..., out_channel): concatenated per-group embeddings mapped
            through out_proj.
        """
        if indices.shape[-1] != self._num_groups:
            raise ValueError(
                f"ProductVQ.get_codebook_entry expects last dim == num_groups "
                f"({self._num_groups}); got {indices.shape[-1]}.")
        embeds = []
        for g in range(self._num_groups):
            cb = self.codebooks[g]._get_codebook()
            if self.use_l2_norm:
                cb = F.normalize(cb, dim=-1)
            embeds.append(F.embedding(indices[..., g], cb))
        latent = self._synthesize(embeds)
        return self.out_proj(latent)


class _GroupedGatherVQ(ProductVQ):
    """Overlapping-group VQ (parallel, not residual).

    ``in_proj`` maps encoder.width -> code_dim. ``num_groups`` overlapping
    ``group_dim``-wide sub-vectors are gathered from the code_dim latent via a
    fixed ``group_idx`` index map (each channel covered exactly ``coverage``
    times). Each group is quantized by its own EMA ``VQ`` atom (codebook
    ``(n_embed, group_dim)``). Reconstruction scatter-adds the quantized
    sub-vectors back and divides by ``coverage`` -> code_dim latent, then
    ``out_proj``. Sliding-window and balanced-overlap subclasses differ ONLY in
    how ``group_idx`` is constructed. Everything else (EMA, dead-code restart,
    usage, STE, result_dict) is inherited unchanged from ProductVQ.
    """

    def __init__(self, in_channel, out_channel, code_dim, num_groups, n_embed,
                 group_dim, grouped_seed=0, **kwargs):
        self._grouped_seed = int(grouped_seed)
        super().__init__(
            in_channel=in_channel, out_channel=out_channel,
            code_dim=code_dim, num_groups=num_groups, n_embed=n_embed,
            group_dim=group_dim, **kwargs)

        idx = self._build_group_idx()                       # (num_groups, group_dim) long
        if tuple(idx.shape) != (self._num_groups, self.group_dim):
            raise ValueError(
                f"group_idx shape {tuple(idx.shape)} != "
                f"({self._num_groups}, {self.group_dim})")
        for g in range(self._num_groups):
            if len(torch.unique(idx[g])) != self.group_dim:
                raise ValueError(f"group {g} has duplicate channel indices")
        coverage = torch.bincount(idx.reshape(-1), minlength=self.code_dim)
        expected = self._num_groups * self.group_dim // self.code_dim
        if not (coverage == expected).all() or expected * self.code_dim != self._num_groups * self.group_dim:
            raise ValueError(
                f"non-uniform channel coverage: expected every dim covered "
                f"{expected}x, got min={coverage.min().item()} max={coverage.max().item()}")
        self.register_buffer("group_idx", idx.long(), persistent=True)
        self.register_buffer("coverage", coverage.to(torch.float32), persistent=True)

    def _build_group_idx(self):
        raise NotImplementedError

    def _analyze(self, z_proj):
        return [z_proj.index_select(-1, self.group_idx[g])
                for g in range(self._num_groups)]

    def _synthesize(self, quant_list):
        lead = quant_list[0].shape[:-1]
        z_hat = torch.zeros(*lead, self.code_dim,
                            dtype=quant_list[0].dtype, device=quant_list[0].device)
        for g in range(self._num_groups):
            z_hat.index_add_(-1, self.group_idx[g], quant_list[g])
        return z_hat / self.coverage


class SlidingWindowVQ(_GroupedGatherVQ):
    """Cyclic sliding-window grouping. ``stride = code_dim // num_groups``; group
    ``g`` owns dims ``(g*stride + t) % code_dim`` for ``t`` in [0, group_dim).
    With ``group_dim = 2*stride`` every dim is covered exactly twice, adjacent
    groups sharing ``stride`` dims."""

    def _build_group_idx(self):
        D, G, r = self.code_dim, self._num_groups, self.group_dim
        stride = D // G
        base = (torch.arange(G).unsqueeze(1) * stride
                + torch.arange(r).unsqueeze(0)) % D
        return base.long()


class BalancedOverlapVQ(_GroupedGatherVQ):
    """Balanced overlapping grouping. Every dim lives in exactly 2 groups; any
    two groups share 9 or 10 dims (vs the sliding window's lopsided ``stride``
    overlap). Each unordered group pair is deterministically assigned 9 or 10
    dims by ring distance so each group gets 144 dims (9 near pairs x10 + 6 far
    pairs x9) and the 1152 dims split as 72 pairs x10 + 48 pairs x9."""

    def _build_group_idx(self):
        import itertools
        D, G, r = self.code_dim, self._num_groups, self.group_dim
        pairs = list(itertools.combinations(range(G), 2))
        near = {1, 2, 3, 4, 8}

        def ring_dist(i, j):
            return min((j - i) % G, (i - j) % G)

        pair_dims = [10 if ring_dist(i, j) in near else 9 for (i, j) in pairs]
        if sum(pair_dims) != D:
            raise ValueError(f"pair dim budget {sum(pair_dims)} != code_dim {D}")
        n10 = pair_dims.count(10)
        n9 = pair_dims.count(9)
        if not (n10 == 72 and n9 == 48):
            raise ValueError(f"expected 72 pairs=10 & 48 pairs=9, got {n10}/{n9}")

        gen = torch.Generator().manual_seed(self._grouped_seed)
        perm = torch.randperm(D, generator=gen).tolist()
        groups = [[] for _ in range(G)]
        cursor = 0
        for (i, j), nd in zip(pairs, pair_dims):
            chunk = perm[cursor:cursor + nd]
            cursor += nd
            groups[i].extend(chunk)
            groups[j].extend(chunk)
        for g in range(G):
            if len(groups[g]) != r:
                raise ValueError(f"group {g} got {len(groups[g])} dims != {r}")
        return torch.tensor([sorted(g) for g in groups], dtype=torch.long)


class FactorizedVQ(ProductVQ):
    """Factorized (tight-frame) grouping. Instead of coordinate subsets, ``z`` is
    projected by ``num_groups`` fixed dense maps ``R_g in R^{group_dim x
    code_dim}``. Stacked ``A in R^{(G*group_dim) x code_dim}`` is a tight frame
    (``A^T A = rho I``, ``rho = G*group_dim / code_dim = 2``), so
    ``z_hat = (1/rho) A^T q`` reconstructs exactly when ``q = A z``. ``R`` is a
    fixed buffer (no learnable projector), keeping the EMA target ``u_g``
    stable across steps."""

    def __init__(self, in_channel, out_channel, code_dim, num_groups, n_embed,
                 group_dim, grouped_seed=0, **kwargs):
        self._grouped_seed = int(grouped_seed)
        super().__init__(
            in_channel=in_channel, out_channel=out_channel,
            code_dim=code_dim, num_groups=num_groups, n_embed=n_embed,
            group_dim=group_dim, **kwargs)

        D, G, r = self.code_dim, self._num_groups, self.group_dim
        rows = G * r
        if rows < D:
            raise ValueError(
                f"tight frame needs G*group_dim ({rows}) >= code_dim ({D})")
        rho = rows / D
        gen = torch.Generator().manual_seed(self._grouped_seed)
        M = torch.randn(rows, D, generator=gen, dtype=torch.float64)
        Q, _ = torch.linalg.qr(M, mode="reduced")          # (rows, D), Q^T Q = I_D
        A = ((rho ** 0.5) * Q).to(torch.float32)           # A^T A = rho I_D
        gram = A.t() @ A
        err = (gram - rho * torch.eye(D)).abs().max().item()
        if err >= 1e-3:
            raise ValueError(f"tight frame A^TA deviates from {rho}I: {err}")
        self._rho = float(rho)
        self.register_buffer("R", A.reshape(G, r, D).contiguous(), persistent=True)

    def _analyze(self, z_proj):
        u = torch.einsum("...d,grd->...gr", z_proj, self.R)   # (..., G, r)
        return [u[..., g, :] for g in range(self._num_groups)]

    def _synthesize(self, quant_list):
        q = torch.stack(quant_list, dim=-2)                   # (..., G, r)
        z_hat = torch.einsum("...gr,grd->...d", q, self.R)    # (..., D) = A^T q
        return z_hat / self._rho


class FSQ(BaseVQ):
    """Finite Scalar Quantization.

    Quantizes continuous features into discrete levels per channel without a learned codebook.
    Binary quantization (levels_per_channel=2) is always used.
    """
    def __init__(self,
                 in_channel=None,
                 out_channel=None,
                 token_size=16,
                 config=None):
        super().__init__()
        self.config = config
        self.token_size = token_size
        self.levels_per_channel = 2  # Binary quantization

        self.in_proj = nn.Linear(in_channel, token_size) if in_channel is not None else nn.Identity()
        self.out_proj = nn.Linear(token_size, out_channel) if out_channel is not None else nn.Identity()

        levels = [self.levels_per_channel] * token_size
        assert config.model.vq_model.codebook_size == self.levels_per_channel ** token_size, \
            f"codebook_size {config.model.vq_model.codebook_size} != " \
            f"levels_per_channel {self.levels_per_channel} ** token_size {token_size}"

        _levels = torch.tensor(levels, dtype=torch.int64)
        self.register_buffer("_levels", _levels, persistent=False)

        _basis = torch.cumprod(torch.tensor([1] + levels[:-1]), dim=0, dtype=torch.int64)
        self.register_buffer("_basis", _basis, persistent=False)

        self._codebook_size = config.model.vq_model.codebook_size

    @property
    def codebook_size(self):
        return self._codebook_size

    def round_ste(self, z):
        """Round with straight-through gradients."""
        zhat = z.round()
        return z + (zhat - z).detach()

    def bound(self, z, eps: float = 1e-3):
        """Bound z to quantization range.

        For binary quantization (levels=2):
        - Maps to range [-1, 0] before rounding
        - After rounding: -1 or 0
        """
        half_l = (self._levels - 1) * (1 + eps) / 2  # For levels=2: 0.5015
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)  # For levels=2: 0.5
        shift = (offset / half_l).atanh()

        normalized_z = (z + shift).tanh()
        return normalized_z * half_l - offset

    def quantize(self, z: torch.Tensor) -> torch.Tensor:
        """Quantize z to discrete levels.

        For binary quantization (levels=2):
        - bound() maps to [-1, 0] range
        - round_ste() rounds to {-1, 0}
        - Final mapping transforms {-1, 0} -> {-1, +1}
        """
        quantized = self.round_ste(self.bound(z))

        half_width = self._levels // 2  # For levels=2: half_width=1
        return (quantized / half_width) * 2 + 1

    def _scale_and_shift(self, zhat_normalized):
        """Scale normalized codes {-1, +1} to integer levels {0, 1}."""
        return (zhat_normalized + 1) / 2

    def _scale_and_shift_inverse(self, zhat):
        """Inverse of _scale_and_shift: {0, 1} -> {-1, +1}."""
        assert torch.all((zhat == 0) | (zhat == 1)), \
            f"Expected zhat to contain only 0 or 1, but got values in range [{zhat.min()}, {zhat.max()}]"
        return zhat.float() * 2 - 1

    def get_codebook_entry(self, indices):
        """Get quantized codes from indices.

        Args:
            indices: (B, L*C) flattened indices per level

        Returns:
            codes: (B, L, C) quantized codes
        """
        indices_per_level = indices.reshape(indices.shape[0], -1, self.token_size)
        codes = self.indices_to_codes_per_level(indices_per_level)
        codes = self.out_proj(codes)
        return codes

    def codes_to_indices_per_level(self, zhat_normalized):
        """Convert normalized codes to per-level indices.

        Args:
            zhat_normalized: (B, L, C) float codes in [-1, 1]

        Returns:
            codes: (B, L, C) integer codes in [0, levels_per_channel-1]
        """
        levels_float = self._scale_and_shift(zhat_normalized)
        codes = torch.round(levels_float).to(torch.int64)
        return torch.clamp(codes, 0, self.levels_per_channel - 1)

    def indices_to_codes_per_level(self, per_level_codes):
        """Convert per-level indices to normalized codes."""
        per_level_codes = per_level_codes.to(torch.int64)
        return self._scale_and_shift_inverse(per_level_codes)

    @autocast(device_type='cuda', enabled=False)
    def forward(self, z, condition=None):
        """Forward pass: quantize continuous features with binary quantization.

        Args:
            z: (B, L, C) continuous features
            condition: (unused, for compatibility)

        Returns:
            z_quantized: (B, L, C) quantized features
            result_dict: Dictionary with quantization info
        """
        z = z.float()
        z = self.in_proj(z)

        assert z.shape[-1] == self.token_size
        z_quantized = self.quantize(z)

        min_encoding_indices = self.codes_to_indices_per_level(z_quantized)

        z_quantized = self.out_proj(z_quantized)

        result_dict = dict(
            min_encoding_indices=min_encoding_indices
        )

        return z_quantized, result_dict



######
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
    """Variable-length attention adapted from ``strong_vq.py``'s ``PlainAttention``.

    Uses ``flash_attn_varlen_func`` (when available) so the same module handles
    both static and dynamic-resolution batches: caller may pass either
      - ``x`` of shape ``(B, L, in_dim)`` (cu_seqlens inferred = arange(0, B+1)*L), or
      - ``x`` of shape ``(seq_total, in_dim)`` + explicit ``cu_seqlens`` kwarg
        (``shape=(B+1,)``, int32, on the same device).
    Falls back to ``F.scaled_dot_product_attention`` with per-image splits when
    flash_attn isn't installed.

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

    def _attn_static(self, q, k, v, B, L):
        # SDPA fallback for (B, L) shape.
        q = q.reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)         # (B, heads, L, head_dim)
        return out.transpose(1, 2).reshape(B * L, self.num_heads, self.head_dim)

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
        # Normalize to (seq_total, in_dim) + B / L info.
        if x.dim() == 3:
            B, L, _ = x.shape
            x_flat = x.reshape(B * L, -1)
            if cu_seqlens is None:
                cu_seqlens = torch.arange(
                    0, (B + 1) * L, L, device=x.device, dtype=torch.int32)
            max_seqlen = L
            static_BL = (B, L)
        elif x.dim() == 2:
            assert cu_seqlens is not None, \
                "PlainAttentionFlash: (seq_total, D) input requires cu_seqlens."
            x_flat = x
            max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())
            static_BL = None
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")

        seq_total = x_flat.shape[0]
        qkv_bias = torch.cat((self.q_bias, self.zero_k_bias, self.v_bias))
        qkv = F.linear(x_flat, self.qkv.weight, qkv_bias)
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
            if static_BL is not None:
                attn_out = self._attn_static(q, k, v, *static_BL)
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

        out_flat = self.proj(flat)                                       # (seq_total, out_dim)
        if static_BL is not None:
            B, L = static_BL
            return out_flat.reshape(B, L, self.out_dim)
        return out_flat


class AttnProjection(nn.Module):
    """Transformer-block-style projector: ``proj+attn`` then ``+mlp``.

    Same skeleton as ``strong_vq.py``'s ``AttnProjection`` but the attention
    sub-module is ``PlainAttentionFlash`` (flash_attn varlen API), so the
    projector works for both static (B, L, D) and dynamic-resolution
    (seq_total, D) + cu_seqlens inputs.
    """

    def __init__(self, in_dim, out_dim, num_heads, norm_layer=nn.LayerNorm, mlp_ratio=2):
        super().__init__()
        # Note: strong_vq.py's original AttnProjection asserted that in/out dims
        # divide each other. That's only needed in the "in_dim <= out_dim" path
        # to ensure num_heads*head_dim == out_dim. The "in_dim > out_dim" path
        # uses adaptive_avg_pool1d which is dim-agnostic, so 1152→1024 etc. are
        # fine. We delegate divisibility checks to PlainAttentionFlash.
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.norm1 = norm_layer(in_dim)
        self.attn = PlainAttentionFlash(in_dim, out_dim, num_heads)
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm3 = norm_layer(in_dim)

        self.norm2 = norm_layer(out_dim)
        hidden_dim = int(out_dim * mlp_ratio)
        self.mlp = GeGluMlp(in_features=out_dim, hidden_features=hidden_dim)

    def forward(self, x, cu_seqlens=None):
        # Linear residual branch (works on any leading shape).
        lin_branch = self.proj(self.norm3(x))
        # Attention branch (handles both 2D and 3D x, plus optional cu_seqlens).
        attn_branch = self.attn(self.norm1(x), cu_seqlens=cu_seqlens)
        x = lin_branch + attn_branch
        x = x + self.mlp(self.norm2(x))
        return x

