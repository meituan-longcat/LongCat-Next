# Copyright (c) 2026 LongCat dNaViT Team.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import json
from typing import Optional
from omegaconf import OmegaConf
from pathlib import Path

from .modules import BaseModel, UnifiedEncoder, UnifiedDecoder
from .modules.blocks import RMSNorm, Block, _expand_token, FeatureDecoder, LinearFeatureDecoder
from .quantizer import (BaseVQ, FSQ, RVQ, VQ, ProductVQ, IBQ,
                        SlidingWindowVQ, BalancedOverlapVQ, FactorizedVQ)


def _make_sem_projector(in_dim, out_dim, hidden=2048, proj_type="mlp"):
    """Head that maps tokenizer features (pre- or post-quant) into the
    encoder-frozen-branch feature space for semantic distillation.

    ``proj_type="mlp"`` (default): 3-layer SiLU MLP — the original behaviour.
    ``proj_type="linear"``: a single ``nn.Linear``. Use with
    ``merger_as_sem_proj=True`` so the non-linearity comes from the frozen
    qwen merger that follows it (``feat -> linear -> merger -> loss``)."""
    if proj_type == "linear":
        return nn.Linear(in_dim, out_dim)
    if proj_type == "mlp":
        return nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )
    raise ValueError(f"Unknown sem_projector_type={proj_type!r}; expected 'mlp' or 'linear'.")


class TokenUnmerger(nn.Module):
    """Inverse of the Qwen 2x2 patch-merger at the *token-count* level: takes
    one merged token of dim ``in_dim`` and produces ``ms**2`` sub-tokens of
    dim ``out_dim``, matching the raster-scan layout the merger collapsed.

    Supports cross-dim (``in_dim != out_dim``) so a continuous-AE stage-A can
    fold dim reduction + token expansion into a single Linear(merger_out_dim
    -> ms2 * vit_dec_width) (e.g. 4096 -> 4*1152=4608, then reshape
    (T/ms2, ms2, 1152) -> (T, 1152)). When ``in_dim == out_dim`` the layer
    degenerates to the original same-dim replicate-init unmerger.

    Optional pre-projection transformer block: when ``use_transformer_block``
    is set, a single pre-norm transformer layer is applied to the input
    (dim = ``in_dim``, which equals 1152 with the baseline rmsnorm_mlp
    out_proj — NOT 4096, because out_proj already did 4096->1152 reduction
    upstream) BEFORE the unmerger's Linear projection. The block reuses
    ``_ResModLayer`` (LayerNorm + PlainAttentionFlash + GeGluMlp) and is
    zero-init at the residual out-projections (attn.proj + mlp.w2) so it
    starts as identity → safe to bolt on top of a baseline-trained
    checkpoint without breaking the forward output.

    The block uses varlen (per-image) attention via ``cu_seqlens``. For flat
    packed ``(T_total, in_dim)`` input the caller passes cu_seqlens from the
    encoder; for batched ``(B, L, in_dim)`` input (e.g. decode_tokens) the
    forward builds a trivial per-image cu_seqlens internally.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        spatial_merge_size: int,
        use_transformer_block: bool = False,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        init_values: Optional[float] = None,
        transformer_zero_init: bool = False,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.ms = int(spatial_merge_size)
        self.ms2 = self.ms ** 2
        self.proj = nn.Linear(self.in_dim, self.ms2 * self.out_dim)
        # Pseudo-identity init: same-dim -> true replicate-init (each sub-token
        # = the input); cross-dim -> projects z onto the first out_dim coords
        # (rest zero). Sensible broadcast warm-start the decoder can
        # differentiate; checkpoint resume overwrites it.
        with torch.no_grad():
            eye = torch.eye(self.out_dim, self.in_dim,
                            dtype=self.proj.weight.dtype)
            self.proj.weight.copy_(eye.repeat(self.ms2, 1))
            self.proj.bias.zero_()

        # Optional 1-layer transformer block, applied BEFORE the Linear
        # projection. Two init modes:
        #   * transformer_zero_init=True  → zero-init attn.proj + mlp.w2 →
        #     identity at init → warm-start safe (initial forward == baseline
        #     forward). But learning is VERY slow (zero-init residuals need
        #     many steps to ramp up).
        #   * transformer_zero_init=False (DEFAULT, 方案3) → standard
        #     trunc_normal_(std=0.02) for all Linear, ones_/zeros_ for
        #     LayerNorm. Transformer has immediate non-zero contribution at
        #     step 0. Initial forward ≠ baseline forward → loss has a small
        #     jump at step 0. Faster convergence of the transformer block.
        self.transformer = None
        if use_transformer_block:
            # Local import to avoid a circular import (res_mod imports from
            # tokenizer for type hints in some downstream modules).
            from .modules.res_mod import _ResModLayer
            self.transformer = _ResModLayer(
                dim=self.in_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                init_values=init_values,
            )
            if transformer_zero_init:
                # Zero-init residuals: block starts as identity. Safe for
                # warm-start from a baseline ckpt that has no transformer.* keys.
                with torch.no_grad():
                    self.transformer.attn.proj.weight.zero_()
                    self.transformer.attn.proj.bias.zero_()
                    self.transformer.mlp.w2.weight.zero_()
                    self.transformer.mlp.w2.bias.zero_()
            else:
                # 方案3: standard ViT-style trunc_normal init. All Linear
                # layers get trunc_normal_(std=0.02) + zero bias; LayerNorm
                # gets ones_(weight) + zeros_(bias). The transformer has
                # immediate non-zero effect on the forward pass.
                import torch.nn.init as init_
                def _vit_init(m):
                    if isinstance(m, nn.Linear):
                        init_.trunc_normal_(m.weight, mean=0.0, std=0.02)
                        if m.bias is not None:
                            init_.zeros_(m.bias)
                    elif isinstance(m, (nn.LayerNorm,)):
                        init_.ones_(m.weight)
                        init_.zeros_(m.bias)
                self.transformer.apply(_vit_init)

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        D = x.shape[-1]
        if D != self.in_dim:
            raise ValueError(
                f"TokenUnmerger built for in_dim={self.in_dim}, got input D={D}.")

        if self.transformer is not None:
            if x.dim() == 2:
                # Flat packed (T_total, in_dim). Caller should pass cu_seqlens
                # from the encoder; if absent, treat as a single segment.
                if cu_seqlens is None:
                    cu_seqlens = torch.tensor(
                        [0, x.shape[0]], dtype=torch.int32, device=x.device)
                x = self.transformer(x, cu_seqlens=cu_seqlens)
            elif x.dim() == 3:
                # Batched (B, L, in_dim). Flatten + build per-image cu_seqlens
                # so attention stays per-image (matches encode-path behavior).
                B, L, _ = x.shape
                flat = x.reshape(B * L, D)
                cu = torch.arange(
                    0, B + 1, device=x.device, dtype=torch.int32) * L
                flat = self.transformer(flat, cu_seqlens=cu)
                x = flat.view(B, L, D)
            else:
                raise ValueError(
                    f"TokenUnmerger.forward: x must be 2D or 3D, got {x.dim()}D")

        y = self.proj(x)                              # (..., ms2 * out_dim)
        if x.dim() == 2:
            T_in = x.shape[0]
            return y.view(T_in, self.ms2, self.out_dim).reshape(
                T_in * self.ms2, self.out_dim)
        B, L_in, _ = x.shape
        return y.view(B, L_in, self.ms2, self.out_dim).reshape(
            B, L_in * self.ms2, self.out_dim)



class BaseTokenizer(BaseModel):
    """Base class for all Tokenizer models.

    All tokenizer variants (FSQ, RVQ, VQ, AlignedVQ, AE, IBQ) must inherit
    from this class
    and implement:
    - encode(x): images -> (z_quantized, result_dict)
    - decode(z_quantized): z_quantized -> (decoded, clip_pred)
    - decode_tokens(tokens): discrete tokens -> decoded images
    - forward(input): full encode -> quantize -> decode pipeline
    """
    def __init__(self):
        super().__init__()

    def set_encoder_trainable(self, flag: bool) -> None:
        """Toggle ``requires_grad`` on the trainable backbone branch of
        ``self.encoder`` (if present). Used by the ``enc_freeze_warmup_step``
        knob to freeze the encoder during early training so the random
        decoder doesn't drag its representation off. ResMod / quantizer /
        decoder / clip projectors are NOT touched — they keep training.
        No-op if the tokenizer subclass doesn't expose ``self.encoder`` with
        the helper method.
        """
        enc = getattr(self, "encoder", None)
        if enc is not None and hasattr(enc, "set_trainable_backbone_grad"):
            enc.set_trainable_backbone_grad(flag)

    def encode(self, x):
        raise NotImplementedError("Subclasses must implement encode")

    def decode(self, z_quantized):
        raise NotImplementedError("Subclasses must implement decode")

    def decode_tokens(self, tokens):
        raise NotImplementedError("Subclasses must implement decode_tokens")

    def forward(self, input):
        raise NotImplementedError("Subclasses must implement forward")

    # ---- sem-loss / merger helpers ------------------------------------------
    def _init_sem_proj_cfg(self, vq_cfg):
        """Read the sem-projector config (type + merger reuse) and validate it
        against the encoder backbone. Sets ``sem_projector_type``,
        ``merger_as_sem_proj``, ``merger_after_quant``, ``merger_out_dim``,
        ``spatial_merge_size``. Subclasses call this from ``__init__``."""
        self.sem_projector_type = vq_cfg.get("sem_projector_type", "mlp")
        self.merger_as_sem_proj = bool(vq_cfg.get("merger_as_sem_proj", False))
        # ``merger_after_quant`` reshapes the pipeline: the Qwen patch-merger
        # moves INSIDE the encoder (ResMod's internal merger -> merger_out_dim +
        # collapse ms2 tokens), so the tokenizer sees a (T/ms2, merger_out_dim)
        # feature directly and quantises in merger-out space. A TokenUnmerger
        # after the quantizer expands back to T tokens before the decoder.
        self.merger_after_quant = bool(vq_cfg.get("merger_after_quant", False))
        if self.merger_as_sem_proj or self.merger_after_quant:
            enc = getattr(self, "encoder", None)
            merger = getattr(enc, "qwen_merger", None) if enc is not None else None
            if merger is None:
                raise ValueError(
                    "merger_as_sem_proj / merger_after_quant requires the "
                    "qwen3vit backbone (no patch-merger on siglip2).")
        if self.merger_after_quant:
            # Encoder output already lives in merger-out space, so the post-hoc
            # ``_apply_sem_merger`` path would double-merge. Force-disable.
            if self.merger_as_sem_proj:
                print("[BaseTokenizer] merger_after_quant=True overrides "
                      "merger_as_sem_proj=True (Qwen merger lives inside the "
                      "encoder; sem-loss compares in merger-out space directly).")
            self.merger_as_sem_proj = False
            enc = self.encoder
            self.merger_out_dim = int(enc.qwen_merger_out_dim)
            self.spatial_merge_size = int(enc.qwen_spatial_merge_size)
        else:
            self.merger_out_dim = None
            self.spatial_merge_size = None

    def _apply_sem_merger(self, x):
        """If ``merger_as_sem_proj`` is on, push ``x`` through the frozen qwen
        patch-merger; otherwise return ``x`` unchanged. Handles ``(B, L, D)``
        dense and ``(T, D)`` packed layouts (tokens pre-grouped 2x2 upstream)."""
        if not getattr(self, "merger_as_sem_proj", False):
            return x
        return self._apply_frozen_merger(x)

    def _apply_frozen_merger(self, x):
        """Unconditional version of ``_apply_sem_merger`` — runs ``x`` through
        the frozen qwen patch-merger regardless of the flag. Used by
        merger_after_quant to merge the teacher clip_gt for sem-loss."""
        merger = self.encoder.qwen_merger
        ms2 = int(self.encoder.qwen_spatial_merge_size) ** 2
        D = x.shape[-1]
        out = merger(x.reshape(-1, D))
        if x.dim() == 3:
            B, L, _ = x.shape
            if L % ms2 != 0:
                raise ValueError(
                    f"_apply_frozen_merger needs L divisible by {ms2}; got L={L}.")
            return out.view(B, L // ms2, -1)
        return out



class FSQTokenizer(BaseTokenizer):
    """Tokenizer with Finite Scalar Quantization (FSQ).

    Uses SigLIP2 encoder/decoder with binary FSQ quantization.
    """

    def __init__(self, config):
        if isinstance(config, dict):
            config = OmegaConf.create(config)

        super().__init__()
        self.config = config

        # Initialize decoder first, then apply weight initialization
        self.decoder = UnifiedDecoder(config)
        self.apply(self._init_weights)

        # Load encoder after weight initialization to preserve pretrained weights
        self.encoder = UnifiedEncoder(config)

        # Only use FSQ quantizer
        self.quantize = FSQ(
            in_channel=self.encoder.width,
            out_channel=self.decoder.width,
            token_size=config.model.vq_model.token_size,
            config=config,
        )

        # FSQ has no codebook → PostQuantSemLoss is not meaningful here; only the
        # PreQuantSemLoss projector is exposed.
        # Sem-loss weights live on the `losses` block — projector is built iff
        # the weight > 0 (single source of truth shared with ReconstructionLoss).
        vq_cfg = config.model.vq_model
        loss_cfg = config.get("losses", {})
        self.pre_quant_sem_weight = loss_cfg.get("pre_quant_sem_weight", 0.0)
        self.post_quant_sem_weight = 0.0
        sem_hidden = vq_cfg.get("sem_projector_hidden", 2048)
        if self.pre_quant_sem_weight > 0:
            self.pre_clip_projector = _make_sem_projector(
                self.encoder.width, self.encoder.width, hidden=sem_hidden)
        else:
            self.pre_clip_projector = None
        self.post_clip_projector = None

    def _save_pretrained(self, save_directory: Path) -> None:
        """Save weights and config to a local directory."""
        dict_config = OmegaConf.to_container(self.config)
        file_path = Path(save_directory) / "config.json"
        with open(file_path, 'w') as json_file:
            json.dump(dict_config, json_file, indent=4)
        super()._save_pretrained(save_directory)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
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

    def encode(self, x, image_grid_thw=None):
        """Encode images to quantized tokens.

        ``image_grid_thw`` is accepted for API uniformity; FSQ doesn't support
        dynamic-resolution yet (FSQ's quantize lacks a cu_seqlens path).
        """
        if image_grid_thw is not None:
            raise NotImplementedError(
                "FSQTokenizer doesn't support dynamic resolution; use RVQTokenizer or VQTokenizer."
            )
        z, clip_gt = self.encoder(pixel_values=x, return_clip_gt=True)
        z_quantized, result_dict = self.quantize(z)
        result_dict["clip_gt"] = clip_gt
        if self.pre_clip_projector is not None:
            result_dict["pre_clip_pred"] = self.pre_clip_projector(z)
        return z_quantized, result_dict

    def decode(self, z_quantized, image_grid_thw=None):
        decoded, clip_pred = self.decoder(z_quantized, image_grid_thw=image_grid_thw)
        return decoded, clip_pred

    def decode_tokens(self, tokens):
        """Decode discrete token indices to images."""
        tokens = tokens.squeeze(1)
        batch, seq_len = tokens.shape  # B x N
        z_quantized = self.quantize.get_codebook_entry(
            tokens.reshape(batch, -1)).reshape(batch, -1, self.decoder.width)
        decoded = self.decode(z_quantized)[0]
        return decoded

    def forward(self, input, image_grid_thw=None):
        """Forward pass: encode -> quantize -> decode.

        Args:
            input: Input images (B, 3, H, W)

        Returns:
            decoded: Reconstructed images (B, 3, H, W)
            result_dict: Dictionary containing quantization results, CLIP predictions and ground truth
        """
        z_quantized, result_dict = self.encode(input)
        decoded, clip_pred = self.decode(z_quantized)
        result_dict["clip_pred"] = clip_pred
        return decoded, result_dict


class RVQTokenizer(BaseTokenizer):
    """Tokenizer with Residual Vector Quantization (RVQ).

    Uses SigLIP2 encoder/decoder with multi-layer residual VQ.
    """

    def __init__(self, config):
        if isinstance(config, dict):
            config = OmegaConf.create(config)
        super().__init__()
        self.config = config

        # Feature-reconstruction-only mode: skip the image decoder entirely and
        # train purely on the quantizer's feature-reconstruction signal
        # (RVQ ``quantizer_loss`` = commitment + vq_loss). Used to verify
        # lossless quantization without the pixel-decode / distillation branches.
        self.feature_recon_only = config.model.vq_model.get("feature_recon_only", False)
        # Continuous-AE (stage-A) mode: a no-quantization autoencoder path. The
        # quantizer is still built (encode_continuous borrows its in/out_proj for
        # the in->code->out projection) but the codebook is NEVER consulted /
        # updated; the encoder feature flows in_proj -> out_proj -> (unmerger) ->
        # pixel decoder. Requires a pixel decoder (feature_recon_only=False) and
        # ResMod. Works for both 32x (merger_after_quant=True, unmerger expands
        # the merged tokens) and 16x (no merger, T tokens straight to the decoder).
        self.continuous = bool(config.model.vq_model.get("continuous", False))
        if self.continuous:
            if self.feature_recon_only:
                raise ValueError("continuous=True needs a pixel decoder; set feature_recon_only=False.")
            if not bool(config.model.vq_model.get("use_res_mod", False)):
                raise ValueError("continuous=True requires use_res_mod=True.")
        # Detached pixel-decoder warm-up (stage-B): build the pixel decoder and run
        # it on a DETACHED copy of the post-quant tokens, so the pixel loss
        # (L2/LPIPS/GAN) trains ONLY the decoder and never backprops into the RVQ /
        # in-out projections / ResMod. RVQ is still supervised purely by the
        # feature-recon path (aux_feature_distill MSE + commitment). Used to
        # pre-warm stage-C's pixel decoder without disturbing the codebook.
        self.detach_pixel_decoder = bool(config.model.vq_model.get("detach_pixel_decoder", False))
        if self.detach_pixel_decoder:
            if self.feature_recon_only:
                raise ValueError("detach_pixel_decoder=True needs a pixel decoder; set feature_recon_only=False.")
            if self.continuous:
                raise ValueError("detach_pixel_decoder is incompatible with continuous=True.")
        # Feature-recon target: which encoder feature the FeatureDecoder + MSE
        # reconstructs. "pre_resmod" (default) = the frozen-merger merge_target
        # (stable anchor); "post_resmod" = the post-ResMod z (used in stage-B
        # where ResMod is frozen and z is a meaningful fixed target).
        self.feature_recon_target = config.model.vq_model.get("feature_recon_target", "pre_resmod")
        if self.feature_recon_target not in ("pre_resmod", "post_resmod"):
            raise ValueError(
                f"feature_recon_target must be 'pre_resmod' or 'post_resmod', got {self.feature_recon_target!r}.")
        if self.feature_recon_only:
            self.decoder = None
        else:
            self.decoder = UnifiedDecoder(config)
            self.apply(self._init_weights)
        self.encoder = UnifiedEncoder(config)

        # Feature decoder: a small transformer that maps post-quant tokens back to
        # the frozen encoder feature ``z``. Built before RVQ so the quantizer's
        # out_proj (= feat_post_quant_conv) targets its width. Used for both 16x
        # and 32x encoders.
        # ``feature_decoder_type`` selects how post-quant tokens map back to ``z``:
        #   - "attn": transformer FeatureDecoder (token mixing); out_proj targets
        #             its width and the decoder head maps back to the z dim.
        #   - "mlp":  no FeatureDecoder; out_proj (rmsnorm_mlp, pointwise) maps
        #             code_dim -> z dim directly and IS the decoder.
        #
        # ``aux_feature_distill`` reuses this branch in PIXEL mode (stage-2
        # warm-started from a feature_recon stage-1): the FeatureDecoder + its MSE
        # to ``feature_target`` is kept as an auxiliary feature-distillation loss
        # alongside the pixel decoder. The shared out_proj (1152 -> vit_dec_width
        # = 1024) feeds BOTH this FeatureDecoder (merged space) and, after the
        # TokenUnmerger, the UnifiedDecoder — so yaml must lock vit_dec_width:1024.
        self.aux_feature_distill = bool(config.model.vq_model.get("aux_feature_distill", False))
        self.feature_decoder_type = config.model.vq_model.get("feature_decoder_type", "attn")
        _build_feat_dec = (self.feature_recon_only or self.aux_feature_distill)
        if _build_feat_dec and self.feature_decoder_type not in ("attn", "mlp", "linear"):
            raise ValueError(
                f"Unsupported feature_decoder_type: {self.feature_decoder_type!r} "
                "(expected 'attn', 'mlp', or 'linear').")
        if _build_feat_dec and self.feature_decoder_type in ("attn", "linear"):
            # The reconstruction target is the pre-quant feature ``z``. Under
            # merger_after_quant (stage-2) z is the ResMod-merger output
            # (qwen_merger_out_dim = 4096), NOT encoder.width (which stays
            # hidden_size = 1152). Under qwen_use_merger (stage-1) z == encoder.width
            # (= 4096). Pick the dim that matches encode()'s feature_target.
            _maq = bool(config.model.vq_model.get("merger_after_quant", False))
            feat_target_dim = (int(self.encoder.qwen_merger_out_dim)
                               if _maq else self.encoder.width)
            # "attn": transformer FeatureDecoder (token mixing + pos-embed + MLP
            # head). "linear": single pointwise nn.Linear(vit_dec_width -> z dim),
            # no token mixing, for the lightweight feature-distill ablation. Both
            # consume the shared out_proj output (vit_dec_width) so out_proj stays
            # code_dim -> vit_dec_width and the pixel decoder is unaffected.
            if self.feature_decoder_type == "attn":
                self.feature_decoder = FeatureDecoder(config, out_channels=feat_target_dim)
            else:
                self.feature_decoder = LinearFeatureDecoder(config, out_channels=feat_target_dim)
            self.feature_decoder.apply(self._init_weights)
        else:
            self.feature_decoder = None

        # RVQ Quantizer
        vq_cfg = config.model.vq_model
        loss_cfg = config.get("losses", {})
        enable_simvq = vq_cfg.get("enable_simvq", False)

        # Quantizer-internal loss knobs live on the `losses:` block
        # (single source of truth, alongside pixel-recon weights).
        entropy_loss_weight = loss_cfg.get("entropy_loss_weight", 0.0)
        if enable_simvq and entropy_loss_weight is None:
            entropy_loss_weight = 0.0

        # ---- Shape resolution -----------------------------------------------
        # Two equivalent ways to specify codebook geometry:
        #
        # A) Dynamic-resolution (recommended for variable H/W):
        #       rvq_depth: 8
        #       code_dim:  1152     # codebook embedding dim (128/256/512/768/1024/1152)
        #    H/W are runtime-determined; spatial packing is disabled
        #    (code_shape/latent_shape's H/W forced to 1, so shape_divisor==(1,1,1)).
        #
        # B) Explicit shape (legacy / static-resolution / spatial-packing):
        #       code_shape:   [16, 16, 8]
        #       latent_shape: [16, 16, 1152]
        #    rvq_depth (optional) must match code_shape[-1] if both given.
        code_dim = vq_cfg.get("code_dim", None)
        rvq_depth = vq_cfg.get("rvq_depth", None)

        if code_dim is not None:
            if rvq_depth is None:
                raise ValueError("code_dim requires rvq_depth to be set explicitly.")
            # Cross-check against any legacy code_shape / latent_shape: only the
            # last (depth / codebook-dim) entry must agree; H/W are ignored
            # because they are dynamic.
            cfg_code_shape = vq_cfg.get("code_shape", None)
            cfg_latent_shape = vq_cfg.get("latent_shape", None)
            if cfg_code_shape is not None:
                cs = list(cfg_code_shape)
                if len(cs) == 3 and int(cs[-1]) != int(rvq_depth):
                    raise ValueError(
                        f"code_shape[-1]={cs[-1]} conflicts with rvq_depth={rvq_depth}.")
            if cfg_latent_shape is not None:
                ls = list(cfg_latent_shape)
                if len(ls) == 3 and int(ls[-1]) != int(code_dim):
                    raise ValueError(
                        f"latent_shape[-1]={ls[-1]} conflicts with code_dim={code_dim}.")
            code_shape = [1, 1, int(rvq_depth)]
            latent_shape = [1, 1, int(code_dim)]
        else:
            # Legacy path: resolve depth from rvq_depth or code_shape[-1].
            code_shape = list(vq_cfg.code_shape)
            if rvq_depth is not None:
                if len(code_shape) == 2:
                    code_shape = code_shape + [int(rvq_depth)]
                elif len(code_shape) == 3:
                    if int(code_shape[-1]) != int(rvq_depth):
                        raise ValueError(
                            f"rvq_depth={rvq_depth} conflicts with code_shape[-1]={code_shape[-1]}; "
                            "drop one or make them match."
                        )
                else:
                    raise ValueError(
                        f"code_shape must be length 2 or 3 (got {len(code_shape)}).")
            latent_shape = vq_cfg.get("latent_shape", None)

        # Shared out_channel for both quantizers: feature-recon targets the
        # FeatureDecoder width (attn) / encoder.width (mlp); pixel mode targets
        # decoder.width.
        quant_out_channel = (
            (self.feature_decoder.width if self.feature_decoder is not None
             else self.encoder.width)
            if self.feature_recon_only else self.decoder.width)

        # ---- merger_after_quant pre-detection -------------------------------
        # When the merger lives BEFORE the quantizer (inside the encoder's
        # ResMod internal merger), the encoder output ``z`` already lives in
        # merger-out space, so the quantizer's in_channel must be sized to
        # qwen_merger_out_dim instead of encoder.width. Full validation +
        # TokenUnmerger construction happen later via _init_sem_proj_cfg.
        _maq_enabled = bool(vq_cfg.get("merger_after_quant", False))
        _quant_in_channel = (
            int(self.encoder.qwen_merger_out_dim) if _maq_enabled
            else self.encoder.width
        )

        # ---- Quantizer dispatch ---------------------------------------------
        # Built via an overridable hook so subclasses (e.g. IBQTokenizer) can swap
        # the quantization method while inheriting all of RVQTokenizer's
        # feature-recon plumbing (encode / forward / FeatureDecoder / losses).
        self.quantize = self._build_quantizer(
            vq_cfg, loss_cfg, config,
            in_channel=_quant_in_channel,
            out_channel=quant_out_channel,
            code_shape=code_shape,
            latent_shape=latent_shape,
            enable_simvq=enable_simvq,
            entropy_loss_weight=entropy_loss_weight,
        )

        # ---- Pre/Post-quant semantic projectors -----------------------------
        # PreQuantSemLoss aligns encoder's last-hidden z with frozen clip_gt;
        # PostQuantSemLoss aligns codebook-lookup features (cumulative up to
        # ``post_quant_align_depth`` levels for RVQ) with clip_gt.
        # Both projectors map into ``encoder.width`` (clip_gt's dim).
        # Sem-loss weights live on the `losses` block — projector is built iff
        # the weight > 0 (single source of truth shared with ReconstructionLoss).
        loss_cfg = config.get("losses", {})
        self.pre_quant_sem_weight = loss_cfg.get("pre_quant_sem_weight", 0.0)
        self.post_quant_sem_weight = loss_cfg.get("post_quant_sem_weight", 0.0)
        # pre_quant_sem target source: "clip_gt" (default; proj(z) -> teacher
        # frozen_merger(clip_gt)) or "pre_resmod" (proj(z) -> same-forward
        # pre-ResMod merge_target, no teacher branch needed).
        self.pre_quant_sem_target = vq_cfg.get("pre_quant_sem_target", "clip_gt")
        if self.pre_quant_sem_target not in ("clip_gt", "pre_resmod"):
            raise ValueError(
                f"pre_quant_sem_target must be 'clip_gt' or 'pre_resmod', "
                f"got {self.pre_quant_sem_target!r}.")
        if (self.continuous and self.pre_quant_sem_weight > 0
                and self.pre_quant_sem_target != "pre_resmod"):
            raise ValueError(
                "continuous=True has no teacher branch; pre_quant_sem_weight>0 "
                "requires pre_quant_sem_target='pre_resmod'.")
        sem_hidden = vq_cfg.get("sem_projector_hidden", 2048)
        self._init_sem_proj_cfg(vq_cfg)

        # ---- merger_after_quant: build the post-quant unmerger --------------
        # ``z_quantized`` carries T/ms2 tokens; the decoder expects T tokens,
        # so expand each merged token back into ms2 sub-tokens. The unmerger's
        # ``in_dim`` is whatever actually flows out of ``out_proj``: Identity
        # (force_identity OR force=False with matching dims) lets code_dim
        # pass through unchanged, so the unmerger must consume code_dim and
        # do dim reduction itself; a real projection already maps to
        # quant_out_channel, so the unmerger is same-dim. ``out_dim`` is
        # always quant_out_channel (decoder.width) — what UnifiedDecoder eats.
        if self.merger_after_quant:
            unmerger_in_dim = (code_dim
                               if isinstance(self.quantize.out_proj, nn.Identity)
                               else quant_out_channel)
            # Optional 1-layer transformer block inside the unmerger. Sub-config
            # under model.vq_model.post_quant_unmerger controls the block; if
            # the section is absent or use_transformer_block is false, the
            # unmerger is the original Linear-only TokenUnmerger.
            unmerger_cfg = vq_cfg.get("post_quant_unmerger", {}) or {}
            self.post_quant_unmerger = TokenUnmerger(
                in_dim=unmerger_in_dim,
                out_dim=quant_out_channel,
                spatial_merge_size=self.spatial_merge_size,
                use_transformer_block=bool(unmerger_cfg.get("use_transformer_block", False)),
                num_heads=int(unmerger_cfg.get("num_heads", 16)),
                mlp_ratio=float(unmerger_cfg.get("mlp_ratio", 4.0)),
                init_values=unmerger_cfg.get("init_values", None),
                transformer_zero_init=bool(unmerger_cfg.get("transformer_zero_init", False)),
            )
        else:
            self.post_quant_unmerger = None

        # Depth K to align against clip_gt; None / 0 / >= depth means "use full
        # depth" (= self.quantize.code_depth). For RVQ depth=8, set 4 to only
        # align the first 4 levels so the AR model can stop early and still get
        # semantically-meaningful tokens.
        align_depth = vq_cfg.get("post_quant_align_depth", None)
        full_depth = self.quantize.code_depth
        if not align_depth or align_depth <= 0 or align_depth > full_depth:
            # When a semantic/pixel split is configured (sem_group_size), default
            # the alignment depth to the semantic group size so post-quant
            # distillation supervises exactly the semantic group. Otherwise fall
            # back to full depth (legacy behavior).
            sem_sz = vq_cfg.get("sem_group_size", None)
            align_depth = int(sem_sz) if sem_sz else full_depth
        self.post_quant_align_depth = align_depth

        if self.pre_quant_sem_weight > 0:
            # Under merger_after_quant the encoder output ``z`` already lives in
            # merger-out space, so the projector is dim-preserving
            # (merger_out_dim -> merger_out_dim); target is frozen_merger(clip_gt).
            pre_in_dim = self.merger_out_dim if self.merger_after_quant else self.encoder.width
            pre_out_dim = self.merger_out_dim if self.merger_after_quant else self.encoder.width
            self.pre_clip_projector = _make_sem_projector(
                pre_in_dim, pre_out_dim, hidden=sem_hidden,
                proj_type=self.sem_projector_type)
        else:
            self.pre_clip_projector = None

        if self.post_quant_sem_weight > 0:
            post_out_dim = self.merger_out_dim if self.merger_after_quant else self.encoder.width
            self.post_clip_projector = _make_sem_projector(
                self.quantize.post_quant_dim, post_out_dim, hidden=sem_hidden,
                proj_type=self.sem_projector_type)
        else:
            self.post_clip_projector = None

    def _build_quantizer(self, vq_cfg, loss_cfg, config, in_channel, out_channel,
                         code_shape, latent_shape, enable_simvq, entropy_loss_weight):
        """Overridable quantizer factory. Base RVQTokenizer builds ProductVQ (when
        ``pq_num_groups`` is set) or residual RVQ; subclasses (e.g. IBQTokenizer)
        override this to swap the quantization method while reusing every other
        piece of the feature-recon pipeline. Returns the quantizer module."""
        # ``pq_num_groups`` switches the channel-split Product Quantizer
        # (channel-split, parallel per-group codebooks) in place of residual RVQ.
        # One patch -> pq_num_groups parallel tokens; ``code_dim`` is split into
        # ``code_dim / pq_num_groups``-wide sub-vectors, each with its own
        # ``codebook_size`` codebook. Everything else (feature-recon, decoder,
        # losses) is identical to the RVQ path.
        pq_num_groups = vq_cfg.get("pq_num_groups", None)
        if pq_num_groups is not None:
            pq_code_dim = vq_cfg.get("code_dim", None)
            if pq_code_dim is None:
                raise ValueError("pq_num_groups requires code_dim to be set explicitly.")
            # ``grouped_quant_method`` swaps the disjoint channel-split for an
            # overlapping / factorized grouping (each patch still -> num_groups
            # parallel tokens, EMA + dead-code restart unchanged). None = vanilla
            # ProductVQ (disjoint split).
            grouped_method = vq_cfg.get("grouped_quant_method", None)
            _PQ_CLASSES = {
                "sliding": SlidingWindowVQ,
                "balanced": BalancedOverlapVQ,
                "factorized": FactorizedVQ,
            }
            pq_common = dict(
                in_channel=in_channel,
                out_channel=out_channel,
                code_dim=int(pq_code_dim),
                num_groups=int(pq_num_groups),
                n_embed=vq_cfg.codebook_size,
                codebook_update_mode=vq_cfg.get("codebook_update_mode", "ema"),
                enable_simvq=enable_simvq,
                commitment_loss_type=loss_cfg.get("commitment_loss_type", "cumsum"),
                commitment_weight=loss_cfg.get("commitment_weight", 0.25),
                decay=vq_cfg.get("decay", 0.99),
                restart_unused_codes=vq_cfg.get("restart_unused_codes", True),
                restart_threshold_ratio=vq_cfg.get("restart_threshold_ratio", 0.03),
                restart_max_per_step=vq_cfg.get("restart_max_per_step", 64),
                use_l2_norm=vq_cfg.get("use_l2_norm", False),
                entropy_loss_weight=entropy_loss_weight,
                sim_weight_norm_codebook=vq_cfg.get("sim_weight_norm_codebook", False),
                sim_sync_buffer=vq_cfg.get("sim_sync_buffer", True),
                sim_codebook_transform=vq_cfg.get("sim_codebook_transform", "linear"),
                sim_transform_mlp_ratio=vq_cfg.get("sim_transform_mlp_ratio", 2.0),
                frozen_codebook_dim=vq_cfg.get("frozen_codebook_dim", None),
                force_in_out_proj=vq_cfg.get("force_in_out_proj", True),
                in_proj_type=vq_cfg.get("in_proj_type", "attn_proj"),
                out_proj_type=vq_cfg.get("out_proj_type", "attn_proj"),
                proj_mlp_ratio=vq_cfg.get("proj_mlp_ratio", 4.0),
                proj_num_heads=vq_cfg.get("proj_num_heads", 16),
                attn_proj_mlp_ratio=vq_cfg.get("attn_proj_mlp_ratio", 2.0),
                pre_pq_l2_norm=vq_cfg.get("pre_pq_l2_norm", False),
                config=config,
            )
            if grouped_method is None:
                return ProductVQ(**pq_common)
            if grouped_method not in _PQ_CLASSES:
                raise ValueError(
                    f"unknown grouped_quant_method '{grouped_method}'; "
                    f"expected one of {sorted(_PQ_CLASSES)}")
            group_dim = vq_cfg.get(
                "group_dim", 2 * int(pq_code_dim) // int(pq_num_groups))
            return _PQ_CLASSES[grouped_method](
                group_dim=int(group_dim),
                grouped_seed=vq_cfg.get("grouped_seed", 0),
                **pq_common,
            )
        return RVQ(
            in_channel=in_channel,
            out_channel=out_channel,
            n_embed=vq_cfg.codebook_size,
            code_shape=code_shape,                                     # (h, w, d)
            latent_shape=latent_shape,
            codebook_update_mode=vq_cfg.get("codebook_update_mode", "ema"),
            enable_simvq=enable_simvq,
            shared_codebook=vq_cfg.get("shared_codebook", False),
            shared_group_num=vq_cfg.get("shared_group_num", 1),
            sem_group_size=vq_cfg.get("sem_group_size", None),
            commitment_loss_type=loss_cfg.get("commitment_loss_type", "cumsum"),
            decay=vq_cfg.get("decay", 0.99),
            restart_unused_codes=vq_cfg.get("restart_unused_codes", True),
            restart_threshold_ratio=vq_cfg.get("restart_threshold_ratio", 0.03),
            restart_max_per_step=vq_cfg.get("restart_max_per_step", 64),
            commitment_weight=loss_cfg.get("commitment_weight", 0.25),
            entropy_loss_weight=entropy_loss_weight,
            use_l2_norm=vq_cfg.get("use_l2_norm", False),   # Pass L2 norm setting
            sim_weight_norm_codebook=vq_cfg.get("sim_weight_norm_codebook", False),
            sim_sync_buffer=vq_cfg.get("sim_sync_buffer", True),
            sim_codebook_transform=vq_cfg.get("sim_codebook_transform", "linear"),
            sim_transform_mlp_ratio=vq_cfg.get("sim_transform_mlp_ratio", 2.0),
            frozen_codebook_dim=vq_cfg.get("frozen_codebook_dim", None),
            rotation_trick=vq_cfg.get("rotation_trick", False),
            quantize_dropout=vq_cfg.get("quantize_dropout", False),
            quantize_dropout_cutoff_index=vq_cfg.get("quantize_dropout_cutoff_index", 0),
            quantize_dropout_multiple_of=vq_cfg.get("quantize_dropout_multiple_of", 1),
            force_in_out_proj=vq_cfg.get("force_in_out_proj", True),
            in_proj_type=vq_cfg.get("in_proj_type", "attn_proj"),
            out_proj_type=vq_cfg.get("out_proj_type", "attn_proj"),
            proj_mlp_ratio=vq_cfg.get("proj_mlp_ratio", 4.0),
            proj_num_heads=vq_cfg.get("proj_num_heads", 16),
            attn_proj_mlp_ratio=vq_cfg.get("attn_proj_mlp_ratio", 2.0),
            use_whiten=vq_cfg.get("use_whiten", False),
            whiten_momentum=vq_cfg.get("whiten_momentum", 0.01),
            whiten_eps=vq_cfg.get("whiten_eps", 1e-5),
            use_center=vq_cfg.get("use_center", False),
            center_momentum=vq_cfg.get("center_momentum", 0.01),
            config=config,
        )

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
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

    def encode(self, x, image_grid_thw=None):
        """Encode images to (z_quantized, result_dict).

        Static (``image_grid_thw=None``):
            ``x``: (B, 3, H, W); returns ``z_quantized=(B, L, decoder.width)``.

        Dynamic varlen:
            ``x``: list of (3, H_i, W_i) tensors; ``image_grid_thw=(B, 3)``;
            returns ``z_quantized=(seq_total, decoder.width)`` and
            ``result_dict['cu_seqlens']=(B+1,)`` int32.
        """
        # Teacher (frozen) branch only produces ``clip_gt`` for the distillation
        # / clip losses. In feature_recon_only mode those are all off, so skip
        # the frozen forward entirely (return_clip_gt=False) to save compute.
        want_clip_gt = not self.feature_recon_only
        if image_grid_thw is None:
            if want_clip_gt:
                z, clip_gt = self.encoder(pixel_values=x, return_clip_gt=True)
            else:
                z, clip_gt = self.encoder(pixel_values=x, return_clip_gt=False), None
            z_quantized, result_dict = self.quantize(z)
            cu_seqlens = None
        else:
            if want_clip_gt:
                z, clip_gt, cu_seqlens = self.encoder(
                    pixel_values=x, return_clip_gt=True, image_grid_thw=image_grid_thw)
            else:
                z, cu_seqlens = self.encoder(
                    pixel_values=x, return_clip_gt=False, image_grid_thw=image_grid_thw)
                clip_gt = None
            z_quantized, result_dict = self.quantize(z, cu_seqlens=cu_seqlens)
            result_dict["cu_seqlens"] = cu_seqlens
            result_dict["image_grid_thw"] = image_grid_thw

        # Bridge merged token count back to the original count for the decoder
        # (decoder.width already matches z_quantized's dim).
        if self.merger_after_quant:
            # Stash the pre-unmerge (merged-space) quantized feature for the
            # FeatureDecoder distillation branch: it reconstructs ``z`` (also in
            # merged space) so it must consume the T/ms2-token feature, not the
            # T-token unmerged one that goes to the pixel decoder.
            if self.feature_recon_only or self.aux_feature_distill:
                result_dict["z_quantized_merged"] = z_quantized
            # Detached pixel-decoder warm-up: cut the graph at the unmerger's
            # INPUT so the pixel branch (post_quant_unmerger + UnifiedDecoder)
            # trains end-to-end, but no gradient reaches the RVQ / out_proj. The
            # FeatureDecoder still consumes the NON-detached z_quantized_merged
            # above, so RVQ keeps its feature-recon gradient.
            unmerge_in = (z_quantized.detach()
                          if self.detach_pixel_decoder else z_quantized)
            z_quantized = self.post_quant_unmerger(
                unmerge_in, cu_seqlens=cu_seqlens)

        result_dict["clip_gt"] = clip_gt
        if self.pre_clip_projector is not None:
            if self.merger_after_quant:
                # Student feature ``z`` is already merged; projector maps it to
                # merger_out_dim. Target is frozen_merger(clip_gt).
                result_dict["pre_clip_pred"] = self.pre_clip_projector(z)
            else:
                result_dict["pre_clip_pred"] = self._apply_sem_merger(self.pre_clip_projector(z))
        if self.post_clip_projector is not None:
            depth_list = result_dict["quant_at_depth_with_grad"]
            target = depth_list[self.post_quant_align_depth - 1]
            if self.merger_after_quant:
                # ``target`` (pre-unmerger) already lives in merger-out space.
                result_dict["post_clip_pred"] = self.post_clip_projector(target)
            else:
                result_dict["post_clip_pred"] = self._apply_sem_merger(self.post_clip_projector(target))

        if self.merger_after_quant and (
                self.pre_clip_projector is not None or self.post_clip_projector is not None):
            with torch.no_grad():
                merged_target = self._apply_frozen_merger(clip_gt)
            if self.pre_clip_projector is not None:
                if self.pre_quant_sem_target == "pre_resmod":
                    # cos(proj(post-ResMod z), pre-ResMod merge_target): leash
                    # ResMod to its own frozen-merger input, not the teacher CLIP.
                    mt = getattr(self.encoder, "_last_merge_target", None)
                    result_dict["pre_clip_target"] = (mt if mt is not None else z).detach()
                else:
                    result_dict["pre_clip_target"] = merged_target
            if self.post_clip_projector is not None:
                result_dict["post_clip_target"] = merged_target
        elif self.merger_as_sem_proj and (
                self.pre_clip_projector is not None or self.post_clip_projector is not None):
            merged_target = self._apply_sem_merger(clip_gt)
            if self.pre_clip_projector is not None:
                if self.pre_quant_sem_target == "pre_resmod":
                    mt = getattr(self.encoder, "_last_merge_target", None)
                    result_dict["pre_clip_target"] = (mt if mt is not None else z).detach()
                else:
                    result_dict["pre_clip_target"] = merged_target
            if self.post_clip_projector is not None:
                result_dict["post_clip_target"] = merged_target
        elif self.pre_clip_projector is not None or self.post_clip_projector is not None:
            # 16x (no Qwen merger): the sem target lives in encoder.width
            # directly. Continuous mode has no teacher, so pre_quant_sem_target
            # is forced to 'pre_resmod' -> target = same-forward pre-ResMod ViT
            # output (self.encoder._last_merge_target), aligned 1:1 with
            # pre_clip_pred = pre_clip_projector(post-ResMod z).
            if self.pre_clip_projector is not None:
                if self.pre_quant_sem_target == "pre_resmod":
                    mt = getattr(self.encoder, "_last_merge_target", None)
                    result_dict["pre_clip_target"] = (mt if mt is not None else z).detach()
                else:
                    result_dict["pre_clip_target"] = clip_gt
            if self.post_clip_projector is not None:
                result_dict["post_clip_target"] = clip_gt

        if self.feature_recon_only or self.aux_feature_distill:
            # Feature-distillation target: the frozen-merger semantic feature
            # WITHOUT ResMod's SFT modulation. Under merger_after_quant (stage-2)
            # ``z`` = ResMod(merge_target) is a moving target that ResMod also
            # feeds into the quantizer, so distilling to it is degenerate; the
            # encoder side-channels the pre-ResMod ``merge_target`` instead. On
            # the stage-1 path (qwen_use_merger, no ResMod) ``z`` already IS the
            # pure frozen-merger output, so it stays the target.
            #
            # ``feature_recon_target='post_resmod'`` (stage-B, ResMod frozen)
            # instead targets the post-ResMod ``z`` directly: ResMod is fixed so
            # z is a stable, meaningful reconstruction target.
            mt = getattr(self.encoder, "_last_merge_target", None)
            tz = getattr(self.encoder, "_last_teacher_z", None)
            if self.feature_recon_target == "post_resmod":
                # With a frozen teacher ResMod (stage-C joint-ft) the distill
                # target is the TEACHER's post-ResMod output (stable, frozen),
                # NOT the student ``z``. This lets the student ``z`` flow into
                # the quantizer un-detached while still being supervised toward
                # a fixed post-ResMod feature. Falls back to ``z`` (the original
                # stage-B behaviour) when no teacher is present.
                feat_target = tz if tz is not None else z
            else:
                feat_target = mt if mt is not None else z
            result_dict["feature_target"] = feat_target.detach()
            # Independent pre-ResMod anchor for the resmod-preserve loss, so it
            # stays meaningful even when feature_target == post-ResMod z
            # (feature_recon_target='post_resmod'). resmod-preserve always pulls
            # post-ResMod z toward the frozen-merger pre-ResMod merge_target.
            if mt is not None:
                result_dict["resmod_preserve_anchor"] = mt.detach()
        # Encoder output ``z`` actually fed to the quantizer (post-ResMod +
        # post-LN on the stage-2 path; pure merger output on stage-1).
        # ``encoder_z`` is detached and exposed read-only for the linear-probe
        # monitor's "post_resmod" tap (the exact tensor entering in_proj).
        result_dict["encoder_z"] = z.detach()
        # ``encoder_z_graph`` keeps the graph so the resmod-preserve cosine loss
        # can backprop into ResMod (target side uses the detached anchor). Route
        # it to the PRE-LN student post-ResMod z when the encoder exposes it
        # (symmetric teacher-distill 16x path) so the leash lives in the same
        # un-normalized space as the pre-ResMod anchor and the student LN is
        # excluded from the preserve objective. Falls back to ``z`` otherwise.
        z_preln = getattr(self.encoder, "_last_student_z_preln", None)
        result_dict["encoder_z_graph"] = z_preln if z_preln is not None else z
        return z_quantized, result_dict

    def encode_continuous(self, x, image_grid_thw=None):
        """Continuous-AE (stage-A) encode: no quantization.

        Reuses the quantizer's in_proj / out_proj for the
        merger_out_dim -> code_dim -> quant_out_channel projection but SKIPS the
        codebook entirely (z_cont = out_proj(in_proj(z))), then expands the
        merged tokens back to T via the post_quant_unmerger for the pixel
        decoder. The codebook is never read or updated.
        """
        if image_grid_thw is None:
            z = self.encoder(pixel_values=x, return_clip_gt=False)
            cu_seqlens = None
        else:
            z, cu_seqlens = self.encoder(
                pixel_values=x, return_clip_gt=False, image_grid_thw=image_grid_thw)
        result_dict = {}
        if cu_seqlens is not None:
            result_dict["cu_seqlens"] = cu_seqlens
            result_dict["image_grid_thw"] = image_grid_thw

        # in_proj (merger_out_dim -> code_dim) then out_proj (code_dim ->
        # quant_out_channel), skipping the codebook lookup / STE.
        z_proj = self.quantize._apply_proj(self.quantize.in_proj, z, cu_seqlens)
        z_cont = self.quantize._apply_proj(self.quantize.out_proj, z_proj, cu_seqlens)

        # merger_after_quant (32x) built an unmerger that expands the merged
        # T/ms2 tokens back to T; the 16x path (no merger) keeps T tokens
        # throughout, so there is no unmerger to apply.
        z_quantized = (self.post_quant_unmerger(z_cont, cu_seqlens=cu_seqlens)
                       if self.post_quant_unmerger is not None else z_cont)

        # Anchor + graph features for the resmod-preserve cosine loss.
        mt = getattr(self.encoder, "_last_merge_target", None)
        result_dict["feature_target"] = (mt if mt is not None else z).detach()
        result_dict["encoder_z"] = z.detach()
        result_dict["encoder_z_graph"] = z

        # pre-quant semantic loss (pre_resmod target only in continuous mode):
        # cos(proj(post-ResMod z), pre-ResMod merge_target). No teacher branch.
        if self.pre_clip_projector is not None:
            result_dict["pre_clip_pred"] = self.pre_clip_projector(z)
            result_dict["pre_clip_target"] = (mt if mt is not None else z).detach()
        return z_quantized, result_dict

    def decode(self, z_quantized, image_grid_thw=None):
        if self.feature_recon_only:
            raise RuntimeError(
                "decode() is unavailable in feature_recon_only mode (image decoder disabled).")
        decoded, clip_pred = self.decoder(z_quantized, image_grid_thw=image_grid_thw)
        return decoded, clip_pred

    def decode_tokens(self, tokens):
        """Decode RVQ tokens: (B, L, D) -> images."""
        if self.feature_recon_only:
            raise RuntimeError(
                "decode_tokens() is unavailable in feature_recon_only mode (image decoder disabled).")
        batch, seq_len, depth = tokens.shape
        z_quantized = self.quantize.get_codebook_entry(tokens).reshape(batch, seq_len, self.decoder.width)
        if self.merger_after_quant:
            z_quantized = self.post_quant_unmerger(z_quantized)
        decoded = self.decode(z_quantized)[0]
        return decoded

    def _merged_grid_thw(self, image_grid_thw):
        """Map the patch grid to the post-quant token grid. With the Qwen merger
        on, each ``spatial_merge_size`` block of patches along h/w collapses to one
        token (encoder already did ``cu //= merge_factor``), so the pos-embed must
        be laid out at the merged resolution. Without the merger, returned as-is."""
        if image_grid_thw is None:
            return None
        merger = getattr(self.encoder, "qwen_merger", None)
        if merger is None:
            return image_grid_thw
        ms = int(getattr(self.encoder, "qwen_spatial_merge_size", 1))
        if ms == 1:
            return image_grid_thw
        grid = image_grid_thw
        if not torch.is_tensor(grid):
            grid = torch.tensor(grid, dtype=torch.long)
        grid = grid.clone()
        grid[:, 1] = grid[:, 1] // ms
        grid[:, 2] = grid[:, 2] // ms
        return grid

    def forward(self, input, image_grid_thw=None):
        if self.continuous:
            # Stage-A continuous-AE path: no quantization, pixel reconstruction
            # + resmod-preserve cosine loss only.
            z_quantized, result_dict = self.encode_continuous(
                input, image_grid_thw=image_grid_thw)
            decoded, clip_pred = self.decode(z_quantized, image_grid_thw=image_grid_thw)
            result_dict["clip_pred"] = clip_pred
            return decoded, result_dict
        z_quantized, result_dict = self.encode(input, image_grid_thw=image_grid_thw)
        if self.feature_recon_only:
            # Decode post-quant tokens back to the frozen encoder feature ``z``.
            # ``reconstructions`` (= recon_feat) is compared against
            # result_dict["feature_target"] (= z) with MSE, plus commit/vq loss.
            if self.feature_decoder is not None:
                # "attn": transformer decoder mixes tokens, head maps to encoder.width.
                # Under merger_after_quant the FeatureDecoder must consume the
                # MERGED-space quantized tokens (T/ms2), since feature_target = z
                # lives in merged space; encode() already unmerged z_quantized to
                # T tokens for the (absent) pixel decoder, so use the stashed
                # pre-unmerge tensor here.
                grid_thw_tok = self._merged_grid_thw(image_grid_thw)
                fd_in = (result_dict["z_quantized_merged"]
                         if self.merger_after_quant else z_quantized)
                recon_feat = self.feature_decoder(fd_in, image_grid_thw=grid_thw_tok)
            else:
                # "mlp": out_proj already mapped code_dim -> encoder.width pointwise.
                recon_feat = z_quantized
            return recon_feat, result_dict
        # Pixel decode. Under detach_pixel_decoder the graph was already cut at
        # the post_quant_unmerger's INPUT (in encode) for merger_after_quant, so
        # post_quant_unmerger + this decoder train on the pixel loss while the RVQ
        # stays untouched. The 16x path (no merger/unmerger) has nothing to cut in
        # encode, so detach the pixel decoder's input HERE instead; the
        # aux_feature_distill branch below keeps the NON-detached z_quantized so
        # the RVQ still receives its feature-recon gradient.
        pixel_in = z_quantized
        if self.detach_pixel_decoder and not self.merger_after_quant:
            pixel_in = z_quantized.detach()
        decoded, clip_pred = self.decode(pixel_in, image_grid_thw=image_grid_thw)
        result_dict["clip_pred"] = clip_pred
        if self.aux_feature_distill and self.feature_decoder is not None:
            # Auxiliary feature-distillation branch (stage-2): reconstruct the
            # pre-quant feature ``z`` from the MERGED-space quantized tokens.
            # ``feature_recon_pred`` is compared against feature_target (= z) with
            # MSE in losses.py. Uses the merged grid so token count == z's.
            grid_thw_tok = self._merged_grid_thw(image_grid_thw)
            z_merged = (result_dict["z_quantized_merged"]
                        if self.merger_after_quant else z_quantized)
            result_dict["feature_recon_pred"] = self.feature_decoder(
                z_merged, image_grid_thw=grid_thw_tok)
        return decoded, result_dict


class IBQTokenizer(RVQTokenizer):
    """Single-level Index-Backpropagation-Quantization tokenizer.

    Reuses RVQTokenizer's entire feature-recon pipeline (encoder / in-out proj /
    FeatureDecoder / feature_target / varlen batching) and only swaps the
    quantizer for :class:`IBQ` via the ``_build_quantizer`` hook. IBQ is depth-1
    and produces a result_dict already aligned with the RVQ contract, so no other
    override is needed. IBQ-specific knobs are read from the ``vq_model`` /
    ``losses`` config blocks.
    """

    def _build_quantizer(self, vq_cfg, loss_cfg, config, in_channel, out_channel,
                         code_shape, latent_shape, enable_simvq, entropy_loss_weight):
        return IBQ(
            in_channel=in_channel,
            out_channel=out_channel,
            n_embed=vq_cfg.codebook_size,
            token_size=int(vq_cfg.get("code_dim", 1152)),
            commitment_cost=loss_cfg.get("commitment_weight", 0.25),
            use_l2_norm=vq_cfg.get("use_l2_norm", False),
            entropy_loss_weight=entropy_loss_weight,
            entropy_temperature=vq_cfg.get("entropy_temperature", 0.01),
            sample_minimization_weight=vq_cfg.get("sample_minimization_weight", 1.0),
            batch_maximization_weight=vq_cfg.get("batch_maximization_weight", 1.0),
            restart_unused_codes=vq_cfg.get("restart_unused_codes", True),
            restart_threshold_ratio=vq_cfg.get("restart_threshold_ratio", 0.03),
            restart_max_per_step=vq_cfg.get("restart_max_per_step", None),
            usage_decay=vq_cfg.get("decay", 0.99),
            force_in_out_proj=vq_cfg.get("force_in_out_proj", True),
            in_proj_type=vq_cfg.get("in_proj_type", "rmsnorm_mlp"),
            out_proj_type=vq_cfg.get("out_proj_type", "rmsnorm_mlp"),
            proj_mlp_ratio=vq_cfg.get("proj_mlp_ratio", 4.0),
            proj_num_heads=vq_cfg.get("proj_num_heads", 16),
            attn_proj_mlp_ratio=vq_cfg.get("attn_proj_mlp_ratio", 2.0),
            config=config,
        )


class VQTokenizer(BaseTokenizer):
    """Tokenizer with standard Vector Quantization.

    Uses SigLIP2 encoder/decoder with a single codebook VQ.
    """

    def __init__(self, config):
        if isinstance(config, dict):
            config = OmegaConf.create(config)
        super().__init__()
        self.config = config

        self.decoder = UnifiedDecoder(config)
        self.apply(self._init_weights)
        self.encoder = UnifiedEncoder(config)

        # Standard VQ
        vq_cfg = config.model.vq_model
        loss_cfg = config.get("losses", {})
        token_size = vq_cfg.get("token_size", 32)
        self.quantize = VQ(
            in_channel=self.encoder.width,
            out_channel=self.decoder.width,
            n_embed=vq_cfg.codebook_size,
            token_size=token_size,
            update_mode=vq_cfg.get("update_mode", "gradient"),
            enable_simvq=vq_cfg.get("enable_simvq", False),
            commitment_cost=loss_cfg.get("commitment_weight", 0.25),
            use_l2_norm=vq_cfg.get("use_l2_norm", False),
            ema_decay=vq_cfg.get("ema_decay", 0.99),
            restart_unused_codes=vq_cfg.get("restart_unused_codes", True),
            restart_threshold_ratio=vq_cfg.get("restart_threshold_ratio", 0.03),
            restart_max_per_step=vq_cfg.get("restart_max_per_step", 64),
            entropy_loss_weight=loss_cfg.get("entropy_loss_weight", 0.1),
            sim_weight_norm_codebook=vq_cfg.get("sim_weight_norm_codebook", False),
            sim_sync_buffer=vq_cfg.get("sim_sync_buffer", True),
            force_in_out_proj=vq_cfg.get("force_in_out_proj", True),
            in_proj_type=vq_cfg.get("in_proj_type", "attn_proj"),
            out_proj_type=vq_cfg.get("out_proj_type", "attn_proj"),
            proj_mlp_ratio=vq_cfg.get("proj_mlp_ratio", 4.0),
            proj_num_heads=vq_cfg.get("proj_num_heads", 16),
            attn_proj_mlp_ratio=vq_cfg.get("attn_proj_mlp_ratio", 2.0),
            config=config,
        )
        self._token_size = token_size

        # Pre/Post-quant semantic projectors. VQ has a single quant level so
        # post_quant_align_depth is irrelevant.
        # Sem-loss weights live on the `losses` block — projector is built iff
        # the weight > 0 (single source of truth shared with ReconstructionLoss).
        loss_cfg = config.get("losses", {})
        self.pre_quant_sem_weight = loss_cfg.get("pre_quant_sem_weight", 0.0)
        self.post_quant_sem_weight = loss_cfg.get("post_quant_sem_weight", 0.0)
        sem_hidden = vq_cfg.get("sem_projector_hidden", 2048)
        if self.pre_quant_sem_weight > 0:
            self.pre_clip_projector = _make_sem_projector(
                self.encoder.width, self.encoder.width, hidden=sem_hidden)
        else:
            self.pre_clip_projector = None
        if self.post_quant_sem_weight > 0:
            self.post_clip_projector = _make_sem_projector(
                self.quantize.post_quant_dim, self.encoder.width, hidden=sem_hidden)
        else:
            self.post_clip_projector = None

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
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

    def encode(self, x, image_grid_thw=None):
        if image_grid_thw is None:
            z, clip_gt = self.encoder(pixel_values=x, return_clip_gt=True)
            z_quantized, result_dict = self.quantize(z)
        else:
            z, clip_gt, cu_seqlens = self.encoder(
                pixel_values=x, return_clip_gt=True, image_grid_thw=image_grid_thw)
            z_quantized, result_dict = self.quantize(z, cu_seqlens=cu_seqlens)
            result_dict["cu_seqlens"] = cu_seqlens
            result_dict["image_grid_thw"] = image_grid_thw

        result_dict["clip_gt"] = clip_gt
        if self.pre_clip_projector is not None:
            result_dict["pre_clip_pred"] = self.pre_clip_projector(z)
        if self.post_clip_projector is not None:
            result_dict["post_clip_pred"] = self.post_clip_projector(
                result_dict["quant_at_depth_with_grad"][0])
        return z_quantized, result_dict

    def decode(self, z_quantized, image_grid_thw=None):
        decoded, clip_pred = self.decoder(z_quantized, image_grid_thw=image_grid_thw)
        return decoded, clip_pred

    def decode_tokens(self, tokens):
        """Decode VQ tokens: (B, L) or (B, L, 1) -> images."""
        tokens = tokens.squeeze(1)
        batch, seq_len = tokens.shape
        z_quantized = self.quantize.get_codebook_entry(tokens).reshape(batch, seq_len, -1)
        decoded = self.decode(z_quantized)[0]
        return decoded

    def forward(self, input, image_grid_thw=None):
        z_quantized, result_dict = self.encode(input, image_grid_thw=image_grid_thw)
        decoded, clip_pred = self.decode(z_quantized, image_grid_thw=image_grid_thw)
        result_dict["clip_pred"] = clip_pred
        return decoded, result_dict


class AETokenizer(BaseTokenizer):
    """Plain auto-encoder tokenizer: NO quantization, NO bottleneck.

    The encoder feature ``z`` (B, L, encoder.width) passes through an identity
    bottleneck straight into the decoder, so ``encoder.width`` must equal
    ``decoder.width``. Set ``model.vq_model.vit_dec_width`` to the encoder width
    (e.g. 1152 for the Qwen3.6 ViT). Use this to validate the encoder+decoder
    reconstruction path in isolation from any quantizer.

    Losses (all driven by ReconstructionLoss):
      - pixel reconstruction: L1 / L2 (+ perceptual / gram / optional GAN)
      - semantic distillation: PreQuantSemLoss on the encoder feature and the
        decoder-intermediate clip alignment (``clip_loss``).
    There are no commitment / codebook / entropy terms (no codebook exists), so
    ReconstructionLoss reads those as zero via its ``.get(..., 0)`` defaults.
    """

    def __init__(self, config):
        if isinstance(config, dict):
            config = OmegaConf.create(config)
        super().__init__()
        self.config = config

        self.decoder = UnifiedDecoder(config)
        self.apply(self._init_weights)
        self.encoder = UnifiedEncoder(config)

        if self.encoder.width != self.decoder.width:
            raise ValueError(
                f"AETokenizer uses an identity bottleneck, so encoder.width "
                f"({self.encoder.width}) must equal decoder.width "
                f"({self.decoder.width}). Set model.vq_model.vit_dec_width="
                f"{self.encoder.width} in the config."
            )

        # Identity bottleneck: no quantization, no in/out projection.
        self.quantize = nn.Identity()

        # ---- Semantic distillation projectors -------------------------------
        # Same pattern as the VQ tokenizers: PreQuantSemLoss aligns the encoder
        # feature z with the frozen-encoder clip_gt. The "post-quant" feature is
        # identical to z under the identity bottleneck, but a separate head is
        # exposed so post_quant_sem can be turned on independently if wanted.
        # Sem-loss weights live on the `losses` block (single source of truth).
        vq_cfg = config.model.vq_model
        loss_cfg = config.get("losses", {})
        self.pre_quant_sem_weight = loss_cfg.get("pre_quant_sem_weight", 0.0)
        self.post_quant_sem_weight = loss_cfg.get("post_quant_sem_weight", 0.0)
        sem_hidden = vq_cfg.get("sem_projector_hidden", 2048)
        if self.pre_quant_sem_weight > 0:
            self.pre_clip_projector = _make_sem_projector(
                self.encoder.width, self.encoder.width, hidden=sem_hidden)
        else:
            self.pre_clip_projector = None
        if self.post_quant_sem_weight > 0:
            self.post_clip_projector = _make_sem_projector(
                self.encoder.width, self.encoder.width, hidden=sem_hidden)
        else:
            self.post_clip_projector = None

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
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

    def encode(self, x, image_grid_thw=None):
        """Encode images to (z, result_dict). The bottleneck is identity, so the
        returned latent IS the encoder feature.

        Static (``image_grid_thw=None``): ``x=(B,3,H,W)`` -> ``z=(B,L,width)``.
        Dynamic varlen: ``x`` is a list of ``(3,H_i,W_i)``; ``image_grid_thw=(B,3)``;
        ``z=(seq_total, width)`` and ``result_dict['cu_seqlens']``.
        """
        if image_grid_thw is None:
            z, clip_gt = self.encoder(pixel_values=x, return_clip_gt=True)
            z_quantized = self.quantize(z)
            result_dict = {}
        else:
            z, clip_gt, cu_seqlens = self.encoder(
                pixel_values=x, return_clip_gt=True, image_grid_thw=image_grid_thw)
            z_quantized = self.quantize(z)
            result_dict = {"cu_seqlens": cu_seqlens, "image_grid_thw": image_grid_thw}

        result_dict["clip_gt"] = clip_gt
        if self.pre_clip_projector is not None:
            result_dict["pre_clip_pred"] = self.pre_clip_projector(z)
        if self.post_clip_projector is not None:
            result_dict["post_clip_pred"] = self.post_clip_projector(z_quantized)
        return z_quantized, result_dict

    def decode(self, z_quantized, image_grid_thw=None):
        decoded, clip_pred = self.decoder(z_quantized, image_grid_thw=image_grid_thw)
        return decoded, clip_pred

    def decode_tokens(self, tokens):
        raise NotImplementedError(
            "AETokenizer has no codebook / discrete tokens; use encode()+decode().")

    def forward(self, input, image_grid_thw=None):
        z_quantized, result_dict = self.encode(input, image_grid_thw=image_grid_thw)
        decoded, clip_pred = self.decode(z_quantized, image_grid_thw=image_grid_thw)
        result_dict["clip_pred"] = clip_pred
        return decoded, result_dict


class CausalUnifiedDecoder(UnifiedDecoder):
    """UnifiedDecoder with a causal attention mask (left-to-right aligned codes).

    The decoder uses causal masking so that the reconstruction of each spatial
    position only depends on previous positions, aligning with AR modeling.
    """

    def __init__(self, config):
        super().__init__(config)
        # Initialize causal mask
        seq_len = self.grid_size ** 2 + 1
        mask = torch.full((seq_len, seq_len), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, z_quantized):
        clip_pred = None
        x = z_quantized
        batchsize, seq_len, _ = x.shape

        x = torch.cat([_expand_token(self.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)

        # Use causal mask instead of full attention
        attn_mask = self.causal_mask[:x.shape[1], :x.shape[1]]
        x = self.ln_pre(x)

        for i in range(self.num_layers):
            if self.use_checkpoint:
                x = torch.utils.checkpoint.checkpoint(
                    self.transformer[i].forward, x, attn_mask, use_reentrant=False)
            else:
                x = self.transformer[i](x, attn_mask=attn_mask)

            if self.clip_align and i == self.clip_align_layer_id:
                clip_pred = self.clip_projector(x)[:, 1:1+self.grid_size**2]

        x = x[:, 1:1+self.grid_size**2]
        x = self.ln_post(x)

        x = x.permute(0, 2, 1).reshape(batchsize, self.width, self.grid_size, self.grid_size)
        x = self.ffn(x.contiguous())
        x = self.conv_out(x)
        return x, clip_pred


class AlignedVQTokenizer(BaseTokenizer):
    """Aligned tokenizer: causal decoder + standard VQ.

    Uses a causal decoder to align the discrete codes with a left-to-right
    (autoregressive-friendly) ordering, on top of the shared dNaViT modules.
    """

    def __init__(self, config):
        if isinstance(config, dict):
            config = OmegaConf.create(config)
        super().__init__()
        self.config = config

        # Causal decoder -> left-to-right aligned codes
        self.decoder = CausalUnifiedDecoder(config)
        self.apply(self._init_weights)
        self.encoder = UnifiedEncoder(config)

        # Standard VQ
        vq_cfg = config.model.vq_model
        loss_cfg = config.get("losses", {})
        self.quantize = VQ(
            in_channel=self.encoder.width,
            out_channel=self.decoder.width,
            n_embed=vq_cfg.codebook_size,
            token_size=vq_cfg.get("token_size", 32),
            update_mode=vq_cfg.get("update_mode", "gradient"),
            enable_simvq=vq_cfg.get("enable_simvq", False),
            commitment_cost=loss_cfg.get("commitment_weight", 0.25),
            use_l2_norm=vq_cfg.get("use_l2_norm", False),
            ema_decay=vq_cfg.get("ema_decay", 0.99),
            restart_unused_codes=vq_cfg.get("restart_unused_codes", True),
            restart_threshold_ratio=vq_cfg.get("restart_threshold_ratio", 0.03),
            restart_max_per_step=vq_cfg.get("restart_max_per_step", 64),
            entropy_loss_weight=loss_cfg.get("entropy_loss_weight", 0.1),
            sim_weight_norm_codebook=vq_cfg.get("sim_weight_norm_codebook", False),
            sim_sync_buffer=vq_cfg.get("sim_sync_buffer", True),
            force_in_out_proj=vq_cfg.get("force_in_out_proj", True),
            in_proj_type=vq_cfg.get("in_proj_type", "attn_proj"),
            out_proj_type=vq_cfg.get("out_proj_type", "attn_proj"),
            proj_mlp_ratio=vq_cfg.get("proj_mlp_ratio", 4.0),
            proj_num_heads=vq_cfg.get("proj_num_heads", 16),
            attn_proj_mlp_ratio=vq_cfg.get("attn_proj_mlp_ratio", 2.0),
            config=config,
        )

        # Pre/Post-quant semantic projectors.
        # Sem-loss weights live on the `losses` block — projector is built iff
        # the weight > 0 (single source of truth shared with ReconstructionLoss).
        loss_cfg = config.get("losses", {})
        self.pre_quant_sem_weight = loss_cfg.get("pre_quant_sem_weight", 0.0)
        self.post_quant_sem_weight = loss_cfg.get("post_quant_sem_weight", 0.0)
        sem_hidden = vq_cfg.get("sem_projector_hidden", 2048)
        if self.pre_quant_sem_weight > 0:
            self.pre_clip_projector = _make_sem_projector(
                self.encoder.width, self.encoder.width, hidden=sem_hidden)
        else:
            self.pre_clip_projector = None
        if self.post_quant_sem_weight > 0:
            self.post_clip_projector = _make_sem_projector(
                self.quantize.post_quant_dim, self.encoder.width, hidden=sem_hidden)
        else:
            self.post_clip_projector = None

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
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

    def encode(self, x, image_grid_thw=None):
        if image_grid_thw is not None:
            raise NotImplementedError(
                "AlignedVQTokenizer doesn't support dynamic resolution (causal "
                "decoder varlen attention is not implemented).")
        z, clip_gt = self.encoder(pixel_values=x, return_clip_gt=True)
        z_quantized, result_dict = self.quantize(z)
        result_dict["clip_gt"] = clip_gt

        if self.pre_clip_projector is not None:
            result_dict["pre_clip_pred"] = self.pre_clip_projector(z)
        if self.post_clip_projector is not None:
            result_dict["post_clip_pred"] = self.post_clip_projector(
                result_dict["quant_at_depth_with_grad"][0])
        return z_quantized, result_dict

    def decode(self, z_quantized, image_grid_thw=None):
        if image_grid_thw is not None:
            raise NotImplementedError(
                "AlignedVQTokenizer doesn't support dynamic resolution decode.")
        decoded, clip_pred = self.decoder(z_quantized)
        return decoded, clip_pred

    def decode_tokens(self, tokens):
        """Decode aligned-VQ tokens: (B, L) or (B, L, 1) -> images."""
        if len(tokens.shape) == 2:
            tokens = tokens.unsqueeze(-1)
        batch, seq_len, _ = tokens.shape
        z_quantized = self.quantize.get_codebook_entry(tokens.squeeze(-1)).reshape(batch, seq_len, -1)
        decoded = self.decode(z_quantized)[0]
        return decoded

    def forward(self, input, image_grid_thw=None):
        z_quantized, result_dict = self.encode(input, image_grid_thw=image_grid_thw)
        decoded, clip_pred = self.decode(z_quantized, image_grid_thw=image_grid_thw)
        result_dict["clip_pred"] = clip_pred
        return decoded, result_dict


