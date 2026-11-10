# Copyright (c) 2026 LongCat dNaViT Team.
# SPDX-License-Identifier: Apache-2.0

"""Training losses: autoencoder reconstruction (L1 / L2 / perceptual / Gram),
quantizer commitment / entropy terms, feature distillation and an optional
adversarial term."""
from typing import List, Mapping, Text, Tuple, Union
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.amp import autocast
from utils.paths import pretrained_path
from .perceptual_loss import PerceptualLoss
from .discriminator_dino import DinoDisc


# A loss-side input/recon sample may be either a dense (B, 3, H, W) tensor
# (fixed-resolution batch) or a list of per-sample (3, H_i, W_i) tensors
# (anyres / varlen batch).
SampleBatch = Union[torch.Tensor, List[torch.Tensor]]


def _is_list_batch(x: SampleBatch) -> bool:
    return isinstance(x, (list, tuple))


def _per_sample_pixel_loss(inp_list, rec_list, loss_fn):
    """Token-weighted average of a pixelwise loss across variable-shape samples.

    Falls back to a plain ``loss_fn(rec, inp, reduction='mean')`` when the
    inputs are dense (single tensor).  For lists, sums element-wise loss with
    ``reduction='sum'`` then divides by total numel — this matches the dense
    case exactly when all samples share a shape, and avoids biasing toward
    small images when shapes differ.
    """
    if not _is_list_batch(inp_list):
        return loss_fn(rec_list, inp_list, reduction='mean')

    total = inp_list[0].new_zeros(())
    count = 0
    for inp_i, rec_i in zip(inp_list, rec_list):
        total = total + loss_fn(rec_i, inp_i, reduction='sum')
        count += inp_i.numel()
    return total / max(count, 1)


def _resize_batch_for_perceptual(samples: SampleBatch, target: int = 256,
                                 mode: str = 'bilinear') -> torch.Tensor:
    """Coerce a (B,3,H,W) tensor OR a list[(3,H_i,W_i)] into a dense
    ``(B, 3, target, target)`` tensor for downstream LPIPS / discriminator /
    Gram. Always returns fp32 to keep LPIPS / disc numerically stable.

    For list inputs, each sample is interpolated then stacked.
    """
    if not _is_list_batch(samples):
        x = samples.float()
        if x.shape[-1] == target and x.shape[-2] == target:
            return x
        return F.interpolate(x, size=(target, target),
                             mode=mode, align_corners=False)

    parts = []
    for s in samples:
        s = s.float()
        if s.dim() == 3:
            s = s.unsqueeze(0)
        if s.shape[-1] != target or s.shape[-2] != target:
            s = F.interpolate(s, size=(target, target),
                              mode=mode, align_corners=False)
        parts.append(s.squeeze(0))
    return torch.stack(parts, dim=0)


def _cosine_distill_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Semantic distillation loss: ``mean(1 - cos_sim(pred, target))``.

    Cosine similarity L2-normalizes ``pred`` and ``target`` over the feature dim
    internally, so it is scale-invariant — the teacher (``clip_gt``) does NOT need
    to be normalized first. The target is detached (frozen teacher). Works for
    both ``(B, L, D)`` dense and ``(seq_total, D)`` packed layouts.
    """
    cos = F.cosine_similarity(pred, target.detach(), dim=-1)
    return (1.0 - cos).mean()


# Gram Loss
class GramLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # ImageNet-1K VGG16 backbone (torchvision weights).
        ckpt_path = pretrained_path("vgg16-397923af.pth")

        vgg16 = models.vgg16(weights=None)
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        vgg16.load_state_dict(state_dict)
        self.feature_extractor = vgg16.features.eval()

        self.register_buffer("imagenet_mean", torch.Tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer("imagenet_std", torch.Tensor([0.229, 0.224, 0.225])[None, :, None, None])

        for param in self.parameters():
            param.requires_grad = False

    @staticmethod
    def gram_matrix(features: torch.Tensor):
        """
        Compute Gram matrix for feature maps
        features: [N, C, H, W]
        returns: [N, C, C]
        """
        N, C, H, W = features.shape
        F_flat = features.view(N, C, H * W)  # [N, C, H*W]
        G = F_flat @ F_flat.transpose(1, 2)  # [N, C, C]
        return G / (C * H * W)  # normalize

    def forward(self, input: torch.Tensor, target: torch.Tensor,):
        """Computes the perceptual loss.

        Args:
            input: A tensor of shape (B, C, L), the input concatenated image. Normalized to [-1, 1].
            target: A tensor of shape (B, C, L), the target concatenated image. Normalized to [-1, 1].

        Returns:
            A scalar tensor, the perceptual loss.
        """
        # Always in eval mode.
        self.feature_extractor.eval()
        # [-1, 1] -> [0, 1]

        input_img = F.interpolate(input, size=(224, 224), mode="bilinear", align_corners=False)
        target_img = F.interpolate(target, size=(224, 224), mode="bilinear", align_corners=False)

        input_imgs = (input_img + 1.0) / 2.0
        target_imgs = (target_img + 1.0) / 2.0

        input_imgs = (input_imgs - self.imagenet_mean) / self.imagenet_std
        target_imgs = (target_imgs - self.imagenet_mean) / self.imagenet_std

        input_features = []
        target_features = []

        x_in = input_imgs
        x_tg = target_imgs
        for layer in self.feature_extractor:
            x_in = layer(x_in)
            x_tg = layer(x_tg)

            input_features.append(x_in)
            target_features.append(x_tg)

        total_loss = 0.0
        for f_in, f_tg in zip(input_features, target_features):
            G_in = self.gram_matrix(f_in)
            G_tg = self.gram_matrix(f_tg)
            total_loss += ((G_in - G_tg) ** 2).mean()

        return total_loss


def hinge_d_loss(logits_real: torch.Tensor, logits_fake: torch.Tensor) -> torch.Tensor:
    """Hinge loss for discriminator.

    Reference:
        the standard autoencoder perceptual-loss interface
    """
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


class ReconstructionLoss(torch.nn.Module):
    def __init__(self, config):
        """Initialize the losses module.

        Args:
            config: Configuration for the model and losses.
        """
        super().__init__()
        loss_config = config.losses
        disc_config = config.model.discriminator

        self.reconstruction_weight_l1 = loss_config.reconstruction_weight_l1
        self.reconstruction_weight_l2 = loss_config.reconstruction_weight_l2
        # Feature-reconstruction-only mode: the tokenizer has no image decoder,
        # so the backward target is purely the quantizer's feature-recon signal
        # (``quantizer_loss``). All pixel-side modules below are skipped.
        self.feature_recon_only = config.model.vq_model.get("feature_recon_only", False)
        # Weight on the explicit feature-recon term MSE(z_quantized, z) used only
        # in feature_recon_only mode (added on top of the quantizer's commit/vq).
        self.feature_recon_weight = loss_config.get("feature_recon_weight", 1.0)
        # Auxiliary feature-distillation (stage-2 pixel mode warm-started from a
        # feature_recon stage-1): keep the FeatureDecoder + MSE(recon_feat, z) as
        # an extra supervision running alongside the pixel-reconstruction losses.
        self.aux_feature_distill = bool(config.model.vq_model.get("aux_feature_distill", False))
        self.feature_distill_weight = loss_config.get("feature_distill_weight", 1.0)
        # Optional cosine companion to the feature-distillation MSE: same pair
        # (FeatureDecoder(quant) vs feature_target) but direction-only, so the
        # codebook is pulled toward the target in angle as well as magnitude.
        # Off by default.
        self.feature_distill_cos_weight = loss_config.get("feature_distill_cos_weight", 0.0)
        # ResMod-preserve cosine loss: pull the post-ResMod feature ``z``
        # (encoder_z_graph, with grad) toward the pre-ResMod frozen-merger anchor
        # (feature_target, detached). Keeps ResMod from drifting semantically
        # while it adds pixel detail. Large weight in stage-A (continuous-AE),
        # small weight in stage-C (fine-tune leash).
        self.resmod_preserve_weight = loss_config.get("resmod_preserve_weight", 0.0)
        # Optional MSE companion to the resmod-preserve cosine: same pair
        # (post-ResMod z vs pre-ResMod anchor) but L2 instead of cosine, so the
        # leash also constrains magnitude, not just direction. Off by default.
        self.resmod_preserve_mse_weight = loss_config.get("resmod_preserve_mse_weight", 0.0)
        if self.feature_recon_only:
            self.perceptual_loss = None
        else:
            self.perceptual_loss = PerceptualLoss(
                loss_config.perceptual_loss).eval()
        self.perceptual_weight = loss_config.perceptual_weight
        # InternVL-style any-res tiled perceptual loss: when enabled, LPIPS +
        # ConvNeXt run on global + local 224 tiles per image (OOM-safe chunking)
        # instead of one 256-resized batch. Default off -> dense path unchanged.
        self.perceptual_anyres = loss_config.get("perceptual_anyres", False)
        # Fixed square side for the plain (non-anyres) perceptual path: each image
        # is bilinear-resized to perceptual_size^2 before LPIPS/ConvNeXt.
        self.perceptual_size = loss_config.get("perceptual_size", 224)
        self.perceptual_tile_size = loss_config.get("perceptual_tile_size", 224)
        self.perceptual_tile_batch_size = loss_config.get("perceptual_tile_batch_size", 32)
        self.perceptual_global_weight = loss_config.get("perceptual_global_weight", 1.0)
        self.perceptual_local_weight = loss_config.get("perceptual_local_weight", 1.0)
        self.discriminator_iter_start = loss_config.discriminator_start
        self.discriminator_factor = loss_config.discriminator_factor
        self.discriminator_weight = loss_config.discriminator_weight

        # Discriminator: only create if discriminator_weight > 0 (never in
        # feature_recon_only mode, which has no reconstructions to feed it).
        self.use_discriminator = self.discriminator_weight > 0.0 and not self.feature_recon_only

        if self.use_discriminator:
            # Only use DinoDisc as discriminator
            # Local DINO ViT-S/8 checkpoint, or an http(s) URL.
            dino_ckpt_path = disc_config.get("dino_ckpt_path", "assets/models/dino_vit_small_patch8_224.pth")
            norm_type = disc_config.get("norm_type", "bn")
            self.discriminator = DinoDisc(dino_ckpt_path=dino_ckpt_path, norm_type=norm_type)
        else:
            print("Discriminator is disabled (discriminator_weight = 0)")
            self.discriminator = None

        # Gram loss: independent of discriminator
        self.gram_loss_weight = loss_config.get("gram_loss_weight", 0.0)
        if self.gram_loss_weight > 0:
            print(f"Gram loss enabled with weight: {self.gram_loss_weight}")
            self.gram_loss = GramLoss()
        else:
            self.gram_loss = None


        self.config = config

        # CLIP loss: frozen CLIP features are now computed in the encoder and passed via result_dict
        self.clip_loss_weight = loss_config.get("clip_loss_weight", 0.0)

        # Semantic distillation losses (independent from pixel reconstruction):
        #   PreQuantSemLoss:  MSE(pre_clip_pred, clip_gt)  — supervises encoder
        #   PostQuantSemLoss: MSE(post_clip_pred, clip_gt) — supervises codebook
        # Both are backwarded but logged separately from pixel-recon-side losses.
        self.pre_quant_sem_weight = loss_config.get("pre_quant_sem_weight", 0.0)
        # Optional MSE companion on the pre_quant_sem pair (same pred/target as the
        # cos term): magnitude leash alongside the cosine direction term.
        self.pre_quant_sem_mse_weight = loss_config.get("pre_quant_sem_mse_weight", 0.0)
        self.post_quant_sem_weight = loss_config.get("post_quant_sem_weight", 0.0)

        # Side length the DINO discriminator (GAN) input is resized to — it
        # needs a fixed square (B,3,N,N) and is pretrained at this scale (256 →
        # DINO ViT-S/8). Gram reuses this dense batch (it re-resizes to 224
        # internally). LPIPS / ConvNeXt do NOT use this: they run at native
        # resolution (anyres = global+local 224 tiles; fixed-res = crop size).
        # L1 / MSE stay on the original per-sample shapes (_per_sample_pixel_loss).
        self.disc_resize_target = int(loss_config.get("disc_resize_target", 256))

        # Note: torch.compile is applied after accelerator.prepare() in train_tok.py
        # Discriminator-related methods use @torch._dynamo.disable to exclude them from compilation

    # @torch.amp.autocast(device_type='cuda', enabled=False)
    def forward(self,
                inputs: torch.Tensor,
                reconstructions: torch.Tensor,
                extra_result_dict: Mapping[Text, torch.Tensor],
                global_step: int,
                mode: str = "generator",
                ) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        # Both inputs and reconstructions are in range [-1, 1].
        # inputs = inputs.float()
        # reconstructions = reconstructions.float()

        if mode == "generator":
            return self._forward_generator(inputs, reconstructions, extra_result_dict, global_step)
        elif mode == "discriminator":
            return self._forward_discriminator(inputs, reconstructions, global_step)
        else:
            raise ValueError(f"Unsupported mode {mode}")
   
    def should_discriminator_be_trained(self, global_step : int):
        return self.use_discriminator and global_step >= self.discriminator_iter_start

    def _forward_generator(self,
                           inputs: SampleBatch,
                           reconstructions: SampleBatch,
                           extra_result_dict: Mapping[Text, torch.Tensor],
                           global_step: int,
                           ) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        """Generator training step.

        ``inputs`` / ``reconstructions`` may be either:
          - a dense ``(B, 3, H, W)`` tensor (fixed-resolution / c2i path), or
          - a list of ``(3, H_i, W_i)`` per-sample tensors (anyres / t2i path).

        L1 / MSE losses are computed per-sample (token-weighted average) so big
        images don't dominate. LPIPS / ConvNeXt run at native resolution (anyres:
        global + local 224 tiles; fixed-res: crop size). Only the discriminator
        (and Gram, which internally goes to 224) get a dense batch resized to
        ``self.disc_resize_target``.
        """
        # Feature-reconstruction-only mode: the tokenizer returns no pixel
        # reconstructions (``reconstructions`` = z_quantized, the out_proj
        # output). Backward = explicit feature-recon MSE(z_quantized, z) plus
        # the quantizer's commit/vq loss; all pixel-side modules are skipped.
        # Diagnostics still flow through so codebook usage is logged.
        if self.feature_recon_only:
            quantizer_loss = extra_result_dict["quantizer_loss"]
            ref_device = quantizer_loss.device
            zero = torch.zeros((), device=ref_device)
            commitment_loss = extra_result_dict.get("commitment_loss", zero)
            vq_loss = extra_result_dict.get("vq_loss", zero)
            entropy_loss = extra_result_dict.get("entropy_loss", zero)
            # Explicit feature reconstruction: out_proj(z_q) should reproduce the
            # frozen encoder feature ``z``. Trains in_proj + out_proj end-to-end.
            feature_target = extra_result_dict["feature_target"]
            feature_recon_loss = F.mse_loss(reconstructions, feature_target)
            total_loss = quantizer_loss + self.feature_recon_weight * feature_recon_loss
            loss_dict = dict(
                total_loss=total_loss.clone().detach(),
                quantizer_loss=quantizer_loss.detach(),
                commitment_loss=commitment_loss.detach() if torch.is_tensor(commitment_loss) else commitment_loss,
                vq_loss=vq_loss.detach() if torch.is_tensor(vq_loss) else vq_loss,
                entropy_loss=entropy_loss.detach() if torch.is_tensor(entropy_loss) else entropy_loss,
                feature_recon_loss=feature_recon_loss.detach(),
            )
            _diag_prefixes = (
                "codebook_usage/", "codebook_std/", "codebook_norm/",
                "commit_per_depth/", "commit_per_group/", "vq_per_group/",
            )
            for k, v in extra_result_dict.items():
                if not isinstance(k, str) or not k.startswith(_diag_prefixes):
                    continue
                if not torch.is_tensor(v):
                    v = torch.as_tensor(float(v), device=ref_device)
                loss_dict[k] = v.detach()
            return total_loss, loss_dict

        list_mode = _is_list_batch(inputs)
        if list_mode:
            inputs_list = [inp.contiguous() for inp in inputs]
            recon_list = [rec.contiguous() for rec in reconstructions]
            ref_device = inputs_list[0].device
        else:
            inputs_list = inputs.contiguous()
            recon_list = reconstructions.contiguous()
            ref_device = inputs_list.device

        # We always ensure the images are normalized to [-1, 1]
        # Track L1 / L2 components separately so the train log can show each
        # one independently. ``reconstruction_loss`` stays as the weighted sum
        # contributing to ``pixel_recon_total``.
        recon_l1 = torch.zeros((), device=ref_device)
        recon_l2 = torch.zeros((), device=ref_device)
        reconstruction_loss = torch.zeros((), device=ref_device)
        if self.reconstruction_weight_l1 > 0:
            recon_l1 = _per_sample_pixel_loss(inputs_list, recon_list, F.l1_loss)
            reconstruction_loss = reconstruction_loss + recon_l1 * self.reconstruction_weight_l1
        if self.reconstruction_weight_l2 > 0:
            recon_l2 = _per_sample_pixel_loss(inputs_list, recon_list, F.mse_loss)
            reconstruction_loss = reconstruction_loss + recon_l2 * self.reconstruction_weight_l2

        # ``disc_resize_target`` sizes the DISCRIMINATOR (GAN) input only — the
        # DINO disc needs a fixed square (B,3,N,N). Gram also reuses this dense
        # batch (it re-resizes to 224 internally, so the value is moot for it).
        # Perceptual/LPIPS never uses it: see below.
        needs_dense = self.gram_loss_weight > 0 or self.use_discriminator
        if needs_dense:
            inputs_dense = _resize_batch_for_perceptual(
                inputs_list, target=self.disc_resize_target)
            recon_dense = _resize_batch_for_perceptual(
                recon_list, target=self.disc_resize_target)
        else:
            inputs_dense = None
            recon_dense = None

        # Compute perceptual loss — fully independent of disc_resize_target.
        if self.perceptual_weight > 0:
            if self.perceptual_anyres:
                # Any-res tiled path: per-image global + local 224 tiles on
                # native-res inputs. recon carries grad; inputs detached inside.
                perceptual_loss = self.perceptual_loss.forward_anyres(
                    recon_list, inputs_list,
                    tile_size=self.perceptual_tile_size,
                    max_batch_size=self.perceptual_tile_batch_size,
                    global_weight=self.perceptual_global_weight,
                    local_weight=self.perceptual_local_weight,
                )
            else:
                # Plain fixed-res LPIPS: bilinear-resize every (possibly
                # variable-res) image to a fixed 224x224 square, stack into a
                # dense (B,3,224,224) batch, then a single perceptual forward.
                # recon keeps grad through interpolate; the dense forward weights
                # LPIPS(VGG) + ConvNeXt-S exactly like the fixed-res path.
                inp_dense = _resize_batch_for_perceptual(
                    inputs_list, target=self.perceptual_size)
                rec_dense = _resize_batch_for_perceptual(
                    recon_list, target=self.perceptual_size)
                perceptual_loss = self.perceptual_loss(inp_dense, rec_dense).mean()
        else:
            perceptual_loss = torch.zeros((), device=ref_device)

        # Compute CLIP loss using frozen CLIP features from encoder. Both
        # tensors are already in the same layout (packed (seq_total, D) for
        # varlen, (B, L, D) for dense) — no reshape needed.
        clip_loss = torch.zeros((), device=ref_device)
        if self.clip_loss_weight > 0:
            clip_pred = extra_result_dict["clip_pred"]
            clip_gt = extra_result_dict["clip_gt"]
            clip_loss = _cosine_distill_loss(clip_pred, clip_gt)

        # Compute Gram loss
        gram_loss = torch.zeros((), device=ref_device)
        generator_loss = torch.zeros((), device=ref_device)
        discriminator_factor = 0.
        d_weight = 0.
        if self.gram_loss_weight > 0:
            gram_loss = self.gram_loss(inputs_dense, recon_dense)

        if self.use_discriminator:
            # Discriminator sees the fixed-res reconstructions.
            generator_loss, d_weight, discriminator_factor = self._compute_generator_loss(
                recon_dense, global_step)

        # Quantizer losses are already weighted inside the quantizer (commitment_weight,
        # entropy_loss_weight, etc.) and pre-summed into ``quantizer_loss``. FSQ has no
        # quantizer_loss; default to zero so the dict access doesn't fail.
        zero = torch.zeros((), device=ref_device)
        quantizer_loss = extra_result_dict.get("quantizer_loss", zero)
        commitment_loss = extra_result_dict.get("commitment_loss", zero)
        codebook_loss = extra_result_dict.get("codebook_loss", zero)  # VQ only
        vq_loss = extra_result_dict.get("vq_loss", zero)              # RVQ only
        entropy_loss = extra_result_dict.get("entropy_loss", zero)

        # ---- Semantic distillation (independent block) ----------------------
        # PreQuantSemLoss aligns encoder's last hidden state with the frozen
        # encoder's CLIP features; PostQuantSemLoss aligns the codebook lookup
        # feature (cumulative up to ``post_quant_align_depth`` for RVQ) with the
        # same target. Kept on a separate ledger from pixel reconstruction so
        # wandb panels for the two stay decoupled.
        pre_sem_loss = zero
        pre_sem_mse_loss = zero
        post_sem_loss = zero
        clip_gt = extra_result_dict.get("clip_gt")
        # When merger_as_sem_proj / merger_after_quant is on, the projector
        # outputs live in the frozen-merger's output space, so the teacher is
        # the explicit ``*_clip_target`` (= frozen_merger(clip_gt)) rather than
        # raw clip_gt. Fall back to clip_gt otherwise.
        pre_target = extra_result_dict.get("pre_clip_target", clip_gt)
        post_target = extra_result_dict.get("post_clip_target", clip_gt)
        if self.pre_quant_sem_weight > 0 and "pre_clip_pred" in extra_result_dict and pre_target is not None:
            pre_sem_loss = _cosine_distill_loss(extra_result_dict["pre_clip_pred"], pre_target)
        if self.pre_quant_sem_mse_weight > 0 and "pre_clip_pred" in extra_result_dict and pre_target is not None:
            pre_sem_mse_loss = F.mse_loss(extra_result_dict["pre_clip_pred"], pre_target)
        if self.post_quant_sem_weight > 0 and "post_clip_pred" in extra_result_dict and post_target is not None:
            post_sem_loss = _cosine_distill_loss(extra_result_dict["post_clip_pred"], post_target)
        weighted_pre_sem = (self.pre_quant_sem_weight * pre_sem_loss
                            + self.pre_quant_sem_mse_weight * pre_sem_mse_loss)
        weighted_post_sem = self.post_quant_sem_weight * post_sem_loss
        semantic_total = weighted_pre_sem + weighted_post_sem

        # ---- Pixel-reconstruction-side total --------------------------------
        pixel_recon_total = (
            reconstruction_loss
            + self.perceptual_weight * perceptual_loss
            + d_weight * discriminator_factor * generator_loss
            + self.clip_loss_weight * clip_loss
            + self.gram_loss_weight * gram_loss
            + quantizer_loss
        )

        # ---- Auxiliary feature-distillation MSE (stage-2) -------------------
        # Reconstruct the pre-quant feature ``z`` via the FeatureDecoder and pull
        # it back with MSE. Replaces post_quant_sem as the semantic supervision
        # when warm-starting from a feature_recon stage-1.
        feature_distill_loss = zero
        feature_distill_cos_loss = zero
        if self.aux_feature_distill and "feature_recon_pred" in extra_result_dict:
            feature_distill_loss = F.mse_loss(
                extra_result_dict["feature_recon_pred"],
                extra_result_dict["feature_target"])
            if self.feature_distill_cos_weight > 0:
                feature_distill_cos_loss = _cosine_distill_loss(
                    extra_result_dict["feature_recon_pred"],
                    extra_result_dict["feature_target"])

        # ---- ResMod-preserve cosine loss (stage-A / stage-C) ----------------
        # cos(post-ResMod z, pre-ResMod frozen-merger anchor). pred side keeps
        # the graph (encoder_z_graph) so gradients reach ResMod; target side is
        # the detached feature_target (frozen-merger output).
        resmod_preserve_loss = zero
        resmod_preserve_mse_loss = zero
        if ((self.resmod_preserve_weight > 0 or self.resmod_preserve_mse_weight > 0)
                and "encoder_z_graph" in extra_result_dict
                and ("resmod_preserve_anchor" in extra_result_dict
                     or "feature_target" in extra_result_dict)):
            _z_graph = extra_result_dict["encoder_z_graph"]
            # Prefer the dedicated pre-ResMod anchor; fall back to feature_target
            # (stage-A continuous mode, where feature_target IS the pre-ResMod
            # merge_target so the two coincide).
            _z_anchor = extra_result_dict.get(
                "resmod_preserve_anchor", extra_result_dict.get("feature_target"))
            if self.resmod_preserve_weight > 0:
                resmod_preserve_loss = _cosine_distill_loss(_z_graph, _z_anchor)
            if self.resmod_preserve_mse_weight > 0:
                resmod_preserve_mse_loss = F.mse_loss(_z_graph, _z_anchor)

        # Backward target: pixel reconstruction + semantic distillation +
        # (optional) auxiliary feature distillation + resmod-preserve cosine/MSE.
        total_loss = (pixel_recon_total + semantic_total
                      + self.feature_distill_weight * feature_distill_loss
                      + self.feature_distill_cos_weight * feature_distill_cos_loss
                      + self.resmod_preserve_weight * resmod_preserve_loss
                      + self.resmod_preserve_mse_weight * resmod_preserve_mse_loss)

        loss_dict = dict(
            total_loss=total_loss.clone().detach(),
            pixel_recon_total=pixel_recon_total.detach(),
            reconstruction_loss=reconstruction_loss.detach(),
            # Per-component recon (unweighted) for diagnostic logging.
            recon_l1=recon_l1.detach() if torch.is_tensor(recon_l1) else recon_l1,
            recon_l2=recon_l2.detach() if torch.is_tensor(recon_l2) else recon_l2,
            perceptual_loss=(self.perceptual_weight * perceptual_loss).detach(),
            weighted_gan_loss=(d_weight * discriminator_factor * generator_loss).detach(),
            discriminator_factor=discriminator_factor,
            clip_loss=(self.clip_loss_weight * clip_loss).detach(),
            gram_loss=(self.gram_loss_weight * gram_loss).detach(),
            d_weight=d_weight,
            gan_loss=generator_loss.detach(),
            quantizer_loss=quantizer_loss.detach() if torch.is_tensor(quantizer_loss) else quantizer_loss,
            commitment_loss=commitment_loss.detach() if torch.is_tensor(commitment_loss) else commitment_loss,
            codebook_loss=codebook_loss.detach() if torch.is_tensor(codebook_loss) else codebook_loss,
            vq_loss=vq_loss.detach() if torch.is_tensor(vq_loss) else vq_loss,
            entropy_loss=entropy_loss.detach() if torch.is_tensor(entropy_loss) else entropy_loss,
            # Semantic distillation block (separate from pixel-recon ledger).
            semantic_total=semantic_total.detach(),
            pre_sem_loss=pre_sem_loss.detach(),
            pre_sem_mse_loss=pre_sem_mse_loss.detach() if torch.is_tensor(pre_sem_mse_loss) else pre_sem_mse_loss,
            post_sem_loss=post_sem_loss.detach(),
            weighted_pre_sem_loss=weighted_pre_sem.detach(),
            weighted_post_sem_loss=weighted_post_sem.detach(),
            # Auxiliary feature-distillation MSE (stage-2; 0 when disabled).
            feature_distill_loss=feature_distill_loss.detach() if torch.is_tensor(feature_distill_loss) else feature_distill_loss,
            feature_distill_cos_loss=feature_distill_cos_loss.detach() if torch.is_tensor(feature_distill_cos_loss) else feature_distill_cos_loss,
            # ResMod-preserve cosine (stage-A / stage-C; 0 when disabled).
            resmod_preserve_loss=resmod_preserve_loss.detach() if torch.is_tensor(resmod_preserve_loss) else resmod_preserve_loss,
            resmod_preserve_mse_loss=resmod_preserve_mse_loss.detach() if torch.is_tensor(resmod_preserve_mse_loss) else resmod_preserve_mse_loss,
        )

        # Forward all quantizer diagnostics (per-group / per-depth codebook
        # usage, std and norm, plus per-group / per-depth commitment & vq
        # loss) into loss_dict so wandb / tensorboard see them.
        # Values from RVQ._update_and_get_usage may be plain Python floats
        # (usage ratio) or 0-d tensors (std / norm); cast to tensor on the
        # right device so accelerator.gather works uniformly.
        _diag_prefixes = (
            "codebook_usage/", "codebook_std/", "codebook_norm/",
            "commit_per_depth/", "commit_per_group/", "vq_per_group/",
        )
        for k, v in extra_result_dict.items():
            if not isinstance(k, str) or not k.startswith(_diag_prefixes):
                continue
            if not torch.is_tensor(v):
                v = torch.as_tensor(float(v), device=ref_device)
            loss_dict[k] = v.detach()

        return total_loss, loss_dict

    @torch._dynamo.disable
    def _compute_generator_loss(self, reconstructions, global_step):
        """Compute generator adversarial loss."""
        # Compute discriminator loss.
        generator_loss = torch.zeros((), device=reconstructions.device)
        discriminator_factor = self.discriminator_factor if self.should_discriminator_be_trained(global_step) else 0
        d_weight = 1.0
        d_weight *= self.discriminator_weight
        if discriminator_factor > 0.0 and self.discriminator_weight > 0.0 and self.discriminator is not None:
            # Disable discriminator gradients during the generator step so
            # backward() does not populate disc.params.grad — those would
            # corrupt the next discriminator update. The companion flip-on
            # happens inside _forward_discriminator.
            # NOTE: this MUST stay here even though it causes per-iter DDP
            # reducer churn — alternatives that move the flip to __init__
            # break create_optimizer (which selects params by requires_grad).
            for param in self.discriminator.parameters():
                param.requires_grad = False
            logits_fake = self.discriminator(reconstructions)
            generator_loss = -torch.mean(logits_fake)
        return generator_loss, d_weight, discriminator_factor

    @torch._dynamo.disable
    def _forward_discriminator(self,
                               inputs: SampleBatch,
                               reconstructions: SampleBatch,
                               global_step: int,
                               ) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        """Discrminator training step.

        Accepts list[(3,H_i,W_i)] anyres batches in addition to dense
        ``(B, 3, H, W)``; both are resized to ``self.disc_resize_target``
        before being fed to the discriminator (it was trained on a fixed
        resolution).
        """
        # Skip discriminator training if using gram loss
        if self.discriminator is None:
            ref_device = (inputs[0] if _is_list_batch(inputs) else inputs).device
            discriminator_loss = torch.zeros((), device=ref_device)
            loss_dict = dict(
                discriminator_loss=discriminator_loss,
                logits_real=torch.zeros((), device=ref_device),
                logits_fake=torch.zeros((), device=ref_device),
            )
            return discriminator_loss, loss_dict

        # Coerce to dense fixed-res batch.
        inputs_dense = _resize_batch_for_perceptual(
            inputs, target=self.disc_resize_target)
        recon_dense = _resize_batch_for_perceptual(
            reconstructions, target=self.disc_resize_target)

        discriminator_factor = self.discriminator_factor if self.should_discriminator_be_trained(global_step) else 0
        loss_dict = {}
        # Turn the gradients on.
        for param in self.discriminator.parameters():
            param.requires_grad = True

        real_images = inputs_dense.detach().requires_grad_(True)
        # quantize reconstructions
        recon_dense = recon_dense.detach()
        recon_dense = recon_dense.clamp(-1.0, 1.0)
        recon_dense = torch.round((recon_dense + 1.0) * 127.5) / 127.5 - 1.0

        logits_real = self.discriminator(real_images)
        logits_fake = self.discriminator(recon_dense.detach())
        discriminator_loss = discriminator_factor * hinge_d_loss(logits_real=logits_real, logits_fake=logits_fake)

        loss_dict = dict(
            discriminator_loss=discriminator_loss.detach(),
            logits_real=logits_real.detach().mean(),
            logits_fake=logits_fake.detach().mean(),
        )
        return discriminator_loss, loss_dict