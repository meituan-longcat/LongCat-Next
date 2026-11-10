# Copyright (c) 2026 LongCat dNaViT Team.
# SPDX-License-Identifier: Apache-2.0

"""This file contains perceptual loss module using LPIPS or ConvNeXt-S."""
import os
from typing import List, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from torchvision import models
from .lpips import LPIPS
from utils.paths import pretrained_path

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

 
class PerceptualLoss(torch.nn.Module):
    def __init__(self, model_name: str = "lpips"):
        """Initializes the PerceptualLoss class.

        Args:
            model_name: A string, the name of the perceptual loss model to use.

        Raise:
            ValueError: If the model_name does not contain "lpips" or "convnext_s".
        """
        super().__init__()
        if ("lpips" not in model_name) and (
            "convnext_s" not in model_name):
            raise ValueError(f"Unsupported Perceptual Loss model name {model_name}")
        self.lpips = None
        self.convnext = None
        self.loss_weight_lpips = None
        self.loss_weight_convnext = None

        # Parsing the model name. We support name formatted in
        # "lpips-convnext_s-{float_number}-{float_number}", where the 
        # {float_number} refers to the loss weight for each component.
        # E.g., lpips-convnext_s-1.0-2.0 refers to compute the perceptual loss
        # using both the convnext_s and lpips, and average the final loss with
        # (1.0 * loss(lpips) + 2.0 * loss(convnext_s)) / (1.0 + 2.0).
        if "lpips" in model_name:
            self.lpips = LPIPS().eval()

        if "convnext_s" in model_name:
            # ImageNet-1K ConvNeXt-Small backbone (torchvision weights).
            ckpt_path = pretrained_path("convnext_small-0c510722.pth")
            self.convnext = models.convnext_small(weights=None).eval()
            ckpt = torch.load(ckpt_path, map_location="cpu")
            self.convnext.load_state_dict(ckpt)
            del ckpt

        if "lpips" in model_name and "convnext_s" in model_name:
            loss_config = model_name.split('-')[-2:]
            self.loss_weight_lpips, self.loss_weight_convnext = float(loss_config[0]), float(loss_config[1])
            print(f"self.loss_weight_lpips, self.loss_weight_convnext: {self.loss_weight_lpips}, {self.loss_weight_convnext}")
 
        self.register_buffer("imagenet_mean", torch.Tensor(_IMAGENET_MEAN)[None, :, None, None])
        self.register_buffer("imagenet_std", torch.Tensor(_IMAGENET_STD)[None, :, None, None])

        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, input: torch.Tensor, target: torch.Tensor):
        """Computes the perceptual loss.

        Args:
            input: A tensor of shape (B, C, H, W), the input image. Normalized to [0, 1].
            target: A tensor of shape (B, C, H, W), the target image. Normalized to [0, 1].

        Returns:
            A scalar tensor, the perceptual loss.
        """
        # Always in eval mode.
        self.eval()
        loss = 0.
        num_losses = 0.
        # Computes LPIPS loss, if available.
        if self.lpips is not None:
            lpips_loss = self.lpips(input, target)
            if self.loss_weight_lpips is None:
                loss += lpips_loss
                num_losses += 1
            else:
                num_losses += self.loss_weight_lpips
                loss += self.loss_weight_lpips * lpips_loss

        if self.convnext is not None:
            # Computes ConvNeXt-s loss, if available.
            pred_input = torch.nn.functional.interpolate(input, size=224, mode="bicubic", align_corners=False, antialias=True)
            pred_target = torch.nn.functional.interpolate(target, size=224, mode="bicubic", align_corners=False, antialias=True)

            # shift from [-1, 1] to [0, 1]
            pred_input = (pred_input + 1.) / 2.
            pred_target = (pred_target + 1.) / 2.

            pred_input = self.convnext((pred_input - self.imagenet_mean) / self.imagenet_std)
            pred_target = self.convnext((pred_target - self.imagenet_mean) / self.imagenet_std)
            convnext_loss = torch.nn.functional.mse_loss(
                pred_input,
                pred_target,
                reduction="mean")
                
            if self.loss_weight_convnext is None:
                num_losses += 1
                loss += convnext_loss
            else:
                num_losses += self.loss_weight_convnext
                loss += self.loss_weight_convnext * convnext_loss

        # weighted avg.
        loss = loss / num_losses
        return loss

    # =====================================================================
    # InternVL-style any-res tiled perceptual loss (OOM-safe, batched).
    #
    # Both branches (LPIPS=VGG, ConvNeXt-S) run on fixed 224x224 tiles:
    #   per image -> 1 global thumbnail + N non-overlapping local 224 tiles
    #               (grid chosen by aspect ratio, InternVL-style).
    # All tiles across the batch are concatenated and pushed through each
    # frozen backbone in chunks of ``max_batch_size`` (with grad checkpointing
    # on the grad-carrying recon branch) so memory stays bounded regardless of
    # how many tiles a high-res batch produces. Losses are averaged per image
    # first, then across images, so images with many tiles don't dominate.
    #
    # Range contract: inputs are in [-1, 1] (the loss-path convention) — fed to
    # LPIPS directly (it expects [-1, 1]); ConvNeXt shifts to [0, 1] +
    # ImageNet-normalizes internally, exactly like the dense ``forward`` above.
    # =====================================================================

    _MAX_GRID_DIM = 3  # max rows or cols when tiling (sole cap; up to _MAX_GRID_DIM^2 tiles)

    @staticmethod
    def _find_closest_grid(aspect_ratio, height, width, tile_size):
        """Pick (rows, cols), each in [1, _MAX_GRID_DIM], best matching the image
        aspect ratio (InternVL heuristic). Max tiles = _MAX_GRID_DIM**2."""
        best_diff = float('inf')
        best_grid = (1, 1)
        area = height * width
        candidates = [
            (r, c)
            for r in range(1, PerceptualLoss._MAX_GRID_DIM + 1)
            for c in range(1, PerceptualLoss._MAX_GRID_DIM + 1)
        ]
        for rows, cols in sorted(candidates, key=lambda x: x[0] * x[1]):
            diff = abs(aspect_ratio - cols / rows)
            if diff < best_diff:
                best_diff = diff
                best_grid = (rows, cols)
            elif diff == best_diff:
                if area > 0.5 * (tile_size * tile_size * rows * cols):
                    best_grid = (rows, cols)
        return best_grid

    @staticmethod
    def _generate_tiles(image, tile_size):
        """One image [*,3,H,W] -> (global [1,3,T,T], list of local [1,3,T,T]).

        Local tiles are empty for a (1,1) grid (small / near-square images).
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
        _, _, h, w = image.shape
        global_tile = F.interpolate(
            image, (tile_size, tile_size), mode='bicubic', align_corners=False)
        rows, cols = PerceptualLoss._find_closest_grid(
            w / max(h, 1), h, w, tile_size)
        if rows == 1 and cols == 1:
            return global_tile, []
        resized = F.interpolate(
            image, (rows * tile_size, cols * tile_size),
            mode='bicubic', align_corners=False)
        local_tiles = []
        for r in range(rows):
            for c in range(cols):
                top, left = r * tile_size, c * tile_size
                local_tiles.append(
                    resized[:, :, top:top + tile_size, left:left + tile_size])
        return global_tile, local_tiles

    def _prepare_tiles(self, recon, target, tile_size):
        """Build batched global + local tiles for recon (grad) and target
        (detached). Accepts a dense (B,3,H,W) tensor or a list of [*,3,H_i,W_i].

        Returns global_in[B,3,T,T], global_tgt(det), local_in[N,3,T,T],
        local_tgt(det), local_counts (len B).
        """
        if isinstance(recon, torch.Tensor):
            recon = [recon[i:i + 1] for i in range(recon.shape[0])]
        if isinstance(target, torch.Tensor):
            target = [target[i:i + 1] for i in range(target.shape[0])]

        g_in, g_tg, l_in, l_tg, counts = [], [], [], [], []
        for r_img, t_img in zip(recon, target):
            rg, rl = self._generate_tiles(r_img, tile_size)
            tg, tl = self._generate_tiles(t_img, tile_size)
            g_in.append(rg)
            g_tg.append(tg)
            l_in.extend(rl)
            l_tg.extend(tl)
            counts.append(len(rl))

        global_in = torch.cat(g_in, dim=0)
        global_tgt = torch.cat(g_tg, dim=0).detach()
        if l_in:
            local_in = torch.cat(l_in, dim=0)
            local_tgt = torch.cat(l_tg, dim=0).detach()
        else:
            local_in = global_in.new_zeros((0, 3, tile_size, tile_size))
            local_tgt = global_tgt.new_zeros((0, 3, tile_size, tile_size))
        return global_in, global_tgt, local_in, local_tgt, counts

    def _lpips_per_tile(self, inp_m11, tgt_m11):
        """Per-tile LPIPS (VGG). Inputs [-1, 1]; returns [N]."""
        return self.lpips(inp_m11, tgt_m11).flatten()

    def _convnext_per_tile(self, inp_m11, tgt_m11):
        """Per-tile ConvNeXt logits-MSE. Inputs [-1, 1] at tile size; returns [N]."""
        a = (inp_m11 + 1.) / 2.
        b = (tgt_m11 + 1.) / 2.
        fa = self.convnext((a - self.imagenet_mean) / self.imagenet_std)
        fb = self.convnext((b - self.imagenet_mean) / self.imagenet_std)
        return F.mse_loss(fa, fb, reduction='none').mean(dim=tuple(range(1, fa.dim())))

    def _branch_per_tile(self, branch_fn, inp, tgt, max_batch_size):
        """Run ``branch_fn`` over all tiles in chunks of ``max_batch_size``
        (grad checkpoint on grad-carrying chunks). Returns [N] per-tile losses."""
        n = inp.shape[0]
        if n == 0:
            return inp.new_zeros((0,))
        out = inp.new_zeros(n)
        for s in range(0, n, max_batch_size):
            e = min(s + max_batch_size, n)
            ic, tc = inp[s:e], tgt[s:e]
            if ic.requires_grad:
                out[s:e] = grad_checkpoint(branch_fn, ic, tc, use_reentrant=False)
            else:
                out[s:e] = branch_fn(ic, tc)
        return out

    @staticmethod
    def _mean_per_image(per_tile, counts):
        """Average local-tile losses within each image, then across images."""
        if per_tile.numel() == 0:
            return per_tile.new_zeros(())
        vals, off = [], 0
        for c in counts:
            if c > 0:
                vals.append(per_tile[off:off + c].mean())
            off += c
        if not vals:
            return per_tile.new_zeros(())
        return torch.stack(vals).mean()

    def forward_anyres(
        self,
        recon: Union[torch.Tensor, Sequence[torch.Tensor]],
        target: Union[torch.Tensor, Sequence[torch.Tensor]],
        tile_size: int = 224,
        max_batch_size: int = 32,
        global_weight: float = 1.0,
        local_weight: float = 1.0,
    ) -> torch.Tensor:
        """Any-res tiled perceptual loss (drop-in for the dense ``forward``).

        Args:
            recon:  grad-carrying reconstruction — dense (B,3,H,W) or list of
                    [*,3,H_i,W_i], range [-1, 1].
            target: original images, same layout; detached internally.
            tile_size / max_batch_size: tile resolution + OOM-batching chunk.
                Tile count per image is set solely by ``_MAX_GRID_DIM`` (up to
                ``_MAX_GRID_DIM**2`` local tiles).
            global_weight / local_weight: combine global thumbnail vs local tiles
                (per branch). local term is 0 when no image produced local tiles.

        Returns:
            Scalar perceptual loss = weighted avg of the LPIPS and ConvNeXt terms
            using the same ``loss_weight_lpips`` / ``loss_weight_convnext`` as the
            dense path (or unweighted mean when only one branch is enabled).
        """
        self.eval()
        g_in, g_tg, l_in, l_tg, counts = self._prepare_tiles(
            recon, target, tile_size)
        # fp32 for stable LPIPS / MSE math.
        g_in, g_tg = g_in.float(), g_tg.float()
        l_in, l_tg = l_in.float(), l_tg.float()
        has_local = l_in.shape[0] > 0

        def branch_term(branch_fn):
            gv = self._branch_per_tile(branch_fn, g_in, g_tg, max_batch_size).mean()
            term = global_weight * gv
            if has_local and local_weight != 0.0:
                lv = self._mean_per_image(
                    self._branch_per_tile(branch_fn, l_in, l_tg, max_batch_size), counts)
                term = term + local_weight * lv
            return term

        loss = 0.
        num = 0.
        if self.lpips is not None:
            lp = branch_term(self._lpips_per_tile)
            w = 1.0 if self.loss_weight_lpips is None else self.loss_weight_lpips
            loss = loss + w * lp
            num += w
        if self.convnext is not None:
            cn = branch_term(self._convnext_per_tile)
            w = 1.0 if self.loss_weight_convnext is None else self.loss_weight_convnext
            loss = loss + w * cn
            num += w
        return loss / num