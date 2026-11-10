# Copyright (c) 2026 LongCat dNaViT Team.
# SPDX-License-Identifier: Apache-2.0

"""In-training reconstruction-metrics monitor for the (RV)Q tokenizer.

For the t2i / anyres training path ``create_dataloader`` builds NO eval loader
(``eval_dataloader is None``), so the standard ``eval_reconstruction`` block in
``train_one_epoch`` never runs and PSNR/SSIM/LPIPS/rFID are never measured.

This module fills that gap: every ``eval_every`` steps it encodes+decodes a
fixed-resolution ImageNet set and feeds the originals/reconstructions through the
already-built ``ReconstructionEvaluator`` to report PSNR / SSIM / LPIPS / rFID / MSE / MAE.

Two data sources are supported:
  * ``recon_eval.val_path`` set  -> stream the FULL ImageNet val set from that
    directory (e.g. the pre-resized 256x256 ``val_cache_s256``, all 50000 images),
    loaded lazily per batch so nothing is cached in RAM. ``num_images`` (if set)
    caps it; unset/null means all images.
  * otherwise                    -> fall back to the small fixed-res subset built
    by ``utils.linear_probe`` (capped at ``num_images``, default 512).

Like the linear probe it runs on the main process only, on EMA weights, under
``no_grad`` with the model in ``eval()`` (which disables the quantizer's EMA
update / dead-code restart so the codebook is untouched), and is wrapped in
try/except so a failure here can never take training down.
"""
import os
import time

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor, normalize, center_crop, resize

from utils.linear_probe import _build_subset

# Module-level cache: the val image-path list (paths only, NOT tensors) so the
# 50000-image directory walk runs once and every later eval reuses it.
_VAL_PATHS = None


def _build_val_paths(cfg, logger):
    """List every image under ``recon_eval.val_path`` (recursively by class dir)."""
    global _VAL_PATHS
    if _VAL_PATHS is not None:
        return _VAL_PATHS
    root = cfg.get("recon_eval", {}).get("val_path")
    exts = {".jpeg", ".jpg", ".png"}
    paths = []
    for cls in sorted(os.listdir(root)):
        d = os.path.join(root, cls)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if os.path.splitext(f)[1].lower() in exts:
                paths.append(os.path.join(d, f))
    _VAL_PATHS = paths
    if logger is not None:
        logger.info(f"[recon_eval] val set: {len(paths)} images from {root}")
    return _VAL_PATHS


def _load_square(path, img_size, mean, std):
    """Center-crop to square + resize to img_size + normalize to [-1,1]."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if w != h:
        s = min(w, h)
        img = center_crop(img, [s, s])
    if img.size[0] != img_size:
        img = resize(img, [img_size, img_size])
    return normalize(to_tensor(img), mean, std)  # (3,S,S)


@torch.no_grad()
def run_recon_eval(config, model, ema_model, accelerator, evaluator, logger, global_step):
    """Main entry: EMA-weight reconstruction metrics, main process only, training-safe."""
    if not accelerator.is_main_process:
        return
    if not config.get("recon_eval", {}).get("enable", False):
        return
    if evaluator is None:
        logger.warning("[recon_eval] evaluator is None (feature_recon_only?); skip")
        return

    re_cfg = config.get("recon_eval", {})
    batch_size = int(re_cfg.get("batch_size", 32))
    use_val = bool(re_cfg.get("val_path"))
    img_size = int(re_cfg.get("img_size", 256))
    patch = int(config.model.vq_model.get("vit_dec_patch_size", 16))
    mean = list(config.dataset.preprocessing.get("normalize_mean", [0.5, 0.5, 0.5]))
    std = list(config.dataset.preprocessing.get("normalize_std", [0.5, 0.5, 0.5]))

    unwrapped = accelerator.unwrap_model(model)
    use_ema = bool(config.training.get("use_ema", False)) and ema_model is not None
    was_training = unwrapped.training
    is_continuous = bool(getattr(unwrapped, "continuous", False))
    device = accelerator.device
    dtype = torch.bfloat16 if accelerator.mixed_precision == "bf16" else (
        torch.float16 if accelerator.mixed_precision == "fp16" else torch.float32)
    try:
        t0 = time.time()
        if use_ema:
            ema_model.store(model.parameters())
            ema_model.copy_to(model.parameters())
        unwrapped.eval()

        # Build the (path or tensor) source. val_path => full ImageNet val,
        # loaded lazily; otherwise the small cached linear-probe subset.
        if use_val:
            paths = _build_val_paths(config, logger)
            cap = re_cfg.get("num_images", None)
            if cap is not None:
                paths = paths[:int(cap)]
            n_total = len(paths)
            h_p = w_p = img_size // patch
            grid_one = torch.tensor([1, h_p, w_p], dtype=torch.long)

            def _batch(start):
                chunk = paths[start:start + batch_size]
                imgs = [_load_square(p, img_size, mean, std).to(device) for p in chunk]
                grids = torch.stack([grid_one.clone() for _ in chunk]).to(device)
                return imgs, grids, len(chunk)
        else:
            samples, (train_idx, test_idx) = _build_subset(config, logger)
            num_images = int(re_cfg.get("num_images", 512))
            order = (train_idx + test_idx)[:num_images]
            n_total = len(order)

            def _batch(start):
                chunk = order[start:start + batch_size]
                imgs = [samples[i][0].to(device) for i in chunk]
                grids = torch.stack([samples[i][1] for i in chunk]).to(device)
                return imgs, grids, len(chunk)

        if n_total == 0:
            logger.warning("[recon_eval] empty image set; skip")
            return

        evaluator.reset_metrics()
        for start in range(0, n_total, batch_size):
            imgs, grids, n = _batch(start)
            with torch.autocast(device_type="cuda", dtype=dtype,
                                enabled=accelerator.mixed_precision != "no"):
                if is_continuous:
                    z, _ = unwrapped.encode_continuous(imgs, image_grid_thw=grids)
                else:
                    z, _ = unwrapped.encode(imgs, image_grid_thw=grids)
                recon_list = unwrapped.decode(z, image_grid_thw=grids)[0]

            reals, fakes = [], []
            for j in range(n):
                rec = recon_list[j]
                if rec.dim() == 4:
                    rec = rec.squeeze(0)
                orig01 = ((imgs[j].float() + 1.) / 2.).clamp(0.0, 1.0)
                rec01 = ((rec.float() + 1.) / 2.).clamp(0.0, 1.0)
                # uint8-quantize the reconstruction to match eval_reconstruction.
                rec01 = torch.round(rec01 * 255.0) / 255.0
                reals.append(orig01)
                fakes.append(rec01)
            real_batch = torch.stack(reals).to(device)
            fake_batch = torch.stack(fakes).to(device)
            evaluator.update(real_batch, fake_batch, None)

        scores = evaluator.result()
        log = {f"recon_eval/{k}": (v.item() if torch.is_tensor(v) else float(v))
               for k, v in scores.items()}
        accelerator.log(log, step=global_step)
        msg = " ".join(f"{k.split('/')[-1]}={v:.4f}" for k, v in log.items())
        logger.info(f"[recon_eval] step {global_step} ({n_total} imgs, "
                    f"{time.time() - t0:.1f}s): {msg}")
    except Exception as e:  # never let the monitor break training
        logger.warning(f"[recon_eval] skipped due to error: {e!r}")
    finally:
        if use_ema:
            ema_model.restore(model.parameters())
        if was_training:
            unwrapped.train()
