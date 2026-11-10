# Copyright (c) 2026 LongCat dNaViT Team.
# SPDX-License-Identifier: Apache-2.0

"""In-training PSNR monitor for the RVQ tokenizer (fixed 256x256 reconstruction).

Every ``eval_every`` steps this decodes a small fixed set of ImageNet images at
256x256 and reports mean reconstruction PSNR. It reuses the LIVE
``model.encode`` / ``model.decode`` (same path as ``reconstruct_images_anyres``),
runs entirely under ``no_grad`` on the main process with the model in ``eval()``
(which disables the quantizer's EMA update / dead-code restart) and EMA weights,
and is wrapped in try/except so a failure can never disturb training.

Data source: reuses ``linear_probe.imagenet_path`` (see the ``psnr_eval`` config
block, which falls back to the ``linear_probe`` block). All images are
center-cropped + resized to a fixed 256x256 so PSNR is computed on same-size
tensors and stays comparable across steps.
"""
import os
import random
import time

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor, normalize, center_crop, resize

# Module-level cache so the image subset is built once and reused across calls.
_CACHED_SAMPLES = None   # list[(image_tensor (3,S,S) in [-1,1], grid_thw (3,))]


def _cfg(config):
    """Resolve the psnr_eval settings, falling back to the linear_probe block."""
    pe = config.get("psnr_eval", {}) or {}
    lp = config.get("linear_probe", {}) or {}
    return {
        "enable": bool(pe.get("enable", False)),
        "imagenet_path": pe.get("imagenet_path", lp.get("imagenet_path")),
        "num_images": int(pe.get("num_images", 128)),
        "img_size": int(pe.get("img_size", 256)),
        "batch_size": int(pe.get("batch_size", 32)),
        "seed": int(pe.get("seed", lp.get("seed", 42))),
    }


def _build_subset(cfg, settings, logger):
    """Sample ``num_images`` ImageNet images and pre-load 256x256 tensors."""
    global _CACHED_SAMPLES
    if _CACHED_SAMPLES is not None:
        return _CACHED_SAMPLES

    root = settings["imagenet_path"]
    num_images = settings["num_images"]
    img_size = settings["img_size"]
    seed = settings["seed"]
    patch = int(cfg.model.vq_model.get("vit_dec_patch_size", 16))
    mean = list(cfg.dataset.preprocessing.get("normalize_mean", [0.5, 0.5, 0.5]))
    std = list(cfg.dataset.preprocessing.get("normalize_std", [0.5, 0.5, 0.5]))

    random.seed(seed)
    all_cls = sorted(c for c in os.listdir(root) if os.path.isdir(os.path.join(root, c)))
    h_p = w_p = img_size // patch
    grid = torch.tensor([1, h_p, w_p], dtype=torch.long)
    exts = {".jpeg", ".jpg", ".png"}

    # Round-robin one image per class until we have num_images.
    per_class = {}
    for wnid in all_cls:
        d = os.path.join(root, wnid)
        imgs = sorted(f for f in os.listdir(d) if os.path.splitext(f)[1].lower() in exts)
        if imgs:
            per_class[wnid] = random.sample(imgs, len(imgs))

    samples = []
    random.shuffle(per_class_keys := list(per_class.keys()))
    k = 0
    # Round-robin over classes that still have images; stop once the pool is
    # exhausted (a small subset directory must not loop forever).
    while len(samples) < num_images:
        avail = [c for c in per_class_keys if per_class[c]]
        if not avail:
            break
        wnid = avail[k % len(avail)]
        k += 1
        f = per_class[wnid].pop()
        try:
            img = Image.open(os.path.join(root, wnid, f)).convert("RGB")
        except Exception:
            continue
        w, h = img.size
        s = min(w, h)
        img = center_crop(img, [s, s])
        img = resize(img, [img_size, img_size])
        t = normalize(to_tensor(img), mean, std)  # (3,S,S) in [-1,1]
        samples.append((t, grid.clone()))

    _CACHED_SAMPLES = samples
    if logger is not None:
        logger.info(f"[psnr_eval] subset: {len(samples)} images @ {img_size}x{img_size}")
    return _CACHED_SAMPLES


@torch.no_grad()
def _psnr_batch(orig01, rec01):
    """Per-image PSNR in dB for (B,3,H,W) tensors in [0,1]; returns a (B,) tensor.

    Matches eval_reconstruction's uint8 rounding so values are comparable.
    """
    rec01 = torch.round(torch.clamp(rec01, 0.0, 1.0) * 255.0) / 255.0
    orig01 = torch.clamp(orig01, 0.0, 1.0)
    mse = (orig01 - rec01).pow(2).flatten(1).mean(dim=1).clamp_min(1e-10)
    return 10.0 * torch.log10(1.0 / mse)


def run_psnr_eval(config, model, ema_model, accelerator, logger, global_step):
    """Main entry: EMA-weight fixed-256 reconstruction PSNR, main process only.

    Wrapped in try/except so any failure degrades to a warning instead of
    crashing the training loop.
    """
    if not accelerator.is_main_process:
        return
    settings = _cfg(config)
    if not settings["enable"] or not settings["imagenet_path"]:
        return

    unwrapped = accelerator.unwrap_model(model)
    use_ema = bool(config.training.get("use_ema", False)) and ema_model is not None
    was_training = unwrapped.training
    dtype = torch.bfloat16 if accelerator.mixed_precision == "bf16" else (
        torch.float16 if accelerator.mixed_precision == "fp16" else torch.float32)
    try:
        t0 = time.time()
        if use_ema:
            ema_model.store(model.parameters())
            ema_model.copy_to(model.parameters())
        unwrapped.eval()  # disables quantizer EMA update / dead-code restart

        samples = _build_subset(config, settings, logger)
        if not samples:
            logger.warning("[psnr_eval] no images found; skipping.")
            return
        device = accelerator.device
        bs = settings["batch_size"]

        psnr_vals = []
        for start in range(0, len(samples), bs):
            chunk = samples[start:start + bs]
            imgs = [s[0].to(device, dtype=dtype) for s in chunk]
            grid = torch.stack([s[1] for s in chunk]).to(device)
            with torch.autocast(device_type="cuda", dtype=dtype,
                                enabled=accelerator.mixed_precision != "no"):
                if getattr(unwrapped, "continuous", False):
                    z_q, _ = unwrapped.encode_continuous(imgs, image_grid_thw=grid)
                else:
                    z_q, _ = unwrapped.encode(imgs, image_grid_thw=grid)
                recon_list = unwrapped.decode(z_q, image_grid_thw=grid)[0]
            for i in range(len(chunk)):
                rec = recon_list[i]
                if rec.dim() == 4:
                    rec = rec.squeeze(0)
                orig01 = (chunk[i][0].to(device).float() + 1.) / 2.
                rec01 = (rec.float() + 1.) / 2.
                psnr_vals.append(_psnr_batch(orig01.unsqueeze(0), rec01.unsqueeze(0)))

        mean_psnr = float(torch.cat(psnr_vals).mean().item())
        log = {"psnr_eval/psnr": mean_psnr}
        accelerator.log(log, step=global_step)
        logger.info(f"[psnr_eval] PSNR={mean_psnr:.3f} dB over {len(samples)} imgs "
                    f"in {time.time() - t0:.1f}s @ step {global_step}")
    except Exception as e:  # never let the monitor break training
        logger.warning(f"[psnr_eval] skipped due to error: {e!r}")
    finally:
        if use_ema:
            ema_model.restore(model.parameters())
        if was_training:
            unwrapped.train()
