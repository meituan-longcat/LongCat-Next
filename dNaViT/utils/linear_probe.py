# Copyright (c) 2026 LongCat dNaViT Team.
# SPDX-License-Identifier: Apache-2.0

"""In-training linear-probe monitor for the RVQ tokenizer.

Every ``eval_every`` steps this measures how linearly-separable the tokenizer's
features are on a small fixed-resolution ImageNet subset, comparing:
  * pre-quant  : the encoder feature ``z`` fed to the quantizer
                 (= result_dict["feature_target"], merger output, 4096-d)
  * post-quant : the cumulative RVQ codebook lookup
                 (= result_dict["quant_at_depth_with_grad"][-1], merged, 1152-d)

It reuses the LIVE ``model.encode`` (no re-implementation of the merger / in_proj
/ codebook math), runs entirely under ``no_grad`` on the main process with the
model in ``eval()`` (which disables the quantizer's EMA update / dead-code
restart), and is wrapped in try/except so a probe failure can never disturb
training.
"""
import os
import random
import time

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor, normalize, center_crop, resize

# Module-level caches so the ImageNet subset (and its normalized tensors) are
# built once and reused across every probe call.
_CACHED_SAMPLES = None   # list[(image_tensor (3,S,S), grid_thw (3,), label)]
_CACHED_SPLIT = None     # (train_idx, test_idx)


def _build_subset(cfg, logger):
    """Sample a fixed ImageNet subset and pre-load normalized square tensors."""
    global _CACHED_SAMPLES, _CACHED_SPLIT
    if _CACHED_SAMPLES is not None:
        return _CACHED_SAMPLES, _CACHED_SPLIT

    lp = cfg.get("linear_probe", {})
    root = lp.get("imagenet_path")
    n_class = int(lp.get("n_class", 200))
    per_train = int(lp.get("per_train", 20))
    per_test = int(lp.get("per_test", 10))
    img_size = int(lp.get("img_size", 256))
    seed = int(lp.get("seed", 42))
    patch = int(cfg.model.vq_model.get("vit_dec_patch_size", 16))
    mean = list(cfg.dataset.preprocessing.get("normalize_mean", [0.5, 0.5, 0.5]))
    std = list(cfg.dataset.preprocessing.get("normalize_std", [0.5, 0.5, 0.5]))

    random.seed(seed)
    all_cls = sorted(c for c in os.listdir(root) if os.path.isdir(os.path.join(root, c)))[:n_class]
    h_p = w_p = img_size // patch
    grid = torch.tensor([1, h_p, w_p], dtype=torch.long)

    samples, labels_train, labels_test = [], [], []
    train_idx, test_idx = [], []
    exts = {".jpeg", ".jpg", ".png"}
    for li, wnid in enumerate(all_cls):
        d = os.path.join(root, wnid)
        imgs = sorted(f for f in os.listdir(d) if os.path.splitext(f)[1].lower() in exts)
        if len(imgs) < per_train + per_test:
            continue
        sel = random.sample(imgs, per_train + per_test)
        for k, f in enumerate(sel):
            try:
                img = Image.open(os.path.join(d, f)).convert("RGB")
            except Exception:
                continue
            w, h = img.size
            s = min(w, h)
            img = center_crop(img, [s, s])
            img = resize(img, [img_size, img_size])
            t = normalize(to_tensor(img), mean, std)  # (3,S,S) in [-1,1]
            idx = len(samples)
            samples.append((t, grid.clone(), li))
            (train_idx if k < per_train else test_idx).append(idx)

    if not samples:
        if logger is not None:
            logger.warning(
                f"[linear_probe] no class under {root} has "
                f"{per_train + per_test} images; probe skipped.")
    _CACHED_SAMPLES = samples
    _CACHED_SPLIT = (train_idx, test_idx)
    if logger is not None:
        logger.info(f"[linear_probe] subset: classes={n_class}, "
                    f"train={len(train_idx)}, test={len(test_idx)}, S={img_size}")
    return _CACHED_SAMPLES, _CACHED_SPLIT


@torch.no_grad()
def _extract(unwrapped, samples, indices, device, batch_size=256):
    """Run model.encode over the subset; return mean-pooled features + labels.

    Returns a dict of per-image mean-pooled feature arrays keyed by tap name
    (``pre_quant`` = encoder feature_target, ``post_resmod`` = ResMod-output z fed
    to the quantizer, ``post_quant`` = cumulative RVQ lookup) plus the labels.

    The forward is wrapped in ``torch.autocast(bf16)`` to match the training /
    eval_reconstruction path: model params are fp32 masters and the Qwen ViT's
    FlashAttention only accepts fp16/bf16, so without autocast the encoder runs
    in fp32 and raises "FlashAttention only support fp16 and bf16 data type".

    The Qwen ViT uses varlen (cu_seqlens) attention, so each image attends only
    within itself — memory grows linearly in batch_size, not quadratically — and
    a large batch (default 256 imgs = 256x256 each => 64 super-tokens/img) fits
    comfortably under no_grad+bf16.
    """
    # Continuous-AE (stage-A) runs encode_continuous (no codebook): it exposes
    # feature_target (pre-ResMod anchor) + encoder_z (post-ResMod z) but NO
    # quant_at_depth_with_grad, so the post-quant tap is undefined and skipped.
    is_continuous = bool(getattr(unwrapped, "continuous", False))
    pre_all, postres_all, post_all, y_all = [], [], [], []
    for start in range(0, len(indices), batch_size):
        chunk = indices[start:start + batch_size]
        imgs = [samples[i][0].to(device, dtype=torch.bfloat16) for i in chunk]
        grids = torch.stack([samples[i][1] for i in chunk]).to(device)
        labels = [samples[i][2] for i in chunk]

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if is_continuous:
                _, rd = unwrapped.encode_continuous(imgs, image_grid_thw=grids)
            else:
                _, rd = unwrapped.encode(imgs, image_grid_thw=grids)
        # feature_target is only populated when feature_recon_only or
        # aux_feature_distill is on (see tokenizer.py). Fall back to encoder_z
        # (always set, = post-ResMod z fed to in_proj) so the probe still runs
        # in pixel-only ablations where aux_feature_distill=false. In that
        # case pre_quant and post_resmod measure the same tensor — the dim
        # label printed in the log makes this obvious.
        pre = (rd["feature_target"] if "feature_target" in rd else rd["encoder_z"]).float()
        postres = rd["encoder_z"].float()                        # post-ResMod z fed to in_proj
        post = None if is_continuous else rd["quant_at_depth_with_grad"][-1].float()
        cu = rd["cu_seqlens"].tolist()
        for j in range(len(chunk)):
            st, en = int(cu[j]), int(cu[j + 1])
            pre_all.append(torch.nan_to_num(pre[st:en].mean(0)).cpu())
            postres_all.append(torch.nan_to_num(postres[st:en].mean(0)).cpu())
            if post is not None:
                post_all.append(torch.nan_to_num(post[st:en].mean(0)).cpu())
            y_all.append(labels[j])
    feats = {
        "pre_quant": torch.stack(pre_all).numpy(),
        "post_resmod": torch.stack(postres_all).numpy(),
    }
    if post_all:
        feats["post_quant"] = torch.stack(post_all).numpy()
    return feats, torch.tensor(y_all).numpy()


def _probe(tr, te, y_tr, y_te, name, logger, seed=42):
    import numpy as np
    from sklearn.linear_model import RidgeClassifier
    from sklearn.preprocessing import StandardScaler
    if not np.isfinite(tr).all() or not np.isfinite(te).all():
        logger.warning(f"[linear_probe] {name}: SKIP (NaN/Inf)")
        return None
    sc = StandardScaler()
    # RidgeClassifier(alpha=10.0): closed-form, fast, and no max_iter
    # convergence warnings — matches the offline linear_probe_lxt_all.py probe.
    clf = RidgeClassifier(alpha=10.0)
    clf.fit(sc.fit_transform(tr), y_tr)
    acc = float((clf.predict(sc.transform(te)) == y_te).mean())
    logger.info(f"[linear_probe] {name}: acc={acc:.4f}")
    return acc


def run_linear_probe(config, model, ema_model, accelerator, logger, global_step):
    """Main entry: EMA-weight linear probe, main process only, training-safe.

    Wrapped in try/except so any failure degrades to a warning instead of
    crashing the training loop.
    """
    if not accelerator.is_main_process:
        return
    if not config.get("linear_probe", {}).get("enable", False):
        return

    unwrapped = accelerator.unwrap_model(model)
    use_ema = bool(config.training.get("use_ema", False)) and ema_model is not None
    was_training = unwrapped.training
    try:
        t0 = time.time()
        # Evaluate on EMA weights (matches eval_reconstruction): stash live
        # params, copy EMA in, restore afterwards.
        if use_ema:
            ema_model.store(model.parameters())
            ema_model.copy_to(model.parameters())
        unwrapped.eval()  # disables quantizer EMA update / dead-code restart

        samples, (train_idx, test_idx) = _build_subset(config, logger)
        if not train_idx or not test_idx:
            logger.warning("[linear_probe] empty probe subset; skipping.")
            return
        device = accelerator.device
        bs = int(config.get("linear_probe", {}).get("batch_size", 256))
        tr, y_tr = _extract(unwrapped, samples, train_idx, device, batch_size=bs)
        te, y_te = _extract(unwrapped, samples, test_idx, device, batch_size=bs)

        # tap name -> wandb metric key, in a stable order. Display dim is taken
        # from the actual feature width (16x=1152, 32x=4096; post_quant absent in
        # continuous mode).
        taps = [
            ("pre_quant", "linear_probe/pre_quant_acc"),
            ("post_resmod", "linear_probe/post_resmod_acc"),
            ("post_quant", "linear_probe/post_quant_acc"),
        ]
        log = {}
        for name, key in taps:
            if name not in tr or name not in te:
                continue
            dim = f"[{tr[name].shape[1]}]"
            acc = _probe(tr[name], te[name], y_tr, y_te, f"{name} {dim}", logger)
            if acc is not None:
                log[key] = acc
        if log:
            accelerator.log(log, step=global_step)
        logger.info(f"[linear_probe] done in {time.time() - t0:.1f}s @ step {global_step}")
    except Exception as e:  # never let the probe break training
        logger.warning(f"[linear_probe] skipped due to error: {e!r}")
    finally:
        if use_ema:
            ema_model.restore(model.parameters())
        if was_training:
            unwrapped.train()
