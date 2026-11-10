# Copyright (c) 2026 LongCat dNaViT Team.
# SPDX-License-Identifier: Apache-2.0

"""ImageNet reconstruction eval (256/512/1024) for the released dNaViT HF repos.

Feeds every image at a fixed square (Resize shorter edge + CenterCrop, identical
to ``eval_tok.sh``) through the HuggingFace model, saves the reconstruction and
reports PSNR / SSIM / LPIPS / rFID against the same GT crop.

Refiner exports reconstruct at 2x; the saved PNG keeps that native size while
the metrics resize it back to the GT resolution.

Launch (8 GPUs)::

    python -m torch.distributed.run --standalone --nproc_per_node=8 \
        scripts/eval_hf_in1k.py --image-size 256 \
        --model-root /path/to/checkpoints --data-path /path/to/imagenet/val \
        --models dNaViT-G16,dNaViT-G64,dNaViT-G16R1 --fid

Each model is loaded from ``<model-root>/<model-name>``. The model root defaults
to ``pretrained/`` and outputs default to ``outputs/imagenet/``; both paths are
relative to the current working directory unless explicitly overridden.

Single GPU also works (no torchrun). Existing PNGs are skipped, so a restarted
job only redoes missing images.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

MODELS = {
    "dNaViT-G16": ("dNaViT-G16", "tokenizer"),
    "dNaViT-G64": ("dNaViT-G64", "tokenizer"),
    "dNaViT-G16R1": ("dNaViT-G16R1", "refiner"),
    "dNaViT-G16R52": ("dNaViT-G16R52", "refiner"),
}
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff"}


def list_images(root: str) -> List[str]:
    imgs = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if Path(fn).suffix.lower() in IMG_EXTS:
                imgs.append(os.path.join(dp, fn))
    imgs.sort()
    if not imgs:
        raise ValueError(f"No images found under {root}")
    return imgs


class FIDAccumulator:
    """Streaming InceptionV3 pool3 statistics (same as eval_recon.py)."""

    def __init__(self, device: torch.device):
        from pytorch_fid.inception import InceptionV3
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        self.model = InceptionV3([block_idx]).to(device).eval()
        self.device = device
        d = 2048
        self.n = torch.zeros(1, dtype=torch.float64, device=device)
        self.s_r = torch.zeros(d, dtype=torch.float64, device=device)
        self.c_r = torch.zeros(d, d, dtype=torch.float64, device=device)
        self.s_g = torch.zeros(d, dtype=torch.float64, device=device)
        self.c_g = torch.zeros(d, d, dtype=torch.float64, device=device)

    @torch.no_grad()
    def _feat(self, imgs01: List[torch.Tensor]) -> torch.Tensor:
        x = torch.stack([
            F.interpolate(im.unsqueeze(0), size=(299, 299), mode="bilinear",
                          align_corners=False).squeeze(0).clamp(0, 1)
            for im in imgs01
        ]).to(self.device)
        f = self.model(x.float())[0].squeeze(-1).squeeze(-1)
        return f.double()

    @torch.no_grad()
    def update(self, rec01: torch.Tensor, gt01: torch.Tensor):
        fr = self._feat([rec01])
        fg = self._feat([gt01])
        self.n += fr.shape[0]
        self.s_r += fr.sum(0)
        self.c_r += fr.t() @ fr
        self.s_g += fg.sum(0)
        self.c_g += fg.t() @ fg

    def reduce(self):
        for t in (self.n, self.s_r, self.c_r, self.s_g, self.c_g):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)

    def compute(self) -> float:
        from scipy import linalg
        n = float(self.n.item())
        mu_r = self.s_r / n
        mu_g = self.s_g / n
        sig_r = (self.c_r - n * torch.outer(mu_r, mu_r)) / (n - 1.0)
        sig_g = (self.c_g - n * torch.outer(mu_g, mu_g)) / (n - 1.0)
        mu_r, mu_g = mu_r.cpu().numpy(), mu_g.cpu().numpy()
        sig_r, sig_g = sig_r.cpu().numpy(), sig_g.cpu().numpy()
        diff = mu_r - mu_g
        covmean, _ = linalg.sqrtm(sig_r @ sig_g, disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        return float(diff @ diff + np.trace(sig_r + sig_g - 2.0 * covmean))


def _rel_png(src: str, data_root: str) -> str:
    rel = os.path.relpath(src, data_root)
    return os.path.splitext(rel)[0] + ".png"


@torch.no_grad()
def main(args):
    from transformers import AutoImageProcessor, AutoModel
    import lpips

    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    distributed = world > 1
    if distributed:
        dist.init_process_group("nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
    else:
        local_rank = 0
    device = torch.device("cuda", local_rank % torch.cuda.device_count())
    torch.cuda.set_device(device)

    size = args.image_size
    pre_tf = transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC,
                          antialias=True),
        transforms.CenterCrop(size),
    ])
    tf = transforms.Compose([
        pre_tf,
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    to_tensor = transforms.ToTensor()

    images = list_images(args.data_path)
    if args.num_samples is not None:
        images = images[: args.num_samples]
    mine = images[rank::world]
    if rank == 0:
        print(f"[eval] size={size} total={len(images)} per_rank={len(mine)}", flush=True)

    lpips_model = lpips.LPIPS(net="vgg").to(device).eval()
    fid = FIDAccumulator(device) if args.fid else None

    out_root = Path(args.out)
    metrics_dir = out_root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    summary = {}

    for name in [m.strip() for m in args.models.split(",") if m.strip()]:
        model_subdir, kind = MODELS[name]
        model_dir = os.path.join(args.model_root, model_subdir)
        kwargs = {"torch_dtype": torch.bfloat16} if kind == "refiner" else {}
        model = AutoModel.from_pretrained(model_dir, trust_remote_code=True,
                                         **kwargs).to(device).eval()
        proc = AutoImageProcessor.from_pretrained(model_dir, trust_remote_code=True)
        model_out = out_root / name / str(size)
        t0 = time.time()

        # ---- phase 1: generate reconstructions (skippable PNGs) ----
        made = 0
        for i, src in enumerate(mine):
            dst = model_out / _rel_png(src, args.data_path)
            if dst.is_file() and dst.stat().st_size > 0:
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            img = pre_tf(Image.open(src).convert("RGB"))
            inputs = proc(images=img)
            out_img = model(**inputs)[0]
            out_img.save(dst)
            made += 1
            if rank == 0 and made % 200 == 0:
                print(f"[{name} {size}] generated {made} ({time.time()-t0:.0f}s)", flush=True)
        if distributed:
            dist.barrier()
        if rank == 0:
            print(f"[{name} {size}] generation done in {time.time()-t0:.0f}s", flush=True)

        # ---- phase 2: metrics on saved reconstructions ----
        t1 = time.time()
        p_sum = s_sum = l_sum = 0.0
        n = 0
        for src in mine:
            dst = model_out / _rel_png(src, args.data_path)
            gt = tf(Image.open(src).convert("RGB"))
            rec_img = Image.open(dst).convert("RGB")
            if rec_img.size != (size, size):
                rec_img = rec_img.resize((size, size), Image.BICUBIC)
            rec = to_tensor(rec_img).mul(2).sub(1)

            rec01 = ((rec + 1) / 2).clamp(0, 1)
            gt01 = ((gt + 1) / 2).clamp(0, 1)
            rec8 = (rec01 * 255.0).round().div(255.0)
            rec_np = rec8.permute(1, 2, 0).numpy()
            gt_np = gt01.permute(1, 2, 0).numpy()
            p_sum += float(compute_psnr(gt_np, rec_np, data_range=1.0))
            s_sum += float(compute_ssim(gt_np, rec_np, data_range=1.0, channel_axis=-1))
            l_sum += float(lpips_model(rec[None].to(device), gt[None].to(device)).item())
            if fid is not None:
                fid.update(rec01, gt01)
            n += 1
            if rank == 0 and n % 200 == 0:
                print(f"[{name} {size}] metrics {n} ({time.time()-t1:.0f}s)", flush=True)

        stats = torch.tensor([n, p_sum, s_sum, l_sum], dtype=torch.float64, device=device)
        if distributed:
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            if fid is not None:
                fid.reduce()

        if rank == 0:
            total = float(stats[0].item())
            rfid = fid.compute() if fid is not None else None
            rec = {
                "model": name,
                "image_size": size,
                "images": int(total),
                "psnr": float(stats[1].item() / total),
                "ssim": float(stats[2].item() / total),
                "lpips": float(stats[3].item() / total),
                "rfid": rfid,
                "out_dir": str(model_out),
                "elapsed_s": time.time() - t0,
            }
            summary[f"{name}@{size}"] = rec
            (metrics_dir / f"{name}_{size}.json").write_text(json.dumps(rec, indent=2))
            print(f"[{name} {size}] PSNR={rec['psnr']:.4f} SSIM={rec['ssim']:.4f} "
                  f"LPIPS={rec['lpips']:.4f} rFID={rfid}", flush=True)
        if fid is not None:
            fid = FIDAccumulator(device)

        del model, proc
        torch.cuda.empty_cache()

    if rank == 0:
        (metrics_dir / f"summary_{size}.json").write_text(json.dumps(summary, indent=2))
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-root", default="pretrained",
                    help="Directory containing the named HuggingFace model exports (default: pretrained)")
    ap.add_argument("--data-path", required=True, help="ImageNet validation image directory")
    ap.add_argument("--out", default="outputs/imagenet", help="Reconstruction and metrics output directory")
    ap.add_argument("--models", default="dNaViT-G16,dNaViT-G64,dNaViT-G16R1")
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--num-samples", type=int, default=None)
    ap.add_argument("--fid", action="store_true")
    main(ap.parse_args())
