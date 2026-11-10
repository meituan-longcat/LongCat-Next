# Copyright (c) 2026 LongCat dNaViT Team.
# SPDX-License-Identifier: Apache-2.0

"""ImageNet-256 reconstruction evaluation for a trained dNaViT tokenizer.

Feeds every image at a fixed 256x256 (resize shorter edge + center crop) through
``encode -> quantize -> decode`` and reports PSNR / SSIM, plus an optional rFID
between the reconstructions and the same ground-truth crops.

Launch (multi-GPU)::

    torchrun --standalone --nproc_per_node=4 scripts/eval_recon.py \
        --config configs/dnavit_stageC_anyres_g16_32x.yaml \
        --ckpt runs/<run>/checkpoint-latest/unwrapped_model \
        --data-path /path/to/imagenet/val \
        --image-size 256 --bf16 --fid

Single GPU also works: ``python scripts/eval_recon.py ...`` (no torchrun).
Images are read from any ImageFolder-style directory tree.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

try:
    from tqdm import tqdm
except ImportError:                       # pragma: no cover
    def tqdm(x, **k):
        return x

from utils.train_utils import build_tokenizer

_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff"}


def _list_images(root: str) -> List[str]:
    imgs = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if Path(fn).suffix.lower() in _IMG_EXTS:
                imgs.append(os.path.join(dp, fn))
    imgs.sort()
    if not imgs:
        raise ValueError(f"No images found under {root}")
    return imgs


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class FixedResDataset(Dataset):
    """Resize (shorter edge) + center-crop to a fixed square; normalize 0.5/0.5."""

    def __init__(self, root: str, image_size: int):
        self.images = _list_images(root)
        self.tf = transforms.Compose([
            transforms.Resize(image_size,
                              interpolation=transforms.InterpolationMode.BICUBIC,
                              antialias=True),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),                       # [0, 1]
            transforms.Normalize([0.5] * 3, [0.5] * 3),  # -> [-1, 1]
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = Image.open(self.images[i]).convert("RGB")
        return self.tf(img)                              # (3, S, S) in [-1, 1]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _to01(x: torch.Tensor) -> torch.Tensor:
    return ((x + 1.0) / 2.0).clamp(0.0, 1.0)


def _psnr_ssim(rec01: torch.Tensor, gt01: torch.Tensor) -> Tuple[float, float]:
    """rec01/gt01: (3, H, W) in [0, 1]. Recon rounded to the 8-bit grid."""
    rec = (rec01 * 255.0).round().div(255.0)
    rec_np = rec.permute(1, 2, 0).cpu().numpy()
    gt_np = gt01.permute(1, 2, 0).cpu().numpy()
    psnr = compute_psnr(gt_np, rec_np, data_range=1.0)
    ssim = compute_ssim(gt_np, rec_np, data_range=1.0, channel_axis=-1)
    return float(psnr), float(ssim)


class FIDAccumulator:
    """Streams InceptionV3 pool3 (2048-d) features for recon and GT, keeping only
    running sufficient statistics (count, feature-sum, feature-gram) per set so
    memory is O(2048^2) regardless of #images. Combine across ranks with
    ``reduce()`` (sums the stats), then ``compute()`` returns the Frechet
    distance FID(recon, real)."""

    def __init__(self, device: torch.device):
        from pytorch_fid.inception import InceptionV3
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        self.model = InceptionV3([block_idx]).to(device).eval()
        self.device = device
        d = 2048
        self.n = torch.zeros(1, dtype=torch.float64, device=device)
        self.s_r = torch.zeros(d, dtype=torch.float64, device=device)     # recon sum
        self.c_r = torch.zeros(d, d, dtype=torch.float64, device=device)  # recon gram
        self.s_g = torch.zeros(d, dtype=torch.float64, device=device)     # gt sum
        self.c_g = torch.zeros(d, d, dtype=torch.float64, device=device)  # gt gram

    @torch.no_grad()
    def _feat(self, imgs01: List[torch.Tensor]) -> torch.Tensor:
        # Resize to 299 so a batch can be stacked; InceptionV3 normalises
        # [0,1] -> [-1,1] internally.
        x = torch.stack([
            F.interpolate(im.unsqueeze(0), size=(299, 299), mode="bilinear",
                          align_corners=False).squeeze(0).clamp(0, 1)
            for im in imgs01
        ])
        f = self.model(x.float())[0].squeeze(-1).squeeze(-1)   # (B, 2048)
        return f.double()

    @torch.no_grad()
    def update(self, rec01_list: List[torch.Tensor], gt01_list: List[torch.Tensor]):
        fr = self._feat(rec01_list)
        fg = self._feat(gt01_list)
        self.n += fr.shape[0]
        self.s_r += fr.sum(0); self.c_r += fr.t() @ fr
        self.s_g += fg.sum(0); self.c_g += fg.t() @ fg

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
        mu_r = mu_r.cpu().numpy(); mu_g = mu_g.cpu().numpy()
        sig_r = sig_r.cpu().numpy(); sig_g = sig_g.cpu().numpy()
        diff = mu_r - mu_g
        covmean, _ = linalg.sqrtm(sig_r @ sig_g, disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        return float(diff @ diff + np.trace(sig_r + sig_g - 2.0 * covmean))


# ---------------------------------------------------------------------------
# Inference loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def _run_fixed(model, args, device, rank, world, fid=None):
    ds = FixedResDataset(args.data_path, args.image_size)
    total = len(ds)
    req = total if args.num_samples is None else min(args.num_samples, total)
    idx = list(range(req))[rank::world]
    loader = DataLoader(Subset(ds, idx), batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers, pin_memory=True)
    if rank == 0:
        print(f"[fixed] image_size={args.image_size} total={req} per_rank={len(idx)}")

    P, S = [], []
    for x in (tqdm(loader, desc="fixed") if rank == 0 else loader):
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.bf16):
            recon, _ = model(x)                          # (B, 3, S, S) [-1, 1]
        recon = _to01(recon.float())
        gt = _to01(x.float())
        for b in range(x.shape[0]):
            p, s = _psnr_ssim(recon[b], gt[b])
            P.append(p); S.append(s)
        if fid is not None:
            fid.update([recon[b] for b in range(x.shape[0])],
                       [gt[b] for b in range(x.shape[0])])
    return P, S, req


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_model(args, device):
    config = OmegaConf.load(args.config)
    if args.override:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(args.override))
    model = build_tokenizer(config).to(device).eval()

    # Resolve the weight file: a .bin/.pt file, or a dir containing
    # pytorch_model.bin (e.g. checkpoint-<step>/unwrapped_model).
    ckpt = args.ckpt
    if os.path.isdir(ckpt):
        cand = os.path.join(ckpt, "pytorch_model.bin")
        if not os.path.isfile(cand):
            raise FileNotFoundError(f"{cand} not found")
        ckpt = cand
    sd = torch.load(ckpt, map_location="cpu")
    if isinstance(sd, dict) and "model" in sd and not any(
            k.startswith(("encoder.", "decoder.", "quantize.")) for k in sd):
        sd = sd["model"]
    # Strip torch.compile / DDP prefixes if present.
    sd = {k.replace("_orig_mod.module.", "").replace("_orig_mod.", "").replace("module.", ""): v
          for k, v in sd.items()}
    msg = model.load_state_dict(sd, strict=False)
    if torch.cuda.current_device() == device.index:
        print(f"[load] {ckpt}\n  missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")
    model.requires_grad_(False)
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    if not torch.cuda.is_available():
        raise RuntimeError("Eval requires a CUDA GPU.")
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if distributed:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
    else:
        rank, world = 0, 1
    device = torch.device("cuda", rank % torch.cuda.device_count())
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model = _load_model(args, device)

    fid = FIDAccumulator(device) if args.fid else None
    P, S, req = _run_fixed(model, args, device, rank, world, fid)

    # Gather across ranks.
    if distributed:
        dist.barrier()
        gP = [None] * world
        gS = [None] * world
        dist.all_gather_object(gP, P)
        dist.all_gather_object(gS, S)
        P = [v for sub in gP for v in sub]
        S = [v for sub in gS for v in sub]

    rfid = None
    if fid is not None:
        if distributed:
            fid.reduce()
        if rank == 0:
            rfid = fid.compute()

    if rank == 0:
        psnr = float(np.mean(P)) if P else float("nan")
        ssim = float(np.mean(S)) if S else float("nan")
        print("\n" + "=" * 50)
        print(f"  mode  : fixed (image_size={args.image_size})")
        print(f"  images: {len(P)}")
        print(f"  PSNR  : {psnr:.4f}")
        print(f"  SSIM  : {ssim:.4f}")
        if rfid is not None:
            print(f"  rFID  : {rfid:.4f}")
        print("=" * 50)
        if args.result_file:
            with open(args.result_file, "w") as f:
                f.write(f"mode: fixed\nimage_size: {args.image_size}\nimages: {len(P)}\n")
                f.write(f"PSNR: {psnr:.6f}\nSSIM: {ssim:.6f}\n")
                if rfid is not None:
                    f.write(f"rFID: {rfid:.6f}\n")
            print(f"Saved to {args.result_file}")

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="dNaViT tokenizer ImageNet-256 reconstruction eval")
    p.add_argument("--config", required=True, help="Tokenizer config YAML (same as training).")
    p.add_argument("--ckpt", required=True,
                   help="pytorch_model.bin, or a dir containing it "
                        "(e.g. checkpoint-<step>/unwrapped_model).")
    p.add_argument("--data-path", required=True, help="ImageNet val root (ImageFolder layout).")
    p.add_argument("--image-size", type=int, default=256, help="Square eval size.")
    p.add_argument("--batch-size", type=int, default=8, help="Per-GPU batch size.")
    p.add_argument("--num-samples", type=int, default=None, help="Cap #images (default: all).")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--bf16", action="store_true", help="Autocast bf16 (recommended).")
    p.add_argument("--fid", action="store_true",
                   help="Also compute rFID (pytorch_fid InceptionV3 2048-d) vs the same GT crops.")
    p.add_argument("--result-file", type=str, default=None, help="Optional txt to write metrics.")
    p.add_argument("--override", nargs="*", default=[],
                   help="OmegaConf dotlist overrides, e.g. "
                        "model.vq_model.qwen3vit_path=/path/to/Qwen3-VL-8B-Instruct")
    main(p.parse_args())
