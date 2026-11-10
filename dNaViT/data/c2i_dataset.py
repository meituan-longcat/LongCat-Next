# Copyright (c) 2026 LongCat dNaViT Team.
# SPDX-License-Identifier: Apache-2.0

"""Fixed-resolution ImageNet dataset for class-to-image tokenizer training.

Reads from a standard ImageFolder layout:
    <root>/<wnid>/*.JPEG

The class id is assigned by sorted wnid order (1000 classes for ILSVRC2012).
For tokenizer training the class id is unused, but it's kept in the batch dict
so the same dataset can drive a downstream c2i AR-generator without changes.

Output (default-collate friendly):
    {
        "image":    torch.Tensor (3, H, W)   # normalized to [-1, 1]
        "class_id": int
        "__key__":  str                       # filename (no extension)
    }
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


_DEFAULT_EXTS = (".JPEG", ".jpeg", ".jpg", ".png", ".webp")


def _build_imagefolder_index(
    root,
    extensions=_DEFAULT_EXTS,
):
    """Walk ``<root>/<wnid>/*.<ext>`` and return ``[(path, class_id), ...]``.

    ``wnids`` are sorted ascending so class ids are deterministic across
    machines without needing a separate mapping file.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"ImageNet root not found: {root}")
    wnids = sorted(d.name for d in root.iterdir() if d.is_dir())
    if not wnids:
        raise RuntimeError(f"No class subdirectories found under {root}")
    wnid_to_id = {w: i for i, w in enumerate(wnids)}

    samples: List[Tuple[str, int]] = []
    for w in wnids:
        cls_dir = root / w
        cid = wnid_to_id[w]
        for p in sorted(cls_dir.iterdir()):
            if p.suffix in extensions:
                samples.append((str(p), cid))
    return samples, wnids


def build_imagenet_transform(
    crop_size: int = 256,
    resize_shorter_edge: int = 256,
    random_crop: bool = True,
    random_flip: bool = True,
    normalize_mean: Iterable[float] = (0.5, 0.5, 0.5),
    normalize_std: Iterable[float] = (0.5, 0.5, 0.5),
    train: bool = True,
) -> transforms.Compose:
    """Resize / crop / flip / normalize pipeline for fixed-resolution training."""
    interpolation = transforms.InterpolationMode.BICUBIC
    if train:
        ops = [
            transforms.Resize(resize_shorter_edge, interpolation=interpolation, antialias=True),
            transforms.RandomCrop(crop_size) if random_crop else transforms.CenterCrop(crop_size),
        ]
        if random_flip:
            ops.append(transforms.RandomHorizontalFlip())
    else:
        ops = [
            transforms.Resize(crop_size, interpolation=interpolation, antialias=True),
            transforms.CenterCrop(crop_size),
        ]
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(list(normalize_mean), list(normalize_std)),
    ]
    return transforms.Compose(ops)


class C2IImageNet(Dataset):
    """ImageFolder-style dataset for fixed-resolution ImageNet training.

    Reads decoded JPEGs from ``<root>/<class>/*.JPEG``.
    """

    def __init__(
        self,
        root,
        crop_size: int = 256,
        resize_shorter_edge: int = 256,
        random_crop: bool = True,
        random_flip: bool = True,
        normalize_mean: Iterable[float] = (0.5, 0.5, 0.5),
        normalize_std: Iterable[float] = (0.5, 0.5, 0.5),
        train: bool = True,
        extensions: Sequence[str] = _DEFAULT_EXTS,
        max_samples: Optional[int] = None,
    ):
        self.root = str(root)
        self.samples, self.wnids = _build_imagefolder_index(root, extensions=extensions)
        if max_samples is not None:
            self.samples = self.samples[:max_samples]
        self.transform = build_imagenet_transform(
            crop_size=crop_size,
            resize_shorter_edge=resize_shorter_edge,
            random_crop=random_crop,
            random_flip=random_flip,
            normalize_mean=normalize_mean,
            normalize_std=normalize_std,
            train=train,
        )
        self.num_classes = len(self.wnids)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, cid = self.samples[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
            tensor = self.transform(img)
        key = os.path.splitext(os.path.basename(path))[0]
        return {"image": tensor, "class_id": int(cid), "__key__": key}
