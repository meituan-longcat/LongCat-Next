# Copyright (c) 2026 LongCat dNaViT Team.
# SPDX-License-Identifier: Apache-2.0

"""Portable resolution of pretrained weight files.

The perceptual / LPIPS / Gram / DINO-discriminator losses need a few ImageNet
backbone checkpoints on disk. They are NOT shipped with the repository; put them
under ``<repo>/pretrained/`` (default) or point ``DNAVIT_PRETRAINED`` at any
directory containing them.

Expected file names::

    vgg16-397923af.pth              # torchvision VGG16 IMAGENET1K_V1
    convnext_small-0c510722.pth     # torchvision ConvNeXt-Small IMAGENET1K_V1
    vgg_lpips.pth                   # LPIPS v0.1 VGG linear layers (richzhang/PerceptualSimilarity)
    dino_vit_small_patch8_224.pth   # DINO ViT-S/8 (facebookresearch/dino)
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_VAR = "DNAVIT_PRETRAINED"

# Where each expected file comes from, quoted verbatim in the error message.
SOURCE_HINTS = {
    "vgg16-397923af.pth":
        "https://download.pytorch.org/models/vgg16-397923af.pth",
    "convnext_small-0c510722.pth":
        "https://download.pytorch.org/models/convnext_small-0c510722.pth",
    "vgg_lpips.pth":
        "LPIPS v0.1 VGG linear weights (lpips/weights/v0.1/vgg.pth from "
        "https://github.com/richzhang/PerceptualSimilarity)",
    "dino_vit_small_patch8_224.pth":
        "DINO ViT-S/8 weights from https://github.com/facebookresearch/dino",
}


def pretrained_dir() -> Path:
    """Directory searched for pretrained backbone weights."""
    return Path(os.environ.get(ENV_VAR, REPO_ROOT / "pretrained"))


def pretrained_path(filename: str) -> str:
    """Absolute path to ``filename`` under :func:`pretrained_dir`.

    Raises ``FileNotFoundError`` with the download hint when missing, so a run
    fails fast with an actionable message instead of a silent random init.
    """
    path = pretrained_dir() / filename
    if not path.is_file():
        hint = SOURCE_HINTS.get(filename, "obtain the upstream checkpoint")
        raise FileNotFoundError(
            f"Pretrained weights not found: {path}\n"
            f"  Download from: {hint}\n"
            f"  Then place it at that path, or set {ENV_VAR} to the directory "
            f"that contains '{filename}'.")
    return str(path)
