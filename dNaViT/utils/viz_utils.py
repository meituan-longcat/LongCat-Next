# Copyright (c) 2026 LongCat dNaViT Team.
# SPDX-License-Identifier: Apache-2.0

""" This file contains some utils functions for visualization."""

import torch
import torchvision.transforms.functional as F
from einops import rearrange

def make_viz_from_samples(
    original_images,
    reconstructed_images
):
    """Generates visualization images from original images and reconstructed images.

    Args:
        original_images: A torch.Tensor, original images.
        reconstructed_images: A torch.Tensor, reconstructed images.

    Returns:
        A tuple containing two lists - images_for_saving and images_for_logging.
    """
    reconstructed_images = torch.clamp(reconstructed_images, 0.0, 1.0)
    reconstructed_images = reconstructed_images * 255.0
    reconstructed_images = reconstructed_images.cpu()
    
    original_images = torch.clamp(original_images, 0.0, 1.0)
    original_images *= 255.0
    original_images = original_images.cpu()

    diff_img = torch.abs(original_images - reconstructed_images)
    to_stack = [original_images, reconstructed_images, diff_img]

    images_for_logging = rearrange(
            torch.stack(to_stack),
            "(l1 l2) b c h w -> b c (l1 h) (l2 w)",
            l1=1).byte()
    images_for_saving = [F.to_pil_image(image) for image in images_for_logging]

    return images_for_saving, images_for_logging
