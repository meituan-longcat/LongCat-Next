# Copyright (c) 2026 LongCat dNaViT Team.
# SPDX-License-Identifier: Apache-2.0

"""Rotary Position Embedding (RoPE) implementation.

Reference:
    https://github.com/LTH14/JiT/blob/main/util/model_util.py
"""

import math
import torch
import torch.nn as nn
from einops import rearrange


def broadcat(tensors, dim=-1):
    """Broadcast and concatenate tensors along a dimension."""
    num_tensors = len(tensors)
    shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
    assert len(shape_lens) == 1, 'tensors must all have the same number of dimensions'
    shape_len = list(shape_lens)[0]

    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))

    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all([*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]), \
        'invalid dimensions for broadcastable concatentation'
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes)))
    return torch.cat(tensors, dim=dim)


def rotate_half(x):
    """Rotate half the hidden dims of the input."""
    x = rearrange(x, '... (d r) -> ... d r', r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, '... d r -> ... (d r)')


class VisionRotaryEmbeddingFast(nn.Module):
    """Fast Rotary Position Embedding for Vision Transformers.

    This implementation supports 2D spatial rotary embeddings for vision transformers,
    with optional support for class tokens and different frequency modes.

    Args:
        dim: Dimension of the rotary embedding (typically half of head_dim)
        pt_seq_len: Pretraining sequence length (spatial size)
        ft_seq_len: Fine-tuning sequence length (spatial size), defaults to pt_seq_len
        custom_freqs: Custom frequency tensor if provided
        freqs_for: Frequency mode - 'lang', 'pixel', or 'constant'
        theta: Base for language model frequencies (default: 10000)
        max_freq: Maximum frequency for pixel mode (default: 10)
        num_freqs: Number of frequencies for constant mode (default: 1)
        num_cls_token: Number of class tokens to prepend (default: 0)
    """

    def __init__(
        self,
        dim,
        pt_seq_len=16,
        ft_seq_len=None,
        custom_freqs=None,
        freqs_for='lang',
        theta=10000,
        max_freq=10,
        num_freqs=1,
        num_cls_token=0
    ):
        super().__init__()

        # Frequency computation based on different modes
        if custom_freqs is not None:
            freqs = custom_freqs
        elif freqs_for == 'lang':
            freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        elif freqs_for == 'pixel':
            freqs = torch.linspace(1., max_freq / 2, dim // 2) * math.pi
        elif freqs_for == 'constant':
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f'unknown modality {freqs_for}')

        # Sequence length handling
        if ft_seq_len is None:
            ft_seq_len = pt_seq_len
        t = torch.arange(ft_seq_len).float() / ft_seq_len * pt_seq_len

        # Frequency computation for 2D spatial positions
        freqs = torch.einsum('..., f -> ... f', t, freqs)
        freqs = freqs.repeat_interleave(2, dim=-1)  # repeat each frequency twice
        freqs = broadcat((freqs[:, None, :], freqs[None, :, :]), dim=-1)

        # Handling class tokens if present
        if num_cls_token > 0:
            freqs_flat = freqs.reshape(-1, freqs.shape[-1])
            cos_img = freqs_flat.cos()
            sin_img = freqs_flat.sin()

            N_img, D = cos_img.shape
            cos_pad = torch.ones(num_cls_token, D, dtype=cos_img.dtype, device=cos_img.device)
            sin_pad = torch.zeros(num_cls_token, D, dtype=sin_img.dtype, device=sin_img.device)

            freqs_cos = torch.cat([cos_pad, cos_img], dim=0)
            freqs_sin = torch.cat([sin_pad, sin_img], dim=0)
        else:
            freqs_cos = freqs.cos().reshape(-1, freqs.shape[-1])
            freqs_sin = freqs.sin().reshape(-1, freqs.shape[-1])

        # Register as buffers
        self.register_buffer('freqs_cos', freqs_cos)
        self.register_buffer('freqs_sin', freqs_sin)

    @torch.amp.autocast(device_type='cuda', enabled=False)
    def forward(self, t, rope_order=None):
        """Apply rotary position embedding to input tensor.

        Args:
            t: Input tensor of shape (B, num_heads, seq_len, head_dim)
            rope_order: Optional tensor of position indices for each token
                       - Training: (B, N) specifying position for each token
                       - Sampling with KV cache: (B, 1) for the current token
                       - If None, uses sequential positions [0, 1, 2, ...]

        Returns:
            Tensor with rotary position embeddings applied
        """
        # Convert to fp32 for numerical stability
        input_dtype = t.dtype
        t = t.float()

        # Determine sequence length from input tensor
        # t shape: (B, num_heads, N, head_dim)
        N = t.shape[2]

        # Select appropriate RoPE frequencies based on rope_order
        # Assume t is always 4D: (B, num_heads, N, head_dim)
        if rope_order is not None:
            # rope_order specifies position indices for each token
            # rope_order shape:
            #   - Training: (B, N) where N is full sequence length
            #   - Sampling with KV cache: (B, 1) for the current token
            # Advanced indexing: freqs_cos[rope_order] outputs (B, N, head_dim) or (B, 1, head_dim)
            # Expand to (B, 1, N, head_dim) or (B, 1, 1, head_dim) for broadcasting
            rope_freqs_cos = self.freqs_cos[rope_order].unsqueeze(1)
            rope_freqs_sin = self.freqs_sin[rope_order].unsqueeze(1)
        else:
            # Standard mode: sequential positions from 0
            # freqs_cos[:N] outputs (N, head_dim)
            # Expand to (1, 1, N, head_dim) for broadcasting
            rope_freqs_cos = self.freqs_cos[:N].unsqueeze(0).unsqueeze(0)
            rope_freqs_sin = self.freqs_sin[:N].unsqueeze(0).unsqueeze(0)

        # Compute RoPE in fp32 and convert back to original dtype
        output = t * rope_freqs_cos + rotate_half(t) * rope_freqs_sin
        return output.to(input_dtype)
