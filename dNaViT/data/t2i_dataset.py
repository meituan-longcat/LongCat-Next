# Copyright (c) 2026 LongCat dNaViT Team.
# SPDX-License-Identifier: Apache-2.0

"""Dynamic-resolution jsonl dataset for tokenizer / t2i pretraining.

Reads one or more jsonl files where each line looks like:

    {"input": "<img_start><fixres_224>/abs/path/img.JPEG<img_end>",
     "source": "imagenet1k_train",
     "img_size": [H, W]}

The ``<fixres_NNN>`` marker is optional:
  - present  -> center-crop-resize the image to NxN (fixed-resolution
                preprocessor).
  - absent   -> smart-resize so H, W become multiples of
                ``patch_size * spatial_merge_size`` (32 for dNaViT), keeping the
                aspect ratio and clamping the token count to
                ``[min_pixels, max_pixels]``.

For tokenizer training only the image is needed; the optional ``text`` /
``source`` fields are kept so the same dataset class can later be reused for
text-to-image pretraining.

Output per sample (use the provided ``t2i_collate``):
    {
        "image":           torch.Tensor (3, H, W)       # normalized to [-1, 1]
        "image_grid_thw":  tuple(1, h_p, w_p)            # h_p = H // 16, etc.
        "__key__":         str                            # the original path
    }
"""
from __future__ import annotations

import json
import math
import os
import random
import re
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import hflip, normalize, to_tensor
from PIL import Image

# Some jsonl entries point at huge source images (e.g. 256MP scanned pages).
# PIL refuses to open anything above ~178MP by default (DecompressionBombError).
# We don't need the protection — smart_resize + max_tokens_per_sample clamp the
# post-resize shape to a few MP at most — so disable the cap globally.
Image.MAX_IMAGE_PIXELS = None


# dNaViT's qwen3vit: patch_size=16, spatial_merge_size=2 -> factor=32.
DEFAULT_PATCH_SIZE = 16
DEFAULT_MERGE_SIZE = 2
DEFAULT_FACTOR = DEFAULT_PATCH_SIZE * DEFAULT_MERGE_SIZE
# Bound the patch count so a single 2048**2 image doesn't OOM a small batch.
DEFAULT_MIN_PIXELS = 56 * 56                # 9 patches at factor=32
DEFAULT_MAX_PIXELS = 28 * 28 * 1280         # ~1k patches at factor=32
DEFAULT_MAX_ASPECT = 200

_IMG_START_RE = re.compile(r"<img_start>(.*?)<img_end>", re.DOTALL)
_FIX_RES_RE = re.compile(r"<fixres_(\d+)>")


def smart_resize(
    height: int,
    width: int,
    factor: int = DEFAULT_FACTOR,
    min_pixels: int = DEFAULT_MIN_PIXELS,
    max_pixels: int = DEFAULT_MAX_PIXELS,
    max_aspect: float = DEFAULT_MAX_ASPECT,
) -> Tuple[int, int]:
    """Round to the nearest multiple of ``factor`` while keeping aspect ratio
    and bounding the total pixel count to ``[min_pixels, max_pixels]``.

    Keeps the aspect ratio, aligns H/W to ``factor`` (= patch_size *
    spatial_merge_size), and clamps the total pixel count into
    ``[min_pixels, max_pixels]``.
    """
    if height < factor or width < factor:
        scale = factor / min(height, width)
        height = int(round(height * scale))
        width = int(round(width * scale))
    if max(height, width) / min(height, width) > max_aspect:
        short_side = min(height, width)
        target_long = int(short_side * max_aspect)
        if height >= width:
            height = target_long
        else:
            width = target_long

    h_align = max(factor, int(round(height / factor)) * factor)
    w_align = max(factor, int(round(width / factor)) * factor)
    if h_align * w_align > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_align = max(factor, math.floor(height / beta / factor) * factor)
        w_align = max(factor, math.floor(width / beta / factor) * factor)
    elif h_align * w_align < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_align = math.ceil(height * beta / factor) * factor
        w_align = math.ceil(width * beta / factor) * factor
    return h_align, w_align


def parse_img_input(field: str) -> Tuple[str, int]:
    """Extract ``(path, fix_res)`` from the ``input`` field of a jsonl line.

    Returns ``(path, -1)`` when no ``<fixres_NNN>`` marker is present.
    """
    m = _IMG_START_RE.search(field)
    if m is None:
        raise ValueError(f"Cannot find <img_start>...<img_end> in input: {field!r}")
    inner = m.group(1)
    fm = _FIX_RES_RE.search(inner)
    if fm is None:
        return inner.strip(), -1
    fix_res = int(fm.group(1))
    return _FIX_RES_RE.sub("", inner).strip(), fix_res


def _center_crop_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    s = min(w, h)
    left = (w - s) // 2
    top = (h - s) // 2
    return img.crop((left, top, left + s, top + s))


class T2IJsonlDataset(Dataset):
    """Variable-resolution image dataset built from a list of jsonl files.

    Each ``__getitem__`` returns a single image tensor together with its
    ``image_grid_thw`` triple so the t2i collate can pack a batch for the
    varlen encoder.

    Args:
        jsonl_paths: list of jsonl files. Each line must contain at least the
            ``input`` field as described in the module docstring.
        patch_size: encoder patch size (=16 for the qwen3vit backbone). Determines
            ``h_p = H // patch_size``.
        spatial_merge_size: encoder spatial merge size (=2 for qwen3vit).
            Together with ``patch_size`` sets the resize ``factor``.
        min_pixels / max_pixels: token-count bounds. Bigger images are scaled
            down; smaller ones up.
        max_tokens_per_sample: hard cap. Post-resize images larger than this
            are scaled down further on the factor grid so the cap is honored.
        random_flip: random horizontal flip aug.
        shuffle_index: if True, shuffle the (path, fix_res) list once at init
            with the given ``seed`` so workers see different shards quickly.
    """

    def __init__(
        self,
        jsonl_paths,
        patch_size: int = DEFAULT_PATCH_SIZE,
        spatial_merge_size: int = DEFAULT_MERGE_SIZE,
        min_pixels: int = DEFAULT_MIN_PIXELS,
        max_pixels: int = DEFAULT_MAX_PIXELS,
        max_tokens_per_sample: Optional[int] = 1024,
        normalize_mean: Iterable[float] = (0.5, 0.5, 0.5),
        normalize_std: Iterable[float] = (0.5, 0.5, 0.5),
        random_flip: bool = True,
        shuffle_index: bool = True,
        seed: int = 0,
        skip_missing: bool = True,
        force_fix_res: int = -1,
    ):
        if isinstance(jsonl_paths, (str, os.PathLike)):
            jsonl_paths = [jsonl_paths]
        self.patch_size = int(patch_size)
        self.merge_size = int(spatial_merge_size)
        self.factor = self.patch_size * self.merge_size
        self.min_pixels = int(min_pixels)
        self.max_pixels = int(max_pixels)
        # Curriculum override: when > 0, EVERY sample is center-crop-resized to
        # force_fix_res x force_fix_res, ignoring each record's own fix_res marker
        # and the smart-resize/anyres path. Used by the fixed-res warmup phase
        # (train first N steps at a single resolution, then rebuild the dataset
        # with force_fix_res=-1 to switch to anyres).
        self.force_fix_res = int(force_fix_res)
        self.max_tokens_per_sample = int(max_tokens_per_sample) if max_tokens_per_sample else None
        self.random_flip = bool(random_flip)
        self.normalize_mean = list(normalize_mean)
        self.normalize_std = list(normalize_std)
        self.skip_missing = skip_missing

        self.records = self._load_jsonls(jsonl_paths)
        if shuffle_index:
            rng = random.Random(seed)
            rng.shuffle(self.records)
        # Memoised post-resize token counts, populated lazily by
        # ``tokens_at_index`` (used by the TokenBudgetBatchSampler).
        self._token_count_cache: dict = {}

    def _load_jsonls(self, paths) -> List[Tuple[str, int, Optional[Tuple[int, int]]]]:
        """Parse every jsonl line into ``(path, fix_res, img_size_or_None)``.

        ``img_size`` (``(H, W)``) is read from the jsonl when present so the
        TokenBudgetBatchSampler can compute per-sample token counts without
        actually loading any image. Lines without ``img_size`` are still
        accepted; their token count falls back to ``max_tokens_per_sample``
        (the conservative upper bound) at batching time.
        """
        records: List[Tuple[str, int, Optional[Tuple[int, int]]]] = []
        for p in paths:
            p = str(p)
            if not os.path.exists(p):
                raise FileNotFoundError(f"jsonl not found: {p}")
            with open(p, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        path, fix_res = parse_img_input(obj["input"])
                    except (KeyError, ValueError, json.JSONDecodeError):
                        continue
                    img_size = obj.get("img_size", None)
                    if img_size and len(img_size) >= 2:
                        # jsonl stores [H, W]; coerce to ints.
                        img_size = (int(img_size[0]), int(img_size[1]))
                    else:
                        img_size = None
                    records.append((path, fix_res, img_size))
        if not records:
            raise RuntimeError(f"No valid records found in jsonls: {paths}")
        return records

    def __len__(self) -> int:
        return len(self.records)

    # ---- shape helpers -----------------------------------------------------
    def _eff_fix_res(self, fix_res: int) -> int:
        """Apply the curriculum override: force_fix_res (>0) wins over the
        per-record marker so the whole dataset renders at a single resolution."""
        return self.force_fix_res if self.force_fix_res > 0 else fix_res

    def _target_size(self, src_h: int, src_w: int, fix_res: int) -> Tuple[int, int]:
        if fix_res > 0:
            # Round fix_res up to the nearest factor multiple so the encoder
            # can patchify cleanly. (224 -> 224, 256 -> 256, 250 -> 256.)
            if fix_res % self.factor != 0:
                fix_res = int(math.ceil(fix_res / self.factor) * self.factor)
            return fix_res, fix_res
        h, w = smart_resize(
            src_h, src_w,
            factor=self.factor,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )
        return h, w

    def _maybe_clamp_tokens(self, h: int, w: int) -> Tuple[int, int]:
        if self.max_tokens_per_sample is None:
            return h, w
        h_p = h // self.patch_size
        w_p = w // self.patch_size
        tokens = h_p * w_p
        if tokens <= self.max_tokens_per_sample:
            return h, w
        scale = math.sqrt(self.max_tokens_per_sample / tokens)
        h_new = max(self.factor, int(math.floor(h * scale / self.factor) * self.factor))
        w_new = max(self.factor, int(math.floor(w * scale / self.factor) * self.factor))
        return h_new, w_new

    # ---- core load + transform --------------------------------------------
    def _load_image(self, idx: int):
        path, fix_res, _ = self.records[idx]
        fix_res = self._eff_fix_res(fix_res)
        try:
            img = Image.open(path).convert("RGB")
        except (FileNotFoundError, OSError, Image.UnidentifiedImageError,
                Image.DecompressionBombError) as _e:
            if not self.skip_missing:
                raise
            return None
        src_w, src_h = img.size

        if fix_res > 0:
            img = _center_crop_square(img)
            src_w, src_h = img.size
        target_h, target_w = self._target_size(src_h, src_w, fix_res)
        target_h, target_w = self._maybe_clamp_tokens(target_h, target_w)

        img = img.resize((target_w, target_h), Image.BICUBIC)
        tensor = to_tensor(img)
        if self.random_flip and random.random() < 0.5:
            tensor = hflip(tensor)
        tensor = normalize(tensor, self.normalize_mean, self.normalize_std)

        h_p = target_h // self.patch_size
        w_p = target_w // self.patch_size
        return {
            "image": tensor,
            "image_grid_thw": (1, h_p, w_p),
            "__key__": path,
        }

    def __getitem__(self, idx: int):
        for _ in range(len(self.records)):
            out = self._load_image(idx)
            if out is not None:
                return out
            idx = (idx + 1) % len(self.records)
        raise RuntimeError("All samples failed to load (corrupt jsonl?).")

    # ---- token-budget helpers ---------------------------------------------
    def tokens_at_index(self, idx: int) -> int:
        """Return the post-resize token count ``h_p * w_p`` for sample ``idx``.

        Uses the jsonl-cached ``img_size`` if available; otherwise falls back
        to ``max_tokens_per_sample`` as a safe upper bound so the batch sampler
        never overshoots its budget.
        """
        cached = self._token_count_cache.get(idx)
        if cached is not None:
            return cached
        path, fix_res, img_size = self.records[idx]
        fix_res = self._eff_fix_res(fix_res)
        if img_size is None:
            # Conservative fallback — assume the sample fills the cap.
            t = self.max_tokens_per_sample or 1
        else:
            src_h, src_w = img_size
            target_h, target_w = self._target_size(src_h, src_w, fix_res)
            target_h, target_w = self._maybe_clamp_tokens(target_h, target_w)
            t = (target_h // self.patch_size) * (target_w // self.patch_size)
        self._token_count_cache[idx] = t
        return t


def t2i_collate(batch):
    """Pack a list of variable-shape samples for the varlen tokenizer.

    Returns:
        {
            "image":           List[torch.Tensor (3, H_i, W_i)],
            "image_grid_thw":  torch.LongTensor (B, 3),
            "__key__":         List[str],
        }
    """
    return {
        "image": [b["image"] for b in batch],
        "image_grid_thw": torch.tensor(
            [b["image_grid_thw"] for b in batch], dtype=torch.long),
        "__key__": [b["__key__"] for b in batch],
    }


class TokenBudgetBatchSampler(torch.utils.data.Sampler):
    """BatchSampler that greedily packs samples up to a total-token budget.

    Used by the anyres / t2i path where a fixed ``batch_size`` is the wrong
    semantic — what actually drives GPU memory is the sum of ``h_p * w_p``
    over the batch. The sampler is rank/world-aware so each process sees a
    disjoint slice of the dataset per epoch.

    Args:
        dataset: a ``T2IJsonlDataset`` exposing ``tokens_at_index(idx) -> int``.
        max_batch_tokens: upper bound on the per-batch token sum. Once the
            running total would exceed the bound, the current batch is yielded
            and a new one started.
        shuffle: shuffle indices each epoch (recommended).
        drop_last: if True, discard the final partial batch.
        seed: base seed; the actual per-epoch seed is ``seed + epoch``.
        num_replicas / rank: distributed sharding info. If ``None``, the
            sampler runs as if single-process.
    """

    def __init__(
        self,
        dataset,
        max_batch_tokens: int,
        shuffle: bool = True,
        drop_last: bool = True,
        seed: int = 0,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
    ):
        if max_batch_tokens <= 0:
            raise ValueError(f"max_batch_tokens must be positive, got {max_batch_tokens}.")
        if not hasattr(dataset, "tokens_at_index"):
            raise TypeError(
                "TokenBudgetBatchSampler requires a dataset exposing "
                "tokens_at_index(idx).")
        self.dataset = dataset
        self.max_batch_tokens = int(max_batch_tokens)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.num_replicas = int(num_replicas) if num_replicas is not None else 1
        self.rank = int(rank) if rank is not None else 0
        if not (0 <= self.rank < self.num_replicas):
            raise ValueError(
                f"rank={self.rank} out of range for num_replicas={self.num_replicas}.")
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def _rank_slice(self, indices: List[int]) -> List[int]:
        # Trim to a multiple of num_replicas so every rank sees the same
        # number of indices (drops at most num_replicas-1 samples per epoch).
        n = (len(indices) // self.num_replicas) * self.num_replicas
        # Stride sharding — rank r gets indices[r::num_replicas].
        return indices[self.rank:n:self.num_replicas]

    def __iter__(self) -> Iterator[List[int]]:
        n = len(self.dataset)
        order = list(range(n))
        if self.shuffle:
            rng = random.Random(self.seed + self._epoch)
            rng.shuffle(order)
        order = self._rank_slice(order)

        batch: List[int] = []
        tokens = 0
        yielded = 0
        for i in order:
            t = max(1, int(self.dataset.tokens_at_index(i)))
            if t > self.max_batch_tokens:
                # Pathologically large sample: yield as a 1-sample batch
                # (the model handles arbitrary L; this just means a smaller
                # effective batch for this iteration).
                if batch:
                    yield batch
                    yielded += 1
                    batch, tokens = [], 0
                yield [i]
                yielded += 1
                continue
            if tokens + t > self.max_batch_tokens and batch:
                yield batch
                yielded += 1
                batch, tokens = [], 0
            batch.append(i)
            tokens += t
        # ``drop_last`` discards the trailing partial batch, but a dataset that
        # fits in less than one budget must still produce a batch -- otherwise an
        # epoch would silently yield nothing and training could never progress.
        if batch and (not self.drop_last or yielded == 0):
            yield batch

    def __len__(self) -> int:
        # Approximate: total / per-rank-avg, lower bound for accelerator's
        # progress tracking. Computed without iterating (assumes uniform avg
        # token count over the dataset).
        n = len(self.dataset)
        per_rank = n // self.num_replicas
        if per_rank == 0:
            return 0
        # Sample a small batch to estimate avg token count cheaply.
        sample_n = min(per_rank, 64)
        avg = max(
            1,
            sum(self.dataset.tokens_at_index(i) for i in range(sample_n)) // sample_n,
        )
        per_batch_n = max(1, self.max_batch_tokens // avg)
        return per_rank // per_batch_n
