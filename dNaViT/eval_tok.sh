#!/usr/bin/env bash
# Copyright (c) 2026 LongCat dNaViT Team.
# SPDX-License-Identifier: Apache-2.0
# Evaluate ImageNet-256 reconstruction (PSNR / SSIM, optional rFID).
#
#   bash eval_tok.sh <config.yaml> <checkpoint> [data-path] [image-size]
#
#   bash eval_tok.sh configs/dnavit_stageC_anyres_g16_32x.yaml \
#       runs/dnavit_stageC_anyres_g16_32x/exp1/checkpoint-latest/unwrapped_model \
#       /path/to/imagenet/val 256
#
# Environment overrides: NPROC (GPUs, default 1), BATCH (per-GPU, default 8),
# FID=1 to also compute rFID, NUM_SAMPLES to cap the image count.
set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "usage: bash eval_tok.sh <config.yaml> <checkpoint> [data-path] [image-size]" >&2
    exit 1
fi
CONFIG="$1"
CKPT="$2"
DATA="${3:-example_data/imagenet_val}"
SIZE="${4:-256}"
NPROC="${NPROC:-1}"
BATCH="${BATCH:-8}"

EXTRA=()
if [ "${FID:-0}" = "1" ]; then EXTRA+=(--fid); fi
if [ -n "${NUM_SAMPLES:-}" ]; then EXTRA+=(--num-samples "$NUM_SAMPLES"); fi

torchrun --standalone --nproc_per_node "$NPROC" scripts/eval_recon.py \
    --config "$CONFIG" \
    --ckpt "$CKPT" \
    --data-path "$DATA" \
    --image-size "$SIZE" \
    --batch-size "$BATCH" \
    --num-workers 8 \
    --bf16 \
    "${EXTRA[@]}"
