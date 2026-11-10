#!/usr/bin/env bash
# Copyright (c) 2026 LongCat dNaViT Team.
# SPDX-License-Identifier: Apache-2.0
# Train the dNaViT tokenizer on N GPUs.
#
#   bash run_tok.sh                                     # stage A (default config)
#   bash run_tok.sh configs/dnavit_stageB_anyres_g16_32x.yaml
#   NPROC=4 bash run_tok.sh configs/dnavit_stageC_anyres_g16_32x.yaml \
#       experiment.init_weight_path=runs/dnavit_stageB_anyres_g16_32x/exp1/checkpoint-latest/unwrapped_model
#
# Any extra arguments are passed through as OmegaConf dotlist overrides.
set -euo pipefail

CONFIG="${1:-configs/dnavit_stageA_anyres_32x.yaml}"
if [ "$#" -gt 0 ]; then shift; fi

NPROC="${NPROC:-8}"
MASTER_PORT="${MASTER_PORT:-9999}"
WANDB_MODE="${WANDB_MODE:-offline}"

export WANDB_MODE
accelerate launch --num_machines 1 --num_processes "$NPROC" \
    --machine_rank 0 --main_process_ip 127.0.0.1 \
    --main_process_port "$MASTER_PORT" --same_network \
    scripts/train_tok.py \
    config="$CONFIG" "$@"
