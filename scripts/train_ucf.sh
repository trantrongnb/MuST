#!/usr/bin/env bash
# UCF101, 5-way K-shot. Usage: bash scripts/train_ucf.sh [shot] [granularity] [span_prior]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

SHOT="${1:-1}"
GRANULARITY="${2:-pyramid}"
SPAN_PRIOR="${3:-box}"
# Anything after the third argument is forwarded verbatim to train.py, e.g.
#   bash scripts/train_hmdb.sh 1 pyramid box --lora_rank 8   # temporal_context=transformer is the default
shift $(( $# < 3 ? $# : 3 ))

"$PYTHON" -u train.py \
  --dataset "${UCF_ROOT:-$DATA_ROOT/UCF101}" \
  --split_path "$PROJECT_DIR/splits/ucf" \
  --subtext_path "$PROJECT_DIR/data/sub_texts/ucf101_class_subtexts.json" \
  --shot "$SHOT" \
  --granularity "$GRANULARITY" \
  --span_prior "$SPAN_PRIOR" \
  --seq_len 8 \
  --optimizer adamw \
  --learning_rate 5e-5 \
  --clip_learning_rate 2e-6 \
  --lora_lr 1e-4 \
  --tasks_per_batch 2 \
  --training_iterations 10000 \
  --val_interval 1000 \
  --num_val_tasks 1000 \
  --num_workers 4 \
  "$@"
