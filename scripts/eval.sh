#!/usr/bin/env bash
# Usage: bash scripts/eval.sh <checkpoint.pt> <hmdb|ucf|kinetics|ssv2_small> [shot] [granularity] [span_prior]
#
# granularity and span_prior are optional and only kept for backwards
# compatibility: eval.py reads the model-shaping arguments (granularity, span
# prior, LoRA, temporal context, ...) back out of the checkpoint's own stored
# args, so evaluation cannot silently disagree with training. Passing them here
# OVERRIDES what the checkpoint says, so leave them out unless you mean it.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

CHECKPOINT="$1"
DATASET_NAME="${2:-hmdb}"
SHOT="${3:-1}"
OVERRIDES=()
[ -n "${4:-}" ] && OVERRIDES+=(--granularity "$4")
[ -n "${5:-}" ] && OVERRIDES+=(--span_prior "$5")

case "$DATASET_NAME" in
  hmdb)
    DATASET="${HMDB_ROOT:-$DATA_ROOT/hmdb_root}"
    SPLITS="$PROJECT_DIR/splits/hmdb"
    SUBTEXTS="$PROJECT_DIR/data/sub_texts/hmdb51_class_subtexts.json"
    ;;
  ucf)
    DATASET="${UCF_ROOT:-$DATA_ROOT/UCF101}"
    SPLITS="$PROJECT_DIR/splits/ucf"
    SUBTEXTS="$PROJECT_DIR/data/sub_texts/ucf101_class_subtexts.json"
    ;;
  kinetics)
    DATASET="${KINETICS_ROOT:-$DATA_ROOT/kinetics_FSAR}"
    SPLITS="$PROJECT_DIR/splits/kinetics"
    SUBTEXTS="$PROJECT_DIR/data/sub_texts/kinetics_class_subtexts.json"
    ;;
  ssv2_small)
    DATASET="${SSV2_ROOT:-$DATA_ROOT/ssv2_small_FSAR}"
    SPLITS="$PROJECT_DIR/splits/ssv2_small"
    SUBTEXTS="$PROJECT_DIR/data/sub_texts/ssv2_small_class_subtexts.json"
    ;;
  *)
    echo "Unknown dataset '$DATASET_NAME', expected hmdb, ucf, kinetics or ssv2_small" >&2
    exit 1
    ;;
esac

"$PYTHON" -u eval.py \
  --dataset "$DATASET" \
  --split_path "$SPLITS" \
  --subtext_path "$SUBTEXTS" \
  --shot "$SHOT" \
  --num_test_tasks 10000 \
  --num_workers 4 \
  -pc "$CHECKPOINT" \
  "${OVERRIDES[@]}"
