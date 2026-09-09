#!/usr/bin/env bash
# Main ablation: a 2x2 grid over the two contributions, plus the control that
# tells temporal alignment apart from generic attention regularisation.
#
#                    | prior off | prior box
#   atomic  (M=4)    |     .     |     .        <- ~ a bag of atomic sub-texts
#   pyramid (M=10)   |     .     |     .        <- full method
#   pyramid + random (span centres permuted)    <- control
#
# The catalog is fixed at N sub-texts throughout, so only M changes across rows:
# the sub-text CONTENT is identical, which is what makes the comparison clean.
#
#   bash scripts/run_ablation.sh hmdb 1
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

DATASET_NAME="${1:-hmdb}"
SHOT="${2:-1}"
# "granularity:span_prior" pairs.
RUNS="${RUNS:-atomic:off atomic:box pyramid:off pyramid:box pyramid:random}"

# train.py derives its own checkpoint directory from the backbone configuration
# as well as the granularity and prior, so the path cannot be reconstructed here
# from the two loop variables alone. Pass it explicitly instead: --checkpoint_dir
# is forwarded to train.py, and eval.sh is then handed the same path.
RUN_ROOT="$PROJECT_DIR/work/$DATASET_NAME/${SHOT}-shot"

for RUN in $RUNS; do
  GRANULARITY="${RUN%%:*}"
  PRIOR="${RUN##*:}"
  RUN_DIR="$RUN_ROOT/ablation_${GRANULARITY}_${PRIOR}"
  echo "=== training $DATASET_NAME ${SHOT}-shot | granularity=$GRANULARITY span_prior=$PRIOR ==="
  bash "$PROJECT_DIR/scripts/train_${DATASET_NAME}.sh" "$SHOT" "$GRANULARITY" "$PRIOR" \
    --checkpoint_dir "$RUN_DIR"

  echo "=== evaluating $RUN_DIR/checkpoint_best.pt ==="
  bash "$PROJECT_DIR/scripts/eval.sh" "$RUN_DIR/checkpoint_best.pt" "$DATASET_NAME" "$SHOT"
done

echo
echo "=== ablation summary ==="
grep -H "" "$RUN_ROOT"/ablation_*/eval_*.txt || true

echo
echo "=== learned gate of the full method ==="
"$PYTHON" "$PROJECT_DIR/scripts/plot_span_prior.py" \
  -pc "$RUN_ROOT/ablation_pyramid_box/checkpoint_best.pt" || true
