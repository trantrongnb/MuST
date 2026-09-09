#!/usr/bin/env bash
# Secondary sweep: every granularity mode at a fixed span_prior, for the table
# that isolates contribution 1 on its own. Run it with the prior OFF so the two
# contributions do not mix.
#
# The 'full' row (M=1, all N phases in one long sentence) is the one reviewers
# ask for first -- it shows the gain comes from having SEVERAL granularities,
# not merely from a longer prompt. Do not drop it.
#
#   bash scripts/run_granularity_sweep.sh hmdb 1 off
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

DATASET_NAME="${1:-hmdb}"
SHOT="${2:-1}"
PRIOR="${3:-off}"
MODES="${MODES:-global atomic full atomic_full pairs pyramid}"

# As in run_ablation.sh: pass the run directory explicitly rather than trying to
# reconstruct the one train.py would pick.
RUN_ROOT="$PROJECT_DIR/work/$DATASET_NAME/${SHOT}-shot"

for MODE in $MODES; do
  RUN_DIR="$RUN_ROOT/sweep_${MODE}_${PRIOR}"
  echo "=== $DATASET_NAME ${SHOT}-shot | granularity=$MODE span_prior=$PRIOR ==="
  bash "$PROJECT_DIR/scripts/train_${DATASET_NAME}.sh" "$SHOT" "$MODE" "$PRIOR" \
    --checkpoint_dir "$RUN_DIR"
  bash "$PROJECT_DIR/scripts/eval.sh" "$RUN_DIR/checkpoint_best.pt" "$DATASET_NAME" "$SHOT"
done

echo
echo "=== summary ==="
grep -H "" "$RUN_ROOT"/sweep_*_"${PRIOR}"/eval_*.txt || true
