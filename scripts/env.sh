# Shared environment for every run script. Source it, do not execute it.
#
# Everything here is overridable from the outside, so nothing needs editing to
# run this repository on another machine:
#
#   PYTHON     interpreter to use                 (default: python3 on PATH)
#   DATA_ROOT  parent directory of the datasets   (default: ./data/datasets)
#
#   PYTHON=~/miniconda3/envs/must/bin/python DATA_ROOT=/mnt/datasets \
#     bash scripts/train_hmdb.sh 1
#
# A single dataset can be pointed elsewhere with HMDB_ROOT, UCF_ROOT,
# KINETICS_ROOT or SSV2_ROOT; see README.md.
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

PYTHON="${PYTHON:-python3}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_DIR/data/datasets}"
export MUST_DATA_ROOT="$DATA_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Some CUDA installations refuse to initialise unless CUDA_MPS_PIPE_DIRECTORY
# points at a writable directory. Harmless to set when MPS is not in use; unset
# MUST_SET_MPS_DIR=0 to skip it.
if [ "${MUST_SET_MPS_DIR:-1}" = "1" ] && [ -z "${CUDA_MPS_PIPE_DIRECTORY:-}" ]; then
  CUDA_MPS_PIPE_DIRECTORY="${TMPDIR:-/tmp}/must_mps_$(id -u)"
  mkdir -p "$CUDA_MPS_PIPE_DIRECTORY"
  export CUDA_MPS_PIPE_DIRECTORY
fi
