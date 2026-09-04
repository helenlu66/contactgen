#!/usr/bin/env bash
# Run ContactGen grasp generation demo.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="${ROOT}/.venv-contactgen"
CG="${ROOT}/contactgen"

if [[ ! -x "${VENV}/bin/python" ]]; then
  echo "Missing ${VENV}. Run: bash contactgen/scripts/setup_env.sh" >&2
  exit 1
fi

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12}"
export PATH="/usr/libexec/gcc/x86_64-linux-gnu/13:${CUDA_HOME}/bin:${PATH}"

cd "${CG}"
exec "${VENV}/bin/python" demo.py \
  --obj_path "${1:-assets/toothpaste.ply}" \
  --n_samples "${2:-10}" \
  --save_root "${3:-exp/demo_results}"
