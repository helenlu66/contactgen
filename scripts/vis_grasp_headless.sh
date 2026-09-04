#!/usr/bin/env bash
# Run contactgen/vis_grasp.py on a headless remote server via Xvfb.
#
# NOTE: On many headless GPU servers (including this one), legacy Visualizer capture
# returns blank PNGs even when create_window() succeeds. Prefer:
#   python motion_feature_knobs/grasp_type_knob/contactgen/render_inference_grasps.py
# which uses Open3D OffscreenRenderer + EGL (no xvfb).
#
# If you still want this path: sudo apt-get install -y xvfb libosmesa6
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="${ROOT}/.venv-contactgen"
CG="${ROOT}/contactgen"

if [[ ! -x "${VENV}/bin/python" ]]; then
  echo "Missing ${VENV}. Run: bash contactgen/scripts/setup_env.sh" >&2
  exit 1
fi

if ! command -v xvfb-run >/dev/null 2>&1; then
  echo "xvfb-run not found. Install with: sudo apt-get install -y xvfb libosmesa6" >&2
  exit 1
fi

HAND_PATH="${1:-exp/demo_results/grasp_0.obj}"
OBJ_PATH="${2:-assets/toothpaste.ply}"
SAVE_PATH="${3:-exp/demo_results/grasp_0.png}"

cd "${CG}"
exec xvfb-run -a -s "-screen 0 1920x1080x24 +extension GLX +render -noreset" \
  "${VENV}/bin/python" vis_grasp.py \
  --hand_path "${HAND_PATH}" \
  --obj_path "${OBJ_PATH}" \
  --save_path "${SAVE_PATH}"
