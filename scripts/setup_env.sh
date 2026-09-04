#!/usr/bin/env bash
# Setup ContactGen Python environment at repo root (.venv-contactgen).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="${ROOT}/.venv-contactgen"
CG="${ROOT}/contactgen"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12}"
export PATH="/usr/libexec/gcc/x86_64-linux-gnu/13:${CUDA_HOME}/bin:${PATH}"

if [[ ! -d "${VENV}" ]]; then
  python3.10 -m venv "${VENV}"
fi

PIP="${VENV}/bin/pip"
PY="${VENV}/bin/python"

"${PIP}" install --upgrade pip setuptools wheel
"${PIP}" install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
"${PIP}" install "numpy<2" trimesh==3.9.8 omegaconf tensorboardX kornia opencv-python==4.5.5.64 open3d
"${PIP}" install --no-build-isolation chumpy
"${PIP}" install iopath fvcore ninja
"${PIP}" install --no-index --no-cache-dir pytorch3d \
  -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt240/download.html

# chumpy uses removed numpy scalar aliases (np.bool, etc.)
CHUMPY_INIT="$("${PY}" -c 'import chumpy, os; print(os.path.join(os.path.dirname(chumpy.__file__), "__init__.py"))')"
if grep -q 'from numpy import bool, int, float' "${CHUMPY_INIT}"; then
  "${PY}" - <<'PY'
import pathlib
p = pathlib.Path(__import__("chumpy").__file__).parent / "__init__.py"
text = p.read_text()
old = "from numpy import bool, int, float, complex, object, unicode, str, nan, inf"
new = """import numpy as np
bool = np.bool_
int = np.int_
float = np.float64
complex = np.complex128
object = np.object_
unicode = np.str_
str = np.str_
nan = np.nan
inf = np.inf"""
if old in text:
    p.write_text(text.replace(old, new))
    print("Patched chumpy for numpy>=1.24 compatibility")
PY
fi

CUDA_HOME="${CUDA_HOME}" PATH="${PATH}" TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}" \
  "${PIP}" install --no-build-isolation "${CG}/pointnet_lib"

echo "ContactGen env ready: ${VENV}"
echo "Activate: source ${VENV}/bin/activate"
echo "Demo: cd ${CG} && python demo.py --obj_path assets/toothpaste.ply --n_samples=10 --save_root exp/demo_results"
echo ""
echo "Headless inference renders (Open3D OffscreenRenderer + EGL, recommended):"
echo "  python motion_feature_knobs/grasp_type_knob/contactgen/render_inference_grasps.py"
echo "  # or: cd contactgen && python vis_grasp_headless.py --save-root exp/demo_results"
echo "Requires EGL vendor libs (usually preinstalled): libegl1, nvidia or mesa EGL vendor JSON."
echo "Legacy vis_grasp + xvfb often captures blank PNGs on GPU servers; use render_inference_grasps instead."
