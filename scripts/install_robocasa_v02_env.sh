#!/usr/bin/env bash
set -euo pipefail

workspace_root=/storage/yukaichengLab/lishiwen/xufanghui/xiaomi_robotics_1_RL
repo_root=${workspace_root}/Xiaomi-Robotics-1
env_prefix=${repo_root}/.env-robocasa-v02
python310=/storage/yukaichengLab/lishiwen/miniconda3/envs/gr00t16/bin/python
upload_root=${workspace_root}/upload

if [[ ! -x "${python310}" ]]; then
  echo "Missing Python 3.10 interpreter: ${python310}" >&2
  exit 1
fi

echo "Using base interpreter: ${python310}"
"${python310}" -c 'import sys; print(sys.version); assert sys.version_info[:2] == (3,10)'

if [[ ! -x "${env_prefix}/bin/python" ]]; then
  "${python310}" -m venv --copies "${env_prefix}"
fi
py=${env_prefix}/bin/python
pip=${env_prefix}/bin/pip

"${py}" -m pip install --upgrade pip setuptools wheel \
  -i https://mirrors.aliyun.com/pypi/simple/

# RoboCasa v0.2 hard requirements and XR-1 client dependencies.
"${pip}" install --prefer-binary \
  numpy==1.23.3 \
  numba==0.56.4 \
  llvmlite==0.39.1 \
  scipy \
  pygame \
  Pillow \
  opencv-python \
  pyyaml \
  pynput \
  tqdm \
  termcolor \
  imageio imageio-ffmpeg \
  h5py lxml hidapi \
  tianshou==0.4.10 \
  transformers==4.57.1 \
  tokenizers==0.22.1 \
  tyro \
  -i https://mirrors.aliyun.com/pypi/simple/

# Install the Python-3.10 Linux MuJoCo wheel uploaded by the user.
mujoco_wheel=$(find "${upload_root}" -maxdepth 2 -type f \
  -name 'mujoco-3.2.6-cp310-cp310-*.whl' -print -quit)
if [[ -z "${mujoco_wheel}" ]]; then
  echo "Missing mujoco-3.2.6 cp310 Linux wheel under ${upload_root}" >&2
  echo "Expected: mujoco-3.2.6-cp310-cp310-manylinux*.whl" >&2
  exit 2
fi
"${pip}" install --no-deps "${mujoco_wheel}"

"${py}" - <<'PY'
import sys
import numpy, mujoco, numba, transformers, tokenizers
print('python', sys.version)
print('numpy', numpy.__version__)
print('mujoco', mujoco.__version__)
print('numba', numba.__version__)
print('transformers', transformers.__version__)
print('tokenizers', tokenizers.__version__)
assert sys.version_info[:2] == (3, 10)
assert numpy.__version__ == '1.23.3'
assert mujoco.__version__ == '3.2.6'
assert numba.__version__ == '0.56.4'
PY

echo "Environment ready: ${env_prefix}"
