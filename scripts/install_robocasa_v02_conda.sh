#!/usr/bin/env bash
set -euo pipefail

workspace_root=/storage/yukaichengLab/lishiwen/xufanghui/xiaomi_robotics_1_RL
repo_root=${workspace_root}/Xiaomi-Robotics-1
env_prefix=${repo_root}/.conda-robocasa-v02
conda_pkgs=${repo_root}/.conda_pkgs_v02
upload_root=${workspace_root}/upload

mkdir -p "${conda_pkgs}"

if [[ -e "${env_prefix}/pyvenv.cfg" ]]; then
  echo "Refusing to overwrite an existing venv at ${env_prefix}" >&2
  exit 2
fi

if [[ ! -x "${env_prefix}/bin/python" ]]; then
  CONDA_PKGS_DIRS="${conda_pkgs}" conda create \
    -p "${env_prefix}" python=3.10 \
    -c conda-forge --override-channels -y
fi

py=${env_prefix}/bin/python
pip=${env_prefix}/bin/pip

"${pip}" install --upgrade pip setuptools wheel \
  -i https://mirrors.aliyun.com/pypi/simple/

"${pip}" install --prefer-binary \
  numpy==1.23.3 \
  numba==0.56.4 \
  llvmlite==0.39.1 \
  scipy pygame Pillow opencv-python pyyaml pynput tqdm termcolor \
  imageio imageio-ffmpeg h5py lxml hidapi \
  tianshou==0.4.10 \
  transformers==4.57.1 tokenizers==0.22.1 tyro torchvision \
  -i https://mirrors.aliyun.com/pypi/simple/

mujoco_wheel=$(find "${upload_root}" -maxdepth 3 -type f \
  -name 'mujoco-3.2.6-cp310-cp310-*.whl' -print -quit)
if [[ -n "${mujoco_wheel}" ]]; then
  "${pip}" install --no-deps "${mujoco_wheel}"
else
  "${pip}" install mujoco==3.2.6 \
    -i https://mirrors.aliyun.com/pypi/simple/
fi

"${py}" - <<'PY'
import sys, numpy, mujoco, numba
print('python', sys.version)
print('numpy', numpy.__version__)
print('mujoco', mujoco.__version__)
print('numba', numba.__version__)
assert sys.version_info[:2] == (3, 10)
assert numpy.__version__ == '1.23.3'
assert mujoco.__version__ == '3.2.6'
assert numba.__version__ == '0.56.4'
PY

echo "Conda environment ready: ${env_prefix}"
