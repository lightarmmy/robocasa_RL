#!/usr/bin/env bash
# Create an isolated RoboCasa365 (RoboCasa v1.x) client environment.
set -euo pipefail

workspace_root=/storage/yukaichengLab/lishiwen/xufanghui/xiaomi_robotics_1_RL
repo_root=${workspace_root}/Xiaomi-Robotics-1
env_prefix=${repo_root}/.conda-robocasa365
conda_pkgs=${repo_root}/.conda_pkgs_robocasa365
robocasa_root=${workspace_root}/robocasa
robosuite_root=${repo_root}/third_party/robosuite
pip_index=https://mirrors.aliyun.com/pypi/simple/

mkdir -p "${conda_pkgs}"

# A failed Conda transaction can leave an unusable prefix without bin/python.
# It is not an environment that can be resumed, so remove only that partial
# prefix before retrying the create transaction.
if [[ -d "${env_prefix}" && ! -x "${env_prefix}/bin/python" ]]; then
  echo "Removing incomplete Conda prefix: ${env_prefix}"
  rm -rf "${env_prefix}"
fi

if [[ ! -x "${env_prefix}/bin/python" ]]; then
  # Retry once: Conda 26 can occasionally finish downloading package archives
  # but fail its first transaction with a stale local package-cache index.
  for attempt in 1 2; do
    if CONDA_PKGS_DIRS="${conda_pkgs}" conda create \
      -p "${env_prefix}" python=3.11 \
      -c conda-forge --override-channels -y; then
      break
    fi
    if [[ "${attempt}" -eq 2 ]]; then
      echo "Conda environment creation failed after two attempts." >&2
      exit 1
    fi
    echo "Retrying Conda environment creation after clearing the incomplete prefix..." >&2
    rm -rf "${env_prefix}"
  done
fi

py=${env_prefix}/bin/python
pip=${env_prefix}/bin/pip

"${pip}" install --no-cache-dir --upgrade pip setuptools wheel -i "${pip_index}"

# The XR-1 deployment guide pins this tested PyTorch / torchvision pair.
"${pip}" install --no-cache-dir \
  torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128

# Editable installs keep this environment on the checked-out v1.x code.
"${pip}" install --no-cache-dir --no-build-isolation -e "${robosuite_root}" -i "${pip_index}"
"${pip}" install --no-cache-dir --no-build-isolation -e "${robocasa_root}" -i "${pip_index}"
"${pip}" install --no-cache-dir \
  transformers==4.57.1 imageio[ffmpeg] tqdm scipy tyro \
  -i "${pip_index}"

"${py}" - <<'PY'
import importlib.metadata
import mujoco, numba, numpy, robocasa, robosuite, torch, torchvision
print('robocasa', importlib.metadata.version('robocasa'), robocasa.__file__)
print('robosuite', robosuite.__version__)
print('python environment is ready')
print('numpy', numpy.__version__)
print('numba', numba.__version__)
print('mujoco', mujoco.__version__)
print('torch', torch.__version__)
print('torchvision', torchvision.__version__)
print('transformers', importlib.metadata.version('transformers'))
assert numpy.__version__ == '2.2.5'
assert numba.__version__ == '0.61.2'
assert mujoco.__version__ == '3.3.1'
PY

echo "RoboCasa365 Conda environment ready: ${env_prefix}"
