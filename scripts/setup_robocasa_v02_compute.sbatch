#!/usr/bin/env bash
# Install an isolated RoboCasa v0.2 client environment on a compute node.
# This script is intentionally separate from the shared gr00tRL environment.
#SBATCH --job-name=xr1-robocasa-env
#SBATCH --partition=yukaichenglab
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=Xiaomi-Robotics-1/logs/xr1_robocasa_env_%j.log

set -euo pipefail

workspace_root=/storage/yukaichengLab/lishiwen/xufanghui/xiaomi_robotics_1_RL
repo_root=${workspace_root}/Xiaomi-Robotics-1
env_prefix=${repo_root}/.env-robocasa-v02

echo "host=$(hostname)"
echo "date=$(date '+%F %T %Z')"
echo "--- DNS ---"
getent hosts pypi.org || true
getent hosts pypi.tuna.tsinghua.edu.cn || true
getent hosts mirrors.aliyun.com || true
echo "--- HTTPS ---"
curl -ILsS --connect-timeout 8 --max-time 20 -o /dev/null -w 'pypi=%{http_code}\n' https://pypi.org/simple/ || true
curl -ILsS --connect-timeout 8 --max-time 20 -o /dev/null -w 'tsinghua=%{http_code}\n' https://pypi.tuna.tsinghua.edu.cn/simple/ || true
curl -ILsS --connect-timeout 8 --max-time 20 -o /dev/null -w 'aliyun=%{http_code}\n' https://mirrors.aliyun.com/pypi/simple/ || true

if [[ ! -x "${env_prefix}/bin/python" ]]; then
  conda create -p "${env_prefix}" python=3.10.20 -y
fi

py="${env_prefix}/bin/python"
"${py}" -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
"${py}" -m pip install \
  numpy==1.23.3 numba==0.56.4 scipy pygame Pillow opencv-python pyyaml \
  pynput tqdm termcolor imageio h5py lxml hidapi tianshou==0.4.10 \
  transformers==4.57.1 tyro \
  -i https://pypi.tuna.tsinghua.edu.cn/simple
"${py}" -m pip install mujoco==3.2.6 -i https://pypi.tuna.tsinghua.edu.cn/simple

"${py}" - <<'PY'
import sys, numpy, mujoco, numba
print('python', sys.version)
print('numpy', numpy.__version__)
print('mujoco', mujoco.__version__)
print('numba', numba.__version__)
PY
