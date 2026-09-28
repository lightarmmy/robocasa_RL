#!/usr/bin/env bash
#SBATCH --job-name=xr1-robocasa-smoke
#SBATCH --partition=yukaichenglab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=Xiaomi-Robotics-1/logs/xr1_robocasa_smoke_%j.log

set -euo pipefail

repo_root=/storage/yukaichengLab/lishiwen/xufanghui/xiaomi_robotics_1_RL/Xiaomi-Robotics-1
workspace_root=/storage/yukaichengLab/lishiwen/xufanghui/xiaomi_robotics_1_RL
server_python=/storage/yukaichengLab/lishiwen/miniconda3/envs/gr00tRL/bin/python
client_python=${repo_root}/.conda-robocasa-v02/bin/python
model_path=${repo_root}/checkpoints/Xiaomi-Robotics-1-RoboCasa
result_root=${repo_root}/eval_runs/smoke_${SLURM_JOB_ID}
server_log=${result_root}/server.log
server_vendor=${repo_root}/.xr1_server_vendor

mkdir -p "${result_root}"
mkdir -p "${server_vendor}"
ln -sfn "${repo_root}/.xr1_vendor/transformers" "${server_vendor}/transformers"
ln -sfn "${repo_root}/.xr1_vendor/tokenizers" "${server_vendor}/tokenizers"
ln -sfn "${repo_root}/.xr1_vendor/tokenizers-0.22.1.dist-info" "${server_vendor}/tokenizers-0.22.1.dist-info"
export NUMBA_CACHE_DIR="${repo_root}/.numba_cache"
mkdir -p "${NUMBA_CACHE_DIR}"
export TOKENIZERS_PARALLELISM=false
export USE_TF=0
export USE_TORCH=1

cd "${repo_root}"
"${server_python}" -c 'import torch, transformers, tokenizers; print("server torch", torch.__version__, "transformers", transformers.__version__, "tokenizers", tokenizers.__version__); assert torch.cuda.is_available()'
"${client_python}" -c 'import numpy, mujoco, numba, robocasa, robosuite; print("client numpy", numpy.__version__, "mujoco", mujoco.__version__, "numba", numba.__version__, "robosuite", robosuite.__version__)'

cleanup() {
  if [[ -n "${server_pid:-}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
    kill "${server_pid}"
    wait "${server_pid}" || true
  fi
}
trap cleanup EXIT

PYTHONPATH="${server_vendor}" "${server_python}" -u deploy/server.py --model "${model_path}" --port 10086 > "${server_log}" 2>&1 &
server_pid=$!

for _ in $(seq 1 180); do
  if grep -q 'Server running on' "${server_log}"; then
    break
  fi
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    cat "${server_log}"
    exit 1
  fi
  sleep 2
done
grep -q 'Server running on' "${server_log}"

"${client_python}" -u eval_robocasa/main.py \
  --task-name OpenDrawer \
  --model-path "${model_path}" \
  --num-eval-episodes 1 \
  --start-seed 0 \
  --server-port 10086 \
  --save-root-dir "${result_root}/results"
