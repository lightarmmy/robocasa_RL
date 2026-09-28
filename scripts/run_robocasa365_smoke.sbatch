#!/usr/bin/env bash
# One-task RoboCasa365 smoke evaluation using the released RoboCasa365 model.
#SBATCH --job-name=xr1-robocasa365-smoke
#SBATCH --partition=yukaichenglab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=/storage/yukaichengLab/lishiwen/xufanghui/xiaomi_robotics_1_RL/Xiaomi-Robotics-1/logs/xr1_robocasa365_smoke_%j.log

set -euo pipefail

workspace_root=/storage/yukaichengLab/lishiwen/xufanghui/xiaomi_robotics_1_RL
repo_root=${workspace_root}/Xiaomi-Robotics-1
server_python=${repo_root}/.conda-robocasa365/bin/python
client_python=${repo_root}/.conda-robocasa365/bin/python
model_path=${repo_root}/checkpoints/Xiaomi-Robotics-1-RoboCasa365
result_root=${repo_root}/eval_results/robocasa365-smoke
server_log=${result_root}/server_${SLURM_JOB_ID}.log

mkdir -p "${result_root}"
cd "${repo_root}"
export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false
export USE_TF=0
export USE_TORCH=1
export HF_HOME=${repo_root}/.hf_cache
mkdir -p "${HF_HOME}"

"${server_python}" -c 'import torch, transformers; print("server torch", torch.__version__, "transformers", transformers.__version__); assert torch.cuda.is_available()'
"${client_python}" -c 'import mujoco, robocasa, robosuite, transformers; print("client mujoco", mujoco.__version__, "robocasa", robocasa.__version__, "robosuite", robosuite.__version__, "transformers", transformers.__version__)'

cleanup() {
  if [[ -n "${server_pid:-}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
    kill "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

"${server_python}" -u deploy/server.py --model "${model_path}" --host 127.0.0.1 --port 10086 >"${server_log}" 2>&1 &
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

CONDA_ENV= PYTHON="${client_python}" NUM_TRIALS=1 RUN_ID="smoke-${SLURM_JOB_ID}" \
  bash scripts/launch_robocasa365.sh \
  1 "${result_root}" "${model_path}" \
  --task-name CloseBlenderLid --horizon 20
