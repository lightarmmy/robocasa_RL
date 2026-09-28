#!/usr/bin/env bash
# RoboCasa365 target50 evaluation: 50 tasks x 5 trials, four model servers.
#SBATCH --job-name=xr1-robocasa365-5x4
#SBATCH --partition=yukaichenglab
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00
#SBATCH --output=/storage/yukaichengLab/lishiwen/xufanghui/xiaomi_robotics_1_RL/Xiaomi-Robotics-1/logs/xr1_robocasa365_target50_5trials_%j.log

set -euo pipefail

workspace_root=/storage/yukaichengLab/lishiwen/xufanghui/xiaomi_robotics_1_RL
repo_root=${workspace_root}/Xiaomi-Robotics-1
python=${repo_root}/.conda-robocasa365/bin/python
model_path=${repo_root}/checkpoints/Xiaomi-Robotics-1-RoboCasa365
result_root=${repo_root}/eval_results/robocasa365-target50-5trials
run_id=target50-5trials-${SLURM_JOB_ID}
server_dir=${result_root}/servers/${SLURM_JOB_ID}

mkdir -p "${server_dir}" "${result_root}"
cd "${repo_root}"
export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false
export USE_TF=0
export USE_TORCH=1
export HF_HOME=${repo_root}/.hf_cache
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
mkdir -p "${HF_HOME}"

"${python}" -c 'import torch, transformers; print("torch",torch.__version__,"transformers",transformers.__version__); assert torch.cuda.is_available() and torch.cuda.device_count() >= 4'
"${python}" -c 'import mujoco, robocasa, robosuite; print("mujoco",mujoco.__version__,"robocasa",robocasa.__version__,"robosuite",robosuite.__version__)'

server_pids=()
cleanup() {
  for pid in "${server_pids[@]:-}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
      wait "${pid}" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT

for gpu in 0 1 2 3; do
  port=$((10086 + gpu))
  log=${server_dir}/server-${gpu}.log
  CUDA_VISIBLE_DEVICES=${gpu} HF_HOME="${HF_HOME}" "${python}" -u deploy/server.py \
    --model "${model_path}" --host 127.0.0.1 --port "${port}" >"${log}" 2>&1 &
  server_pids+=("$!")
done

for gpu in 0 1 2 3; do
  log=${server_dir}/server-${gpu}.log
  for _ in $(seq 1 300); do
    if grep -q 'Server running on' "${log}"; then break; fi
    if ! kill -0 "${server_pids[$gpu]}" 2>/dev/null; then
      cat "${log}"
      exit 1
    fi
    sleep 2
  done
  grep -q 'Server running on' "${log}"
done

NUM_TRIALS=5 RUN_ID="${run_id}" PYTHON="${python}" HF_HOME="${HF_HOME}" EGL_DEVICE_PER_WORKER=1 \
  bash scripts/launch_robocasa365.sh \
  4 "${result_root}" "${model_path}"
