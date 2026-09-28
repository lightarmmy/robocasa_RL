#!/usr/bin/env bash
# Stable RoboCasa365 evaluation: one simulator worker per Slurm task/GPU.
#SBATCH --job-name=xr1-robocasa365-stable
#SBATCH --partition=yukaichenglab
#SBATCH --array=0-3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --output=/storage/yukaichengLab/lishiwen/xufanghui/xiaomi_robotics_1_RL/Xiaomi-Robotics-1/logs/xr1_robocasa365_stable_%A_%a.log

set -euo pipefail
workspace_root=/storage/yukaichengLab/lishiwen/xufanghui/xiaomi_robotics_1_RL
repo_root=${workspace_root}/Xiaomi-Robotics-1
python=${repo_root}/.conda-robocasa365/bin/python
model_path=${repo_root}/checkpoints/Xiaomi-Robotics-1-RoboCasa365
result_root=${repo_root}/eval_results/robocasa365-target50-5trials-stable
run_id=target50-5trials-stable-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}
server_log=${result_root}/server_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log
server_port=$((10086 + SLURM_ARRAY_TASK_ID))

mkdir -p "${result_root}"
cd "${repo_root}"
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0
export TOKENIZERS_PARALLELISM=false
export USE_TF=0
export USE_TORCH=1
export HF_HOME=${repo_root}/.hf_cache
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
mkdir -p "${HF_HOME}"

mapfile -t tasks < <("${python}" - <<'PY'
import os
from robocasa.utils.dataset_registry import TASK_SET_REGISTRY
shard = int(os.environ['SLURM_ARRAY_TASK_ID'])
for i, task in enumerate(TASK_SET_REGISTRY['target50']):
    if i % 4 == shard:
        print(task)
PY
)
# RoboCasa emits an import warning to stdout in this environment; never treat
# that diagnostic line as a task name.
filtered_tasks=()
for task in "${tasks[@]}"; do
  [[ "${task}" == WARNING:* ]] && continue
  filtered_tasks+=("${task}")
done
tasks=("${filtered_tasks[@]}")
if (( ${#tasks[@]} == 0 )); then echo "No tasks assigned" >&2; exit 2; fi

"${python}" -c 'import torch, transformers; print("torch",torch.__version__,"transformers",transformers.__version__); assert torch.cuda.is_available()'
"${python}" -u deploy/server.py --model "${model_path}" --host 127.0.0.1 --port "${server_port}" >"${server_log}" 2>&1 &
server_pid=$!
cleanup() { kill "${server_pid}" 2>/dev/null || true; wait "${server_pid}" 2>/dev/null || true; }
trap cleanup EXIT
for _ in $(seq 1 300); do
  if grep -q 'Server running on' "${server_log}"; then break; fi
  if ! kill -0 "${server_pid}" 2>/dev/null; then cat "${server_log}"; exit 1; fi
  sleep 2
done
grep -q 'Server running on' "${server_log}"

task_args=()
for task in "${tasks[@]}"; do task_args+=(--task-name "${task}"); done
NUM_TRIALS=5 RUN_ID="${run_id}" PYTHON="${python}" BASE_PORT="${server_port}" \
  bash scripts/launch_robocasa365.sh 1 "${result_root}" "${model_path}" \
  "${task_args[@]}" --save-videos --save-failure-videos
