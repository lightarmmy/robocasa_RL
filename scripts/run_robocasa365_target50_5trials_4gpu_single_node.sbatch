#!/usr/bin/env bash
# One Slurm job / one node: four independent GPU evaluation lanes.
#SBATCH --job-name=xr1-robocasa365-single
#SBATCH --partition=yukaichenglab
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00
#SBATCH --output=/storage/yukaichengLab/lishiwen/xufanghui/xiaomi_robotics_1_RL/Xiaomi-Robotics-1/logs/xr1_robocasa365_single_%j.log

set -euo pipefail
workspace_root=/storage/yukaichengLab/lishiwen/xufanghui/xiaomi_robotics_1_RL
repo_root=${workspace_root}/Xiaomi-Robotics-1
python=${repo_root}/.conda-robocasa365/bin/python
model_path=${repo_root}/checkpoints/Xiaomi-Robotics-1-RoboCasa365
result_root=${EVAL_RESULT_ROOT:-${repo_root}/eval_results/robocasa365-target50-5trials-single}
run_id=${EVAL_RUN_ID:-target50-5trials-single-${SLURM_JOB_ID}}
mkdir -p "${result_root}"
cd "${repo_root}"
export MUJOCO_GL=egl TOKENIZERS_PARALLELISM=false USE_TF=0 USE_TORCH=1
export HF_HOME=${repo_root}/.hf_cache OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
mkdir -p "${HF_HOME}"

"${python}" -c 'import torch, transformers; print("torch",torch.__version__,"transformers",transformers.__version__); assert torch.cuda.is_available() and torch.cuda.device_count() >= 4'

lane_pids=()
cleanup() {
  for pid in "${lane_pids[@]:-}"; do kill "${pid}" 2>/dev/null || true; done
  for pid in "${lane_pids[@]:-}"; do wait "${pid}" 2>/dev/null || true; done
}
trap cleanup EXIT

gpu_offset=${GPU_OFFSET:-0}
for local_gpu in 0 1 2 3; do
  gpu=$((gpu_offset + local_gpu))
  port=$((BASE_PORT + local_gpu))
  queue_dir=${result_root}/scheduler/${run_id}-lane${gpu}
  lane_log=${result_root}/lane-${gpu}-${SLURM_JOB_ID}.log
  mkdir -p "${result_root}/scheduler"
  # Emit a prefix so import diagnostics can never become a task name.
  mapfile -t tasks < <(SLURM_LANE="${local_gpu}" "${python}" - <<'PY' | sed -n 's/^TASK:://p'
import os
from robocasa.utils.dataset_registry import TASK_SET_REGISTRY
lane = int(os.environ['SLURM_LANE'])
for i, task in enumerate(TASK_SET_REGISTRY['target50']):
    if i % 4 == lane:
        print('TASK::' + task)
PY
  )
  if (( ${#tasks[@]} == 0 )); then echo "lane ${gpu}: no tasks" >&2; exit 2; fi
  task_args=()
  for task in "${tasks[@]}"; do task_args+=(--task-name "${task}"); done
  (
    # RoboSuite validates MUJOCO_EGL_DEVICE_ID against the literal ids in
    # CUDA_VISIBLE_DEVICES. Since each lane exposes one physical GPU, both
    # variables must use that same physical id.
    export CUDA_VISIBLE_DEVICES=${gpu} MUJOCO_EGL_DEVICE_ID=${gpu} BASE_PORT=${port}
    echo "lane=${gpu} host=$(hostname) job=${SLURM_JOB_ID} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} MUJOCO_GL=${MUJOCO_GL} MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID}"
    nvidia-smi -L || true
    server_log=${result_root}/server-${gpu}-${SLURM_JOB_ID}.log
    server_args=(--model "${model_path}" --host 127.0.0.1 --port "${port}")
    if [[ -n "${POLICY_CHECKPOINT:-}" ]]; then
      server_args+=(--policy-checkpoint "${POLICY_CHECKPOINT}")
    fi
    "${python}" -u deploy/server.py "${server_args[@]}" >"${server_log}" 2>&1 &
    server_pid=$!
    lane_cleanup() { kill "${server_pid}" 2>/dev/null || true; wait "${server_pid}" 2>/dev/null || true; }
    trap lane_cleanup EXIT
    for _ in $(seq 1 300); do
      if grep -q 'Server running on' "${server_log}"; then break; fi
      if ! kill -0 "${server_pid}" 2>/dev/null; then cat "${server_log}"; exit 1; fi
      sleep 2
    done
    grep -q 'Server running on' "${server_log}"
    NUM_TRIALS=5 RUN_ID="${run_id}-lane${gpu}" PYTHON="${python}" \
      bash scripts/launch_robocasa365.sh 1 "${result_root}" "${model_path}" \
      "${task_args[@]}" --save-videos --save-failure-videos
  ) >"${lane_log}" 2>&1 &
  lane_pids+=("$!")
done

status=0
for pid in "${lane_pids[@]}"; do wait "${pid}" || status=1; done
exit "${status}"
