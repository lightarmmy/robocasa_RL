#!/usr/bin/env bash
# One 4-GPU Slurm job. Each lane evaluates one task at a time and restarts all
# native state between tasks; a SIGABRT therefore affects only one task.
#SBATCH --job-name=xr1-robocasa365-isolated
#SBATCH --partition=yukaichenglab
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00
#SBATCH --output=/storage/yukaichengLab/lishiwen/xufanghui/xiaomi_robotics_1_RL/Xiaomi-Robotics-1/logs/xr1_robocasa365_isolated_%j.log

set -euo pipefail
workspace_root=/storage/yukaichengLab/lishiwen/xufanghui/xiaomi_robotics_1_RL
repo_root=${workspace_root}/Xiaomi-Robotics-1
python=${repo_root}/.conda-robocasa365/bin/python
model_path=${repo_root}/checkpoints/Xiaomi-Robotics-1-RoboCasa365
result_root=${repo_root}/eval_results/robocasa365-target50-5trials-isolated
run_id=target50-5trials-isolated-${SLURM_JOB_ID}
mkdir -p "${result_root}/lanes"
cd "${repo_root}"
export MUJOCO_GL=egl TOKENIZERS_PARALLELISM=false USE_TF=0 USE_TORCH=1
export HF_HOME=${repo_root}/.hf_cache OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
mkdir -p "${HF_HOME}"

"${python}" -c 'import torch; assert torch.cuda.is_available() and torch.cuda.device_count() >= 4'
lane_pids=()
cleanup() { for pid in "${lane_pids[@]:-}"; do kill "${pid}" 2>/dev/null || true; done; }
trap cleanup EXIT

for gpu in 0 1 2 3; do
  lane_log=${result_root}/lanes/lane-${gpu}-${SLURM_JOB_ID}.log
  (
    set -uo pipefail
    export CUDA_VISIBLE_DEVICES=${gpu} MUJOCO_EGL_DEVICE_ID=0
    mapfile -t tasks < <(SLURM_LANE="${gpu}" "${python}" - <<'PY' | sed -n 's/^TASK:://p'
import os
from robocasa.utils.dataset_registry import TASK_SET_REGISTRY
lane = int(os.environ['SLURM_LANE'])
for i, task in enumerate(TASK_SET_REGISTRY['target50']):
    if i % 4 == lane:
        print('TASK::' + task)
PY
    )
    failed=0
    for task in "${tasks[@]}"; do
      task_done=0
      for attempt in 1 2 3; do
        port=$((11000 + gpu))
        task_run=${run_id}-lane${gpu}-${task}-attempt${attempt}
        task_log=${result_root}/lanes/${task_run}.log
        server_log=${result_root}/lanes/${task_run}-server.log
        "${python}" -u deploy/server.py --model "${model_path}" --host 127.0.0.1 --port "${port}" >"${server_log}" 2>&1 &
        server_pid=$!
        ready=0
        for _ in $(seq 1 180); do
          if grep -q 'Server running on' "${server_log}"; then ready=1; break; fi
          kill -0 "${server_pid}" 2>/dev/null || break
          sleep 2
        done
        if (( ready )); then
          set +e
          BASE_PORT=${port} NUM_TRIALS=5 RUN_ID="${task_run}" PYTHON="${python}" \
            bash scripts/launch_robocasa365.sh 1 "${result_root}" "${model_path}" \
            --task-name "${task}" --save-videos --save-failure-videos >"${task_log}" 2>&1
          task_status=$?
          set -e
        else
          task_status=1
          cat "${server_log}" >>"${task_log}"
        fi
        kill "${server_pid}" 2>/dev/null || true
        wait "${server_pid}" 2>/dev/null || true
        if (( task_status == 0 )) && [[ -f "${result_root}/${task_run}/summary.json" ]]; then
          echo "completed ${task} attempt ${attempt}"
          task_done=1
          break
        fi
        echo "retry ${task}: attempt ${attempt} failed with status ${task_status}" >&2
        sleep 5
      done
      if (( ! task_done )); then
        echo "FAILED_TASK ${task}" >&2
        failed=1
      fi
    done
    exit "${failed}"
  ) >"${lane_log}" 2>&1 &
  lane_pids+=("$!")
done

status=0
for pid in "${lane_pids[@]}"; do wait "${pid}" || status=1; done
exit "${status}"
