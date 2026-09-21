#!/usr/bin/env bash
# Evaluate one PPO checkpoint on all RoboCasa365 target50 tasks.  Each task
# gets a fresh model server and simulator process so a native MuJoCo failure
# can be retried without invalidating the rest of the lane.
#SBATCH --job-name=xr1-ppo100-eval
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --output=/storage/yukaichengLab/lishiwen/xufanghui/xiaomi_robotics_1_RL/Xiaomi-Robotics-1/logs/xr1_ppo_policy_eval_%j.log

set -euo pipefail

workspace_root=/storage/yukaichengLab/lishiwen/xufanghui/xiaomi_robotics_1_RL
repo_root=${workspace_root}/Xiaomi-Robotics-1
python=${repo_root}/.conda-robocasa365/bin/python
model_path=${repo_root}/checkpoints/Xiaomi-Robotics-1-RoboCasa365
policy_checkpoint=${POLICY_CHECKPOINT:-${repo_root}/results/xr1_ppo_full_stable_100u_resume/xr1_robocasa365_ppo_full/checkpoints/global_step_100/actor/model_state_dict/full_weights.pt}
policy_name=${POLICY_NAME:-full_step100}
eval_split=${EVAL_SPLIT:-pretrain}
eval_seed=${EVAL_SEED:-7}
num_trials=${NUM_TRIALS:-5}
num_lanes=${NUM_LANES:-4}
eval_tasks=${EVAL_TASKS:-}
result_root=${EVAL_RESULT_ROOT:-${repo_root}/eval_results/robocasa365-${policy_name}-${SLURM_JOB_ID}}
# Multiple evaluation jobs may share one node. Give each Slurm job a disjoint
# four-port block instead of colliding on the old fixed 12086..12089 range.
port_base=${EVAL_PORT_BASE:-$((20000 + (SLURM_JOB_ID % 10000) * 4))}

# Slurm's device cgroup remaps an allocated physical subset to job-local CUDA
# ordinals 0..N-1.  Child processes must select those logical ordinals; using
# SLURM_JOB_GPUS physical ids here makes CUDA report no available devices.
slurm_gpu_allocation=${SLURM_JOB_GPUS:-unknown}

if [[ "${policy_checkpoint}" == "none" ]]; then
  policy_checkpoint=
fi
if [[ -n "${policy_checkpoint}" && ! -f "${policy_checkpoint}" ]]; then
  echo "Missing PPO policy checkpoint: ${policy_checkpoint}" >&2
  exit 2
fi
[[ "${num_trials}" =~ ^[1-9][0-9]*$ ]] || {
  echo "NUM_TRIALS must be a positive integer, got: ${num_trials}" >&2
  exit 2
}
[[ "${num_lanes}" =~ ^[1-9][0-9]*$ ]] || {
  echo "NUM_LANES must be a positive integer, got: ${num_lanes}" >&2
  exit 2
}
(( port_base >= 1024 && port_base + num_lanes - 1 <= 65535 )) || {
  echo "Invalid evaluation port range: ${port_base}-$((port_base + num_lanes - 1))" >&2
  exit 2
}

mkdir -p "${result_root}/lanes" "${result_root}/results"
cd "${repo_root}"
export MUJOCO_GL=${MUJOCO_GL:-egl} TOKENIZERS_PARALLELISM=false USE_TF=0 USE_TORCH=1
export HF_HOME=${repo_root}/.hf_cache OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1

echo "job=${SLURM_JOB_ID} host=$(hostname) policy_name=${policy_name}"
echo "policy_checkpoint=${policy_checkpoint:-PRETRAINED_BASE_MODEL}"
echo "result_root=${result_root} split=${eval_split} seed=${eval_seed} trials=${num_trials} lanes=${num_lanes}"
echo "port_range=${port_base}-$((port_base + num_lanes - 1))"
[[ -n "${eval_tasks}" ]] && echo "task_filter=${eval_tasks}"
echo "slurm_gpu_allocation=${slurm_gpu_allocation}; using logical_gpu_ids=0-$((num_lanes - 1))"
echo "mujoco_gl=${MUJOCO_GL}"
NUM_LANES="${num_lanes}" "${python}" -c 'import os, torch, transformers; print("torch", torch.__version__, "transformers", transformers.__version__); assert torch.cuda.is_available() and torch.cuda.device_count() >= int(os.environ["NUM_LANES"])'

lane_pids=()
cleanup() {
  for pid in "${lane_pids[@]:-}"; do kill "${pid}" 2>/dev/null || true; done
  for pid in "${lane_pids[@]:-}"; do wait "${pid}" 2>/dev/null || true; done
}
trap cleanup EXIT INT TERM

for lane in $(seq 0 $((num_lanes - 1))); do
  gpu=${lane}
  lane_log=${result_root}/lanes/lane-${lane}.log
  (
    set -uo pipefail
    export CUDA_VISIBLE_DEVICES=${gpu}
    if [[ "${MUJOCO_GL}" == "egl" ]]; then
      # robosuite validates that the EGL device id appears literally in
      # CUDA_VISIBLE_DEVICES. Each lane exposes one job-local Slurm ordinal,
      # so keep the two variables on the same lane id.
      export MUJOCO_EGL_DEVICE_ID=${gpu}
    else
      unset MUJOCO_EGL_DEVICE_ID
    fi
    mapfile -t tasks < <(SLURM_LANE="${lane}" NUM_LANES="${num_lanes}" EVAL_TASKS="${eval_tasks}" "${python}" - <<'PY' | sed -n 's/^TASK:://p'
import os
from robocasa.utils.dataset_registry import TASK_SET_REGISTRY

lane = int(os.environ["SLURM_LANE"])
num_lanes = int(os.environ["NUM_LANES"])
requested = {name for name in os.environ.get("EVAL_TASKS", "").split(",") if name}
all_tasks = list(TASK_SET_REGISTRY["target50"])
unknown = requested - set(all_tasks)
if unknown:
    raise SystemExit(f"Unknown target50 task(s): {sorted(unknown)}")
for index, task in enumerate(all_tasks):
    if requested and task not in requested:
        continue
    if index % num_lanes == lane:
        print("TASK::" + task)
PY
    )
    failed=0
    for task in "${tasks[@]}"; do
      task_done=0
      for attempt in 1 2 3; do
        port=$((port_base + lane))
        task_run=target50-${num_trials}trials-${policy_name}-${SLURM_JOB_ID}-lane${lane}-${task}-attempt${attempt}
        task_log=${result_root}/lanes/${task_run}.log
        server_log=${result_root}/lanes/${task_run}-server.log

        server_args=(
          --model "${model_path}"
          --host 127.0.0.1
          --port "${port}"
        )
        if [[ -n "${policy_checkpoint}" ]]; then
          server_args+=(--policy-checkpoint "${policy_checkpoint}")
        fi
        "${python}" -u deploy/server.py "${server_args[@]}" >"${server_log}" 2>&1 &
        server_pid=$!
        ready=0
        for _ in $(seq 1 180); do
          if grep -q 'Server running on' "${server_log}"; then
            ready=1
            break
          fi
          kill -0 "${server_pid}" 2>/dev/null || break
          sleep 2
        done

        task_status=1
        if (( ready )); then
          set +e
          BASE_PORT=${port} NUM_TRIALS=${num_trials} SPLIT=${eval_split} SEED=${eval_seed} \
            RUN_ID="${task_run}" PYTHON="${python}" \
            bash scripts/launch_robocasa365.sh \
              1 "${result_root}/results" "${model_path}" \
              --task-name "${task}" --save-videos --save-failure-videos \
              >"${task_log}" 2>&1
          task_status=$?
          set -e
        else
          cat "${server_log}" >>"${task_log}"
        fi

        kill "${server_pid}" 2>/dev/null || true
        wait "${server_pid}" 2>/dev/null || true
        summary=${result_root}/results/${task_run}/summary.json
        if (( task_status == 0 )) && [[ -f "${summary}" ]]; then
          echo "COMPLETED task=${task} attempt=${attempt} summary=${summary}"
          task_done=1
          break
        fi
        echo "RETRY task=${task} attempt=${attempt} status=${task_status}" >&2
        sleep 3
      done
      if (( ! task_done )); then
        echo "FAILED_TASK task=${task}" >&2
        failed=1
      fi
    done
    exit "${failed}"
  ) >"${lane_log}" 2>&1 &
  lane_pids+=("$!")
done

status=0
for pid in "${lane_pids[@]}"; do
  wait "${pid}" || status=1
done

POLICY_NAME="${policy_name}" POLICY_CHECKPOINT="${policy_checkpoint}" \
EVAL_SPLIT="${eval_split}" EVAL_SEED="${eval_seed}" NUM_TRIALS="${num_trials}" EVAL_STATUS="${status}" \
EVAL_TASKS="${eval_tasks}" \
"${python}" - "${result_root}" <<'PY'
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for path in sorted((root / "results").glob("*/summary.json")):
    try:
        data = json.loads(path.read_text())
    except Exception:
        continue
    rows.append({"path": str(path), "summary": data})

episodes = sum(int(row["summary"].get("num_episodes", 0)) for row in rows)
successes = sum(int(row["summary"].get("successes", 0)) for row in rows)
trials = int(os.environ["NUM_TRIALS"])
requested = [name for name in os.environ.get("EVAL_TASKS", "").split(",") if name]
expected_tasks = len(requested) if requested else 50
out = {
    "policy": os.environ["POLICY_NAME"],
    "policy_checkpoint": os.environ["POLICY_CHECKPOINT"] or None,
    "split": os.environ["EVAL_SPLIT"],
    "seed": int(os.environ["EVAL_SEED"]),
    "num_trials_per_task": trials,
    "completed_tasks": len(rows),
    "expected_tasks": expected_tasks,
    "episodes": episodes,
    "expected_episodes": expected_tasks * trials,
    "successes": successes,
    "success_rate": successes / episodes if episodes else 0.0,
    "task_summaries": rows,
}
(root / "aggregate.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps({k: out[k] for k in (
    "policy", "completed_tasks", "episodes", "successes", "success_rate"
)}))

complete = len(rows) == expected_tasks and episodes == expected_tasks * trials
sys.exit(0 if int(os.environ.get("EVAL_STATUS", "1")) == 0 and complete else 1)
PY
