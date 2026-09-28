#!/usr/bin/env bash
# Robust two-policy RoboCasa365 evaluation. Each task gets a fresh server and
# worker process so a native MuJoCo abort cannot invalidate the whole lane.
#SBATCH --job-name=xr1-robocasa365-2policy-iso
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=72:00:00
#SBATCH --output=/storage/yukaichengLab/lishiwen/xufanghui/xiaomi_robotics_1_RL/Xiaomi-Robotics-1/logs/xr1_robocasa365_2policy_iso_%j.log

set -euo pipefail
workspace_root=/storage/yukaichengLab/lishiwen/xufanghui/xiaomi_robotics_1_RL
repo_root=${workspace_root}/Xiaomi-Robotics-1
python=${repo_root}/.conda-robocasa365/bin/python
model_path=${repo_root}/checkpoints/Xiaomi-Robotics-1-RoboCasa365
result_root=${repo_root}/eval_results/robocasa365-two-policy-isolated-${SLURM_JOB_ID}
action_expert_ckpt=${ACTION_EXPERT_CKPT:-${repo_root}/results/xr1_ppo_stable_lr1e9_10u/xr1_robocasa365_ppo_full/checkpoints/global_step_10/actor/model_state_dict/full_weights.pt}
full_ckpt=${FULL_CKPT:-${repo_root}/results/xr1_ppo_full_test/xr1_robocasa365_ppo_full/checkpoints/global_step_10/actor/model_state_dict/full_weights.pt}
mkdir -p "${result_root}"
cd "${repo_root}"
export MUJOCO_GL=egl TOKENIZERS_PARALLELISM=false USE_TF=0 USE_TORCH=1
export HF_HOME=${repo_root}/.hf_cache OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1

[[ -f "${action_expert_ckpt}" ]] || { echo "Missing action-expert checkpoint: ${action_expert_ckpt}" >&2; exit 2; }
[[ -f "${full_ckpt}" ]] || { echo "Missing full checkpoint: ${full_ckpt}" >&2; exit 2; }

run_policy() {
  local policy_name=$1
  local gpu_start=$2
  local checkpoint=$3
  local policy_root=${result_root}/${policy_name}
  mkdir -p "${policy_root}/lanes"

  for local_gpu in 0 1 2 3; do
    local gpu=$((gpu_start + local_gpu))
    local lane_log=${policy_root}/lanes/lane-${gpu}.log
    (
      set -uo pipefail
      export CUDA_VISIBLE_DEVICES=${gpu} MUJOCO_EGL_DEVICE_ID=${gpu}
      mapfile -t tasks < <(SLURM_LANE="${local_gpu}" "${python}" - <<'PY' | sed -n 's/^TASK:://p'
import os
from robocasa.utils.dataset_registry import TASK_SET_REGISTRY
lane = int(os.environ["SLURM_LANE"])
for i, task in enumerate(TASK_SET_REGISTRY["target50"]):
    if i % 4 == lane:
        print("TASK::" + task)
PY
      )
      failed=0
      for task in "${tasks[@]}"; do
        task_done=0
        for attempt in 1 2 3; do
          port=$((11086 + gpu))
          task_run=target50-5trials-${policy_name}-${SLURM_JOB_ID}-lane${gpu}-${task}-attempt${attempt}
          task_log=${policy_root}/lanes/${task_run}.log
          server_log=${policy_root}/lanes/${task_run}-server.log
          task_root=${policy_root}/results
          mkdir -p "${task_root}"
          "${python}" -u deploy/server.py --model "${model_path}" --policy-checkpoint "${checkpoint}" \
            --host 127.0.0.1 --port "${port}" >"${server_log}" 2>&1 &
          server_pid=$!
          ready=0
          for _ in $(seq 1 180); do
            if grep -q 'Server running on' "${server_log}"; then ready=1; break; fi
            kill -0 "${server_pid}" 2>/dev/null || break
            sleep 2
          done
          task_status=1
          if (( ready )); then
            set +e
            BASE_PORT=${port} NUM_TRIALS=5 RUN_ID="${task_run}" PYTHON="${python}" \
              bash scripts/launch_robocasa365.sh 1 "${task_root}" "${model_path}" \
              --task-name "${task}" --save-videos --save-failure-videos >"${task_log}" 2>&1
            task_status=$?
            set -e
          else
            cat "${server_log}" >>"${task_log}"
          fi
          kill "${server_pid}" 2>/dev/null || true
          wait "${server_pid}" 2>/dev/null || true
          if (( task_status == 0 )) && [[ -f "${task_root}/${task_run}/summary.json" ]]; then
            echo "COMPLETED task=${task} attempt=${attempt} summary=${task_root}/${task_run}/summary.json"
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
  done
  wait
}

run_policy action_expert 0 "${action_expert_ckpt}" &
pid_action=$!
run_policy full 4 "${full_ckpt}" &
pid_full=$!
status=0
wait "${pid_action}" || status=1
wait "${pid_full}" || status=1

EVAL_STATUS="${status}" "${python}" - "${result_root}" <<'PY'
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
for policy in ("action_expert", "full"):
    rows = []
    for path in sorted((root / policy / "results").glob("*/summary.json")):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        rows.append({"path": str(path), "summary": data})
    episodes = sum(int(row["summary"].get("num_episodes", 0)) for row in rows)
    successes = sum(int(row["summary"].get("successes", 0)) for row in rows)
    out = {
        "policy": policy,
        "completed_tasks": len(rows),
        "expected_tasks": 50,
        "episodes": episodes,
        "expected_episodes": 250,
        "successes": successes,
        "success_rate": successes / episodes if episodes else 0.0,
        "task_summaries": rows,
    }
    (root / f"{policy}_aggregate.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in ("policy", "completed_tasks", "episodes", "successes", "success_rate")}))
sys.exit(int(os.environ.get("EVAL_STATUS", "1")))
PY
