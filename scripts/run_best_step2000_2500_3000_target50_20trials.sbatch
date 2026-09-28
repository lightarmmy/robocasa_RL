#!/usr/bin/env bash
# Select the best fixed five-seed score among step2000/2500/3000. If a new
# checkpoint wins, evaluate it on the complete fixed 20-seed manifest.
#SBATCH --job-name=xr1-best2k3k-eval20
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=36:00:00
#SBATCH --exclude=gnho006,gnho011,gnho020
#SBATCH --output=/storage/yukaichengLab/lishiwen/xufanghui/xiaomi_robotics_1_RL/Xiaomi-Robotics-1/logs/xr1_best_step2000_3000_eval20_%j.log

set -euo pipefail
: "${SCREEN_JOB_ID:?SCREEN_JOB_ID must identify the completed screen job}"

repo_root=/storage/yukaichengLab/lishiwen/xufanghui/xiaomi_robotics_1_RL/Xiaomi-Robotics-1
train_root=${repo_root}/results/xr1_ppo_flow_target50_full_resume2000_to3000_20260928/xr1_robocasa365_ppo_full/checkpoints
screen_root=${repo_root}/eval_results/fixedseed-target50-step2500-step3000-screen5-${SCREEN_JOB_ID}
step2000_final=${repo_root}/eval_results/fixedseed-target50-step1500-step2000-20trials-final-111086/step2000/aggregate.json

cd "${repo_root}"
selection=$(STEP2000_FINAL="${step2000_final}" SCREEN_ROOT="${screen_root}" \
  "${repo_root}/.conda-robocasa365/bin/python" - <<'PY'
import json
import os
from pathlib import Path

def load(path):
    data = json.loads(Path(path).read_text())
    if data["completed_tasks"] != 50:
        raise RuntimeError(f"Incomplete aggregate: {path}")
    return data

base = load(os.environ["STEP2000_FINAL"])
if base["episodes"] != 1000 or base["num_trials_per_task"] != 20:
    raise RuntimeError("Unexpected step2000 protocol")
base_first5 = 0
base_episodes = 0
for row in base["task_summaries"]:
    tasks = row["summary"]["tasks"]
    if len(tasks) != 1:
        raise RuntimeError("Expected one task per summary")
    task = next(iter(tasks.values()))
    first = [episode for episode in task["episodes"] if episode["episode"] < 5]
    if len(first) != 5:
        raise RuntimeError("Incomplete step2000 five-seed prefix")
    base_first5 += sum(int(episode["success"]) for episode in first)
    base_episodes += len(first)

scores = {"step2000": (base_first5, base_episodes)}
for policy in ("step2500", "step3000"):
    data = load(Path(os.environ["SCREEN_ROOT"]) / policy / "aggregate.json")
    if data["episodes"] != 250 or data["num_trials_per_task"] != 5:
        raise RuntimeError(f"Unexpected {policy} screen protocol")
    scores[policy] = (int(data["successes"]), int(data["episodes"]))

# Prefer the later checkpoint only on an exact score tie.
winner = max(scores, key=lambda name: (scores[name][0] / scores[name][1], int(name[4:])))
print(winner)
print(json.dumps(scores), file=os.sys.stderr)
PY
)

echo "selected_policy=${selection} screen_job=${SCREEN_JOB_ID}"
if [[ "${selection}" == step2000 ]]; then
  echo "STEP2000_REMAINS_BEST existing_full20=${step2000_final}"
  exit 0
fi

selected_step=${selection#step}
checkpoint=${train_root}/global_step_${selected_step}/actor/model_state_dict/full_weights.pt
[[ -s "${checkpoint}" ]] || { echo "Missing checkpoint: ${checkpoint}" >&2; exit 2; }
gate_root=${repo_root}/eval_results/fixedseed-StoreLeftoversInBowl-${selection}-native-gate-${SLURM_JOB_ID}
result_root=${repo_root}/eval_results/fixedseed-target50-${selection}-20trials-${SLURM_JOB_ID}

POLICY_CHECKPOINT="${checkpoint}" POLICY_NAME="${selection}_native_gate20" \
EVAL_SPLIT=pretrain EVAL_SEED=327 NUM_TRIALS=5 NUM_LANES=1 \
EVAL_TASKS=StoreLeftoversInBowl EVAL_DIRECT_GL_READBACK=0 \
EVAL_SAVE_VIDEOS=0 EVAL_RESULT_ROOT="${gate_root}" \
  bash scripts/run_robocasa365_policy_eval_isolated.sbatch
echo "NATIVE_READBACK_GATE_PASSED host=$(hostname)"

POLICY_CHECKPOINT="${checkpoint}" POLICY_NAME="${selection}_eval20" \
EVAL_SPLIT=pretrain EVAL_SEED=7 EVAL_SEED_STRIDE=20 \
NUM_TRIALS=20 NUM_LANES=4 EVAL_DIRECT_GL_READBACK=0 \
EVAL_SAVE_VIDEOS=0 EVAL_RESULT_ROOT="${result_root}" \
  bash scripts/run_robocasa365_policy_eval_isolated.sbatch

echo "BEST_CHECKPOINT_EVAL20_COMPLETE policy=${selection} result_root=${result_root}"
