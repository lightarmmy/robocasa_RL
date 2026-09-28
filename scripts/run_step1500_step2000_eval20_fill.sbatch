#!/usr/bin/env bash
# Fill only the task-policy pairs missing from job 109969, then merge them with
# its valid task summaries and require exactly 50 tasks / 1000 episodes each.
#SBATCH --job-name=xr1-s15-s20-eval20-fill
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=36:00:00
# Use a currently idle node; failed partial results are rejected by the strict
# 50-task / 1000-episode merge below.
#SBATCH --nodelist=gnho031
#SBATCH --output=/storage/yukaichengLab/lishiwen/xufanghui/xiaomi_robotics_1_RL/Xiaomi-Robotics-1/logs/xr1_step1500_step2000_eval20_fill_%j.log

set -euo pipefail

repo_root=/storage/yukaichengLab/lishiwen/xufanghui/xiaomi_robotics_1_RL/Xiaomi-Robotics-1
train_root=${repo_root}/results/xr1_ppo_flow_target50_full_resume1000_to2000_20260924_v3/xr1_robocasa365_ppo_full/checkpoints
source_root=${repo_root}/eval_results/fixedseed-target50-step1500-step2000-20trials-109969
fill_root=${repo_root}/eval_results/fixedseed-target50-step1500-step2000-20trials-fill-${SLURM_JOB_ID}
final_root=${repo_root}/eval_results/fixedseed-target50-step1500-step2000-20trials-final-${SLURM_JOB_ID}

step1500=${train_root}/global_step_1500/actor/model_state_dict/full_weights.pt
step2000=${train_root}/global_step_2000/actor/model_state_dict/full_weights.pt
missing1500=ArrangeBreadBasket,CategorizeCondiments,CuttingToolSelection,DeliverStraw,HeatKebabSandwich,LoadDishwasher,MakeIceLemonade,PackIdenticalLunches,RecycleBottlesByType,RinseSinkBasin,SeparateFreezerRack,StackBowlsCabinet,SteamInMicrowave,WashLettuce,WeighIngredients
missing2000=ArrangeBreadBasket,CategorizeCondiments,CuttingToolSelection,DeliverStraw,HeatKebabSandwich,LoadDishwasher,MakeIceLemonade,OpenCabinet,PackIdenticalLunches,RecycleBottlesByType,RinseSinkBasin,SeparateFreezerRack,StackBowlsCabinet,SteamInMicrowave,WashLettuce,WeighIngredients

for path in \
  "${step1500}" "${step2000}" \
  "${source_root}/step1500/aggregate.json" \
  "${source_root}/step2000/aggregate.json"; do
  [[ -s "${path}" ]] || { echo "Missing input: ${path}" >&2; exit 2; }
done

cd "${repo_root}"
echo "job=${SLURM_JOB_ID} host=$(hostname) source_job=109969"
echo "protocol=target50 split=pretrain seed=7 trials_per_task=20"
echo "readback=native_mujoco"
echo "fill_root=${fill_root}"
echo "final_root=${final_root}"

run_fill() {
  local policy=$1 checkpoint=$2 tasks=$3 offset=$4 port_base=$5
  POLICY_CHECKPOINT="${checkpoint}" POLICY_NAME="${policy}_fill" \
  EVAL_SPLIT=pretrain EVAL_SEED=7 NUM_TRIALS=20 NUM_LANES=4 \
  EVAL_TASKS="${tasks}" EVAL_LANE_OFFSET="${offset}" \
  EVAL_PORT_BASE="${port_base}" EVAL_DIRECT_GL_READBACK=0 \
  EVAL_SAVE_VIDEOS=0 EVAL_RESULT_ROOT="${fill_root}/${policy}" \
    bash scripts/run_robocasa365_policy_eval_isolated.sbatch
}

run_fill step1500 "${step1500}" "${missing1500}" 0 32000 &
pid1500=$!
run_fill step2000 "${step2000}" "${missing2000}" 4 32100 &
pid2000=$!

status=0
wait "${pid1500}" || status=1
wait "${pid2000}" || status=1
if (( status != 0 )); then
  echo "FILL_EVALUATION_INCOMPLETE fill_root=${fill_root}" >&2
  exit 1
fi

SOURCE_ROOT="${source_root}" FILL_ROOT="${fill_root}" FINAL_ROOT="${final_root}" \
  "${repo_root}/.conda-robocasa365/bin/python" - <<'PY'
import json
import os
import re
from pathlib import Path

from robocasa.utils.dataset_registry import TASK_SET_REGISTRY

source_root = Path(os.environ["SOURCE_ROOT"])
fill_root = Path(os.environ["FILL_ROOT"])
final_root = Path(os.environ["FINAL_ROOT"])
expected_tasks = list(TASK_SET_REGISTRY["target50"])


def task_name(row):
    summary = row["summary"]
    task_stats = summary.get("task_stats", {})
    if len(task_stats) == 1:
        return next(iter(task_stats))
    match = re.search(r"-lane\d+-(.+)-attempt\d+$", Path(row["path"]).parent.name)
    if not match:
        raise RuntimeError(f"Cannot identify task for {row['path']}")
    return match.group(1)


final_root.mkdir(parents=True, exist_ok=False)
for policy in ("step1500", "step2000"):
    source = json.loads((source_root / policy / "aggregate.json").read_text())
    fill = json.loads((fill_root / policy / "aggregate.json").read_text())
    if (source["split"], source["seed"], source["num_trials_per_task"]) != ("pretrain", 7, 20):
        raise RuntimeError(f"Unexpected source protocol for {policy}")
    if (fill["split"], fill["seed"], fill["num_trials_per_task"]) != ("pretrain", 7, 20):
        raise RuntimeError(f"Unexpected fill protocol for {policy}")

    rows = source["task_summaries"] + fill["task_summaries"]
    by_task = {}
    for row in rows:
        task = task_name(row)
        if task in by_task:
            raise RuntimeError(f"Duplicate task for {policy}: {task}")
        if int(row["summary"].get("num_episodes", -1)) != 20:
            raise RuntimeError(f"Incomplete task for {policy}: {task}")
        by_task[task] = row

    missing = sorted(set(expected_tasks) - set(by_task))
    unexpected = sorted(set(by_task) - set(expected_tasks))
    if missing or unexpected or len(by_task) != 50:
        raise RuntimeError(
            f"Invalid task coverage for {policy}: missing={missing}, unexpected={unexpected}"
        )

    ordered_rows = [by_task[task] for task in expected_tasks]
    episodes = sum(int(row["summary"]["num_episodes"]) for row in ordered_rows)
    successes = sum(int(row["summary"]["successes"]) for row in ordered_rows)
    if episodes != 1000:
        raise RuntimeError(f"Invalid episode count for {policy}: {episodes}")

    out = {
        "policy": policy,
        "policy_checkpoint": source["policy_checkpoint"],
        "split": "pretrain",
        "seed": 7,
        "num_trials_per_task": 20,
        "completed_tasks": 50,
        "expected_tasks": 50,
        "episodes": episodes,
        "expected_episodes": 1000,
        "successes": successes,
        "success_rate": successes / episodes,
        "source_aggregates": [
            str(source_root / policy / "aggregate.json"),
            str(fill_root / policy / "aggregate.json"),
        ],
        "task_summaries": ordered_rows,
    }
    policy_dir = final_root / policy
    policy_dir.mkdir()
    (policy_dir / "aggregate.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({key: out[key] for key in (
        "policy", "completed_tasks", "episodes", "successes", "success_rate"
    )}))
PY

echo "FINAL_EVALUATION_COMPLETE final_root=${final_root}"
