#!/usr/bin/env bash
# Evaluate the completed continuation checkpoints concurrently: each policy
# receives four dedicated GPU/EGL lanes and all 50 target tasks x 20 trials.
#SBATCH --job-name=xr1-s15-s20-eval20
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=/storage/yukaichengLab/lishiwen/xufanghui/xiaomi_robotics_1_RL/Xiaomi-Robotics-1/logs/xr1_step1500_step2000_eval20_%j.log

set -euo pipefail

repo_root=/storage/yukaichengLab/lishiwen/xufanghui/xiaomi_robotics_1_RL/Xiaomi-Robotics-1
train_root=${repo_root}/results/xr1_ppo_flow_target50_full_resume1000_to2000_20260924_v3/xr1_robocasa365_ppo_full/checkpoints
step1500=${train_root}/global_step_1500/actor/model_state_dict/full_weights.pt
step2000=${train_root}/global_step_2000/actor/model_state_dict/full_weights.pt
eval_root=${repo_root}/eval_results/fixedseed-target50-step1500-step2000-20trials-${SLURM_JOB_ID}

for checkpoint in "${step1500}" "${step2000}"; do
  [[ -s "${checkpoint}" ]] || {
    echo "Missing or empty checkpoint: ${checkpoint}" >&2
    exit 2
  }
done

cd "${repo_root}"
echo "job=${SLURM_JOB_ID} host=$(hostname) dependency_train_job=109745"
echo "protocol=target50 split=pretrain seed=7 trials_per_task=20 expected_per_policy=1000"
echo "step1500_checkpoint=${step1500}"
echo "step2000_checkpoint=${step2000}"
echo "eval_root=${eval_root}"

run_policy() {
  local policy_name=$1
  local checkpoint=$2
  local lane_offset=$3
  local port_base=$4
  local result_root=${eval_root}/${policy_name}

  POLICY_CHECKPOINT="${checkpoint}" \
  POLICY_NAME="${policy_name}" \
  EVAL_SPLIT=pretrain EVAL_SEED=7 NUM_TRIALS=20 NUM_LANES=4 \
  EVAL_LANE_OFFSET="${lane_offset}" EVAL_PORT_BASE="${port_base}" \
  EVAL_DIRECT_GL_READBACK=0 EVAL_SAVE_VIDEOS=0 \
  EVAL_RESULT_ROOT="${result_root}" \
    bash scripts/run_robocasa365_policy_eval_isolated.sbatch
}

run_policy step1500 "${step1500}" 0 31000 &
pid1500=$!
run_policy step2000 "${step2000}" 4 31100 &
pid2000=$!

status=0
wait "${pid1500}" || status=1
wait "${pid2000}" || status=1

for policy in step1500 step2000; do
  aggregate=${eval_root}/${policy}/aggregate.json
  [[ -s "${aggregate}" ]] || {
    echo "Missing aggregate: ${aggregate}" >&2
    status=1
  }
done

if (( status != 0 )); then
  echo "EVALUATION_INCOMPLETE eval_root=${eval_root}" >&2
  exit 1
fi

echo "EVALUATION_COMPLETE eval_root=${eval_root}"
