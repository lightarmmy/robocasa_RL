#!/usr/bin/env bash
# Screen step2500 and step3000 on the first five seeds of the fixed 20-seed
# manifest. Both policies run concurrently on four dedicated GPUs each.
#SBATCH --job-name=xr1-s25-s30-screen5
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --exclude=gnho006,gnho011,gnho020
#SBATCH --output=/storage/yukaichengLab/lishiwen/xufanghui/xiaomi_robotics_1_RL/Xiaomi-Robotics-1/logs/xr1_step2500_step3000_screen5_%j.log

set -euo pipefail

repo_root=/storage/yukaichengLab/lishiwen/xufanghui/xiaomi_robotics_1_RL/Xiaomi-Robotics-1
checkpoint_root=${repo_root}/results/xr1_ppo_flow_target50_full_resume2000_to3000_20260928/xr1_robocasa365_ppo_full/checkpoints
step2500=${checkpoint_root}/global_step_2500/actor/model_state_dict/full_weights.pt
step3000=${checkpoint_root}/global_step_3000/actor/model_state_dict/full_weights.pt
gate_root=${repo_root}/eval_results/fixedseed-StoreLeftoversInBowl-screen5-native-gate-${SLURM_JOB_ID}
result_root=${repo_root}/eval_results/fixedseed-target50-step2500-step3000-screen5-${SLURM_JOB_ID}

for checkpoint in "${step2500}" "${step3000}"; do
  [[ -s "${checkpoint}" ]] || { echo "Missing checkpoint: ${checkpoint}" >&2; exit 2; }
done
cd "${repo_root}"

echo "job=${SLURM_JOB_ID} host=$(hostname) seed_stride=20 trials=5"
echo "result_root=${result_root}"

# Reject allocations with the known native long-horizon renderer failure.
POLICY_CHECKPOINT="${step3000}" POLICY_NAME=screen5_native_gate \
EVAL_SPLIT=pretrain EVAL_SEED=327 NUM_TRIALS=5 NUM_LANES=1 \
EVAL_TASKS=StoreLeftoversInBowl EVAL_DIRECT_GL_READBACK=0 \
EVAL_SAVE_VIDEOS=0 EVAL_RESULT_ROOT="${gate_root}" \
  bash scripts/run_robocasa365_policy_eval_isolated.sbatch
echo "NATIVE_READBACK_GATE_PASSED host=$(hostname)"

run_screen() {
  local policy=$1 checkpoint=$2 offset=$3 port_base=$4
  POLICY_CHECKPOINT="${checkpoint}" POLICY_NAME="${policy}_screen5" \
  EVAL_SPLIT=pretrain EVAL_SEED=7 EVAL_SEED_STRIDE=20 \
  NUM_TRIALS=5 NUM_LANES=4 EVAL_LANE_OFFSET="${offset}" \
  EVAL_PORT_BASE="${port_base}" EVAL_DIRECT_GL_READBACK=0 \
  EVAL_SAVE_VIDEOS=0 EVAL_RESULT_ROOT="${result_root}/${policy}" \
    bash scripts/run_robocasa365_policy_eval_isolated.sbatch
}

run_screen step2500 "${step2500}" 0 33000 &
pid2500=$!
run_screen step3000 "${step3000}" 4 33100 &
pid3000=$!
status=0
wait "${pid2500}" || status=1
wait "${pid3000}" || status=1
(( status == 0 )) || exit 1

echo "SCREEN5_COMPLETE result_root=${result_root}"
