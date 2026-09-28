#!/usr/bin/env bash
# Gate native MuJoCo readback on the allocated node, then evaluate step1000
# with the exact 50-task x 20-trial protocol used by step1500 and step2000.
#SBATCH --job-name=xr1-s1000-eval20
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=36:00:00
#SBATCH --exclude=gnho006,gnho011,gnho020
#SBATCH --output=/storage/yukaichengLab/lishiwen/xufanghui/xiaomi_robotics_1_RL/Xiaomi-Robotics-1/logs/xr1_step1000_eval20_%j.log

set -euo pipefail

repo_root=/storage/yukaichengLab/lishiwen/xufanghui/xiaomi_robotics_1_RL/Xiaomi-Robotics-1
checkpoint=${repo_root}/results/xr1_ppo_flow_target50_full_1000u_dynamicpad_20260921/xr1_robocasa365_ppo_full/checkpoints/global_step_1000/actor/model_state_dict/full_weights.pt
gate_root=${repo_root}/eval_results/fixedseed-StoreLeftoversInBowl-step1000-20protocol-native-gate-${SLURM_JOB_ID}
result_root=${repo_root}/eval_results/fixedseed-target50-step1000-20trials-${SLURM_JOB_ID}

[[ -s "${checkpoint}" ]] || { echo "Missing checkpoint: ${checkpoint}" >&2; exit 2; }
cd "${repo_root}"

echo "job=${SLURM_JOB_ID} host=$(hostname) checkpoint=${checkpoint}"
echo "protocol=target50 split=pretrain seed=7 trials_per_task=20 expected=1000"
echo "gate_root=${gate_root}"
echo "result_root=${result_root}"

# This task and seed range previously exposed node-specific long-horizon
# mjr_readPixels aborts. Keep the gate and benchmark in the same allocation.
POLICY_CHECKPOINT="${checkpoint}" POLICY_NAME=step1000_native_gate20 \
EVAL_SPLIT=pretrain EVAL_SEED=327 NUM_TRIALS=5 NUM_LANES=1 \
EVAL_TASKS=StoreLeftoversInBowl EVAL_DIRECT_GL_READBACK=0 \
EVAL_SAVE_VIDEOS=0 EVAL_RESULT_ROOT="${gate_root}" \
  bash scripts/run_robocasa365_policy_eval_isolated.sbatch

echo "NATIVE_READBACK_GATE_PASSED host=$(hostname)"

POLICY_CHECKPOINT="${checkpoint}" POLICY_NAME=step1000_eval20 \
EVAL_SPLIT=pretrain EVAL_SEED=7 EVAL_SEED_STRIDE=20 NUM_TRIALS=20 NUM_LANES=4 \
EVAL_DIRECT_GL_READBACK=0 EVAL_SAVE_VIDEOS=0 \
EVAL_RESULT_ROOT="${result_root}" \
  bash scripts/run_robocasa365_policy_eval_isolated.sbatch

echo "STEP1000_EVAL20_COMPLETE result_root=${result_root}"
