#!/usr/bin/env bash
# Complete the 20 missing step-1000 target50 tasks only after a long-horizon
# native-MuJoCo-readback gate passes in the same Slurm allocation and node.
#SBATCH --job-name=xr1-s1000-eval-gated
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --exclude=gnho011,gnho020
#SBATCH --output=/storage/yukaichengLab/lishiwen/xufanghui/xiaomi_robotics_1_RL/Xiaomi-Robotics-1/logs/xr1_step1000_eval_gated_%j.log

set -euo pipefail

workspace_root=/storage/yukaichengLab/lishiwen/xufanghui/xiaomi_robotics_1_RL
repo_root=${workspace_root}/Xiaomi-Robotics-1
checkpoint=${repo_root}/results/xr1_ppo_flow_target50_full_1000u_dynamicpad_20260921/xr1_robocasa365_ppo_full/checkpoints/global_step_1000/actor/model_state_dict/full_weights.pt
gate_root=${repo_root}/eval_results/fixedseed-StoreLeftoversInBowl-step1000-5trials-native-gate-${SLURM_JOB_ID}
result_root=${repo_root}/eval_results/fixedseed-target50-step1000-5trials-v10-fill3-${SLURM_JOB_ID}
remaining_tasks=ArrangeBreadBasket,ArrangeTea,BreadSelection,CategorizeCondiments,CuttingToolSelection,GarnishPancake,GatherTableware,HeatKebabSandwich,MakeIceLemonade,PanTransfer,PortionHotDogs,RecycleBottlesByType,SearingMeat,SeparateFreezerRack,SteamInMicrowave,StirVegetables,StoreLeftoversInBowl,WaffleReheat,WashFruitColander,WeighIngredients

[[ -f "${checkpoint}" ]] || {
  echo "Missing checkpoint: ${checkpoint}" >&2
  exit 2
}

cd "${repo_root}"
echo "gated_eval_job=${SLURM_JOB_ID} host=$(hostname) checkpoint=${checkpoint}"
echo "gate_root=${gate_root}"
echo "result_root=${result_root}"

# Native readback is required so the observation protocol remains identical to
# the valid pretrained/step500 results. StoreLeftoversInBowl seed 167 used to
# abort around steps 750--800 on bad nodes, so five complete episodes are the
# admission test for this exact allocation.
POLICY_CHECKPOINT="${checkpoint}" \
POLICY_NAME=step1000_native_gate \
EVAL_SPLIT=pretrain EVAL_SEED=7 NUM_TRIALS=5 NUM_LANES=1 \
EVAL_TASKS=StoreLeftoversInBowl EVAL_DIRECT_GL_READBACK=0 \
EVAL_RESULT_ROOT="${gate_root}" \
bash scripts/run_robocasa365_policy_eval_isolated.sbatch

echo "NATIVE_READBACK_GATE_PASSED host=$(hostname)"

POLICY_CHECKPOINT="${checkpoint}" \
POLICY_NAME=step1000_fill3 \
EVAL_SPLIT=pretrain EVAL_SEED=7 NUM_TRIALS=5 NUM_LANES=4 \
EVAL_TASKS="${remaining_tasks}" EVAL_DIRECT_GL_READBACK=0 \
EVAL_RESULT_ROOT="${result_root}" \
bash scripts/run_robocasa365_policy_eval_isolated.sbatch

echo "STEP1000_REMAINING_EVAL_COMPLETED result_root=${result_root}"
