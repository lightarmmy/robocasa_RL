#!/usr/bin/env bash
# Two independent four-GPU RoboCasa365 eval processes on one node.
# Set ACTION_EXPERT_CKPT and FULL_CKPT to RLinf full_weights.pt files.
#SBATCH --job-name=xr1-robocasa365-2policy
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=/storage/yukaichengLab/lishiwen/xufanghui/xiaomi_robotics_1_RL/Xiaomi-Robotics-1/logs/xr1_robocasa365_2policy_%j.log

set -euo pipefail
workspace_root=/storage/yukaichengLab/lishiwen/xufanghui/xiaomi_robotics_1_RL
repo_root=${workspace_root}/Xiaomi-Robotics-1
python=${repo_root}/.conda-robocasa365/bin/python
model_path=${repo_root}/checkpoints/Xiaomi-Robotics-1-RoboCasa365
action_expert_ckpt=${ACTION_EXPERT_CKPT:-${repo_root}/results/xr1_ppo_stable_lr1e9_10u/xr1_robocasa365_ppo_full/checkpoints/global_step_10/actor/model_state_dict/full_weights.pt}
full_ckpt=${FULL_CKPT:-${repo_root}/results/xr1_ppo_full_test/xr1_robocasa365_ppo_full/checkpoints/global_step_10/actor/model_state_dict/full_weights.pt}
result_root=${repo_root}/eval_results/robocasa365-two-policy-${SLURM_JOB_ID}
mkdir -p "${result_root}"
cd "${repo_root}"
export MUJOCO_GL=egl TOKENIZERS_PARALLELISM=false USE_TF=0 USE_TORCH=1
export HF_HOME=${repo_root}/.hf_cache OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1

run_policy() {
  local policy_name=$1
  local gpu_start=$2
  local checkpoint=$3
  local result_dir=${result_root}/${policy_name}
  local run_id=target50-5trials-${policy_name}-${SLURM_JOB_ID}
  mkdir -p "${result_dir}"
  POLICY_CHECKPOINT="${checkpoint}" \
    GPU_OFFSET=${gpu_start} \
    EVAL_RESULT_ROOT="${result_dir}" \
    EVAL_RUN_ID="${run_id}" \
    BASE_PORT=$((11086 + gpu_start)) \
    bash scripts/run_robocasa365_target50_5trials_4gpu_single_node.sbatch
}

[[ -f "${action_expert_ckpt}" ]] || { echo "Missing action-expert checkpoint: ${action_expert_ckpt}" >&2; exit 2; }
[[ -f "${full_ckpt}" ]] || { echo "Missing full checkpoint: ${full_ckpt}" >&2; exit 2; }

run_policy action_expert 0 "${action_expert_ckpt}" >"${result_root}/action_expert.log" 2>&1 &
pid_action=$!
run_policy full 4 "${full_ckpt}" >"${result_root}/full.log" 2>&1 &
pid_full=$!
status=0
wait "${pid_action}" || status=1
wait "${pid_full}" || status=1
exit "${status}"
