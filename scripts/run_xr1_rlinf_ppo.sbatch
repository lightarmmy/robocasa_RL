#!/usr/bin/env bash
#SBATCH --job-name=xr1-rlinf-ppo
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/storage/yukaichengLab/lishiwen/xufanghui/xiaomi_robotics_1_RL/Xiaomi-Robotics-1/logs/xr1_rlinf_ppo_%j.log

set -euo pipefail
# Slurm executes a staged copy under /var/spool; resolve paths from the
# directory where sbatch was invoked on the shared filesystem.
repo_root=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}
repo_root=$(cd "${repo_root}" && pwd)
rlinf_root=${repo_root}/third_party/RLinf
python_bin=${repo_root}/.conda-robocasa365/bin/python
config_name=${XR1_RLINF_CONFIG:-xr1_robocasa365_ppo_full}

# RLinf is intentionally an external checkout (and therefore not committed
# with this repository).  Keep the XR-1 PPO config overlay under version
# control here and install it into that checkout at job start.
config_overlay=${repo_root}/xr1/mibot/rlinf/configs
install -D -m 644 "${config_overlay}/xr1_robocasa365_ppo_full.yaml" \
  "${rlinf_root}/examples/embodiment/config/xr1_robocasa365_ppo_full.yaml"
install -D -m 644 "${config_overlay}/xr1_robocasa365_ppo_flow_smoke.yaml" \
  "${rlinf_root}/examples/embodiment/config/xr1_robocasa365_ppo_flow_smoke.yaml"
install -D -m 644 "${config_overlay}/xr1_ppo.yaml" \
  "${rlinf_root}/examples/embodiment/config/model/xr1_ppo.yaml"
install -D -m 644 "${config_overlay}/xr1_ppo_flow.yaml" \
  "${rlinf_root}/examples/embodiment/config/model/xr1_ppo_flow.yaml"

export REPO_PATH=${rlinf_root}
export EMBODIED_PATH=${rlinf_root}/examples/embodiment
export PYTHONPATH=${repo_root}/xr1:${rlinf_root}:${repo_root}:${PYTHONPATH:-}
export XR1_ROBOCASA365_CKPT=${XR1_ROBOCASA365_CKPT:-${repo_root}/checkpoints/Xiaomi-Robotics-1-RoboCasa365}
export MUJOCO_GL=${XR1_MUJOCO_GL:-${MUJOCO_GL:-egl}}
export PYOPENGL_PLATFORM=${XR1_PYOPENGL_PLATFORM:-${PYOPENGL_PLATFORM:-${MUJOCO_GL}}}
export RLINF_ROBOCASA_START_METHOD=${RLINF_ROBOCASA_START_METHOD:-spawn}
export RLINF_ROBOCASA_SHARE_MEMORY=${RLINF_ROBOCASA_SHARE_MEMORY:-1}
export RLINF_ROBOCASA_MAX_RESTARTS=${RLINF_ROBOCASA_MAX_RESTARTS:-8}
# A lost native-simulator transition cannot be reconstructed. Full training
# therefore fails with the real child exit status instead of silently joining
# pre-crash and post-reset states. `restart` is reserved for fault-injection
# tests and diagnostics where invalid transitions are explicitly inspected.
export RLINF_ROBOCASA_EOF_POLICY=${RLINF_ROBOCASA_EOF_POLICY:-raise}
export RLINF_ROBOCASA_RECONFIGURE_RESTART=${RLINF_ROBOCASA_RECONFIGURE_RESTART:-1}
export TOKENIZERS_PARALLELISM=false

cd "${rlinf_root}"
cmd=(
  "${python_bin}" examples/embodiment/train_embodied_agent.py
  --config-path config
  --config-name "${config_name}"
  "runner.logger.log_path=${XR1_RLINF_LOG_DIR:-${repo_root}/results/${config_name}}"
)
# Pass numeric smoke overrides through Hydra so they retain numeric types;
# oc.env values are strings and fail RLinf's config validation.
[[ -n "${XR1_RLINF_MAX_STEPS:-}" ]] && cmd+=("runner.max_steps=${XR1_RLINF_MAX_STEPS}")
[[ -n "${XR1_RLINF_SAVE_INTERVAL:-}" ]] && cmd+=("runner.save_interval=${XR1_RLINF_SAVE_INTERVAL}")
[[ -n "${XR1_RLINF_RESUME_DIR:-}" ]] && cmd+=("runner.resume_dir=${XR1_RLINF_RESUME_DIR}")
[[ -n "${XR1_RLINF_TRAIN_ENVS:-}" ]] && cmd+=("env.train.total_num_envs=${XR1_RLINF_TRAIN_ENVS}")
[[ -n "${XR1_RLINF_EPISODE_STEPS:-}" ]] && cmd+=("env.train.max_episode_steps=${XR1_RLINF_EPISODE_STEPS}")
[[ -n "${XR1_RLINF_ROLLOUT_STEPS:-}" ]] && cmd+=("env.train.max_steps_per_rollout_epoch=${XR1_RLINF_ROLLOUT_STEPS}")
[[ -n "${XR1_RLINF_ACTION_CHUNKS:-}" ]] && cmd+=("actor.model.num_action_chunks=${XR1_RLINF_ACTION_CHUNKS}" "actor.model.action_horizon=${XR1_RLINF_ACTION_CHUNKS}")
[[ -n "${XR1_RLINF_SPLIT:-}" ]] && cmd+=("env.train.split=${XR1_RLINF_SPLIT}")
[[ -n "${XR1_RLINF_TASK_SOUP:-}" ]] && cmd+=("env.train.task_soup=${XR1_RLINF_TASK_SOUP}")
[[ -n "${XR1_RLINF_TASK_FILTER_INCLUDE:-}" ]] && cmd+=("env.train.task_filter=[${XR1_RLINF_TASK_FILTER_INCLUDE}]")
[[ -n "${XR1_RLINF_TASK_SAMPLING:-}" ]] && cmd+=("env.train.task_sampling_strategy=${XR1_RLINF_TASK_SAMPLING}")
[[ -n "${XR1_RLINF_SEED:-}" ]] && cmd+=("env.train.seed=${XR1_RLINF_SEED}")
[[ -n "${XR1_RLINF_AUTO_RESET:-}" ]] && cmd+=("env.train.auto_reset=${XR1_RLINF_AUTO_RESET}")
[[ -n "${XR1_RLINF_ROTATE_AUTO_RESET:-}" ]] && cmd+=("env.train.rotate_tasks_on_auto_reset=${XR1_RLINF_ROTATE_AUTO_RESET}")
[[ -n "${XR1_RLINF_ROTATE_ROLLOUT:-}" ]] && cmd+=("env.train.rotate_tasks_on_rollout=${XR1_RLINF_ROTATE_ROLLOUT}")
[[ -n "${XR1_RLINF_EPISODE_HORIZON_SOURCE:-}" ]] && cmd+=("env.train.episode_horizon_source=${XR1_RLINF_EPISODE_HORIZON_SOURCE}")
[[ -n "${XR1_RLINF_USE_REL_REWARD:-}" ]] && cmd+=("env.train.use_rel_reward=${XR1_RLINF_USE_REL_REWARD}")
[[ -n "${XR1_RLINF_MICRO_BATCH:-}" ]] && cmd+=("actor.micro_batch_size=${XR1_RLINF_MICRO_BATCH}")
[[ -n "${XR1_RLINF_GLOBAL_BATCH:-}" ]] && cmd+=("actor.global_batch_size=${XR1_RLINF_GLOBAL_BATCH}")
[[ -n "${XR1_RLINF_TRAINABLE_MODE:-}" ]] && cmd+=("actor.model.trainable_mode=${XR1_RLINF_TRAINABLE_MODE}")
[[ -n "${XR1_RLINF_ACTION_STD:-}" ]] && cmd+=("actor.model.action_std=${XR1_RLINF_ACTION_STD}")
[[ -n "${XR1_RLINF_CLIP_ACTION:-}" ]] && cmd+=("actor.model.clip_normalized_action=${XR1_RLINF_CLIP_ACTION}")
[[ -n "${XR1_RLINF_LR:-}" ]] && cmd+=("actor.optim.lr=${XR1_RLINF_LR}")
[[ -n "${XR1_RLINF_CLIP_GRAD:-}" ]] && cmd+=("actor.optim.clip_grad=${XR1_RLINF_CLIP_GRAD}")
[[ -n "${XR1_RLINF_NORMALIZE_ADVANTAGES:-}" ]] && cmd+=("algorithm.normalize_advantages=${XR1_RLINF_NORMALIZE_ADVANTAGES}")
[[ -n "${XR1_RLINF_RECOMPUTE_LOGPROBS:-}" ]] && cmd+=("algorithm.recompute_logprobs=${XR1_RLINF_RECOMPUTE_LOGPROBS}")
[[ -n "${XR1_RLINF_UPDATE_EPOCH:-}" ]] && cmd+=("algorithm.update_epoch=${XR1_RLINF_UPDATE_EPOCH}")
[[ "${XR1_RLINF_DISABLE_AUTO_WRAP:-0}" == "1" || "${XR1_RLINF_DISABLE_FSDP:-0}" == "1" ]] && cmd+=("+actor.fsdp_config.disable=true")
[[ "${XR1_RLINF_USE_ORIG_PARAMS:-0}" == "1" ]] && cmd+=("actor.fsdp_config.use_orig_params=true")
[[ "${XR1_RLINF_DISABLE_GRAD_CLIP:-0}" == "1" ]] && cmd+=("actor.optim.clip_grad=0")
exec "${cmd[@]}"
