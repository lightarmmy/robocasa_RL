#!/usr/bin/env bash
#SBATCH --job-name=pi05-robocasa-rl
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/pi05_robocasa_rl_%j.log

set -euo pipefail

repo_root=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}
repo_root=$(cd "${repo_root}" && pwd)
rlinf_root=${repo_root}/third_party/RLinf
openpi_root=${repo_root}/third_party/openpi-g01-sft
python_bin=${repo_root}/.conda-robocasa365/bin/python
config_name=${PI05_RLINF_CONFIG:-pi05_robocasa365_ppo}
: "${PI05_BASE_CKPT:?Set PI05_BASE_CKPT to the external pi0.5 checkpoint directory}"
base_ckpt=${PI05_BASE_CKPT}

: "${PI05_ROBOCASA_NORM_STATS:?Set PI05_ROBOCASA_NORM_STATS to matching RoboCasa norm_stats.json}"

"${python_bin}" "${repo_root}/scripts/multimodel_robocasa_preflight.py" \
  --model pi05 \
  --algorithm "${PI05_RLINF_ALGORITHM:-ppo}" \
  --checkpoint "${base_ckpt}" \
  --norm-stats "${PI05_ROBOCASA_NORM_STATS}"

config_overlay=${repo_root}/multimodel_robocasa/configs
install -D -m 644 "${config_overlay}/pi05_robocasa365_ppo.yaml" \
  "${rlinf_root}/examples/embodiment/config/pi05_robocasa365_ppo.yaml"
install -D -m 644 "${config_overlay}/pi05_robocasa365_grpo.yaml" \
  "${rlinf_root}/examples/embodiment/config/pi05_robocasa365_grpo.yaml"

export REPO_PATH=${rlinf_root}
export EMBODIED_PATH=${rlinf_root}/examples/embodiment
export PI05_BASE_CKPT=${base_ckpt}
export PYTHONPATH=${openpi_root}/src:${rlinf_root}:${repo_root}:${PYTHONPATH:-}
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-${MUJOCO_GL}}
export TOKENIZERS_PARALLELISM=false

cd "${rlinf_root}"
exec "${python_bin}" examples/embodiment/train_embodied_agent.py \
  --config-path config \
  --config-name "${config_name}"
