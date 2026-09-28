#!/usr/bin/env bash
#SBATCH --job-name=xr1-robocasa-assets
#SBATCH --partition=yukaichenglab
#SBATCH --cpus-per-task=2
#SBATCH --time=04:00:00
#SBATCH --output=Xiaomi-Robotics-1/logs/xr1_robocasa_assets_%j.log

set -euo pipefail

workspace_root=/storage/yukaichengLab/lishiwen/xufanghui/xiaomi_robotics_1_RL
repo_root=${workspace_root}/Xiaomi-Robotics-1
python_bin=/storage/yukaichengLab/lishiwen/miniconda3/envs/gr00tRL/bin/python

export PYTHONPATH="${repo_root}/.xr1_vendor:${repo_root}/third_party/robosuite:${workspace_root}/robocasa"
export NUMBA_CACHE_DIR="${repo_root}/.numba_cache"
export USE_TF=0
export USE_TORCH=1
mkdir -p "${NUMBA_CACHE_DIR}"

cd "${workspace_root}/robocasa"
printf 'y\n' | "${python_bin}" -m robocasa.scripts.download_kitchen_assets --type objs_lw

test -f robocasa/models/assets/objects/lightwheel/utensil_rack/UtensilRack014/model.xml
