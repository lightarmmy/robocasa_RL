#!/usr/bin/env bash
#SBATCH --job-name=xr1-robosuite-setup
#SBATCH --partition=yukaichenglab
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --output=Xiaomi-Robotics-1/logs/xr1_robosuite_setup_%j.log

set -euo pipefail

cd /storage/yukaichengLab/lishiwen/xufanghui/xiaomi_robotics_1_RL
repo=Xiaomi-Robotics-1/third_party/robosuite

if [[ ! -d "${repo}/.git" ]]; then
  mkdir -p Xiaomi-Robotics-1/third_party
  git clone --depth=1 https://github.com/ARISE-Initiative/robosuite "${repo}"
else
  git -C "${repo}" fetch --depth=1 origin master
  git -C "${repo}" checkout --detach FETCH_HEAD
fi

rg -q 'class PandaOmron' "${repo}/robosuite/models/robots/manipulators/panda_omron_robot.py"
