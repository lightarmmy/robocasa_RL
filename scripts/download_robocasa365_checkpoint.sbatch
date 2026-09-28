#!/usr/bin/env bash
# Download the checkpoint that matches eval_robocasa365 and RoboCasa v1.x.
#SBATCH --job-name=xr1-robocasa365-download
#SBATCH --partition=yukaichenglab
#SBATCH --output=Xiaomi-Robotics-1/logs/xr1_robocasa365_download_%j.log
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4

set -euo pipefail

cd /storage/yukaichengLab/lishiwen/xufanghui/xiaomi_robotics_1_RL
mkdir -p Xiaomi-Robotics-1/checkpoints Xiaomi-Robotics-1/logs

/storage/yukaichengLab/lishiwen/miniconda3/bin/modelscope download \
  --model XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365 \
  --local_dir Xiaomi-Robotics-1/checkpoints/Xiaomi-Robotics-1-RoboCasa365
