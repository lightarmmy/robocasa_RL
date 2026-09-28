#!/usr/bin/env bash
#SBATCH --job-name=xr1-robocasa-download
#SBATCH --output=Xiaomi-Robotics-1/logs/xr1_robocasa_download_%j.log
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4

set -euo pipefail

cd /storage/yukaichengLab/lishiwen/xufanghui/xiaomi_robotics_1_RL
mkdir -p Xiaomi-Robotics-1/checkpoints Xiaomi-Robotics-1/logs

/storage/yukaichengLab/lishiwen/miniconda3/bin/modelscope download \
  --model XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa \
  --local_dir Xiaomi-Robotics-1/checkpoints/Xiaomi-Robotics-1-RoboCasa
