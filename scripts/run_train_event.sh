#!/bin/bash
#$ -l h_rt=240:0:0
#$ -l h_vmem=11G
#$ -pe smp 8
#$ -l gpu=1
#$ -cwd
#$ -j y

set -euo pipefail

# Activate virtual environment
source .venv/bin/activate



# Enable WandB Online Mode
export WANDB_MODE=online

echo "================================================================================"
echo "RUNNING HGT EVENT-BASED TRAINING (WANDB ONLINE)"
echo "================================================================================"

# Event-Based (HGT with RTE)
python -m src.train_event --config config/experiments/hgt_rte.yaml