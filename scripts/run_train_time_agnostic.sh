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



# Enable WandB Offline Mode
export WANDB_MODE=offline

echo "================================================================================"
echo "RUNNING GATv2 TIME-AGNOSTIC TRAINING (WANDB OFFLINE)"
echo "================================================================================"

# Time-Agnostic (GATv2)
python -m src.train_time_agnostic --config config/experiments/gatv2_agnostic.yaml