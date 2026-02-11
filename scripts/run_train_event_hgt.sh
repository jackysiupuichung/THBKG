#!/bin/bash
#$ -l h_rt=240:0:0
#$ -l h_vmem=11G
#$ -pe smp 8
#$ -l gpu=1
#$ -cwd
#$ -j y

set -euo pipefail

# Activate virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

echo "================================================================================"
echo "RUNNING EVENT-BASED SELF-SUPERVISED PRETRAINING (HGT)"
echo "================================================================================"

# 1. HGT Pretrain
echo "▶️  Running HGT (Event)..."
python src/train_self_supervised_event.py --config config/experiments/pretrain_event_hgt.yaml
