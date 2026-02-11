#!/bin/bash
#$ -l tmem=16G
#$ -l h_rt=24:00:00
#$ -l gpu=true
#$ -N event_hgt_finetune
#$ -wd /data/home/bty414/opentarget_temporal_study/src/opentarget_het_graph
#$ -j y
#$ -o logs/event_hgt_finetune.log
#$ -t 1

# Event-based HGT Clinical Multi-Task Finetuning
# Uses pretrained event HGT encoder with RTE

hostname
date

source /data/home/bty414/opentarget_temporal_study/src/opentarget_het_graph/.venv/bin/activate

echo "Starting Event-based HGT Finetuning..."
echo "Config: config/experiments/event_hgt.yaml"

python src/train_clinical_multitask.py \
    --config config/experiments/event_hgt.yaml

echo "Event HGT Finetuning Complete!"
date
