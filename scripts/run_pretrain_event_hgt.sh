#!/bin/bash
#$ -l tmem=16G
#$ -l h_rt=72:00:00
#$ -l gpu=true
#$ -N event_hgt_pretrain
#$ -wd /data/home/bty414/opentarget_temporal_study/src/opentarget_het_graph
#$ -j y
#$ -o logs/event_hgt_pretrain.log
#$ -t 1

# Event-based HGT Self-Supervised Pretraining with RTE 
# Uses causal temporal sampling with Relative Temporal Encoding

hostname
date

source /data/home/bty414/opentarget_temporal_study/src/opentarget_het_graph/.venv/bin/activate

echo "Starting Event-based HGT Pretraining with RTE..."
echo "Config: config/experiments/event_hgt.yaml"

python src/train_self_supervised_event.py \
    --config config/experiments/event_hgt.yaml

echo "Event HGT Pretraining Complete!"
date
