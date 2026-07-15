#!/bin/bash
#SBATCH -J w3_decay
#SBATCH -o %x.o%j
#SBATCH -p computeshort
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -t 1:0:0
#SBATCH --mem-per-cpu=48G
# Temporal-decay analysis: does performance drop for pairs whose decision year
# is further in the future? Models: GATv2-both (injected) + RDG/OTS (from the w3
# zarr). Uses the same w3 eval inputs as run_w3_eval.sh so masking/labels match.
# Writes to a separate results dir so the full w3 eval is untouched.
set -euo pipefail
cd /data/home/bty414/opentarget_temporal_study/src/opentarget_het_graph
source .venv/bin/activate
export WANDB_MODE="disabled"

W3RUNS=/gpfs/scratch/bty414/opentarget_evidences/26.03/runs/w3_retrain
W3ZARR=/gpfs/scratch/bty414/clinical_advancement_paper/data/datasets_26.03_w3
W3GRAPH=/gpfs/scratch/bty414/opentarget_evidences/26.03/graph/hetero_graph_with_features_datatype_w3.pt
W3MAP=/gpfs/scratch/bty414/opentarget_evidences/26.03/progression/temporal_graph_datatype_w3_mappings.pt

INJECT="[{\"path\":\"$W3RUNS/gatv2_ablation/both/ensemble/test_predictions.parquet\",\"model_name\":\"gatv2_both_w3\"}]"

python evaluate_advancement.py evaluate \
    --inject "$INJECT" \
    --datasets_dir "$W3ZARR" \
    --graph_file "$W3GRAPH" \
    --mappings_file "$W3MAP" \
    --results_dir headline_results/w3_temporal_decay

echo "Done -> headline_results/w3_temporal_decay/ (temporal_decay.csv + plots/temporal_decay_*.png)"
