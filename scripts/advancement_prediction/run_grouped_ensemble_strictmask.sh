#!/bin/bash
#SBATCH --job-name=grpens_strictmask
#SBATCH --partition=computeshort
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=1:0:0
#SBATCH --output=%x.o%j

set -euo pipefail
cd /data/home/bty414/opentarget_temporal_study/src/opentarget_het_graph

RUNS=/gpfs/scratch/bty414/opentarget_evidences/26.03/runs
OUT_DIR=headline_results/grouped_ensemble_strictmask_eval

# 1. Build the strict-mask (< transition_year) 5-seed percentile-rank ensemble parquet
uv run python scripts/advancement_prediction/build_grouped_ensemble_strictmask.py

# 2. Evaluate it via official evaluate_advancement (inject the ensemble parquet).
uv run python evaluate_advancement.py evaluate \
  --results_dir "$OUT_DIR" \
  --only "" \
  --inject "[{\"path\": \"$RUNS/grouped_ensemble_strictmask_s5/test_predictions.parquet\", \"model_name\": \"grouped_ensemble_strictmask_s5\"}]"

echo "DONE. Eval outputs in $OUT_DIR"
