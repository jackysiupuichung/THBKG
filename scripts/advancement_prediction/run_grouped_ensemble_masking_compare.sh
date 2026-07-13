#!/bin/bash
#SBATCH --job-name=grpens_maskcmp
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
OUT_DIR=headline_results/grouped_ensemble_masking_compare_eval

# strict ensemble parquet already built by build_grouped_ensemble_strictmask.py.
# loose ensemble = existing lr_grouped_k100_latest fusion; alias it under a
# distinct name so it labels as EAHGT-loose (latest_s5 maps to plain EAHGT).
STRICT=$RUNS/grouped_ensemble_strictmask_s5/test_predictions.parquet
LOOSE_SRC=$RUNS/grouped_ensemble_latest_s5/test_predictions.parquet
LOOSE=$RUNS/grouped_ensemble_loose_s5/test_predictions.parquet
mkdir -p "$(dirname "$LOOSE")"
cp -f "$LOOSE_SRC" "$LOOSE"

# One eval, both EA-HGT variants injected; RDG/OTS auto-added as references.
uv run python evaluate_advancement.py evaluate \
  --results_dir "$OUT_DIR" \
  --only "" \
  --inject "[{\"path\": \"$STRICT\", \"model_name\": \"grouped_ensemble_strictmask_s5\"}, {\"path\": \"$LOOSE\", \"model_name\": \"grouped_ensemble_loose_s5\"}]"

echo "DONE. Eval outputs in $OUT_DIR"
