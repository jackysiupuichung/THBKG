#!/bin/bash
# Submit the 4-row early-stopping metric sweep for p3_eahgt_both.
# All 4 jobs share architecture, hyperparameters, and lambdarank.ndcg_k=50.
# Only the early_stopping.metric varies:
#   - ndcg@10        (current baseline)
#   - ndcg@50        (matches lambdarank.ndcg_k)
#   - ndcg_ta_mean@10
#   - ndcg_ta_mean@50  (primary candidate)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$REPO_ROOT"

SWEEP_DIR="$SCRIPT_DIR/early_stop_sweep"

for job in \
  run_p3_es_ndcg10_flat.sh \
  run_p3_es_ndcg50_flat.sh \
  run_p3_es_ndcgta10.sh \
  run_p3_es_ndcgta50.sh; do
    echo "Submitting $job"
    sbatch "$SWEEP_DIR/$job"
done

echo "All 4 jobs submitted. Use 'squeue -u $USER' to monitor."
