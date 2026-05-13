#!/bin/bash
# Submit the val-window experiment: 5 val-window variants × 3 seeds = 15 jobs.
# Each variant retrains p3_eahgt_both with a different val_min/max_year (train
# cutoff and test window held constant). The goal is to find whether a
# narrower, more test-adjacent val window makes the val_ndcg/rr signal
# predictive of test_rr (Spearman ρ > 0.5, selection regret < 1.0).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$REPO_ROOT"

SWEEP_DIR="$SCRIPT_DIR/val_window_sweep"

JOBS=("run_v1_2011_15_s42.sh" "run_v1_2011_15_s17.sh" "run_v1_2011_15_s123.sh" "run_v2_2013_15_s42.sh" "run_v2_2013_15_s17.sh" "run_v2_2013_15_s123.sh" "run_v3_2014_15_s42.sh" "run_v3_2014_15_s17.sh" "run_v3_2014_15_s123.sh" "run_v4_2015_s42.sh" "run_v4_2015_s17.sh" "run_v4_2015_s123.sh" "run_v5_2012_13_s42.sh" "run_v5_2012_13_s17.sh" "run_v5_2012_13_s123.sh")

for job in "${JOBS[@]}"; do
    echo "Submitting $job"
    sbatch "$SWEEP_DIR/$job"
done

echo "All ${#JOBS[@]} jobs submitted. Use 'squeue -u $USER' to monitor."
