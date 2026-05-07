#!/bin/bash
# Submit the 2-row LR sweep for p3_eahgt_both, with early stopping disabled
# so the full 50-epoch trajectory is captured.
#
# Goal: test the overfitting hypothesis. If test_rr_ta_mean@50 still peaks
# at epoch ~2 with halved/quartered LR, the brittle-val-signal story stands.
# If it peaks later (epoch 4–6), LR is the lever and we should retune.
#
# Both runs land on the gpu partition (240h cap; expect ~4h for 50 epochs).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$REPO_ROOT"

SWEEP_DIR="$SCRIPT_DIR/lr_sweep"

for job in \
  run_p3_lr_half.sh \
  run_p3_lr_quarter.sh; do
    echo "Submitting $job"
    sbatch "$SWEEP_DIR/$job"
done

echo "Both jobs submitted. Use 'squeue -u $USER' to monitor."
