#!/bin/bash
#SBATCH -J tune_advancement_hgt_leaked
#SBATCH -o %x.o%j
#SBATCH -n 8
#SBATCH --cpus-per-gpu=8
#SBATCH -t 240:0:0
#SBATCH --mem-per-cpu=11G
#SBATCH --gres=gpu:1
#SBATCH -p sae
#SBATCH -A pilot_sae_gpu

# W&B sweep-based hyperparameter tuning — LEAKED/ORACLE variant.
#
# Trains on train+val, evaluates on test. Gives a theoretical upper bound
# on performance when hyperparameters are selected with full test-set knowledge.
#
# Usage:
#   sbatch scripts/advancement_prediction/tune_advancement_hgt_leaked.sh [options]
#
# Options:
#   --config PATH        Experiment config with a tune: section (default: p3_eahgt_both.yaml)
#   --sweep_id ID        Join an existing W&B sweep instead of creating a new one
#   --n_trials N         Max trials for this agent run
#   --output_dir PATH    Override output directory
#   --entity NAME        W&B entity
#   --wandb              Enable W&B cloud sync (default: offline)

set -euo pipefail

REPO_ROOT="/data/home/bty414/opentarget_temporal_study/src/opentarget_het_graph"
cd "$REPO_ROOT"

source .venv/bin/activate

# ── Defaults ──────────────────────────────────────────────────────────────────
CONFIG="config/experiments/p3_eahgt_both.yaml"
WANDB_MODE_VAL="offline"
SWEEP_ID=""
N_TRIALS=""
OUTPUT_DIR=""
ENTITY=""

# ── Parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)      CONFIG="$2";      shift 2 ;;
        --sweep_id)    SWEEP_ID="$2";    shift 2 ;;
        --n_trials)    N_TRIALS="$2";    shift 2 ;;
        --output_dir)  OUTPUT_DIR="$2";  shift 2 ;;
        --entity)      ENTITY="$2";      shift 2 ;;
        --wandb)       WANDB_MODE_VAL="online"; shift ;;
        *)             echo "Unknown argument: $1"; exit 1 ;;
    esac
done

echo "Config     : $CONFIG"
echo "W&B        : $WANDB_MODE_VAL"
echo "[LEAKED]   : train=train+val, eval=test (theoretical upper bound)"
[[ -n "$SWEEP_ID"   ]] && echo "sweep_id   : $SWEEP_ID"
[[ -n "$N_TRIALS"   ]] && echo "n_trials   : $N_TRIALS (override)"
[[ -n "$OUTPUT_DIR" ]] && echo "output_dir : $OUTPUT_DIR (override)"
[[ -n "$ENTITY"     ]] && echo "entity     : $ENTITY"
echo "--------------------------------------------"

# ── Environment ───────────────────────────────────────────────────────────────
export WANDB_MODE="$WANDB_MODE_VAL"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

# ── Build command ─────────────────────────────────────────────────────────────
CMD=(
    python src/tune_advancement_hgt_leaked.py
    --config "$CONFIG"
    --create_sweep
)

[[ -n "$SWEEP_ID"   ]] && CMD+=(--sweep_id   "$SWEEP_ID")
[[ -n "$N_TRIALS"   ]] && CMD+=(--n_trials   "$N_TRIALS")
[[ -n "$OUTPUT_DIR" ]] && CMD+=(--output_dir "$OUTPUT_DIR")
[[ -n "$ENTITY"     ]] && CMD+=(--entity     "$ENTITY")

"${CMD[@]}"
