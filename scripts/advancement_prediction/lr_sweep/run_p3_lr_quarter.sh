#!/bin/bash
#SBATCH -J p3_lr_quarter
#SBATCH -o %x.o%j
#SBATCH -p andrena
#SBATCH -A pilot_andrena
#SBATCH -n 12
#SBATCH --cpus-per-gpu=12
#SBATCH -t 240:0:0
#SBATCH --mem-per-cpu=7500M
#SBATCH --gres=gpu:1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")}"
cd "$REPO_ROOT"

source .venv/bin/activate
export WANDB_MODE="disabled"

python src/train_advancement_lambdarank.py \
  --config config/experiments/lr_sweep/p3_lr_quarter.yaml
