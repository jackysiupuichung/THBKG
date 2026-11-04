#!/bin/bash
#SBATCH -J smoke_train
#SBATCH -o %x.o%j
#SBATCH -p gpushort
#SBATCH -A pilot
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -t 1:0:0

# Smoke-train a small GNN on the release and score T1 and T2 on the last
# model-selection fold (val block 2016-2019 inside train)
# (scripts/benchmark/smoke_train.py): the release, its loader and the harness
# work end to end on a GPU. Results in ${OUTPUT_BASE}/smoke/.
set -euo pipefail
source scripts/benchmark/bench_env.sh
export PYTHONDONTWRITEBYTECODE=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for task in discovery hypothesis; do
  python scripts/benchmark/smoke_train.py --release-dir "$RELEASE_DIR" --task "$task" \
    --out "${OUTPUT_BASE}/smoke/${task}_cv_fold.json"
done
echo "[$(date)] ✅ smoke_train done"
