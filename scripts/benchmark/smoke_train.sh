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

# Smoke-train a small GNN on the release with each training recipe and score T1
# and T2 on val (fit on what is dated before 2017, score 2017-2019)
# (scripts/benchmark/smoke_train.py): the release, its loader and the harness
# work end to end on a GPU. Results in ${OUTPUT_BASE}/smoke/, one record per
# run in results.json.
set -euo pipefail
source scripts/benchmark/bench_env.sh
export PYTHONDONTWRITEBYTECODE=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
rm -f "${OUTPUT_BASE}/smoke/results.json"
for task in discovery hypothesis; do
  for recipe in single stacked; do
    python scripts/benchmark/smoke_train.py --release-dir "$RELEASE_DIR" --task "$task" --recipe "$recipe" \
      --out "${OUTPUT_BASE}/smoke/${task}_${recipe}_val.json"
  done
done
echo "[$(date)] ✅ smoke_train done"
