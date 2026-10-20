#!/bin/bash
#SBATCH -J bench_05_snapshots
#SBATCH -o %x.o%j
#SBATCH -p computeshort
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=64G
#SBATCH -t 1:0:0

# Discontinuous graph: one snapshot per decision year (1995 .. last edge year + 1)
# built from the exported continuous graph, with raw score and recency per edge.
# Brute-force checks a sample of pairs in three years; then loads one snapshot
# through the shared loader as a smoke test.
set -euo pipefail
source scripts/benchmark/bench_env.sh

rm -rf "${RELEASE_DIR:?}/snapshots"
python scripts/benchmark/build_snapshots.py --release-dir "$RELEASE_DIR"

python - <<'PY'
import os
from src.benchmark.snapshots import load_snapshot
d = load_snapshot(os.environ["RELEASE_DIR"], 2016, normalise="max")
print(d)
for et in d.edge_types:
    a = d[et].edge_attr
    assert a.min() >= 0 and a.max() <= 1 + 1e-6, (et, float(a.min()), float(a.max()))
    assert int(d[et].edge_time.max()) < 2016, et
print("✅ 2016 snapshot loads; max-normalised edge_attr within [0, 1]; no edge dated >= 2016")
PY

echo "[$(date)] ✅ bench_05_snapshots done -> ${RELEASE_DIR}/snapshots"
