#!/bin/bash
#SBATCH -J export_hf
#SBATCH -o %x.o%j
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=32G
#SBATCH -t 0:30:0

# Export one OT release's w3 graph to a HuggingFace-ready release directory.
# Produces pickle-free graph.safetensors + metadata.json, node-id mapping and
# label parquets, and stages load_thbkg.py / the dataset card / croissant.json.
# Submit from the repo root: OT_VERSION=26.09 sbatch scripts/hf_release/export_hf.sh
set -euo pipefail
source .venv/bin/activate

OT_VERSION="${OT_VERSION:-26.09}"
BASE="/gpfs/scratch/bty414/opentarget_evidences/${OT_VERSION}"
OUT="${BASE}/hf_release/THBKG"

# 26.03 was relabelled to w3 after its first build, so its w3 files carry a
# _w3 suffix; later releases are built w3 from the start.
SUFFIX=""
[ "$OT_VERSION" = "26.03" ] && SUFFIX="_w3"
LABELS="data/clinical_trial_advancement/${OT_VERSION}_w3"

echo "[$(date)] Exporting HF release (OT ${OT_VERSION}) -> ${OUT}"
python scripts/hf_release/export_hf.py \
  --graph-file "${BASE}/graph/hetero_graph_with_features_datatype${SUFFIX}.pt" \
  --mappings-file "${BASE}/progression/temporal_graph_datatype${SUFFIX}_mappings.pt" \
  --train-csv "${LABELS}/train_dataset.csv" \
  --test-csv "${LABELS}/test_dataset.csv" \
  --out "$OUT"

# Pinned upstream source versions, recorded by scripts/download.sh.
if [ -f "${BASE}/evidenceDated/source_versions.json" ]; then
  cp "${BASE}/evidenceDated/source_versions.json" "${OUT}/source_versions.json"
fi
echo "[$(date)] DONE -> ${OUT}"
