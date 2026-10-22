#!/bin/bash
#SBATCH -J bench_04_assemble
#SBATCH -o %x.o%j
#SBATCH -p computeshort
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=96G
#SBATCH -t 1:0:0

# Attach features, check the graph against the v26.09 release, export a
# release directory, and write the T1 benchmark labels into it.
set -euo pipefail
source scripts/benchmark/bench_env.sh

for level in datatype datasource; do
  python preprocessing/temporal_graph/pipeline/attach_features.py \
    --graph-file "${EVENT_OUTPUT_DIR}/temporal_graph_${level}.pt" \
    --output-file "${FINAL_GRAPH_DIR}/hetero_graph_with_features_${level}.pt" \
    --feature-dir "$FEATURE_OUTPUT_DIR"
done

# Hard checks against the release; a failure stops the job before export.
python scripts/benchmark/verify_bench_graph.py \
  --bench-graph "${FINAL_GRAPH_DIR}/hetero_graph_with_features_datatype.pt" \
  --ref-graph "${REF_BASE}/graph/hetero_graph_with_features_datatype.pt" \
  --out "${OUTPUT_BASE}/verify_vs_26.09.json"

# Export into a fresh directory so no file from an earlier build lingers.
rm -rf "${RELEASE_DIR:?}"
python scripts/hf_release/export_hf.py \
  --graph-file "${FINAL_GRAPH_DIR}/hetero_graph_with_features_datatype.pt" \
  --mappings-file "${EVENT_OUTPUT_DIR}/temporal_graph_datatype_mappings.pt" \
  --train-csv "${ADV_DIR}/train_dataset.csv" \
  --test-csv "${ADV_DIR}/test_dataset.csv" \
  --out "$RELEASE_DIR" \
  --score-only

# Continuous graph keeps raw scores; record each relation's theoretical maximum
# so loaders can normalise without fitting anything to the data.
python scripts/benchmark/write_edge_scales.py --release-dir "$RELEASE_DIR" --raw-edges-dir "$RAW_EDGES_DIR"

# Upstream versions: the 26.09 sources plus the ChEMBL release the sequences came from.
python - <<'PY'
import json, os
ref = os.path.join(os.environ["INPUT_EVIDENCE_DIR"], "source_versions.json")
out = os.path.join(os.environ["OUTPUT_BASE"], "hf_release/THBKG/source_versions.json")
v = json.load(open(ref)) if os.path.exists(ref) else {}
v["chembl_biotherapeutics"] = json.load(open(os.path.join(os.environ["OUTPUT_BASE"], "features/raw/chembl_biotherapeutics.json")))
json.dump(v, open(out, "w"), indent=2)
print(f"wrote {out}")
PY

python scripts/benchmark/build_task_labels.py --release-dir "$RELEASE_DIR"
cp -p "${OUTPUT_BASE}/verify_vs_26.09.json" "${FEATURE_OUTPUT_DIR}/molecule_features.yaml" "$RELEASE_DIR/"
# full-width ESM-2 embeddings of the biologics (the graph carries a 64-d PCA)
mkdir -p "$RELEASE_DIR/features"
cp -p "${FEATURE_OUTPUT_DIR}/molecule_esm2_full.safetensors" "${FEATURE_OUTPUT_DIR}/molecule_esm2_full_ids.parquet" \
      "${FEATURE_OUTPUT_DIR}/disease_embeddings.yaml" "$RELEASE_DIR/features/"

echo "[$(date)] ✅ bench_04_assemble done -> ${RELEASE_DIR}"
