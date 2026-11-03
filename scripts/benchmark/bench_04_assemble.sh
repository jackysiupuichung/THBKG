#!/bin/bash
#SBATCH -J bench_04_assemble
#SBATCH -o %x.o%j
#SBATCH -p computeshort
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=96G
#SBATCH -t 1:0:0

# Attach features, export a release directory, write the benchmark labels (T1,
# T2, T3, therapeutic areas), the per-source evidence scores and the Croissant
# metadata into it, and check the release on its own: against the release
# contract and against its own Open Targets input. Nothing is compared with
# another release.
set -euo pipefail
source scripts/benchmark/bench_env.sh

for level in datatype datasource; do
  python preprocessing/temporal_graph/pipeline/attach_features.py \
    --graph-file "${EVENT_OUTPUT_DIR}/temporal_graph_${level}.pt" \
    --output-file "${FINAL_GRAPH_DIR}/hetero_graph_with_features_${level}.pt" \
    --feature-dir "$FEATURE_OUTPUT_DIR"
done

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
# per-source target-disease scores, for baselines that need the sources apart
python scripts/benchmark/write_source_scores.py \
  --events "${EVENT_OUTPUT_DIR}/events_datasource.parquet" --release-dir "$RELEASE_DIR"
cp -p "${FEATURE_OUTPUT_DIR}/molecule_features.yaml" "$RELEASE_DIR/"
# raw ESM-2 embeddings of the biologics (the graph carries them centred and scaled)
mkdir -p "$RELEASE_DIR/features"
cp -p "${FEATURE_OUTPUT_DIR}/molecule_esm2_full.safetensors" "${FEATURE_OUTPUT_DIR}/molecule_esm2_full_ids.parquet" \
      "${FEATURE_OUTPUT_DIR}/disease_embeddings.yaml" "${FEATURE_OUTPUT_DIR}/target_features_imputed.parquet" \
      "$RELEASE_DIR/features/"
# Croissant metadata with the sha256 of every file it lists, now that all are in place
python scripts/hf_release/write_croissant.py --release-dir "$RELEASE_DIR"

# What every release must hold whatever the version (config/release_contract.yaml),
# and that it holds all of its own Open Targets input. A failure stops the chain
# before the snapshots.
python scripts/benchmark/check_release.py --release-dir "$RELEASE_DIR" --ot-version "$OT_VERSION"
python scripts/benchmark/check_inputs.py --release-dir "$RELEASE_DIR" --evidence-dir "$INPUT_EVIDENCE_DIR" \
  --trials-dir "$RAW_TRIALS_DIR" --pubmed-cache "$PUBMED_YEAR_CACHE" --edge-schema "config/edge_schema_${OT_VERSION}.yaml"

# safetensors writes 0600 files; make the release readable to the group and others
chmod -R a+rX "$RELEASE_DIR"

echo "[$(date)] ✅ bench_04_assemble done -> ${RELEASE_DIR}"
