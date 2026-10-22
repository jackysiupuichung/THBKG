#!/bin/bash
#SBATCH -J bench_03_features
#SBATCH -o %x.o%j
#SBATCH -p gpushort
#SBATCH -n 1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-cpu=11G
#SBATCH --gres=gpu:1
#SBATCH -t 1:0:0

# Node features for the benchmark graph.
#
# The target node set is unchanged from 26.09 (guarded: aborts if any node set
# differs from the release). Rebuilt here:
#   - targets: from the 26.09 raw inputs, to record which values are
#     mean-imputed (target_features_imputed.parquet); the vectors must match
#     the release's;
#   - GO / Reactome: autoencoders retrained with a learnable input per node (the
#     release fed torch.eye(N, 128), leaving 90% of GO terms with zero input);
#   - molecules: per-modality tensors (Morgan, ESM-2 reduced to 64 by PCA,
#     drug type, presence masks), see molecule_multimodal.py;
#   - diseases: PubMedBERT over "<name>. <description>" at full width (768),
#     padding excluded from the mean (the release kept the first 256 dims,
#     averaged padding in, and embedded missing descriptions as "None").
#
# Needs, prepared on a login node (compute nodes are offline):
#   $CHEMBL_CACHE   from preprocessing/node_features/fetch_chembl_biotherapeutics.py
#   ESM-2 weights   facebook/esm2_t33_650M_UR50D in the HF cache
set -euo pipefail
source scripts/benchmark/bench_env.sh
[ -f "$CHEMBL_CACHE" ] || { echo "❌ missing $CHEMBL_CACHE (run the fetch on a login node)"; exit 1; }

python -u - <<'PY'
import os, torch
ref = torch.load(f"{os.environ['REF_BASE']}/progression/temporal_graph_datatype_mappings.pt", weights_only=False)["node_mapping"]
for level in ("datatype", "datasource"):
    new = torch.load(f"{os.environ['OUTPUT_BASE']}/progression/temporal_graph_{level}_mappings.pt", weights_only=False)["node_mapping"]
    for nt in ("target", "disease", "go", "reactome"):
        assert new[nt] == ref[nt], f"{level}: {nt} node mapping differs from 26.09; cannot reuse its features"
    print(f"{level}: target/disease/go/reactome mappings identical to 26.09; molecule {len(ref['molecule']):,} -> {len(new['molecule']):,}")
PY

mkdir -p "$FEATURE_OUTPUT_DIR"
python -m preprocessing.node_features.build_all_features \
  --mappings-file "${EVENT_OUTPUT_DIR}/temporal_graph_datatype_mappings.pt" \
  --evidence-dir "$INPUT_EVIDENCE_DIR" \
  --feature-data-dir "${REF_BASE}/features/raw" \
  --output-dir "$FEATURE_OUTPUT_DIR" \
  --temp-dir "$TEMP_NODE_DIR" \
  --node-types target

python -u - <<'PY'
import os, torch
ref = torch.load(f"{os.environ['REF_BASE']}/features/processed/integrated_target_features.pt")
new = torch.load(f"{os.environ['FEATURE_OUTPUT_DIR']}/integrated_target_features.pt")
assert ref.keys() == new.keys(), "target feature IDs differ from 26.09"
bad = [k for k in ref if not torch.allclose(ref[k], new[k], atol=1e-5)]
assert not bad, f"{len(bad)} target vectors differ from 26.09, e.g. {bad[:3]}"
print(f"target features: {len(new):,} vectors match 26.09")
PY

python -m preprocessing.node_features.build_all_features \
  --mappings-file "${EVENT_OUTPUT_DIR}/temporal_graph_datatype_mappings.pt" \
  --evidence-dir "$INPUT_EVIDENCE_DIR" \
  --feature-data-dir "$FEATURE_RAW_DIR" \
  --output-dir "$FEATURE_OUTPUT_DIR" \
  --temp-dir "$TEMP_NODE_DIR" \
  --chembl-cache "$CHEMBL_CACHE" \
  --node-types molecule disease go reactome

echo "[$(date)] ✅ bench_03_features done"
