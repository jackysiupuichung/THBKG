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
# target/go/reactome node sets are unchanged from 26.09, so their processed
# features are copied (the GO/Reactome autoencoders are not re-trained, which
# would change them). The copy is guarded: it aborts if any of those node sets
# differs from the release. Rebuilt here:
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
for f in integrated_target_features.pt \
         go_embeddings.pt go_embeddings_full.pt reactome_embeddings.pt reactome_embeddings_full.pt; do
  cp -p "${REF_BASE}/features/processed/${f}" "$FEATURE_OUTPUT_DIR/"
done

python -m preprocessing.node_features.build_all_features \
  --mappings-file "${EVENT_OUTPUT_DIR}/temporal_graph_datatype_mappings.pt" \
  --evidence-dir "$INPUT_EVIDENCE_DIR" \
  --feature-data-dir "$FEATURE_RAW_DIR" \
  --output-dir "$FEATURE_OUTPUT_DIR" \
  --temp-dir "$TEMP_NODE_DIR" \
  --chembl-cache "$CHEMBL_CACHE" \
  --node-types molecule disease

echo "[$(date)] ✅ bench_03_features done"
