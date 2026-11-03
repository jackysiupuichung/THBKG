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
# Every node type's features are rebuilt here, for this graph's own node sets:
#   - targets: from the Open Targets 26.09 and Human Protein Atlas raw inputs,
#     recording which values are mean-imputed (target_features_imputed.parquet);
#   - GO / Reactome: autoencoders retrained with a learnable input per node (the
#     release fed torch.eye(N, 128), leaving 90% of GO terms with zero input);
#   - molecules: per-modality tensors (Morgan, ESM-2 at full width (1,280),
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

mkdir -p "$FEATURE_OUTPUT_DIR"
python -m preprocessing.node_features.build_all_features \
  --mappings-file "${EVENT_OUTPUT_DIR}/temporal_graph_datatype_mappings.pt" \
  --evidence-dir "$INPUT_EVIDENCE_DIR" \
  --feature-data-dir "${REF_BASE}/features/raw" \
  --output-dir "$FEATURE_OUTPUT_DIR" \
  --temp-dir "$TEMP_NODE_DIR" \
  --node-types target

python -m preprocessing.node_features.build_all_features \
  --mappings-file "${EVENT_OUTPUT_DIR}/temporal_graph_datatype_mappings.pt" \
  --evidence-dir "$INPUT_EVIDENCE_DIR" \
  --feature-data-dir "$FEATURE_RAW_DIR" \
  --output-dir "$FEATURE_OUTPUT_DIR" \
  --temp-dir "$TEMP_NODE_DIR" \
  --chembl-cache "$CHEMBL_CACHE" \
  --node-types molecule disease go reactome

echo "[$(date)] ✅ bench_03_features done"
