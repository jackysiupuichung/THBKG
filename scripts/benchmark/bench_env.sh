# Shared paths for the benchmark graph build (sourced by bench_0*.sh).
#
# Reads the 26.09 release build (REF_BASE) and writes a separate tree
# (OUTPUT_BASE), so the published v26.09 artifacts are never touched. The
# clinical_precedence input is already trial-resolved (Stage B), so Stage B is
# not re-run here.

OT_VERSION="26.09"
BUILD_TAG="${BUILD_TAG:-bench}"
REF_BASE="/gpfs/scratch/bty414/opentarget_evidences/${OT_VERSION}"
OUTPUT_BASE="${REF_BASE}_${BUILD_TAG}"
if [ -z "$BUILD_TAG" ] || [ "$OUTPUT_BASE" = "$REF_BASE" ]; then
  echo "❌ BUILD_TAG must be non-empty: refusing to write into ${REF_BASE}"; exit 1
fi

INPUT_EVIDENCE_DIR="${REF_BASE}/evidenceDated"
KG_OUTPUT_DIR="${OUTPUT_BASE}/evidences"
RAW_EDGES_DIR="${KG_OUTPUT_DIR}/edges"
RAW_NODES_DIR="${KG_OUTPUT_DIR}/nodes"
STATIC_EDGES_DIR="${KG_OUTPUT_DIR}/static_edges"
EVENT_OUTPUT_DIR="${OUTPUT_BASE}/progression"
FEATURE_RAW_DIR="${OUTPUT_BASE}/features/raw"
FEATURE_OUTPUT_DIR="${OUTPUT_BASE}/features/processed"
TEMP_NODE_DIR="${OUTPUT_BASE}/features/temp_nodes"
FINAL_GRAPH_DIR="${OUTPUT_BASE}/graph"
RELEASE_DIR="${OUTPUT_BASE}/hf_release/THBKG"
CHEMBL_CACHE="${FEATURE_RAW_DIR}/chembl_biotherapeutics.parquet"
ADV_DIR="data/clinical_trial_advancement/${OT_VERSION}_w3"

export OT_VERSION REF_BASE OUTPUT_BASE INPUT_EVIDENCE_DIR RAW_EDGES_DIR RAW_NODES_DIR
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1  # compute nodes are offline; ESM-2 must already be in the HF cache

source .venv/bin/activate
echo "[$(date)] ${SLURM_JOB_NAME:-local}: ${REF_BASE} -> ${OUTPUT_BASE}"
