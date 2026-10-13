#!/bin/bash
#SBATCH -J building_event_graph_02
#SBATCH -o %x.o%j
#SBATCH -p compute
#SBATCH -n 1
#SBATCH -t 1:0:0
#SBATCH --mem-per-cpu=32G

set -euo pipefail

# Activate venv
source .venv/bin/activate

# === Configuration ===
OT_VERSION="${OT_VERSION:-26.09}"
OUTPUT_BASE="/gpfs/scratch/bty414/opentarget_evidences/${OT_VERSION}"
KG_OUTPUT_DIR="${OUTPUT_BASE}/evidences"
STATIC_EDGES_DIR="${KG_OUTPUT_DIR}/static_edges"
RAW_EDGES_DIR="${KG_OUTPUT_DIR}/edges"
EVENT_OUTPUT_DIR="${OUTPUT_BASE}/progression"

# Advancement datasets — w3 window of the same OT release. Passed here to
# REGISTER their target/disease nodes in the graph; the outcome labels
# themselves are NOT written into the graph and are loaded externally at
# train/eval time (see src/data/temporal_loader.py: load_advancement_labels).
# The 23.06 label CSVs had a same-year clinical-trial-edge leak and are
# retired. Labels must come from the same release as the graph, since disease
# IDs differ between releases. See data/clinical_trial_advancement/.
ADV_DIR="data/clinical_trial_advancement/${OT_VERSION}_w3"
ADV_TRAIN="${ADV_DIR}/train_dataset.csv"
ADV_TEST="${ADV_DIR}/test_dataset.csv"
for f in "$ADV_TRAIN" "$ADV_TEST"; do
  if [ ! -f "$f" ]; then echo "❌ Advancement labels not found: $f"; exit 1; fi
done

# === Build Graph Structures (datasource-level and datatype-level) ===

# --- Datasource-level ---
echo "🚀 Building Event Graph (datasource-level)..."
python preprocessing/temporal_graph/pipeline/build_event_graph.py \
  --input "${EVENT_OUTPUT_DIR}/events_datasource.parquet" \
  --output "${EVENT_OUTPUT_DIR}/temporal_graph_datasource.pt" \
  --static-edges "$STATIC_EDGES_DIR" \
  --raw-edges "$RAW_EDGES_DIR" \
  --advancement-train-csv "$ADV_TRAIN" \
  --advancement-test-csv "$ADV_TEST" \
  --edge-type-mode relation_only

echo "✅ Datasource graph: ${EVENT_OUTPUT_DIR}/temporal_graph_datasource.pt"

# --- Datatype-level ---
echo ""
echo "🚀 Building Event Graph (datatype-level)..."
python preprocessing/temporal_graph/pipeline/build_event_graph.py \
  --input "${EVENT_OUTPUT_DIR}/events_datatype.parquet" \
  --output "${EVENT_OUTPUT_DIR}/temporal_graph_datatype.pt" \
  --static-edges "$STATIC_EDGES_DIR" \
  --raw-edges "$RAW_EDGES_DIR" \
  --advancement-train-csv "$ADV_TRAIN" \
  --advancement-test-csv "$ADV_TEST" \
  --edge-type-mode relation_only

echo "✅ Datatype graph: ${EVENT_OUTPUT_DIR}/temporal_graph_datatype.pt"
