#!/bin/bash
#$ -pe smp 8
#$ -l h_vmem=64G
#$ -l h_rt=24:0:0
#$ -cwd
#$ -j y

set -euo pipefail

# Activate venv
source .venv/bin/activate

# === Configuration ===
CONFIG="config/event_graph_config.yaml"

# --- Step 0 Config (Raw Input) ---
# Default to cluster path, but allow override via env var
INPUT_EVIDENCE_DIR="data/evidenceDated_subset/23.06"
NODE_SCHEMA="config/node_schema.yaml"
EDGE_SCHEMA="config/edge_schema.yaml"
STATIC_EDGE_SCHEMA="config/static_edge_schema.yaml"

# Output Directories
OUTPUT_BASE="output"

# Step 0 Output
KG_OUTPUT_DIR="${OUTPUT_BASE}/evidences"
RAW_EDGES_DIR="${KG_OUTPUT_DIR}/edges"
RAW_NODES_DIR="${KG_OUTPUT_DIR}/nodes"
STATIC_EDGES_DIR="${KG_OUTPUT_DIR}/static_edges"

# Step 1-4 Outputs
EVENT_OUTPUT_DIR="${OUTPUT_BASE}/progression"
FEATURE_RAW_DIR="data/node_features"    # External raw features (RNA, etc.)
FEATURE_OUTPUT_DIR="${OUTPUT_BASE}/features/processed"
FINAL_GRAPH_DIR="${OUTPUT_BASE}/graph"

# Files
EVENTS_FILE="${EVENT_OUTPUT_DIR}/events.parquet"
GRAPH_STRUCT_FILE="${EVENT_OUTPUT_DIR}/temporal_graph_structure.pt"
FINAL_GRAPH_FILE="${FINAL_GRAPH_DIR}/hetero_graph_with_features.pt"

# === 0. KG Pipeline (Raw Evidence -> Nodes/Edges) ===
echo "🚀 [0/5] Running KG Pipeline..."
echo "   Input: $INPUT_EVIDENCE_DIR"
echo "   Output: $KG_OUTPUT_DIR"

python -m src.pipeline.kg_pipeline \
  --input "$INPUT_EVIDENCE_DIR" \
  --node-schema "$NODE_SCHEMA" \
  --edge-schema "$EDGE_SCHEMA" \
  --static-edge-schema "$STATIC_EDGE_SCHEMA" \
  --node-output "$RAW_NODES_DIR" \
  --edge-output "$RAW_EDGES_DIR" \
  --static-edge-output "$STATIC_EDGES_DIR"

# === 1. Build Event List ===
echo "🚀 [1/5] Building Event List..."
if [ ! -f "$CONFIG" ]; then echo "❌ Config $CONFIG not found!"; exit 1; fi

python -m src.pipeline.build_event_list \
  --input-dir "$RAW_EDGES_DIR" \
  --config "$CONFIG" \
  --output "$EVENTS_FILE" \
  --aggregation-method "harmonic_sum"

# === 2. Build Graph Structure ===
echo "🚀 [2/5] Building Graph Structure (Nodes + Edges)..."
# Note: passing static edges if available? 
# build_event_graph supports --static-edges. Let's include it.
python -m src.pipeline.build_event_graph \
  --input "$EVENTS_FILE" \
  --output "$GRAPH_STRUCT_FILE" \
  --static-edges "$STATIC_EDGES_DIR"

# === 3. Build Node Features ===
echo "🚀 [3/5] Building Node Features..."
# This invokes target_features, disease_description, molecule_structure
python -m src.node_features.build_all_features \
  --node-dir "$RAW_NODES_DIR" \
  --feature-data-dir "$FEATURE_RAW_DIR" \
  --output-dir "$FEATURE_OUTPUT_DIR"

# === 4. Attach Features to Graph ===
echo "🚀 [4/5] Attaching Features to Graph..."
python -m src.pipeline.attach_features \
  --graph-file "$GRAPH_STRUCT_FILE" \
  --output-file "$FINAL_GRAPH_FILE" \
  --feature-dir "$FEATURE_OUTPUT_DIR" 

# === 5. Analysis ===
echo "🚀 [5/5] Analyze Final Graph..."
python -m src.data.analyze_graph \
  --file "$FINAL_GRAPH_FILE" \
  --output "${OUTPUT_BASE}/analysis"

echo "✅ Full Graph Build Complete!"
echo "   Final Graph: $FINAL_GRAPH_FILE"
