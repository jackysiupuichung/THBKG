#!/bin/bash
#SBATCH -J bench_01_edges
#SBATCH -o %x.o%j
#SBATCH -p compute
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=200G
#SBATCH -t 8:0:0

# Raw edges + event lists for the benchmark graph.
#
# Only clinical_precedence edges change:
#   - modulated_by, because all drug types are now molecule nodes;
#   - the target->disease trial buckets, because the start-dated ongoing edge
#     now carries the phase weight instead of Open Targets' score, which is
#     halved for trials that later stopped for safety or efficacy (a leak of
#     the trial's outcome into its start year).
# So clinical_precedence is re-resolved from the original download into this
# tree and re-parsed; every other raw edge parquet is copied from the 26.09
# build, keeping all non-trial evidence byte-identical to the release. The
# event lists are then rebuilt over the full edge set.
#
# The 26.09 event-list step needed 200G (it OOMed at 64G) and ran 2.5 h.
set -euo pipefail
source scripts/benchmark/bench_env.sh

# --- 0. trial resolution (Stage B) from the original download, into this tree
python -u -m preprocessing.temporal_graph.parsers.apply_trial_resolution \
  --cache "$NCT_CACHE" \
  --input-dir "${REF_BASE}/raw_backup/sourceId=clinical_precedence" \
  --output-dir "${RESOLVED_EVIDENCE_DIR}/sourceId=clinical_precedence"

# --- 1. seed raw edges from the release build (minus clinical_precedence files)
mkdir -p "$RAW_EDGES_DIR" "$STATIC_EDGES_DIR"
rm -f "$RAW_EDGES_DIR"/*.parquet
n=0
for f in "$REF_BASE"/evidences/edges/*.parquet; do
  case "$(basename "$f")" in *_clinical_precedence.parquet) continue ;; esac
  cp -p "$f" "$RAW_EDGES_DIR/"; n=$((n + 1))
done
cp -p "$REF_BASE"/evidences/static_edges/*.parquet "$STATIC_EDGES_DIR/"
echo "Copied ${n} edge files and $(ls "$STATIC_EDGES_DIR" | wc -l) static edge files from ${REF_BASE}"

# --- 2. nodes (all drug types) + clinical_precedence edges from the fixed file
python -u - <<'PY'
import os, sys
sys.path.insert(0, os.path.abspath("preprocessing/temporal_graph"))
from parsers.parser import NodeParser, EdgeParser

inp = os.environ["INPUT_EVIDENCE_DIR"] + "/"
_, node_store = NodeParser(inp, "config/node_schema.yaml", os.environ["RAW_NODES_DIR"], node_store=None).parse()
EdgeParser(os.environ["RESOLVED_EVIDENCE_DIR"] + "/", "config/edge_schema_26.09_clinical_only.yaml",
           os.environ["RAW_EDGES_DIR"], node_store=node_store).parse()
PY
ls -la "$RAW_EDGES_DIR"/*_clinical_precedence.parquet

# --- 3. event lists (same two levels as scripts/collecting_edges_01.sh)
CONFIG="config/event_graph_config.yaml"
python preprocessing/temporal_graph/pipeline/build_event_list.py \
  --input-dir "$RAW_EDGES_DIR" --config "$CONFIG" \
  --output "${EVENT_OUTPUT_DIR}/events_datatype.parquet" \
  --datatype-mapping "config/datatype_mapping.yaml"
python preprocessing/temporal_graph/pipeline/build_event_list.py \
  --input-dir "$RAW_EDGES_DIR" --config "$CONFIG" \
  --output "${EVENT_OUTPUT_DIR}/events_datasource.parquet"

echo "[$(date)] ✅ bench_01_edges done"
