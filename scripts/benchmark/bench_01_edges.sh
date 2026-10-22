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
# tree and re-parsed. GO annotations and IntAct interactions are re-parsed too:
# their rows are dated through PubMed, and the 26.09 build lost 48,582 GO and
# 105,422 IntAct rows to one failed PubMed request cached as "no year" (the
# lookup now retries and fails loudly; answers are kept in $PUBMED_YEAR_CACHE).
# Every other raw edge parquet is copied from the 26.09 build. The event lists
# are then rebuilt over the full edge set.
#
# The 26.09 event-list step needed 200G (it OOMed at 64G) and ran 2.5 h.
set -euo pipefail
source scripts/benchmark/bench_env.sh

# --- 0. trial resolution (Stage B) from the original download, into this tree
python -u -m preprocessing.temporal_graph.parsers.apply_trial_resolution \
  --cache "$NCT_CACHE" \
  --input-dir "${REF_BASE}/raw_backup/sourceId=clinical_precedence" \
  --output-dir "${RESOLVED_EVIDENCE_DIR}/sourceId=clinical_precedence"

# --- 1. seed raw edges from the release build (minus the files re-parsed below)
mkdir -p "$RAW_EDGES_DIR" "$STATIC_EDGES_DIR"
rm -f "$RAW_EDGES_DIR"/*.parquet
n=0
for f in "$REF_BASE"/evidences/edges/*.parquet; do
  case "$(basename "$f")" in
    *_clinical_precedence.parquet) continue ;;
    target_has_function_in_go_gene_ontology.parquet|target_interacts_with_target_intact.parquet) continue ;;
  esac
  cp -p "$f" "$RAW_EDGES_DIR/"; n=$((n + 1))
done
rm -f "$STATIC_EDGES_DIR"/*.parquet
cp -p "$REF_BASE"/evidences/static_edges/*.parquet "$STATIC_EDGES_DIR/"
echo "Copied ${n} edge files and $(ls "$STATIC_EDGES_DIR" | wc -l) static edge files from ${REF_BASE}"

# --- 2. nodes (all drug types), clinical_precedence edges from the fixed file,
#        and the PubMed-dated GO annotation and IntAct edges
python -u - <<'PY'
import os, sys
sys.path.insert(0, os.path.abspath("preprocessing/temporal_graph"))
from parsers.parser import NodeParser, EdgeParser
from parsers.intact.parser import IntActParser

inp = os.environ["INPUT_EVIDENCE_DIR"] + "/"
out = os.environ["RAW_EDGES_DIR"]
_, node_store = NodeParser(inp, "config/node_schema.yaml", os.environ["RAW_NODES_DIR"], node_store=None).parse()
EdgeParser(os.environ["RESOLVED_EVIDENCE_DIR"] + "/", "config/edge_schema_26.09_clinical_only.yaml",
           out, node_store=node_store).parse()
EdgeParser(inp, "config/edge_schema_26.09_pubmed_dated.yaml", out, node_store=node_store).parse()
IntActParser(inp, "config/edge_schema_26.09_pubmed_dated.yaml", out, node_store=node_store).parse()
PY
ls -la "$RAW_EDGES_DIR"/*_clinical_precedence.parquet "$RAW_EDGES_DIR"/target_has_function_in_go_gene_ontology.parquet \
       "$RAW_EDGES_DIR"/target_interacts_with_target_intact.parquet

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
