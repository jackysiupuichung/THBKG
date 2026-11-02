#!/bin/bash
#SBATCH -J bench_01_edges
#SBATCH -o %x.o%j
#SBATCH -p compute
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=200G
#SBATCH -t 12:0:0

# Raw edges + event lists for the benchmark graph.
#
# Every edge is parsed from the Open Targets download with the current parsers,
# as preprocessing/temporal_graph/pipeline/kg_pipeline.py does. Nothing is copied
# from the 26.09 build, so a parser change cannot miss a source: copying its
# Europe PMC parse kept literature dated by publicationYear after 29fef74, and
# scripts/benchmark/check_inputs.py found 97,956 pairs with another first year.
#
# Clinical trials are parsed from this tree's own trial resolution of the
# original download, not from the 26.09 build's: all drug types are molecule
# nodes (modulated_by); the start-dated ongoing edge carries the phase weight,
# not Open Targets' score, which is halved for trials that later stopped for
# safety or efficacy (a leak of the outcome into the start year); trials of
# UNKNOWN status get no outcome edge; trials with no start date are left out.
# GO annotations and IntAct are dated through PubMed: the lookup retries and
# fails loudly (the 26.09 build lost 48,582 GO and 105,422 IntAct rows to one
# failed request cached as "no year"), and answers are kept in $PUBMED_YEAR_CACHE.
# The event lists are then built over the full edge set.
#
# The 26.09 event-list step needed 200G (it OOMed at 64G) and ran 2.5 h; the
# full parse took about 2 h of the 26.09 build.
set -euo pipefail
source scripts/benchmark/bench_env.sh

# --- 0. trial resolution (Stage B) from the original download, into this tree
python -u -m preprocessing.temporal_graph.parsers.apply_trial_resolution \
  --cache "$NCT_CACHE" \
  --input-dir "${REF_BASE}/raw_backup/sourceId=clinical_precedence" \
  --output-dir "${RESOLVED_EVIDENCE_DIR}/sourceId=clinical_precedence"

# --- 1. nodes and every edge, parsed into this tree. The edge schema is split:
#        clinical_precedence is read from the trial resolution above, every
#        other source from the download. Both halves are written next to the
#        edges as a record of what was parsed.
mkdir -p "$RAW_EDGES_DIR" "$STATIC_EDGES_DIR" "${KG_OUTPUT_DIR}/schemas"
rm -f "$RAW_EDGES_DIR"/*.parquet "$STATIC_EDGES_DIR"/*.parquet
python -u - <<'PY'
import os, sys
import yaml
sys.path.insert(0, os.path.abspath("preprocessing/temporal_graph"))
from parsers.parser import NodeParser, EdgeParser
from parsers.intact.parser import IntActParser
from parsers.go_ontology.parser import GOOntologyParser

inp = os.environ["INPUT_EVIDENCE_DIR"] + "/"
out, static_out = os.environ["RAW_EDGES_DIR"], os.environ["STATIC_EDGES_DIR"]
edge_schema = f"config/edge_schema_{os.environ['OT_VERSION']}.yaml"
static_schema = "config/static_edge_schema.yaml"
schema = yaml.safe_load(open(edge_schema))
split = {"clinical": {k: v for k, v in schema.items() if k == "clinical_precedence"},
         "evidence": {k: v for k, v in schema.items() if k != "clinical_precedence"}}
paths = {}
for part, spec in split.items():
    paths[part] = os.path.join(os.environ["KG_OUTPUT_DIR"], "schemas", f"edge_schema_{part}.yaml")
    with open(paths[part], "w") as f:
        f.write(f"# {part} part of {edge_schema}, written by bench_01_edges.sh\n")
        yaml.safe_dump(spec, f, sort_keys=False)
print(f"{edge_schema}: {len(split['evidence'])} sources from {inp}, clinical_precedence from the trial resolution")

_, node_store = NodeParser(inp, "config/node_schema.yaml", os.environ["RAW_NODES_DIR"], node_store=None).parse()
EdgeParser(os.environ["RESOLVED_EVIDENCE_DIR"] + "/", paths["clinical"], out, node_store=node_store).parse()
EdgeParser(inp, paths["evidence"], out, node_store=node_store).parse()
IntActParser(inp, edge_schema, out, node_store=node_store).parse()
GOOntologyParser(inp, static_schema, static_out, node_store=node_store, static=True).parse()
EdgeParser(inp, static_schema, static_out, node_store=node_store, static=True).parse()
PY
echo "Parsed $(ls "$RAW_EDGES_DIR"/*.parquet | wc -l) edge files and $(ls "$STATIC_EDGES_DIR"/*.parquet | wc -l) static edge files"
ls -la "$RAW_EDGES_DIR" "$STATIC_EDGES_DIR"

# --- 2. event lists (same two levels as scripts/collecting_edges_01.sh)
CONFIG="config/event_graph_config.yaml"
python preprocessing/temporal_graph/pipeline/build_event_list.py \
  --input-dir "$RAW_EDGES_DIR" --config "$CONFIG" \
  --output "${EVENT_OUTPUT_DIR}/events_datatype.parquet" \
  --datatype-mapping "config/datatype_mapping.yaml"
python preprocessing/temporal_graph/pipeline/build_event_list.py \
  --input-dir "$RAW_EDGES_DIR" --config "$CONFIG" \
  --output "${EVENT_OUTPUT_DIR}/events_datasource.parquet"

echo "[$(date)] ✅ bench_01_edges done"
