#!/bin/bash
#SBATCH -J bench_02_graph
#SBATCH -o %x.o%j
#SBATCH -p computeshort
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=96G
#SBATCH -t 1:0:0

# HeteroData graphs from the benchmark event lists (as scripts/building_event_graph_02.sh).
# Advancement CSVs only register their endpoints as nodes; labels stay external.
set -euo pipefail
source scripts/benchmark/bench_env.sh

for level in datatype datasource; do
  python preprocessing/temporal_graph/pipeline/build_event_graph.py \
    --input "${EVENT_OUTPUT_DIR}/events_${level}.parquet" \
    --output "${EVENT_OUTPUT_DIR}/temporal_graph_${level}.pt" \
    --static-edges "$STATIC_EDGES_DIR" \
    --raw-edges "$RAW_EDGES_DIR" \
    --advancement-train-csv "${ADV_DIR}/train_dataset.csv" \
    --advancement-test-csv "${ADV_DIR}/test_dataset.csv" \
    --edge-type-mode relation_only
done

echo "[$(date)] ✅ bench_02_graph done"
