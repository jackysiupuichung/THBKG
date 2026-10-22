#!/bin/bash
# Submit the benchmark graph build as one dependency chain. Run from the repo root
# on a login node, after the two online prerequisites:
#
#   source scripts/benchmark/bench_env.sh   # (or set the paths by hand)
#   python preprocessing/node_features/fetch_chembl_biotherapeutics.py \
#     --clinical-precedence-dir "$INPUT_EVIDENCE_DIR/sourceId=clinical_precedence" \
#     --output "$CHEMBL_CACHE"
#   python -c "from huggingface_hub import snapshot_download; \
#     snapshot_download('facebook/esm2_t33_650M_UR50D', allow_patterns=['*.json','*.txt','model.safetensors'])"
#
#   bash scripts/benchmark/submit_bench.sh          # all five stages
#   FROM=3 bash scripts/benchmark/submit_bench.sh   # resume at stage 3
set -euo pipefail
cd "$(dirname "$0")/../.."

FROM="${FROM:-1}"
STAGES=(bench_01_edges bench_02_graph bench_03_features bench_04_assemble bench_05_snapshots)
dep="${AFTER:-}"  # AFTER=<jobid>: make the first submitted stage wait for that job
for i in "${!STAGES[@]}"; do
  n=$((i + 1))
  [ "$n" -lt "$FROM" ] && continue
  jid=$(sbatch --parsable ${dep:+--dependency=afterok:$dep} "scripts/benchmark/${STAGES[$i]}.sh")
  echo "stage ${n} ${STAGES[$i]}: ${jid}${dep:+ (after ${dep})}"
  dep="$jid"
done
