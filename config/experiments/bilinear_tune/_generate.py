#!/usr/bin/env python3
"""6-run bilinear-decoder HP sweep on EAHGT s42.

Tests whether the per-TA underperformance of the bilinear decoder is a
tuning artifact (HPs inherited from MLP) or an architectural limit.

Variants (all seed=42, val_year=2013, ES patience=10 on rs_ta_mean@50):
  run1 — baseline:        bilinear, lr=4e-4, dropout=0.2, wd=0.0018, 40 ep
  run2 — higher lr:       bilinear, lr=1e-3, dropout=0.2, wd=0.0018, 60 ep
  run3 — stronger reg:    bilinear, lr=4e-4, dropout=0.4, wd=0.01,   60 ep
  run4 — low-rank 16:     bilinear_lr16, lr=4e-4, dropout=0.2, wd=0.0018, 60 ep
  run5 — low-rank 8:      bilinear_lr8,  lr=4e-4, dropout=0.2, wd=0.0018, 60 ep
  run6 — lr16+aggressive: bilinear_lr16, lr=1e-3, dropout=0.4, wd=0.01,   80 ep
"""
from pathlib import Path

CFG_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = (Path(__file__).resolve().parents[3]
               / "scripts" / "advancement_prediction" / "bilinear_tune")
SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_ROOT = "/gpfs/scratch/bty414/opentarget_evidences/23.06/runs/bilinear_tune"

CONFIG_TMPL = """\
# Bilinear-decoder tuning run {tag}. Inherits EAHGT p3_eahgt_both HPs from
# headline/p3_eahgt_both_s42.yaml; overrides only the variants listed.
experiment:
  name: bilinear_tune_{tag}
data:
  graph_file: /gpfs/scratch/bty414/opentarget_evidences/23.06/graph/hetero_graph_with_features_datatype.pt
  mappings_file: /gpfs/scratch/bty414/opentarget_evidences/23.06/progression/temporal_graph_datatype_mappings.pt
  undirected: true
  train_cutoff_year: 2010
  val_min_year: 2013
  val_max_year: 2013
model:
  name: hgt
  hidden_dim: 128
  num_heads: 2
  num_layers: 2
  dropout: {dropout}
  use_rte: false
  use_edge_features: true
  edge_feat_cols: [0, 1]
  edge_feat_dim: 2
  decoder_kind: {decoder_kind}
  decoder_dropout: {decoder_dropout}
train:
  output_dir: {output_root}/{tag}
  num_epochs: {num_epochs}
  lr: {lr}
  weight_decay: {weight_decay}
  eta_min: 1.0e-06
  cosine_t_max: 10
  batch_size: 256
  num_neighbors: [20, 10]
  lambdarank:
    impl: allrank
    weighing_scheme: lambdaRank_scheme
    sigma: 1.4319388789983414
    ndcg_k: 50
  early_stopping:
    enabled: true
    patience: 10
    metric: rs_ta_mean@50
    mode: max
seed: 42
"""

SBATCH_TMPL = """\
#!/bin/bash
#SBATCH -J bt_{tag}
#SBATCH -o %x.o%j
#SBATCH -p gpushort
#SBATCH -n 8
#SBATCH --cpus-per-gpu=8
#SBATCH -t 1:0:0
#SBATCH --mem-per-cpu=11G
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1

set -euo pipefail

REPO_ROOT="/data/home/bty414/opentarget_temporal_study/src/opentarget_het_graph"
cd "$REPO_ROOT"

source .venv/bin/activate
export WANDB_MODE="disabled"
export SAVE_PER_EPOCH_TOPK=100

python src/train_advancement_lambdarank.py \\
    --config config/experiments/bilinear_tune/{tag}.yaml
"""

SWEEP_TMPL = r"""#!/bin/bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
for f in run_*.sh; do
    sbatch "$f"
done
"""


RUNS = [
    # tag                 decoder_kind   lr      dropout  weight_decay  num_epochs
    ("run1_baseline",     "bilinear",      4e-4,  0.2,    0.0018,       40),
    ("run2_lr1e3",        "bilinear",      1e-3,  0.2,    0.0018,       60),
    ("run3_strongreg",    "bilinear",      4e-4,  0.4,    0.01,         60),
    ("run4_lr16",         "bilinear_lr16", 4e-4,  0.2,    0.0018,       60),
    ("run5_lr8",          "bilinear_lr8",  4e-4,  0.2,    0.0018,       60),
    ("run6_lr16_agg",     "bilinear_lr16", 1e-3,  0.4,    0.01,         80),
]


def main():
    for tag, decoder_kind, lr, dropout, weight_decay, num_epochs in RUNS:
        cfg = CONFIG_TMPL.format(
            tag=tag, decoder_kind=decoder_kind,
            lr=lr, dropout=dropout, decoder_dropout=dropout,
            weight_decay=weight_decay, num_epochs=num_epochs,
            output_root=OUTPUT_ROOT,
        )
        (CFG_DIR / f"{tag}.yaml").write_text(cfg)
        sb = SBATCH_TMPL.format(tag=tag)
        sb_path = SCRIPTS_DIR / f"run_{tag}.sh"
        sb_path.write_text(sb)
        sb_path.chmod(0o755)
    sweep = SCRIPTS_DIR / "submit_all.sh"
    sweep.write_text(SWEEP_TMPL)
    sweep.chmod(0o755)
    print(f"Wrote {len(RUNS)} configs and sbatch scripts.")


if __name__ == "__main__":
    main()
