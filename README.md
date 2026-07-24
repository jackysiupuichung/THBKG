[![Dataset: Zenodo](https://img.shields.io/badge/dataset-Zenodo%2010.5281%2Fzenodo.20795231-1682D4)](https://doi.org/10.5281/zenodo.20795231)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![PyTorch Geometric](https://img.shields.io/badge/PyG-CUDA%2011.8-EE4C2C)](https://pyg.org/)

# THBKG — Temporal Heterogeneous Biomedical Knowledge Graph

A dated biomedical knowledge graph built from
[Open Targets](https://www.opentargets.org/) 26.03 (with Reactome, ChEMBL, and
ClinicalTrials.gov), plus the code for its **clinical-advancement benchmark**:
ranking target–disease pairs by their likelihood of advancing to Phase II,
scored only from evidence datable *strictly before* each pair's decision year.

Every temporal edge carries the year its evidence first appeared, so the graph
can be queried as of any historical decision point without leakage. Full dataset
description and statistics are in [croissant.json](croissant.json).

## Load the dataset

Packaged 26.03 graph + advancement benchmark are archived on Zenodo
(CC-BY-4.0). Cite the concept DOI
[10.5281/zenodo.20795231](https://doi.org/10.5281/zenodo.20795231) (all
versions); the built graph tensors — `hetero_graph_with_features.pt` and
`temporal_graph_mappings.pt` — are attached to version
[10.5281/zenodo.21529524](https://doi.org/10.5281/zenodo.21529524).

```python
import torch
from src.data.temporal_loader import load_event_graph

graph    = load_event_graph("hetero_graph_with_features.pt")   # PyG HeteroData
mappings = torch.load("temporal_graph_mappings.pt", weights_only=False)
```

To run the CLI against an unpacked copy, point it at the data root:

```bash
export THBKG_DATA_ROOT=/path/to/opentarget_evidences
```

Paths resolve under `$THBKG_DATA_ROOT/26.03/...`; individual files can be
overridden per call (`--graph_file`, `--mappings_file`).

## Setup

Dependencies are managed with [uv](https://github.com/astral-sh/uv); Python ≥
3.11, PyTorch + PyG on CUDA 11.8. Training and evaluation are heavy GPU/CPU jobs,
run via SLURM (`sbatch`), not in the foreground.

```bash
uv sync
```

## Evaluate

```bash
python evaluate_advancement.py                              # all registered runs
python evaluate_advancement.py --only p3_eahgt_both,b1_hgt  # a subset
```

The primary metric is **Relative Success @ K** (an importance-weighted hit rate),
reported per therapeutic area (TA-mean over 13 areas) and Wilcoxon-tested against
a randomized-decisions baseline.

| Reference | What it is | Source |
|---|---|---|
| **EA-HGT** | Grouped five-seed, validation-selected, percentile-rank-fused ensemble (official result) | this repo |
| **RDG** | Ridge regression on decision-time features (Czech et al.) | `evaluation_dataset.zarr` |
| **OTS** | Open Targets global association score | `evaluation_dataset.zarr` |

## Explainability

Post-hoc explanation code lives under `explain/`: the library in `src/explain/`
(integrated gradients, attention, PaGE-Link), command-line entrypoints in
`explain/cli/`, evidence-assembly helpers in `explain/evidence/`, case-study
tooling in `explain/casestudy/`, and SLURM drivers in `explain/drivers/`.

## Rebuild from source

Four ordered SLURM stages under `scripts/` (require an Open Targets 26.03
evidence dump and the IntAct / GO / Reactome / ChEMBL / EFO sources):

```bash
sbatch scripts/collecting_edges_01.sh          # dated event lists per datasource
sbatch scripts/building_event_graph_02.sh      # HeteroData temporal + advancement edges
sbatch scripts/collecting_node_features_03.sh  # node features
sbatch scripts/assembling_graph_04.sh          # attach features → final graph
```

Then train the ensemble (per-seed configs in `config/experiments/headline/`) and
fuse:

```bash
sbatch scripts/advancement_prediction/run_grouped_ensemble_strictmask.sh
```

## License

Source code: [MIT](LICENSE). Dataset artifacts: CC-BY-4.0 (see Zenodo above).

> The canonical build is Open Targets 26.03; the earlier 23.06 build had a
> same-year clinical-trial-edge leak and is **deprecated**.
