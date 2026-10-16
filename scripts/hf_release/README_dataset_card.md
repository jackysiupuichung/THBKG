---
license: cc-by-4.0
pretty_name: THBKG — Temporal Heterogeneous Biomedical Knowledge Graph
task_categories:
- graph-ml
- tabular-classification
- other
tags:
- biomedical
- knowledge-graph
- temporal
- drug-discovery
- clinical-trials
- leak-free-evaluation
size_categories:
- 10K<n<100K
configs:
- config_name: advancement
  data_files:
  - split: train
    path: labels/train.parquet
  - split: eval
    path: labels/eval.parquet
---

# THBKG — Temporal Heterogeneous Biomedical Knowledge Graph

A dated biomedical knowledge graph built from
[Open Targets](https://www.opentargets.org/) **26.09** (with Gene Ontology,
Reactome, IntAct, the Human Protein Atlas and ClinicalTrials.gov), plus a
**clinical-advancement benchmark**: rank target–disease pairs that entered
Phase II by how likely they are to advance to Phase III, scored only from
evidence dated **strictly before** each pair's decision year.

Every temporal edge carries the year its evidence first appeared, so the graph
can be queried as of any historical decision point without leakage.

## Files

| Path | What it is |
|---|---|
| `graph.safetensors` | All graph tensors (node features, `edge_index`, `edge_attr`, `edge_time`), pickle-free. |
| `metadata.json` | Schema to reassemble the graph: node/edge types, tensor keys, shapes, dtypes. |
| `mappings/<ntype>.parquet` | `node_id ↔ index` for each node type (`target`, `disease`, `molecule`, `go`, `reactome`). |
| `labels/train.parquet`, `labels/eval.parquet` | Advancement labels: `target_id, disease_id, transition_year, outcome`. |
| `load_thbkg.py` | Standalone loader — rebuilds a PyG `HeteroData`, reads the label splits. |
| `source_versions.json` | Pinned upstream versions (Open Targets release, GO, IntAct, HPA). |
| `croissant.json` | MLCommons Croissant metadata. |

## Graph at a glance

- **Nodes:** 109,087 across five types — `target` (19,620), `disease` (45,896),
  `molecule` (2,596), Gene Ontology term `go` (38,092), Reactome pathway
  `reactome` (2,883). Feature dims: molecule 1024, disease 256, target 56,
  GO 64, Reactome 64.
- **Edges:** ~13.6M over nineteen relations (sixteen temporal, three static);
  each temporal edge carries `edge_time` (year first observed) and
  `edge_attr = [weight, novelty]`.
- **Clinical-trial edges are dated by outcome.** Every registered trial adds an
  `ongoing` edge at its start date; a finished trial adds an outcome edge
  (positive, unmet efficacy, adverse effects, unknown/operational) at its actual
  completion date from ClinicalTrials.gov.
- **Supervision is external.** The graph holds only nodes and *evidence* edges;
  the clinical-advancement outcomes live in the label parquets and are joined to
  the graph by node ID at train/eval time.

## Advancement benchmark

47,802 target–disease pairs, split temporally: **34,740 train** (Phase II entry
1995–2015, 17.7% positive) / **13,062 eval** (2016–2021, 7.9% positive).
`outcome` is `True` iff the pair reached Phase III by a fixed horizon: 2018 for
train, 2024 for eval. Eval entries stop at 2021 so every pair has at least three
years of follow-up (the protocol of Czech et al.). Each pair is scored using only
graph evidence dated strictly before its `transition_year`.

Primary metric: **Relative Success @ K** (an importance-weighted hit rate),
reported per therapeutic area and Wilcoxon-tested against a randomized-decisions
baseline.

## Quick start

```python
from huggingface_hub import snapshot_download
from load_thbkg import load_graph, load_labels, node_index

path  = snapshot_download("jackysiupuichung/THBKG", repo_type="dataset")
data  = load_graph(path)              # PyG HeteroData (reverse edges added, training view)
train = load_labels(path, "train")    # DataFrame: target_id, disease_id, transition_year, outcome
eval_ = load_labels(path, "eval")

# resolve a label endpoint to its graph node index
t2i, d2i = node_index(path, "target"), node_index(path, "disease")
```

The label parquets also load directly as a 🤗 `datasets` config:

```python
from datasets import load_dataset
ds = load_dataset("jackysiupuichung/THBKG", "advancement")   # splits: train, eval
```

## Versions

This is the **26.09** build. The previous **26.03** build stays available at
revision `v26.03`:

```python
snapshot_download("jackysiupuichung/THBKG", repo_type="dataset", revision="v26.03")
```

Results are not directly comparable across builds: Open Targets replaced many
EFO disease IDs with MONDO IDs in 26.06 and merged duplicate terms in 26.09, and
26.09 has about 60% more clinical records. An earlier 23.06 build had a
same-year clinical-trial-edge leak and is **deprecated**.

## License & citation

Dataset artifacts: **CC-BY-4.0**. Source code (construction pipeline, benchmark
harness): MIT, at <https://github.com/jackysiupuichung/THBKG>. Archival DOI on
Zenodo: concept [10.5281/zenodo.20795231](https://doi.org/10.5281/zenodo.20795231).

```
Siu, Cabrera, Mudaliar, and Zubiaga.
THBKG: A Temporal Biomedical Knowledge Graph for Leak-Free Clinical Advancement Prediction.
```
