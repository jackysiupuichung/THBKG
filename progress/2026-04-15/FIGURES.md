# Knowledge Graph Visualisation Figures

This document describes the two main visualisation figures produced for the
temporal heterogeneous knowledge graph derived from Open Targets 23.06.
These notes serve as the basis for figure captions and methods-section writing.

---

## Figure 1 — Sunburst: Static Edge Distribution by Datatype and Datasource

**Script:** `visualisation/sunburst_diagram.py`

**Output:** `sunburst_output/sunburst.html`

**Run command:**
```bash
python visualisation/sunburst_diagram.py \
    --edges-dir /gpfs/scratch/bty414/opentarget_evidences/23.06/evidences/edges \
    --datatype-mapping config/datatype_mapping.yaml \
    --event-config config/event_graph_config.yaml \
    --output-dir /gpfs/scratch/bty414/opentarget_evidences/23.06/progression/sunburst_output
```

### Description

The sunburst diagram provides a static, aggregate view of all edges in the
knowledge graph, organised into two concentric rings:

- **Inner ring (datatype):** The seven high-level evidence categories defined
  by Open Targets — `genetic_association`, `somatic_mutation`, `known_drug`,
  `affected_pathway`, `literature`, `rna_expression`, and `animal_model`. Arc
  size is proportional to the total number of qualifying edges in that category.
- **Outer ring (datasource):** Individual data providers contributing edges
  within each datatype (e.g. `eva`, `chembl`, `europepmc`). Arc size is
  proportional to the number of qualifying edges from that source.

Score cutoffs from `config/event_graph_config.yaml` are applied before counting,
matching the filtering used during graph construction:
- `eva`: score ≥ 0.6
- `europepmc`: score ≥ 0.3

Colours are consistent with the shared palette defined in
`colours_schema.yaml` — each datatype retains the same hue whether it appears
here or in the chord diagrams.

### Suggested Caption

**Figure 1.** Sunburst diagram of edge composition in the Open Targets
heterogeneous knowledge graph (release 23.06). The inner ring shows the seven
evidence datatypes; the outer ring shows individual datasources nested within
each datatype. Arc area is proportional to edge count after applying per-source
score thresholds (EVA ≥ 0.6, EuroPMC ≥ 0.3). Colours are shared with Figure 2.

---

## Figure 2 — Chord: Temporal Evolution of Edge-Type Interactions

**Script:** `visualisation/chord_diagram.py`

**Output:** `chord_output/chord_split_{year}.html` (one per cutoff year),
`chord_output/chord_all_splits.html` (combined), `chord_output/edge_counts_by_split.csv`

**Run command:**
```bash
python visualisation/chord_diagram.py \
    --events /gpfs/scratch/bty414/opentarget_evidences/23.06/progression/events_datatype.parquet \
    --output-dir /gpfs/scratch/bty414/opentarget_evidences/23.06/progression/chord_output \
    --cutoffs 2005 2010 2015 2020 2025
```

### Description

The chord diagrams reveal how the distribution and co-occurrence of edge types
change across five temporal snapshots of the knowledge graph (cutoffs: ≤2005,
≤2010, ≤2015, ≤2020, ≤2025). Each snapshot includes all evidence accumulated
up to and including that cutoff year.

Each chord node represents a `(source_type, relation, target_type)` triple
(e.g. `target → genetic_association → disease`). Node arc size reflects the
total edge count for that triple at the given cutoff. Ribbons connect triples
that share an entity type, weighted by co-occurrence, indicating how different
evidence types cluster around the same biological entities.

Key observations expected across cutoffs:

- **Early snapshots (≤2005, ≤2010):** The graph is sparse and dominated by
  genetic association and literature evidence, reflecting the maturity of
  genetic databases and text mining at that period. Few datasources contribute
  and interaction ribbons are limited to a small number of triple pairs.
- **Mid snapshots (≤2015):** Expansion of somatic mutation and known drug
  evidence becomes visible as ChEMBL and cancer genomics databases grow.
  Pathway and RNA expression datasources begin to appear, broadening the
  relation type distribution.
- **Late snapshots (≤2020, ≤2025):** The graph reaches near-complete coverage.
  Clinical trial edges (`disease → clinical_trial_* → target`) emerge
  prominently, reflecting the systematic curation of clinical advancement.
  Ribbon density increases substantially, indicating that entities now
  accumulate evidence from multiple independent datatypes simultaneously.

This temporal progression motivates the use of a temporal graph model that
can learn from the evolving structure of evidence rather than treating the
knowledge graph as static.

Colours for each relation type are fixed across all cutoff panels using the
shared palette in `colours_schema.yaml`, enabling direct visual comparison
between snapshots.

### Suggested Caption

**Figure 2.** Chord diagrams of the temporal knowledge graph at five cumulative
cutoff years (≤2005, ≤2010, ≤2015, ≤2020, ≤2025). Each node represents a
directed edge type `(source → relation → target)`; arc size is proportional to
edge count at that cutoff. Ribbons indicate co-occurrence of edge types sharing
an entity, weighted by the minimum count of the two connected triples. Colour
encoding is consistent across panels and with Figure 1. The progression from
sparse, genetically dominated evidence (early cutoffs) to a densely connected,
multi-evidence graph (late cutoffs) illustrates the temporal dynamics that
motivate the temporal graph learning approach.

---

## Figure 3 — Cumulative Advancement Labels by Therapeutic Area

**Script:** `visualisation/advancement_cumulative_by_ta.py`

**Output:** `advancement_cumulative_output/cumulative_advancement_positive.png`,
`advancement_cumulative_output/cumulative_advancement_negative.png`,
`advancement_cumulative_output/cumulative_advancement_by_ta.csv`

**Run command:**
```bash
python visualisation/advancement_cumulative_by_ta.py \
    --train-csv data/clinical_trial_advancement/23.06/train_dataset.csv \
    --test-csv  data/clinical_trial_advancement/23.06/test_dataset.csv \
    --ta-parquet advancement_data/features/therapeutic_areas.parquet \
    --primary-tas-json advancement_data/results/primary_therapeutic_areas.json \
    --output-dir /gpfs/scratch/bty414/opentarget_evidences/23.06/progression/advancement_cumulative_output
```

### Description

Two stacked-area plots (one for positive labels, one for negative labels) show
how advancement evidence accumulates over time. Each coloured band is a
therapeutic area; stack height at year Y is the cumulative number of labels of
that sign with `transition_year <= Y`. Counts come from the union of the train
and test advancement datasets (`outcome = True` → positive; `False` → negative).

Therapeutic areas are resolved by joining each label's `disease_id` against
`advancement_data/features/therapeutic_areas.parquet`. A disease that maps to
multiple therapeutic areas contributes to each, matching the per-TA
stratification used in `evaluate_advancement.py`. The synthetic `all` TA is
excluded from the stack (it would double-count) but is retained in the CSV so
its total matches the raw positive/negative counts.

By default only the primary therapeutic areas listed in
`advancement_data/results/primary_therapeutic_areas.json` are plotted; pass
`--all-tas` to include every therapeutic area present in the data.

Styling follows `evaluate_advancement.py`: plotnine with `theme_minimal`, 150
dpi PNG, matplotlib `tab20` palette for TA colours.

### Suggested Caption

**Figure 3.** Cumulative counts of (a) positive and (b) negative clinical
advancement labels as a function of `transition_year`, stacked by therapeutic
area. Each coloured band represents one therapeutic area; full stack height
equals the cumulative label count. Diseases mapping to multiple therapeutic
areas are counted in each band they belong to. The plot exposes both the
overall growth of clinical trial evidence and the relative contribution of
each therapeutic area to the signal — with oncology dominating positive
advancements and a broader spread across TAs for negative outcomes.

---

## Shared Colour Scheme

All figures use a common colour palette defined in `colours_schema.yaml` and
loaded via `colours.py`. The four colour namespaces are:

| Namespace | Used in |
|-----------|---------|
| `datatype_colours` | Sunburst inner ring; chord relation colours where relation maps to a datatype |
| `datasource_colours` | Sunburst outer ring |
| `relation_colours` | Chord node and ribbon colours |
| `node_colours` | Reserved for future node-level visualisations |

This ensures that, for example, `genetic_association` always appears in the
same blue (#4C72B0) whether it is a sunburst segment or a chord arc.
