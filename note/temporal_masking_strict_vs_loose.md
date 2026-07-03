# Temporal masking: strict (`<`) vs loose (`<=`)

Advancement is a **future-link-prediction** task: a target–disease pair transitions
at year `Y` (the advancement `edge_time`), and the model must score it using only
evidence available **before** the decision. The masking convention determines what
context edges the neighbor sampler is allowed to see relative to `Y`.

## The two behaviours

PyG's `LinkNeighborLoader` keeps a context edge when
`edge_time <= edge_label_time`. So the label time we pass decides the boundary:

| variant | `edge_label_time` passed | edges kept | convention |
|---|---|---|---|
| **loose** | `edge_time` (`= Y`) | `edge_time <= Y` — **includes year `Y`** | inclusive |
| **strict** | `edge_time - 1` (`= Y-1`) | `edge_time <= Y-1`, i.e. **`< Y`** | temporal-LP standard (`t < target`) |

Loose admits **a full year of same-year context** — every edge stamped `Y`, including
evidence that co-occurs with (or is a consequence of) the transition itself. Because
the graph's time granularity is coarse (year-level), "same year" is a wide window and
a real leakage channel. Strict masking passes `edge_time - 1` at every loader site to
recover the standard `t < target_time` boundary. See memory `masking_strict_before`.

## Strict `<` is the benchmark-conformant convention

The task and its label come from the Related Sciences *clinical_advancement_paper*
(`analysis.py`). Their definitions fix which masking is correct:

- **Evidence cutoff (reference):** `feature_year < transition_year` — **strictly before** `T`.
- **Decision point:** `transition_year` = `T` (the year the phase transition is *reached*,
  from `year_first_advanced` / `clinicalPhase` + `studyStartDate` — **not** trial start).
- **Label horizon:** advanced iff `year_first_advanced <= T + W`, with
  `DEFAULT_CLINICAL_ADVANCEMENT_WINDOW = W = 2`. So the outcome looks ahead over `[T, T+2]`.
- **Split:** `max_training_advancement_year = max_training_transition_year + W`;
  `min_evaluation_transition_year = max_training_transition_year + 1`.

So the task is: **standing at year `T`, using evidence strictly before `T`, will this pair
advance a clinical phase within the next `W=2` years?** The label deliberately encodes a
future outcome (`year_first_advanced ≥ T`) — that is the supervision target, not leakage.
Leakage would be letting *features* see `≥ T`.

**This makes strict (`<`) the canonical variant**, not merely the "honest" one: our strict
loader masks context at `edge_time - 1 = T - 1` (i.e. context `edge_time <= T-1`, so `< T`),
which matches the reference `feature_year < transition_year`. The old loose (`<=`) run
admitted year-`T` evidence that the reference methodology explicitly excludes — its higher
RS was an artifact of that extra year, not a real gain.

**Provenance — VERIFIED (2026-07-03).** The per-edge `edge_time` on context edges is set by the
upstream event-graph builder (produces `hetero_graph_with_features_datatype.pt`), not in this repo.
This loader only *consumes* it: `LinkNeighborLoader(time_attr="edge_time")` filters each context
edge by its own `edge_time <= T-1`. Empirically confirmed that `edge_time` is each evidence's OWN
year (not a constant, not a copy of the advancement `transition_year`) — via
`scripts/inspect_edge_time_provenance.py` on the 26.03 graph:

- Edge types span **different, sensible year ranges** (`clinical_trial_positive` 2005–2025,
  `rna_expression` 2004–2022, `associated_with` 1995–2021), not the advancement shape (1990–2022,
  peak 2015). If dates were copied from the label, all types would share that shape.
- **Millions of context edges are dated OUTSIDE the advancement range** (e.g. 1.15M `genetic_association`
  and 1.77M `literature` edges dated 2023–2025, beyond the max transition year 2022) — impossible
  if they inherited the label year.
- No data-carrying type is a single constant (each has 18–33 distinct years).

So `edge_time <= T-1` genuinely restricts context to evidence dated before decision year `T` — the
correct analogue of the reference's `feature_year < transition_year`. **Strict masking is sound.**

**Ontology-edge nuance.** Three structural types — `disease is_subtype_of disease`,
`go is_subtype_of go`, `reactome is_subpathway_of reactome` — carry a NaT sentinel
(`edge_time = -9223372036854775808`, int64 min). Because it's a large *negative* number,
`edge_time <= T-1` is always true, so these timeless ontology-backbone edges are **always kept**
under BOTH strict and loose masking. Benign (ontology structure isn't time-varying, carries no
temporal signal to leak) and identical across variants, so it doesn't affect the comparison.

## Where it lives in code

`src/train_advancement_lambdarank.py`:

- `seed_time = edge_time - 1` (the strict shift) — single source of truth.
- Applied at **all three** loader/scoring sites so train, val, and test share the boundary:
  - train loader: `edge_label_time=seed_time[train_mask]`
  - val loader:   `edge_label_time=seed_time[val_mask]`
  - test:         `test_edge_times = seed_time[test_mask]` (feeds `test_predictions.parquet`
    and `per_epoch_preds/`, i.e. the scores the ensemble fuses)

**Important:** masking changes which *context* edges are visible, **not** the label set —
test positives stay `n_pos=739` under both variants. A change in label counts would signal
a bug, not a masking change.

The committed `HEAD` (eadd43d) still uses the **loose** form (`edge_label_time=edge_time[...]`,
no `seed_time`). The strict form currently lives as an **uncommitted working-tree change**;
a clean checkout reproduces loose, not strict. Commit the trainer edit to make strict results
reproducible.

## Runs / artifacts

| variant | trained runs | 5-seed ensemble parquet |
|---|---|---|
| loose (`<=`) | `runs/lr_grouped_k100_latest/lrgrpk100lat_s{1,7,42,123,2024}` | `runs/grouped_ensemble_latest_s5` (alias `grouped_ensemble_loose_s5`) |
| strict (`<`) | `runs/lr_grouped_k100_strictmask/strictmask_s{1,7,42,123,2024}` | `runs/grouped_ensemble_strictmask_s5` |

Both ensembles use the **identical** fusion pipeline
(`scripts/advancement_prediction/build_grouped_ensemble_{latest,strictmask}.py` — byte-identical
apart from the run-dir path): per-seed val-select the epoch by `val_rs_ta_median@50`,
percentile-rank each seed's test scores, mean over 5 seeds. They differ **only** in the
masking of the underlying checkpoints. Combined eval:
`scripts/advancement_prediction/run_grouped_ensemble_masking_compare.sh`
→ `headline_results/grouped_ensemble_masking_compare_eval/`.

## Effect on results (26.03, official grouped 5-seed ensemble)

Pooled RS, stratum = all:

| limit | strict (`<`) | loose (`<=`) | RDG | OTS |
|---|---|---|---|---|
| @10 | 3.27 | 4.27 | 4.10 | 2.53 |
| @50 | 2.58 | 4.03 | 1.94 | 1.78 |
| @100 | 2.63 | 3.11 | 1.68 | 1.64 |

- Strict lowers RS **everywhere** (removes the same-year leak). It is **not** a pooled-vs-per-TA
  artifact: strict's real-TA mean/median move together and it loses to loose across the majority
  of TAs — uniformly weaker, not inflated by one TA.
- Strict still beats both baselines at @50/@100, but **loses its @10 lead to RDG** (3.27 vs 4.10).
- **Where the leak lived — the top of the ranking.** Rank agreement strict-vs-loose is moderate
  overall (Spearman 0.72, Kendall 0.53) but collapses at the head: top-10 overlap = 2/10,
  top-50 = 18/50. The bulk/tail ordering is preserved; removing same-year context most disrupts
  the highest-confidence predictions — exactly where RS@10 is decided.

**Takeaway:** strict (`<`) is the **benchmark-conformant** convention — it matches the reference
paper's `feature_year < transition_year` cutoff and its `W=2` look-ahead label. The strict model
is broadly consistent with the loose one (Spearman 0.72) but no longer wins the *head* of the
ranking via year-`T` evidence the reference excludes (top-10 overlap 2/10, @10 lead over RDG
lost). Report **strict as the canonical result**; treat the loose (`<=`) numbers as an upper bound
inflated by one year of otherwise-excluded evidence.
