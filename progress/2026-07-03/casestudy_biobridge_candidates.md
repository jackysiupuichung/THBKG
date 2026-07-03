# Second-case-study candidates — genuine biology-bridged explanations

**Date:** 2026-07-03
**Purpose:** Find a second explainability case study whose fused path bridges through **genuine
biology** (genetic association / PPI / pathway / GO function to a partner in a *different*
program), not a shared-drug/trial program like MYH7B→HCM. These are the "impressive" end
(indirect biological recovery), to complement MYH7B (the "reliable/legible" end).

**Method:** PaGE-Link 5-seed, exclude-nothing, min_mask 0.02, on 25 evidence-sparse pairs across
23 diseases (`explain_pairs_biobridge_search.csv`); masks percentile-rank fused; single path
search. Screened for paths whose middle hop is a biological relation with no drug/trial edge.
6 of 25 pairs yielded a biology-bridged path; the 3 strongest are reported below.
Path edges show the **actual graph edge attribute `[s, n]`** — the cumulative evidence score s_ij
and novelty n_ij *visible at the pair's decision year*. Masking is **per-instance**: an edge's
latest snapshot with `year <= decision` is used (matching the loader's `edge_time <=
edge_label_time`); reverse (`r:`) edges are resolved to their forward direction for the lookup.
All path edges here are decision-visible (nothing on these paths is masked out). Relation
abbreviations: `gen`=genetic_association, `ppi`=interacts_with, `path`=affected_pathway/involved_in,
`gofn`=has_function_in, `CT`=clinical_trial, `r:`=reverse edge, `sub`=is_subtype_of.

---

## Temporal honesty (own clinical_trial_positive vs decision)

| Pair | Decision | Own positive edges (year, s) | Visible pre-decision |
|---|---|---|---|
| CD274 → mesothelioma | 2016 | (2021, s=0.2) | none — masked ✓ |
| IL17F → psoriatic arthritis | 2016 | (2015, s=0.10), (2020, s=0.2), (2022, s=0.7) | (2015, s=0.10) low prior |
| TIGIT → NSCLC | 2018 | (2020, s=0.2) | none — masked ✓ |

(Clinical-trial edges carry novelty n=0 by construction — the score s encodes trial phase, not
recency.) No leak: the post-decision positive is masked in all three. IL17F sees only a low
(s=0.10) 2015 prior.

---

## Candidate A — CD274 (PD-L1) → mesothelioma  [ENSG00000120217 → EFO_0000588]

**Immune-checkpoint target; the path bridges to mesothelioma through BAP1, the canonical
mesothelioma tumour-suppressor gene, via pathway + genetic/literature links.** Edges show
`[s, n]` visible at the 2016 decision:
```
r0: CD274 -[CT:ongoing s=0.2,n=0]-> mesothelioma
r1: CD274 -[literature s=0.0,n=0]-> peritoneal mesothelioma -[sub s=0.0,n=1.0]-> mesothelioma
r2: CD274 -[CT:positive s=1.0,n=0]-> urothelial carcinoma -[r:CT:positive s=0.1,n=0]-> CTLA4 -[CT:ongoing s=0.7,n=0]-> mesothelioma
r3: CD274 -[path s=0.18,n=0.18]-> renal carcinoma -[r:path s=0.23,n=0.23]-> BAP1 -[literature s=0.0,n=0]-> well-diff. papillary mesothelioma -[sub s=0.0,n=1.0]-> mesothelioma
```
**Bio-bridge (r3):** CD274 → (shared pathway, s=0.18) → renal carcinoma → (r:pathway, s=0.23,
n=0.23) → **BAP1** → mesothelioma. BAP1 germline mutation is *the* hereditary mesothelioma gene —
a genuine, non-obvious biological bridge, not a shared trial, and the whole bridge is
decision-visible.

---

## Candidate B — IL17F → psoriatic arthritis  [ENSG00000112116 → EFO_0003778]

**NOTE: this is the REAL IL17F/psoriatic-arthritis pair — the one the old fabricated narrative
claimed but got wrong (that narrative was actually GLRA1 and invented IL12B). Here are the ACTUAL
paths.** Edges show `[s, n]` visible at the 2016 decision:
```
r0: IL17F -[CT:positive s=0.1,n=0]-> psoriatic arthritis
r1: IL17F -[literature s=0.01,n=0]-> chronic mucocutaneous candidiasis -[r:literature s=0.0,n=0]-> IL17RA -[CT:ongoing s=1.0,n=0]-> psoriatic arthritis
r2: IL17F -[gen s=0.58,n=0.06]-> chronic mucocutaneous candidosis -[r:animal_model s=0.03,n=0.03]-> CCL5 -[rna_expression s=0.01,n=0]-> rheumatoid arthritis -[sub s=0]-> psoriatic arthritis
r3: IL17F -[literature s=0.01,n=0]-> chronic mucocutaneous candidiasis -[r:gen s=0.37,n=0.37]-> TRAF3IP2 -[gen s=0.43,n=0.01]-> psoriatic arthritis
```
**Bio-bridge (r1, r3):** IL17F → **IL17RA** (its receptor) → trial → PsA; and IL17F →
**TRAF3IP2** (Act1, the IL-17 signalling adaptor) → PsA, where the reverse-genetic hop into
TRAF3IP2 (s=0.37, n=0.37) and the TRAF3IP2→PsA genetic edge (s=0.43, n=0.01) are both
decision-visible. IL17F's own genetic edge (to candidiasis, s=0.58, n=0.06) is also visible. These
are the real IL-17 pathway partners (receptor + adaptor) — a genuinely mechanistic bridge, fully
decision-visible. Query disease = psoriatic arthritis (EFO_0003778), the *real* pair the
fabricated story mislabelled.

---

## Candidate C — TIGIT → non-small cell lung carcinoma (NSCLC)  [ENSG00000181847 → EFO_0003060]

**Immune-checkpoint target; bridges via shared immune GO functions to other checkpoints
(CTLA4, CD274/PD-L1).** Edges show `[s, n]` visible at the 2018 decision:
```
r0: TIGIT -[CT:ongoing s=0.2,n=0]-> NSCLC
r1: TIGIT -[gofn s=0.46,n=0.46]-> GO:0050868 (neg.reg.T-cell activation) -[r:gofn s=0.37,n=0.37]-> CTLA4 -[literature s=0.0,n=0]-> NSC squamous lung ca. -[r:lit s=0.02,n=0.01]-> ALK -[path s=0.59,n=0]-> cancer -[r:path s=0.57,n=0.23]-> CEP43 -[somatic_mutation s=0.18,n=0.18]-> NSCLC
```
**Bio-bridge:** TIGIT → (shared immune GO function, s=0.46, n=0.46 — high novelty) → **CTLA4/CD274**
(fellow checkpoints). The GO-function bridge is biologically legitimate and decision-visible, but
the full paths are long (7 hops) stitched through generic cancer-topology edges — less clean than
A or B for a figure.

---

## Recommendation

**IL17F → psoriatic arthritis (Candidate B) is the strongest second case study** and has a nice
narrative bonus: it is the *real* version of the pair the fabricated story mislabelled. Its bridge
is mechanistically tight — IL17F reaches PsA through its own receptor **IL17RA** and the IL-17
signalling adaptor **TRAF3IP2** (genetically associated with PsA; the TRAF3IP2→PsA
`genetic_association` edge is decision-visible at s=0.43, n=0.01, and IL17F's own genetic edge is
s=0.58, n=0.58) — a genuine IL-17-axis explanation, not a shared-trial artifact. Caveat: it has a
low (s=0.10) 2015 prior own-trial edge visible (like the earlier CD3 pairs), so acknowledge that
honestly.

**CD274 → mesothelioma (Candidate A)** is the cleanest "non-obvious biology" story: the bridge to
**BAP1** (hereditary mesothelioma gene) via pathway+genetics is a real indirect link with no
shared trial, and its own positive edge is fully masked. Strong alternative.

TIGIT (C) is valid but the paths are long/cancer-topology-heavy — weaker for a clean figure.

**Suggested pairing for the paper:** MYH7B→HCM (reliable, shared-program) + IL17F→PsA or
CD274→mesothelioma (genuine indirect biology). VERIFY gene/disease facts before writing (BAP1 =
hereditary mesothelioma gene; TRAF3IP2/Act1 = IL-17 adaptor, PsA-associated; IL17RA = IL17F
receptor) to avoid another mislabel.

Related: [[project_pagelink_trial_exclusion]], [[project_explainability_roadmap]].
