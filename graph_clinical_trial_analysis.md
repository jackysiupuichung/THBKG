# Clinical Trial Data Analysis Report
**Date:** 2026-02-03  
**Graph:** `/Users/pui.chungsiu/Documents/opentarget_het_graph/output/graph/hetero_graph_with_features.pt`

---

## Executive Summary

✅ **The graph DOES contain clinical trial edges** (`disease --[clinical_trial::chembl]--> target`)  
❌ **The graph does NOT contain max phase by outcome type** as required by the research plan  
⚠️ **Critical data is available but not being extracted**

---

## Current State

### What's in the Graph (86,971 edges)
The current graph edge attributes:
- `edge_attr`: Single float32 score (shape: [86971, 1])
- `edge_time`: Temporal information (int64, shape: [86971])

### What's in the Processed Edges (2,494 edges after filtering)
File: `output/evidences/edges/disease_clinical_trial_target_chembl.parquet`

**Columns captured:**
1. ✅ `sourceId` (disease)
2. ✅ `targetId` (target/gene)
3. ✅ `clinicalStatus` - Trial status (Completed, Recruiting, Terminated, etc.)
4. ✅ `studyStopReasonCategories` - Why trials stopped (list of reasons)
5. ✅ `year` - Temporal information
6. ✅ `score` - Evidence score

**Distribution of Clinical Status:**
```
Completed                  1,388  (55.7%)
Recruiting                   281  (11.3%)
Terminated                   277  (11.1%)
Active, not recruiting       193  (7.7%)
Unknown status               164  (6.6%)
Withdrawn                     86  (3.4%)
Not yet recruiting            84  (3.4%)
Suspended                     13  (0.5%)
Enrolling by invitation        8  (0.3%)
```

**Distribution of Stop Reason Categories:**
```
Insufficient enrollment                105
Business or administrative              69
Negative (efficacy failure)             30
Safety or side effects                  15
Logistics or resources                  12
Study design                            11
Another study                           10
[+ 20+ other combinations]
```

---

## What's MISSING (Critical Gap)

### ❌ Clinical Phase NOT Extracted

The source ChEMBL data contains `clinicalPhase` with **3,069 trials** distributed as:

| Phase | Count | Percentage |
|-------|-------|------------|
| 0.5 (Phase I/II) | 33 | 1.1% |
| 1.0 (Phase I) | 614 | 20.0% |
| 2.0 (Phase II) | 979 | 31.9% |
| 3.0 (Phase III) | 571 | 18.6% |
| 4.0 (Phase IV) | 872 | 28.4% |

**This data exists in the source but is NOT being extracted!**

### ❌ Outcome Type Classification NOT Computed

The research plan requires 4 outcome categories:
1. **`trial_with_positive_outcome`** - Successful trials
2. **`trial_with_unmet_efficacy`** - Failed due to lack of efficacy
3. **`trial_with_adverse_effects`** - Failed due to safety issues
4. **`trial_with_operation_or_unknown`** - Administrative/other reasons

**Current issue:** Raw `clinicalStatus` and `studyStopReasonCategories` are stored but not classified into these categories.

### ❌ Max Phase by Outcome NOT Aggregated

Research plan requires (per disease-target pair, per time T):
- `y_pos(T)`: max phase for positive outcomes
- `y_unmet(T)`: max phase for unmet efficacy
- `y_adv(T)`: max phase for adverse effects  
- `y_op(T)`: max phase for operational/unknown

**Current issue:** No aggregation logic exists to compute these values.

---

## Why `extract_edge_props` Isn't Fully Activated

### The Function IS Being Called
Line 295 in `parser.py`:
```python
extracted = extract_edge_props(row, props, datasource)
```

### The Issue: Missing from Schema
File: `config/edge_schema.yaml`, line 30:
```yaml
props: [source_type=constant:disease, target_type=constant:target, 
        clinicalStatus, studyStopReasonCategories, 
        datasourceId, score, year, id]
```

**`clinicalPhase` is NOT in the props list!**

This means:
- ✅ `clinicalStatus` is extracted
- ✅ `studyStopReasonCategories` is extracted  
- ❌ **`clinicalPhase` is IGNORED** (not in props)
- ❌ **`studyStopReason` is IGNORED** (not in props, though text version exists)

---

## Cross-Tabulation: Phase vs Status vs Outcome

Based on source data analysis:

### Completed Trials by Phase
- Phase 1: 355 completed
- Phase 2: 511 completed
- Phase 3: 331 completed
- Phase 4: 187 completed

### Terminated Trials by Phase (and reasons)
Examples from sample data:
- **Phase 1 + Terminated + Safety**: 15 trials
- **Phase 3 + Terminated + Negative**: 30 trials
- **Phase 2 + Terminated + Insufficient enrollment**: 105 trials

---

## Required Actions to Implement Research Plan

### 1. **Extract `clinicalPhase`** ✏️
**File:** `config/edge_schema.yaml` (line 30)

**Change:**
```yaml
props: [source_type=constant:disease, target_type=constant:target, 
        clinicalStatus, studyStopReasonCategories, clinicalPhase,
        datasourceId, score, year, id]
```

### 2. **Update `edge_extractor.py`** ✏️
Add handling for `clinicalPhase` in the ChEMBL section:
```python
if p == "clinicalPhase":
    if pd.notnull(val):
        extracted[p] = float(val)  # Ensure it's a float
    continue
```

### 3. **Create Outcome Classification Module** 🆕
**New file:** `src/parsers/outcome_classifier.py`

Function to map (`clinicalStatus`, `studyStopReasonCategories`) → outcome type:
```python
def classify_outcome(status, stop_reasons):
    """
    Returns: 'positive', 'unmet_efficacy', 'adverse_effects', 'operational'
    """
    # Logic:
    # - Completed → positive
    # - Suspended/Terminated + "Negative" → unmet_efficacy
    # - Suspended/Terminated + "Safety" → adverse_effects
    # - Withdrawn/Terminated + other reasons → operational
    ...
```

### 4. **Create Label Generation Script** 🆕
**New file:** `src/pipeline/generate_clinical_labels.py`

Purpose: Generate `labels_{T}.parquet` files with columns:
- `diseaseId`
- `targetId`
- `y_pos_T` (max phase for positive outcomes up to year T)
- `y_unmet_T` (max phase for unmet efficacy up to year T)
- `y_adv_T` (max phase for adverse effects up to year T)
- `y_op_T` (max phase for operational/unknown up to year T)

For T ∈ {2015, 2017, 2024}

### 5. **Create Progression Labels** 🆕
**New file:** `src/pipeline/generate_progression_labels.py`

Purpose: Generate `eval_anchor2017.parquet` with:
- Candidate pairs (with evidence by 2017)
- `y_prog = 1` if `y_pos(2024) > y_pos(2017)` else 0
- `y_risk = 1` if adverse/unmet increased between 2017-2024

---

## Proposed Classification Logic

### Outcome Type Mapping

**`positive_outcome`:**
- `clinicalStatus == "Completed"` AND no negative stop reasons
- Phase progression observed

**`unmet_efficacy`:**
- `clinicalStatus in ["Terminated", "Suspended"]` 
- `"Negative" in studyStopReasonCategories`

**`adverse_effects`:**
- `clinicalStatus in ["Terminated", "Suspended", "Withdrawn"]`
- `"Safety or side effects" in studyStopReasonCategories`

**`operational`:**
- `clinicalStatus in ["Withdrawn", "Terminated"]`
- Stop reasons: "Business or administrative", "Insufficient enrollment", "Logistics", etc.

---

## Data Pipeline Update Required

### Current Pipeline
```
Source ChEMBL parquet 
  → EdgeParser.apply_spec (extracts only what's in schema)
  → saves to edges/disease_clinical_trial_target_chembl.parquet
  → loads into graph (aggregates to single score)
```

### Updated Pipeline Needed
```
Source ChEMBL parquet
  → Extract: clinicalPhase, clinicalStatus, studyStopReasonCategories ✏️
  → Classify outcome type (new module) 🆕
  → Generate temporal labels by year (new script) 🆕
  → Create labels_{2015,2017,2024}.parquet files 🆕
  → Create eval_anchor2017.parquet 🆕
```

---

## Example Output Format

### labels_2017.parquet
```
diseaseId       | targetId        | y_pos | y_unmet | y_adv | y_op
----------------|-----------------|-------|---------|-------|------
MONDO_0007254   | ENSG00000137267 |   3   |    0    |   0   |   0
EFO_0000181     | ENSG00000146648 |   0   |    0    |   1   |   0
EFO_0003843     | ENSG00000105464 |   0   |    3    |   0   |   0
```

Where values 0-4 represent:
- 0: No trial observed
- 1: Phase I max
- 2: Phase II max
- 3: Phase III max
- 4: Phase IV max

---

## Verification Examples

### Example 1: Positive Outcome
```
diseaseId: MONDO_0007254
targetId: ENSG00000137267  
clinicalPhase: 3.0
clinicalStatus: Recruiting
studyStopReasonCategories: None
→ Classification: positive_outcome (if completed in future data)
```

### Example 2: Safety Issue
```
diseaseId: EFO_0000181
targetId: ENSG00000146648
clinicalPhase: 1.0
clinicalStatus: Terminated
studyStopReasonCategories: ['Safety or side effects']
→ Classification: adverse_effects
→ y_adv = 1 (Phase I)
```

### Example 3: Lack of Efficacy
```
diseaseId: EFO_0003843
targetId: ENSG00000105464
clinicalPhase: 3.0
clinicalStatus: Terminated
studyStopReasonCategories: ['Negative']
→ Classification: unmet_efficacy
→ y_unmet = 3 (Phase III)
```

---

## Summary

### ✅ Good News
1. Clinical trial data exists in the graph (86,971 edges)
2. Source data contains all required fields (`clinicalPhase`, `clinicalStatus`, `studyStopReasonCategories`)
3. The extraction infrastructure is in place (just needs schema update)

### ⚠️ Critical Gaps
1. **`clinicalPhase` not extracted** - Simple schema fix needed
2. **Outcome classification missing** - Requires new logic  
3. **Max phase aggregation missing** - Requires new pipeline scripts
4. **Temporal labels not generated** - Requires new data artifacts

### 📋 Next Steps
1. Update `config/edge_schema.yaml` to include `clinicalPhase`
2. Re-run edge extraction pipeline
3. Implement outcome classifier
4. Generate label files for 2015, 2017, 2024
5. Generate progression/evaluation dataset

---

**Conclusion:** The graph contains clinical trial data but NOT in the format required by the research plan. The missing piece is primarily the `clinicalPhase` field and downstream processing to classify outcomes and aggregate max phases per outcome type.
