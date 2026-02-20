import pandas as pd
import json
from collections import Counter
from typing import List, Union


def _parse_reason_field(val: Union[str, list, float]) -> List[str]:
    """Normalise studyStopReasonCategories into a list of clean strings."""
    if val is None:
        return []
    if isinstance(val, list):
        return [str(v).strip() for v in val]
    if isinstance(val, float):
        return []  # NaN
    if isinstance(val, str):
        # Try literal list: "['Lack of efficacy', 'Recruitment']"
        try:
            parsed = json.loads(val.replace("'", "\""))
            if isinstance(parsed, list):
                return [str(v).strip() for v in parsed]
        except Exception:
            pass
        # Otherwise treat as a single string reason
        return [val.strip()]
    return [str(val).strip()]


def inspect_chembl_stop_reasons(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract all stop reasons from a ChEMBL clinical-trial dataframe.
    
    Returns a dataframe showing:
        reason, count
    """
    all_reasons = []

    for val in df.get("studyStopReasonCategories", []):
        reasons = _parse_reason_field(val)
        all_reasons.extend(reasons)

    counter = Counter(all_reasons)

    result = pd.DataFrame(
        [{"reason": r, "count": c} for r, c in counter.items()]
    ).sort_values("count", ascending=False)

    return result


def inspect_clinical_status(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count occurrences of each clinicalStatus string.
    """
    status_counter = Counter(str(v).strip().lower() for v in df["clinicalStatus"].fillna("unknown"))
    return (
        pd.DataFrame([{"clinicalStatus": s, "count": c} for s, c in status_counter.items()])
        .sort_values("count", ascending=False)
    )


def inspect_reason_by_status(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a grouped table:
        clinicalStatus   reason    count
    """
    rows = []

    for _, row in df.iterrows():
        status = str(row.get("clinicalStatus", "unknown")).lower()
        reasons = _parse_reason_field(row.get("studyStopReasonCategories"))
        if not reasons:
            rows.append({"clinicalStatus": status, "reason": "(none)", "count": 1})
        else:
            for r in reasons:
                rows.append({"clinicalStatus": status, "reason": r, "count": 1})

    result = (
        pd.DataFrame(rows)
        .groupby(["clinicalStatus", "reason"])
        .sum()
        .reset_index()
        .sort_values(["clinicalStatus", "count"], ascending=[True, False])
    )

    return result


def _norm_reasons(val):
    if val is None:
        return []
    if isinstance(val, float):
        return []
    if isinstance(val, list):
        return [str(r).lower().strip() for r in val]
    # Try JSON
    try:
        parsed = json.loads(val.replace("'", "\""))
        if isinstance(parsed, list):
            return [str(r).lower().strip() for r in parsed]
    except:
        pass
    return [str(val).lower().strip()]


# -----------------------------------------
# 1. High-level "bucket" mapping (GATher style)
# -----------------------------------------
def map_clinical_outcome(status: str, reason: str) -> str:
    """
    Collapse the clinical trial into 4 major outcome buckets:
      - positive        -> success
      - unmet_efficacy  -> failed efficacy
      - adverse_effects -> failed safety
      - unknown         -> operational/ongoing/unclear
    """

    s = (status or "").lower()
    r = (reason or "").lower()

    # ----- Completed → positive efficacy -----
    if "completed" in s:
        return "clinical_trial_positive"

    # ----- Safety / side effects → adverse effect failures -----
    if "safety" in r or "side effect" in r:
        return "clinical_trial_adverse_effects"

    # ----- Negative OR insufficient efficacy -----
    if "negative" in r:
        return "clinical_trial_unmet_efficacy"

    # ----- Enrollment problems (recruitment fail) -----
    if "insufficient enrollment" in r or "enrollment" in r:
        return "clinical_trial_Unknown/Operational"

    # ----- Business, logistics, administrative → unknown/operational -----
    if any(k in r for k in [
        "business", "administrative", "logistics",
        "resources", "moved", "interim analysis",
        "invalid reason", "regulatory", "no context"
    ]):
        return "clinical_trial_Unknown/Operational"

    # ----- Suspended / withdrawn but no clear reason → unknown -----
    if any(k in s for k in ["withdrawn", "suspended", "unknown"]):
        return "clinical_trial_Unknown/Operational"

    # Catch-all
    return "clinical_trial_Unknown/Operational"



# -----------------------------------------
# 3. Main expander
# -----------------------------------------
def expand_chembl_clinical_trials(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes the raw ChEMBL clinical-trial edges and expands:
      1) into 4 major buckets (GATher-compatible)
      2) optionally adds extra edges for specific fail categories

    Required columns:
        - relation == "clinical_trial"
        - clinicalStatus
        - studyStopReasonCategories (single or list)
    """

    df = df.copy()
    out_rows = []

    for _, row in df.iterrows():
        # only include those target disease relationships are clinical trials
        if row.get("relation") != "clinical_trial":
            out_rows.append(row.to_dict())
            continue
        
        status = str(row.get("clinicalStatus"))
        reasons = row.get("studyStopReasonCategories")
        if isinstance(reasons, list):
            reasons_list = reasons
        elif pd.isna(reasons) or reasons in (None, "", "(none)"):
            reasons_list = []
        else:
            reasons_list = [reasons]

        # ---------------------------------------
        # 1) MAIN BUCKET
        # ---------------------------------------
        bucket_relation = map_clinical_outcome(status, reasons_list[0] if reasons_list else "")
        base = row.to_dict()
        base["relation"] = bucket_relation
        out_rows.append(base)

    return pd.DataFrame(out_rows)
