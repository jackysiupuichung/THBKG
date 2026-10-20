"""
Stage B: rewrite the OT_VERSION (default 26.09) clinical_precedence parquets,
in place by default or into --output-dir.

Dual-edge mode: every NCT-shaped row becomes 1 OR 2 output rows.
  - All rows emit an `ongoing` evidence at studyStartDate
    (relation = clinical_trial_ongoing, resolvedTrialDate = studyStartDate).
    Its score is the phase weight of the trial's clinicalStage (STAGE_SCORE):
    Open Targets halves the score of trials that stopped for safety or
    efficacy, and that outcome is not known at the start date.
  - Rows whose trial has resolved additionally emit an outcome evidence at
    terminal_date (relation = clinical_trial_<bucket>, resolvedTrialDate =
    terminal_date, clipped to >= studyStartDate)

Outcome bucket rules:
  - stop_reason has 'Safety'/'sideeffect'        -> adverse_effects
  - stop_reason has 'Negative'                   -> unmet_efficacy
  - stop_reason present, no safety/negative      -> Unknown/Operational
  - no stop_reason and overall_status
      COMPLETED / APPROVED_FOR_MARKETING         -> positive
  - no stop_reason and overall_status
      TERMINATED / WITHDRAWN / NO_LONGER_AVAILABLE / SUSPENDED / UNKNOWN
      / other                                    -> Unknown/Operational
  - no stop_reason and overall_status
      RECRUITING / ACTIVE_NOT_RECRUITING /
      NOT_YET_RECRUITING / ENROLLING_BY_INVITATION
                                                 -> still ongoing, no outcome row

Non-NCT rows are dropped (no terminal date available).
Idempotent: tolerates running on already-mutated parquets.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

OT_VERSION = os.environ.get("OT_VERSION", "26.09")
CLIN_PREC_DIR = Path(
    f"/gpfs/scratch/bty414/opentarget_evidences/{OT_VERSION}/evidenceDated/sourceId=clinical_precedence"
)
DEFAULT_CACHE = (
    Path(__file__).resolve().parents[3] / "data" / f"nct_terminal_dates_{OT_VERSION}.parquet"
)

NCT_RE = re.compile(r"^NCT\d{8}$")

ONGOING_STATUSES = {
    "RECRUITING",
    "ACTIVE_NOT_RECRUITING",
    "NOT_YET_RECRUITING",
    "ENROLLING_BY_INVITATION",
}
POSITIVE_STATUSES = {"COMPLETED", "APPROVED_FOR_MARKETING"}

BUCKET_POS = "clinical_trial_positive"
BUCKET_NEG = "clinical_trial_unmet_efficacy"
BUCKET_ADV = "clinical_trial_adverse_effects"
BUCKET_OPS = "clinical_trial_Unknown/Operational"
BUCKET_ONG = "clinical_trial_ongoing"

# Open Targets clinical-precedence score per clinicalStage, before the halving
# it applies to trials stopped for safety or efficacy (checked against the
# 26.09 data: every non-halved row matches this table).
STAGE_SCORE = {
    "UNKNOWN": 0.01,
    "EARLY_PHASE_1": 0.05,
    "PHASE_1": 0.1,
    "PHASE_1_2": 0.15,
    "PHASE_2": 0.2,
    "PHASE_2_3": 0.5,
    "PHASE_3": 0.7,
    "PHASE_4": 1.0,
    "APPROVAL": 1.0,
}


def _has_stop_reason(v) -> bool:
    if v is None or isinstance(v, float):
        return False
    try:
        return len(v) > 0
    except TypeError:
        return bool(v)


def _failure_subbucket(reasons) -> str:
    if reasons is None or isinstance(reasons, float):
        return BUCKET_OPS
    try:
        joined = " ".join(str(r).strip() for r in reasons).lower()
    except TypeError:
        return BUCKET_OPS
    if "safety" in joined or "sideeffect" in joined or "side effect" in joined:
        return BUCKET_ADV
    if "negative" in joined:
        return BUCKET_NEG
    return BUCKET_OPS


def outcome_bucket(row) -> str | None:
    """Return the outcome bucket if the trial has resolved, else None (still ongoing)."""
    if _has_stop_reason(row["trialStopReasonCategories"]):
        return _failure_subbucket(row["trialStopReasonCategories"])
    s = (row.get("overall_status") or "")
    s = s.upper() if isinstance(s, str) else ""
    if s in POSITIVE_STATUSES:
        return BUCKET_POS
    if s in ONGOING_STATUSES:
        return None
    # Anything else (TERMINATED/WITHDRAWN/NO_LONGER_AVAILABLE/SUSPENDED/UNKNOWN/other):
    # treat as a resolved-but-uninformative outcome.
    return BUCKET_OPS


def _clip_to_start(term, start) -> str | None:
    if pd.isna(term) or term is None:
        return str(start) if pd.notna(start) else None
    if pd.isna(start) or start is None:
        return str(term)
    try:
        if pd.to_datetime(term, errors="coerce") < pd.to_datetime(start, errors="coerce"):
            return str(start)
    except Exception:
        pass
    return str(term)


def _rescore_ongoing(ongoing: pd.DataFrame) -> int:
    """Give each start-dated ongoing row its phase weight, in place; return rows changed.

    The only difference allowed is OT's halving: a changed row whose score is
    not half its phase weight means the weight table no longer matches OT.
    """
    full = ongoing["clinicalStage"].map(STAGE_SCORE)
    unknown = sorted(ongoing.loc[full.isna(), "clinicalStage"].dropna().unique())
    if unknown:
        raise ValueError(f"clinicalStage values without a phase weight: {unknown}")
    old = ongoing["score"].astype(float)
    changed = ~np.isclose(old, full)
    halved = np.isclose(2 * old, full)
    if (changed & ~halved).any():
        bad = ongoing.loc[changed & ~halved, ["clinicalStage", "score"]].value_counts().head(10)
        raise ValueError(f"ongoing scores that are neither the phase weight nor half of it:\n{bad}")
    ongoing["score"] = full.astype(float)
    return int(changed.sum())


def process_file(path: Path, cache: pd.DataFrame, dry_run: bool, out_path: Path | None = None) -> dict:
    df = pd.read_parquet(path)
    n_before = len(df)

    # Drop any prior Stage B mutations so we start clean. Safe because the
    # source columns (clinicalReportId, trialStopReasonCategories, studyStartDate)
    # are untouched.
    df = df.drop(columns=[c for c in ("trial_outcome_bucket", "resolvedTrialDate") if c in df.columns])

    df["_nct_id"] = df["clinicalReportId"].astype(str).str.upper()
    is_nct = df["_nct_id"].str.match(NCT_RE).fillna(False)
    n_dropped = int((~is_nct).sum())
    df = df[is_nct].copy()

    df = df.merge(
        cache[["nct_id", "terminal_date", "overall_status"]].rename(
            columns={"nct_id": "_nct_id"}
        ),
        on="_nct_id",
        how="left",
    )

    # 1) ongoing row for every NCT trial, dated at studyStartDate
    ongoing = df.copy()
    ongoing["trial_outcome_bucket"] = BUCKET_ONG
    ongoing["resolvedTrialDate"] = ongoing["studyStartDate"].astype(object)
    ongoing.loc[ongoing["studyStartDate"].isna(), "resolvedTrialDate"] = None
    n_rescored = _rescore_ongoing(ongoing)

    # 2) outcome row only for resolved trials, dated at terminal_date (clipped)
    df["_outcome"] = df.apply(outcome_bucket, axis=1)
    resolved = df[df["_outcome"].notna()].copy()
    resolved["trial_outcome_bucket"] = resolved["_outcome"]
    resolved["resolvedTrialDate"] = resolved.apply(
        lambda r: _clip_to_start(r.get("terminal_date"), r.get("studyStartDate")),
        axis=1,
    )
    resolved = resolved.drop(columns=["_outcome"])

    out = pd.concat([ongoing, resolved], ignore_index=True)
    out = out.drop(columns=["_nct_id", "terminal_date", "overall_status"])

    counts = out["trial_outcome_bucket"].value_counts().to_dict()

    if not dry_run:
        dest = out_path or path
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        out.to_parquet(tmp, index=False)
        shutil.move(str(tmp), str(dest))

    return {
        "file": path.name,
        "rows_before": n_before,
        "rows_dropped_non_nct": n_dropped,
        "rows_after": int(len(out)),
        "ongoing_rescored": n_rescored,
        "buckets": counts,
    }


def main(cache_path: Path, dry_run: bool, input_dir: Path, output_dir: Path | None) -> None:
    cache = pd.read_parquet(cache_path)
    cache["nct_id"] = cache["nct_id"].astype(str).str.upper()

    files = sorted(input_dir.glob("*.parquet"))
    if not files:
        raise SystemExit(f"no parquet files at {input_dir}")
    if output_dir is not None and not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
    print(f"{'DRY-RUN: ' if dry_run else ''}processing {len(files)} files from {input_dir}"
          f" -> {output_dir or 'in place'}")

    totals = {
        "rows_before": 0,
        "rows_dropped_non_nct": 0,
        "rows_after": 0,
        "ongoing_rescored": 0,
        "buckets": {},
    }
    for path in files:
        r = process_file(path, cache, dry_run, out_path=(output_dir / path.name) if output_dir else None)
        print(
            f"  {r['file']}: {r['rows_before']} -> {r['rows_after']} "
            f"(dropped non-NCT {r['rows_dropped_non_nct']}, ongoing rescored {r['ongoing_rescored']})"
        )
        totals["rows_before"] += r["rows_before"]
        totals["rows_dropped_non_nct"] += r["rows_dropped_non_nct"]
        totals["rows_after"] += r["rows_after"]
        totals["ongoing_rescored"] += r["ongoing_rescored"]
        for k, v in r["buckets"].items():
            totals["buckets"][k] = totals["buckets"].get(k, 0) + int(v)

    print("\n=== TOTALS ===")
    print(f"  rows before:           {totals['rows_before']}")
    print(f"  rows dropped (non-NCT):{totals['rows_dropped_non_nct']}")
    print(f"  rows after:            {totals['rows_after']}")
    print(f"  ongoing rows rescored: {totals['ongoing_rescored']}")
    print("\n  bucket distribution:")
    for k in sorted(totals["buckets"], key=lambda x: -totals["buckets"][x]):
        print(f"    {k:42s} {totals['buckets'][k]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--input-dir", type=Path, default=CLIN_PREC_DIR,
                    help="clinical_precedence parquets as downloaded (default: the release's, in place)")
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="write here instead of overwriting --input-dir")
    args = ap.parse_args()
    main(args.cache, args.dry_run, args.input_dir, args.output_dir)
