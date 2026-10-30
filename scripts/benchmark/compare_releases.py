#!/usr/bin/env python
"""Compare two THBKG builds; between builds of one Open Targets version, fail on undeclared drift.

Two builds from the same Open Targets input differ only by our code, so every
difference is ours. This measures them per node type, edge type and label file
and fails on any metric outside its default band (config/release_contract.yaml
drift_defaults) unless a changes file declares it with a reason: the rebuild's
change log, one per rebuild under config/release_changes/.

Across Open Targets versions the input itself changes (25-37% of literature,
IMPC and trial pairs were replaced between 26.03 and 26.09), so there the
comparison only reports what changed, for the release notes, and never fails.
Whether a release holds all of its own input is check_inputs.py's job.

    python scripts/benchmark/compare_releases.py --old OLD --new NEW \\
        [--changes config/release_changes/X.yaml] [--mode auto|same-version|cross-version] \\
        [--diseases-dir NEW_OT/diseases] [--out report.json] [--report-only] [--write-draft draft.yaml]

--mode auto (the default) reads open_targets.release from both releases'
source_versions.json: the same version needs --changes and fails on undeclared
drift; different versions only report. For a rebuild that changes the graph,
run once with --write-draft: it writes every failing metric as a declaration
with reason TODO (which still fails), to fill in or, where a metric reveals a
bug, to fix in the build instead.

Nodes are matched by id. With --diseases-dir (the new release's Open Targets
disease index), old disease ids the new release lacks are first mapped through
obsoleteTerms. Dated rows are compared from the later of the two releases'
fold years (the year older evidence is counted in), so moving the fold changes
no first year. Only the score column of edge_attr is read, so releases exported
with or without novelty compare alike. interacts_with is compared on (min, max)
pairs, so a release storing both directions compares with one storing each
pair once.

Metrics (ratios are new / old; fractions are of the old release's nodes or pairs)
  nodes.<type>.removed, .added       fraction of old nodes
  edges.<src|rel|dst>.rows, .pairs   new / old
  edges.<k>.missing@Y, .lower@Y      of the old pairs with a score before
                                     decision year Y (both endpoints in the new
                                     release): fraction the new release lacks
                                     before Y, or scores lower before Y
  edges.<k>.gone                     fraction of old pairs the new release lacks
                                     at every year (missing@Y minus gone: re-dated)
  edges.<k>.first_year_changed       fraction of pairs in both whose first year
                                     differs (dated relations only)
  edges.<k>.score_p50, .score_p99    quantiles of each pair's last score, new / old
  labels.<file>.rows                 new / old
  labels.<file>.positive_rate        new - old
A node or edge type in only one release fails unless declared (metric
new_type / dropped_type).

Changes file
  changes:
    - key: edges.target|animal_model|disease   # or nodes.<type>, labels.<file>
      metric: rows                             # or missing (every year), missing@2016, ...
      allow: [3.5, 5.0]                        # the declared band
      reason: upstream IMPC rows 1.52M -> 6.91M
A declaration matching no metric fails (a typo would otherwise pass silently);
one whose metric is within the default band anyway is reported as not needed.

Heavy (reads both releases): run via sbatch, as bench_04_assemble.sh does.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
from src.benchmark.release import Release, edge_key  # noqa: E402

ALL_TIME = np.iinfo(np.int64).max
LABEL_FILES = ("train", "eval", "discovery")


def ot_release(release_dir: Path) -> str | None:
    """open_targets.release from a release's source_versions.json, if it has one."""
    p = release_dir / "source_versions.json"
    return (json.loads(p.read_text()).get("open_targets") or {}).get("release") if p.exists() else None


def obsolete_disease_ids(diseases_dir: Path) -> dict[str, str]:
    """Obsolete disease id -> the one current id that lists it in obsoleteTerms."""
    import pyarrow.dataset as ds
    d = ds.dataset(str(diseases_dir), format="parquet").to_table(columns=["id", "obsoleteTerms"]).to_pandas()
    ex = d.explode("obsoleteTerms").dropna(subset=["obsoleteTerms"])
    n_targets = ex.groupby("obsoleteTerms")["id"].nunique()
    ex = ex[ex["obsoleteTerms"].isin(n_targets.index[n_targets == 1])]  # split terms stay unmapped
    return dict(zip(ex["obsoleteTerms"].astype(str), ex["id"].astype(str)))


def match_nodes(old: Release, new: Release, obsolete: dict) -> tuple[dict, dict, dict]:
    """Old node index -> new node index (-1: gone) per node type shared by both,
    with removed / added fractions."""
    to_new, metrics, info = {}, {}, {}
    for nt in sorted(set(old.node_types) & set(new.node_types)):
        o_ids, n_ids = old.node_ids(nt), new.node_ids(nt)
        n_index = {v: i for i, v in enumerate(n_ids)}
        mapped = np.array([n_index.get(v, -1) for v in o_ids], dtype=np.int64)
        remapped = 0
        if nt == "disease" and obsolete:
            for i in np.flatnonzero(mapped < 0):
                j = n_index.get(obsolete.get(o_ids[i], ""), -1)
                if j >= 0:
                    mapped[i], remapped = j, remapped + 1
        to_new[nt] = mapped
        kept = np.unique(mapped[mapped >= 0])
        metrics[f"nodes.{nt}.removed"] = float((mapped < 0).mean())
        metrics[f"nodes.{nt}.added"] = float((len(n_ids) - len(kept)) / len(o_ids))
        info[nt] = {"old": len(o_ids), "new": len(n_ids), "removed": int((mapped < 0).sum()),
                    "added": int(len(n_ids) - len(kept)), "remapped_via_obsolete_terms": remapped}
    return to_new, metrics, info


class Pairs:
    """One edge type's rows as pair keys, sorted by (pair, year, score)."""

    def __init__(self, src: np.ndarray, dst: np.ndarray, t: np.ndarray, s: np.ndarray, n_dst: int, symmetric: bool):
        if symmetric:
            src, dst = np.minimum(src, dst), np.maximum(src, dst)
        key = src * n_dst + dst
        order = np.lexsort((s, t, key))
        self.k, self.t, self.s = key[order], t[order], s[order]

    def before(self, year: int) -> tuple[np.ndarray, np.ndarray]:
        """Unique pair keys with a row dated < year, and the score of the latest (highest on ties)."""
        m = self.t < year
        k, s = self.k[m], self.s[m]
        last = np.r_[k[1:] != k[:-1], True] if len(k) else np.zeros(0, bool)
        return k[last], s[last]

    def first_year(self) -> tuple[np.ndarray, np.ndarray]:
        first = np.r_[True, self.k[1:] != self.k[:-1]] if len(self.k) else np.zeros(0, bool)
        return self.k[first], self.t[first]


def lookup(keys: np.ndarray, values: np.ndarray, query: np.ndarray) -> np.ndarray:
    """values at each query key (keys sorted and unique); NaN where absent."""
    out = np.full(len(query), np.nan)
    if len(keys):
        pos = np.minimum(np.searchsorted(keys, query), len(keys) - 1)
        hit = keys[pos] == query
        out[hit] = values[pos[hit]]
    return out


def fold_year(r: Release) -> int:
    """Earliest year of any dated edge: the year a release folds older evidence into
    (time_range.first_year). Static edges (edge_time < 0) are left out."""
    years = []
    for et in r.edge_types:
        t = r.tensor(r.edge_meta(et)["tensors"]["edge_time"]["key"])
        if (t >= 0).any():
            years.append(int(t[t >= 0].min()))
    return min(years)


def edge_metrics(old: Release, new: Release, et, to_new: dict, kind: str, years: list[int], floor: int) -> dict:
    """``floor``: the later of the two releases' fold years. Dated rows are compared
    on the common time range, so a changed fold year alone changes no first year."""
    src, _, dst = et
    symmetric, static = kind == "symmetric", kind == "static"
    n_dst = new.num_nodes(dst)
    o_ei, o_t, o_s = old.edges(et)
    n_ei, n_t, n_s = new.edges(et)
    if not static:
        o_t, n_t = np.maximum(o_t, floor), np.maximum(n_t, floor)
    s_new, d_new = to_new[src][o_ei[0]], to_new[dst][o_ei[1]]
    keep = (s_new >= 0) & (d_new >= 0)
    old_raw = Pairs(o_ei[0], o_ei[1], o_t, o_s, old.num_nodes(dst), symmetric)
    o = Pairs(s_new[keep], d_new[keep], o_t[keep], o_s[keep], n_dst, symmetric)
    n = Pairs(n_ei[0], n_ei[1], n_t, n_s, n_dst, symmetric)

    k = edge_key(et)
    m = {f"edges.{k}.rows": len(n_t) / max(len(o_t), 1),
         f"edges.{k}.pairs": len(np.unique(n.k)) / max(len(np.unique(old_raw.k)), 1)}
    for y in years:
        ok, os_ = o.before(y)
        ok, os_ = ok[os_ > 0], os_[os_ > 0]
        v = lookup(*n.before(y), ok)
        m[f"edges.{k}.missing@{y}"] = float(np.isnan(v).mean()) if len(ok) else 0.0
        m[f"edges.{k}.lower@{y}"] = float((v < os_ - 1e-6).mean()) if len(ok) else 0.0  # NaN compares False
    ok, ot = o.first_year()
    nk, nt_ = n.first_year()
    # a missing pair is either gone (not in the new release at all) or re-dated later
    m[f"edges.{k}.gone"] = float((~np.isin(ok, nk, assume_unique=True)).mean()) if len(ok) else 0.0
    if not static:  # static edges have no year
        common, oi, ni = np.intersect1d(ok, nk, assume_unique=True, return_indices=True)
        m[f"edges.{k}.first_year_changed"] = float((ot[oi] != nt_[ni]).mean()) if len(common) else 0.0
    _, o_last = old_raw.before(ALL_TIME)
    _, n_last = n.before(ALL_TIME)
    for q, name in ((50, "score_p50"), (99, "score_p99")):
        qo, qn = np.percentile(o_last, q), np.percentile(n_last, q)
        m[f"edges.{k}.{name}"] = float(qn / qo) if qo > 0 else float("nan")
    return m


def label_metrics(old: Release, new: Release) -> tuple[dict, list[str]]:
    m, only = {}, []
    for name in LABEL_FILES:
        po, pn = old.path / "labels" / f"{name}.parquet", new.path / "labels" / f"{name}.parquet"
        if not (po.exists() and pn.exists()):
            if po.exists() != pn.exists():
                only.append(f"labels.{name} only in the {'new' if pn.exists() else 'old'} release")
            continue
        lo, ln = pd.read_parquet(po), pd.read_parquet(pn)
        m[f"labels.{name}.rows"] = len(ln) / max(len(lo), 1)
        if "outcome" in lo.columns and "outcome" in ln.columns:
            m[f"labels.{name}.positive_rate"] = float(ln["outcome"].astype(bool).mean() - lo["outcome"].astype(bool).mean())
    return m, only


def split_metric(name: str) -> tuple[str, str]:
    """'edges.a|b|c.missing@2016' -> ('edges.a|b|c', 'missing@2016')."""
    obj, metric = name.rsplit(".", 1)
    return obj, metric


def judge(metrics: dict, bands: dict, declared: list[dict]) -> tuple[dict, list[str], list[str]]:
    """Status of every metric: ok, declared or fail."""
    used, results, failures = set(), {}, []
    for name, v in metrics.items():
        obj, metric = split_metric(name)
        base = metric.split("@")[0]
        default = bands[f"{obj.split('.')[0]}.{base}"]
        # the most specific declaration wins: metric@year before metric
        decl = next((i for i, d in enumerate(declared) if d["key"] == obj and d["metric"] == metric), None)
        if decl is None:
            decl = next((i for i, d in enumerate(declared) if d["key"] == obj and d["metric"] == base), None)
        within_default = not math.isnan(v) and default[0] - 1e-9 <= v <= default[1] + 1e-9
        res = {"value": None if math.isnan(v) else round(v, 6), "band": default}
        if decl is not None:
            used.add(decl)
            d = declared[decl]
            res.update(band=d["allow"], reason=d["reason"], default_band=default)
            ok = not math.isnan(v) and d["allow"][0] - 1e-9 <= v <= d["allow"][1] + 1e-9
            todo = str(d["reason"]).strip().upper() == "TODO"  # a --write-draft entry nobody explained yet
            res["status"] = "declared" if ok and not todo else "fail"
            if todo:
                res["reason"] = "TODO: reason not filled in"
            if within_default:
                res["declaration_not_needed"] = True
        else:
            res["status"] = "ok" if within_default or math.isnan(v) else "fail"
        if res["status"] == "fail":
            failures.append(name)
        results[name] = res
    unmatched = [f"{d['key']}.{d['metric']}" for i, d in enumerate(declared)
                 if i not in used and d["metric"] not in ("new_type", "dropped_type")]
    return results, failures, unmatched


def fmt(v) -> str:
    return "n/a" if v is None else (f"{v:.4f}" if abs(v) < 10 else f"{v:.1f}")


RATIO_METRICS = {"rows", "pairs", "score_p50", "score_p99"}


def draft_band(metric: str, v: float) -> list[float]:
    """A band just around a measured value, for a draft declaration."""
    if metric in RATIO_METRICS:
        return [math.floor(v * 0.95 * 1000) / 1000, math.ceil(v * 1.05 * 1000) / 1000]
    if metric == "positive_rate":  # a difference, either sign
        return sorted([0.0, math.ceil(v * 1.15 * 1000) / 1000 if v > 0 else math.floor(v * 1.15 * 1000) / 1000])
    return [0.0, math.ceil((v * 1.15 + 0.005) * 1000) / 1000]  # fractions


def write_draft(path: Path, results: dict, presence: list, declared_presence: set, args) -> None:
    """Every failing metric as a declaration with reason TODO, which still fails
    until someone replaces it with the change that explains the metric."""
    entries = []
    for k, m in presence:
        if (k, m) not in declared_presence:
            entries.append({"key": k, "metric": m, "reason": "TODO"})
    for name, res in results.items():
        if res["status"] == "fail" and res["value"] is not None:
            obj, metric = split_metric(name)
            entries.append({"key": obj, "metric": metric, "observed": res["value"],
                            "allow": draft_band(metric.split("@")[0], res["value"]), "reason": "TODO"})
    header = (f"# Draft changes: {args.old} -> {args.new}\n"
              "# Replace each reason TODO with the change that explains the metric (and its commit),\n"
              "# merge entries that share one reason (metric without @year covers every year),\n"
              "# and delete any that reveal a bug instead: fix the build.\n")
    path.write_text(header + yaml.safe_dump({"old": str(args.old), "new": str(args.new), "changes": entries},
                                            sort_keys=False, width=120))
    print(f"draft with {len(entries)} declarations -> {path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old", required=True, type=Path, help="previous release directory")
    ap.add_argument("--new", required=True, type=Path, help="release directory to check")
    ap.add_argument("--changes", type=Path, default=None,
                    help="declared changes (config/release_changes/*.yaml); required between builds of one version")
    ap.add_argument("--mode", choices=("auto", "same-version", "cross-version"), default="auto",
                    help="same-version: fail on undeclared drift; cross-version: report only; "
                         "auto: decide from source_versions.json")
    ap.add_argument("--contract", type=Path, default=_REPO / "config" / "release_contract.yaml")
    ap.add_argument("--diseases-dir", type=Path, default=None,
                    help="new Open Targets disease index, to map retired disease ids through obsoleteTerms")
    ap.add_argument("--out", type=Path, default=None, help="JSON report (default <new>/compare_releases.json)")
    ap.add_argument("--report-only", action="store_true", help="never fail (calibrating bands)")
    ap.add_argument("--write-draft", type=Path, default=None,
                    help="write the failing metrics as a draft changes file (reasons TODO)")
    args = ap.parse_args()

    contract = yaml.safe_load(args.contract.read_text())
    drift = contract["drift_defaults"]
    kinds = {tuple(e[:3]): e[3] for e in contract["edge_types"]}
    mode = args.mode
    if mode == "auto":
        versions = ot_release(args.old), ot_release(args.new)
        if None in versions:
            raise SystemExit(f"--mode auto needs open_targets.release in both source_versions.json "
                             f"(old {versions[0]}, new {versions[1]}): pass --mode")
        mode = "same-version" if versions[0] == versions[1] else "cross-version"
        print(f"Open Targets {versions[0]} -> {versions[1]}: {mode}", flush=True)
    if mode == "same-version" and args.changes is None:
        raise SystemExit("builds of one Open Targets version: --changes is required")
    report_only = args.report_only or mode == "cross-version"
    changes = (yaml.safe_load(args.changes.read_text()) or {}) if args.changes else {}
    declared = changes.get("changes") or []
    for d in declared:
        if d.get("metric") not in ("new_type", "dropped_type") and not (d.get("allow") and d.get("reason")):
            raise SystemExit(f"declaration needs allow and reason: {d}")
    old, new = Release(args.old), Release(args.new)

    obsolete = obsolete_disease_ids(args.diseases_dir) if args.diseases_dir else {}
    to_new, metrics, node_info = match_nodes(old, new, obsolete)
    for nt, i in node_info.items():
        print(f"nodes {nt:9s} old {i['old']:>7,}  new {i['new']:>7,}  removed {i['removed']:>6,}  "
              f"added {i['added']:>6,}" + (f"  ({i['remapped_via_obsolete_terms']:,} via obsoleteTerms)"
                                           if i["remapped_via_obsolete_terms"] else ""), flush=True)

    # node and edge types in only one release
    presence = []
    for kind, o_set, n_set, fmt_key in (("nodes", set(old.node_types), set(new.node_types), str),
                                        ("edges", set(old.edge_types), set(new.edge_types), edge_key)):
        presence += [(f"{kind}.{fmt_key(x)}", "new_type") for x in n_set - o_set]
        presence += [(f"{kind}.{fmt_key(x)}", "dropped_type") for x in o_set - n_set]
    declared_presence = {(d["key"], d["metric"]) for d in declared if d["metric"] in ("new_type", "dropped_type")}
    type_failures = [f"{k} ({m}, not declared)" for k, m in presence if (k, m) not in declared_presence]

    folds = fold_year(old), fold_year(new)
    print(f"fold year old {folds[0]}, new {folds[1]}: dated rows compared from {max(folds)}", flush=True)
    for et in [et for et in new.edge_types if et in set(old.edge_types)]:
        if et[0] not in to_new or et[2] not in to_new:
            continue
        metrics.update(edge_metrics(old, new, et, to_new, kinds.get(et, "evidence"), drift["check_years"],
                                    max(folds)))
        print(f"edges {edge_key(et)} done", flush=True)
    lm, label_only = label_metrics(old, new)
    metrics.update(lm)

    results, failures, unmatched = judge(metrics, drift["bands"], declared)
    failures = type_failures + failures + [f"{u} (declaration matches no metric)" for u in unmatched]

    fail_tag = "CHANGED " if mode == "cross-version" else "FAIL    "  # across versions nothing fails
    print("\nOut-of-band and declared metrics:")
    for name, res in results.items():
        if res["status"] == "ok":
            continue
        tag = fail_tag if res["status"] == "fail" else "DECLARED"
        extra = f"  ({res['reason']})" if "reason" in res else ""
        extra += "  [not needed: within the default band]" if res.get("declaration_not_needed") else ""
        print(f"  {tag} {name} = {fmt(res['value'])}  band {res['band']}{extra}")
    for k, m in presence:
        print(f"  {'DECLARED' if (k, m) in declared_presence else fail_tag} {k} {m}")
    for u in unmatched:
        print(f"  {fail_tag} declaration {u} matches no metric")
    for line in label_only:
        print(f"  INFO     {line}")

    report = {"old": str(args.old), "new": str(args.new), "mode": mode, "changes": str(args.changes),
              "nodes": node_info, "metrics": results, "types_only_in_one_release": presence,
              "labels_only_in_one_release": label_only, "failures": failures}
    out = args.out or args.new / "compare_releases.json"
    out.write_text(json.dumps(report, indent=2))
    if args.write_draft:
        write_draft(args.write_draft, results, presence, declared_presence, args)
    n_ok = sum(r["status"] == "ok" for r in results.values())
    n_decl = sum(r["status"] == "declared" for r in results.values())
    if mode == "cross-version":
        print(f"\n{len(results)} metrics: {n_ok} within the default band, {len(failures)} changed beyond it "
              f"(Open Targets versions differ: reported, not checked) -> {out}")
        return 0
    print(f"\n{len(results)} metrics: {n_ok} within the default band, {n_decl} declared, {len(failures)} failed")
    print(f"{'❌ undeclared drift' if failures else '✅ no undeclared drift'} -> {out}")
    return 1 if failures and not report_only else 0


if __name__ == "__main__":
    sys.exit(main())
