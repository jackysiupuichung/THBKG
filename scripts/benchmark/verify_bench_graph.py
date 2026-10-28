#!/usr/bin/env python
"""Check the benchmark graph against the released v26.09 graph.

Relative to the release, the benchmark build: makes all drug types molecule
nodes with per-modality features; rescores the start-dated clinical_trial_ongoing
edge with the trial's phase weight; gives static ontology edges (score 1,
novelty 0) and edge_time -1; computes each year's top-50 evidence from that
year's rows only; recovers PubMed-dated GO / IntAct rows lost in 26.09; stores
IntAct pairs once in (min, max) order without self-loops; counts each
(drug, target, trial) once in modulated_by; drops rows scoring 0; dates
literature by publicationDate; emits no outcome edge for trials of UNKNOWN
status; leaves out trials with no start date; and re-encodes disease, GO and
Reactome features. The checks below
pin down what must NOT change. Fails (exit 1) on any hard check.

Nodes are matched by id: edits to the evidence add or remove a few targets
(dropped score-0 rows, recovered PubMed-dated rows, evidence dated 2026), which
shifts node indices.

Hard checks
  1. disease/go/reactome: same node ids in the same order; re-encoded (disease
     768-d text, GO / Reactome 64-d autoencoder with a learnable node input),
     so their shape, finiteness and distinct rows are checked. target: every
     release target missing from the bench graph had only score-0 evidence;
     the targets both graphs share keep identical features.
  2. edge types: exactly the release's types.
  3. every dated edge type without a molecule endpoint: EVIDENCE ONLY GROWS.
     At each check year Y, every pair the release has before Y is present
     before Y with a score >= the release's. The per-year top-50 fix and the
     recovered PubMed-dated GO / IntAct rows only add evidence; anything else
     is a bug. (Up to 0.1% violations allowed for the two PubMed-dated
     relations, where PubMed itself may have re-dated a record.)
     Release pairs whose score before Y is 0 are skipped (score-0 rows are
     dropped). literature is compared one year later in the bench graph
     (bench before Y+1 vs release before Y), with the same 0.1% allowance:
     its publicationDate falls a year after publicationYear in 179k rows.
     The trial outcome relations (positive, unmet_efficacy, adverse_effects,
     Unknown/Operational) are reported, not checked: trials of UNKNOWN status
     emit no outcome edge, and trials with no start date are left out.
     interacts_with is compared on (min, max) pairs without self-loops, the
     release's two directions taking their higher score.
  3b. static ontology types: identical edge_index/edge_time; edge_attr is
     exactly (1, 0).
  4. clinical_trial_ongoing, on the release's time range (folded into 1995,
     to 2025): the same (target, disease) pairs, except that a pair whose
     trials ALL lack a start date may be gone; for pairs whose trials all have
     one, the same first edge_time (so first-trial labels are unchanged) and a
     final score that never drops; novelty unchanged (always 0). In check 3, a
     missing or lower ongoing pair-year is allowed only for a pair with at
     least one trial lacking a start date (the release dated those trials by
     publication date). Trials are read from --clinical-evidence.
  5. molecules: a release molecule may be gone only if all its trials lack a
     start date. modulated_by on the release's molecules: at most 1% of
     release pair-years go missing (drug-target pairs whose only trials have
     no start date; 34 of 10,253 pairs in 26.09); scores may fall (the release
     counted each trial up to twice).
  6. molecules both graphs have keep their exact Morgan fingerprint (block 0)
     unless the release had mean-imputed them.
Reported
  - molecule coverage by input modality; how many ongoing pairs changed score.

Heavy (loads two full graphs): run via scripts/benchmark/bench_04_assemble.sh.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd
import torch

MOD_EDGE = ("molecule", "modulated_by", "target")
ONGOING_EDGE = ("target", "clinical_trial_ongoing", "disease")


CHECK_YEARS = (1996, 2000, 2005, 2010, 2016, 2020, 2025, 2026)
# The release's time range: evidence before 1995 was folded into 1995 (the
# bench graph folds at 1985) and nothing after 2025 (the bench graph runs to
# September 2026)
RELEASE_YEARS = (1995, 2025)
PUBMED_DATED = {"has_function_in", "interacts_with"}
# Re-dated relations: years the bench graph may lag the release by
REDATED_LAG = {"literature": 1}
# Relations that lose release edges by design (no outcome for UNKNOWN-status
# trials; trials with no start date left out): reported only
REPORT_ONLY = {"clinical_trial_Unknown/Operational", "clinical_trial_positive",
               "clinical_trial_unmet_efficacy", "clinical_trial_adverse_effects"}


def _scores_before(src, dst, t, w, year, symmetric=False) -> pd.Series:
    """Score of each pair's latest row dated < year, indexed by (src, dst).

    symmetric: drop self-loops and index by (min, max), taking the higher score
    when both directions are present.
    """
    m = t < year
    df = pd.DataFrame({"s": src[m], "d": dst[m], "t": t[m], "w": w[m]}).sort_values(["s", "d", "t"])
    last = df.groupby(["s", "d"], sort=False).tail(1)
    if symmetric:
        last = last[last.s != last.d]
        lo, hi = np.minimum(last.s, last.d), np.maximum(last.s, last.d)
        last = pd.DataFrame({"s": lo, "d": hi, "w": last.w.to_numpy()}).groupby(["s", "d"]).w.max().reset_index()
    return last.set_index(["s", "d"])["w"]


def evidence_only_grows(b_store, r_store, src_map, dst_map, symmetric=False, lag=0, excused=None) -> dict:
    """Per check year: release pairs missing from the bench graph, and with a lower score.

    ``src_map`` / ``dst_map`` take release node indices to bench indices (-1:
    not in the bench graph). Release pairs with score 0 are skipped; the bench
    graph is read ``lag`` years later than the release. Misses on ``excused``
    (a MultiIndex of bench (src, dst) pairs) are counted apart, as "excused".
    """
    ei = r_store.edge_index.numpy().astype(np.int64)
    s, d = src_map[ei[0]], dst_map[ei[1]]
    keep = (s >= 0) & (d >= 0)
    w = r_store.edge_attr[:, 0].numpy().astype(np.float64)
    rr = (s[keep], d[keep], r_store.edge_time.numpy()[keep], w[keep])
    eb = b_store.edge_index.numpy().astype(np.int64)
    rb = (eb[0], eb[1], b_store.edge_time.numpy(), b_store.edge_attr[:, 0].numpy().astype(np.float64))
    out = {"pairs": 0, "missing": 0, "lower": 0, "excused": 0, "earlier_in_bench": 0, "release_score0_skipped": 0,
           "release_rows_on_dropped_nodes": int((~keep).sum())}
    for y in CHECK_YEARS:
        r = _scores_before(*rr, y, symmetric=symmetric)
        out["release_score0_skipped"] += int((r <= 0).sum())
        r = r[r > 0]
        b = _scores_before(*rb, y + lag, symmetric=symmetric)
        j = b.reindex(r.index)
        miss, low = j.isna().to_numpy(), (j < r - 1e-6).to_numpy()
        ok = r.index.isin(excused) if excused is not None else np.zeros(len(r), bool)
        out["pairs"] += len(r)
        out["missing"] += int((miss & ~ok).sum())
        out["lower"] += int((low & ~ok).sum())
        out["excused"] += int(((miss | low) & ok).sum())
        out["earlier_in_bench"] += int(len(b) - j.notna().sum())
    return out


def undated_trials(clinical_dir: str) -> tuple[pd.DataFrame, pd.Series]:
    """ClinicalTrials.gov trials with no start date, which the bench graph leaves out.

    Returns, per (targetId, diseaseId), whether ANY / ALL of its trials lack a
    start date, and per drugId whether all of them do.
    """
    import pyarrow.dataset as ds
    t = ds.dataset(clinical_dir, format="parquet").to_table(
        columns=["targetId", "diseaseId", "drugId", "clinicalReportId", "studyStartDate"]).to_pandas()
    t = t[t["clinicalReportId"].astype(str).str.upper().str.match(r"^NCT\d{8}$")]
    t["undated"] = t["studyStartDate"].isna()
    pairs = t.groupby(["targetId", "diseaseId"])["undated"].agg(any_undated="any", all_undated="all")
    drugs = t.groupby("drugId")["undated"].all()
    return pairs, drugs


def _edges_df(store, src_ids, dst_ids) -> pd.DataFrame:
    ei = store.edge_index.numpy()
    df = pd.DataFrame({"src": np.asarray(src_ids)[ei[0]], "dst": np.asarray(dst_ids)[ei[1]],
                       "time": store.edge_time.numpy()})
    attr = store.edge_attr.numpy()
    df["w"], df["n"] = attr[:, 0], attr[:, 1]
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench-graph", required=True)
    ap.add_argument("--ref-graph", required=True)
    ap.add_argument("--out", required=True, help="JSON report path")
    ap.add_argument("--clinical-evidence", required=True,
                    help="raw clinical_precedence evidence (to identify trials with no start date)")
    args = ap.parse_args()

    bench = torch.load(args.bench_graph, weights_only=False)
    ref = torch.load(args.ref_graph, weights_only=False)
    report, failures = {}, []

    def check(name, ok, detail=None):
        report[name] = {"ok": bool(ok), **({"detail": detail} if detail is not None else {})}
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  {detail}" if detail is not None else ""))
        if not ok:
            failures.append(name)

    # release node index -> bench node index (-1: not in the bench graph)
    to_bench, b_index = {}, {}
    for nt in ref.node_types:
        b_index[nt] = {n: i for i, n in enumerate(bench[nt].node_id)}
        to_bench[nt] = np.array([b_index[nt].get(n, -1) for n in ref[nt].node_id], dtype=np.int64)

    # trials with no start date, which the bench graph leaves out (see module docstring)
    undated_pairs, undated_drugs = undated_trials(args.clinical_evidence)
    any_undated = undated_pairs.index[undated_pairs["any_undated"]]
    all_undated = set(undated_pairs.index[undated_pairs["all_undated"]])
    excused_ongoing = pd.MultiIndex.from_tuples(
        [(b_index["target"][t], b_index["disease"][d]) for t, d in any_undated
         if t in b_index["target"] and d in b_index["disease"]], names=["s", "d"])

    # 1. node types
    for nt in ("disease", "go", "reactome"):  # same nodes, re-encoded (see module docstring)
        same_ids = list(bench[nt].node_id) == list(ref[nt].node_id)
        check(f"nodes.{nt}.ids", same_ids, f"bench={bench[nt].num_nodes:,} ref={ref[nt].num_nodes:,}")
        if same_ids:
            bxd = bench[nt].x
            width = {"disease": 768, "go": 64, "reactome": 64}[nt]
            ok = bxd.dim() == 2 and bxd.size(0) == ref[nt].num_nodes and bool(torch.isfinite(bxd).all())
            n_distinct = int(torch.unique(bxd.round(decimals=4), dim=0).size(0))
            check(f"nodes.{nt}.x_reencoded", ok and bxd.size(1) == width and n_distinct >= 0.99 * bxd.size(0),
                  f"shape {list(bxd.shape)}, distinct rows {n_distinct:,} of {bxd.size(0):,}")
    tmap = to_bench["target"]
    gone = np.flatnonzero(tmap < 0)
    # a release target may only disappear if all its dated evidence scored 0
    best = np.zeros(ref["target"].num_nodes)
    for et in ref.edge_types:
        if "target" not in (et[0], et[2]) or bool((ref[et].edge_time < 0).all()):
            continue
        w = ref[et].edge_attr[:, 0].numpy().astype(np.float64)
        for side, nt in ((0, et[0]), (1, et[2])):
            if nt == "target":
                np.maximum.at(best, ref[et].edge_index[side].numpy(), w)
    added = bench["target"].num_nodes - int((tmap >= 0).sum())
    check("nodes.target.dropped_only_score0", bool((best[gone] <= 0).all()),
          f"release {ref['target'].num_nodes:,}, bench {bench['target'].num_nodes:,}: {len(gone)} dropped "
          f"(max release score {float(best[gone].max()) if len(gone) else 0:.3f}), {added} added")
    shared = np.flatnonzero(tmap >= 0)
    check("nodes.target.x_shared_identical",
          torch.equal(bench["target"].x[torch.from_numpy(tmap[shared])], ref["target"].x[torch.from_numpy(shared)]),
          f"{len(shared):,} shared targets")

    # 2. edge types
    b_types, r_types = set(bench.edge_types), set(ref.edge_types)
    check("edge_types.identical", b_types == r_types,
          {"missing": sorted(r_types - b_types), "extra": sorted(b_types - r_types)} if b_types != r_types else None)

    # 3. edge types without a molecule endpoint, except the rescored ongoing edge
    # and the static ontology edges (checked in 3b)
    static = {et for et in r_types if bool((ref[et].edge_time < 0).all())}  # INT64_MIN in the release
    dated = [et for et in sorted(r_types & b_types)
             if "molecule" not in (et[0], et[2]) and et not in static]
    report["evidence_only_grows"] = {}
    for et in dated:
        lag = REDATED_LAG.get(et[1], 0)
        g = evidence_only_grows(bench[et], ref[et], to_bench[et[0]], to_bench[et[2]],
                                symmetric=(et[0] == et[2]), lag=lag,
                                excused=excused_ongoing if et == ONGOING_EDGE else None)
        report["evidence_only_grows"]["|".join(et)] = g
        bad = g["missing"] + g["lower"]
        limit = 0.001 * g["pairs"] if et[1] in PUBMED_DATED or lag else 0
        if et[1] in REPORT_ONLY:
            print(f"INFO  edges.grows.{et[1]}  missing {g['missing']:,}, lower {g['lower']:,} "
                  f"of {g['pairs']:,} release pair-years (not checked)")
            continue
        check(f"edges.grows.{et[1]}", bad <= limit,
              f"missing {g['missing']:,}, lower {g['lower']:,} of {g['pairs']:,} release pair-years"
              + (f" (bench read {lag} year later)" if lag else "")
              + (f"; {g['excused']:,} excused (pair has a trial with no start date)" if g["excused"] else "")
              + f"; {g['earlier_in_bench']:,} bench pair-years absent from the release; "
              f"{g['release_score0_skipped']:,} score-0 release pair-years skipped")

    # 3b. static ontology edges: same edges, attributes fixed to (score 1, novelty 0)
    for et in sorted(static & b_types):
        r_ei = np.stack([to_bench[et[0]][ref[et].edge_index[0].numpy()], to_bench[et[2]][ref[et].edge_index[1].numpy()]])
        same_edges = (r_ei.min() >= 0 and r_ei.shape == tuple(bench[et].edge_index.shape)
                      and np.array_equal(np.unique(r_ei, axis=1), np.unique(bench[et].edge_index.numpy(), axis=1)))
        same_struct = same_edges and bool((bench[et].edge_time == -1).all())  # explicit static edge_time
        a = bench[et].edge_attr
        fixed = bool((a[:, 0] == 1).all() and (a[:, 1] == 0).all())
        check(f"edges.static.{et[1]}.{et[0]}", same_struct and fixed,
              f"same edges/times={same_struct}, attr (1,0)={fixed}")

    # 4. ongoing trial edge: same pairs and first years, scores only go up
    if ONGOING_EDGE in b_types and ONGOING_EDGE in r_types:
        b_on = _edges_df(bench[ONGOING_EDGE], list(bench["target"].node_id), list(bench["disease"].node_id))
        r_on = _edges_df(ref[ONGOING_EDGE], list(ref["target"].node_id), list(ref["disease"].node_id))
        # compare on the release's time range
        b_on = b_on[b_on["time"] <= RELEASE_YEARS[1]].assign(time=lambda x: x["time"].clip(lower=RELEASE_YEARS[0]))
        agg = {"time": "min", "w": "max", "n": "max"}
        b_pair = b_on.groupby(["src", "dst"]).agg(agg)
        r_pair = r_on.groupby(["src", "dst"]).agg(agg)
        gone = r_pair.index.difference(b_pair.index)
        new = b_pair.index.difference(r_pair.index)
        gone_unexplained = [p for p in gone if p not in all_undated]
        check("edges.ongoing.same_pairs", not gone_unexplained and len(new) == 0,
              f"bench={len(b_pair):,} ref={len(r_pair):,}; {len(gone)} release pairs gone "
              f"({len(gone) - len(gone_unexplained)} whose trials all lack a start date); {len(new)} new")
        # first years and final scores, on the pairs whose trials all have a start date
        shared = r_pair.index.intersection(b_pair.index)
        shared = shared[~shared.isin(any_undated)]
        bp, rp = b_pair.loc[shared], r_pair.loc[shared]
        check("edges.ongoing.same_first_year", bool((bp["time"] == rp["time"]).all()),
              f"{len(shared):,} pairs compared")
        drops = int((bp["w"] < rp["w"] - 1e-9).sum())
        check("edges.ongoing.final_score_never_drops", drops == 0, f"pairs with lower score: {drops}")
        check("edges.ongoing.novelty_unchanged", bool(((b_on["n"] == 0).all()) and ((r_on["n"] == 0).all())))
        report["ongoing_rescore"] = {
            "pairs": len(b_pair), "pairs_compared": len(shared),
            "pairs_final_score_raised": int((bp["w"] > rp["w"] + 1e-9).sum()),
            "events_bench": len(b_on), "events_ref": len(r_on),
        }
        print(json.dumps(report["ongoing_rescore"]))
    else:
        check("edges.ongoing.present", False, "clinical_trial_ongoing missing")

    # 5. modulated_by on the release's molecules
    b_mol, r_mol = list(bench["molecule"].node_id), list(ref["molecule"].node_id)
    gone_mol = sorted(set(r_mol) - set(b_mol))
    unexplained_mol = [m for m in gone_mol if not undated_drugs.get(m, False)]
    check("nodes.molecule.release_kept", not unexplained_mol,
          f"bench={len(b_mol):,} ref={len(r_mol):,}; {len(gone_mol)} release molecules gone "
          f"({len(gone_mol) - len(unexplained_mol)} whose trials all lack a start date)"
          + (f"; unexplained e.g. {unexplained_mol[:5]}" if unexplained_mol else ""))
    g = evidence_only_grows(bench[MOD_EDGE], ref[MOD_EDGE], to_bench["molecule"], to_bench["target"])
    report["evidence_only_grows"]["|".join(MOD_EDGE)] = g
    check("edges.modulated_by.few_pairs_missing", g["missing"] <= 0.01 * g["pairs"],
          f"missing {g['missing']:,} of {g['pairs']:,} release pair-years; lower score {g['lower']:,} "
          f"(expected: trials no longer counted twice); bench molecules {len(b_mol):,} vs release {len(r_mol):,}")

    # 6. Morgan block for release molecules
    b_idx = {m: i for i, m in enumerate(b_mol)}
    bm, rx = bench["molecule"], ref["molecule"].x
    check("features.molecule.per_modality", "x" not in bm and all(k in bm for k in (
        "x_morgan", "x_esm", "has_morgan", "has_esm", "drug_type")),
        sorted(k for k in bm.keys() if k != "node_id"))
    kept = [i for i, m in enumerate(r_mol) if m in b_idx]
    rows = torch.tensor([b_idx[r_mol[i]] for i in kept])
    rx = rx[torch.tensor(kept)]
    match = (bm.x_morgan[rows] == rx).all(dim=1)
    imputed = ~((rx == 0) | (rx == 1)).all(dim=1)  # the release mean-imputed unparsable SMILES
    unexplained = int((~match & ~imputed).sum())
    check("features.molecule.release_morgan_preserved", unexplained == 0,
          f"match={int(match.sum()):,} release_imputed={int(imputed.sum()):,} unexplained={unexplained}")

    # a modality's rows are zero exactly where it is absent
    zero_ok = all(bool((bm[f"x_{m}"][~bm[f"has_{m}"]] == 0).all()) for m in ("morgan", "esm"))
    check("features.molecule.absent_rows_zero", zero_ok)

    # reported: modality coverage
    hm, he = bm.has_morgan.bool(), bm.has_esm.bool()
    report["molecule_coverage"] = {
        "n": len(b_mol),
        "morgan_only": int((hm & ~he).sum()),
        "esm_only": int((~hm & he).sum()),
        "both": int((hm & he).sum()),
        "neither": int((~hm & ~he).sum()),
        "x_morgan": list(bm.x_morgan.shape), "x_esm": list(bm.x_esm.shape),
    }

    print(json.dumps(report["molecule_coverage"], indent=2))

    report["failures"] = failures
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n{'❌ ' + str(len(failures)) + ' check(s) failed' if failures else '✅ all checks passed'} -> {args.out}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
