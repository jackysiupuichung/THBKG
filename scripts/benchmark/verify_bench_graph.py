#!/usr/bin/env python
"""Check the benchmark graph against the released v26.09 graph.

Relative to the release, the benchmark build: makes all drug types molecule
nodes with per-modality features; rescores the start-dated clinical_trial_ongoing
edge with the trial's phase weight; gives static ontology edges (score 1,
novelty 0) and edge_time -1; computes each year's top-50 evidence from that
year's rows only; recovers PubMed-dated GO / IntAct rows lost in 26.09; stores
IntAct pairs once in (min, max) order without self-loops; counts each
(drug, target, trial) once in modulated_by; and re-encodes disease, GO and
Reactome features. The checks below pin down what must NOT change. Fails
(exit 1) on any hard check.

Hard checks
  1. target/disease/go/reactome: same node ids in the same order; target
     features identical; disease (768-d text) and GO / Reactome (64-d
     autoencoder, now with a learnable node input) re-encoded, so their shape,
     finiteness and distinct rows are checked.
  2. edge types: exactly the release's types.
  3. every dated edge type without a molecule endpoint: EVIDENCE ONLY GROWS.
     At each check year Y, every pair the release has before Y is present
     before Y with a score >= the release's. The per-year top-50 fix and the
     recovered PubMed-dated GO / IntAct rows only add evidence; anything else
     is a bug. (Up to 0.1% violations allowed for the two PubMed-dated
     relations, where PubMed itself may have re-dated a record.)
     interacts_with is compared on (min, max) pairs without self-loops, the
     release's two directions taking their higher score.
  3b. static ontology types: identical edge_index/edge_time; edge_attr is
     exactly (1, 0).
  4. clinical_trial_ongoing: same (target, disease) pairs with the same first
     edge_time (so first-trial labels are unchanged); the final score per pair
     never drops; novelty unchanged (always 0).
  5. modulated_by on the release's molecules: no pair goes missing at any check
     year (scores may fall: the release counted each trial up to twice).
  6. release molecules keep their exact Morgan fingerprint (block 0) unless the
     release had mean-imputed them.
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
PUBMED_DATED = {"has_function_in", "interacts_with"}


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


def evidence_only_grows(b_store, r_store, r_src_to_b=None, symmetric=False) -> dict:
    """Per check year: release pairs missing from the bench graph, and with a lower score."""
    def cols(store, src_map=None):
        ei = store.edge_index.numpy().astype(np.int64)
        src = ei[0] if src_map is None else src_map[ei[0]]
        return src, ei[1], store.edge_time.numpy(), store.edge_attr[:, 0].numpy().astype(np.float64)
    rb, rr = cols(b_store), cols(r_store, r_src_to_b)
    out = {"pairs": 0, "missing": 0, "lower": 0, "earlier_in_bench": 0}
    for y in CHECK_YEARS:
        r = _scores_before(*rr, y, symmetric=symmetric)
        b = _scores_before(*rb, y, symmetric=symmetric)
        j = b.reindex(r.index)
        out["pairs"] += len(r)
        out["missing"] += int(j.isna().sum())
        out["lower"] += int((j < r - 1e-6).sum())
        out["earlier_in_bench"] += int(len(b) - j.notna().sum())
    return out


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
    args = ap.parse_args()

    bench = torch.load(args.bench_graph, weights_only=False)
    ref = torch.load(args.ref_graph, weights_only=False)
    report, failures = {}, []

    def check(name, ok, detail=None):
        report[name] = {"ok": bool(ok), **({"detail": detail} if detail is not None else {})}
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  {detail}" if detail is not None else ""))
        if not ok:
            failures.append(name)

    # 1. unchanged node types
    for nt in ("target", "disease", "go", "reactome"):
        same_ids = list(bench[nt].node_id) == list(ref[nt].node_id)
        check(f"nodes.{nt}.ids", same_ids, f"bench={bench[nt].num_nodes:,} ref={ref[nt].num_nodes:,}")
        if same_ids:
            if nt in ("disease", "go", "reactome"):  # re-encoded (see module docstring)
                bxd = bench[nt].x
                width = {"disease": 768, "go": 64, "reactome": 64}[nt]
                ok = bxd.dim() == 2 and bxd.size(0) == ref[nt].num_nodes and bool(torch.isfinite(bxd).all())
                n_distinct = int(torch.unique(bxd.round(decimals=4), dim=0).size(0))
                check(f"nodes.{nt}.x_reencoded", ok and bxd.size(1) == width and n_distinct >= 0.99 * bxd.size(0),
                      f"shape {list(bxd.shape)}, distinct rows {n_distinct:,} of {bxd.size(0):,}")
            else:
                check(f"nodes.{nt}.x", torch.equal(bench[nt].x, ref[nt].x))

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
        g = evidence_only_grows(bench[et], ref[et], symmetric=(et[0] == et[2]))
        report["evidence_only_grows"]["|".join(et)] = g
        bad = g["missing"] + g["lower"]
        limit = 0.001 * g["pairs"] if et[1] in PUBMED_DATED else 0
        check(f"edges.grows.{et[1]}", bad <= limit,
              f"missing {g['missing']:,}, lower {g['lower']:,} of {g['pairs']:,} release pair-years; "
              f"{g['earlier_in_bench']:,} bench pair-years absent from the release")

    # 3b. static ontology edges: same edges, attributes fixed to (score 1, novelty 0)
    for et in sorted(static & b_types):
        same_struct = (torch.equal(bench[et].edge_index, ref[et].edge_index)
                       and bool((bench[et].edge_time == -1).all()))  # explicit static edge_time
        a = bench[et].edge_attr
        fixed = bool((a[:, 0] == 1).all() and (a[:, 1] == 0).all())
        check(f"edges.static.{et[1]}.{et[0]}", same_struct and fixed,
              f"same edges/times={same_struct}, attr (1,0)={fixed}")

    # 4. ongoing trial edge: same pairs and first years, scores only go up
    if ONGOING_EDGE in b_types and ONGOING_EDGE in r_types:
        tgt_ids, dis_ids = list(bench["target"].node_id), list(bench["disease"].node_id)
        b_on = _edges_df(bench[ONGOING_EDGE], tgt_ids, dis_ids)
        r_on = _edges_df(ref[ONGOING_EDGE], tgt_ids, dis_ids)
        agg = {"time": "min", "w": "max", "n": "max"}
        b_pair = b_on.groupby(["src", "dst"]).agg(agg)
        r_pair = r_on.groupby(["src", "dst"]).agg(agg)
        same_pairs = b_pair.index.equals(r_pair.index)  # groupby sorts both
        check("edges.ongoing.same_pairs", same_pairs, f"bench={len(b_pair):,} ref={len(r_pair):,}")
        if same_pairs:
            check("edges.ongoing.same_first_year", bool((b_pair["time"] == r_pair["time"]).all()))
            drops = int((b_pair["w"] < r_pair["w"] - 1e-9).sum())
            check("edges.ongoing.final_score_never_drops", drops == 0, f"pairs with lower score: {drops}")
            check("edges.ongoing.novelty_unchanged", bool(((b_on["n"] == 0).all()) and ((r_on["n"] == 0).all())))
            report["ongoing_rescore"] = {
                "pairs": len(b_pair),
                "pairs_final_score_raised": int((b_pair["w"] > r_pair["w"] + 1e-9).sum()),
                "events_bench": len(b_on), "events_ref": len(r_on),
            }
            print(json.dumps(report["ongoing_rescore"]))
    else:
        check("edges.ongoing.present", False, "clinical_trial_ongoing missing")

    # 5. modulated_by on the release's molecules
    b_mol, r_mol = list(bench["molecule"].node_id), list(ref["molecule"].node_id)
    r_set = set(r_mol)
    superset = r_set <= set(b_mol)
    check("nodes.molecule.superset", superset, f"bench={len(b_mol):,} ref={len(r_mol):,}")
    if not superset:  # the checks below index release molecules in the bench graph
        report["failures"] = failures
        json.dump(report, open(args.out, "w"), indent=2)
        print(f"\n❌ {len(failures)} check(s) failed -> {args.out}")
        return 1
    b_index = {m: i for i, m in enumerate(b_mol)}
    r_to_b = np.array([b_index[m] for m in r_mol], dtype=np.int64)
    g = evidence_only_grows(bench[MOD_EDGE], ref[MOD_EDGE], r_src_to_b=r_to_b)
    report["evidence_only_grows"]["|".join(MOD_EDGE)] = g
    check("edges.modulated_by.no_pair_missing", g["missing"] == 0,
          f"missing {g['missing']:,} of {g['pairs']:,} release pair-years; lower score {g['lower']:,} "
          f"(expected: trials no longer counted twice); bench molecules {len(b_mol):,} vs release {len(r_mol):,}")

    # 6. Morgan block for release molecules
    b_idx = {m: i for i, m in enumerate(b_mol)}
    bm, rx = bench["molecule"], ref["molecule"].x
    check("features.molecule.per_modality", "x" not in bm and all(k in bm for k in (
        "x_morgan", "x_esm", "has_morgan", "has_esm", "drug_type")),
        sorted(k for k in bm.keys() if k != "node_id"))
    rows = torch.tensor([b_idx[m] for m in r_mol])
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
