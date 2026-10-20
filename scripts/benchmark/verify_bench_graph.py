#!/usr/bin/env python
"""Check the benchmark graph against the released v26.09 graph.

The benchmark build changes two things: all drug types become molecule nodes,
with multimodal features, and the start-dated clinical_trial_ongoing edge
carries the trial's phase weight instead of Open Targets' score (which is halved
for trials that later stopped for safety or efficacy). Everything else must be
bit-identical to the release. Fails (exit 1) on any hard check.

Hard checks
  1. target/disease/go/reactome: same node ids in the same order, same features.
  2. edge types: exactly the release's types.
  3. every edge type without a molecule endpoint, except clinical_trial_ongoing:
     identical edge_index/attr/time.
  4. clinical_trial_ongoing: same (target, disease) pairs with the same first
     edge_time (so first-trial labels are unchanged); the final score per pair
     never drops; novelty unchanged (always 0).
  5. modulated_by restricted to the release's molecules equals the release's.
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
    ap.add_argument("--fp-dim", type=int, default=1024)
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
            check(f"nodes.{nt}.x", torch.equal(bench[nt].x, ref[nt].x))

    # 2. edge types
    b_types, r_types = set(bench.edge_types), set(ref.edge_types)
    check("edge_types.identical", b_types == r_types,
          {"missing": sorted(r_types - b_types), "extra": sorted(b_types - r_types)} if b_types != r_types else None)

    # 3. edge types without a molecule endpoint, except the rescored ongoing edge
    same = [et for et in sorted(r_types) if "molecule" not in (et[0], et[2]) and et != ONGOING_EDGE]
    diff = [et for et in same
            if not all(torch.equal(bench[et][a], ref[et][a]) for a in ("edge_index", "edge_attr", "edge_time"))]
    check("edges.non_molecule_identical", not diff, f"{len(same)} types compared" if not diff else diff)

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
    check("nodes.molecule.superset", r_set <= set(b_mol), f"bench={len(b_mol):,} ref={len(r_mol):,}")
    tgt = list(bench["target"].node_id)
    b_mod = _edges_df(bench[MOD_EDGE], b_mol, tgt)
    r_mod = _edges_df(ref[MOD_EDGE], r_mol, tgt)
    b_mod_old = b_mod[b_mod["src"].isin(r_set)]
    key = ["src", "dst", "time", "w", "n"]
    same_mod = (len(b_mod_old) == len(r_mod)
                and b_mod_old.sort_values(key).reset_index(drop=True).equals(r_mod.sort_values(key).reset_index(drop=True)))
    check("edges.modulated_by.release_molecules_identical", same_mod,
          f"bench(old mols)={len(b_mod_old):,} ref={len(r_mod):,} bench(all)={len(b_mod):,}")

    # 6. Morgan block for release molecules
    b_idx = {m: i for i, m in enumerate(b_mol)}
    bx, rx = bench["molecule"].x, ref["molecule"].x
    rows = torch.tensor([b_idx[m] for m in r_mol])
    match = (bx[rows, : args.fp_dim] == rx).all(dim=1)
    imputed = ~((rx == 0) | (rx == 1)).all(dim=1)  # the release mean-imputed unparsable SMILES
    unexplained = int((~match & ~imputed).sum())
    check("features.molecule.release_morgan_preserved", unexplained == 0,
          f"match={int(match.sum()):,} release_imputed={int(imputed.sum()):,} unexplained={unexplained}")

    # reported: modality coverage (last two columns are has_morgan, has_esm)
    flags = bx[:, -2:].bool()
    report["molecule_coverage"] = {
        "n": len(b_mol),
        "morgan_only": int((flags[:, 0] & ~flags[:, 1]).sum()),
        "esm_only": int((~flags[:, 0] & flags[:, 1]).sum()),
        "both": int((flags[:, 0] & flags[:, 1]).sum()),
        "neither": int((~flags[:, 0] & ~flags[:, 1]).sum()),
        "feature_dim": int(bx.size(1)),
    }

    print(json.dumps(report["molecule_coverage"], indent=2))

    report["failures"] = failures
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n{'❌ ' + str(len(failures)) + ' check(s) failed' if failures else '✅ all checks passed'} -> {args.out}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
