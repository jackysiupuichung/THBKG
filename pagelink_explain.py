"""PaGE-Link path explanations for the EAHGT advancement predictor.

Reimplements PaGE-Link (Zhang et al., WWW 2023, arXiv:2302.12465) against our
PyG EAHGT model. Two stages per (target, disease) pair:

  STAGE 1 (mask learning): learn a soft edge mask m_e in [0,1] over the pair's
    subgraph that, when scaled into HGTConv message passing, preserves the
    model's advancement logit (BCE to the unmasked prediction), with size +
    entropy regularisers driving the mask sparse and decisive. Model frozen.

  STAGE 2 (path enforcement): turn the soft mask into connected target->disease
    PATHS. Edge cost = -log(m_e); the k lowest-cost simple paths (Yen, via
    networkx) are the explanation — connection-interpretable chains through the
    relation types, the ChronoMedKG-style decomposition.

Outputs reuse the per_pair_edges.parquet schema (mask weight in the ig_total
slot) so export_pair_evidence_json.py / present_pair_evidence.py render
PaGE-Link explanations with OT evidence unchanged, plus per_pair_paths.parquet.

GPU job via sbatch (mask learning is per-pair gradient descent). See
note/explain_pagelink.md.

Invocation:
    uv run python pagelink_explain.py \
        --config <run>/config.yaml --checkpoint <run>/best_model.pt \
        --pairs-csv explain_pairs_evfree_diverse.csv \
        --mask-epochs 200 --lr 0.01 --size-coeff 5e-3 --entropy-coeff 1e-1 \
        --num-paths 5 --out-dir <run>/explanations/pagelink
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data.temporal_loader import ADV_ETYPE, build_edge_time_dict
from src.explain.runtime import ExplainRuntime, build_edge_feat_dict
from src.explain.edge_mask import EdgeMask, apply_edge_mask

EdgeType = Tuple[str, str, str]


# ----------------------------------------------------------------------------
# Stage 1 — learn the soft edge mask.
# ----------------------------------------------------------------------------
def learn_edge_mask(rt: ExplainRuntime, batch, edge_time_dict, args) -> EdgeMask:
    """Optimise an EdgeMask so the masked subgraph reproduces the model's logit.

    Loss = BCE(masked_logit, sigmoid(base_logit)) + size/entropy reg. The model
    is frozen; only mask logits are optimised (Adam). The mask is built over the
    SAME edge_index_dict the conv iterates, so flat() aligns with message order.
    """
    for p in rt.model.parameters():
        p.requires_grad_(False)

    edge_order = list(batch.edge_index_dict.keys())
    edge_counts = {et: batch[et].edge_index.size(1) for et in edge_order}
    mask = EdgeMask(edge_counts, edge_order, init=1.0, device=rt.device)

    with torch.no_grad():
        base_logit = rt.predict_logit(batch, edge_time_dict=edge_time_dict).detach()
    base_p = torch.sigmoid(base_logit)

    opt = torch.optim.Adam(mask.parameters(), lr=args.lr)
    for ep in range(args.mask_epochs):
        opt.zero_grad()
        with apply_edge_mask(rt.model, mask):
            logit = rt.predict_logit(batch, edge_time_dict=edge_time_dict)
        # Keep the masked prediction matched to the original (PaGE-Link fidelity
        # objective): BCE against the model's own probability, plus mask reg.
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logit, base_p)
        loss = bce + mask.regularisation(args.size_coeff, args.entropy_coeff)
        loss.backward()
        opt.step()
        if args.verbose and (ep % max(1, args.mask_epochs // 5) == 0 or ep == args.mask_epochs - 1):
            with torch.no_grad():
                m = mask.flat()
                print(f"[pagelink]     ep{ep:>3} loss={loss.item():.4f} "
                      f"bce={bce.item():.4f} mask[mean={m.mean():.3f} "
                      f">0.5={float((m > 0.5).float().mean()):.3f}]", flush=True)
    return mask


# ----------------------------------------------------------------------------
# Stage 2 — enforce target->disease paths from the soft mask.
# ----------------------------------------------------------------------------
def _global_node(batch, ntype: str, local_idx: int) -> Tuple[str, int]:
    return (ntype, int(batch[ntype].n_id[local_idx].item()))


def enforce_paths(rt: ExplainRuntime, batch, mask: EdgeMask, num_paths: int,
                  min_mask: float) -> Tuple[List[dict], Dict[Tuple, float]]:
    """k lowest-cost simple target->disease paths under cost -log(m_e).

    Builds a directed multigraph over (node_type, global_idx) nodes from the
    subgraph edges (canonicalised to forward orientation), weighting each edge by
    -log(m_e). Returns (paths, edge_mask_by_key) where each path is an ordered
    list of typed edges and edge_mask_by_key maps a global edge key to its m_e
    (for the per_pair_edges output).
    """
    import networkx as nx

    mvals = {et: v.detach().cpu().numpy() for et, v in mask.values().items()}
    G = nx.DiGraph()
    edge_mask_by_key: Dict[Tuple, float] = {}

    for et in batch.edge_index_dict:
        st, rel, dt = et
        ei = batch[et].edge_index.cpu().numpy()
        m = mvals.get(et)
        if m is None:
            continue
        for j in range(ei.shape[1]):
            su, dv = int(ei[0, j]), int(ei[1, j])
            me = float(m[j])
            if me < min_mask:
                continue
            gn_s = _global_node(batch, st, su)
            gn_d = _global_node(batch, dt, dv)
            cost = -math.log(max(me, 1e-6))
            key = (et, gn_s[1], gn_d[1])
            edge_mask_by_key[key] = me
            # Keep the lowest-cost parallel edge between the same node pair.
            if G.has_edge(gn_s, gn_d):
                if cost < G[gn_s][gn_d]["cost"]:
                    G[gn_s][gn_d].update(cost=cost, et=et, me=me)
            else:
                G.add_edge(gn_s, gn_d, cost=cost, et=et, me=me)

    # Endpoints: the query edge in global node space.
    ls = int(batch[ADV_ETYPE].edge_label_index[0, 0].item())
    ld = int(batch[ADV_ETYPE].edge_label_index[1, 0].item())
    src = ("target", int(batch["target"].n_id[ls].item()))
    dst = ("disease", int(batch["disease"].n_id[ld].item()))

    paths: List[dict] = []
    if src in G and dst in G:
        try:
            gen = nx.shortest_simple_paths(G, src, dst, weight="cost")
            for rank, nodes in enumerate(gen):
                if rank >= num_paths:
                    break
                edges = []
                total = 0.0
                for a, b in zip(nodes[:-1], nodes[1:]):
                    d = G[a][b]
                    edges.append({"edge_type": "::".join(d["et"]),
                                  "src_type": a[0], "src_global": a[1],
                                  "dst_type": b[0], "dst_global": b[1],
                                  "m_e": d["me"]})
                    total += d["cost"]
                paths.append({"rank": rank, "n_hops": len(edges),
                              "total_cost": total, "edges": edges})
        except nx.NetworkXNoPath:
            pass
    return paths, edge_mask_by_key


# ----------------------------------------------------------------------------
# Output.
# ----------------------------------------------------------------------------
def _edge_rows(rt: ExplainRuntime, batch, mask: EdgeMask, t_id: str, d_id: str) -> List[dict]:
    """Per-edge rows in the per_pair_edges.parquet schema, mask weight in the
    ig_total slot so the existing decomposition tooling consumes it unchanged."""
    mvals = {et: v.detach().cpu().numpy() for et, v in mask.values().items()}
    rows = []
    for et in batch.edge_index_dict:
        ei = batch[et].edge_index.cpu().numpy()
        m = mvals.get(et)
        if m is None:
            continue
        st, rel, dt = et
        for j in range(ei.shape[1]):
            su, dv = int(ei[0, j]), int(ei[1, j])
            rows.append({
                "target_id": t_id, "disease_id": d_id,
                "edge_type": "::".join(et),
                "src": int(batch[st].n_id[su].item()),
                "dst": int(batch[dt].n_id[dv].item()),
                "ig_total": float(m[j]),      # mask weight as the attribution
                "mask_weight": float(m[j]),
            })
    return rows


def main(args: argparse.Namespace) -> None:
    rt = ExplainRuntime.from_config(args.config, args.checkpoint)
    print(f"[pagelink] device={rt.device}; mask_epochs={args.mask_epochs}; "
          f"num_paths={args.num_paths}", flush=True)

    pair_idx = rt.select_pairs_from_csv(args.pairs_csv)
    print(f"[pagelink] {len(pair_idx)} pairs from {args.pairs_csv}", flush=True)
    if len(pair_idx) == 0:
        raise SystemExit("[pagelink] no pairs resolved to the test split")
    loader = rt.pair_loader(pair_idx)

    all_edge_rows: List[dict] = []
    all_path_rows: List[dict] = []
    for bi, batch in enumerate(loader):
        batch = batch.to(rt.device)
        etd = build_edge_time_dict(batch, ADV_ETYPE)
        t_id, d_id = rt.pair_ids(batch)
        print(f"[pagelink] {bi+1}/{len(pair_idx)} {t_id}->{d_id}: learning mask...",
              flush=True)

        mask = learn_edge_mask(rt, batch, etd, args)
        all_edge_rows.extend(_edge_rows(rt, batch, mask, t_id, d_id))

        paths, _ = enforce_paths(rt, batch, mask, args.num_paths, args.min_mask)
        for p in paths:
            chain = " -> ".join(
                [f"{p['edges'][0]['src_type']}#{p['edges'][0]['src_global']}"]
                + [f"[{e['edge_type']}] {e['dst_type']}#{e['dst_global']}" for e in p["edges"]]
            )
            all_path_rows.append({
                "target_id": t_id, "disease_id": d_id, "rank": p["rank"],
                "n_hops": p["n_hops"], "total_cost": p["total_cost"],
                "min_m_e": min((e["m_e"] for e in p["edges"]), default=float("nan")),
                "path": chain,
            })
        print(f"[pagelink]   -> {len(paths)} path(s); "
              f"{len([r for r in all_edge_rows if r['target_id']==t_id and r['disease_id']==d_id])} edges",
              flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_edge_rows).to_parquet(out_dir / "per_pair_edges.parquet", index=False)
    paths_df = pd.DataFrame(all_path_rows)
    paths_df.to_parquet(out_dir / "per_pair_paths.parquet", index=False)
    if not paths_df.empty:
        print("\n[pagelink] top paths:", flush=True)
        print(paths_df.sort_values(["target_id", "rank"])
              .head(20).to_string(index=False), flush=True)
    print(f"\n[pagelink] wrote {len(all_edge_rows)} edge rows, "
          f"{len(all_path_rows)} path rows -> {out_dir}", flush=True)


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PaGE-Link path explanations for EAHGT advancement.")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--pairs-csv", required=True,
                   help="target_id,disease_id pairs to explain.")
    p.add_argument("--mask-epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--size-coeff", type=float, default=5e-3)
    p.add_argument("--entropy-coeff", type=float, default=1e-1)
    p.add_argument("--num-paths", type=int, default=5,
                   help="k lowest-cost target->disease paths per pair.")
    p.add_argument("--min-mask", type=float, default=0.1,
                   help="Drop edges with m_e below this from path search.")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    main(args)
