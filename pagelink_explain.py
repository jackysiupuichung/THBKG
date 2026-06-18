"""PaGE-Link path explanations for the EAHGT advancement predictor (SCAFFOLD).

Reimplements PaGE-Link (Zhang et al., WWW 2023, arXiv:2302.12465) against our
PyG EAHGT model: learn a soft edge mask over the query pair's subgraph, then
enforce connected target -> ... -> disease PATHS from that mask. Produces
path-structured, connection-interpretable explanations through the 20 relation
types — the publication-grade structural answer the IG/attention signals can't
give.

See note/explain_pagelink.md for the algorithm and adaptation plan.

STATUS: scaffold. The two stages and output writers are stubbed with intended
signatures; model/subgraph plumbing reuses explain_advancement.py (shared
runtime helper, same one branch #1 needs). No graph/model load is triggered.
Implementation is a GPU sbatch job (mask learning = per-pair gradient descent).

Intended invocation (GPU, via sbatch):
    uv run python pagelink_explain.py \
        --config runs/<exp>/config.yaml \
        --checkpoint runs/<exp>/best_model.pt \
        --pairs-csv explain_pairs_evfree_diverse.csv \
        --mask-epochs 200 --size-coeff 5e-3 --entropy-coeff 1e-1 \
        --num-paths 5 --out-dir runs/<exp>/explanations/pagelink
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Tuple

import pandas as pd

EdgeType = Tuple[str, str, str]


def learn_edge_mask(model, batch, args) -> "object":
    """STAGE 1 — learn m_e in [0,1] over the subgraph edges.

    Freeze ``model``; optimise an EdgeMask (src/explain/edge_mask.py) to maximise
    the masked query-edge logit's agreement with the unmasked prediction, plus
    size + entropy regularisers. Returns the trained EdgeMask.

    TODO(impl): Adam over mask.logits for args.mask_epochs; loss =
    pred_match(masked_logit, base_logit) + size_coeff*size + entropy_coeff*entropy.
    """
    raise NotImplementedError("scaffold: mask-learning loop (see note)")


def enforce_paths(batch, mask, src_idx: int, dst_idx: int,
                  num_paths: int) -> List[dict]:
    """STAGE 2 — convert the soft mask into connected target->disease paths.

    Build a weighted digraph over the subgraph edges with cost w_e = -log(m_e);
    find the ``num_paths`` lowest-cost (target -> disease) paths; keep those with
    an admissible relation sequence. Canonicalise rev_ edges to forward
    orientation (reuse join_pair_evidence._canonicalise) so paths read forward.

    Returns one dict per path: ordered [(edge_type, src_acc, dst_acc, m_e)],
    total_cost, rank.

    TODO(impl): k-shortest-paths over the mask-weighted subgraph.
    """
    raise NotImplementedError("scaffold: path enforcement (see note)")


def write_outputs(pair_paths: Dict[Tuple[str, str], List[dict]],
                  edge_attr: pd.DataFrame, out_dir: str) -> None:
    """Emit:
      * per_pair_edges.parquet  -- mask weight in the ``ig_total`` slot, same
        schema export_pair_evidence_json.py / present_pair_evidence.py consume,
        so PaGE-Link explanations render with OT evidence unchanged.
      * per_pair_paths.parquet  -- one row per path (ordered typed edges + cost).
    TODO(impl).
    """
    raise NotImplementedError("scaffold: output writers (see note)")


def _load_model_and_loader(args):
    """Reuse explain_advancement.py model build + checkpoint load +
    LinkNeighborLoader (shared runtime helper). TODO(impl): extract + import."""
    raise NotImplementedError("scaffold: share runtime with explain_advancement.py")


def main(args: argparse.Namespace) -> None:
    pairs = pd.read_csv(args.pairs_csv)  # columns: target_id, disease_id
    print(f"[pagelink] {len(pairs)} pair(s); mask_epochs={args.mask_epochs}; "
          f"num_paths={args.num_paths}", flush=True)
    # model, loader, context = _load_model_and_loader(args)               # TODO
    # for each pair:
    #   batch = subgraph for (t, d); src_idx, dst_idx from edge_label_index
    #   mask  = learn_edge_mask(model, batch, args)
    #   paths = enforce_paths(batch, mask, src_idx, dst_idx, args.num_paths)
    #   ...accumulate per-edge mask weights + paths
    # write_outputs(pair_paths, edge_attr, args.out_dir)
    print("[pagelink] SCAFFOLD: stages stubbed (see note/explain_pagelink.md). "
          f"Would write {args.out_dir}", flush=True)


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PaGE-Link path explanations for EAHGT advancement (scaffold).")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--pairs-csv", required=True,
                   help="target_id,disease_id pairs to explain.")
    p.add_argument("--mask-epochs", type=int, default=200)
    p.add_argument("--size-coeff", type=float, default=5e-3)
    p.add_argument("--entropy-coeff", type=float, default=1e-1)
    p.add_argument("--num-paths", type=int, default=5,
                   help="k shortest target->disease paths to return per pair.")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    main(_parse())
