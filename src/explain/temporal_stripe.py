"""Temporal evidence-accumulation stripe (Layout C) for explained pairs.

For one (target, disease) prediction, render small-multiples histograms — one
row per edge in the explainer subgraph — showing how supporting evidence
accumulated over time. A vertical line marks the prediction cutoff
(``edge_label_time``); years to the left are what the model saw, years to
the right are post-hoc.

Reads two parquets written by ``explain_advancement.py``:
  - ``per_pair_edges.parquet``    (for IG attribution per edge type)
  - ``per_pair_evidence.parquet`` (for year-by-year evidence counts)

The stripe needs only ``year`` from per_pair_evidence — no PubMed resolution.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _top_edge_types(
    edges_df: pd.DataFrame,
    target_id: str,
    disease_id: str,
    top_n: int,
) -> list[str]:
    """Pick the top-N edge types by total |ig_total| for one pair.

    Aggregating by edge_type (not edge) matches the granularity of the
    accumulation strips — one row per edge type, not per individual edge.
    """
    pair = edges_df[
        (edges_df["target_id"] == target_id)
        & (edges_df["disease_id"] == disease_id)
    ]
    if pair.empty:
        return []
    grouped = (
        pair.assign(abs_ig=pair["ig_total"].abs())
        .groupby("edge_type")["abs_ig"]
        .sum()
        .sort_values(ascending=False)
    )
    return grouped.head(top_n).index.tolist()


def _pair_cutoff_year(evidence_df: pd.DataFrame, target_id: str, disease_id: str) -> Optional[int]:
    """Recover the cutoff used when the bridge was queried.

    The bridge filters with ``year <= cutoff`` per pair (the pair's
    ``edge_label_time``). So the cutoff is simply the maximum year present
    in this pair's evidence rows. Returns None if the pair has no rows.
    """
    pair = evidence_df[
        (evidence_df["target_id"] == target_id)
        & (evidence_df["disease_id"] == disease_id)
    ]
    if pair.empty or "year" not in pair.columns:
        return None
    years = pair["year"].dropna()
    if years.empty:
        return None
    return int(years.max())


def plot_temporal_stripe(
    edges_df: pd.DataFrame,
    evidence_df: pd.DataFrame,
    target_id: str,
    disease_id: str,
    output_path: Path,
    top_n: int = 6,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
    cutoff_year: Optional[int] = None,
    target_name: Optional[str] = None,
    disease_name: Optional[str] = None,
) -> Optional[Path]:
    """Render Option C for one (target, disease) pair.

    Args:
        edges_df:     per_pair_edges.parquet content
        evidence_df:  per_pair_evidence.parquet content
        target_id:    Ensembl gene ID
        disease_id:   EFO/MONDO/DOID/Orphanet disease ID
        output_path:  PNG destination
        top_n:        How many top-attributed edge types to show as rows
        year_min:     X-axis lower bound (default: min evidence year across all pairs)
        year_max:     X-axis upper bound (default: max evidence year across all pairs)
        cutoff_year:  Override the per-pair cutoff. If None, infer from evidence_df.
        target_name:  Optional human-readable target label
        disease_name: Optional human-readable disease label

    Returns:
        Path to the written PNG, or None if there was nothing to plot.
    """
    top_etypes = _top_edge_types(edges_df, target_id, disease_id, top_n)
    if not top_etypes:
        return None

    pair_ev = evidence_df[
        (evidence_df["target_id"] == target_id)
        & (evidence_df["disease_id"] == disease_id)
        & evidence_df["edge_type"].isin(top_etypes)
    ].copy()

    # Fall back: if the pair has no evidence rows for any top edge type,
    # there's nothing temporal to show.
    if pair_ev.empty:
        return None

    pair_ev["year"] = pd.to_numeric(pair_ev["year"], errors="coerce")
    pair_ev = pair_ev.dropna(subset=["year"])
    pair_ev["year"] = pair_ev["year"].astype(int)

    if cutoff_year is None:
        cutoff_year = _pair_cutoff_year(evidence_df, target_id, disease_id)

    # X-axis bounds: default to the union range across the dataframe so
    # multiple pairs in the same paper are visually comparable.
    if year_min is None:
        year_min = int(evidence_df["year"].dropna().min()) if not evidence_df.empty else int(pair_ev["year"].min())
    if year_max is None:
        # +1 so the rightmost bar isn't clipped
        year_max = int(evidence_df["year"].dropna().max()) + 1 if not evidence_df.empty else int(pair_ev["year"].max()) + 1

    n_rows = len(top_etypes)
    fig, axes = plt.subplots(
        n_rows, 1,
        figsize=(8.5, 0.7 * n_rows + 1.0),
        sharex=True,
        gridspec_kw={"hspace": 0.35},
    )
    if n_rows == 1:
        axes = [axes]

    bin_edges = np.arange(year_min, year_max + 1)

    for ax, etype in zip(axes, top_etypes):
        rows = pair_ev[pair_ev["edge_type"] == etype]
        years = rows["year"].to_numpy()
        # Pre/post-cutoff masks for two-tone shading.
        if cutoff_year is not None:
            pre = years[years <= cutoff_year]
            post = years[years > cutoff_year]
        else:
            pre, post = years, np.array([], dtype=int)

        ax.hist(
            pre, bins=bin_edges,
            color="#2b5d8c", alpha=0.95,
            label="≤ cutoff (model-visible)" if ax is axes[0] else None,
        )
        if post.size:
            ax.hist(
                post, bins=bin_edges,
                color="#c4a747", alpha=0.7,
                label="> cutoff (post-hoc)" if ax is axes[0] else None,
            )

        if cutoff_year is not None:
            ax.axvline(cutoff_year + 0.5, color="black", lw=1.0, ls="--", alpha=0.7)

        # Row label: edge_type, plus the datasourceId(s) it spans.
        ds_set = sorted(set(str(d) for d in rows["datasourceId"].dropna().unique()))
        ds_str = ", ".join(ds_set) if ds_set else "—"
        label = f"{etype}\n({ds_str})"
        ax.set_ylabel(label, rotation=0, ha="right", va="center",
                      fontsize=8, labelpad=8)

        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", left=False)
        ax.tick_params(axis="x", labelsize=8)
        # Per-row count annotation, top-right.
        n_pre = pre.size
        n_post = post.size
        ax.text(
            0.995, 0.92,
            f"n={n_pre}" + (f"  (+{n_post} post)" if n_post else ""),
            transform=ax.transAxes,
            fontsize=7, ha="right", va="top", color="#444",
        )

    axes[-1].set_xlabel("Publication year", fontsize=9)
    axes[-1].set_xlim(year_min, year_max)

    title = f"Evidence accumulation: {target_name or target_id} → {disease_name or disease_id}"
    if cutoff_year is not None:
        title += f"   (cutoff: {cutoff_year})"
    fig.suptitle(title, fontsize=10, y=0.995)

    # Legend on the top subplot.
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles, labels, loc="upper left", fontsize=7,
                       frameon=False, ncol=2)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--explanations-dir", required=True,
                   help="Directory containing per_pair_edges.parquet and "
                        "per_pair_evidence.parquet")
    p.add_argument("--target-id", required=True)
    p.add_argument("--disease-id", required=True)
    p.add_argument("--output", required=True, help="Destination PNG path")
    p.add_argument("--top-n", type=int, default=6,
                   help="How many top-attributed edge types to show as rows")
    p.add_argument("--year-min", type=int, default=None)
    p.add_argument("--year-max", type=int, default=None)
    p.add_argument("--cutoff-year", type=int, default=None,
                   help="Override per-pair cutoff (default: infer from data)")
    args = p.parse_args()

    explanations_dir = Path(args.explanations_dir)
    edges_df = pd.read_parquet(explanations_dir / "per_pair_edges.parquet")
    evidence_df = pd.read_parquet(explanations_dir / "per_pair_evidence.parquet")

    out = plot_temporal_stripe(
        edges_df=edges_df,
        evidence_df=evidence_df,
        target_id=args.target_id,
        disease_id=args.disease_id,
        output_path=Path(args.output),
        top_n=args.top_n,
        year_min=args.year_min,
        year_max=args.year_max,
        cutoff_year=args.cutoff_year,
    )
    if out is None:
        print(f"[temporal_stripe] no plot generated — no evidence rows for "
              f"{args.target_id} → {args.disease_id}")
    else:
        print(f"[temporal_stripe] wrote {out}")


if __name__ == "__main__":
    main()
