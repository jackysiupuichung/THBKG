"""Per-pair subgraph visualization showing top-N edges by |ig_total|.

Renders a static matplotlib PNG with the queried (target, disease) at the
center, top-N most-attributed edges by absolute IG, color-coded by sign of
ig_total, sized by magnitude. Node labels show entity IDs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import yaml


_DEFAULT_SCHEMA_PATH = (Path(__file__).resolve().parents[2]
                        / "visualisation" / "colours_schema.yaml")


def _load_colour_schema(path: Optional[Path] = None) -> dict:
    """Load the shared colour palette. Falls back to an empty dict if the
    file is unavailable."""
    path = path or _DEFAULT_SCHEMA_PATH
    try:
        with open(path) as fh:
            return yaml.safe_load(fh) or {}
    except (FileNotFoundError, OSError):
        return {}


# Aliases from the relation names used in the heterograph (as they appear in
# edge_type tuples after rev_ stripping) to the keys in colours_schema.yaml.
# Unknown relations fall through to a neutral grey.
_REL_ALIASES = {
    "modulated_by":                       "modulated_by_molecule",
    "has_function_in":                    "has_function_in_go",
    "involved_in":                        "involved_in_reactome",
    "associated_with":                    "associated_with_reactome",
    "clinical_trial_Unknown/Operational": "clinical_trial_unknown_operational",
    "clinical_trial_unknown/operational": "clinical_trial_unknown_operational",
}


def _relation_colour(rel_name: str, schema: dict, default: str = "#bdbdbd") -> str:
    rels = schema.get("relation_colours", {})
    key = _REL_ALIASES.get(rel_name, rel_name)
    return rels.get(key, default)


def plot_pair_subgraph(
    edges_df: pd.DataFrame,
    target_id: str,
    disease_id: str,
    out_path: Path,
    top_k: int = 25,
    id_maps: Optional[Dict[str, Dict[int, str]]] = None,
    target_idx: Optional[int] = None,
    disease_idx: Optional[int] = None,
    name_maps: Optional[Dict[str, Dict[str, str]]] = None,
    nodes_df: Optional[pd.DataFrame] = None,
) -> None:
    """Render a subgraph of the top-K *nodes* by total attribution, plus
    every attributed edge between any two of them. Mirrors the node-ranking
    convention used in published explainer figures (e.g. "top-N nodes by
    total attribution") — gives richer cross-edges than edge-first ranking,
    so spring_layout produces a structured layout rather than a star.

    Node score = sum of |ig_total| over all incident edges (in + out),
    falling back to attention when ig_total is NaN. The queried target and
    disease are always kept regardless of their rank.

    id_maps: optional ``{node_type: {idx: external_id}}`` used to label
    nodes with human-readable IDs. If omitted, nodes are labeled by
    internal index.
    """
    df = edges_df.copy()
    # Edge-level score (|ig_total| with attention as fallback).
    score = df["ig_total"].abs()
    score = score.fillna(df["attention"].fillna(0))
    df["_rank_score"] = score

    # Strip "rev_" and flip src↔dst so reverse edges fold onto their
    # canonical forward direction *before* we rank nodes (otherwise the
    # same underlying entity appears twice — once as a src on rev_X edges
    # and once as a dst on forward X edges).
    parts = df["edge_type"].str.split("::", n=2, expand=True)
    parts.columns = ["_src_type", "_rel", "_dst_type"]
    df = pd.concat([df, parts], axis=1)

    is_rev = df["_rel"].str.startswith("rev_", na=False)
    df.loc[is_rev, "_rel"] = df.loc[is_rev, "_rel"].str.slice(4)
    df.loc[is_rev, ["src", "dst"]] = df.loc[is_rev, ["dst", "src"]].values
    df.loc[is_rev, ["_src_type", "_dst_type"]] = df.loc[is_rev, ["_dst_type", "_src_type"]].values

    df["_src_node"] = df["_src_type"] + "#" + df["src"].astype(str)
    df["_dst_node"] = df["_dst_type"] + "#" + df["dst"].astype(str)

    # Node-level score. Prefer NATIVE node-IG when nodes_df is supplied
    # (sum of |feature-dim IG| per node — what GATher-style papers report).
    # Fall back to a derived score = sum of |ig_total| over incident edges
    # when nodes_df is unavailable.
    if nodes_df is not None and len(nodes_df):
        node_score = pd.Series(dtype=float)
        for _, r in nodes_df.iterrows():
            key = f"{r['node_type']}#{int(r['node_global_idx'])}"
            node_score[key] = float(r["ig_node_abs"])
    else:
        src_contrib = df.groupby("_src_node")["_rank_score"].sum()
        dst_contrib = df.groupby("_dst_node")["_rank_score"].sum()
        node_score = src_contrib.add(dst_contrib, fill_value=0.0)

    # Always include the queried target and disease, even if they didn't
    # crack the top by attribution (the supervision edge itself is in the
    # held-out graph, so their accrued attribution is via context edges
    # only — usually they're top anyway).
    must_keep = set()
    if target_idx is not None:
        must_keep.add(f"target#{target_idx}")
    if disease_idx is not None:
        must_keep.add(f"disease#{disease_idx}")

    selected_nodes = set(node_score.sort_values(ascending=False).head(top_k).index)
    selected_nodes |= must_keep

    # Now keep every attributed edge whose endpoints are BOTH in the
    # selected node set. This is what brings in the cross-edges between
    # peer nodes that pure edge-ranking would have dropped.
    df = df[df["_src_node"].isin(selected_nodes) & df["_dst_node"].isin(selected_nodes)]

    # Use a simple DiGraph so nx.draw_networkx_edge_labels can place labels.
    # The rev_ stripping + src/dst flip happened during node ranking above,
    # so here we just consume the already-normalised columns.
    G = nx.DiGraph()
    edge_styles = []
    pair_relations: Dict[Tuple[str, str], list] = {}
    for _, r in df.iterrows():
        src_type = r["_src_type"]
        dst_type = r["_dst_type"]
        rel_name = r["_rel"]
        src_idx = int(r["src"])
        dst_idx = int(r["dst"])
        src_node = r["_src_node"]
        dst_node = r["_dst_node"]

        sign = 1 if (r.get("ig_total", 0) or 0) >= 0 else -1
        G.add_node(src_node, ntype=src_type, idx=src_idx)
        G.add_node(dst_node, ntype=dst_type, idx=dst_idx)
        if G.has_edge(src_node, dst_node):
            # Keep the entry with the larger |ig_total| and merge relation names.
            existing = G[src_node][dst_node]
            if r["_rank_score"] > existing["weight"]:
                existing["sign"] = sign
                existing["weight"] = float(r["_rank_score"])
                existing["ig_total"] = float(r["ig_total"]) if pd.notna(r["ig_total"]) else None
                existing["attention"] = float(r["attention"]) if pd.notna(r["attention"]) else None
            pair_relations[(src_node, dst_node)].append(rel_name)
        else:
            G.add_edge(src_node, dst_node, rel=rel_name, sign=sign,
                       weight=float(r["_rank_score"]),
                       ig_total=float(r["ig_total"]) if pd.notna(r["ig_total"]) else None,
                       attention=float(r["attention"]) if pd.notna(r["attention"]) else None)
            pair_relations[(src_node, dst_node)] = [rel_name]
        edge_styles.append((src_node, dst_node, sign, r["_rank_score"]))

    if G.number_of_nodes() == 0:
        return

    # Layout: spring with the queried disease pinned at the centre (and the
    # queried target nudged just to its right) so every figure has a stable,
    # disease-centric framing.
    initial_pos = {}
    fixed = []
    disease_node = f"disease#{disease_idx}" if disease_idx is not None else None
    target_node = f"target#{target_idx}" if target_idx is not None else None
    if disease_node and disease_node in G.nodes:
        initial_pos[disease_node] = (0.0, 0.0)
        fixed.append(disease_node)
    if target_node and target_node in G.nodes:
        initial_pos[target_node] = (0.25, 0.0)
        fixed.append(target_node)
    pos = nx.spring_layout(
        G, seed=42, k=0.8,
        pos=initial_pos or None,
        fixed=fixed or None,
    )

    fig, ax = plt.subplots(figsize=(11, 9))

    # Schema-driven node colours (visualisation/colours_schema.yaml).
    # Falls back to tab10 for any unknown node types so we never hard-fail.
    schema = _load_colour_schema()
    node_colour_schema = schema.get("node_colours", {})
    ntypes = sorted({d["ntype"] for _, d in G.nodes(data=True)})
    palette = plt.cm.tab10.colors
    color_map = {
        t: node_colour_schema.get(t, palette[i % len(palette)])
        for i, t in enumerate(ntypes)
    }

    def label_for(node):
        d = G.nodes[node]
        idx = d["idx"]
        accession = None
        if id_maps and d["ntype"] in id_maps and idx in id_maps[d["ntype"]]:
            accession = id_maps[d["ntype"]][idx]
        if accession is None:
            return f"{d['ntype']}:{idx}"
        if name_maps and d["ntype"] in name_maps:
            name = name_maps[d["ntype"]].get(accession)
            if name:
                return name
        return accession

    node_colors = [color_map[G.nodes[n]["ntype"]] for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                            node_size=300, alpha=0.85)

    # Edge widths from the merged-edge weights actually stored on G (so they
    # line up 1:1 with G.edges() iteration order, which the rest of the
    # draw calls follow). Edge COLOUR is schema-driven by relation type
    # (so all `genetic_association` edges share one colour, all `literature`
    # edges another, etc.). The direction of effect (positive vs negative
    # IG attribution) is encoded as linestyle — solid = pushes prediction
    # up, dashed = pushes down — so colour can carry relation semantics.
    edges_in_order = list(G.edges(data=True))
    weights = np.array([d["weight"] for *_, d in edges_in_order], dtype=float)
    if weights.size and weights.max() > 0:
        widths = 0.5 + 4 * (weights / weights.max())
    else:
        widths = np.full(len(edges_in_order), 1.0)
    edge_colors = [_relation_colour(d["rel"], schema) for *_, d in edges_in_order]
    # Unsigned display: thickness encodes |IG|, sign is intentionally omitted
    # so casual readers don't over-interpret low-magnitude dashed edges (which
    # are mostly path-integration noise on near-zero edge features).
    edge_styles_per_edge = ["solid" for _ in edges_in_order]
    nx.draw_networkx_edges(G, pos, ax=ax, width=list(widths),
                            edge_color=edge_colors,
                            style=edge_styles_per_edge,
                            alpha=0.7,
                            arrows=True, arrowsize=10,
                            connectionstyle="arc3,rad=0.05")

    labels = {n: label_for(n) for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=7)

    # Edge labels intentionally omitted: relation type is encoded by
    # edge colour from colours_schema.yaml, and the legend below
    # explicitly lists every relation present in this figure.

    # ── Highlight the queried (target, disease) pair ────────────────────
    # The query nodes may or may not appear in the top-K subgraph (the
    # supervision edge itself isn't in the message-passing graph, but the
    # endpoints almost always are via reverse advancement / evidence edges).
    query_nodes = []
    if target_idx is not None:
        query_nodes.append((f"target#{target_idx}", target_id, "target"))
    if disease_idx is not None:
        query_nodes.append((f"disease#{disease_idx}", disease_id, "disease"))

    present_query = [(node, ext, nt) for node, ext, nt in query_nodes
                     if node in G.nodes]

    def _display(ext, nt):
        if name_maps and nt in name_maps:
            nm = name_maps[nt].get(ext)
            if nm:
                return nm
        return ext

    if present_query:
        # Gold outline around the queried nodes (visual anchor only).
        # We do NOT redraw the labels — they keep the standard
        # black/small styling from the main label pass above.
        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            nodelist=[n for n, _, _ in present_query],
            node_size=700, node_color="none",
            edgecolors="#ffb000", linewidths=3.0,
        )

    # Dashed gold arrow showing the predicted edge, even if it's not in
    # the sampled subgraph (it's the *held-out* supervision edge).
    if len(present_query) == 2:
        (src_node, _, _), (dst_node, _, _) = present_query
        src_xy = pos[src_node]
        dst_xy = pos[dst_node]
        ax.annotate(
            "",
            xy=dst_xy, xytext=src_xy,
            arrowprops=dict(arrowstyle="-|>", color="#ffb000",
                             linestyle="--", linewidth=2.5,
                             connectionstyle="arc3,rad=-0.25",
                             shrinkA=18, shrinkB=18),
            zorder=10,
        )

    handles = [plt.Line2D([0], [0], marker="o", color="w",
                           markerfacecolor=color_map[t], markersize=10, label=t)
               for t in ntypes]
    # One entry per relation actually present in this figure, coloured per
    # the shared schema. Sorted by display name for stability.
    rels_present = sorted({d["rel"] for *_, d in edges_in_order})
    handles += [plt.Line2D([0], [0],
                           color=_relation_colour(r, schema), lw=2.5, label=r)
                for r in rels_present]
    handles += [
        plt.Line2D([0], [0], color="black", lw=2, linestyle="solid",
                   label="positive IG (pushes prediction up)"),
        plt.Line2D([0], [0], color="black", lw=2, linestyle="dashed",
                   label="negative IG (pushes down)"),
        plt.Line2D([0], [0], color="#ffb000", lw=2.5, linestyle="--",
                   label="queried advancement edge"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=7,
              ncol=2, framealpha=0.85)
    # Title uses human-readable names when available (falls back to
    # accessions). Format mirrors the reference paper: "TARGET–DISEASE".
    target_disp = (name_maps.get("target", {}).get(target_id)
                   if name_maps else None) or target_id
    disease_disp = (name_maps.get("disease", {}).get(disease_id)
                    if name_maps else None) or disease_id
    ax.set_title(f"{target_disp} — {disease_disp}", fontsize=12,
                  fontweight="bold")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
