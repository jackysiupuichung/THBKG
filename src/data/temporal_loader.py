#!/usr/bin/env python3
"""
Temporal graph loader utilities for event-based graphs.

Handles loading of event-based HeteroData and creating temporal masks.
"""

import torch
import pandas as pd
from torch_geometric.data import HeteroData
from typing import Dict, Tuple, Optional, List, Union
from pathlib import Path
from torch_geometric.utils import coalesce
import torch_geometric.transforms as T


def load_event_graph(
    filepath: str,
    to_undirected: bool = True,
    normalize_features: bool = False,
) -> HeteroData:
    """
    Load event-based temporal graph (HeteroData).
    
    Note: Features should be attached separately using src/pipeline/attach_features.py
    
    Args:
        filepath: Path to temporal graph file (.pt)
        to_undirected: Whether to add reverse edges for message passing
        normalize_features: Whether to apply StandardScaler to node features
        
    Returns:
        HeteroData object with edge_time and edge_weight attributes
    """
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Temporal graph file not found: {filepath}")
    
    # Load HeteroData object
    data = torch.load(filepath, weights_only=False)
    
    if not isinstance(data, HeteroData):
        raise TypeError(f"Expected HeteroData, got {type(data)}")
    
    # 1. Feature Normalization (if requested)
    if normalize_features:
        print("   Normalizing features...")
        data = T.NormalizeFeatures()(data)

    # 2. Convert to undirected for GNN message passing
    if to_undirected:
        print("🔄 Converting to undirected graph (adding reverse edges)...")
        data = T.ToUndirected()(data)
    
    # Remove node_id attribute if present to avoid PyG loader errors
    # (PyG loader tries to slice all attributes, and list[str] fails)
    for node_type in data.node_types:
        if hasattr(data[node_type], 'node_id'):
            del data[node_type].node_id
            
    return data


def is_clinical_trial_edge(edge_type: Tuple[str, str, str]) -> bool:
    """
    Check if edge type is a clinical trial edge (to exclude from context).
    
    Args:
        edge_type: Tuple of (src_type, relation, dst_type)
        
    Returns:
        True if edge is a clinical trial edge
    """
    CLINICAL_TRIAL_KEYWORDS = ['clinical_trial']
    return any(kw in edge_type[1] for kw in CLINICAL_TRIAL_KEYWORDS)


def build_num_neighbors(
    data: HeteroData,
    base: List[int],
    strategy: str = "off",
    overrides: Optional[Dict[str, List[int]]] = None,
    cap_relations: Optional[List[str]] = None,
    boost_relations: Optional[List[str]] = None,
    cap_value: int = 2,
    boost_value: int = 40,
) -> Union[List[int], Dict[Tuple[str, str, str], List[int]]]:
    """Build a per-edge-type ``num_neighbors`` budget for LinkNeighborLoader.

    The default sampler draws ``base`` (e.g. [20, 10]) neighbors per hop blind to
    edge type, so frequency-dominant relations (literature ~73% of target-disease
    edges) crowd out rare-but-predictive ones (clinical_trial_*, genetic). PyG
    2.6.x accepts ``num_neighbors`` as a ``Dict[EdgeType, List[int]]`` to give each
    edge type its own budget — this builds that dict (no custom sampler needed).

    PyG requires EVERY edge type present in the dict, so we start every edge type
    at ``base`` and then adjust per strategy.

    Args:
        data: the context HeteroData (its ``.edge_types`` define the keys).
        base: per-hop budget applied to every edge type by default, e.g. [20, 10].
        strategy:
            ``"off"``      -> return ``base`` unchanged (flat list; no behaviour
                              change — the A/B control).
            ``"manual"``   -> apply ``overrides`` on top of ``base``.
            ``"boosted"``  -> cap ``cap_relations`` to ``cap_value`` per hop and
                              give ``boost_relations`` ``boost_value`` per hop
                              (a generous FINITE budget, not -1: an unbounded -1
                              on a high-cardinality type like genetic_association
                              (~1.7M edges) blows up the subgraph and OOMs).
            ``"equalized"``-> per hop, split that hop's budget evenly across the
                              edge types incident to each destination node type.
        overrides: relation-name (middle tuple element) -> per-hop list. Matched by
            substring so ``"clinical_trial"`` covers all clinical_trial_* relations
            and their ``rev_`` reverses.
        cap_relations / boost_relations: relation-name substrings for "boosted".
        cap_value: per-hop budget assigned to capped relations.

    Returns:
        ``base`` (list) when strategy is "off", else a ``Dict[EdgeType, List[int]]``
        covering all edge types.
    """
    if strategy == "off":
        return list(base)

    edge_types = list(data.edge_types)
    n_hops = len(base)

    def _matches(rel: str, patterns: List[str]) -> bool:
        return any(p in rel for p in (patterns or []))

    budget: Dict[Tuple[str, str, str], List[int]] = {
        et: list(base) for et in edge_types
    }

    if strategy == "manual":
        for et in edge_types:
            for rel_pat, hops in (overrides or {}).items():
                if rel_pat in et[1]:
                    budget[et] = list(hops)

    elif strategy == "boosted":
        for et in edge_types:
            rel = et[1]
            if _matches(rel, cap_relations):
                budget[et] = [cap_value] * n_hops
            elif _matches(rel, boost_relations):
                budget[et] = [boost_value] * n_hops
            # explicit per-relation overrides win over cap/boost
            for rel_pat, hops in (overrides or {}).items():
                if rel_pat in rel:
                    budget[et] = list(hops)

    elif strategy == "equalized":
        # group edge types by destination node type; each hop's budget is split
        # evenly across the relations feeding that destination type.
        from collections import defaultdict
        by_dst = defaultdict(list)
        for et in edge_types:
            by_dst[et[2]].append(et)
        for dst, ets in by_dst.items():
            for hop in range(n_hops):
                share = max(base[hop] // max(len(ets), 1), 1)
                for et in ets:
                    budget[et][hop] = share

    else:
        raise ValueError(f"unknown num_neighbors strategy: {strategy!r}")

    return budget


def remove_clinical_trial_edges(data: HeteroData) -> HeteroData:
    """
    Remove all clinical trial edges from graph.
    
    Clinical trial edges are supervision targets for downstream tasks,
    so they should not be part of the encoder's context to prevent leakage.
    
    Args:
        data: HeteroData object
        
    Returns:
        New HeteroData object without clinical trial edges
    """
    new_data = data.clone()
    
    to_remove = [et for et in new_data.edge_types if is_clinical_trial_edge(et)]
    
    if to_remove:
        print(f"   Removing {len(to_remove)} clinical trial edge types from context:")
        for et in to_remove:
            print(f"      ❌ {et}")
            del new_data[et]
    
    return new_data


def filter_graph_by_time(data: HeteroData, year: int) -> HeteroData:
    """
    Filter graph to include only edges up to a specific year.
    Creates a temporal cut off view of the graph.
    
    Args:
        data: HeteroData object with edge_time
        year: Max year to include (inclusive)
        
    Returns:
        New HeteroData object with filtered edges
    """
    new_data = data.clone()
    
    for et in new_data.edge_types:
        if 'edge_time' in new_data[et]:
            edge_time = new_data[et].edge_time
            mask = edge_time <= year
            
            # Filter edge_index
            new_data[et].edge_index = new_data[et].edge_index[:, mask]
            
            # Filter attributes
            for key in ['edge_time', 'edge_weight', 'edge_attr']:
                if key in new_data[et]:
                    new_data[et][key] = new_data[et][key][mask]
        else:
            # Keep static edges as is
            pass
            
    return new_data


def get_temporal_masks(
    data: HeteroData,
    split_config: Dict[str, List[int]]
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Create train/val/test masks based on edge_time using split configuration.
    
    Args:
        data: HeteroData object with edge_time
        split_config: Dict with 'train', 'val', 'test' keys containing [start, end] lists.
                     Example: {'train': [1990, 2015], 'val': [2016, 2017], 'test': [2018, 2024]}
        
    Returns:
        Dictionary mapping edge_type -> (train_mask, val_mask, test_mask)
        
    Raises:
        ValueError: If split_config is None or missing required keys
    """
    if split_config is None:
        raise ValueError(
            "split_config is required. Must provide a dict with 'train', 'val', 'test' keys.\n"
            "Example: {'train': [1990, 2015], 'val': [2016, 2017], 'test': [2018, 2024]}"
        )
    
    # Validate split_config has required keys
    required_keys = {'train', 'val', 'test'}
    missing_keys = required_keys - set(split_config.keys())
    if missing_keys:
        raise ValueError(
            f"split_config missing required keys: {missing_keys}. "
            f"Must provide 'train', 'val', and 'test' ranges."
        )
    
    masks = {}
    
    # Helper to check if times fall within range
    def is_in_range(times, rng):
        start, end = rng
        mask = torch.ones(times.size(0), dtype=torch.bool, device=times.device)
        if start is not None:
            mask &= (times >= int(start))
        if end is not None:
            mask &= (times <= int(end))
        return mask

    for edge_type in data.edge_types:
        if 'edge_time' not in data[edge_type]:
            # If no time, assume context (all train)
            num_edges = data[edge_type].edge_index.size(1)
            train_mask = torch.ones(num_edges, dtype=torch.bool)
            val_mask = torch.zeros(num_edges, dtype=torch.bool)
            test_mask = torch.zeros(num_edges, dtype=torch.bool)
        else:
            edge_time = data[edge_type].edge_time
            
            # Get ranges from split_config (handles both dict and OmegaConf)
            tr_range = split_config.get('train') or split_config['train']
            val_range = split_config.get('val') or split_config['val']
            test_range = split_config.get('test') or split_config['test']
            
            train_mask = is_in_range(edge_time, tr_range)
            val_mask = is_in_range(edge_time, val_range)
            test_mask = is_in_range(edge_time, test_range)
            
        masks[edge_type] = (train_mask, val_mask, test_mask)
        
    return masks


def print_temporal_summary(data: HeteroData):
    """
    Print summary of temporal graph events.
    
    Args:
        data: HeteroData object
    """
    print(f"\n📊 Temporal Graph Summary")
    print(f"{'='*80}")
    
    print(f"Nodes:")
    for nt in data.node_types:
        print(f"   {nt}: {data[nt].num_nodes:,}")
        
    print(f"\nEdges:")
    for et in data.edge_types:
        num_edges = data[et].edge_index.size(1)
        has_time = 'edge_time' in data[et]
        has_weight = 'edge_weight' in data[et] or 'edge_attr' in data[et]
        
        info = []
        if has_time: 
            min_t = int(data[et].edge_time.min())
            max_t = int(data[et].edge_time.max())
            info.append(f"Time: {min_t}-{max_t}")
        if has_weight: 
            info.append("Weighted")
            
        print(f"   {et}: {num_edges:,} {' | '.join(info)}")


def to_time_agnostic(data: HeteroData) -> HeteroData:
    """
    Collapse temporal graph into a static time-agnostic graph.
    
    Aggregates multiple edges between the same (src, dst) pair into a single edge.
    Aggregation method: 'max' for edge weights/attributes.
    Removes 'edge_time' attribute.
    
    Args:
        data: HeteroData object (temporal)
        
    Returns:
        New HeteroData object (static)
    """
    new_data = data.clone()
    
    print(f"\nTime-Agnostic Collapsing:")
    
    for et in new_data.edge_types:
        edge_index = new_data[et].edge_index
        num_edges_before = edge_index.size(1)
        
        # Gather attributes to aggregate
        # We assume 'edge_weight' or 'edge_attr' are the scores to max.
        # If both exist, we need to handle them. Coalesce handles one 'edge_attr'.
        # If we have multiple, we might need multiple passes or stack them?
        # Typically HGT uses 'edge_attr' or 'edge_weight'.
        
        edge_attr = None
        if 'edge_weight' in new_data[et]:
            edge_attr = new_data[et].edge_weight
            if edge_attr.dim() == 1: edge_attr = edge_attr.view(-1, 1) # Make sure it's [N, 1]
        elif 'edge_attr' in new_data[et]:
            edge_attr = new_data[et].edge_attr
        
        has_time = 'edge_time' in new_data[et]
        time_attr = new_data[et].edge_time.float().unsqueeze(-1) if has_time else None  # [E, 1]

        if edge_attr is not None:
            # Stack time alongside score so both are coalesced in one pass
            if time_attr is not None:
                combined = torch.cat([edge_attr, time_attr], dim=-1)  # [E, score_dim+1]
                new_index, new_combined = coalesce(edge_index, combined, reduce='max')
                new_attr = new_combined[:, :-1]
                new_time = new_combined[:, -1].long()
            else:
                new_index, new_attr = coalesce(edge_index, edge_attr, reduce='max')
                new_time = None

            new_data[et].edge_index = new_index

            # Restore attribute name
            if 'edge_weight' in new_data[et]:
                new_data[et].edge_weight = new_attr.squeeze()  # [N] or [N, 1]
            elif 'edge_attr' in new_data[et]:
                new_data[et].edge_attr = new_attr

        else:
            if time_attr is not None:
                new_index, new_time_combined = coalesce(edge_index, time_attr, reduce='max')
                new_time = new_time_combined[:, 0].long()
            else:
                new_index = coalesce(edge_index, None)
                new_time = None
            new_data[et].edge_index = new_index

        # Update or remove edge_time
        if has_time:
            if new_time is not None:
                new_data[et].edge_time = new_time
            else:
                del new_data[et].edge_time
            
        num_edges_after = new_data[et].edge_index.size(1)
        print(f"   {et}: {num_edges_before:,} -> {num_edges_after:,} edges (Max Aggregation)")
        
    return new_data


def to_temporal_snapshots(
    data: HeteroData,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    verbose: bool = True
) -> Dict[int, HeteroData]:
    """
    Materialize yearly snapshots of the graph.
    
    For each year y, creates a static graph containing the max score of edges
    observed up to year y (cumulative).
    
    Args:
        data: HeteroData object (temporal)
        start_year: Start year (inclusive). Defaults to min edge time.
        end_year: End year (inclusive). Defaults to max edge time.
        verbose: Print progress
        
    Returns:
        Dictionary {year: static_hetero_data}
    """
    if verbose:
        print("\n📸 Materializing Temporal Snapshots...")
        
    # Determine range
    all_times = []
    for et in data.edge_types:
        if 'edge_time' in data[et]:
            all_times.append(data[et].edge_time)
            
    if not all_times:
        print("⚠️ No temporal information found. Returning single snapshot.")
        return {0: to_time_agnostic(data)}
        
    all_times = torch.cat(all_times)
    min_t = int(all_times.min().item())
    max_t = int(all_times.max().item())
    
    if start_year is None: start_year = min_t
    if end_year is None: end_year = max_t
    
    snapshots = {}
    
    for year in range(start_year, end_year + 1):
        if verbose: print(f"\n🗓️  Year: {year}")
        
        # 1. Filter (Cumulative <= year)
        # Note: filter_graph_by_time follows cumulative logic
        snapshot_temporal = filter_graph_by_time(data, year)
        
        # 2. Collapse to static
        # This keeps the MAX score for duplicate edges
        snapshot_static = to_time_agnostic(snapshot_temporal)
        
        snapshots[year] = snapshot_static
        
    if verbose: print(f"\n✅ Created {len(snapshots)} snapshots ({start_year}-{end_year})")
    return snapshots


# ── Literature traceability ───────────────────────────────────────────────────

def load_literature_index(graph_path: str) -> pd.DataFrame:
    """
    Load the evidence-literature sidecar parquet built alongside a graph file.

    The sidecar is expected at <graph_path with .pt replaced by _literature.parquet>,
    e.g. temporal_graph_datasource_literature.parquet next to temporal_graph_datasource.pt.
    Each row is one (sourceId, targetId, datasourceId, year, pmid, evidence_id) tuple.

    Args:
        graph_path: Path to the .pt graph file

    Returns:
        DataFrame with columns: sourceId, targetId, datasourceId, year, pmid, evidence_id
    """
    sidecar = graph_path.replace(".pt", "_literature.parquet")
    if not Path(sidecar).exists():
        raise FileNotFoundError(
            f"Literature sidecar not found: {sidecar}\n"
            "Re-run build_event_graph.py with --raw-edges <RAW_EDGES_DIR> to generate it."
        )
    return pd.read_parquet(sidecar)


def get_edge_literature(
    lit_index: pd.DataFrame,
    source_id: str,
    target_id: str,
    datasource_id: str = None,
    year: int = None,
    year_max: int = None,
) -> pd.DataFrame:
    """
    Return PMIDs and OT evidence IDs supporting a given edge (or subgraph of edges).

    The lookup key is (sourceId, targetId, datasourceId, year) — all four uniquely
    identify an evidence stream since the same node pair can appear under multiple
    datasources and in multiple years.

    Args:
        lit_index:      DataFrame from load_literature_index()
        source_id:      Source node ID (e.g. Ensembl gene ID)
        target_id:      Target node ID (e.g. EFO disease ID)
        datasource_id:  Filter to one datasource (None = all datasources)
        year:           Filter to an exact year (None = all years)
        year_max:       Cumulative cutoff — return rows where row.year <= year_max
                        (mirrors filter_graph_by_time; takes precedence over `year`)

    Returns:
        DataFrame subset with columns: sourceId, targetId, datasourceId, year, pmid, evidence_id

    Subgraph use:
        results = pd.concat([
            get_edge_literature(lit, src, dst, datasource_id=ds, year_max=cutoff)
            for (src, dst, ds) in edge_list
        ]).drop_duplicates()
    """
    mask = (lit_index['sourceId'] == source_id) & (lit_index['targetId'] == target_id)
    if datasource_id is not None:
        mask &= lit_index['datasourceId'] == datasource_id
    if year_max is not None:
        mask &= lit_index['year'] <= year_max
    elif year is not None:
        mask &= lit_index['year'] == year
    return lit_index[mask].reset_index(drop=True)


def format_literature_links(
    result: pd.DataFrame,
    include_ot: bool = True,
) -> pd.DataFrame:
    """
    Add human-readable URL columns to a get_edge_literature() result.

    Args:
        result:     DataFrame returned by get_edge_literature()
        include_ot: Whether to add an OpenTargets evidence link column

    Returns:
        result with additional columns: pubmed_url (and optionally ot_url)
    """
    out = result.copy()
    out['pubmed_url'] = out['pmid'].apply(
        lambda p: f"https://pubmed.ncbi.nlm.nih.gov/{p}/" if pd.notna(p) else None
    )
    if include_ot:
        out['ot_url'] = out.apply(
            lambda r: (
                f"https://platform.opentargets.org/evidence/{r['sourceId']}/{r['targetId']}"
                if pd.notna(r['sourceId']) and pd.notna(r['targetId']) else None
            ),
            axis=1,
        )
    return out


# ---------------------------------------------------------------------------
# Advancement task helpers
#
# These used to be duplicated across the trainers and the explainer. Kept
# here so there is one source of truth for "what counts as the supervision
# edge", "which edges leak the label", and how to build the context graph.
# ---------------------------------------------------------------------------

ADV_ETYPE = ("target", "advancement", "disease")
REV_ADV_ETYPE = ("disease", "rev_advancement", "target")
TRAIN_YEAR_MAX = 2015   # transition years from train_dataset.csv
TEST_YEAR_MIN = 2016    # transition years from test_dataset.csv


def split_advancement_edges(data, cutoff_year=2010, val_min_year=None, val_max_year=None,
                            random_val_frac=None, random_seed=42):
    """Chronological split of advancement edges. Default (cutoff_year=2010)
    matches the canonical setup: train = edge_time ≤ 2010, val = 2011..2015,
    test = ≥ 2016.

    Passing explicit ``val_min_year`` / ``val_max_year`` restricts val to the
    inclusive year range — used by the val-window experiments.

    ``random_val_frac`` (e.g. 0.2): instead of a temporal train/val split, pool
    ALL pre-test edges (edge_time ≤ TRAIN_YEAR_MAX = 2015) and randomly assign
    ``random_val_frac`` of them to val, the rest to train (seeded by
    ``random_seed``). TEST stays strictly temporal (edge_time ≥ 2016) so the
    forecasting evaluation is unchanged — only model SELECTION uses the random
    val. NOTE: random val measures interpolation, not forecasting, so it is an
    optimistic selection proxy and may correlate less with the temporal test.

    Returns: train_mask, val_mask, test_mask, cutoff_year
    """
    import torch
    edge_time = data[ADV_ETYPE].edge_time

    train_year_mask = edge_time <= TRAIN_YEAR_MAX
    test_year_mask = edge_time >= TEST_YEAR_MIN

    if random_val_frac is not None:
        # Random 1-frac / frac split over ALL pre-test edges; test untouched.
        g = torch.Generator().manual_seed(int(random_seed))
        pool_idx = torch.nonzero(train_year_mask, as_tuple=False).flatten()
        perm = pool_idx[torch.randperm(pool_idx.numel(), generator=g)]
        n_val = int(round(float(random_val_frac) * perm.numel()))
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]
        train_mask = torch.zeros_like(train_year_mask)
        val_mask = torch.zeros_like(train_year_mask)
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask = test_year_mask
        return train_mask, val_mask, test_mask, cutoff_year

    train_mask = train_year_mask & (edge_time <= cutoff_year)
    if val_min_year is not None or val_max_year is not None:
        lo = int(val_min_year) if val_min_year is not None else (cutoff_year + 1)
        hi = int(val_max_year) if val_max_year is not None else TRAIN_YEAR_MAX
        val_mask = (edge_time >= lo) & (edge_time <= hi)
    else:
        val_mask = train_year_mask & (edge_time > cutoff_year)
    test_mask = test_year_mask

    return train_mask, val_mask, test_mask, cutoff_year


def build_context_graph(data, collapse: bool = False):
    """Remove advancement edges (BOTH directions) from the graph.

    Both the forward ``('target','advancement','disease')`` and the reverse
    ``('disease','rev_advancement','target')`` edge types are excluded.

    The reverse type is added by ``load_event_graph(to_undirected=True)``
    for symmetric message passing. If only the forward tuple were dropped,
    the queried (target, disease) supervision edge would still be visible
    to the model through its reverse mirror — every positive label has an
    exact rev_advancement edge with the same edge_time, which passes the
    ``LinkNeighborLoader`` temporal filter (``edge_time <= edge_label_time``).
    That was a label-leak in earlier training runs; dropping the rev tuple
    here closes it. Negatives are unaffected since no forward (and thus
    no reverse) edge exists for them.
    """
    from torch_geometric.data import HeteroData

    context = HeteroData()
    for node_type in data.node_types:
        for key, val in data[node_type].items():
            context[node_type][key] = val
    for edge_type in data.edge_types:
        if edge_type == ADV_ETYPE or edge_type == REV_ADV_ETYPE:
            continue
        for key, val in data[edge_type].items():
            context[edge_type][key] = val

    if collapse:
        context = to_time_agnostic(context)

    return context


def build_edge_time_dict(batch, exclude_etype=ADV_ETYPE):
    """Build edge_time_dict for RTE, covering all context edge types.

    Edge types with no edge_time get zeros so the RTE validator
    (which requires every edge type in edge_index_dict to be present)
    does not raise.
    """
    import torch
    result = {}
    for et in batch.edge_types:
        if et == exclude_etype:
            continue
        store = batch[et]
        n = store.edge_index.size(1)
        if hasattr(store, "edge_time") and store.edge_time is not None:
            result[et] = store.edge_time
        else:
            result[et] = torch.zeros(n, dtype=torch.long, device=store.edge_index.device)
    return result if result else None
