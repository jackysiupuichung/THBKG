"""Data module for heterogeneous graph benchmarking."""

from .graph_builder import build_hetero_graph, load_edges
from .utils import temporal_split, cold_start_split, attach_node_features

__all__ = [
    "build_hetero_graph",
    "load_edges",
    "temporal_split",
    "cold_start_split",
    "attach_node_features",
]
