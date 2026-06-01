"""Capture HGT post-softmax attention weights for explainability.

HGTConv.message() computes per-edge softmax attention over the concatenated
bipartite edge_index built by construct_bipartite_edge_index. That tensor is
not exposed in the public API, so we monkey-patch each conv's bound message()
to also stash the attention on the module. After a single forward pass on the
sampled subgraph, we slice the [E_total, heads] tensor back to per-edge-type
[E_t] vectors using the same edge_index_dict iteration order.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, List, Tuple

import torch

EdgeType = Tuple[str, str, str]


def _patch_conv_to_record_alpha(conv) -> Tuple[callable, callable]:
    """Wrap ``conv.message`` so it stores post-softmax alpha on the module
    each call. Returns ``(original_message, original_forward)`` for restoration.

    We also wrap ``forward`` so we can snapshot ``edge_index_dict.keys()`` order
    used in this particular call — needed later to slice alpha per edge type.
    """
    from torch_geometric.utils import softmax as _softmax
    import math

    original_message = conv.message
    original_forward = conv.forward

    def patched_message(k_j, q_i, v_j, edge_attr, index, ptr,
                        temporal_features, ef_scalar, size_i):
        # Recompute exactly what conv.message does, but record alpha.
        if temporal_features is not None:
            k_j = k_j + temporal_features
            v_j = v_j + temporal_features
        alpha = (q_i * k_j).sum(dim=-1) * edge_attr
        if ef_scalar is not None:
            alpha = alpha * ef_scalar
        alpha = alpha / math.sqrt(q_i.size(-1))
        alpha = _softmax(alpha, index, ptr, size_i)
        # Stash a detached copy so autograd graph isn't held.
        conv._captured_alpha = alpha.detach()
        out = v_j * alpha.view(-1, conv.heads, 1)
        return out.view(-1, conv.out_channels)

    def patched_forward(x_dict, edge_index_dict, edge_time_diff_dict=None,
                         edge_feat_dict=None):
        # Snapshot edge type order used by this call.
        conv._captured_edge_type_order = list(edge_index_dict.keys())
        conv._captured_edge_type_sizes = [edge_index_dict[et].size(1)
                                          for et in conv._captured_edge_type_order]
        return original_forward(x_dict, edge_index_dict,
                                edge_time_diff_dict=edge_time_diff_dict,
                                edge_feat_dict=edge_feat_dict)

    conv.message = patched_message
    conv.forward = patched_forward
    return original_message, original_forward


@contextmanager
def capture_attention(model):
    """Context manager: patch every HGTConv inside ``model`` to capture
    post-softmax attention. On exit, restore the original methods.

    Inside the context, run a single forward pass with the queried batch,
    then call ``read_attention(model)`` to extract per-edge-type attention.
    """
    convs = []
    originals = []
    for m in model.modules():
        if m.__class__.__name__ == "HGTConv":
            orig_msg, orig_fwd = _patch_conv_to_record_alpha(m)
            convs.append(m)
            originals.append((orig_msg, orig_fwd))
    try:
        yield convs
    finally:
        for m, (orig_msg, orig_fwd) in zip(convs, originals):
            m.message = orig_msg
            m.forward = orig_fwd
            for attr in ("_captured_alpha", "_captured_edge_type_order",
                          "_captured_edge_type_sizes"):
                if hasattr(m, attr):
                    delattr(m, attr)


def read_attention(convs: List) -> Dict[EdgeType, torch.Tensor]:
    """Aggregate captured alpha across all HGT layers and heads into one
    per-edge-type [E_t] tensor (mean across heads, mean across layers)."""
    if not convs:
        return {}

    # All convs see the same edge_index_dict during a single forward, so use
    # the first conv's order/sizes as canonical.
    etype_order = convs[0]._captured_edge_type_order
    sizes = convs[0]._captured_edge_type_sizes

    per_layer_per_et: List[Dict[EdgeType, torch.Tensor]] = []
    for conv in convs:
        alpha = conv._captured_alpha  # [E_total, heads]
        alpha = alpha.mean(dim=-1).cpu()  # [E_total]
        # Slice back into per-edge-type chunks in the same order.
        per_et: Dict[EdgeType, torch.Tensor] = {}
        offset = 0
        for et, n in zip(etype_order, sizes):
            per_et[et] = alpha[offset:offset + n]
            offset += n
        per_layer_per_et.append(per_et)

    # Mean across layers
    out: Dict[EdgeType, torch.Tensor] = {}
    for et in etype_order:
        stacked = torch.stack([d[et] for d in per_layer_per_et], dim=0)
        out[et] = stacked.mean(dim=0)
    return out
