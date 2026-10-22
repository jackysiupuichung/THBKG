"""Per-node-type input projections, including node types with several feature modalities.

Most node types carry one feature matrix ``x`` and get one linear projection to
the hidden size. A node type can instead carry separate modality tensors,
row-aligned with its nodes (molecules: Morgan fingerprints for small molecules,
ESM-2 embeddings for biologics):

    x_<modality>    [N, d_m] features; rows of nodes without it are zero
    has_<modality>  [N] bool, which nodes have that modality
    drug_type       [N] long, optional category (one learned embedding each)

``ModalityEncoder`` projects each modality with its own linear layer into the
shared hidden space and adds the category embedding:

    h_v = sum_m has_m(v) * W_m x_m(v) + E[drug_type(v)]

so every node of the type ends up in one space, with no concatenated feature
vector and no zero block reaching a projection. Nodes with no modality get the
category embedding alone, which is their learned missing-modality embedding.

Models receive such a node type as a dict of tensors in ``x_dict`` (built by
``node_inputs``) and project it with the module ``make_input_projection``
returns, so the rest of each model is unchanged.
"""
from __future__ import annotations

from typing import Dict, Union

import torch
from torch import nn
from torch_geometric.nn import Linear

MODALITY_PREFIX = "x_"
MASK_PREFIX = "has_"
CATEGORY_KEY = "drug_type"


def is_modality_store(store) -> bool:
    return "x" not in store and any(k.startswith(MODALITY_PREFIX) for k in store.keys())


def input_spec(store) -> Union[int, dict, None]:
    """Feature width of a node store: an int for ``x``, a dict for modality stores."""
    if "x" in store and store.x is not None:
        return int(store.x.size(1))
    if is_modality_store(store):
        if "num_nodes" not in store:
            raise ValueError("a modality node store needs an explicit num_nodes (else PyG infers it "
                             "from edge indices and samplers do not slice its tensors)")
        mods = {k[len(MODALITY_PREFIX):]: int(store[k].size(1))
                for k in store.keys() if k.startswith(MODALITY_PREFIX)}
        missing = [m for m in mods if f"{MASK_PREFIX}{m}" not in store]
        if missing:
            raise ValueError(f"modality tensors without has_ masks: {missing}")
        n_cat = int(store[CATEGORY_KEY].max()) + 1 if CATEGORY_KEY in store else 0
        return {"modalities": mods, "num_categories": n_cat}
    return None


def node_inputs(data) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
    """``x_dict`` for a (sub)graph: ``x`` per node type, or a dict of modality tensors."""
    out = {}
    for nt in data.node_types:
        store = data[nt]
        if "x" in store and store.x is not None:
            out[nt] = store.x
        elif is_modality_store(store):
            out[nt] = {k: store[k] for k in store.keys()
                       if k.startswith(MODALITY_PREFIX) or k.startswith(MASK_PREFIX) or k == CATEGORY_KEY}
    return out


class ModalityEncoder(nn.Module):
    """One linear projection per modality into a shared space, plus a category embedding."""

    def __init__(self, modalities: Dict[str, int], hidden_dim: int, num_categories: int = 0):
        super().__init__()
        self.proj = nn.ModuleDict({m: Linear(d, hidden_dim) for m, d in modalities.items()})
        self.category = nn.Embedding(num_categories, hidden_dim) if num_categories else None
        self.hidden_dim = hidden_dim

    def forward(self, inp: Dict[str, torch.Tensor]) -> torch.Tensor:
        ref = inp[f"{MODALITY_PREFIX}{next(iter(self.proj))}"]
        h = ref.new_zeros(ref.size(0), self.hidden_dim)
        for m, lin in self.proj.items():
            mask = inp[f"{MASK_PREFIX}{m}"]
            if bool(mask.any()):
                idx = mask.nonzero(as_tuple=True)[0]
                h = h.index_add(0, idx, lin(inp[f"{MODALITY_PREFIX}{m}"][idx].float()))
        if self.category is not None:
            h = h + self.category(inp[CATEGORY_KEY].long())
        return h


def make_input_projection(spec, hidden_dim: int) -> nn.Module:
    """Linear projection for an int width (-1 = lazy), ModalityEncoder for a modality spec."""
    if isinstance(spec, dict):
        return ModalityEncoder(spec["modalities"], hidden_dim, spec.get("num_categories", 0))
    return Linear(-1 if spec is None else spec, hidden_dim)
