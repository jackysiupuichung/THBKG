import os
import torch

from src.models.ncf import NCF
# from src.models.temporal_gat import TemporalGAT
# from src.models.temporal_transformer import TemporalTransformer


import os
import torch

from src.models.ncf import NCF
# from src.models.temporal_gat import TemporalGAT
# from src.models.temporal_transformer import TemporalTransformer


def initialise_model(cfg, user_map, item_map, hetero_data=None, pretrained_embeddings=None):
    """
    Initialise the recommender model.

    Args:
        cfg: config object (YAML via OmegaConf/Hydra)
        user_map: dict {user_id -> index}, built from ALL disease nodes
        item_map: dict {item_id -> index}, built from ALL target nodes
        hetero_data: PyG HeteroData (for graph models)
        pretrained_embeddings: dict with optional "user" and "item" embeddings (torch.Tensor)

    Returns:
        model (torch.nn.Module)
    """

    model_name = cfg.model.name.lower()

    # --------------------
    # Classic Neural CF
    # --------------------
    if model_name == "ncf":
        num_users = len(user_map)
        num_items = len(item_map)

        model = NCF(
            num_users=num_users,
            num_items=num_items,
            embed_dim=cfg.model.embed_dim,
            user_emb=pretrained_embeddings.get("user") if pretrained_embeddings else None,
            item_emb=pretrained_embeddings.get("item") if pretrained_embeddings else None,
        )
        return model

    # --------------------
    # Graph-based models
    # --------------------
    elif model_name in ["graph", "temporal_gat", "temporal_transformer"]:
        if not cfg.data.graph or not os.path.exists(cfg.data.graph):
            raise ValueError("Graph model requires a valid graph path in config")
        if hetero_data is None:
            hetero_data = torch.load(cfg.data.graph)
        print(f"✅ Loaded graph object from {cfg.data.graph}")

        if model_name == "temporal_gat":
            # Placeholder: build embeddings for all node types
            raise NotImplementedError("TemporalGAT integration needed")

        elif model_name == "temporal_transformer":
            # Placeholder: build embeddings for all node types
            raise NotImplementedError("Temporal Transformer integration needed")

        else:
            raise NotImplementedError("Generic Graph model placeholder")

    else:
        raise ValueError(f"❌ Unknown model: {cfg.model.name}")
    
def collate_variable(batch):
        collated = {}
        for key in batch[0]:
            if key == "neg_items":
                # keep as list of tensors
                collated[key] = [d[key] for d in batch]
            else:
                collated[key] = torch.stack([d[key] for d in batch])
        return collated
