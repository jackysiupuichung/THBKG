import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, GATv2Conv

class HetGATv2(nn.Module):
    def __init__(self, hetero_data, hidden_dim, num_layers, heads,
                 pair_src_type, pair_dst_type, pair_mlp_hidden, dropout=0.2):
        super().__init__()
        self.metadata = hetero_data.metadata()
        self.pair_src_type = pair_src_type
        self.pair_dst_type = pair_dst_type

        # === Trainable embeddings from hetero_data.x ===
        self.embeddings = nn.ParameterDict()
        for ntype in hetero_data.node_types:
            if hasattr(hetero_data[ntype], "x") and hetero_data[ntype].x is not None:
                init_tensor = hetero_data[ntype].x.clone().detach()
                self.embeddings[ntype] = nn.Parameter(init_tensor)  # trainable!
            else:
                num_nodes = hetero_data[ntype].num_nodes
                self.embeddings[ntype] = nn.Parameter(
                    torch.randn(num_nodes, hidden_dim)  # random init
                )

        # === Graph encoder ===
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv(
                {
                    (src, rel, dst): GATv2Conv(
                        in_channels=(self.embeddings[src].size(1), self.embeddings[dst].size(1)),
                        out_channels=hidden_dim,
                        heads=heads,
                        concat=False,
                        dropout=dropout,
                        edge_dim=1,
                        add_self_loops=False,
                    )
                    for src, rel, dst in self.metadata[1]
                },
                # refer to aggregation in message passing
                aggr="sum",
            )
            self.convs.append(conv)

        # === Pairwise head ===
        input_dim = 2 * hidden_dim
        layers = []
        for h in pair_mlp_hidden:
            layers += [nn.Linear(input_dim, h), nn.ReLU()]
            input_dim = h
        layers += [nn.Linear(input_dim, 1)]
        self.pair_mlp = nn.Sequential(*layers)
        

    def forward(self, x_dict, edge_index_dict, pairs, edge_attr_dict):
        # override x_dict with trainable parameters
        h_dict = {nt: self.embeddings[nt] for nt in self.metadata[0]}

        # pass through GAT layers
        for conv in self.convs:
            h_dict = conv(h_dict, edge_index_dict, edge_attr_dict)

        src_ids, dst_ids = pairs
        src_emb = h_dict[self.pair_src_type][src_ids]
        dst_emb = h_dict[self.pair_dst_type][dst_ids]
        return self.pair_mlp(torch.cat([src_emb, dst_emb], dim=-1)).squeeze(-1)
