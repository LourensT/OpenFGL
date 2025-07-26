import torch.nn as nn
from torch_geometric.nn import GINEConv, BatchNorm, Linear
import torch.nn.functional as F
import torch

class GINe(torch.nn.Module):
    def __init__(self, 
                input_dim_node, 
                input_dim_edge,
                hid_dim, 
                num_mp_layers, 
                output_dim, 
                final_dropout=0.1):
        super().__init__()

        self.n_hidden = hid_dim
        self.output_dim = output_dim
        self.num_mp_layers = num_mp_layers
        self.final_dropout = final_dropout

        self.node_emb = nn.Linear(input_dim_node, self.n_hidden)
        self.edge_emb = nn.Linear(input_dim_edge, self.n_hidden)

        # message passing layers
        self.convs = nn.ModuleList()
        # edge update layers
        self.emlps = nn.ModuleList()
        # batch normalization layers
        self.batch_norms = nn.ModuleList()
        for _ in range(self.num_mp_layers):
            # GINEConv uses a MLP as aggregation function
            conv = GINEConv(
                nn.Sequential(
                    nn.Linear(self.n_hidden, self.n_hidden), 
                    nn.ReLU(), 
                    nn.Linear(self.n_hidden, self.n_hidden)
                ),
                edge_dim=self.n_hidden
            )
            self.convs.append(conv)
            
            # edges are updated by taking the concatenation of the source node, destination node, and edge features
            self.emlps.append(
                nn.Sequential(
                    nn.Linear(3 * self.n_hidden, self.n_hidden),
                    nn.ReLU(),
                    nn.Linear(self.n_hidden, self.n_hidden),
                )
            )

            self.batch_norms.append(BatchNorm(self.n_hidden))

        # Final prediction MLP (does not use graph structure)
        self.mlp = nn.Sequential(
            Linear(self.n_hidden*3, 50), nn.ReLU(), nn.Dropout(self.final_dropout),
            Linear(50, 25), nn.ReLU(), nn.Dropout(self.final_dropout),
            Linear(25, output_dim)
        )

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        src, dst = edge_index

        x = self.node_emb(x)
        edge_attr = self.edge_emb(edge_attr)

        for i in range(self.num_mp_layers):
            # update node representation
            x = (x + F.relu(self.batch_norms[i](self.convs[i](x, edge_index, edge_attr)))) / 2
            # update edge representation
            edge_attr = edge_attr + self.emlps[i](torch.cat([x[src], x[dst], edge_attr], dim=-1)) / 2

        # concatenate src and dst node features 
        edge_repr = x[edge_index.T].reshape(-1, 2 * self.n_hidden).relu()
        # Concatenate edge feature too
        edge_repr = torch.cat((edge_repr, edge_attr.view(-1, edge_attr.shape[1])), 1)
        # TODO cant this be: edge_repr = torch.cat((edge_repr, edge_attr), 1)
        # Final prediction
        edge_logit = self.mlp(edge_repr)
        
        return x, edge_logit
    
    def __repr__(self):
        total_params = sum(p.numel() for p in self.parameters())
        return f"GINe(input_dim_node={self.node_emb.in_features}, input_dim_edge={self.edge_emb.in_features}, hid_dim={self.n_hidden}, output_dim={self.output_dim}, num_layers={self.num_mp_layers}, num_parameters={total_params})"