import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GENConv

class GEN(nn.Module):
    """
    Using GENConv for edge attributed message passing.
    """
    def __init__(self, input_dim_node, input_dim_edge, hid_dim, output_dim, num_layers=2, dropout=0.5):
        """
        input_dim_node: int, dimension of node features
        input_dim_edge: int, dimension of edge features
        hid_dim: int, dimension of hidden layers
        output_dim: int, dimension of output features
        num_layers: int, number of layers in the model
        dropout: float, dropout rate for regularization during training
        """
        super(GEN, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        # linear layer to make sure input dimensions match
        self.edge_encoder = nn.Linear(input_dim_edge, hid_dim)
        # node feature transformation
        self.node_encoder = nn.Linear(input_dim_node, hid_dim)

        self.mp_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.mp_layers.append(GENConv(hid_dim, hid_dim))
        
        self.lin = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.lin.append(nn.Linear(3 * hid_dim, hid_dim))
        self.lin.append(nn.Linear(3 * hid_dim, output_dim))


    def forward(self, data):
        """
        returns both node embeddings and edge logits.

        node logits have shape (num_nodes, hid_dim)
        edge logits have shape (num_edges, output_dim)
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # Encode edge features to match node feature dimensions
        edge_attr = self.edge_encoder(edge_attr)
        # Encoding node features
        x = self.node_encoder(x)

        for i, conv in enumerate(self.mp_layers):
            # update node representation
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            # update edge representation
            x_src, x_dst = x[edge_index[0]], x[edge_index[1]]
            edge_feat = torch.cat([x_src, edge_attr, x_dst], dim=-1)
            edge_attr = self.lin[i](edge_feat)
            edge_attr = F.dropout(edge_attr, p=self.dropout, training=self.training)
        
        # print shapes of x, edge_index, edge_attr
        return x, edge_attr
    
    def __repr__(self):
        total_params = sum(p.numel() for p in self.parameters())
        return f"GEN(input_dim_node={self.node_encoder.in_features}, input_dim_edge={self.edge_encoder.in_features}, hid_dim={self.mp_layers[0].in_channels}, output_dim={self.lin[-1].out_features}, num_layers={self.num_layers}, num_parameters={total_params})"
