import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_mean

class Classifier(nn.Module):

    def __init__(self, net: callable, feat_dim, nclass, node_sz, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.net = nn.ModuleList([
            net(feat_dim, feat_dim),
            nn.LeakyReLU(),
            net(feat_dim, nclass)
        ])
        if isinstance(self.net[0], MessagePassing):
            self.nettype = 'gnn'
        else:
            self.nettype = 'mlp'
        # self.pool = scatter_mean
        self.pool = nn.Sequential(nn.Linear(node_sz, 1), nn.LeakyReLU())
        # self.head = net(feat_dim, nclass)
    
    def forward(self, x, edge_index, batch):
        if self.nettype == 'gnn':
            x = self.net[0](x, edge_index)
            x = self.net[1](x)
            x = self.net[2](x, edge_index)
        else:
            x = self.net[0](x)
            x = self.net[1](x)
            x = self.net[2](x)
    
        x = self.pool(x.view(batch.max()+1, len(torch.where(batch==0)[0]), x.shape[-1]).transpose(-1, -2))[..., 0]
        return x