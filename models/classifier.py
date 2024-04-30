import torch.nn as nn
from torch_geometric.nn import MessagePassing

class Classifier(nn.Module):

    def __init__(self, net: callable, feat_dim, nclass, *args, **kwargs) -> None:
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
    
    def forward(self, x, edge_index):
        if self.nettype == 'gnn':
            x = self.net[0](x, edge_index)
            x = self.net[1](x)
            x = self.net[2](x, edge_index)
            return x
        else:
            x = self.net[0](x)
            x = self.net[1](x)
            x = self.net[2](x)
            return self.net(x)