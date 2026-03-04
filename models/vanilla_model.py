import torch
from torch import nn

from typing import Optional, Tuple, Union

from torch_geometric.nn import GCNConv, SAGEConv, SGConv, MessagePassing

import torch.nn.functional as F

class Transformer(nn.Module):

    def __init__(self, 
        nlayer: int = 1,
        node_sz: int=116,
        in_channel: Union[int, Tuple[int, int]] = 10,
        out_channel: int = 10,
        dropout: float = 0.1,
        hiddim: int = 1024,
        *args, **kwargs) -> None:
        super().__init__()
        
        heads: int = 2 if in_channel % 2 == 0 else 3
        self.nlayer = nlayer
        self.node_sz = node_sz

        self.lin_first = nn.Sequential(
            nn.Linear(in_channel, in_channel), 
            nn.BatchNorm1d(in_channel), 
            nn.LeakyReLU(),
        )
        self.lin_in = nn.Sequential(
            nn.Linear(in_channel, out_channel), 
            nn.BatchNorm1d(out_channel), 
            nn.LeakyReLU(),
        )
        self.net = nn.ModuleList([torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(d_model=in_channel, nhead=heads, dim_feedforward=hiddim, dropout=dropout, batch_first=True),
            num_layers=1
        ) for _ in range(nlayer)])
        self.heads = heads
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, data):
        node_feature = data.x
        node_feature = self.lin_first(node_feature)
        node_feature = node_feature.view(data.batch.max()+1, len(torch.where(data.batch==0)[0]), data.x.shape[1])
        for i in range(self.nlayer):
            node_feature = self.net[i](node_feature)
        return self.lin_in(node_feature.reshape(node_feature.shape[0] * node_feature.shape[1], self.in_channel))


class GCN(nn.Module):
    def __init__(self, in_channel, out_channel, **kwargs) -> None:
        super().__init__()
        self.dropout = 0.3
        self.net = nn.ModuleList([
            GCNConv(in_channel, in_channel),
            nn.LeakyReLU(),
            GCNConv(in_channel, in_channel),
            nn.LeakyReLU(),
            GCNConv(in_channel, in_channel),
            nn.LeakyReLU(),
            GCNConv(in_channel, out_channel),
            nn.LeakyReLU(),
        ])

    def forward(self, batch):
        x = batch.x
        for net in self.net:
            if isinstance(net, MessagePassing):
                x = F.dropout(x, self.dropout, training=self.training)
                x = net(x, batch.edge_index)
            else:
                x = net(x)
        return x


class SAGE(nn.Module):
    def __init__(self, in_channel, out_channel, **kwargs) -> None:
        super().__init__()
        self.net = nn.ModuleList([
            SAGEConv(in_channel, in_channel),
            nn.BatchNorm1d(in_channel),
            nn.LeakyReLU(),
            SAGEConv(in_channel, in_channel),
            nn.BatchNorm1d(in_channel),
            nn.LeakyReLU(),
            SAGEConv(in_channel, in_channel),
            nn.BatchNorm1d(in_channel),
            nn.LeakyReLU(),
            SAGEConv(in_channel, out_channel),
            nn.BatchNorm1d(out_channel),
            nn.LeakyReLU(),
        ])

    def forward(self, batch):
        x = batch.x
        for net in self.net:
            if isinstance(net, MessagePassing):
                x = net(x, batch.edge_index)
            else:
                x = net(x)
        return x


class SGC(nn.Module):
    def __init__(self, in_channel, out_channel, **kwargs) -> None:
        super().__init__()
        self.net = nn.ModuleList([
            SGConv(in_channel, in_channel),
            nn.BatchNorm1d(in_channel),
            nn.LeakyReLU(),
            SGConv(in_channel, in_channel),
            nn.BatchNorm1d(in_channel),
            nn.LeakyReLU(),
            SGConv(in_channel, in_channel),
            nn.BatchNorm1d(in_channel),
            nn.LeakyReLU(),
            SGConv(in_channel, out_channel),
            nn.BatchNorm1d(out_channel),
            nn.LeakyReLU(),
        ])

    def forward(self, batch):
        x = batch.x
        for net in self.net:
            if isinstance(net, MessagePassing):
                x = net(x, batch.edge_index)
            else:
                x = net(x)
        return x
