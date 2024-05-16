import torch
from torch import nn

from typing import Optional, Tuple, Union
# from torch_geometric.nn.dense.linear import Linear

import networkx as nx
# import nx_cugraph as nxcg

class Graphormer(nn.Module):

    def __init__(self, 
        nlayer: int = 2,
        node_sz: int=116,
        in_channel: Union[int, Tuple[int, int]] = 10,
        out_channel: int = 10,
        heads: int = 2,
        dropout: float = 0.1,
        detour_type = 'node',
        batch_size = 32,
        device='cuda:0',
        *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        org_in_channel = in_channel
        if in_channel % heads != 0:
            in_channel = in_channel  + heads - (in_channel % heads)
        self.detour_type = detour_type
        self.nlayer = nlayer
        self.node_sz = node_sz

        self.lin_first = nn.Sequential(
            nn.Linear(org_in_channel, in_channel), 
            nn.BatchNorm1d(in_channel), 
            nn.LeakyReLU(),
        )
        self.lin_in = nn.Sequential(
            nn.Linear(in_channel, out_channel), 
            nn.BatchNorm1d(out_channel), 
            nn.LeakyReLU(),
        )
        self.net = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(d_model=in_channel, nhead=heads, dim_feedforward=in_channel, dropout=dropout, batch_first=True),
            num_layers=nlayer,
            norm=None#nn.LayerNorm(in_channel)
        )
        num_spatial = 100
        max_degree = 1000
        self.deg_embedding = nn.Embedding(max_degree, in_channel)
        self.spd_embedding = nn.Embedding(num_spatial, heads, padding_idx=0)

        self.heads = heads
        self.in_channel = in_channel
        self.out_channel = out_channel


    def forward(self, data):
        self.loss = 0
        node_feature = data.x
        node_feature = self.lin_first(node_feature)
        node_feature = node_feature.view(data.batch.max()+1, len(torch.where(data.batch==0)[0]), self.in_channel)
        node_feature = node_feature + self.deg_embedding(data.adj_fc.bool().sum(1))
        spd_dist = data.spd_dist.long()
        spd_mask = spd_dist < 0
        spd_dist[spd_mask] = 0
        att_mask = self.spd_embedding(spd_dist)
        att_mask[spd_mask, :] = -1
        att_mask = torch.cat([att_mask[:, :, :, i] for i in range(self.heads)])
        node_feature = self.net(node_feature, mask=att_mask)

        return self.lin_in(node_feature.view(node_feature.shape[0] * node_feature.shape[1], self.in_channel))


class ShortestDistance:

    def __init__(self, cutoff=10, **kargs) -> None:
        self.k = cutoff

    def __call__(self, data):
        nodesz = data.x.shape[0]
        adj = torch.zeros(nodesz, nodesz)
        adj[data.edge_index[0], data.edge_index[1]] = 1
        G = nx.from_numpy_array(adj.numpy())
        length = nx.all_pairs_shortest_path_length(G, cutoff=self.k, backend="parallel")
        spd = torch.zeros(nodesz, nodesz) - 1
        length = list(length)
        for n, path in length:
            ind = torch.LongTensor(list(path.keys()))
            spd[n, ind] = torch.FloatTensor(list(path.values()))
        return {'spd_dist': spd[None]}

        