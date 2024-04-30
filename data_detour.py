import torch
import networkx as nx
from torch_geometric.utils import remove_self_loops, add_self_loops
# import numpy as np

class NeuroDetour:
    def __init__(self, k=5, node_num=116) -> None:
        self.k = k
        self.dim_num = {
            'token': node_num*2,
            'PE': node_num*2,
            'DE': k-1,
            'ID': 1,
        }

    def __call__(self, edge_index1, edge_index2, features):
        N = features.shape[0]
        PE_K = N
        G1 = nx.from_edgelist(edge_index1.T.tolist())
        G2 = nx.from_edgelist(edge_index2.T.tolist())
        de_list = []
        for j in range(edge_index1.shape[1]):
            de_list.append(get_de(G2, edge_index1[0, j].item(), edge_index1[1, j].item(), self.k))
        dee = torch.FloatTensor(de_list)#[:, None]
        lap = torch.from_numpy(nx.laplacian_matrix(G1).toarray())
        L, V = torch.linalg.eig(lap)
        pe = V[:, :PE_K].real
        xlist, pad_mask = segment_node_with_neighbor(edge_index1, node_attrs=[features, pe], edge_attrs=[dee])
        return {
            'token': xlist[0],
            'PE': xlist[1],
            'DE': xlist[2],
            'ID': xlist[3],
            'mask': pad_mask
        }

def segment_node_with_neighbor(edge_index, node_attrs=[], edge_attrs=[], pad_value=0):
    edge_attr_ch = [edge_attr.shape[1] for edge_attr in edge_attrs]
    edge_index, edge_attrs = remove_self_loops(edge_index, torch.cat(edge_attrs, -1) if len(edge_attrs)>0 else None)
    edge_index, edge_attrs = add_self_loops(edge_index, edge_attrs)
    if len(node_attrs[0]) > edge_index.max()+1:
        if edge_attrs is not None:
            edge_attrs = torch.cat([edge_attrs] + [torch.zeros(1, edge_attrs.shape[1]) for i in range(edge_index.max()+1, len(node_attrs[0]))], 0)
        edge_index = torch.cat([edge_index] + [torch.LongTensor([[i, i]]).T for i in range(edge_index.max()+1, len(node_attrs[0]))], 1)

    sortid = edge_index[0].argsort()
    edge_index = edge_index[:, sortid]
    if edge_attrs is not None:
        edge_attrs = edge_attrs[sortid]
    edge_attr_ch = [0] + torch.LongTensor(edge_attr_ch).cumsum(0).tolist()
    edge_attrs = [edge_attrs[:, edge_attr_ch[i]:edge_attr_ch[i+1]] for i in range(len(edge_attr_ch)-1)]
    id_mask = edge_index[0] == edge_index[1]
    edge_attrs.append(id_mask.float()[:, None])
    for i in range(len(node_attrs)):
        node_attrs[i] = torch.cat([node_attrs[i][edge_index[0]], node_attrs[i][edge_index[1]]], -1)
    attrs = node_attrs + edge_attrs
    segment = [torch.where(edge_index[0]==e)[0][0].item() for e in edge_index.unique()] + [edge_index.shape[1]]
    seq = [[] for _ in range(len(attrs))]
    seq_mask = []
    for i in range(len(segment)-1):
        for j in range(len(attrs)):
            attr = attrs[j][segment[i]:segment[i+1]]
            selfloop = torch.where(edge_index[0, segment[i]:segment[i+1]]==edge_index[1, segment[i]:segment[i+1]])[0].item()
            attr = torch.cat([attr[selfloop:selfloop+1], attr[:selfloop], attr[selfloop+1:]]) # Move self loop to the first place
            seq[j].append(attr)
        # seq_mask.append(torch.ones(seq[j][0].shape[0], 1))
        seq_mask.append(seq[j][0].shape[0])
    # seq = [pad_sequence(s, batch_first=True, padding_value=pad_value) for s in seq] # [(N, S, C)]
    # seq_mask = pad_sequence(seq_mask, batch_first=True, padding_value=0).float()
    # return seq, seq_mask, edge_index
    seq_mask = torch.LongTensor(segment[:-1])
    seq = [torch.cat(s) for s in seq]
    C = int(node_attrs[0].shape[1]/2)
    assert (seq[0][seq_mask, :C]==seq[0][seq_mask, C:]).all()
    return seq, seq_mask # [(N*S, C)]

def get_de(G, ni, nj, k):
    de = [0 for _ in range(k-1)]
    for path in nx.all_simple_paths(G, source=ni, target=nj, cutoff=k):
        if len(path) < 2: continue
        de[len(path)-2] += 1 
    return de

