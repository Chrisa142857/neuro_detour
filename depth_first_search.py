import torch
import networkx as nx
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_scatter import scatter_max, scatter_mean, scatter_min, scatter_std



class NeuroDetourNode:
    
    def __init__(self, k=5, node_num=116) -> None:
        self.PEK = node_num
        self.node_num = node_num
        self.k = k
        self.node_list = [i for i in range(node_num)]


    def __call__(self, data):
        edge_index1, edge_index2, features = data.edge_index_fc, data.edge_index_sc, data.x
        edge_index1 = remove_self_loops(edge_index1)[0]
        edge_index1  = edge_index1[:, edge_index1[0].argsort()]
        G1 = nx.Graph()
        G1.add_nodes_from(self.node_list)
        G1.add_edges_from(edge_index1.T.tolist())
        G2 = nx.Graph()
        G2.add_nodes_from(self.node_list)
        G2.add_edges_from(edge_index2.T.tolist())
        de_list = []
        for j in range(edge_index1.shape[1]):
            de_list.append(get_de(G2, edge_index1[0, j].item(), edge_index1[1, j].item(), self.k))
        dee = torch.FloatTensor(de_list)#[edge, k-1]
        node_dee = torch.cat([
            scatter_max(dee, index=edge_index1[0], dim=0, out=torch.zeros(self.node_num, dee.shape[1]))[0],
            scatter_mean(dee, index=edge_index1[0], dim=0, out=torch.zeros(self.node_num, dee.shape[1])),
            scatter_min(dee, index=edge_index1[0], dim=0, out=torch.zeros(self.node_num, dee.shape[1]))[0],
            scatter_std(dee, index=edge_index1[0], dim=0, out=torch.zeros(self.node_num, dee.shape[1])),
        ], -1)#[node, (k-1)*4]
        token = data.x
        # token = torch.cat([data.x, torch.cat([edge_index1[0:1].T, dee, edge_index1[1:2].T, torch.zeros(dee.shape[0], data.x.shape[1]-dee.shape[1]-2)], dim=1)], dim=0)
        lap = torch.from_numpy(nx.laplacian_matrix(G1).toarray()).float()
        L, V = torch.linalg.eig(lap)
        pe = V[:, :self.PEK].real
        # xlist, pad_mask = segment_node_with_neighbor(edge_index1, node_attrs=[features, pe], edge_attrs=[dee])
        return {
            'token': token,
            'PE': pe,
            'DE': node_dee,
            'ID': torch.ones(data.x.shape[0], 1),#torch.cat([torch.ones(data.x.shape[0], 1), torch.ones(dee.shape[0], 1)*2]),
            'mask': torch.ones(data.x.shape[0]).bool()
        }
    
class NeuroDetourEdge:
    def __init__(self, k=5, node_num=116) -> None:
        self.k = k
        self.PEK = node_num
        self.node_num = node_num
        self.node_list = [i for i in range(node_num)]


    def __call__(self, data):
        edge_index1, edge_index2, features = data.edge_index_fc, data.edge_index_sc, data.x
        N = features.shape[0]
        G1 = nx.Graph()
        G1.add_nodes_from(self.node_list)
        G1.add_edges_from(edge_index1.T.tolist())
        G2 = nx.Graph()
        G2.add_nodes_from(self.node_list)
        G2.add_edges_from(edge_index2.T.tolist())
        de_list = []
        for j in range(edge_index1.shape[1]):
            de_list.append(get_de(G2, edge_index1[0, j].item(), edge_index1[1, j].item(), self.k))
        dee = torch.FloatTensor(de_list)#[:, None]
        lap = torch.from_numpy(nx.laplacian_matrix(G1).toarray()).float()
        L, V = torch.linalg.eig(lap)
        pe = V[:, :self.PEK].real
        xlist, pad_mask = segment_node_with_neighbor(edge_index1, node_attrs=[features, pe], edge_attrs=[dee])
        mask = torch.zeros(xlist[0].shape[0]).bool()
        mask[pad_mask] = True
        return {
            'token': xlist[0],
            'PE': xlist[1],
            'DE': xlist[2],
            'ID': xlist[3],
            'mask': mask
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
    de = [0 for _ in range(k)]
    for path in nx.all_simple_paths(G, source=ni, target=nj, cutoff=k):
        if len(path) < 2: continue
        de[len(path)-2] += 1 
    return de

