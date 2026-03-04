import os
import torch
from torch_geometric.utils import to_networkx

from tqdm import tqdm
import argparse
import time
import numpy as np
from torch_geometric.data import Data#, download_url

import igraph as ig

class PathTransform(object):
    def __init__(self, path_type, cutoff):
        self.cutoff = cutoff
        self.path_type = path_type

    def __call__(self, data):
        if self.path_type is None and self.cutoff is None:
            data.pp_time = 0
            return data

        setattr(data, f"path_2", data.edge_index.T.flip(1))
        setattr(
            data,
            f"edge_indices_2",
            get_edge_indices(
                data.x.size(0), data.edge_index, data.edge_index.T.flip(1)
            ),
        )

        if self.cutoff == 2 and self.path_type is None:
            data.pp_time = 0
            return ModifData(**data.stores[0])

        t0 = time.time()
        G = ig.Graph.from_networkx(to_networkx(data, to_undirected=True))
        if self.path_type == "all_simple_paths":
            setattr(
                data,
                f"sp_dists_2",
                torch.cat(
                    [torch.ones(data.num_edges, 1), torch.zeros(data.num_edges, 1)],
                    dim=1,
                ).long(),
            )
        graph_info = fast_generate_paths2(
            G, self.cutoff, self.path_type, undirected=True
        )

        cnt = 0
        for jj in range(1, self.cutoff - 1):
            paths = torch.LongTensor(graph_info[0][jj]).view(-1, jj + 2)
            setattr(data, f"path_{jj+2}", paths.flip(1))
            setattr(
                data,
                f"edge_indices_{jj+2}",
                get_edge_indices(data.x.size(0), data.edge_index, paths.flip(1)),
            )
            if self.path_type == "all_simple_paths":
                if len(paths) > 0:
                    setattr(
                        data,
                        f"sp_dists_{jj+2}",
                        torch.Tensor(graph_info[2][jj]).long().flip(1),
                    )
                else:
                    setattr(data, f"sp_dists_{jj+2}", torch.empty(0, jj + 2).long())
                    cnt += 1
        data.max_cutoff = self.cutoff
        data.cnt = cnt
        data.pp_time = time.time() - t0
        return ModifData(**data.stores[0])


def get_edge_indices(size, edge_index_n, paths):
    index_tensor = torch.zeros(size, size, dtype=torch.long, device=paths.device)
    index_tensor[edge_index_n[0], edge_index_n[1]] = torch.arange(
        edge_index_n.size(1), dtype=torch.long, device=paths.device
    )
    indices = []
    for i in range(paths.size(1) - 1):
        indices.append(index_tensor[paths[:, i], paths[:, i + 1]].unsqueeze(1))

    return torch.cat(indices, -1)


class ModifData(Data):
    def __init__(self, edge_index=None, x=None, *args, **kwargs):
        super().__init__(x=x, edge_index=edge_index, **kwargs)

    def __inc__(self, key, value, *args, **kwargs):

        if "index" in key or "path" in key:
            return self.num_nodes
        elif "indices" in key:
            return self.num_edges
        else:
            return 0

    def __cat_dim__(self, key, value, *args, **kwargs):
        if "index" in key or "face" in key:
            return 1
        else:
            return 0



def fast_generate_paths2(g, cutoff, path_type, weights=None, undirected=True, r=None):
    if undirected and g.is_directed():
        g.to_undirected()

    path_length = np.array(g.distances())
    if path_type != "all_simple_paths":
        diameter = g.diameter(directed=False)
        diameter = diameter + 1 if diameter + 1 < cutoff else cutoff

    else:
        diameter = cutoff

    X = [[] for i in range(cutoff - 1)]
    sp_dists = [[] for i in range(cutoff - 1)]

    for n1 in range(g.vcount()):
        if path_type == "all_simple_paths":
            paths_ = g.get_all_simple_paths(n1, cutoff=cutoff - 1)

            for path in paths_:
                # if len(path) >= min_length and len(path) <= cutoff :
                idx = len(path) - 2
                if len(path) > 0:
                    X[idx].append(path)
                    sp_dist = []
                    for node in path:
                        sp_dist.append(path_length[n1, node])
                    sp_dists[idx].append(sp_dist)

        else:
            valid_ngb = [
                i
                for i in np.where(
                    (path_length[n1] <= cutoff - 1) & (path_length[n1] > 0)
                )[0]
                if i > n1
            ]
            for n2 in valid_ngb:
                if path_type == "shortest_path":
                    paths_ = g.get_shortest_paths(n1, n2, weights=weights)
                elif path_type == "all_shortest_paths":
                    paths_ = g.get_all_shortest_paths(n1, n2, weights=weights)

                for path in paths_:
                    idx = len(path) - 2
                    X[idx].append(path)
                    X[idx].append(list(reversed(path)))

    return X, diameter, sp_dists


def get_dataset(cutoff, path_type):
    data = NeuroNetworkDataset(dname='ukb', node_attr = 'BOLD', atlas_name='Gordon_333', transform=PathTransform(path_type, cutoff))
    return data

if __name__ == "__main__":
    from datasets import NeuroNetworkDataset
    parser = argparse.ArgumentParser()
    # parser.add_argument("--dataset", type=str, default="ogbg-molhiv")
    parser.add_argument("--min_cutoff", type=int, default=3)
    parser.add_argument(
        "--max_cutoff", type=int, default=5, help="Max length of shortest paths"
    )
    args = parser.parse_args()

    for cutoff in range(args.min_cutoff, args.max_cutoff + 1):
        # for dataset in ["ogbg-molhiv", "ogbg-molpcba", "ZINC", "peptides-functional", "peptides-structural"] :
        for dataset in [args.dataset]:

            for path_type in [
                "shortest_path",
                "all_shortest_paths",
                "all_simple_paths",
            ]:

                data = get_dataset(dataset, cutoff, path_type)
                print(data.preprocessing_time)
