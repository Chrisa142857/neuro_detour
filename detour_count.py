'''
Faster structural-detour (DE) computation for NeuroDetour.

The original `get_de` in depth_first_search.py calls
`nx.all_simple_paths(SC, ni, nj, cutoff=k)` once **per FC edge** and materializes
every path as a Python list just to count them. That restarts a DFS from `ni`
for every functional pair and pays the cost of building/yielding ~10^4 path
lists per edge at k=8.

`source_de` here replaces that with a single bounded DFS **per source node** that
*counts* simple paths to every target at once (no path list is ever built), then
reads off the DE vector for each FC edge. Semantics are identical to `get_de`:
de[len(path)-2] is the number of simple SC paths with that many nodes, for
path lengths up to `k` edges.
'''
import numpy as np


def build_adj(edge_index, num_nodes):
    '''edge_index: LongTensor/ndarray [2, E] (directed, both dirs present). -> list[np.ndarray]'''
    ei = np.asarray(edge_index)
    adj = [[] for _ in range(num_nodes)]
    for a, b in zip(ei[0].tolist(), ei[1].tolist()):
        if a != b:
            adj[a].append(b)
    return [np.asarray(sorted(set(nb)), dtype=np.int64) for nb in adj]


def source_de(adj, s, k, num_nodes):
    '''
    One bounded DFS from source s. Returns counts[N, k] where counts[t, L] is the
    number of simple paths s->t using exactly L+1 edges (i.e. de index len(path)-2),
    for paths of up to k edges. Matches get_de(SC, s, t, k) for every t.
    '''
    counts = np.zeros((num_nodes, k), dtype=np.int64)
    visited = bytearray(num_nodes)
    visited[s] = 1

    def dfs(u, eu):
        for v in adj[u]:
            if visited[v]:
                continue
            counts[v, eu] += 1          # path s..v has eu+1 edges -> idx = eu
            if eu + 1 < k:
                visited[v] = 1
                dfs(v, eu + 1)
                visited[v] = 0

    dfs(s, 0)
    return counts


def all_de(edge_index_fc, edge_index_sc, k, num_nodes):
    '''
    Drop-in replacement for the de_list loop in NeuroDetourNode/Edge.
    Returns dee [E_fc, k] aligned with the columns of edge_index_fc.
    '''
    sc_adj = build_adj(edge_index_sc, num_nodes)
    ei_fc = np.asarray(edge_index_fc)
    # cache one DFS per distinct source node that appears as an FC source
    cache = {}
    out = np.zeros((ei_fc.shape[1], k), dtype=np.int64)
    for col in range(ei_fc.shape[1]):
        s = int(ei_fc[0, col]); t = int(ei_fc[1, col])
        if s == t:
            continue
        if s not in cache:
            cache[s] = source_de(sc_adj, s, k, num_nodes)
        out[col] = cache[s][t]
    return out
