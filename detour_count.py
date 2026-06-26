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


def dense_adj(edge_index, num_nodes):
    '''edge_index [2,E] -> symmetric dense {0,1} [N,N] (self-loops dropped).'''
    import numpy as _np
    ei = _np.asarray(edge_index)
    A = _np.zeros((num_nodes, num_nodes), dtype=_np.float64)
    a, b = ei[0], ei[1]
    m = a != b
    A[a[m], b[m]] = 1.0
    A[b[m], a[m]] = 1.0
    return A


def estimate_paths_per_source(avg_deg, k):
    '''Rough count of simple paths from one node up to k edges; drives auto switch.'''
    b = max(avg_deg - 1.0, 1.0)
    return sum(b ** i for i in range(1, k))


def compute_dee(edge_index_fc, edge_index_sc, k, num_nodes,
                method='auto', path_budget=5e5, trials=40, seed=0):
    '''
    Unified structural-detour (DE) backend. Returns a torch.FloatTensor [E_fc, k]
    aligned with the columns of edge_index_fc -- a drop-in for the de_list loop.

      method='exact'  -> per-source DFS counter (detour_count.all_de), bit-exact.
      method='color'  -> color-coding estimate (detour_colorcoding), density-independent.
      method='auto'   -> exact while estimated paths/source <= path_budget, else color.

    Exact and the original nx-based get_de produce identical output; 'auto' only
    falls back to the (unbiased, approximate) color-coding estimator when exact
    enumeration would blow up (dense SC and/or large k).
    '''
    import numpy as _np
    import torch as _torch
    ei_fc = _np.asarray(edge_index_fc)
    ei_sc = _np.asarray(edge_index_sc)
    if ei_fc.size == 0 or ei_fc.shape[1] == 0:
        return _torch.zeros(0, k)
    avg_deg = ei_sc.shape[1] / max(num_nodes, 1)
    if method == 'auto':
        method = 'exact' if estimate_paths_per_source(avg_deg, k) <= path_budget else 'color'
    if method == 'exact':
        dee = all_de(ei_fc, ei_sc, k, num_nodes)
    else:
        from detour_colorcoding import colorcoding_de
        A = dense_adj(ei_sc, num_nodes)
        est = colorcoding_de(A, k, trials=trials, seed=seed)
        dee = est[ei_fc[0], ei_fc[1]]
    return _torch.from_numpy(_np.ascontiguousarray(dee)).float()


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
