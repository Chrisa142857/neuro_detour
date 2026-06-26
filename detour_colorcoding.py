'''
Sub-exponential structural-detour (DE) counter for the DENSE-SC regime.

The DFS counter in detour_count.py is exact but *output-sensitive*: its cost is
proportional to the number of simple paths, which explodes when SC is dense
(avg degree >= 16). Color-coding (Alon-Yuster-Zwick, 1995) breaks that link: its
cost is poly(N, E) * 2^K and is **independent of the number of paths**, so it
stays bounded exactly where enumeration/DFS dies.

Idea: randomly color the N nodes with K colors. A walk that uses K *distinct*
colors is necessarily a simple path (no repeated vertex). We count "colorful"
walks with a DP over (color-subset, vertex); a length-L simple path uses L+1
distinct colors, so K = k_max_edges + 1 colors lets us catch paths up to k edges.
A given (L+1)-vertex path is colorful with probability
    p_L = K! / ((K-L-1)! * K^(L+1)),
so dividing the colorful count by p_L gives an UNBIASED estimate of the true
simple-path count; averaging over `trials` random colorings reduces variance.

`colorcoding_de` returns counts[s, t, L] = estimated number of simple paths with
L+1 nodes (i.e. the get_de index L = len(path)-2) from s to t, for L in 0..k-1 --
the same DE semantics as get_de / detour_count, but approximate.

DP is layered by popcount (subset of size p feeds only size p+1), so only two
adjacent layers are held in memory at once.
'''
import math
import numpy as np


def _subsets_by_popcount(K):
    layers = [[] for _ in range(K + 1)]
    for C in range(1 << K):
        layers[bin(C).count('1')].append(C)
    return layers


def colorcoding_de(adj, k, K=None, trials=40, seed=0, src_batch=None):
    '''
    adj: dense [N, N] {0,1} numpy array (undirected, symmetric).
    k:   max path length in EDGES (paths up to k edges / k+1 nodes).
    K:   number of colors (default k+1; must be >= k+1 to catch the longest path).
    Returns counts[N, N, k] float estimates aligned with get_de index (len(path)-2).
    '''
    N = adj.shape[0]
    if K is None:
        K = k + 1
    assert K >= k + 1, 'need K >= k+1 colors to color a (k+1)-node path'
    A = adj.astype(np.float64)
    layers = _subsets_by_popcount(K)
    if src_batch is None:
        src_batch = N

    # correction 1/p_L for DE index L: such a path has L+2 nodes (len(path)-2 == L)
    inv_p = np.zeros(k)
    for L in range(k):
        nodes = L + 2
        if nodes <= K:
            logp = (math.lgamma(K + 1) - math.lgamma(K - nodes + 1)) - nodes * math.log(K)
            inv_p[L] = math.exp(-logp)

    out = np.zeros((N, N, k), dtype=np.float64)
    rng = np.random.default_rng(seed)

    for start in range(0, N, src_batch):
        srcs = np.arange(start, min(start + src_batch, N))
        B = len(srcs)
        acc = np.zeros((B, N, k), dtype=np.float64)  # accumulated colorful counts over trials
        for _ in range(trials):
            color = rng.integers(0, K, size=N)
            color_bit = (1 << color).astype(np.int64)
            color_eq = [np.where(color == c)[0] for c in range(K)]  # vertices of each color

            # layer p=1: F[{color[s]}][b, v] = 1 iff v==s and that singleton
            F = {}
            for c in range(K):
                C = 1 << c
                mat = np.zeros((B, N))
                sc = srcs[color[srcs] == c]
                if len(sc):
                    bidx = np.searchsorted(srcs, sc)
                    mat[bidx, sc] = 1.0
                if mat.any():
                    F[C] = mat
            # contribute length-0 (single node) -- not a detour, skip (L starts at edges>=1)

            for p in range(1, K):           # extend walks of p nodes -> p+1 nodes
                nextF = {}
                for C, X in F.items():
                    if not X.any():
                        continue
                    Y = X @ A               # inflow to every vertex
                    for c in range(K):
                        if C & (1 << c):
                            continue
                        Vc = color_eq[c]
                        if len(Vc) == 0:
                            continue
                        Cn = C | (1 << c)
                        contrib = Y[:, Vc]
                        if Cn in nextF:
                            nextF[Cn][:, Vc] += contrib
                        else:
                            m = np.zeros((B, N)); m[:, Vc] = contrib; nextF[Cn] = m
                # after building layer p+1 (paths of p edges), accumulate
                edges = p                    # path with p+1 nodes has p edges
                idx = edges - 1              # get_de index = len(path)-2 = (p+1)-2 = p-1
                if idx < k:
                    tot = np.zeros((B, N))
                    for Cn, m in nextF.items():
                        tot += m
                    acc[:, :, idx] += tot
                F = nextF
                if not F:
                    break
        # average trials and apply correction
        est = acc / trials * inv_p[None, None, :]
        out[srcs] = est
    return out
