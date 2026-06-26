'''
Validate + benchmark the fast structural-detour counter (detour_count.py)
against the original networkx `get_de` (depth_first_search.py).

  python detour_count_benchmark.py            # validate + benchmark on synthetic UKB-scale graphs

The counter produces the *exact same* DE vectors as `get_de`; it just replaces
per-FC-edge `nx.all_simple_paths(...)` enumeration with one bounded DFS per
source node that counts simple paths to all targets at once.
'''
import time
import numpy as np
import networkx as nx

from detour_count import all_de


def get_de(G, ni, nj, k):  # original, copied from depth_first_search.py
    de = [0] * k
    for path in nx.all_simple_paths(G, ni, nj, cutoff=k):
        if len(path) < 2:
            continue
        de[len(path) - 2] += 1
    return de


def _both_dirs(G):
    ei = np.array(list(G.edges())).T
    return np.concatenate([ei, ei[::-1]], axis=1)


def validate(seeds=6):
    print('== correctness vs networkx get_de ==')
    rng = np.random.default_rng(0)
    ok = True
    for t in range(seeds):
        N = int(rng.choice([30, 50, 80]))
        p = float(rng.choice([0.08, 0.12, 0.18]))
        k = int(rng.choice([4, 5, 6]))
        Gsc = nx.gnp_random_graph(N, p, seed=t)
        Gfc = nx.gnp_random_graph(N, 0.2, seed=t + 100)
        ei_sc, ei_fc = _both_dirs(Gsc), _both_dirs(Gfc)
        mine = all_de(ei_fc, ei_sc, k, N)
        ref = np.array([get_de(Gsc, int(ei_fc[0, c]), int(ei_fc[1, c]), k)
                        for c in range(ei_fc.shape[1])])
        m = np.array_equal(mine, ref)
        ok &= m
        print(f'  N={N:3d} p={p:.2f} k={k}  match={m}')
    print('  ALL EXACT MATCH:', ok)
    return ok


def benchmark(N=333, sc_deg=8, fc_density=0.15, ks=(5, 6, 7), sample=400):
    print(f'\n== benchmark  N={N} SC_avg_deg={sc_deg} FC_density={fc_density} ==')
    Gsc = nx.watts_strogatz_graph(N, sc_deg, 0.1, seed=1)
    Gfc = nx.gnp_random_graph(N, fc_density, seed=2)
    ei_sc, ei_fc = _both_dirs(Gsc), _both_dirs(Gfc)
    E = ei_fc.shape[1]
    print(f'  FC directed edges/subject = {E}')
    sub = ei_fc[:, :sample]
    for k in ks:
        t = time.time()
        for c in range(sub.shape[1]):
            get_de(Gsc, int(sub[0, c]), int(sub[1, c]), k)
        orig = (time.time() - t) / sub.shape[1] * E          # s / subject (scaled from sample)
        t = time.time()
        all_de(ei_fc, ei_sc, k, N)
        new = time.time() - t                                # s / subject (full)
        print(f'  k={k}: original ~{orig:9.1f} s/subj | counter {new:7.2f} s/subj '
              f'| speedup ~{orig / new:5.0f}x | 50k @counter = {new * 50000 / 3600:6.0f} core-h')


if __name__ == '__main__':
    validate()
    benchmark()
