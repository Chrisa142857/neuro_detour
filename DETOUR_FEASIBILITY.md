# Feasibility of max-length-8 structural detours across atlases (50k UKB FC–SC pairs)

## What is being computed
For every FC edge `(i,j)`, count the simple SC paths between `i` and `j` up to
length `k` (the detour-encoding `DE`). Original implementation: `get_de` in
`depth_first_search.py` calls `nx.all_simple_paths(SC, i, j, cutoff=k)` **per FC
edge**. Fast exact replacement: `detour_count.py` (`all_de`) — one bounded DFS
**per source node**, counting paths to all targets at once (validated bit-identical).

Benchmarks below use synthetic graphs at the relevant node count `N`, SC modelled
as a clustered (Watts–Strogatz) graph, FC density ~0.15. Numbers are single-core;
the workload is embarrassingly parallel across subjects and cached to disk once.

## Scaling laws (the important part)
- **Original (per-FC-edge):** cost ∝ `#FC_edges × per-edge DFS` ∝ **N²** × (path-count).
- **New counter (per-source):** `#sources(≈N) × per-source DFS`. Per-source DFS is
  ~independent of N, so cost is **linear in N**. Confirmed: at SC degree 8, going
  N=100→1000 (10×) raises s/subject by ~10–11× at every `k`.
- Because the new method removes the N² term, **its speedup over the original grows
  ∝ N** (≈ avg-FC-degree × constant): ~140× at N=333, ~450× at N=1000.
- **SC density (avg degree) is the dominant exponential lever**, not node count.

## New counter — cost per subject / 50k core-hours, SC avg degree 8 (sparse SC)
| N (atlas) | k=5 | k=6 | k=7 | k=8 (s/subj) | k=8 → 50k core-h |
|---|---|---|---|---|---|
| 100        | 0.8 | 4.8 | 30 | 195   | 2,710 |
| 116 (AAL)  | 0.9 | 5.6 | 38 | 219   | 3,037 |
| 200        | 1.6 | 10  | 67 | 394   | 5,465 |
| 333 (Gordon)| 2.7| 16  | 107| 613   | 8,520 |
| 400        | 3.2 | 21  | 130| 805   | 11,177 |
| 600        | 5.0 | 32  | 195| 1,168 | 16,227 |
| 800        | 7.0 | 42  | 281| 1,661 | 23,062 |
| 1000       | 8.7 | 55  | 340| 2,116 | 29,394 |

(s/subject; last col = 50,000 subjects in core-hours.)

## Effect of SC density — SC avg degree 16 (denser atlas / lower sc_th)
| N | k=5 (s/subj) | k=6 | k=7 |
|---|---|---|---|
| 100 | 30  | 394 | 5,570 (~1.5 h) |
| 116 | 31  | 446 | 6,215 |

Doubling SC degree (8→16) costs ~40× at k=5 and explodes by k=7. At degree 16,
**k=8 is intractable at any atlas size.** `get_de` (original) at degree 8 was
already ~26 h/subject at N=333, k=8 (~150 core-years for 50k) — not feasible.

## Verdict per atlas (max length k=8)
With the new counter and **SC kept sparse (avg degree ~8, i.e. a sensible `sc_th`)**:

| Atlas | k=8 per subject | 50k on 64 cores | 50k on 1000 cores |
|---|---|---|---|
| AAL-116    | ~3.7 min  | ~2 days   | ~3 h |
| Gordon-333 | ~10 min   | ~5.5 days | ~9 h |
| ~1000-ROI  | ~35 min   | ~19 days  | ~1.2 days |

**Feasible across the whole 100–1000 range** on a cluster, because the counter
makes cost linear in N. The binding constraint is **SC density, not node count** —
keep SC sparse (or stack a C/igraph backend + FC symmetry) and even the 1000-ROI
atlas is tractable; leave SC dense and k=8 is impossible at any resolution.
