---
name: structural-detour-counting
description: >
  Compute NeuroDetour structural-detour encodings (DE) -- counts of bounded-length
  simple SC paths between functionally-connected node pairs -- at scale. Use when
  working with detour features (`get_de`, `NeuroDetourNode`/`Edge`), estimating
  feasibility of large detour computations (e.g. UKB 50k FC-SC pairs), or choosing
  between exact and approximate counters for sparse vs dense SC / small vs large k.
---

# Structural-detour (DE) counting

## What a "structural detour" is
For each **FC edge** `(i,j)` (functional connection), a detour is a **simple path in
the SC graph** between `i` and `j`. The **DE vector** for that edge counts how many
such SC paths exist at each length, up to `k` edges:

```
DE[L] = number of simple SC paths from i to j with (L+2) nodes / (L+1) edges
        i.e. exactly get_de's index L = len(path) - 2,   for L in 0..k-1
```

`k` = max detour length in **edges**. The original reference implementation is
`get_de(G, ni, nj, k)` in `depth_first_search.py` (uses `nx.all_simple_paths`).

## Cost model (read before scaling anything)
- Naive `get_de` is **per-FC-edge** and **output-sensitive**: cost grows with the
  number of paths, which is ~`(avg_degree-1)^k`. It explodes with SC density and k.
- Per-subject cost ~ `O(#FC_edges * #paths)` ∝ **N^2** in atlas size N.
- **SC density (avg degree), not node count, is the dominant exponential lever.**
- See `DETOUR_FEASIBILITY.md` for full benchmarks (N=100..1000, k=5..8, 50k pairs).

## The three backends

### 1. `detour_count.compute_dee(...)` — unified entry point (use this)
```python
from detour_count import compute_dee
dee = compute_dee(edge_index_fc, edge_index_sc, k, num_nodes,
                  method='auto',      # 'auto' | 'exact' | 'color'
                  path_budget=5e5,    # auto -> 'color' when est. paths/source exceeds this
                  trials=40, seed=0)  # color-coding sampling (ignored when exact)
# returns torch.FloatTensor [E_fc, k], aligned with the columns of edge_index_fc
```
- `edge_index_*`: LongTensor/ndarray `[2, E]` (directed, both directions), as produced
  by `torch.where(adj > th)` in `datasets.py`.
- `method='auto'`: exact while `estimate_paths_per_source(avg_deg,k) <= path_budget`,
  else color-coding. Keeps existing experiments (sparse SC, k=5) **bit-exact**.

### 2. `detour_count` — exact, per-source DFS counter (sparse SC / low k)
```python
all_de(edge_index_fc, edge_index_sc, k, num_nodes)  # -> np.ndarray [E_fc, k], exact
source_de(adj_lists, s, k, num_nodes)               # -> np.ndarray [N, k] from one source
build_adj(edge_index, num_nodes)                    # -> list[np.ndarray] adjacency lists
dense_adj(edge_index, num_nodes)                    # -> np.ndarray [N,N] {0,1} symmetric
estimate_paths_per_source(avg_deg, k)               # -> float, drives the auto switch
```
- **Exact**: `all_de` output is identical to looping `get_de` over FC edges (validated).
- One bounded DFS **per source node** (counts to all targets at once, no path lists).
  ~140x faster than `get_de` at N=333; cost is **linear in N**, but still
  output-sensitive (explodes on dense SC).

### 3. `detour_colorcoding.colorcoding_de(...)` — sub-exponential, dense SC / large k
```python
from detour_colorcoding import colorcoding_de
est = colorcoding_de(adj_dense, k, K=None, trials=40, seed=0, src_batch=None)
# adj_dense: [N,N] {0,1} symmetric numpy. K defaults to k+1 (colors). Must be >= k+1.
# returns np.ndarray [N, N, k]:  est[s, t, L] ~= DE count, index L = len(path)-2
```
- Alon–Yuster–Zwick **color-coding**: layered DP over (color-subset, vertex) counting
  *colorful* walks, rescaled to an **unbiased estimate**.
- Cost `O(trials * 2^K * N * E)` — polynomial in graph size, FPT in k, and
  **independent of the path count**. Runtime is flat as SC density grows.
- **Approximate**: aggregate detour totals within ~1–5%, per-edge ~4–16% at 40–80
  trials (tightens with more `trials`; estimator is unbiased). Index with
  `est[edge_index_fc[0], edge_index_fc[1]]` to align to FC edges.

## Using it inside the model transforms
`NeuroDetourNode` / `NeuroDetourEdge` (`depth_first_search.py`) now call `compute_dee`
and expose the switch via constructor args (defaults preserve old behavior):
```python
from depth_first_search import NeuroDetourNode, NeuroDetourEdge
transform = NeuroDetourNode(k=8, node_num=333,
                            de_method='auto',     # 'auto' | 'exact' | 'color'
                            de_trials=40,          # color-coding samples
                            de_path_budget=5e5)    # auto exact->color threshold
# pass `transform` to NeuroNetworkDataset / dataloader_generator in datasets.py
```
The transform is applied once and cached to disk (`data/<dir>/processed_*.pt`), so the
DE cost is a one-time preprocessing step, embarrassingly parallel across subjects.

## Which backend to pick
| Situation | Backend | Why |
|---|---|---|
| Sparse SC (avg deg ≲ 12) and small k (≤6) | `exact` | fast and bit-exact |
| Dense SC (avg deg ≥ 16) or large k (7–8) | `color` | exact explodes; color-coding is flat |
| Not sure / mixed cohort | `auto` | exact until it would blow up, then color |
| Need reproducible published numbers | `exact` | identical to original `get_de` |

## Validate / benchmark
```bash
python detour_count_benchmark.py     # validate exact==get_de, then benchmark vs original
```
`detour_count_benchmark.py` exposes `validate()` and `benchmark(N, sc_deg, fc_density, ks)`.

## Feasibility summary (50k UKB FC-SC pairs, k=8)
- Original `get_de`: ~26 h/subject at Gordon-333 → ~150 core-years. Not feasible.
- `compute_dee` exact (sparse SC): AAL-116 ~3.7 min, Gordon-333 ~10 min, 1000-ROI ~35 min
  per subject → cluster-days. Linear in N.
- `compute_dee` color (any density): ~32 s/subject at N=333, **flat in SC density**
  → ~440 core-hours (~7 h on 64 cores) for all 50k.
- Full numbers and tables: `DETOUR_FEASIBILITY.md`.
