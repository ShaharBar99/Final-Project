# Step 2 - Breaking the problem into small parts

The Phase A model (Chapters 4-5) is split into 7 blocks. Each block is one module
in `src/`, has a clear input and output, and can be tested alone.

```
 clean graph G0
      |
 [A] features.py     centralities of G  ->  table (19,717 x 5)
      |
 [B] scaling.py      fit Min-Max on G0 ONCE (frozen), transform any table
      |
 [C] model.py        Random Forest: features -> importance score (target = PageRank)
      |
 [D] ranking.py      scores -> ranking (also for each single centrality = baselines)
      |
 [E] perturb.py      G0 -> K noisy copies (remove / add edges, 5/10/20 %)
      |   (for every copy: A -> B(frozen) -> C(no retraining) -> D)
 [F] stability.py    node CV, Spearman / Kendall vs base ranking
      |
 [G] core.py         network core + importance-vs-stability roles, hypotheses
```

## Blocks in detail

| Block | Module | Main functions | Input -> Output | Phase A req. |
|---|---|---|---|---|
| A | `features.py` | `compute_features(g)` | DiGraph -> DataFrame, 5 columns | 1,2,3,5,11 |
| B | `scaling.py` | `fit_scaler(df)`, `apply_scaler(scaler, df)` | features -> values in [0,1] on G0 (may exceed on perturbed) | 4,12 |
| C | `model.py` | `train_model(X, y)`, `predict_scores(model, X)` | scaled features -> score per node | 6,8,13 |
| D | `ranking.py` | `to_ranking(scores)` | scores -> rank 1..N | 7 |
| E | `perturb.py` | `remove_edges(g, frac, rng)`, `add_edges(g, frac, rng)`, `make_perturbed(g, kind, frac, k, seed)` | graph -> perturbed graph, same nodes | 9,10 |
| F | `stability.py` | `node_instability(score_matrix)`, `rank_correlations(base, perturbed)`, `instability_profile_corr(...)` | score matrix (nodes x K) -> CV vector, correlations | 14,15,16 |
| G | `core.py` | `find_core(importance, instability, pct)`, `node_roles(...)` | -> core set, role labels | 17 |

## Experiment grid (what actually gets run)

- Methods compared: Random Forest (with PageRank as feature), Random Forest (without PageRank, decision D2), and the 5 single centralities as baselines.
- Perturbation kinds: removal, addition (and optionally both together).
- Levels: 5%, 10%, 20% of edges.
- Repetitions K per setting (start with K = 10 to test, final K decided by runtime).
- Seeds: every run is driven by `config.SEED` plus the run number, so results can be repeated exactly.

## Points that need a precise rule (so we do not decide them later by accident)

1. **Closeness on a directed graph that is not strongly connected.** Most papers cannot reach most others. Rule: use the Wasserman-Faust formula (networkx default; computed with igraph since step 6, D11), which for directed graphs uses *incoming* distances, i.e. how close the rest of the network is to the paper (how fast influence reaches it from the citing side). Nodes that reach nobody get 0 (also what the Phase A test plan asks for).
2. **Betweenness**: exact, via igraph, divided by (n-1)(n-2) (D4, changed after timing it in step 3).
3. **k-core on a directed graph**: computed on the undirected version, self-loops removed.
4. **Degree**: total degree / (N-1) as in Appendix B.
5. **Scaler**: fitted on G0 only. On perturbed graphs values may fall outside [0,1] and are NOT clipped (that is the point of the frozen scaler).
6. **Zero mean in CV**: if the mean score of a node is 0, its instability is set to 0 (Phase A, Section 4.4).
7. **Training**: the forest is trained once on G0, all nodes. We also report a 80/20 hold-out R^2 as a sanity check of the model itself; the stability experiments do not depend on it.
