# Decisions and assumptions

| # | Decision | Reason | Date |
|---|---|---|---|
| D1 | Use PubMed-Diabetes instead of Cora | Dataset provided for the project (Phase A mentioned Cora as preliminary) | 2026-10-10 |
| D2 | Run the model both with and without PageRank as an input feature | PageRank is also the target; with it as a feature the forest can just copy it | 2026-10-10 |
| D3 | Temporal consistency checked structurally (no cycles), not by year | No publication dates in the data | 2026-10-10 |
| D4 | ~~Betweenness approximated by sampling~~ **Replaced 2026-10-23: exact betweenness, computed with igraph** | See below | 2026-10-10 / 2026-10-23 |
| D5 | Mutual citations: keep the edge from the higher PubMed ID to the lower | PubMed IDs grow over time, so higher ID ~ newer paper. This is an assumption; 11 edges out of 44k, so the effect is tiny | 2026-10-10 |
| D7 | Forest settings: 200 trees, `min_samples_leaf=5`, seed 42, other options default | Leaf size 5 keeps the saved model at 18-33 MB instead of several hundred and avoids memorising single papers; no tuning was done, since the goal is stability, not the best possible fit | 2026-10-23 |
| D8 | Model files are not committed to GitHub; they are rebuilt from the notebook (fixed seed) | 20-30 MB each; cheap to rebuild | 2026-10-23 |
| D9 | Rank 1 = most important; tied papers share the average rank | Avoids inventing an order between papers with identical scores; the same rule is used for every method and every graph | 2026-10-23 |
| D6 | Drop the 500 word-feature columns | Not part of the structural model | 2026-10-10 |

## D4 in more detail
**Original idea (2026-10-10):** approximate betweenness from sampled source nodes, because we assumed exact betweenness would be too slow on 19.7k nodes.

**What we measured in step 3 (2026-10-23):**
- Exact betweenness with networkx: 161 s per graph. With igraph: under 1 s.
- Only 1,807 of the 19,717 papers have betweenness above zero (the graph is a DAG and most papers are sources or sinks).
- The sampled version is a poor copy of the exact one: Spearman with exact = 0.22 (k=100), 0.55 (k=500), 0.63 (k=1000), 0.70 (k=2000), and it finds only 535 of the 1,807 non-zero papers at k=500.
- Sampling noise would also show up as "instability" and could be mistaken for the effect we are trying to measure.

**Decision:** compute exact betweenness on every graph with igraph. It is cheap, exact and removes one source of noise. igraph is listed as an option in Phase A. The igraph result is checked against networkx in `tests/test_features.py`.
