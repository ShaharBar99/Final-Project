# Decisions and assumptions

| # | Decision | Reason | Date |
|---|---|---|---|
| D1 | Use PubMed-Diabetes instead of Cora | Dataset provided for the project (Phase A mentioned Cora as preliminary) | 2026-10-10 |
| D2 | Run the model both with and without PageRank as an input feature | PageRank is also the target; with it as a feature the forest can just copy it | 2026-10-10 |
| D3 | Temporal consistency checked structurally (no cycles), not by year | No publication dates in the data | 2026-10-10 |
| D4 | Betweenness will be approximated (sampled sources, fixed seed) | See below | 2026-10-10 |
| D5 | Mutual citations: keep the edge from the higher PubMed ID to the lower | PubMed IDs grow over time, so higher ID ~ newer paper. This is an assumption; 11 edges out of 44k, so the effect is tiny | 2026-10-10 |
| D6 | Drop the 500 word-feature columns | Not part of the structural model | 2026-10-10 |

## D4 in more detail
Exact betweenness needs a shortest-path search from every node (about 19.7k searches) and we need it for many perturbed graphs. The approximation runs the search from only k randomly chosen source nodes (e.g. k = 500) and scales the result up. **All 19,717 nodes still get a score** - only the sources are sampled, not the papers we rank or train on. The same k and seed are used for every graph so the numbers stay comparable.
