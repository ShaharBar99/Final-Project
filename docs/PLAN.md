# Phase B – Implementation Plan

Project: Stability and Structural Validity of Learned Node Importance in Citation Networks
Data: PubMed-Diabetes (19,717 papers, 44,338 directed citations)

## Working rules
- Every step = one notebook (or module) + one commit + push to GitHub.
- Fixed random seeds everywhere; all parameters live in `config.py` / `config.yaml`.
- Logic lives in `src/` (importable, pytest-tested); notebooks only call it. This makes the Colab -> VS Code move trivial.
- Notebooks must run top-to-bottom in Colab (clone repo in first cell).

## Repo layout
```
data/raw/            original PubMed files (git-lfs or excluded if too large)
data/processed/      cleaned edge list + node list (parquet/csv)
src/                 data.py, centralities.py, model.py, perturb.py, stability.py, config.py
notebooks/           01_clean ... 07_core_analysis
tests/               pytest
results/             tables, figures
docs/                user_guide.md, maintainer_guide.md, poster/
```

## Steps (map to lecturer's tasks)

| # | Step | Output | Task |
|---|------|--------|------|
| 0 | Repo setup, skeleton, requirements, .gitignore | first commit | – |
| 1 | **Clean DB**: parse the 2 files -> edge list; drop 500 TF-IDF feature columns (not used by our model); remove self-loops (3), duplicates (0), isolated nodes (0); handle 12 reciprocal pairs (violate time-acyclicity); keep largest component or document the decision; export cleaned graph + cleaning report | `data/processed/*`, report | 1 |
| 2 | **Decompose**: formalise pipeline (see below) into functions with signatures + unit tests | `src/` skeleton, tests | 2 |
| 3 | Centralities: degree, betweenness (approximate with k-sampling or igraph for speed), closeness (directed/disconnected convention), PageRank, k-core. Min-Max scaler fitted ONCE on original graph (frozen) | feature table | 2 |
| 4 | **Random Forest**: X = 5 normalised features, y = PageRank; train on original graph | trained model + sanity metrics | 3 |
| 5 | **Base ranking**: RF scores -> global ranking; baseline rankings from each single centrality | `base_ranking.csv` | 4 |
| 6 | Perturbation engine: random edge removal / addition at 5/10/20 %, K repetitions, nodes fixed | perturbed-graph generator | – |
| 7 | Re-compute features on each perturbed graph, frozen scaler, apply RF without retraining | score matrix (nodes x K) per method per level | – |
| 8 | Stability metrics: node-level CV, Spearman/Kendall vs base ranking, correlation of instability profiles | tables | – |
| 9 | Network core (e.g. 10th-percentile CV x high importance), importance-vs-stability quadrants, test the 3 hypotheses | figures + conclusions | – |
| 10 | pytest suite, reproducibility test, memory/runtime profiling | green tests | – |
| 11 | User guide, maintainer guide | `docs/` | 5, 6 |
| 12 | Poster + demo notebook / small app (e.g. Streamlit or Colab demo) | `docs/poster`, demo | 7, 8 |

## Decisions needed from the team (they affect the design)
1. **Dataset**: Phase A says Cora; the provided data is PubMed-Diabetes. Confirm PubMed is final.
2. **PageRank is both a feature and the target** (Phase A design). The RF can then just copy that column, which makes "ML beats PageRank on stability" trivial/misleading. Options: (a) keep as in paper and discuss it, (b) drop PageRank from the features, (c) run both and compare. Suggested: (c).
3. **Temporal validation**: the dataset has no publication years, so "validation of temporal consistency" can only be done structurally (cycles / reciprocal edges). OK?
4. **Betweenness cost**: first assumed to need approximation; later measured and replaced by exact igraph betweenness (see decisions.md, D4).
