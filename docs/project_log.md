# Project log (macro view)

One entry per step: what we did, why, and where the result is.
Newest at the bottom.

## Step 0 - Setup (2026-10-10)
- Read Phase A and the PubMed-Diabetes files, wrote `docs/PLAN.md`.
- Team decisions: PubMed instead of Cora; PageRank run both as feature and without it; structural (not date-based) consistency check; approximate betweenness.
- Created the folder layout, `requirements.txt`, `.gitignore`, `src/config.py`.
- Moved the raw files to `data/raw/` (untouched) and the paper to `docs/PhaseA.docx`.

## Step 1 - Data cleaning (2026-10-10)
- **Goal:** a clean directed citation graph, ready for centrality computation.
- **Input:** `data/raw/` - 19,717 papers, 44,338 citations.
- **What changed:** 3 self-loops removed, 11 mutual-citation edges removed. Nothing else needed fixing.
- **Output:** `data/processed/nodes.csv` (19,717), `edges.csv` (44,324), `cleaning_log.json`.
- **Result of checks:** 1 connected component, no cycles (the graph is a DAG), no isolated papers.
- **Code:** `src/data.py`, tests in `tests/test_cleaning.py` (5 passing), notebook `notebooks/01_clean_data.ipynb`.
- Details of every action: `docs/cleaning_report.md`. Assumptions: `docs/decisions.md`.

## Step 2 - Decomposition (2026-10-10)
- **Goal:** split the Phase A model into small blocks with clear inputs/outputs before writing the heavy code.
- **Output:** `docs/decomposition.md` (7 blocks A-G, experiment grid, 7 precise rules for ambiguous points such as closeness on a directed graph and the frozen scaler) and the matching empty skeleton in `src/` (`features.py`, `scaling.py`, `model.py`, `ranking.py`, `perturb.py`, `stability.py`, `core.py`). The functions are placeholders that raise `NotImplementedError("step N")` and are filled in by the step named in the message.
- No data was touched in this step.

## Step 3 - Centrality features and frozen scaler (2026-10-23)
- **Goal:** the 5 features (degree, betweenness, closeness, PageRank, k-core) for every paper, and the Min-Max scaler fitted once on the original graph.
- **Change of plan:** timing showed exact betweenness is cheap with igraph (under 1 s; networkx needs 161 s), while the sampled version was inaccurate and would add its own noise. So betweenness is exact everywhere (decision D4 updated).
- **Findings on the real graph:**
  - 1,807 papers (9 %) have betweenness above zero; 2,047 papers (the never-cited ones) have closeness 0.
  - Degree and k-core are almost the same ranking (Spearman 0.97) - the forest gets little new from having both.
  - PageRank is correlated with closeness (0.73) but weakly with degree (0.29).
  - PageRank has many near-identical values, so rankings will contain many ties (handled with average ranks in step 5).
- **Output:** `data/processed/features_raw.csv`, `features_scaled.csv`, `scaler.joblib`.
- **Code:** `src/features.py`, `src/scaling.py`, 9 new tests (14 passing in total), notebook `notebooks/02_features.ipynb`.
