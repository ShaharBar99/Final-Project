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
