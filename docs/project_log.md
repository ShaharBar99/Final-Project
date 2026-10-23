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

## Step 4 - Random Forest (2026-10-23)
- **Goal:** one model that turns the five centralities into an importance score; target = PageRank on the original graph (Phase A). Trained twice: with and without PageRank as an input (D2).
- **Results (80/20 hold-out, on papers the forest never saw):**
  - With PageRank as input: R^2 = 0.999, Spearman 1.000. The forest puts 99.8 % of its importance on the PageRank column, i.e. it just copies it.
  - Without PageRank: R^2 = 0.75, Spearman 0.80. Importance is split mainly between closeness (0.47) and degree (0.44).
  - For comparison, the best single centrality (closeness) follows PageRank with Spearman 0.73; degree 0.29, k-core 0.37, betweenness 0.19. So the forest combines the features into something closer to PageRank than any single one.
- **Consequence for the experiment:** the "with PageRank" model will behave almost exactly like PageRank in the stability tests. The "without PageRank" model is the real test of the Phase A idea. We keep both and report both.
- **Not tuned on purpose:** we want a reasonable model, not the best fit; the question is about stability.
- **Code:** `src/model.py`, 4 new tests (18 passing), notebook `notebooks/03_model.ipynb`. Output: `data/processed/model_report.json`. Models themselves are rebuilt locally and not committed (D8).

## Step 5 - Base ranking (2026-10-23)
- **Goal:** the reference ranking of every method on the original graph (7 methods: 2 forests + 5 single centralities). Perturbed rankings are compared against this in step 8.
- **Tie rule (D9):** rank 1 = most important, tied papers share the average rank.
- **Findings:**
  - Ties are everywhere. Share of papers that sit in a tie: k-core 100 % (only 10 distinct values), degree 99.9 % (82 values), betweenness 94 %, closeness 88 %, forest without PageRank 79 %, PageRank 66 %, forest with PageRank 65 %. Coarse methods cannot separate most papers at all, so their rank correlations will be low-resolution.
  - The forest with PageRank has the same ranking as PageRank (Spearman 1.00, identical top 100) - confirms step 4.
  - The forest without PageRank agrees most with closeness (0.88) and PageRank (0.81), and less with degree (0.39). Top-100 overlap with PageRank: 65 papers.
  - Betweenness is the odd one out: almost no overlap in the top 100 with anything (3 with PageRank).
- **Code:** `src/ranking.py`, `load_or_train` added to `src/model.py`, 4 new tests (22 passing), notebook `notebooks/04_base_ranking.ipynb`.
- **Output:** `data/processed/base_scores.csv`, `base_ranking.csv`.

## Step 6 - Perturbation engine (2026-10-23)
- **Goal:** make damaged copies of the graph - edge removal and edge addition at 5 / 10 / 20 % - with the node set fixed and every run reproducible on its own.
- **Two addition variants (D10):** `add_random` exactly as in Phase A, and `add_dag` which only adds "newer -> older" edges. Before building `add_dag` we checked the PubMed-ID idea: only 89.6 % of real citations go from higher to lower ID, so we used a topological order of the cleaned graph instead (agrees with 100 % of citations, guarantees no cycles).
- **Validation on the real graph (one run per setting, notebook 05):**
  - node set unchanged in all 9 settings; exact edge counts (-/+ 2,216 / 4,432 / 8,865);
  - `remove` deletes only existing edges; it leaves 469 / 943 / 1,981 papers with no edges at all (they stay in the graph);
  - both addition kinds keep all original edges; no self-loops or duplicate edges;
  - `add_dag` stays acyclic; `add_random` creates cycle groups of 328 / 901 / 2,498 papers;
  - the same (seed, kind, level, run) always gives the same graph.
- **Problem found and fixed:** on `add_random` graphs networkx closeness took ~20 s per graph. Closeness moved to igraph with the same formula (D11); a new test checks it equals networkx on a graph with cycles, and the original-graph features are identical to step 3. Feature time now: 0.5 s (remove), 1 s (add_dag), 5 s (add_random - more papers reach each other, so there is simply more work).
- **Code:** `src/perturb.py`, `src/features.py` (closeness), 11 new tests (33 passing), notebook `notebooks/05_perturbation.ipynb`.
