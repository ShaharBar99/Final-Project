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

## Step 7 - Scoring the perturbed graphs (2026-10-23)
- **Goal:** for every setting (remove / add_random / add_dag x 5 / 10 / 20 %) and K = 30 runs: perturb -> recompute features -> frozen scaler -> both forests without retraining; keep all 7 methods' scores.
- **Validation before the full run:**
  - Scoring the original graph through the new code must reproduce the step 5 rankings exactly. **It did not** at first: 4 methods had small differences (up to 135 rank places for the PageRank forest). Cause: values that should be tied differed by ~1e-16 (floating point), and scaling split those ties differently. Fixed by rounding to 12 significant digits before ranking (D13, new test); now all 7 rankings match exactly. Step 5 was re-run; its tie counts moved by at most 4 distinct values.
  - The "out of range" check flagged values on the original graph itself - again floating point (max scaled to 1.0000000000000002). Added a 1e-9 tolerance.
  - Two of the new tests failed on the first try: one exposed the bug above, the other was too strict for a 119-paper toy graph and was replaced by an exact check of the unit conversion.
- **Score units (D12):** natural units for the CV; see decisions.md.
- **Run:** 270 graphs in ~6 minutes (0.5 s per graph, 5.8 s for add_random at 20 % because of cycles). Raw scores in `results/runs/` (not on GitHub, ~150 MB, rebuilt by notebook 06).
- **Extrapolation check (out-of-range inputs):** after edge *addition*, 42-60 % of papers get a scaled PageRank below the original minimum (adding edges lowers the smallest PageRank values), and with add_random 20 % also 60 % of closeness values exceed the original maximum. After *removal*, up to 10 % of degree / k-core values drop below the minimum. The forest cannot extrapolate - it gives all these papers the edge-of-range answer. This becomes important in step 8.
- **Code:** `src/experiment.py`, rounding in `src/ranking.py`, 5 new tests, notebook `notebooks/06_run_experiment.ipynb`.

## Step 8 - Stability metrics (2026-10-23)
- **Goal:** Phase A metrics on the 270 runs: Spearman / Kendall vs the base ranking (mean, std, ranking instability), node-level CV, instability-profile correlation; plus Spearman over each method's top 10 % and top-k retention (D15).
- **Validation:** unit tests on hand-made cases (CV, zero mean, perfect / reversed rankings, top-k, "no change = perfect"); all 9 settings are checked to use the same paper order as the base scores; checked that add_random and add_dag really differ (0 shared new edges) even though degree / k-core give near-identical numbers for both - expected, since both add uniformly random pairs and degree ignores direction. Ranking instability (std / mean of Spearman over the 30 runs) is <= 0.017 everywhere, so K = 30 is enough.
- **Main findings** (forest = the one *without* PageRank input, the real test of Phase A):
  1. **Global ranking (Spearman, all papers):** the forest is *never* the most stable method. Degree and k-core are the most stable under addition (0.94-0.86), betweenness under removal (but see 3). The forest is close to PageRank: worse under removal (0.83 vs 0.86 at 20 %), slightly better under addition from 10 % up (add_dag 20 %: 0.72 vs 0.67).
  2. **Top of the ranking:** under edge addition the forest keeps its top much better than PageRank: top-1000 retention 0.82 vs 0.52 (add_dag 5 %), 0.60 vs 0.39 (add_dag 20 %); mean CV of its top 10 % is 0.06-0.10 vs 0.18-0.31 for PageRank. Under removal they are equal. Degree, k-core and closeness are still better at the top.
  3. **Ties make betweenness look stable under removal** (Spearman 0.99) - 91 % of papers have betweenness 0 in every run. Over its top 10 % it drops to 0.86-0.96, and its top-10 % CV is the worst of all methods (0.29-2.7).
  4. **Saturation is real:** under add_random the forest's top-10 % CV *falls* as damage grows (0.16 -> 0.13 -> 0.06), the same happens to closeness (0.30 -> 0.06). Cycles inflate closeness above the training range, the trees return one edge value for all those papers, so the scores stop moving - "stable" because the model is blind there, not because it is robust. The with-PageRank forest is *less* stable than PageRank itself under addition (0.54 vs 0.67 at add_dag 20 %) for the same reason: half of the PageRank inputs fall below the training range and get lumped together.
  5. **Instability profiles:** the forest's CV pattern follows PageRank's (Spearman 0.45-0.69) more than any input feature's, except closeness under add_random 20 % (0.94 - the saturation case). It is almost unrelated to degree, k-core and betweenness instability (-0.4 to 0.26).
- **Output:** `results/tables/` (ranking_stability.csv, cv_summary.csv, instability_profiles.csv, per_run_correlations.csv), `results/spearman_vs_level.png`.
- **Code:** `src/stability.py`, `src/analysis.py`, 7 new tests, notebook `notebooks/07_stability.ipynb`.

## Step 9 - Network core and hypotheses (2026-10-23)
- **Goal:** answer the 5 research questions and the 3 hypotheses of Phase A.
- **Problem found in the Phase A core rule:** "stable = CV <= 10th percentile of all papers" gives an empty core under removal for the forest, PageRank and closeness - the threshold is 0 and the only papers below it are at the bottom of the ranking (mean base rank ~18,700). Kept as a finding; added an adjusted rule that judges stability among the top-10 % papers (D16).
- **New measures:** for each important paper, the share of runs in which it stays in the top 10 %; "fragile important" = papers that drop out in more than half of the runs; Jaccard overlap of cores between settings.
- **Validation:** unit tests for both core rules and for stay-in-top; checked that the adjusted core really picks robust papers (core papers stay in the top more often than the other important papers in every setting); figures checked visually (legend moved off the bars); every number in results_summary.md checked against the tables.
- **Verdicts:** H1 partly supported (better than PageRank under addition, not under removal, never better than degree / k-core); H2 not supported as stated (a forest that leans on one feature collapses with it); H3 supported with the adjusted rule. Full answers: `docs/results_summary.md`.
- **Code:** `src/core.py`, `core_analysis` / `core_consistency` in `src/analysis.py`, 4 tests (50 passing), notebook `notebooks/08_core_and_hypotheses.ipynb`.
- **Output:** `results/tables/core_summary.csv`, `results/importance_vs_cv.png`, `results/top10_retention.png`.

## Demo notebook (2026-10-23)
- **Goal:** show the whole system on one damaged graph in about a second, for the presentation (task 8).
- **What it does:** choose kind / level / seed -> damage the graph -> recompute features, frozen scaler, trained forests -> Spearman / Kendall / top-100 kept for all 7 methods, before / after table of the top-20 papers, rank-before vs rank-after plot, and a zoom on the top-100 paper that fell the most, with its features before / after.
- **Validation:** the demo's scores for (add_dag, 10 %, seed 42) are identical to run 0 of the same setting in the full experiment.
- **Example found by the demo:** paper 15004560 falls from rank 67 to 1,612 (forest) after a single new outgoing citation - its betweenness jumps from 0 to ~0.1, and the forest had learned "cited a lot, betweenness 0" as a sign of an important classic paper. Explained in the notebook.
- **Code:** `src/demo.py` (shared with the app), `notebooks/09_demo.ipynb`.

## Interactive app (2026-10-23)
- **Goal:** an interactive version of the demo for the presentation (task 8).
- **What it is:** a Streamlit app, `app/app.py`, run with `streamlit run app/app.py`. Sidebar: kind of damage, level (1-30 %), seed, method. Tabs: *Damage the graph* (graph change, stability of every method, top-20 before / after, rank scatter), *Look up a paper* (ranks in all methods and features before / after for any PubMed ID), *Experiment results* (the 270-run results, charts and the full write-up), *How it works*.
- **Validation (clicked through in a browser):** numbers match the demo notebook for the same settings; removal 10 % shows 39,892 citations and 943 isolated papers, the same as step 6; an unknown ID shows a warning.
- **Bugs found while testing and fixed:** the results charts sorted the damage levels as text ("10 %", "20 %", "5 %"), which drew a fake V shape - replaced by a matplotlib chart with numeric levels; the default method was the with-PageRank forest instead of the without-PageRank one; metric labels were cut off; Streamlit's deprecated `use_container_width` replaced with `width`.
- `.claude/launch.json` has a run configuration for the app; `streamlit` and `psutil` added to `requirements.txt`.

## Step 10 - Testing and performance (2026-10-23)
- **Goal:** complete the Phase A testing plan - integration, end-to-end, reproducibility, core-threshold, app and performance tests.
- **New tests** (`tests/test_pipeline.py`, 10 tests, 60 in total): the whole pipeline on a synthetic citation graph; same seed -> identical output; core threshold; real data reproduces the saved base ranking and the experiment; the Streamlit app runs. Real-data tests are marked `slow` (`pytest -m "not slow"` skips them).
- **Bug found:** same seed did not give bit-identical forest scores (~1e-16) because prediction ran on several threads. Fixed by predicting on one thread. Rankings were never affected (D13); saved results match the new code to 5e-16.
- **Also fixed:** the app used matplotlib's GUI backend, which warns when drawing from a server thread; it now uses the non-GUI "Agg" backend.
- **Performance** (`scripts/profile_pipeline.py`): ~0.6 s per damaged graph (2 s for add_random), peak ~350 MB, memory grows only by the stored scores (no leak), runtime grows roughly with the number of edges. Details: `docs/testing_and_performance.md`, raw numbers: `results/profiling/`.

## User guide and maintainer guide (2026-10-23)
- **Goal:** tasks 5 and 6 - `docs/user_guide.md` (install, quick start, notebooks, app, how to read the results, settings, troubleshooting) and `docs/maintainer_guide.md` (architecture, data formats, rules the code relies on, how to add a measure / damage kind / model / dataset, tests, environment, pitfalls).
- **Validation:** cloned the repository fresh from GitHub into an empty folder - all 60 tests pass, the demo notebook runs, and the automatically rebuilt forests give bit-identical scores to the local ones. All timings quoted in the guides were measured (notebooks 6-25 s each, tests ~20 s).
- **Added:** `docs/review_before_sharing.md` - the changes to the Phase A plan, the findings and the details the team and the advisor should review before the work is shared.
