# Maintainer guide

For someone who will change, extend or debug the code. For running the system and reading its results, see `docs/user_guide.md`.

Related documents:
- `docs/decisions.md` - every design decision (D1-D16) with its reason. Read it before changing behaviour; most "strange" choices in the code are explained there.
- `docs/project_log.md` - what was done in each step, including the bugs that were found.
- `docs/decomposition.md` - how the Phase A model was split into blocks.
- `docs/testing_and_performance.md` - what the tests cover, runtime and memory numbers.

---

## 1. Architecture

The Phase A model is split into seven blocks (A-G). Each block is one module in `src/`; the notebooks only call these modules and show the results.

```
data/raw  --(data.py)-->  cleaned graph G0
                              |
        [A] features.py   five centralities of any graph
        [B] scaling.py    Min-Max, fitted ONCE on G0, then frozen
        [C] model.py      Random Forest: scaled features -> score (target = PageRank of G0)
        [D] ranking.py    scores -> ranks (rounding + average ranks for ties)
                              |
        [E] perturb.py    G0 -> damaged copy (remove / add_random / add_dag)
            experiment.py for every damaged copy: A -> B (frozen) -> C (no retraining)
                              |
        [F] stability.py  CV per paper, Spearman / Kendall, top-k kept
        [G] core.py       network core, node roles
            analysis.py   runs F and G over every saved setting, builds the result tables
            demo.py       one damaged graph end to end (used by notebook 09 and the app)
```

| Module | Main functions | Notes |
|---|---|---|
| `config.py` | `ROOT`, `RAW_DIR`, `PROCESSED_DIR`, `RESULTS_DIR`, `SEED` | all paths come from here - never hard-code a path |
| `data.py` | `clean_dataset()`, `save_dataset()`, `load_clean_graph()` | each cleaning step appends what it did to a `log` list |
| `features.py` | `compute_features(g)` | rows follow `g.nodes`; betweenness and closeness via igraph |
| `scaling.py` | `fit_scaler()`, `apply_scaler()` | no clipping, on purpose |
| `model.py` | `train_model()`, `predict_scores()`, `holdout_check()`, `load_or_train()` | prediction is single-threaded, on purpose |
| `ranking.py` | `to_ranking()`, `round_scores()`, `base_scores()`, `tie_share()` | **always** rank through `to_ranking()` |
| `perturb.py` | `perturb()`, `run_rng()`, `time_order()`, `KINDS` | never modifies the input graph |
| `experiment.py` | `score_graph()`, `run_setting()`, `run_all()`, `load_setting()`, `METHODS`, `LEVELS` | scores are stored in natural units |
| `stability.py` | `evaluate_setting()`, `node_instability()`, `rank_correlations()`, `top_k_retention()` | |
| `core.py` | `node_roles()` (Phase A rule), `node_roles_within_important()` (adjusted rule), `stay_in_top_share()` | |
| `analysis.py` | `evaluate_all()`, `ranking_table()`, `core_analysis()`, `core_consistency()` | reads `results/runs/` |
| `demo.py` | `load_pipeline()`, `damage_and_score()`, `method_summary()`, `rank_changes()` | shared by notebook 09 and `app/app.py` |

---

## 2. Files and data formats

| File | Format | Made by | In git? |
|---|---|---|---|
| `data/raw/*.tab*` | original PubMed-Diabetes files (LINQS format) | - | yes, never modified |
| `data/processed/nodes.csv` | `paper_id, label` | notebook 01 | yes |
| `data/processed/edges.csv` | `source, target` - source cites target | notebook 01 | yes |
| `data/processed/cleaning_log.json` | list of `{step, removed, ...}` | notebook 01 | yes |
| `data/processed/features_raw.csv`, `features_scaled.csv` | `paper_id` + 5 feature columns | notebook 02 | yes |
| `data/processed/scaler.joblib` | fitted scikit-learn `MinMaxScaler` | notebook 02 | yes |
| `data/processed/models/rf_*.joblib` | trained forests, 18-33 MB each | notebook 03 / `load_or_train` | **no** - rebuilt in seconds, identical (fixed seed) |
| `data/processed/base_scores.csv`, `base_ranking.csv` | `paper_id` + 7 method columns | notebook 04 | yes |
| `results/runs/<kind>_<level>.npz` | `scores` array (K runs x papers x 7 methods), `nodes`, `methods` | notebook 06 | **no** - ~150 MB, ~6 min to rebuild |
| `results/runs/<kind>_<level>_info.json` | per run: edges, seconds, share of inputs out of range | notebook 06 | no |
| `results/runs/base_scores.csv` | base scores in natural units, same paper order as the npz files | notebook 06 | no |
| `results/tables/*.csv` | final result tables | notebooks 07-08 | yes |
| `results/profiling/*.csv` | performance numbers | `scripts/profile_pipeline.py` | yes |

**Paper IDs are strings everywhere.** When reading a CSV, pass `dtype={"paper_id": str}`, otherwise pandas turns them into integers and lookups by ID fail.

**Paper order.** All arrays use the order of `g.nodes` of the cleaned graph. `analysis.evaluate_all()` asserts that the saved runs and the base scores use the same order; keep it that way.

---

## 3. Rules the code relies on

Break one of these and the results quietly become wrong - the tests catch most, but not all.

1. **The scaler is fitted once, on the original graph.** Never call `fit_scaler` on a damaged graph. Values outside [0, 1] after damage are expected (Phase A requirement 12, "frozen scaler").
2. **The forests are never retrained on a damaged graph** (Phase A requirement 13). `experiment.score_graph()` only calls `predict_scores()`.
3. **Always rank with `ranking.to_ranking()`.** It rounds scores to 12 significant digits first. Without rounding, ties that differ only by floating-point noise (~1e-16) are broken at random and show up as fake rank changes (D13). Do not call `pandas.rank` or `scipy.stats.rankdata` directly on scores.
4. **The coefficient of variation is computed on scores in natural units** - raw centralities, and forest predictions converted back to PageRank units (`experiment.to_pagerank_units`). Never compute it on Min-Max-scaled values: CV changes when values are shifted (D12).
5. **Every random choice goes through a seeded generator.** For damage, use `perturb.run_rng(seed, kind, level, run)`; it makes each run reproducible on its own. Do not use `random` or `np.random` global functions.
6. **Forest prediction is single-threaded.** With several threads the sum over the trees changes in the 16th digit from call to call, which breaks bit-identical reproducibility (found in step 10). Training can stay multi-threaded.
7. **Functions do not modify their input graph.** `perturb.*` returns a copy.

---

## 4. Common changes

### Add a new centrality measure

1. In `src/features.py`, add the column in `compute_features()` and its name to `FEATURE_NAMES`.
2. Add it to the forest inputs in `src/model.py` (`FEATURES_WITH_PAGERANK`, `FEATURES_WITHOUT_PAGERANK`) if the forest should use it.
3. Add a test in `tests/test_features.py` - if possible compare with a reference implementation (as betweenness and closeness are compared with networkx).
4. Add a label in `src/demo.py` - `METHOD_LABELS`.
5. Re-run notebooks 02-08 and delete `data/processed/models/` first (the forests' inputs changed).

The new measure automatically becomes a baseline method, because `experiment.METHODS` is built from `FEATURE_NAMES`.

If the new measure is slow, check it on an `add_random` graph: random edges create large citation cycles, and path-based measures become much slower there (closeness went from 0.5 s to 20 s in networkx, D11).

### Add a new kind of damage

1. In `src/perturb.py`, write a function that returns a damaged **copy** with the **same nodes**, add a branch in `perturb()`, and append the name to the **end** of `KINDS` (the position is part of the random seed - inserting in the middle changes the damage of the existing kinds).
2. Add tests in `tests/test_perturb.py` (the parametrized "nodes never change" test picks it up automatically).
3. Add a label in `src/demo.py` - `KIND_LABELS`; the app shows it automatically.
4. Run notebook 06 (only the new setting is computed; existing ones are skipped), then 07-08.

### Add another model (for example a neural network)

1. Train it in `src/model.py` (or a new module) with the same interface: fitted on the scaled features of the original graph, `predict` on any scaled feature table.
2. In `src/experiment.py`, `score_graph()` currently loops over `forests`; add the new model there with a new method name, convert its output to natural units, and add the name to `METHODS`.
3. Make prediction deterministic (rule 6).

### Use a different dataset

1. Write a loader that returns the same two tables as `data.load_nodes()` (`paper_id`, `label`) and `data.load_edges()` (`source`, `target`, meaning source cites target). Paper IDs as strings. If there is no label, use a constant.
2. Point `config.NODES_FILE` / `EDGES_FILE` to the new files, or add new paths.
3. Run all notebooks from 01. Check the cleaning log: `resolve_mutual_citations` assumes that a higher ID means a newer paper (D5). This holds for PubMed IDs only roughly (89.6 % of citations); for another dataset check it, or use publication years if the dataset has them.
4. `add_dag` needs the cleaned graph to have no cycles. `perturb.time_order()` fails on a graph with cycles - clean them first.

### Change the experiment size

`LEVELS` in `src/experiment.py`, `k` in notebook 06. Delete `results/runs/` or call `run_all(..., redo=True)`; otherwise old settings are skipped.

---

## 5. Tests

```
pytest                    all 60 tests (~20 s)
pytest -m "not slow"      skip the 3 tests that need the full dataset or start the app (~6 s)
pytest tests/test_perturb.py -v     one file
```

- `tests/test_pipeline.py` runs the whole pipeline on a small synthetic citation graph - the quickest way to check that a change did not break the flow between modules.
- The `slow` tests compare the code with the saved data: if you change something that *should* change results (a feature, the scaler, the forest), these tests will fail until you re-run the notebooks and commit the new files. That is intended.
- The test that compares the demo with the saved experiment is skipped automatically when `results/runs/` does not exist.

Run the tests before every commit.

---

## 6. Rebuilding everything from scratch

```
jupyter nbconvert --to notebook --execute --inplace notebooks/01_clean_data.ipynb
... the same for 02 to 09, in order ...
python scripts/profile_pipeline.py        (optional, ~3 min)
```

Notebook 06 needs `--ExecutePreprocessor.timeout=1800` because it runs for about 6 minutes. Everything is deterministic: with the same code and package versions, the same files come out.

---

## 7. Environment

Tested with Python 3.14.7 on Windows 11 and these versions:

| Package | Version | Used for |
|---|---|---|
| networkx | 3.7 | graph structure, degree, PageRank, k-core, topological order |
| igraph | 1.0.0 | betweenness and closeness (fast C implementation) |
| pandas | 3.0.6 | tables |
| numpy | 2.5.3 | arrays |
| scikit-learn | 1.9.1 | Min-Max scaler, Random Forest |
| scipy | 1.18.1 | Spearman, Kendall |
| matplotlib | 3.11.2 | figures |
| streamlit | 1.66.1 | the app |
| pytest | 9.1.1 | tests |
| psutil | 7.2.2 | memory profiling |

`requirements.txt` does not pin versions. If a future version changes results (for example a different PageRank tolerance in networkx), install the versions above.

The saved forests (`.joblib`) only load with the same scikit-learn version. They are not in git anyway; delete `data/processed/models/` and they are rebuilt.

---

## 8. Known pitfalls and lessons

| Pitfall | What happened | Where it is handled |
|---|---|---|
| Floating-point noise breaks ties | Equal PageRank values differed by ~1e-16 and were ranked apart at random; the step 7 code did not reproduce the step 5 ranking | `ranking.round_scores()` (D13) |
| Multi-threaded prediction is not reproducible | Same seed, scores differed by ~1e-16 | `model.predict_scores()` single thread |
| Ties inflate rank correlation | Betweenness looked like the most stable method (90 % of papers stay at 0) | top-10 % metrics in `stability.py` (D15) |
| Trees cannot extrapolate | After damage, inputs outside the training range all get the same forest output - looks "stable" | measured in `experiment.score_graph()` (`out_of_range`), reported in results |
| Random edges create citation cycles | `add_random` at 20 % made a group of 2,498 papers that all reach each other; closeness in networkx took 20 s per graph | `add_dag` variant (D10), igraph closeness (D11) |
| The Phase A core rule is empty under removal | Its CV threshold is 0; only bottom-ranked papers qualify | adjusted rule `core.node_roles_within_important()` (D16) |
| PubMed ID is only a rough proxy for time | 10 % of citations go from a lower to a higher ID | `time_order()` uses a topological order instead |
| Paper IDs read as integers | pandas reads `paper_id` as int; lookups by string fail | always `dtype={"paper_id": str}` |
| An apostrophe in a notebook string | `'method's'` broke a notebook cell | use double quotes for text with apostrophes |

---

## 9. Git

- Commit code, notebooks (with their outputs), small data files and result tables.
- Do not commit `data/processed/models/` or `results/runs/` (already in `.gitignore`).
- One commit per step, with an entry in `docs/project_log.md`, and a new row in `docs/decisions.md` for every decision that changes behaviour.
