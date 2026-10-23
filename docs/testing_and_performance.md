# Testing and performance (step 10)

How the Phase A testing plan (Section 7.2) is covered. Run everything with `pytest`
(about 1.5 minutes), or only the fast tests with `pytest -m "not slow"` (about 10 seconds).

## Tests

| Phase A item | Where | What is checked |
|---|---|---|
| Unit - dataset cleaning | `tests/test_cleaning.py` | duplicates, self-loops, mutual citations, isolated nodes; the cleaned real graph has no self-loops, duplicates or cycles |
| Unit - centralities and edge cases | `tests/test_features.py` | degree by hand; betweenness and closeness equal to networkx (also on a graph with cycles); a node nobody reaches gets closeness 0; k-core ignores self-loops; PageRank sums to 1 |
| Unit - scaler | `tests/test_features.py` | bounded in [0, 1] on the original graph; frozen and not clipped on new data |
| Unit - model, ranking, stability, core | `test_model.py`, `test_ranking.py`, `test_stability.py`, `test_core.py` | seed reproducibility, column order, no retraining; ties and floating-point noise; CV and correlations on hand-made cases; both core rules |
| Integration - data flow networkx -> pandas -> scikit-learn | `tests/test_pipeline.py` | whole pipeline on a 300-paper synthetic citation graph: all stages use the same papers in the same order, shapes are right, correlations are in [-1, 1], CV >= 0 |
| Integration - perturbations | `tests/test_perturb.py`, `tests/test_experiment.py` | node set unchanged, exact edge counts, old edges kept / only existing edges removed, add_dag stays acyclic, zero damage = identical scores |
| E2E - reproducibility | `tests/test_pipeline.py` | same seed -> bit-identical scores; different seed -> different damage; on the real data, the pipeline reproduces the saved base ranking and the demo reproduces run 0 of the experiment |
| E2E - core threshold | `tests/test_pipeline.py` | every core paper is important and at or below the threshold, and no other paper is core |
| E2E - app | `tests/test_pipeline.py` | the Streamlit app runs without an exception, also after changing the damage kind |
| Performance and load | `scripts/profile_pipeline.py` | see below |

**Bug found by these tests:** the same seed did not give bit-identical forest scores (differences of ~1e-16). The forest predicted on several threads, and the 200 tree outputs were added up in a different order each time. Prediction now runs on one thread (`src/model.py`, +0.1 s per graph). Rankings were never affected, because scores are rounded to 12 significant digits before ranking (D13); the saved experiment results stay valid (checked on run 0 of every kind at 10 %: largest relative difference to the new code is 5e-16).

## Performance (`python scripts/profile_pipeline.py`, results in `results/profiling/`)

Machine: the development laptop (Windows 11, Python 3.14). Numbers will differ elsewhere, the proportions should not.

**Stages on the full graph** (19,717 papers, 44,324 citations)

| Stage | Time |
|---|---|
| load cleaned graph | 0.13 s |
| compute the 5 features | 0.2 s |
| train both forests | 0.8 s |
| predict both forests | 0.4 s |
| perturb (any kind, 10 %) | 0.06 s |
| score a damaged graph - remove / add_dag | 0.6 s |
| score a damaged graph - add_random | 2.0 s (cycles make closeness and betweenness more work) |

The whole Python process uses about 350 MB at its peak, well within free Google Colab limits (~12 GB). The Colab-vs-local fallback planned in Phase A was not needed.

**Memory over K runs** (remove 10 %)

| K | Time | Score array | Memory growth |
|---|---|---|---|
| 1 | 0.7 s | 1 MB | 12 MB |
| 10 | 7.5 s | 11 MB | 15 MB |
| 30 | 23 s | 33 MB | 36 MB |

Memory grows only by the size of the stored scores - damaged graphs are freed after each run, there is no leak.

**Scalability** (random induced subgraphs, 10 % damage, time per graph)

| Papers | Edges | remove | add_dag | add_random |
|---|---|---|---|---|
| 4,929 | 2,887 | 0.08 s | 0.08 s | 0.08 s |
| 9,858 | 11,160 | 0.25 s | 0.21 s | 0.21 s |
| 14,787 | 24,405 | 0.36 s | 0.49 s | 0.58 s |
| 19,717 | 44,324 | 0.61 s | 0.70 s | 2.17 s |

Time grows roughly with the number of edges, except add_random, which grows faster because random edges create large citation cycles. The full experiment (270 graphs) takes about 6 minutes.
