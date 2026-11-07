# Stability and Structural Validity of Learned Node Importance in Citation Networks

Braude College of Engineering - Final Project, Phase B.
Question: does a Random Forest trained to approximate PageRank from four other centralities give rankings that are more stable under edge noise than PageRank or the five centrality baselines? The no-PageRank forest is the primary scientific model; the forest that also receives its PageRank target is retained only as a direct-leakage control.

## Layout
- `data/raw/` original PubMed-Diabetes files, `data/processed/` cleaned graph
- `src/` the code, `notebooks/` one notebook per step, `tests/` pytest
- `docs/` paper (Phase A), plan, project log, decisions, cleaning report

## Quick start

Python 3.12.x (64-bit) is the supported local version. In Windows Git Bash:

```
python --version
python -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest -m "not slow" -q
python -m pytest -q
```

The first test command verifies the code without the real-data/app tests. Run the
complete command after it passes; on a fresh clone, the real-data tests may build
the ignored model artifacts because they are not stored in Git.
Then open the notebooks in order (also works in Google Colab: the first cell clones the repo).

## Status
See `docs/project_log.md`. Done: step 1 (cleaning), step 2 (decomposition), step 3 (features + frozen scaler), step 4 (Random Forest), step 5 (base ranking), step 6 (perturbation engine), step 7 (scores on 270 perturbed graphs), step 8 (stability metrics), step 9 (network core + hypotheses). Results: `docs/results_summary.md`. Step 10 (tests + profiling): `docs/testing_and_performance.md`. Demo: `notebooks/09_demo.ipynb`, app: `streamlit run app/app.py`. Guides: `docs/user_guide.md`, `docs/maintainer_guide.md`. Before sharing: `docs/review_before_sharing.md`. Poster: `poster/poster.pdf` (A0).
