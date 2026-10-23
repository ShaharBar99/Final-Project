# Stability and Structural Validity of Learned Node Importance in Citation Networks

Braude College of Engineering - Final Project, Phase B.
Question: does a Random Forest that combines several centrality measures give node rankings that are more stable under edge noise than any single measure?

## Layout
- `data/raw/` original PubMed-Diabetes files, `data/processed/` cleaned graph
- `src/` the code, `notebooks/` one notebook per step, `tests/` pytest
- `docs/` paper (Phase A), plan, project log, decisions, cleaning report

## Quick start
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pytest
```
Then open the notebooks in order (also works in Google Colab: the first cell clones the repo).

## Status
See `docs/project_log.md`. Done: step 1 (cleaning), step 2 (decomposition), step 3 (features + frozen scaler), step 4 (Random Forest), step 5 (base ranking), step 6 (perturbation engine), step 7 (scores on 270 perturbed graphs), step 8 (stability metrics), step 9 (network core + hypotheses). Results: `docs/results_summary.md`. Step 10 (tests + profiling): `docs/testing_and_performance.md`. Demo: `notebooks/09_demo.ipynb`, app: `streamlit run app/app.py`. Guides: `docs/user_guide.md`, `docs/maintainer_guide.md`. Before sharing: `docs/review_before_sharing.md`. Poster: `poster/poster.pdf` (A0).
