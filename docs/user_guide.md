# User guide

**Stability and Structural Validity of Learned Node Importance in Citation Networks**
Braude College of Engineering - Final Project, Phase B. Niran Czuckermann and Shahar Bar.

This guide is for someone who wants to *run* the system and *read its results*. If you want to change the code, see `docs/maintainer_guide.md`.

---

## 1. What the system does

The system takes a citation network (papers that cite each other) and ranks the papers by importance in seven ways:

| Method | In one sentence |
|---|---|
| Degree | how many citations a paper has (given + received) |
| Betweenness | how often a paper lies on the shortest path between two other papers |
| Closeness | how quickly the rest of the network reaches the paper through citations |
| PageRank | a paper is important if important papers cite it |
| k-core | how deep the paper sits inside a densely connected group |
| **Random Forest (without PageRank input)** | a machine-learning model that combines degree, betweenness, closeness and k-core into one score |
| Random Forest (with PageRank input) | the same, but it also gets PageRank as an input (the design of Phase A as written) |

Both forests are trained to predict PageRank on the original network.

Then the system **damages** the network on purpose - it deletes some citations or adds fake ones - and checks **how much each ranking changes**. A ranking that hardly changes is *stable*. The research question is whether the Random Forest ranks papers more stably than any single measure.

The data is the PubMed-Diabetes dataset: 19,717 papers about diabetes and 44,324 citations between them.

---

## 2. Installation

You need **Python 3.10 or newer** (the project was developed and tested with Python 3.14 on Windows 11) and **git**.

### On your own computer

```
git clone https://github.com/ShaharBar99/Final-Project.git
cd Final-Project
python -m venv .venv
.venv\Scripts\activate            (on macOS / Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

Check that everything works:

```
pytest
```

You should see 60 tests passing in about 20 seconds.

### On Google Colab

Open any notebook from the `notebooks/` folder in Colab (*File -> Open notebook -> GitHub*, paste the repository address). Run the first code cell: it downloads the project and installs the packages. Everything else works as on a local computer. The interactive app (section 5) does not run on Colab - use a local computer for it.

---

## 3. Quick start - see it work in one minute

**Option A - the app (recommended for a first look):**

```
streamlit run app/app.py
```

A browser tab opens at `http://localhost:8501`. The first load takes a few seconds. Section 5 explains the screens.

**Option B - the demo notebook:** open `notebooks/09_demo.ipynb` and run all cells. At the top you can change what kind of damage to apply and how much.

You do **not** need to run anything else first: the cleaned data, the features and the scaler are part of the repository, and the two trained models are rebuilt automatically (a few seconds) the first time they are needed.

---

## 4. The notebooks - the full pipeline step by step

The notebooks are numbered in the order they should run. Each one saves what the next one needs, so you can start at any step whose input already exists.

| Notebook | What it does | Time | Saves to |
|---|---|---|---|
| `01_clean_data` | reads the raw files, removes self-citations and impossible mutual citations, checks the graph | ~6 s | `data/processed/nodes.csv`, `edges.csv`, `cleaning_log.json` |
| `02_features` | computes the five centrality measures for every paper and fits the scaler | ~7 s | `features_raw.csv`, `features_scaled.csv`, `scaler.joblib` |
| `03_model` | trains the two Random Forests and checks them on held-out papers | ~8 s | `data/processed/models/`, `model_report.json` |
| `04_base_ranking` | ranks all papers with all seven methods on the undamaged network | ~6 s | `base_scores.csv`, `base_ranking.csv` |
| `05_perturbation` | shows and checks the three kinds of damage | ~7 s | nothing (checks only) |
| `06_run_experiment` | the main experiment: 270 damaged networks, every method scored on each | **~6 min** | `results/runs/` |
| `07_stability` | how much each ranking moved - tables and charts | ~20 s | `results/tables/`, `results/spearman_vs_level.png` |
| `08_core_and_hypotheses` | the "network core", the research questions and the hypotheses | ~25 s | `results/tables/core_summary.csv`, two figures |
| `09_demo` | one damaged network end to end, with explanations | ~10 s | nothing |

**Important:** the raw results of notebook 06 (`results/runs/`, about 150 MB) are too large for GitHub. After a fresh download, run notebook 06 once before notebooks 07 and 08. The final tables and figures *are* in the repository, so you can read the results without running anything.

---

## 5. The interactive app

Start it with `streamlit run app/app.py`. To stop it, press `Ctrl+C` in the terminal.

### The sidebar (left)

- **What happens to the citations?**
  - *Remove citations* - simulates missing data.
  - *Add citations, newer -> older only* - simulates wrong citations that are still possible in time (a paper only cites older papers).
  - *Add citations at random* - the version from Phase A; can create impossible citation loops.
- **How much?** - the share of the citations to remove or add (1-30 %).
- **Random seed** - the same settings with the same seed always give exactly the same damaged network. Change it to see a different random damage of the same size.
- **Method to look at in detail** - which ranking the tables and the plot below should show.

### Tab "Damage the graph"

- Four numbers at the top: how many citations there are now; how many papers lost all their citations; whether the network still has no citation loops; and the largest share of papers whose inputs went outside the range the forest was trained on (see *saturation* in section 6).
- **How much did each ranking move?** - one row per method, coloured from red (moved a lot) to green (hardly moved). Explained in section 6.
- **Top 20 papers** of the chosen method before and after the damage. *moved* > 0 means the paper went up.
- **Every paper - rank before vs rank after.** Points on the dashed line did not move.

### Tab "Look up a paper"

Type a PubMed ID (the tab suggests a few papers that fell the most). You see the paper's rank in every method before and after the damage, and the five inputs the forest sees. This is the place to understand *why* a paper moved.

### Tab "Experiment results"

The results of the full experiment (270 damaged networks), not of the single damage on the left. Choose a measure to see it for all methods and all damage levels. The full written conclusions are in the expandable section at the bottom.

### Tab "How it works"

A one-screen summary of the pipeline.

---

## 6. How to read the results

| Term | Meaning | Good value |
|---|---|---|
| **Rank** | position of a paper in a ranking; 1 = most important. Papers with exactly the same score share the average rank | - |
| **Spearman** | agreement between the ranking before and after the damage, over all papers. 1 = identical order, 0 = unrelated | close to 1 |
| **Kendall** | the same idea, but counts pairs of papers that swapped order. Usually a bit lower than Spearman | close to 1 |
| **Spearman over top 10 %** | Spearman computed only on the method's 1,971 most important papers | close to 1 |
| **Top-100 kept** | share of the 100 most important papers still in the top 100 after the damage | close to 1 |
| **Stay in top** | share of the damaged networks in which a top-10 % paper is still in the top 10 % | close to 1 |
| **CV (instability)** | how much one paper's score jumps between the 30 damaged networks: standard deviation divided by mean. 0 = never changed | close to 0 |
| **Network core** | important papers (top 10 %) whose CV is low compared with the other important papers | - |
| **Important but fragile** | top-10 % papers that fall out of the top 10 % in more than half of the damaged networks | few |

Things to keep in mind when comparing methods:

- **Ties make a method look stable.** k-core has only 10 different values and degree only 82, so most papers are tied and stay tied. Betweenness is 0 for 91 % of papers. The *top 10 %* columns are a fairer comparison.
- **Saturation.** A Random Forest cannot predict outside the range it saw during training. After strong damage, many papers fall outside that range and the forest gives them all the same score. Their score then stops moving - it looks stable, but only because the model can no longer tell them apart.
- **Compare the forest mainly with PageRank.** The forest was trained to imitate PageRank, so PageRank is its natural comparison. Degree and k-core measure a different kind of importance.

The main findings are written up in `docs/results_summary.md`.

---

## 7. Changing the experiment

Most users will not need this. Settings and where to change them:

| Setting | Default | Where |
|---|---|---|
| Random seed of the whole project | 42 | `src/config.py` - `SEED` |
| Damage levels | 5 %, 10 %, 20 % | `src/experiment.py` - `LEVELS` |
| Number of damaged networks per setting (K) | 30 | `notebooks/06_run_experiment.ipynb` - `run_all(..., k=30)` |
| Forest settings | 200 trees, min. 5 papers per leaf | `src/model.py` - `FOREST_SETTINGS` |

After changing the levels or K, delete the folder `results/runs/` (or call `run_all(..., redo=True)`) and run notebooks 06-08 again. After changing the forest settings or the seed, also delete `data/processed/models/` and run from notebook 03.

---

## 8. Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError: No module named 'src'` | Run commands from the project's main folder (the one with `README.md`), not from inside `notebooks/` or `app/`. The notebooks fix this themselves in their first cell. |
| `FileNotFoundError: ... results/runs/...` in notebook 07 or 08 | The raw results are not on GitHub. Run notebook 06 first (~6 min). |
| Notebook 06 seems stuck at `add_random_20` | This setting is the slowest (about 3 minutes). Settings that are already done are skipped if you restart. |
| `pip install` fails on `igraph` | Upgrade pip (`python -m pip install --upgrade pip`) and try again; igraph ships ready-made packages for Windows, macOS and Linux. |
| The app shows an old version after a code change | Press `R` in the browser tab, or stop the app and start it again. |
| `pytest` is slow | `pytest -m "not slow"` skips the tests that use the full dataset and the app (about 6 seconds). |

---

## 9. Where things are

```
app/app.py                 the interactive app
notebooks/                 the pipeline, one notebook per step (01-09)
data/raw/                  the original PubMed-Diabetes files - never changed
data/processed/            cleaned data, features, scaler, base ranking
results/tables/            result tables (CSV)
results/*.png              result figures
docs/results_summary.md    the answers to the research questions
docs/project_log.md        what was done in each step and why
docs/decisions.md          every decision and assumption, with its reason
```
