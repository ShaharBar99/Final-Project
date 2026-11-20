"""Loads the result tables of every run (dataset x feature set) for the results report."""
import json

import pandas as pd

ROOT_RESULTS = None  # filled by build_report.py (project results folder)
ROOT_PROCESSED = None

DATASETS = ["pubmed", "cora", "citeseer"]
DATASET_LABEL = {"pubmed": "PubMed-Diabetes", "cora": "Cora", "citeseer": "CiteSeer"}
KINDS = ["remove", "add_random", "add_dag"]
KIND_LABEL = {"remove": "Remove citations", "add_random": "Add random citations",
              "add_dag": "Add citations newer to older"}
LEVELS = [0.05, 0.10, 0.20]

PRIMARY, CONTROL = "rf_without_pagerank", "rf_with_pagerank"
SIMPLE = ["degree", "betweenness", "closeness", "kcore"]
METHOD_LABEL = {
    PRIMARY: "Forest without PageRank (primary)",
    CONTROL: "Forest with PageRank (control)",
    "pagerank": "PageRank", "degree": "Degree", "betweenness": "Betweenness",
    "closeness": "Closeness", "kcore": "k-core",
}
# Okabe-Ito colours (readable for colour-blind readers)
METHOD_COLOR = {
    PRIMARY: "#D55E00", CONTROL: "#E69F00", "pagerank": "#0072B2", "degree": "#009E73",
    "kcore": "#56B4E9", "betweenness": "#CC79A7", "closeness": "#7F7F7F",
}


def run_name(dataset, feature_set="full"):
    """Folder name of a run, the same rule as src/config.py."""
    if (dataset, feature_set) == ("pubmed", "full"):
        return ""
    return dataset + ("" if feature_set == "full" else "_" + feature_set)


def load_run(results_dir, processed_dir, dataset, feature_set="full"):
    name = run_name(dataset, feature_set)
    tables = results_dir / name / "tables"
    processed = processed_dir / name
    run = {
        "rank": pd.read_csv(tables / "ranking_stability.csv"),
        "core": pd.read_csv(tables / "core_summary.csv"),
        "cv": pd.read_csv(tables / "cv_summary.csv"),
        "profiles": pd.read_csv(tables / "instability_profiles.csv"),
        "model": json.load(open(processed / "model_report.json", encoding="utf-8")),
        "cleaning": json.load(open(processed / "cleaning_log.json", encoding="utf-8")),
    }
    return run


def metric(run, table, column, kind, level, method):
    df = run[table]
    row = df[(df.kind == kind) & (df.level == level) & (df.method == method)]
    assert len(row) == 1, (table, kind, level, method)
    return float(row.iloc[0][column])
