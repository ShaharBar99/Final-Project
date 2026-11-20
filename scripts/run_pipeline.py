"""Run the whole pipeline for one dataset and one feature set, from the raw files to the result tables.

    CITATION_DATASET=cora FEATURE_SET=no_closeness python scripts/run_pipeline.py

CITATION_DATASET  pubmed | cora | citeseer        (default pubmed)
FEATURE_SET       full   | no_closeness           (default full)
Everything is written to data/processed/<run>/ and results/<run>/ (see src/config.py), so the
runs never overwrite each other. The notebooks 01-08 do the same steps one at a time for PubMed.

Options:  --k N   damaged graphs per setting (default 30)
          --redo  rebuild saved models and experiment files even if they exist
"""
import argparse
import json
import time

import joblib
import pandas as pd

from src import analysis, config, data, experiment, features, model, ranking, scaling


def stamp(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def prepare(log):
    """Steps 1-3: clean data, features, frozen scaler. Returns (graph, raw, scaled, scaler)."""
    nodes, edges, cleaning_log = data.clean_dataset()
    data.save_dataset(nodes, edges, cleaning_log)
    stamp(f"cleaned: {len(nodes)} papers, {len(edges)} citations")

    g, _ = data.load_clean_graph()
    raw = features.compute_features(g)
    scaler = scaling.fit_scaler(raw)
    scaled = scaling.apply_scaler(scaler, raw)
    raw.to_csv(config.PROCESSED_DIR / "features_raw.csv", index_label="paper_id")
    scaled.to_csv(config.PROCESSED_DIR / "features_scaled.csv", index_label="paper_id")
    joblib.dump(scaler, config.PROCESSED_DIR / "scaler.joblib")
    stamp(f"features: {list(raw.columns)}")
    return g, raw, scaled, scaler


def train(scaler, redo):
    """Step 4: both forests, hold-out check, importances. Returns (x, forests)."""
    x = pd.read_csv(config.PROCESSED_DIR / "features_scaled.csv", index_col="paper_id", dtype={"paper_id": str})
    y = x[model.TARGET]
    variants = experiment.VARIANTS
    models_dir = config.PROCESSED_DIR / "models"
    forests = {name: model.load_or_train(name, x[cols], y, models_dir, scaler=scaler, rebuild_invalid=redo)
               for name, cols in variants.items()}
    report = {
        "dataset": config.DATASET, "feature_set": config.FEATURE_SET, "features": list(x.columns),
        "holdout": {name: model.holdout_check(x[cols], y) for name, cols in variants.items()},
        "single_centrality_spearman_vs_pagerank": {
            c: float(x[c].corr(y, method="spearman")) for c in model.FEATURES_WITHOUT_PAGERANK},
        "feature_importance": {name: dict(zip(cols, forests[name].feature_importances_.round(3).tolist()))
                               for name, cols in variants.items()},
        "forest_settings": model.FOREST_SETTINGS, "seed": config.SEED,
    }
    with open(config.PROCESSED_DIR / "model_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=float)
    stamp("forests trained; hold-out " + str({k: round(v["r2"], 3) for k, v in report["holdout"].items()}))
    return x, forests


def base_ranking(g, x, forests, scaler):
    """Step 5: scores and ranks of every method on the original graph."""
    scores = ranking.base_scores(x, forests)
    ranks = scores.apply(ranking.to_ranking)
    scores.to_csv(config.PROCESSED_DIR / "base_scores.csv", index_label="paper_id")
    ranks.to_csv(config.PROCESSED_DIR / "base_ranking.csv", index_label="paper_id")
    # the experiment scores the original graph again; both must give the same ranking
    again, _ = experiment.score_graph(g, scaler, forests)
    same = {m: bool((ranks.loc[again.index, m] == ranking.to_ranking(again[m])).all()) for m in again}
    assert all(same.values()), same
    stamp("base ranking saved and reproduced by the experiment code")


def tables():
    """Steps 7-9: stability, instability, network core -> results/<run>/tables/."""
    per_run, cv_table, profiles, cvs = analysis.evaluate_all()
    analysis.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    per_run.to_csv(analysis.TABLES_DIR / "per_run_correlations.csv", index=False)
    analysis.ranking_table(per_run).to_csv(analysis.TABLES_DIR / "ranking_stability.csv", index=False)
    cv_table.to_csv(analysis.TABLES_DIR / "cv_summary.csv", index=False)
    profiles.to_csv(analysis.TABLES_DIR / "instability_profiles.csv", index=False)
    summary, _ = analysis.core_analysis(cvs)
    summary.to_csv(analysis.TABLES_DIR / "core_summary.csv", index=False)
    analysis.out_of_range_table().to_csv(analysis.TABLES_DIR / "out_of_range.csv", index=False)
    stamp(f"tables written to {analysis.TABLES_DIR}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--k", type=int, default=30)
    parser.add_argument("--redo", action="store_true")
    args = parser.parse_args()

    stamp(f"dataset={config.DATASET} feature_set={config.FEATURE_SET} run={config.RUN_NAME or '(original PubMed paths)'}")
    g, _, _, scaler = prepare(None)
    x, forests = train(scaler, args.redo)
    base_ranking(g, x, forests, scaler)
    experiment.run_all(g, scaler, forests, k=args.k, redo=args.redo)
    tables()
    stamp("done")


if __name__ == "__main__":
    main()
