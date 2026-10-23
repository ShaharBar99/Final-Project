"""Run the stability metrics (block F) over every saved setting and write the result tables."""
import json

import pandas as pd

from . import config, experiment, perturb, stability

TABLES_DIR = config.RESULTS_DIR / "tables"


def load_base_scores():
    return pd.read_csv(experiment.RUNS_DIR / "base_scores.csv", index_col="paper_id",
                       dtype={"paper_id": str})


def evaluate_all(kinds=perturb.KINDS, levels=experiment.LEVELS, reference="rf_without_pagerank"):
    """Evaluate every (kind, level). Returns (per_run, cv_table, profiles, cvs).

    per_run   one row per (kind, level, method, run): spearman, kendall, top-k kept
    cv_table  one row per (kind, level, method): mean / median CV, ...
    profiles  one row per (kind, level, method): Spearman of CV vectors vs `reference`
    cvs       dict (kind, level) -> DataFrame papers x methods with each paper's CV
    """
    base = load_base_scores()
    per_run, cv_rows, profile_rows, cvs = [], [], [], {}
    for kind in kinds:
        for level in levels:
            scores, nodes, methods = experiment.load_setting(kind, level)
            assert nodes == list(base.index), "paper order differs from base scores"
            runs_table, cv = stability.evaluate_setting(scores, base, methods)
            runs_table.insert(0, "level", level)
            runs_table.insert(0, "kind", kind)
            per_run.append(runs_table)

            zero = {m: stability.zero_mean_share(scores[:, :, i]) for i, m in enumerate(methods)}
            summary = stability.cv_summary(cv, base, zero)
            summary.insert(0, "level", level)
            summary.insert(0, "kind", kind)
            cv_rows.append(summary.rename_axis("method").reset_index())

            for m, rho in stability.instability_profile_correlation(cv, reference).items():
                profile_rows.append({"kind": kind, "level": level, "method": m, "spearman_cv_vs_ref": rho})
            cvs[(kind, level)] = cv
    return (pd.concat(per_run, ignore_index=True), pd.concat(cv_rows, ignore_index=True),
            pd.DataFrame(profile_rows), cvs)


def ranking_table(per_run):
    """Mean / std of Spearman and Kendall, and ranking instability, per (kind, level, method)."""
    rows = []
    for (kind, level, method), grp in per_run.groupby(["kind", "level", "method"], sort=False):
        row = {"kind": kind, "level": level, "method": method}
        for metric in ["spearman", "kendall", "spearman_top10pct"]:
            s = stability.ranking_summary(grp[metric])
            row[f"{metric}_mean"], row[f"{metric}_std"], row[f"{metric}_instability"] = s["mean"], s["std"], s["instability"]
        for col in [c for c in grp.columns if c.endswith("_kept")]:
            row[col] = grp[col].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def out_of_range_table(kinds=perturb.KINDS, levels=experiment.LEVELS):
    """Average share of scaled feature values outside [0, 1] per setting (from the run info files)."""
    rows = []
    for kind in kinds:
        for level in levels:
            with open(experiment.RUNS_DIR / f"{experiment.setting_name(kind, level)}_info.json") as f:
                info = json.load(f)
            share = pd.DataFrame([r["out_of_range"] for r in info]).mean()
            rows.append({"kind": kind, "level": level, **share.to_dict(),
                         "seconds_per_run": sum(r["seconds"] for r in info) / len(info)})
    return pd.DataFrame(rows)
