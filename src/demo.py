"""One damaged graph, end to end - shared by the demo notebook and the Streamlit app.

    g  --perturb-->  damaged graph  --features + frozen scaler + forests-->  scores
and then: how far did each method's ranking move?
"""
import pandas as pd

from . import config, data, experiment, model, perturb, ranking, stability

METHOD_LABELS = {
    "rf_without_pagerank": "Random Forest (without PageRank input)",
    "rf_with_pagerank": "Random Forest (with PageRank input)",
    "pagerank": "PageRank",
    "closeness": "Closeness",
    "degree": "Degree",
    "kcore": "k-core",
    "betweenness": "Betweenness",
}
KIND_LABELS = {
    "remove": "Remove citations (missing data)",
    "add_dag": "Add citations, newer -> older only (noise that respects time)",
    "add_random": "Add citations at random (Phase A, can create cycles)",
}


def load_pipeline():
    """Everything frozen after training: graph, scaler, both forests and the base scores.

    Forests are rebuilt (fixed seed, a few seconds) if their files are missing,
    e.g. on a fresh clone of the repository.
    """
    import joblib

    g, nodes = data.load_clean_graph()
    scaler = joblib.load(config.PROCESSED_DIR / "scaler.joblib")
    x = pd.read_csv(config.PROCESSED_DIR / "features_scaled.csv", index_col="paper_id",
                    dtype={"paper_id": str})
    forests = {name: model.load_or_train(name, x[cols], x[model.TARGET], config.PROCESSED_DIR / "models")
               for name, cols in experiment.VARIANTS.items()}
    base, _ = experiment.score_graph(g, scaler, forests)
    return g, nodes, scaler, forests, base


def damage_and_score(g, scaler, forests, kind, level, seed=config.SEED):
    """Perturb g once and score every method on the result.

    Returns (damaged graph, scores, out_of_range). The run number is fixed at 0,
    so (kind, level, seed) always gives the same graph.
    """
    order = perturb.time_order(g) if kind == "add_dag" else None
    rng = perturb.run_rng(seed, kind, level, 0)
    damaged = perturb.perturb(g, kind, level, rng, order)
    scores, out_of_range = experiment.score_graph(damaged, scaler, forests)
    return damaged, scores.loc[list(g.nodes)], out_of_range


def method_summary(base, scores, top_k=100):
    """Per method: Spearman / Kendall with the base ranking and how much of the top-k survived."""
    rows = {}
    for method in experiment.METHODS:
        base_ranks = ranking.to_ranking(base[method]).to_numpy()
        new_ranks = ranking.to_ranking(scores[method]).to_numpy()
        spearman, kendall = stability.rank_correlations(base_ranks, new_ranks)
        rows[METHOD_LABELS[method]] = {
            "Spearman": spearman,
            "Kendall": kendall,
            f"top-{top_k} kept": stability.top_k_retention(base_ranks, new_ranks, top_k),
        }
    return pd.DataFrame(rows).T.sort_values("Spearman", ascending=False)


def rank_changes(base, scores, method, nodes=None, top=20):
    """The base top papers of one method, with their new rank and how far they moved."""
    base_ranks = ranking.to_ranking(base[method])
    new_ranks = ranking.to_ranking(scores[method])
    table = pd.DataFrame({"rank before": base_ranks, "rank after": new_ranks})
    table["moved"] = table["rank before"] - table["rank after"]  # positive = went up
    if nodes is not None:
        table["label"] = nodes.set_index("paper_id")["label"].reindex(table.index)
    return table.sort_values("rank before").head(top)


def graph_change(g, damaged):
    """Short description of what the damage did to the graph."""
    import networkx as nx
    return {
        "edges before": g.number_of_edges(),
        "edges after": damaged.number_of_edges(),
        "papers with no edges": sum(1 for n in damaged if damaged.degree(n) == 0),
        "still acyclic": nx.is_directed_acyclic_graph(damaged),
    }
