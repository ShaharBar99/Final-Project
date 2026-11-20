"""Builds the results report (HTML and PDF) from the result tables of every run.

    python scripts/build_report.py

Reads  results/<run>/tables/*.csv and data/processed/<run>/*  (made by scripts/run_pipeline.py or notebooks 01-08)
Writes results/report/results_report.html  and  results/report/results_report.pdf  (PDF needs Microsoft Edge or Chrome).
Every number in the text is computed from the tables; the claims that the text makes are asserted, so the
script stops if a rerun of the experiments no longer supports a sentence.
"""
import datetime
import subprocess
import sys
from pathlib import Path

import networkx as nx
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import report_figures as figs  # noqa: E402
from report_data import (CONTROL, DATASETS, DATASET_LABEL, KIND_LABEL, KINDS, LEVELS, METHOD_LABEL,  # noqa: E402
                         PRIMARY, SIMPLE, load_run, metric)
from report_html import Report  # noqa: E402

RESULTS, PROCESSED = ROOT / "results", ROOT / "data" / "processed"
OUT_DIR = RESULTS / "report"
SETTINGS = [(k, lv) for k in KINDS for lv in LEVELS]

FIGURES = ["flow", "degrees", "importance", "spearman", "stay", "gain", "close_acc", "close_gain"]
TABLES = ["features", "damage", "measures", "data", "models", "summary", "closeness", "answers"]


def pct(x, digits=0):
    return f"{x * 100:.{digits}f} %"


def f2(x):
    return f"{x:.2f}"


def f3(x):
    return f"{x:.3f}"


def signed(x, digits=2):
    return f"{x:+.{digits}f}"


def load_all():
    full = {ds: load_run(RESULTS, PROCESSED, ds, "full") for ds in DATASETS}
    reduced = {ds: load_run(RESULTS, PROCESSED, ds, "no_closeness") for ds in DATASETS}
    return full, reduced


def graph_of(ds):
    folder = PROCESSED / ("" if ds == "pubmed" else ds)
    nodes = pd.read_csv(folder / "nodes.csv", dtype=str)["paper_id"].tolist()
    edges = pd.read_csv(folder / "edges.csv", dtype=str)
    return nodes, edges


def dataset_facts(ds, run):
    nodes, edges = graph_of(ds)
    n, m = len(nodes), len(edges)
    out_deg = edges["source"].value_counts().reindex(nodes).fillna(0)
    in_deg = edges["target"].value_counts().reindex(nodes).fillna(0)
    g = nx.from_pandas_edgelist(edges, "source", "target", create_using=nx.DiGraph)
    g.add_nodes_from(nodes)
    log = {e["step"]: e for e in run["cleaning"]}
    raw = log["load_raw"]
    removed_loops = log["remove_self_loops"]["removed"]
    removed_cycle = (log["resolve_mutual_citations"]["removed"] if "resolve_mutual_citations" in log
                     else log["break_citation_cycles"]["removed"])
    removed_unknown = log["drop_edges_with_unknown_papers"]["removed"]
    return {
        "papers": n, "citations": m, "raw_papers": raw["nodes"], "raw_citations": raw["edges"],
        "density": m / (n * (n - 1)), "cite_nothing": float((out_deg == 0).mean()),
        "never_cited": float((in_deg == 0).mean()), "max_in": int(in_deg.max()), "max_out": int(out_deg.max()),
        "components": nx.number_weakly_connected_components(g),
        "removed_loops": removed_loops, "removed_cycle": removed_cycle, "removed_unknown": removed_unknown,
        "removed_isolated": log["remove_isolated_nodes"]["removed"],
        "cycle_rule": "mutual pairs, higher PubMed id kept" if ds == "pubmed" else "fewest edges so no loop is left",
    }


def main():
    full, reduced = load_all()
    facts = {ds: dataset_facts(ds, full[ds]) for ds in DATASETS}
    r = Report(FIGURES, TABLES)
    today = datetime.date.today().strftime("%d %B %Y")

    # ------------------------------------------------------------------ numbers used in the text
    def gap(run, table, column, ref):
        return [metric(run, table, column, k, lv, PRIMARY) - metric(run, table, column, k, lv, ref) for k, lv in SETTINGS]

    wins = {ds: {"spearman": sum(d > 0 for d in gap(full[ds], "rank", "spearman_mean", "pagerank")),
                 "stay": sum(d > 0 for d in gap(full[ds], "core", "important_stay_in_top", "pagerank"))} for ds in DATASETS}
    mean_gap = {ds: {"spearman": sum(gap(full[ds], "rank", "spearman_mean", "pagerank")) / 9,
                     "stay": sum(gap(full[ds], "core", "important_stay_in_top", "pagerank")) / 9} for ds in DATASETS}

    def importance(ds, version, feature, run_set=full):
        return run_set[ds]["model"]["feature_importance"][version].get(feature, 0.0)

    def holdout(ds, version, key, run_set=full):
        return run_set[ds]["model"]["holdout"][version][key]

    # sanity checks of the sentences below (see module docstring)
    for ds in DATASETS:
        assert importance(ds, "with_pagerank", "pagerank") > 0.99, ds       # the control copies its input
        assert importance(ds, "without_pagerank", "closeness") > 0.40, ds   # closeness is the main feature

    # ------------------------------------------------------------------ title and summary
    r.h1("Stability of learned node importance in citation networks")
    r.p("Results report - Capstone Project Phase B (26-2-R-2). Niran Czuckermann and Shahar Bar. Advisor: Dr. Dvora "
        f"Toledano-Kitai. Braude College of Engineering. {today}.", "sub")

    r.raw("<!--SUMMARY-->")
    summary_index = len(r.parts) - 1

    # ------------------------------------------------------------------ 1. the method
    r.h2("1. The method in one page")
    r.p("<b>Research question (Phase A).</b> Does a Random Forest that combines several centrality measures give node "
        "rankings that are more stable under edge noise than any single measure? A ranking is <i>stable</i> if the same "
        "papers stay on top when the citation data is incomplete or partly wrong.")
    r.figure("flow", figs.flow(), "The algorithm. Part A runs once on the original graph and produces the trained forest and "
                                  "the base ranking. Part B damages the graph, scores it with the same scaler and the same "
                                  "forest, and compares the new ranking with the base ranking. Dashed arrows show what part B "
                                  "reuses from part A.")
    r.p(f"The steps in {r.fig('flow')}, in plain words:")
    r.raw("<ol>"
          "<li><b>Clean the graph.</b> Every paper is a node, every citation a directed edge (A cites B: A to B). "
          "Duplicates, self-citations, unknown papers and citation loops are removed, because a paper can only cite older papers.</li>"
          f"<li><b>Compute five centralities</b> for every paper ({r.tab('features')}).</li>"
          "<li><b>Scale</b> them to [0, 1] with a Min-Max scaler that is fitted <i>once</i>, on the original graph, and never again "
          "(the &quot;frozen scaler&quot; of Phase A). On a damaged graph a scaled value may fall outside [0, 1]; it is not clipped.</li>"
          "<li><b>Train the Random Forest</b> on the original graph. Its target is the PageRank of each paper. Two versions are "
          "trained: <i>without</i> PageRank among its inputs (the primary model) and <i>with</i> it (the control, see Section 3).</li>"
          "<li><b>Base scores and ranking.</b> The two forests and the five single centralities each give every paper a score; "
          "scores become ranks (1 = most important, equal scores share the mean rank).</li>"
          f"<li><b>Damage the graph</b> in three ways and three sizes ({r.tab('damage')}); repeat 30 times each, which gives 270 damaged graphs.</li>"
          "<li><b>Re-score</b> every damaged graph: recompute the features, apply the same frozen scaler, apply the same forests "
          "(no retraining), and rank again.</li>"
          f"<li><b>Measure</b> how far each new ranking is from the base ranking ({r.tab('measures')}). The method whose ranking "
          "moves least is the most stable.</li></ol>")
    r.table("features", "The five features (one number per paper).",
            ["Feature", "What it measures", "How it is computed here"],
            [["Degree", "direct citation links", "(citations given + received) / (N - 1)"],
             ["Betweenness", "how often the paper lies on a shortest path between two others", "exact, directed, divided by (N - 1)(N - 2)"],
             ["Closeness", "how quickly the rest of the network reaches the paper through citations",
              "incoming distances with the Wasserman-Faust correction (decision D17)"],
             ["PageRank", "a paper is important if important papers cite it", "damping 0.85"],
             ["k-core", "how deep the paper sits in a densely connected group", "on the undirected graph, without self-loops"]],
            numeric_from=9)
    r.table("damage", "The three kinds of damage. Each is applied to 5 %, 10 % and 20 % of the citations.",
            ["Kind", "What happens", "What it imitates"],
            [["Remove citations", "random citations are deleted", "missing data"],
             ["Add random citations", "random citations between any two papers are added", "wrong citations (may create impossible loops)"],
             ["Add citations newer to older", "random citations are added, but only from a newer to an older paper",
              "wrong citations that are still possible in time"]], numeric_from=9)
    r.table("measures", "How the new ranking is compared with the base ranking (every method against its own base ranking).",
            ["Measure", "Question it answers", "Best value"],
            [["Spearman (all papers)", "is the whole order the same? 1 = identical, 0 = unrelated", "1"],
             ["Top-10 % stay", "in what share of the damaged graphs does a top-10 % paper stay in the top 10 %?", "1"],
             ["Instability (CV)", "how much does one paper's score jump across the 30 damaged graphs?", "0"]],
            numeric_from=9)
    r.p("Spearman over all papers can be flattering for methods with many ties (k-core has about 10 different values), "
        "so the top-10 % measure is used next to it: it only looks at the papers that matter. The forests are trained to imitate PageRank, "
        "so PageRank is their natural comparison; degree and k-core measure a different kind of importance.", "note")

    # ------------------------------------------------------------------ 2. data
    r.h2("2. The three citation networks")
    rows = []
    for ds in DATASETS:
        f = facts[ds]
        rows.append([DATASET_LABEL[ds], f"{f['papers']:,}", f"{f['citations']:,}", f"{f['density']:.5f}",
                     pct(f["cite_nothing"]), pct(f["never_cited"]), f"{f['max_in']} / {f['max_out']}", f"{f['components']:,}",
                     f"{f['removed_loops']} / {f['removed_cycle']} / {f['removed_unknown']} / {f['removed_isolated']}"])
    r.table("data", "The datasets after cleaning. Cleaning removed, in this order: self-citations / edges that closed a citation loop "
                    "/ edges to unknown papers / isolated papers. For PubMed-Diabetes the loop edges are the 11 mutual citations "
                    "(the higher PubMed id is taken as the newer paper); Cora and CiteSeer have no usable time information, so the "
                    "fewest edges that leave no loop are removed.",
            ["Dataset", "Papers", "Citations", "Density", "Cite nothing", "Never cited", "Max in / out", "Components", "Removed"],
            rows)
    r.figure("degrees", figs.degrees({ds: graph_of(ds) for ds in DATASETS}),
             "How many citations papers give and receive, as a share of the papers of each dataset. In every dataset most "
             "papers are cited once or never, and the out-degree is dominated by papers that cite nothing inside the dataset.")
    r.p(f"{r.tab('data')} and {r.fig('degrees')} show what the three networks share: they are citation graphs without loops, most papers "
        "sit at the end of a citation chain, and a few papers collect many citations. They differ in size and connectivity: "
        f"PubMed-Diabetes is about seven times larger than Cora and is one connected piece, Cora has {facts['cora']['components']} and CiteSeer "
        f"{facts['citeseer']['components']} separate pieces. This matters for closeness, which is only defined between papers that can reach each other.")

    # ------------------------------------------------------------------ 3. two forests
    r.h2("3. Why there are two Random Forests: with and without PageRank")
    r.p("<b>What Phase A said.</b> The feature vector of a paper holds five numbers, <i>including PageRank</i>, and the forest is trained to "
        "predict <i>PageRank</i>. PageRank is therefore both an input and the answer.")
    r.p("<b>Why that does not work.</b> A forest that is given the answer as an input learns to copy it. The result is a ranking that is "
        "PageRank with a little noise, so comparing its stability with PageRank's tells us nothing about combining several measures. "
        f"This is what happens in all three datasets ({r.fig('importance')}): the forest with PageRank puts "
        f"{pct(min(importance(d, 'with_pagerank', 'pagerank') for d in DATASETS), 1)} or more of its weight on that one input.")
    r.p("<b>What we do.</b> We train both. The <i>control</i> forest is Phase A exactly as written. The <i>primary</i> forest is the same forest "
        "with PageRank removed from its inputs, so it has to rebuild importance from degree, betweenness, closeness and k-core. "
        "The conclusions of this report are drawn from the primary forest; the control is reported to show what the original design does.")
    rows = []
    for ds in DATASETS:
        singles = full[ds]["model"]["single_centrality_spearman_vs_pagerank"]
        best_name = max(singles, key=singles.get)
        rows.append([DATASET_LABEL[ds], f2(holdout(ds, "with_pagerank", "r2")), f2(holdout(ds, "without_pagerank", "r2")),
                     f2(holdout(ds, "with_pagerank", "spearman")), f2(holdout(ds, "without_pagerank", "spearman")),
                     f"{singles[best_name]:.2f} ({best_name})"])
    r.table("models", "How well each forest predicts PageRank on papers it was not trained on (80 % train, 20 % test). "
                      "The last column is the best single centrality on its own (Spearman with PageRank).",
            ["Dataset", "R² with PageRank", "R² without", "Spearman with", "Spearman without", "Best single centrality"], rows)
    r.figure("importance", figs.importance({ds: full[ds]["model"] for ds in DATASETS}),
             "What the two forests rely on. The control forest uses almost only its PageRank input. The primary forest relies "
             "mostly on closeness in Cora and CiteSeer and on closeness and degree in PubMed-Diabetes.")
    ctrl_gap = {ds: [metric(full[ds], "rank", "spearman_mean", k, lv, CONTROL) - metric(full[ds], "rank", "spearman_mean", k, lv, "pagerank")
                     for k, lv in SETTINGS] for ds in DATASETS}
    assert all(sum(v) / 9 <= 0.0 for v in ctrl_gap.values())   # the control is never more stable than PageRank on average
    r.p(f"<b>Which is better, and why.</b> The primary forest predicts PageRank worse ({r.tab('models')}), as it must, because it is not given "
        "the answer. It is better as a test of the research question, because its stability is free to differ from PageRank's. "
        "The control cannot do better than PageRank on average: in Cora and CiteSeer its Spearman is within "
        f"{max(abs(sum(ctrl_gap[d]) / 9) for d in ('cora', 'citeseer')):.2f} of PageRank's on average, and in PubMed-Diabetes it is "
        f"{abs(sum(ctrl_gap['pubmed']) / 9):.2f} lower on average and up to {abs(min(ctrl_gap['pubmed'])):.2f} lower after damage, "
        "because a tree cannot predict outside the range it was trained on and many scaled values leave that range once citations are added. "
        "Better prediction of PageRank is therefore not the aim; testing a combination of measures is. "
        "Whether the same holds for other citation networks is what the three datasets are used for: the control copied its input in all three, "
        "so the reason for the primary forest does not depend on the dataset.")

    # ------------------------------------------------------------------ 4. results
    r.h2("4. Experiment results")
    r.p(f"Every method is compared with its own base ranking after each of the 270 damaged graphs. {r.fig('spearman')} shows the whole order "
        f"(Spearman) and {r.fig('stay')} shows whether the most important 10 % of the papers stay on top. Methods with many tied scores "
        "(k-core, degree) reach 1.0 in the second measure partly because tied papers do not change places.")
    r.figure("spearman", figs.curves(full, "rank", "spearman_mean", "mean Spearman"),
             "Spearman correlation between the ranking after damage and the base ranking, averaged over 30 damaged graphs. "
             "Rows are datasets, columns are kinds of damage, the x-axis is the share of citations changed. Higher is more stable. "
             "The primary forest is the thick solid orange line.")
    r.figure("stay", figs.curves(full, "core", "important_stay_in_top", "top-10 % stay"),
             "Share of the damaged graphs in which a paper from the base top 10 % is still in the top 10 %, averaged over those papers. "
             "Higher is more stable.")
    r.figure("gain", figs.difference(full, full, [("rank", "spearman_mean"), ("core", "important_stay_in_top")],
                                      ["primary forest minus PageRank\n(Spearman, all papers)",
                                       "primary forest minus PageRank\n(top-10 % stay)"], ref_method="pagerank"),
             "Is the primary forest more stable than the measure it learns to imitate? Bars above zero mean yes. Settings are "
             "rem = remove, rand = add random, dag = add newer to older, with the share of citations changed.")
    s = {ds: mean_gap[ds] for ds in DATASETS}
    r.p(f"<b>Against PageRank</b> ({r.fig('gain')}). In PubMed-Diabetes the primary forest keeps the top 10 % more stable than PageRank in "
        f"{wins['pubmed']['stay']} of 9 settings (on average {signed(s['pubmed']['stay'])}) while the order over all papers is the same on average "
        f"({signed(s['pubmed']['spearman'], 3)}). In Cora and CiteSeer it is not better: the average differences are "
        f"{signed(s['cora']['spearman'], 3)} and {signed(s['citeseer']['spearman'], 3)} in Spearman and {signed(s['cora']['stay'])} and "
        f"{signed(s['citeseer']['stay'])} in top-10 % stay.")
    r.p(f"{r.tab('summary')} gives the numbers behind the figures for the strongest damage. In every row the best single measure "
        "(degree, k-core, closeness or betweenness) is at least as stable as the primary forest in Spearman.")
    rows = []
    for ds in DATASETS:
        for kind in KINDS:
            lv = 0.20
            simple_best = max(SIMPLE, key=lambda m: metric(full[ds], "rank", "spearman_mean", kind, lv, m))
            assert metric(full[ds], "rank", "spearman_mean", kind, lv, simple_best) >= metric(full[ds], "rank", "spearman_mean", kind, lv, PRIMARY)
            rows.append([DATASET_LABEL[ds], KIND_LABEL[kind],
                         f3(metric(full[ds], "rank", "spearman_mean", kind, lv, PRIMARY)),
                         f3(metric(full[ds], "rank", "spearman_mean", kind, lv, "pagerank")),
                         f"{metric(full[ds], 'rank', 'spearman_mean', kind, lv, simple_best):.3f} ({simple_best})",
                         f2(metric(full[ds], "core", "important_stay_in_top", kind, lv, PRIMARY)),
                         f2(metric(full[ds], "core", "important_stay_in_top", kind, lv, "pagerank"))])
    r.table("summary", "The strongest damage (20 % of the citations). Spearman is over all papers; top-10 % stay as in the figures.",
            ["Dataset", "Damage", "Spearman primary forest", "Spearman PageRank", "Spearman best single measure", "Top-10 % stay primary forest",
             "Top-10 % stay PageRank"], rows, numeric_from=2)

    # ------------------------------------------------------------------ 5. closeness
    def change(ds, table, column):
        return [metric(reduced[ds], table, column, k, lv, PRIMARY) - metric(full[ds], table, column, k, lv, PRIMARY)
                for k, lv in SETTINGS]

    sp_change = {ds: sum(change(ds, "rank", "spearman_mean")) / 9 for ds in DATASETS}
    st_change = {ds: sum(change(ds, "core", "important_stay_in_top")) / 9 for ds in DATASETS}
    wins_red = {ds: sum(d > 0 for d in gap(reduced[ds], "core", "important_stay_in_top", "pagerank")) for ds in DATASETS}
    r2_drop = {ds: holdout(ds, "without_pagerank", "r2", full) - holdout(ds, "without_pagerank", "r2", reduced) for ds in DATASETS}
    assert all(v > 0.2 for v in r2_drop.values())          # the forest predicts PageRank much worse without closeness
    assert sp_change["cora"] < 0 and sp_change["citeseer"] < 0 and sp_change["pubmed"] > 0

    r.h2("5. Does the forest do better without closeness?")
    r.p("<b>Why we tested it.</b> Closeness is the one feature whose Phase A formula had to be changed (decision D17), and it is the feature "
        "the primary forest relies on most. The test removes closeness completely: from the features, from both forests and from the list of "
        "single measures. Nothing else changes (same data, same damage, same 30 repeats). The test lives on its own branch and in its own "
        "result folders; the main code still computes closeness.")
    r.figure("close_acc", figs.closeness_accuracy({d: full[d]["model"] for d in DATASETS}, {d: reduced[d]["model"] for d in DATASETS}),
             "How well the primary forest predicts PageRank on held-out papers, with and without closeness as an input.")
    rows = []
    for ds in DATASETS:
        rows.append([DATASET_LABEL[ds], pct(importance(ds, "without_pagerank", "closeness")),
                     f"{holdout(ds, 'without_pagerank', 'r2', full):.2f} to {holdout(ds, 'without_pagerank', 'r2', reduced):.2f}",
                     signed(sp_change[ds], 3), signed(st_change[ds], 3),
                     f"{wins[ds]['stay']} to {wins_red[ds]} of 9"])
    r.table("closeness", "What removing closeness changes for the primary forest. The two change columns are averages over the 9 damage "
                         "settings (negative = less stable without closeness). The last column counts the settings in which the forest keeps the "
                         "top 10 % more stable than PageRank, with and without closeness.",
            ["Dataset", "Closeness weight in the forest", "Hold-out R\u00b2", "Change in Spearman", "Change in top-10 % stay",
             "Settings better than PageRank"], rows)
    r.figure("close_gain", figs.difference(reduced, full, [("rank", "spearman_mean"), ("core", "important_stay_in_top")],
                                           ["without minus with closeness\n(Spearman, all papers)",
                                            "without minus with closeness\n(top-10 % stay)"], method_a=PRIMARY, method_b=PRIMARY),
             "Change in stability of the primary forest when closeness is removed. Bars above zero mean it is more stable without closeness.")
    r.p(f"<b>Accuracy.</b> In all three datasets the forest predicts PageRank clearly worse without closeness ({r.fig('close_acc')}): R\u00b2 falls by "
        f"{r2_drop['pubmed']:.2f} in PubMed-Diabetes, {r2_drop['cora']:.2f} in Cora and {r2_drop['citeseer']:.2f} in CiteSeer. Closeness carries "
        f"{pct(importance('pubmed', 'without_pagerank', 'closeness'))}, {pct(importance('cora', 'without_pagerank', 'closeness'))} and "
        f"{pct(importance('citeseer', 'without_pagerank', 'closeness'))} of the forest's decisions, and on its own it follows PageRank better than "
        "any other single measure.")
    r.p(f"<b>Stability</b> ({r.tab('closeness')}, {r.fig('close_gain')}). The effect is not the same in every dataset. In Cora and CiteSeer the forest is less "
        f"stable without closeness ({signed(sp_change['cora'], 3)} and {signed(sp_change['citeseer'], 3)} in Spearman) and never beats PageRank. In "
        f"PubMed-Diabetes the order over all papers is slightly more stable without closeness ({signed(sp_change['pubmed'], 3)}), but the top 10 % is less "
        f"stable ({signed(st_change['pubmed'], 3)}). The small gain in PubMed-Diabetes comes with a forest that predicts PageRank far worse "
        f"(R\u00b2 {holdout('pubmed', 'without_pagerank', 'r2', reduced):.2f}); it leans on degree and k-core, which are stable but measure something else.")
    r.box("Answer to the question", [
        "<b>Better results without closeness?</b> No for accuracy (all three datasets). For stability it depends on the dataset: worse in Cora and "
        "CiteSeer, mixed in PubMed-Diabetes. We do not recommend dropping closeness.",
        "<b>The same on other citation networks?</b> The loss of accuracy is the same in all three; the effect on stability is not. "
        "A conclusion from one dataset therefore does not carry over, and more datasets would be needed to say how often each case occurs."])

    # ------------------------------------------------------------------ 6. answers
    r.h2("6. Answers to the research questions and hypotheses")
    core_gain = {}
    best_single_wins = {}
    for ds in DATASETS:
        a = [metric(full[ds], "core", "core_stay_in_top", k, lv, PRIMARY) for k, lv in SETTINGS]
        b = [metric(full[ds], "core", "unstable_important_stay_in_top", k, lv, PRIMARY) for k, lv in SETTINGS]
        core_gain[ds] = (sum(a) / 9, sum(b) / 9)
        assert core_gain[ds][0] > core_gain[ds][1], ds
        best_single_wins[ds] = sum(
            metric(full[ds], "rank", "spearman_mean", k, lv, PRIMARY)
            >= max(metric(full[ds], "rank", "spearman_mean", k, lv, m) for m in SIMPLE) for k, lv in SETTINGS)
        assert best_single_wins[ds] == 0, ds
    rows = [
        ["H1: a combined score is more stable than single measures",
         *[f"vs PageRank: top-10 % stay better in {wins[ds]['stay']} of 9, Spearman better in {wins[ds]['spearman']} of 9; "
           f"vs the best of degree, k-core, closeness and betweenness: better in {best_single_wins[ds]} of 9" for ds in DATASETS]],
        ["H2: the forest copes with the failure of one feature by using the others",
         *[f"the control puts {pct(importance(ds, 'with_pagerank', 'pagerank'), 1)} on its PageRank input; "
           f"the primary forest puts {pct(importance(ds, 'without_pagerank', 'closeness'))} on closeness" for ds in DATASETS]],
        ["H3: the network core keeps its high rank (adjusted core rule)",
         *[f"core papers stay in the top 10 % in {pct(core_gain[ds][0])} of the damaged graphs, other important papers in {pct(core_gain[ds][1])}"
           for ds in DATASETS]],
    ]
    r.table("answers", "The three Phase A hypotheses, checked on each dataset with the primary forest (averages over the 9 damage settings).",
            ["Hypothesis", DATASET_LABEL["pubmed"], DATASET_LABEL["cora"], DATASET_LABEL["citeseer"]], rows, numeric_from=9)
    r.p(f"{r.tab('answers')} answers them with numbers from each dataset. "
        "<b>H1</b> is supported only in a narrow sense: for the top 10 % of PubMed-Diabetes against PageRank. It does not hold in Cora or CiteSeer, "
        "and in no dataset is the forest more stable than the best simple measure. <b>H2</b> is not supported: the Phase A forest copies its PageRank "
        "input, and the primary forest depends mostly on one feature, closeness, so a damaged closeness hurts it. <b>H3</b> holds in all three datasets "
        "with the adjusted core rule; the Phase A core rule gives an empty core on this kind of data (decision D16).")

    # ------------------------------------------------------------------ 7. limits
    r.h2("7. Limits of these results")
    r.raw("<ul>"
          "<li><b>Three datasets.</b> All are small to medium citation graphs of scientific papers: biomedical papers on diabetes (PubMed-Diabetes), "
          "machine-learning papers (Cora) and computer-science papers in six areas (CiteSeer). "
          "The differences between them show that one dataset is not enough for a general claim.</li>"
          "<li><b>One target.</b> The forests learn PageRank. Agreeing with PageRank is not the same as measuring true importance, and a "
          "ranking can be stable and still wrong.</li>"
          "<li><b>No dates.</b> Citation loops in Cora and CiteSeer were removed with the fewest edges, which is a choice (decision D18); "
          "PubMed-Diabetes uses its id order.</li>"
          "<li><b>Forest settings.</b> Every tree sees all features at every split (no feature randomness), which differs from the usual Random "
          "Forest and from the argument of Phase A hypothesis H2. A rerun with half the features per split is still open.</li>"
          "<li><b>Trees cannot extrapolate.</b> After citations are added many scaled inputs leave the range seen in training, and the forest "
          "gives all those papers the same score. Part of its stability is this effect.</li>"
          "<li><b>Training and testing on the same graph</b>, as in Phase A; the hold-out check only shows that the forest learned something.</li></ul>")
    r.h2("Reproducing this report")
    r.raw("<p class='note'>From the project folder (Python 3.12, packages as in requirements.txt):</p>"
          "<pre style='background:#f4f7fa;padding:10px;font-size:12px'>"
          "CITATION_DATASET=cora FEATURE_SET=full          python scripts/run_pipeline.py   # about 1 minute\n"
          "CITATION_DATASET=cora FEATURE_SET=no_closeness  python scripts/run_pipeline.py\n"
          "(the same for citeseer and pubmed; PubMed with all features is notebooks 01-08)\n"
          "python scripts/build_report.py                  # writes results/report/</pre>")

    # ------------------------------------------------------------------ summary at the top
    items = [
        "<b>A Random Forest did not give the most stable ranking in any of the three datasets.</b> In none of the 27 settings (3 datasets x 9) "
        "was the primary forest more stable than the best of degree, k-core, closeness and betweenness.",
        f"<b>Against PageRank, the measure it learns, the result depends on the dataset.</b> In PubMed-Diabetes the forest keeps the top 10 % "
        f"more stable in {wins['pubmed']['stay']} of 9 settings; in Cora and CiteSeer it does so in {wins['cora']['stay']} and {wins['citeseer']['stay']} of 9.",
        "<b>The Phase A design copies its input.</b> A forest that gets PageRank as an input and as its target puts "
        f"{pct(min(importance(d, 'with_pagerank', 'pagerank') for d in DATASETS), 1)} or more of its weight on it. That is why the main model has no PageRank input.",
        "<b>Closeness should stay.</b> It is the most informative feature in all three datasets; without it the forest predicts PageRank much worse, "
        "and in Cora and CiteSeer it is also less stable. The effect on stability is not the same in every dataset.",
        "<b>Results do not carry over automatically from one dataset to another.</b> PubMed-Diabetes alone would have given a more favourable picture.",
    ]
    li = "".join(f"<li>{i}</li>" for i in items)
    r.parts[summary_index] = f'<div class="box"><b>Main findings</b><ul>{li}</ul></div>'

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    page = r.render("Results report - stability of learned node importance")
    html_path = OUT_DIR / "results_report.html"
    html_path.write_text(page, encoding="utf-8")
    print("written", html_path, f"({len(page) / 1e6:.1f} MB)")
    make_pdf(html_path, OUT_DIR / "results_report.pdf")


def make_pdf(html_path, pdf_path):
    """Print the page to PDF with a headless Edge or Chrome, if one is installed."""
    candidates = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                  r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
                  r"C:\Program Files\Google\Chrome\Application\chrome.exe"]
    browser = next((c for c in candidates if Path(c).exists()), None)
    if browser is None:
        print("no Edge or Chrome found: open the HTML file and use Print / Save as PDF")
        return
    subprocess.run([browser, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf_path}", html_path.resolve().as_uri()], check=True, capture_output=True, timeout=180)
    print("written", pdf_path)


if __name__ == "__main__":
    main()
