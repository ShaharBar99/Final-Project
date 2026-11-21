"""Builds the results report (HTML and PDF) from the result tables of every run.

    python scripts/build_report.py

Reads  results/<run>/tables/*.csv and data/processed/<run>/*  (made by scripts/run_pipeline.py or notebooks 01-08)
Writes results/report/results_report.html  and  results/report/results_report.pdf  (PDF needs Microsoft Edge or Chrome).

Part I is the main study on PubMed-Diabetes. Part II repeats it on Cora and CiteSeer and tests the forest
without closeness. Every number in the text is computed from the tables, and the claims that the text makes
are asserted, so the script stops if a rerun of the experiments no longer supports a sentence.
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
from report_data import (CONTROL, DATASETS, DATASET_LABEL, KIND_LABEL, KINDS, LEVELS,  # noqa: E402
                         PRIMARY, SIMPLE, load_run, metric)
from report_html import Report  # noqa: E402

RESULTS, PROCESSED = ROOT / "results", ROOT / "data" / "processed"
OUT_DIR = RESULTS / "report"
SETTINGS = [(k, lv) for k in KINDS for lv in LEVELS]
MAIN, OTHERS = "pubmed", ["cora", "citeseer"]

FIGURES = ["flow", "degrees_main", "importance_main", "spearman_main", "stay_main", "gain_main",
           "degrees_all", "importance_all", "spearman_others", "stay_others", "gain_all", "close_acc", "close_gain"]
TABLES = ["features", "damage", "measures", "models_main", "summary_main", "answers_main",
          "data_all", "models_all", "summary_others", "closeness", "answers_all"]

SOURCES = {
    "pubmed": "Namata, London, Getoor and Huang (2012), Query-driven active surveying for collective classification, "
              "Workshop on Mining and Learning with Graphs (MLG)",
    "cora": "McCallum, Nigam, Rennie and Seymore (2000), Automating the construction of internet portals with machine learning, "
            "Information Retrieval 3(2)",
    "citeseer": "Giles, Bollacker and Lawrence (1998), CiteSeer: an automatic citation indexing system, ACM Digital Libraries",
    "linqs": "Sen, Namata, Bilgic, Getoor, Gallagher and Eliassi-Rad (2008), Collective classification in network data, "
             "AI Magazine 29(3)",
}
SHORT_SOURCE = {"pubmed": "Namata et al., 2012", "cora": "McCallum et al., 2000; LINQS version, Sen et al., 2008",
                "citeseer": "Giles et al., 1998; LINQS version, Sen et al., 2008"}
CONTENT = {"pubmed": "papers from PubMed about diabetes, in three classes",
           "cora": "machine-learning papers in seven classes",
           "citeseer": "computer-science papers in six classes"}


def pct(x, digits=0):
    return f"{x * 100:.{digits}f} %"


def f2(x):
    return f"{x:.2f}"


def f3(x):
    return f"{x:.3f}"


def signed(x, digits=2):
    return f"{x:+.{digits}f}"


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
    removed = {"self-citations": log["remove_self_loops"]["removed"],
               "citations that closed a loop": (log["resolve_mutual_citations"]["removed"] if "resolve_mutual_citations" in log
                                                else log["break_citation_cycles"]["removed"]),
               "citations to papers missing from the dataset": log["drop_edges_with_unknown_papers"]["removed"],
               "duplicates": log["remove_duplicate_edges"]["removed"]}
    return {
        "papers": n, "citations": m, "raw_papers": raw["nodes"], "raw_citations": raw["edges"],
        "density": m / (n * (n - 1)), "cite_nothing": float((out_deg == 0).mean()),
        "never_cited": float((in_deg == 0).mean()), "max_in": int(in_deg.max()), "max_out": int(out_deg.max()),
        "components": nx.number_weakly_connected_components(g), "removed": removed,
        "removed_citations": raw["edges"] - m, "removed_papers": log["remove_isolated_nodes"]["removed"],
    }


def cleaning_sentence(ds, f):
    """'<Dataset> (<source>): <content>. ... Cleaning removed ... Result: ...' in two or three sentences."""
    steps = sorted(((v, k) for k, v in f["removed"].items() if v), reverse=True)
    detail = ", ".join(f"{v:,} {k}" for v, k in steps)
    loops = ("the higher PubMed id is kept as the newer paper" if ds == "pubmed"
             else "the fewest citations that leave no loop, since there are no dates")
    papers = (f", and then {f['removed_papers']:,} papers left without any citation ({pct(f['removed_papers'] / f['raw_papers'], 1)})"
              if f["removed_papers"] else "; no paper was removed")
    share = f["removed_citations"] / f["raw_citations"]
    pieces = "one connected network" if f["components"] == 1 else f"{f['components']:,} separate pieces"
    return (f"<b>{DATASET_LABEL[ds]}</b> ({SHORT_SOURCE[ds]}): {f['raw_papers']:,} {CONTENT[ds]} and {f['raw_citations']:,} directed "
            f"citations. Cleaning removed {f['removed_citations']:,} citations ({pct(share, 2 if share < 0.001 else 1)}): "
            f"{detail} (loops: {loops}){papers}. Result: {f['papers']:,} papers and {f['citations']:,} citations, no loops, {pieces}.")


def main():
    full = {ds: load_run(RESULTS, PROCESSED, ds, "full") for ds in DATASETS}
    reduced = {ds: load_run(RESULTS, PROCESSED, ds, "no_closeness") for ds in DATASETS}
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
    ctrl_gap = {ds: [metric(full[ds], "rank", "spearman_mean", k, lv, CONTROL) - metric(full[ds], "rank", "spearman_mean", k, lv, "pagerank")
                     for k, lv in SETTINGS] for ds in DATASETS}
    best_single_wins = {ds: sum(metric(full[ds], "rank", "spearman_mean", k, lv, PRIMARY)
                                >= max(metric(full[ds], "rank", "spearman_mean", k, lv, m) for m in SIMPLE) for k, lv in SETTINGS)
                        for ds in DATASETS}
    core_stay = {}
    for ds in DATASETS:
        a = [metric(full[ds], "core", "core_stay_in_top", k, lv, PRIMARY) for k, lv in SETTINGS]
        b = [metric(full[ds], "core", "unstable_important_stay_in_top", k, lv, PRIMARY) for k, lv in SETTINGS]
        core_stay[ds] = (sum(a) / 9, sum(b) / 9)

    def importance(ds, version, feature, run_set=full):
        return run_set[ds]["model"]["feature_importance"][version].get(feature, 0.0)

    def holdout(ds, version, key, run_set=full):
        return run_set[ds]["model"]["holdout"][version][key]

    # the sentences below rely on these (see module docstring)
    for ds in DATASETS:
        assert importance(ds, "with_pagerank", "pagerank") > 0.99, ds       # the control copies its input
        assert importance(ds, "without_pagerank", "closeness") > 0.40, ds   # closeness is the main feature
        assert sum(ctrl_gap[ds]) / 9 <= 0.0, ds                             # the control never beats PageRank on average
        assert best_single_wins[ds] == 0, ds                                # never more stable than the best single measure
        assert core_stay[ds][0] > core_stay[ds][1], ds                      # core papers stay on top more often
    assert wins[MAIN]["stay"] >= 5 and all(wins[d]["stay"] == 0 for d in OTHERS)

    # ------------------------------------------------------------------ title and summary
    r.h1("Stability of learned node importance in citation networks")
    r.p("Results report - Capstone Project Phase B (26-2-R-2). Niran Czuckermann and Shahar Bar. Advisor: Dr. Dvora "
        f"Toledano-Kitai. Braude College of Engineering. {today}.", "sub")
    r.p("<b>How this report is organised.</b> Part I is the study planned in Phase A, on the PubMed-Diabetes citation network. "
        "Part II checks how far its conclusions go: the same experiment on two more citation networks (Cora and CiteSeer) and "
        "a version of the forest without the closeness feature.")
    r.raw("<!--SUMMARY-->")
    summary_index = len(r.parts) - 1

    # ================================================================== PART I
    r.raw('<h1 style="margin-top:30px;font-size:24px">Part I - The main study: PubMed-Diabetes</h1>')

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

    # ------------------------------------------------------------------ 2. the dataset
    fm = facts[MAIN]
    r.h2("2. The dataset and its cleaning")
    r.p(cleaning_sentence(MAIN, fm))
    r.figure("degrees_main", figs.degrees({MAIN: graph_of(MAIN)}),
             "How many citations the PubMed-Diabetes papers receive (left) and give (right), as a share of all papers.")
    r.p(f"As {r.fig('degrees_main')} shows, {pct(fm['never_cited'])} of the papers are never cited and {pct(fm['cite_nothing'])} cite no other "
        f"paper of the dataset (it only keeps citations between its own papers), while the most cited paper has {fm['max_in']} citations. "
        "Most papers therefore sit at the end of a citation chain; this is why betweenness is 0 for most papers and why closeness needed "
        "a careful definition.")

    # ------------------------------------------------------------------ 3. two forests
    r.h2("3. Why there are two Random Forests: with and without PageRank")
    r.p("<b>What Phase A said.</b> The feature vector of a paper holds five numbers, <i>including PageRank</i> (Phase A, Model Definition: "
        "Node Representation), and the forest is trained to predict <i>PageRank</i> (Phase A, Model Definition: Target Variable, and "
        "Appendix A). PageRank is therefore both an input and the answer.")
    r.p("<b>Why that does not work.</b> A forest that is given the answer as an input learns to copy it. The result is PageRank with a "
        "little noise, so comparing its stability with PageRank's says nothing about combining several measures. "
        f"In PubMed-Diabetes the forest with PageRank puts {pct(importance(MAIN, 'with_pagerank', 'pagerank'), 1)} of its weight on that "
        f"one input ({r.fig('importance_main')}).")
    r.p("<b>What we do.</b> We train both. The <i>control</i> forest is Phase A exactly as written. The <i>primary</i> forest is the same forest "
        "with PageRank removed from its inputs, so it has to rebuild importance from degree, betweenness, closeness and k-core. "
        "The conclusions are drawn from the primary forest; the control shows what the original design does.")
    singles = full[MAIN]["model"]["single_centrality_spearman_vs_pagerank"]
    r.table("models_main", "How well each forest predicts PageRank on papers it was not trained on (80 % train, 20 % test), PubMed-Diabetes.",
            ["Forest", "R\u00b2", "Spearman", "Largest input weight"],
            [["With PageRank (control)", f2(holdout(MAIN, "with_pagerank", "r2")), f2(holdout(MAIN, "with_pagerank", "spearman")),
              f"PageRank {pct(importance(MAIN, 'with_pagerank', 'pagerank'), 1)}"],
             ["Without PageRank (primary)", f2(holdout(MAIN, "without_pagerank", "r2")), f2(holdout(MAIN, "without_pagerank", "spearman")),
              f"closeness {pct(importance(MAIN, 'without_pagerank', 'closeness'))}, degree {pct(importance(MAIN, 'without_pagerank', 'degree'))}"],
             ["Best single centrality alone", "-", f2(max(singles.values())), max(singles, key=singles.get)]])
    r.figure("importance_main", figs.importance({MAIN: full[MAIN]["model"]}),
             "What the two forests rely on (PubMed-Diabetes). The control uses almost only its PageRank input; the primary forest combines "
             "closeness and degree.")
    r.p(f"<b>Which is better, and why.</b> The primary forest predicts PageRank worse ({r.tab('models_main')}), as it must, because it is not "
        "given the answer; it still follows PageRank better than any single centrality. It is the better test of the research question, "
        "because its stability is free to differ from PageRank's. The control cannot do better than PageRank: its Spearman is "
        f"{abs(sum(ctrl_gap[MAIN]) / 9):.2f} lower on average and up to {abs(min(ctrl_gap[MAIN])):.2f} lower after damage, because a tree "
        "cannot predict outside the range it was trained on, and many scaled values leave that range once citations are added.")

    # ------------------------------------------------------------------ 4. results
    s = mean_gap[MAIN]
    r.h2("4. Experiment results")
    r.p(f"Every method is compared with its own base ranking after each of the 270 damaged graphs. {r.fig('spearman_main')} shows the whole "
        f"order (Spearman) and {r.fig('stay_main')} whether the most important 10 % of the papers stay on top. Methods with many tied scores "
        "(k-core, degree) reach 1.0 in the second measure partly because tied papers do not change places.")
    r.figure("spearman_main", figs.curves(full, "rank", "spearman_mean", "mean Spearman", datasets=[MAIN]),
             "Spearman correlation between the ranking after damage and the base ranking, averaged over 30 damaged graphs, for each kind "
             "of damage. Higher is more stable. The primary forest is the thick solid orange line.")
    r.figure("stay_main", figs.curves(full, "core", "important_stay_in_top", "top-10 % stay", datasets=[MAIN]),
             "Share of the damaged graphs in which a paper from the base top 10 % is still in the top 10 %, averaged over those papers.")
    r.figure("gain_main", figs.difference(full, full, [("rank", "spearman_mean"), ("core", "important_stay_in_top")],
                                           ["primary forest minus PageRank\n(Spearman, all papers)",
                                            "primary forest minus PageRank\n(top-10 % stay)"], ref_method="pagerank", datasets=[MAIN]),
             "Is the primary forest more stable than PageRank, the measure it learns to imitate? Bars above zero mean yes. "
             "rem = remove, rand = add random, dag = add newer to older, with the share of citations changed.")
    r.p(f"<b>Against PageRank</b> ({r.fig('gain_main')}). The primary forest keeps the top 10 % more stable than PageRank in {wins[MAIN]['stay']} "
        f"of 9 settings (on average {signed(s['stay'])}), mainly when citations are added, while the order over all papers is the same on "
        f"average ({signed(s['spearman'], 3)}): better when citations are added, worse when they are removed.")
    rows = []
    for kind in KINDS:
        lv = 0.20
        simple_best = max(SIMPLE, key=lambda m: metric(full[MAIN], "rank", "spearman_mean", kind, lv, m))
        rows.append([KIND_LABEL[kind], f3(metric(full[MAIN], "rank", "spearman_mean", kind, lv, PRIMARY)),
                     f3(metric(full[MAIN], "rank", "spearman_mean", kind, lv, "pagerank")),
                     f"{metric(full[MAIN], 'rank', 'spearman_mean', kind, lv, simple_best):.3f} ({simple_best})",
                     f2(metric(full[MAIN], "core", "important_stay_in_top", kind, lv, PRIMARY)),
                     f2(metric(full[MAIN], "core", "important_stay_in_top", kind, lv, "pagerank"))])
    r.p(f"<b>Against the single measures</b> ({r.tab('summary_main')}). In none of the 9 settings is the primary forest more stable than the best "
        "of degree, k-core, closeness and betweenness.")
    r.table("summary_main", "PubMed-Diabetes at the strongest damage (20 % of the citations).",
            ["Damage", "Spearman primary forest", "Spearman PageRank", "Spearman best single measure", "Top-10 % stay primary forest",
             "Top-10 % stay PageRank"], rows)

    # ------------------------------------------------------------------ 5. answers (main)
    r.h2("5. Answers to the hypotheses (PubMed-Diabetes)")
    r.table("answers_main", "The three Phase A hypotheses on PubMed-Diabetes (primary forest, averages over the 9 damage settings).",
            ["Hypothesis", "Verdict", "Evidence"],
            [["H1: a combined score is more stable than single measures", "Partly",
              f"more stable than PageRank at the top in {wins[MAIN]['stay']} of 9 settings; never more stable than the best single measure"],
             ["H2: the forest copes with the failure of one feature by using the others", "Not supported",
              f"the control relies {pct(importance(MAIN, 'with_pagerank', 'pagerank'), 1)} on PageRank and is less stable than PageRank; "
              f"the primary forest relies {pct(importance(MAIN, 'without_pagerank', 'closeness'))} on closeness"],
             ["H3: the network core keeps its high rank", "Supported (adjusted core rule)",
              f"core papers stay in the top 10 % in {pct(core_stay[MAIN][0])} of the damaged graphs, other important papers in "
              f"{pct(core_stay[MAIN][1])}; Phase A's own core rule gives an empty core (decision D16)"]], numeric_from=9)
    r.p(f"{r.tab('answers_main')} sums up Part I: combining centralities makes the top of the ranking more robust than PageRank against "
        "added citations, but not against missing ones, and simple local measures remain the most stable.")

    # ================================================================== PART II
    r.raw('<h1 style="margin-top:34px;font-size:24px">Part II - How far do the results go?</h1>')
    r.p("The lecturer asked two questions about Part I: would the results be the same on other citation networks, and does the forest "
        "do better without closeness? Part II answers both with the same code, the same damage and the same 30 repeats.")

    # ------------------------------------------------------------------ 6. datasets
    r.h2("6. Two more citation networks")
    r.raw("<ul>" + "".join(f"<li>{cleaning_sentence(ds, facts[ds])}</li>" for ds in DATASETS) + "</ul>")
    r.p("All three are cleaned with the same steps in the same order; only the rule for citation loops differs, because Cora and CiteSeer "
        "have no time information (decision D18). In every dataset cleaning removed less than 5 % of the citations.")
    assert all(facts[d]["removed_citations"] / facts[d]["raw_citations"] < 0.05 for d in DATASETS)
    rows = [[DATASET_LABEL[ds], f"{facts[ds]['papers']:,}", f"{facts[ds]['citations']:,}", f"{facts[ds]['density']:.5f}",
             pct(facts[ds]["cite_nothing"]), pct(facts[ds]["never_cited"]), f"{facts[ds]['max_in']} / {facts[ds]['max_out']}",
             f"{facts[ds]['components']:,}"] for ds in DATASETS]
    r.table("data_all", "The three datasets after cleaning.",
            ["Dataset", "Papers", "Citations", "Density", "Cite nothing", "Never cited", "Max in / out", "Separate pieces"], rows)
    r.figure("degrees_all", figs.degrees({ds: graph_of(ds) for ds in DATASETS}),
             "Citations received and given, as a share of the papers of each dataset.")
    r.p(f"{r.tab('data_all')} and {r.fig('degrees_all')} show what the networks share: no loops, most papers cited once or never, a few heavily "
        "cited ones. They differ in size and connectivity: PubMed-Diabetes is about seven times larger than Cora and is one connected piece, "
        f"Cora has {facts['cora']['components']} and CiteSeer {facts['citeseer']['components']} separate pieces. In the two smaller networks far "
        "fewer papers cite nothing.")

    # ------------------------------------------------------------------ 7. results on the other networks
    r.h2("7. Do the results hold on Cora and CiteSeer?")
    rows = []
    for ds in DATASETS:
        sg = full[ds]["model"]["single_centrality_spearman_vs_pagerank"]
        best_name = max(sg, key=sg.get)
        rows.append([DATASET_LABEL[ds], f2(holdout(ds, "with_pagerank", "r2")), f2(holdout(ds, "without_pagerank", "r2")),
                     f2(holdout(ds, "without_pagerank", "spearman")), f"{sg[best_name]:.2f} ({best_name})"])
    r.table("models_all", "How well the forests predict PageRank on held-out papers in each dataset. The last column is the best single "
                          "centrality on its own (Spearman with PageRank).",
            ["Dataset", "R\u00b2 with PageRank (control)", "R\u00b2 without (primary)", "Spearman primary", "Best single centrality"], rows)
    r.figure("importance_all", figs.importance({ds: full[ds]["model"] for ds in DATASETS}),
             "What the forests rely on in each dataset. The control copies PageRank everywhere; the primary forest relies mostly on closeness "
             "in Cora and CiteSeer.")
    r.p(f"<b>The two forests</b> ({r.tab('models_all')}, {r.fig('importance_all')}). The control copies its PageRank input in all three datasets "
        f"({pct(min(importance(d, 'with_pagerank', 'pagerank') for d in DATASETS), 1)} or more of its weight), so the reason for the primary "
        "forest does not depend on the dataset. In Cora and CiteSeer closeness alone already follows PageRank closely, and the primary forest "
        "is mostly a copy of closeness.")
    r.figure("spearman_others", figs.curves(full, "rank", "spearman_mean", "mean Spearman", datasets=OTHERS),
             "Spearman with the base ranking in Cora and CiteSeer (compare with Figure "
             f"{r.fig_no('spearman_main')} for PubMed-Diabetes).")
    r.figure("stay_others", figs.curves(full, "core", "important_stay_in_top", "top-10 % stay", datasets=OTHERS),
             f"Top-10 % stay in Cora and CiteSeer (compare with Figure {r.fig_no('stay_main')}).")
    r.figure("gain_all", figs.difference(full, full, [("rank", "spearman_mean"), ("core", "important_stay_in_top")],
                                          ["primary forest minus PageRank\n(Spearman, all papers)",
                                           "primary forest minus PageRank\n(top-10 % stay)"], ref_method="pagerank"),
             "Primary forest minus PageRank in the three datasets. Bars above zero mean the forest is more stable.")
    r.p(f"<b>Stability</b> ({r.fig('spearman_others')}, {r.fig('stay_others')}, {r.fig('gain_all')}). The advantage over PageRank at the top of the "
        f"ranking is found only in PubMed-Diabetes. In Cora and CiteSeer the primary forest is not more stable than PageRank: the average "
        f"differences are {signed(mean_gap['cora']['spearman'], 3)} and {signed(mean_gap['citeseer']['spearman'], 3)} in Spearman and "
        f"{signed(mean_gap['cora']['stay'])} and {signed(mean_gap['citeseer']['stay'])} in top-10 % stay, and it beats PageRank at the top in "
        f"{wins['cora']['stay']} of 9 settings in both. As in PubMed-Diabetes, it is never more stable than the best single measure "
        f"({r.tab('summary_others')}).")
    rows = []
    for ds in OTHERS:
        for kind in KINDS:
            lv = 0.20
            simple_best = max(SIMPLE, key=lambda m: metric(full[ds], "rank", "spearman_mean", kind, lv, m))
            rows.append([DATASET_LABEL[ds], KIND_LABEL[kind], f3(metric(full[ds], "rank", "spearman_mean", kind, lv, PRIMARY)),
                         f3(metric(full[ds], "rank", "spearman_mean", kind, lv, "pagerank")),
                         f"{metric(full[ds], 'rank', 'spearman_mean', kind, lv, simple_best):.3f} ({simple_best})",
                         f2(metric(full[ds], "core", "important_stay_in_top", kind, lv, PRIMARY)),
                         f2(metric(full[ds], "core", "important_stay_in_top", kind, lv, "pagerank"))])
    r.table("summary_others", "Cora and CiteSeer at the strongest damage (20 % of the citations).",
            ["Dataset", "Damage", "Spearman primary forest", "Spearman PageRank", "Spearman best single measure",
             "Top-10 % stay primary forest", "Top-10 % stay PageRank"], rows, numeric_from=2)

    # ------------------------------------------------------------------ 8. closeness
    def change(ds, table, column):
        return [metric(reduced[ds], table, column, k, lv, PRIMARY) - metric(full[ds], table, column, k, lv, PRIMARY)
                for k, lv in SETTINGS]

    sp_change = {ds: sum(change(ds, "rank", "spearman_mean")) / 9 for ds in DATASETS}
    st_change = {ds: sum(change(ds, "core", "important_stay_in_top")) / 9 for ds in DATASETS}
    wins_red = {ds: sum(d > 0 for d in gap(reduced[ds], "core", "important_stay_in_top", "pagerank")) for ds in DATASETS}
    r2_drop = {ds: holdout(ds, "without_pagerank", "r2", full) - holdout(ds, "without_pagerank", "r2", reduced) for ds in DATASETS}
    assert all(v > 0.2 for v in r2_drop.values())
    assert sp_change["cora"] < 0 and sp_change["citeseer"] < 0 and sp_change["pubmed"] > 0

    r.h2("8. Does the forest do better without closeness?")
    r.p("<b>Why we tested it.</b> Closeness is the one feature whose Phase A formula had to be changed (decision D17), and it is the feature "
        "the primary forest relies on most. The test removes closeness completely: from the features, from both forests and from the list of "
        "single measures; nothing else changes. The test has its own settings and result folders; the main code still computes closeness.")
    r.figure("close_acc", figs.closeness_accuracy({d: full[d]["model"] for d in DATASETS}, {d: reduced[d]["model"] for d in DATASETS}),
             "How well the primary forest predicts PageRank on held-out papers, with and without closeness as an input.")
    rows = [[DATASET_LABEL[ds], pct(importance(ds, "without_pagerank", "closeness")),
             f"{holdout(ds, 'without_pagerank', 'r2', full):.2f} to {holdout(ds, 'without_pagerank', 'r2', reduced):.2f}",
             signed(sp_change[ds], 3), signed(st_change[ds], 3), f"{wins[ds]['stay']} to {wins_red[ds]} of 9"] for ds in DATASETS]
    r.table("closeness", "What removing closeness changes for the primary forest. The change columns are averages over the 9 damage settings "
                         "(negative = less stable without closeness). The last column counts the settings in which the forest keeps the top 10 % "
                         "more stable than PageRank, with and without closeness.",
            ["Dataset", "Closeness weight in the forest", "Hold-out R\u00b2", "Change in Spearman", "Change in top-10 % stay",
             "Settings better than PageRank"], rows)
    r.figure("close_gain", figs.difference(reduced, full, [("rank", "spearman_mean"), ("core", "important_stay_in_top")],
                                           ["without minus with closeness\n(Spearman, all papers)",
                                            "without minus with closeness\n(top-10 % stay)"], method_a=PRIMARY, method_b=PRIMARY),
             "Change in stability of the primary forest when closeness is removed. Bars above zero mean it is more stable without closeness.")
    r.p(f"<b>Accuracy.</b> In all three datasets the forest predicts PageRank clearly worse without closeness ({r.fig('close_acc')}): R\u00b2 falls by "
        f"{r2_drop['pubmed']:.2f} in PubMed-Diabetes, {r2_drop['cora']:.2f} in Cora and {r2_drop['citeseer']:.2f} in CiteSeer.")
    r.p(f"<b>Stability</b> ({r.tab('closeness')}, {r.fig('close_gain')}). The effect differs between datasets. In Cora and CiteSeer the forest is less "
        f"stable without closeness ({signed(sp_change['cora'], 3)} and {signed(sp_change['citeseer'], 3)} in Spearman). In PubMed-Diabetes the "
        f"order over all papers is slightly more stable ({signed(sp_change['pubmed'], 3)}) but the top 10 % is less stable "
        f"({signed(st_change['pubmed'], 3)}); this small gain comes with a forest that predicts PageRank far worse "
        f"(R\u00b2 {holdout('pubmed', 'without_pagerank', 'r2', reduced):.2f}) and leans on degree and k-core.")
    r.box("Answer to the lecturer's two questions", [
        "<b>Better results without closeness?</b> No for accuracy (all three datasets). For stability it depends on the dataset: worse in Cora "
        "and CiteSeer, mixed in PubMed-Diabetes. We recommend keeping closeness.",
        "<b>The same on other citation networks?</b> Partly. The reason for the primary forest and the loss of accuracy without closeness are "
        "the same in all three; the advantage over PageRank at the top of the ranking, and the effect of closeness on stability, are not."])

    # ------------------------------------------------------------------ 9. answers across datasets
    r.h2("9. The hypotheses on all three datasets")
    rows = [
        ["H1: a combined score is more stable than single measures",
         *[f"top 10 % better than PageRank in {wins[ds]['stay']} of 9; better than the best single measure in {best_single_wins[ds]} of 9"
           for ds in DATASETS]],
        ["H2: the forest copes with the failure of one feature",
         *[f"control: {pct(importance(ds, 'with_pagerank', 'pagerank'), 1)} on PageRank; primary: "
           f"{pct(importance(ds, 'without_pagerank', 'closeness'))} on closeness" for ds in DATASETS]],
        ["H3: the network core keeps its high rank (adjusted rule)",
         *[f"core {pct(core_stay[ds][0])} vs other important papers {pct(core_stay[ds][1])}" for ds in DATASETS]],
    ]
    r.table("answers_all", "The Phase A hypotheses on each dataset (primary forest, averages over the 9 damage settings).",
            ["Hypothesis", *[DATASET_LABEL[d] for d in DATASETS]], rows, numeric_from=9)
    r.p(f"{r.tab('answers_all')}: <b>H1</b> holds only in a narrow sense and only in PubMed-Diabetes (the top of the ranking against PageRank). "
        "<b>H2</b> is not supported in any dataset: the forest relies mostly on one feature. <b>H3</b> holds in all three with the adjusted core rule.")

    # ------------------------------------------------------------------ 10. limits
    r.h2("10. Limits of these results")
    r.raw("<ul>"
          "<li><b>Three datasets</b>, all small to medium citation graphs of scientific papers. Their differences show that one dataset is not "
          "enough for a general claim.</li>"
          "<li><b>One target.</b> The forests learn PageRank. Agreeing with PageRank is not the same as measuring true importance, and a "
          "ranking can be stable and still wrong.</li>"
          "<li><b>No dates.</b> Citation loops in Cora and CiteSeer were removed with the fewest edges, which is a choice (decision D18).</li>"
          "<li><b>Forest settings.</b> Every tree sees all features at every split (no feature randomness), unlike the usual Random Forest "
          "and unlike the argument of Phase A hypothesis H2.</li>"
          "<li><b>Trees cannot extrapolate.</b> After citations are added many scaled inputs leave the range seen in training, and the forest "
          "gives those papers the same score. Part of its stability is this effect.</li>"
          "<li><b>Training and testing on the same graph</b>, as in Phase A; the hold-out check only shows that the forest learned something.</li></ul>")

    r.h2("Data sources")
    r.raw("<ul class='note'>" + "".join(f"<li>{SOURCES[k]}.</li>" for k in ("pubmed", "cora", "citeseer", "linqs")) +
          "<li>Files: linqs-data.soe.ucsc.edu/public/lbc/ (Cora, CiteSeer) and the PubMed-Diabetes files given for the project.</li></ul>")
    r.h2("Reproducing this report")
    r.raw("<pre style='background:#f4f7fa;padding:10px;font-size:12px'>"
          "CITATION_DATASET=cora FEATURE_SET=full          python scripts/run_pipeline.py   # about 1 minute\n"
          "CITATION_DATASET=cora FEATURE_SET=no_closeness  python scripts/run_pipeline.py\n"
          "(the same for citeseer and pubmed; PubMed with all features can also be built with notebooks 01-08)\n"
          "python scripts/build_report.py                  # writes results/report/</pre>")

    # ------------------------------------------------------------------ summary at the top
    part1 = [
        f"The primary forest keeps the top 10 % of the ranking more stable than PageRank, the measure it learns, in {wins[MAIN]['stay']} of 9 "
        "settings, mainly when citations are added; over all papers it is as stable as PageRank on average.",
        "It is never more stable than the best simple measure (degree, k-core or closeness).",
        "The Phase A design, with PageRank as input and target, gives a forest that copies PageRank; that is why the primary forest has no "
        "PageRank input.",
    ]
    part2 = [
        "On Cora and CiteSeer the forest is not more stable than PageRank: it keeps the top 10 % more stable in none of the 9 settings "
        "in either dataset, so the PubMed-Diabetes result does not carry over.",
        "Without closeness the forest predicts PageRank much worse in all three datasets and is less stable in two of them: keep closeness.",
    ]
    r.parts[summary_index] = ('<div class="box"><b>Main findings - Part I (PubMed-Diabetes)</b><ul>' +
                              "".join(f"<li>{i}</li>" for i in part1) + "</ul><b>Part II (more datasets, no closeness)</b><ul>" +
                              "".join(f"<li>{i}</li>" for i in part2) + "</ul></div>")

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
