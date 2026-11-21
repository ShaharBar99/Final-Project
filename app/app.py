"""Interactive demo of the project.

Run from the project root:
    streamlit run app/app.py

Tabs:
    Damage the graph    - damage the citation graph and watch every ranking move
    Look up a paper     - one paper: its features and ranks before / after the damage
    Experiment results  - the results of the full experiment (270 damaged graphs)
    How it works        - the pipeline in short
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no GUI backend: the app draws in a server thread
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config, dashboard_report, demo, experiment, features, model, ranking, scaling  # noqa: E402

st.set_page_config(page_title="Node importance under damage", layout="wide")
st.markdown("""
<style>
  [data-testid="stSidebar"] [data-testid="stDownloadButton"] button {
    background: linear-gradient(135deg, #d33f49 0%, #a8202a 100%);
    color: white;
    border: 2px solid #8d1821;
    border-radius: 10px;
    box-shadow: 0 4px 12px rgba(168, 32, 42, 0.35);
    font-weight: 700;
    min-height: 3.25rem;
  }
  [data-testid="stSidebar"] [data-testid="stDownloadButton"] button:hover {
    background: linear-gradient(135deg, #e24d58 0%, #b72732 100%);
    border-color: #74131a;
    color: white;
    transform: translateY(-1px);
    box-shadow: 0 6px 16px rgba(168, 32, 42, 0.45);
  }
</style>
""", unsafe_allow_html=True)

ABOUT_TEXT = """
**Question.** Does a Random Forest that combines several centrality measures rank papers more
stably than any single measure when the citation data is incomplete or noisy?

**Pipeline** (every step is a module in `src/`):
1. **Clean graph** - 19,717 papers, 44,324 citations, no self-loops, no cycles.
2. **Features** - degree, betweenness, closeness, PageRank, k-core for every paper.
3. **Frozen scaler** - Min-Max fitted once on the original graph and never refitted.
4. **Random Forest** - the primary model predicts PageRank from the other four measures. A leakage
   control also receives PageRank itself and therefore mostly copies its target.
5. **Damage** - remove or add a share of the citations; the papers stay the same.
6. **Re-score** - recompute the features on the damaged graph, use the same scaler and the same trained forest.
7. **Compare** - Spearman / Kendall between rankings, how much of the top survives, per-paper variation.

**Main finding.** The primary forest is more stable than PageRank when citations are *added*, especially
at the top, but a little less stable when citations are *removed*. It is not the most robust method:
simple local measures (degree, k-core) are more stable overall. Agreement with its PageRank target does
not establish an independent or true notion of importance.
"""


# ------------------------------------------------------------------ cached work

@st.cache_resource(show_spinner="Loading the graph and the trained models...")
def pipeline():
    return demo.load_pipeline()


@st.cache_resource(show_spinner="Damaging the graph and scoring it again...", max_entries=20)
def damaged_run(kind, level, seed):
    g, _, scaler, forests, _ = pipeline()
    return demo.damage_and_score(g, scaler, forests, kind, level, seed)


@st.cache_resource(max_entries=20)
def scaled_features(kind, level, seed):
    """Scaled features before and after the damage (what the forest sees)."""
    g, _, scaler, _, _ = pipeline()
    damaged, _, _ = damaged_run(kind, level, seed)
    return (scaling.apply_scaler(scaler, features.compute_features(g)),
            scaling.apply_scaler(scaler, features.compute_features(damaged)))


g, nodes, scaler, forests, base = pipeline()

# ------------------------------------------------------------------ sidebar

st.sidebar.title("Damage settings")
kind = st.sidebar.radio("What happens to the citations?", list(demo.KIND_LABELS),
                        format_func=demo.KIND_LABELS.get, index=1)
level = st.sidebar.slider("How much? (% of the edges)", 1, 30, 10, step=1) / 100
seed = st.sidebar.number_input("Random seed", min_value=0, value=config.SEED, step=1)
method = st.sidebar.selectbox("Method to look at in detail", experiment.METHODS,
                              format_func=demo.METHOD_LABELS.get,
                              index=experiment.METHODS.index(model.PRIMARY_MODEL))
st.sidebar.caption("Same settings + same seed = same damaged graph. "
                   "The forests are never retrained on the damaged graph.")

damaged, scores, out_of_range = damaged_run(kind, level, int(seed))

st.title("How stable is the importance of a paper?")
st.write("PubMed-Diabetes citation network - 19,717 papers. Damage the graph on the left and see "
         "how much each way of ranking papers changes.")

tab_damage, tab_paper, tab_results, tab_about = st.tabs(
    ["Damage the graph", "Look up a paper", "Experiment results", "How it works"])

paper_details = None
paper_ranks = None
paper_features = None

# ------------------------------------------------------------------ tab 1

with tab_damage:
    change = demo.graph_change(g, damaged)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Citations", f"{change['edges after']:,}",
              f"{change['edges after'] - change['edges before']:+,}")
    c2.metric("Isolated papers", f"{change['papers with no edges']:,}")
    c3.metric("No citation loops", "yes" if change["still acyclic"] else "no")
    c4.metric("Out of training range",
              f"{max(out_of_range.values()):.0%}",
              help="Largest share of papers, over the 5 features, whose scaled value is outside "
                   "[0, 1]. The forest cannot extrapolate for these papers.")

    st.subheader("How much did each ranking move?")
    st.caption("1.0 = the ranking did not change. Top-100 kept = share of the top-100 papers still in the top 100.")
    damage_summary = demo.method_summary(base, scores)
    st.dataframe(damage_summary.style.format("{:.3f}").background_gradient(cmap="RdYlGn", vmin=0.4, vmax=1.0),
                 width="stretch")

    st.subheader(f"Top 20 papers of: {demo.METHOD_LABELS[method]}")
    st.caption("moved > 0: the paper went up; moved < 0: it went down. label = diabetes type (not used by the model).")
    top_papers = demo.rank_changes(base, scores, method, nodes, top=20)
    st.dataframe(top_papers, width="stretch")

    st.subheader("Every paper - rank before vs rank after")
    before = ranking.to_ranking(base[method])
    after = ranking.to_ranking(scores[method])
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(before, after, s=2, alpha=0.3)
    ax.plot([1, len(before)], [1, len(before)], "k--", lw=0.8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("rank before"); ax.set_ylabel("rank after")
    ax.set_title(demo.METHOD_LABELS[method])
    rank_plot_png = dashboard_report.figure_png(fig)
    st.pyplot(fig, width="content")
    st.caption("On the dashed line = did not move. Log scale, so the top of the ranking is on the left.")

# ------------------------------------------------------------------ tab 2

with tab_paper:
    top_examples = demo.rank_changes(base, scores, method, top=100)
    fallers = top_examples.sort_values("moved").index[:5].tolist()
    st.write("Enter a PubMed ID. Papers from the top 100 of the selected method that fell the most "
             f"with the current damage: {', '.join(fallers)}")
    paper = st.text_input("PubMed ID", value=fallers[0]).strip()

    if paper not in g:
        st.warning("This ID is not in the dataset.")
    else:
        paper_details = (f"Paper {paper} - label {nodes.set_index('paper_id').loc[paper, 'label']}; "
                         f"cited by {g.in_degree(paper)} -> {damaged.in_degree(paper)} papers, "
                         f"cites {g.out_degree(paper)} -> {damaged.out_degree(paper)} papers "
                         f"(before -> after the damage)")
        st.markdown(f"**{paper_details}**")
        paper_ranks = pd.DataFrame({
            "rank before": {demo.METHOD_LABELS[m]: ranking.to_ranking(base[m])[paper] for m in experiment.METHODS},
            "rank after": {demo.METHOD_LABELS[m]: ranking.to_ranking(scores[m])[paper] for m in experiment.METHODS},
        })
        paper_ranks["moved"] = paper_ranks["rank before"] - paper_ranks["rank after"]
        left, right = st.columns(2)
        left.write("Rank in every method (1 = most important)")
        left.dataframe(paper_ranks.style.format("{:,.0f}"), width="stretch")
        f_before, f_after = scaled_features(kind, level, int(seed))
        right.write("Scaled features - what the forest sees")
        paper_features = pd.DataFrame({"before": f_before.loc[paper], "after": f_after.loc[paper]})
        right.dataframe(paper_features.style.format("{:.4f}"), width="stretch")

# ------------------------------------------------------------------ tab 3

with tab_results:
    tables = config.RESULTS_DIR / "tables"
    st.write("The full experiment: 3 kinds of damage x 3 levels x 30 runs = 270 damaged graphs. "
             "These numbers are averages over the 30 runs (notebooks 06-08).")
    ranking_table = pd.read_csv(tables / "ranking_stability.csv")
    metrics = {
        "spearman_mean": "Spearman with the base ranking (all papers)",
        "kendall_mean": "Kendall with the base ranking (all papers)",
        "spearman_top10pct_mean": "Spearman over the method's own top 10 %",
        "top100_kept": "Share of the top 100 still in the top 100",
        "top1000_kept": "Share of the top 1000 still in the top 1000",
    }
    metric = st.selectbox("Measure", list(metrics), format_func=metrics.get)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, k in zip(axes, ["remove", "add_dag", "add_random"]):
        t = ranking_table[ranking_table.kind == k].pivot(index="level", columns="method", values=metric)
        for m in experiment.SCIENTIFIC_REPORT_ORDER:
            style = "-o" if m.startswith("rf_") else "--."
            ax.plot(t.index * 100, t[m], style, label=demo.METHOD_LABELS[m])
        ax.set_title(k); ax.set_xlabel("% of edges changed"); ax.set_xticks([5, 10, 20])
    axes[0].set_ylabel(metrics[metric])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.12))
    plt.tight_layout()
    experiment_plot_png = dashboard_report.figure_png(fig)
    st.pyplot(fig, width="stretch")
    st.caption("Solid lines = the primary forest and leakage control; dashed = five centrality baselines. "
               "Predictive agreement with PageRank and robustness to damage are different questions.")

    core_path = tables / "core_summary.csv"
    core_view = None
    if core_path.exists():
        st.subheader("Do the important papers stay important?")
        st.caption("Share of runs in which a top-10 % paper is still in the top 10 %.")
        core_table = pd.read_csv(core_path)
        core_view = core_table.pivot_table(index="method", columns=["kind", "level"], values="important_stay_in_top")
        core_view.index = [demo.METHOD_LABELS[m] for m in core_view.index]
        st.dataframe(core_view.style.format("{:.2f}").background_gradient(cmap="RdYlGn", vmin=0.3, vmax=1.0),
                     width="stretch")

    summary_file = config.DOCS_DIR / "results_summary.md"
    results_summary = "Results summary is not available."
    if summary_file.exists():
        results_summary = summary_file.read_text(encoding="utf-8")
        with st.expander("Research questions and hypotheses - full write-up"):
            st.markdown(results_summary)

    report_html = dashboard_report.build_dashboard_html(
        settings={
            "damage_type": demo.KIND_LABELS[kind],
            "level": level,
            "seed": int(seed),
        },
        graph_change=change,
        out_of_range=out_of_range,
        damage_summary=damage_summary,
        top_papers=top_papers,
        rank_plot_png=rank_plot_png,
        method_label=demo.METHOD_LABELS[method],
        paper_id=paper,
        paper_details=paper_details,
        paper_ranks=paper_ranks,
        paper_features=paper_features,
        experiment_metric=metrics[metric],
        experiment_plot_png=experiment_plot_png,
        core_table=core_view,
        results_summary=results_summary,
        about_text=ABOUT_TEXT,
    )
    st.sidebar.divider()
    st.sidebar.markdown("### 📥 Export this dashboard")
    st.sidebar.caption("Includes all four sections, the current paper lookup, tables, and embedded charts.")
    st.sidebar.download_button(
        "Download complete HTML report",
        data=report_html,
        file_name="citation_network_dashboard.html",
        mime="text/html",
        help="A self-contained report with all four tabs and embedded images. Upload it to Google "
             "Drive, open it in a browser, or use its Print / Save as PDF button.",
        type="primary",
        icon=":material/download:",
        width="stretch",
        on_click="ignore",
    )

# ------------------------------------------------------------------ tab 4

with tab_about:
    st.markdown(ABOUT_TEXT)
    st.markdown("Braude College of Engineering - Final project, Phase B. "
                "Niran Czuckermann and Shahar Bar.  \nAdvisor: Dr. Dvora Toledano-Kitai.")
