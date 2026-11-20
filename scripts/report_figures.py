"""Figures of the results report. Every function returns a matplotlib figure."""
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

from report_data import (CONTROL, DATASETS, DATASET_LABEL, KIND_LABEL, KINDS, LEVELS, METHOD_COLOR,
                         METHOD_LABEL, PRIMARY, metric)

DATASET_COLOR = {"pubmed": "#264653", "cora": "#2A9D8F", "citeseer": "#E9C46A"}
ORDER = [PRIMARY, CONTROL, "pagerank", "degree", "kcore", "closeness", "betweenness"]

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 10, "axes.titleweight": "bold", "figure.dpi": 100})


# ------------------------------------------------------------------ Figure: algorithm flow

def flow():
    fig, ax = plt.subplots(figsize=(11.2, 4.7))
    ax.set_xlim(0, 11.2)
    ax.set_ylim(0, 4.8)
    ax.axis("off")
    width, height, gap = 1.62, 1.25, 0.28
    xs = [0.15 + i * (width + gap) for i in range(6)]
    top_y, bottom_y = 3.0, 0.85

    top = [("Cleaned\ncitation graph", "papers = nodes\ncitations = edges"),
           ("Five\ncentralities", "degree, betweenness,\ncloseness, PageRank,\nk-core"),
           ("Min-Max\nscaler", "fitted once on this\ngraph (frozen)"),
           ("Random\nForest", "trained to predict\nPageRank\n(two versions)"),
           ("Base\nscores", "7 methods:\n2 forests +\n5 single measures"),
           ("Base\nranking", "rank 1 = most\nimportant; ties\nshare the mean rank")]
    bottom = [("Damage\nthe graph", "remove or add\ncitations:\n5, 10 or 20 %"),
              ("Recompute\nfeatures", "on the damaged\ngraph"),
              ("Same frozen\nscaler", "not fitted again"),
              ("Same\nforest", "not retrained"),
              ("New\nranking", "of every method"),
              ("Compare", "Spearman,\ntop-10 % stay,\ninstability (CV)")]

    def box(x, y, title, body, face):
        ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc=face, ec="#13293D", lw=1.0))
        ax.text(x + width / 2, y + height - 0.30, title, ha="center", va="center", fontsize=8.2,
                fontweight="bold", color="#13293D", linespacing=1.15)
        ax.text(x + width / 2, y + 0.36, body, ha="center", va="center", fontsize=7, color="#2B3A47", linespacing=1.15)

    def arrow(x0, y0, x1, y1, style="-|>", dashed=False):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle=style, color="#13293D", lw=1.1,
                                    linestyle=(0, (4, 3)) if dashed else "-"))

    for i, (title, body) in enumerate(top):
        box(xs[i], top_y, title, body, "#E8F0F7")
    for i, (title, body) in enumerate(bottom):
        box(xs[i], bottom_y, title, body, "#FDF1E3")
    for i in range(5):
        arrow(xs[i] + width + 0.03, top_y + height / 2, xs[i + 1] - 0.03, top_y + height / 2)
        arrow(xs[i] + width + 0.03, bottom_y + height / 2, xs[i + 1] - 0.03, bottom_y + height / 2)
    # what is reused from the top lane
    for i in (2, 3, 5):
        arrow(xs[i] + width / 2, top_y - 0.04, xs[i] + width / 2, bottom_y + height + 0.04, dashed=True)
    ax.text(0.15, 4.55, "A  Once, on the original graph", fontsize=9.5, fontweight="bold", color="#13293D")
    ax.text(0.15, 2.55, "B  Repeated for every damaged graph: 3 kinds x 3 levels x 30 repeats = 270 graphs per dataset",
            fontsize=9.5, fontweight="bold", color="#8A4B08")
    ax.text(xs[2] + width / 2 + 0.07, 2.33, "reused", fontsize=7, color="#5B6770", va="center")
    ax.text(xs[3] + width / 2 + 0.07, 2.33, "reused", fontsize=7, color="#5B6770", va="center")
    ax.text(xs[5] + width / 2 + 0.07, 2.33, "reference", fontsize=7, color="#5B6770", va="center")
    return fig


# ------------------------------------------------------------------ Figure: degrees of the datasets

def degrees(graphs):
    """graphs: dict dataset -> DataFrame of edges (source, target) and node list."""
    groups = [-1, 0, 1, 2, 5, 10, 20, 10 ** 6]
    names = ["0", "1", "2", "3-5", "6-10", "11-20", "21+"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), sharey=True)
    x = np.arange(len(names))
    w = 0.26
    for ax, which, title in zip(axes, ("in", "out"), ("In-degree: times a paper is cited",
                                                       "Out-degree: papers it cites")):
        for k, ds in enumerate(DATASETS):
            nodes, edges = graphs[ds]
            col = "target" if which == "in" else "source"
            deg = edges[col].value_counts().reindex(nodes).fillna(0)
            share = pd.cut(deg, groups, labels=names).value_counts(normalize=True).reindex(names) * 100
            bars = ax.bar(x + (k - 1) * w, share.values, w, color=DATASET_COLOR[ds], label=DATASET_LABEL[ds])
            if which == "out":
                ax.text(x[0] + (k - 1) * w, share.values[0] + 1.5, f"{share.values[0]:.0f}%", ha="center", fontsize=7)
            else:
                ax.text(x[1] + (k - 1) * w, share.values[1] + 1.5, f"{share.values[1]:.0f}%", ha="center", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_xlabel("degree of the paper")
        ax.set_title(title)
    axes[0].set_ylabel("share of papers (%)")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ Figure: what the forests rely on

def importance(models):
    """models: dict dataset -> model_report of the full feature set."""
    features = ["degree", "betweenness", "closeness", "pagerank", "kcore"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.2), sharey=True)
    for ax, version, title in zip(axes, ("without_pagerank", "with_pagerank"),
                                  ("Forest without PageRank (primary)", "Forest with PageRank (control)")):
        for row, ds in enumerate(DATASETS):
            imp = models[ds]["feature_importance"][version]
            left = 0
            for f in features:
                value = imp.get(f, 0.0)
                ax.barh(row, value, left=left, color=METHOD_COLOR.get(f, "#999"), edgecolor="white",
                        label=f if row == 0 else None)
                if value >= 0.07:
                    ax.text(left + value / 2, row, f"{value:.0%}", ha="center", va="center", fontsize=7.5,
                            color="white" if f in ("degree", "closeness", "kcore") and f != "kcore" else "black")
                left += value
        ax.set_yticks(range(3))
        ax.set_yticklabels([DATASET_LABEL[d] for d in DATASETS])
        ax.set_xlim(0, 1)
        ax.set_xlabel("share of the forest's decisions")
        ax.set_title(title)
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=5, frameon=False, loc="lower center",
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return fig


# ------------------------------------------------------------------ Figures: stability curves, datasets x damage kinds

def curves(runs, table, column, ylabel):
    """ylabel is written under the dataset name."""
    """3 x 3 grid: rows = datasets, columns = kinds of damage, one line per method."""
    fig, axes = plt.subplots(3, 3, figsize=(11, 8.6), sharex=True)
    for r, ds in enumerate(DATASETS):
        lo = 1.0
        for c, kind in enumerate(KINDS):
            ax = axes[r, c]
            for m in ORDER:
                y = [metric(runs[ds], table, column, kind, lv, m) for lv in LEVELS]
                lo = min(lo, np.nanmin(y))
                primary = m == PRIMARY
                ax.plot([lv * 100 for lv in LEVELS], y, marker="o" if m in (PRIMARY, CONTROL, "pagerank") else ".",
                        ls=":" if m == CONTROL else ("-" if m in (PRIMARY, "pagerank") else "--"),
                        lw=2.6 if primary else 1.3, ms=5 if primary else 3.5, color=METHOD_COLOR[m],
                        label=METHOD_LABEL[m], zorder=5 if primary else 2)
            if r == 0:
                ax.set_title(KIND_LABEL[kind])
            if c == 0:
                ax.set_ylabel(f"{DATASET_LABEL[ds]}\n{ylabel}")
            if r == 2:
                ax.set_xlabel("% of citations changed")
            ax.set_xticks([5, 10, 20])
            ax.grid(alpha=0.25)
        for c in range(3):
            axes[r, c].set_ylim(max(0.0, lo - 0.04), 1.01)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig


# ------------------------------------------------------------------ Figure: primary forest minus a reference

def difference(runs_a, runs_b, panels, ylabels, title_a=None, ref_method=None, method_a=PRIMARY, method_b=None):
    """Bars: value of method_a in runs_a minus value of method_b in runs_b, for 9 settings x 3 datasets.

    panels: list of (table, column). runs_*: dict dataset -> run.
    """
    settings = [(k, lv) for k in KINDS for lv in LEVELS]
    fig, axes = plt.subplots(1, len(panels), figsize=(5.6 * len(panels), 3.8), sharex=True)
    axes = np.atleast_1d(axes)
    x = np.arange(len(settings))
    w = 0.26
    for ax, (table, column), ylabel in zip(axes, panels, ylabels):
        for k, ds in enumerate(DATASETS):
            vals = [metric(runs_a[ds], table, column, kind, lv, method_a)
                    - metric(runs_b[ds], table, column, kind, lv, method_b or ref_method) for kind, lv in settings]
            ax.bar(x + (k - 1) * w, vals, w, color=DATASET_COLOR[ds], label=DATASET_LABEL[ds])
        ax.axhline(0, color="#13293D", lw=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{ {'remove': 'rem', 'add_random': 'rand', 'add_dag': 'dag'}[k]} {int(lv * 100)}%"
                            for k, lv in settings], rotation=60, ha="right", fontsize=7.5)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ Figure: accuracy with / without closeness

def closeness_accuracy(full, reduced):
    """Bars: hold-out R^2 and Spearman of the primary forest, with and without closeness."""
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4), sharey=True)
    x = np.arange(len(DATASETS))
    for ax, key, title in zip(axes, ("r2", "spearman"), ("Hold-out R²", "Hold-out Spearman")):
        a = [full[d]["holdout"]["without_pagerank"][key] for d in DATASETS]
        b = [reduced[d]["holdout"]["without_pagerank"][key] for d in DATASETS]
        ax.bar(x - 0.19, a, 0.38, color="#D55E00", label="with closeness")
        ax.bar(x + 0.19, b, 0.38, color="#7F7F7F", label="without closeness")
        for i, (u, v) in enumerate(zip(a, b)):
            ax.text(i - 0.19, u + 0.015, f"{u:.2f}", ha="center", fontsize=8)
            ax.text(i + 0.19, v + 0.015, f"{v:.2f}", ha="center", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([DATASET_LABEL[d] for d in DATASETS])
        ax.set_ylim(0, 1.1)
        ax.set_title(title)
    axes[0].set_ylabel("primary forest vs PageRank, held-out papers")
    axes[0].legend(frameon=False, loc="lower left")
    fig.tight_layout()
    return fig
