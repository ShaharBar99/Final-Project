"""Create a self-contained, printable HTML snapshot of the Streamlit dashboard."""
from __future__ import annotations

import base64
from html import escape
from io import BytesIO
import re

import pandas as pd


def figure_png(figure) -> bytes:
    """Render a matplotlib figure to PNG bytes for embedding in the report."""
    output = BytesIO()
    figure.savefig(output, format="png", dpi=160, bbox_inches="tight")
    return output.getvalue()


def _image(png: bytes, alt: str) -> str:
    encoded = base64.b64encode(png).decode("ascii")
    return f'<img src="data:image/png;base64,{encoded}" alt="{escape(alt)}">'


def _table(table: pd.DataFrame | None, index: bool = True) -> str:
    if table is None:
        return '<p class="muted">Not available for the current selection.</p>'
    return table.to_html(
        classes="report-table",
        border=0,
        index=index,
        escape=True,
        float_format=lambda value: f"{value:.4f}",
    )


def _cards(values: dict) -> str:
    return '<div class="cards">' + "".join(
        f'<div class="card"><span>{escape(str(label))}</span>'
        f'<strong>{escape(str(value))}</strong></div>'
        for label, value in values.items()
    ) + "</div>"


def _markdown(text: str) -> str:
    """Render the small Markdown subset used by the project documentation."""
    rendered = escape(text)
    rendered = re.sub(r"`([^`]+)`", r"<code>\1</code>", rendered)
    rendered = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", rendered)
    rendered = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", rendered)
    rendered = re.sub(
        r"^(#{1,6})\s+(.+)$",
        lambda match: f"<h{len(match.group(1)) + 2}>{match.group(2)}</h{len(match.group(1)) + 2}>",
        rendered,
        flags=re.MULTILINE,
    )
    return rendered


def build_dashboard_html(*, settings: dict, graph_change: dict, out_of_range: dict,
                         damage_summary: pd.DataFrame, top_papers: pd.DataFrame,
                         rank_plot_png: bytes, method_label: str,
                         paper_id: str, paper_details: str | None,
                         paper_ranks: pd.DataFrame | None,
                         paper_features: pd.DataFrame | None,
                         experiment_metric: str, experiment_plot_png: bytes,
                         core_table: pd.DataFrame | None,
                         results_summary: str, about_text: str) -> bytes:
    """Return a complete HTML document representing all four dashboard tabs."""
    setting_cards = {
        "Damage type": settings["damage_type"],
        "Edges changed": f'{settings["level"]:.0%}',
        "Random seed": settings["seed"],
        "Selected method": method_label,
    }
    change_cards = {
        "Citations before": f'{graph_change["edges before"]:,}',
        "Citations after": f'{graph_change["edges after"]:,}',
        "Isolated papers": f'{graph_change["papers with no edges"]:,}',
        "No citation loops": "Yes" if graph_change["still acyclic"] else "No",
        "Largest out-of-range share": f'{max(out_of_range.values()):.1%}',
    }
    paper_content = (
        f'<p>{escape(paper_details)}</p>{_table(paper_ranks)}'
        f'<h3>Scaled features before and after damage</h3>{_table(paper_features)}'
        if paper_details else
        f'<p class="warning">Paper {escape(paper_id)} is not in the dataset.</p>'
    )

    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Citation-network dashboard report</title>
<style>
  :root {{ --ink:#17212b; --muted:#5d6975; --line:#d8dee5; --panel:#f4f7fa;
           --accent:#bc6c25; --blue:#315f7d; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:#eef2f5; color:var(--ink); font:15px/1.5 Arial,sans-serif; }}
  main {{ width:min(1120px,calc(100% - 32px)); margin:24px auto; background:white;
          padding:42px 48px; box-shadow:0 3px 18px #0002; }}
  h1 {{ margin:0; font-size:32px; }} h2 {{ color:var(--blue); border-bottom:2px solid var(--line);
       padding-bottom:8px; margin-top:38px; }} h3 {{ margin-top:24px; }}
  .subtitle,.muted {{ color:var(--muted); }}
  .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:12px; margin:18px 0; }}
  .card {{ background:var(--panel); border-left:4px solid var(--accent); padding:12px 14px; }}
  .card span {{ color:var(--muted); display:block; font-size:12px; text-transform:uppercase; }}
  .card strong {{ display:block; font-size:20px; margin-top:4px; overflow-wrap:anywhere; }}
  img {{ display:block; max-width:100%; height:auto; margin:18px auto; }}
  .report-table {{ width:100%; border-collapse:collapse; font-size:13px; margin:14px 0 24px; }}
  .report-table th,.report-table td {{ border:1px solid var(--line); padding:7px 9px; text-align:right; }}
  .report-table th:first-child,.report-table td:first-child {{ text-align:left; }}
  .report-table thead th {{ background:#e9eff4; }}
  .prose {{ white-space:pre-wrap; overflow-wrap:anywhere; background:var(--panel); padding:18px;
            border:1px solid var(--line); font:14px/1.55 Arial,sans-serif; }}
  .warning {{ background:#fff3cd; padding:12px; border-left:4px solid #d39e00; }}
  .print {{ position:fixed; top:12px; right:12px; border:0; border-radius:5px; padding:10px 14px;
            background:var(--blue); color:white; cursor:pointer; }}
  section {{ break-inside:auto; }}
  @media print {{ body {{ background:white; }} main {{ width:100%; margin:0; padding:18mm; box-shadow:none; }}
    .print {{ display:none; }} section {{ break-before:page; }} section:first-of-type {{ break-before:auto; }} }}
</style>
</head>
<body>
<button class="print" onclick="window.print()">Print / Save as PDF</button>
<main>
  <header><h1>How stable is the importance of a paper?</h1>
  <p class="subtitle">Complete dashboard snapshot — PubMed-Diabetes citation network</p></header>
  {_cards(setting_cards)}

  <section id="damage">
    <h2>1. Damage the graph</h2>
    {_cards(change_cards)}
    <h3>How much did each ranking move?</h3>
    {_table(damage_summary)}
    <h3>Top 20 papers — {escape(method_label)}</h3>
    {_table(top_papers)}
    <h3>Every paper — rank before versus rank after</h3>
    {_image(rank_plot_png, "Rank before versus rank after")}
  </section>

  <section id="paper">
    <h2>2. Look up a paper</h2>
    <h3>Paper {escape(paper_id)}</h3>
    {paper_content}
  </section>

  <section id="experiment">
    <h2>3. Experiment results</h2>
    <p>Full experiment: 3 damage types × 3 levels × 30 repetitions = 270 perturbed graphs.</p>
    <h3>{escape(experiment_metric)}</h3>
    {_image(experiment_plot_png, experiment_metric)}
    <h3>Do important papers stay important?</h3>
    {_table(core_table)}
    <h3>Research questions and hypotheses</h3>
    <div class="prose">{_markdown(results_summary)}</div>
  </section>

  <section id="method">
    <h2>4. How it works</h2>
    <div class="prose">{_markdown(about_text)}</div>
  </section>
</main>
</body>
</html>
"""
    return document.encode("utf-8")
