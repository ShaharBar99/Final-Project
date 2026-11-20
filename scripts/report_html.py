"""Small HTML helpers for the results report: numbered figures and tables, references, page style."""
import base64
import html
import io

CSS = """
:root { --ink:#17212b; --muted:#5d6975; --line:#d8dee5; --panel:#f4f7fa; --accent:#bc6c25; --blue:#1f4e79; }
* { box-sizing: border-box; }
body { margin:0; background:#eef2f5; color:var(--ink); font:14px/1.55 Calibri, Arial, sans-serif; }
main { width:min(1000px, calc(100% - 32px)); margin:20px auto; background:white; padding:40px 52px;
       box-shadow:0 3px 18px #0002; }
h1 { margin:0 0 4px; font-size:28px; color:var(--blue); }
h2 { color:var(--blue); border-bottom:2px solid var(--line); padding-bottom:6px; margin-top:34px; font-size:20px; }
h3 { margin:20px 0 6px; font-size:15px; }
.sub { color:var(--muted); margin:0 0 14px; }
p { margin:7px 0; }
figure { margin:16px 0 20px; break-inside:avoid; }
figure img { display:block; max-width:100%; height:auto; margin:0 auto; }
figcaption, .caption { font-size:12.5px; color:#2b3a47; margin-top:6px; }
.caption { font-weight:bold; margin:14px 0 4px; }
table { border-collapse:collapse; width:100%; font-size:12.5px; margin:4px 0 18px; break-inside:avoid; }
th, td { border:1px solid var(--line); padding:5px 8px; text-align:left; vertical-align:top; }
th { background:#e9eff4; }
td.num, th.num { text-align:right; font-variant-numeric:tabular-nums; }
.box { background:var(--panel); border-left:4px solid var(--accent); padding:10px 14px; margin:12px 0; }
.box ul { margin:4px 0 2px 18px; padding:0; }
.note { color:var(--muted); font-size:12.5px; }
a.ref { color:var(--blue); text-decoration:none; }
.btn { position:fixed; top:12px; right:12px; border:0; border-radius:5px; padding:9px 13px; background:var(--blue);
       color:white; cursor:pointer; }
@page { size:A4; margin:16mm 14mm; }
@media print { body { background:white; } main { width:100%; margin:0; padding:0; box-shadow:none; }
  .btn { display:none; } h2 { break-after:avoid; } }
"""


def to_data_uri(fig, dpi=130):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


class Report:
    """Collects the page. Figures and tables are numbered in the order given at the start."""

    def __init__(self, figure_order, table_order):
        self.figure_order = list(figure_order)
        self.table_order = list(table_order)
        self.parts = []
        self.defined = set()
        self.referenced = set()

    # ---- numbering and references
    def fig_no(self, key):
        return self.figure_order.index(key) + 1

    def tab_no(self, key):
        return self.table_order.index(key) + 1

    def fig(self, key):
        """Reference to a figure, for use inside text: 'Figure 3'."""
        self.referenced.add(("fig", key))
        return f'<a class="ref" href="#fig-{key}">Figure {self.fig_no(key)}</a>'

    def tab(self, key):
        self.referenced.add(("tab", key))
        return f'<a class="ref" href="#tab-{key}">Table {self.tab_no(key)}</a>'

    # ---- content
    def h1(self, text):
        self.parts.append(f"<h1>{text}</h1>")

    def h2(self, text):
        self.parts.append(f"<h2>{text}</h2>")

    def h3(self, text):
        self.parts.append(f"<h3>{text}</h3>")

    def p(self, text, cls=""):
        self.parts.append(f'<p class="{cls}">{text}</p>' if cls else f"<p>{text}</p>")

    def raw(self, text):
        self.parts.append(text)

    def box(self, title, items):
        li = "".join(f"<li>{i}</li>" for i in items)
        self.parts.append(f'<div class="box"><b>{title}</b><ul>{li}</ul></div>')

    def figure(self, key, fig, caption):
        assert key not in self.defined, key
        self.defined.add(key)
        number = self.fig_no(key)
        self.parts.append(
            f'<figure id="fig-{key}"><img src="{to_data_uri(fig)}" alt="Figure {number}">'
            f"<figcaption><b>Figure {number}.</b> {caption}</figcaption></figure>")

    def table(self, key, caption, header, rows, numeric_from=1):
        """rows: list of lists of already formatted strings. Columns from numeric_from on are right-aligned."""
        assert key not in self.defined, key
        self.defined.add(key)
        number = self.tab_no(key)
        head = "".join(f'<th class="{"num" if i >= numeric_from else ""}">{h}</th>' for i, h in enumerate(header))
        body = "".join(
            "<tr>" + "".join(f'<td class="{"num" if i >= numeric_from else ""}">{c}</td>' for i, c in enumerate(r)) + "</tr>"
            for r in rows)
        self.parts.append(f'<div class="caption" id="tab-{key}">Table {number}. {caption}</div>'
                          f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>")

    def check(self):
        figures = {("fig", k) for k in self.figure_order}
        tables = {("tab", k) for k in self.table_order}
        defined = {("fig", k) for k in self.figure_order if k in self.defined} | \
                  {("tab", k) for k in self.table_order if k in self.defined}
        assert defined == figures | tables, f"declared but not built: {(figures | tables) - defined}"
        assert self.referenced >= figures | tables, f"never mentioned in the text: {(figures | tables) - self.referenced}"

    def render(self, title):
        self.check()
        return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" '
                f'content="width=device-width, initial-scale=1"><title>{html.escape(title)}</title>'
                f"<style>{CSS}</style></head><body>"
                '<button class="btn" onclick="window.print()">Print / Save as PDF</button>'
                f"<main>{''.join(self.parts)}</main></body></html>")
