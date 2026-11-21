import base64

import pandas as pd

from src import dashboard_report


def sample_report(**overrides):
    arguments = {
        "settings": {"damage_type": "Add citations", "level": 0.1, "seed": 42},
        "graph_change": {
            "edges before": 10,
            "edges after": 11,
            "papers with no edges": 2,
            "still acyclic": True,
        },
        "out_of_range": {"degree": 0.1, "pagerank": 0.2},
        "damage_summary": pd.DataFrame({"Spearman": [0.9]}, index=["Primary RF"]),
        "top_papers": pd.DataFrame({"rank before": [1], "rank after": [2]}, index=["123"]),
        "rank_plot_png": b"rank-image",
        "method_label": "Primary RF",
        "paper_id": "123",
        "paper_details": "Paper 123 details",
        "paper_ranks": pd.DataFrame({"rank before": [1]}, index=["PageRank"]),
        "paper_features": pd.DataFrame({"before": [0.2], "after": [0.3]}, index=["degree"]),
        "experiment_metric": "Spearman stability",
        "experiment_plot_png": b"experiment-image",
        "core_table": pd.DataFrame({"remove": [0.8]}, index=["Primary RF"]),
        "results_summary": "Mixed findings: better under addition, worse under removal.",
        "about_text": "The scaler and models remain frozen.",
    }
    arguments.update(overrides)
    return dashboard_report.build_dashboard_html(**arguments).decode("utf-8")


def test_report_contains_all_dashboard_sections_and_current_values():
    report = sample_report()
    assert "1. Damage the graph" in report
    assert "2. Look up a paper" in report
    assert "3. Experiment results" in report
    assert "4. How it works" in report
    assert "Paper 123 details" in report
    assert "Mixed findings" in report
    assert "data:image/png;base64," + base64.b64encode(b"rank-image").decode("ascii") in report
    assert "data:image/png;base64," + base64.b64encode(b"experiment-image").decode("ascii") in report
    assert "Print / Save as PDF" in report
    assert "Print / Save as PDF" in report


def test_report_embeds_both_images_as_data_uris():
    report = sample_report()
    assert base64.b64encode(b"rank-image").decode("ascii") in report
    assert base64.b64encode(b"experiment-image").decode("ascii") in report
    assert report.count("data:image/png;base64,") == 2


def test_report_escapes_user_controlled_text():
    report = sample_report(paper_id='<script>alert("x")</script>',
                           paper_details="<b>not markup</b>")
    assert '<script>alert("x")</script>' not in report
    assert "&lt;script&gt;" in report
    assert "&lt;b&gt;not markup&lt;/b&gt;" in report


def test_missing_paper_is_reported_without_tables():
    report = sample_report(paper_id="missing", paper_details=None,
                           paper_ranks=None, paper_features=None)
    assert "Paper missing is not in the dataset." in report
