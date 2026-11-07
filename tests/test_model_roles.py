from src import demo, experiment, features, model


def test_primary_model_excludes_pagerank_but_predicts_it():
    spec = model.MODEL_ROLES[model.PRIMARY_MODEL]
    assert spec["role"] == "primary"
    assert "pagerank" not in spec["features"]
    assert spec["target"] == "pagerank" == model.TARGET


def test_control_model_includes_target_as_direct_leakage():
    spec = model.MODEL_ROLES[model.CONTROL_MODEL]
    assert spec["role"] == "leakage_control"
    assert "pagerank" in spec["features"]
    assert spec["target"] == "pagerank" == model.TARGET


def test_reports_label_primary_and_control_roles():
    assert "primary" in demo.METHOD_LABELS[model.PRIMARY_MODEL].lower()
    assert "leakage control" in demo.METHOD_LABELS[model.CONTROL_MODEL].lower()
    assert experiment.SCIENTIFIC_REPORT_ORDER[0] == model.PRIMARY_MODEL
    assert experiment.SCIENTIFIC_REPORT_ORDER[-1] == model.CONTROL_MODEL


def test_existing_methods_and_score_column_order_are_preserved():
    expected = [
        "rf_with_pagerank",
        "rf_without_pagerank",
        "degree",
        "betweenness",
        "closeness",
        "pagerank",
        "kcore",
    ]
    assert experiment.METHODS == expected
    assert experiment.BASELINE_METHODS == features.FEATURE_NAMES
    assert set(experiment.BASELINE_METHODS) == {
        "degree", "betweenness", "closeness", "pagerank", "kcore"
    }
