"""Tests for src.evaluation."""
import pytest

from src.evaluation import evaluate

LABELS = ["a", "b", "c"]


def test_metrics_on_known_predictions():
    y_true = ["a", "a", "a", "b", "b", "c"]
    y_pred = ["a", "a", "b", "b", "b", "a"]

    result = evaluate(y_true, y_pred, LABELS)

    assert result["accuracy"] == pytest.approx(4 / 6)
    assert result["confusion"].loc["a"].tolist() == [2, 1, 0]
    assert result["confusion"].loc["c"].tolist() == [1, 0, 0]
    a = result["per_class"].loc["a"]
    assert a["sensitivity"] == pytest.approx(2 / 3)
    assert a["precision"] == pytest.approx(2 / 3)
    assert a["specificity"] == pytest.approx(2 / 3)  # TN=2 (b,b), FP=1 (c->a)
    assert a["support"] == 3


def test_class_never_predicted_has_zero_scores():
    result = evaluate(["a", "c"], ["a", "a"], LABELS)
    c = result["per_class"].loc["c"]
    assert c["sensitivity"] == 0 and c["precision"] == 0 and c["f1"] == 0
    assert result["per_class"].loc["b", "specificity"] == 1.0
