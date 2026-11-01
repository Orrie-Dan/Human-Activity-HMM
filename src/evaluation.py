"""Model evaluation metrics (accuracy, confusion matrix, per-class scores)."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix


def confusion_table(y_true: Sequence, y_pred: Sequence, labels: Sequence[str]) -> pd.DataFrame:
    """Confusion matrix as a DataFrame (rows = true, columns = predicted)."""
    cm = confusion_matrix(y_true, y_pred, labels=list(labels))
    return pd.DataFrame(
        cm,
        index=pd.Index(labels, name="true"),
        columns=pd.Index(labels, name="predicted"),
    )


def per_class_metrics(confusion: pd.DataFrame) -> pd.DataFrame:
    """Sensitivity (recall), specificity, precision, F1, and support for each class."""
    cm = confusion.to_numpy(dtype=float)
    tp = np.diag(cm)
    fn = cm.sum(axis=1) - tp
    fp = cm.sum(axis=0) - tp
    tn = cm.sum() - tp - fn - fp

    def ratio(a, b):
        return np.divide(a, b, out=np.zeros_like(a), where=b > 0)

    sensitivity = ratio(tp, tp + fn)
    precision = ratio(tp, tp + fp)
    return pd.DataFrame(
        {
            "sensitivity": sensitivity,
            "specificity": ratio(tn, tn + fp),
            "precision": precision,
            "f1": ratio(2 * precision * sensitivity, precision + sensitivity),
            "support": cm.sum(axis=1).astype(int),
        },
        index=confusion.index,
    )


def evaluate(y_true: Sequence, y_pred: Sequence, labels: Sequence[str]) -> dict:
    """Overall accuracy, confusion matrix, and per-class metrics in one dict."""
    confusion = confusion_table(y_true, y_pred, labels)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion": confusion,
        "per_class": per_class_metrics(confusion),
    }


def report_table(confusion: pd.DataFrame) -> pd.DataFrame:
    """Results table in the assignment's format, one row per activity plus an overall row.

    Per activity: number of windows, sensitivity, specificity, and one-vs-rest
    accuracy ((TP + TN) / N). The overall row gives the total window count and
    the usual multi-class accuracy (correct windows / all windows).
    """
    cm = confusion.to_numpy(dtype=float)
    n = cm.sum()
    tp = np.diag(cm)
    fn = cm.sum(axis=1) - tp
    fp = cm.sum(axis=0) - tp
    tn = n - tp - fn - fp
    per_class = per_class_metrics(confusion)
    table = pd.DataFrame(
        {
            "Number of samples": per_class["support"],
            "Sensitivity": per_class["sensitivity"],
            "Specificity": per_class["specificity"],
            "Accuracy": (tp + tn) / n if n else np.zeros_like(tp),
        },
        index=pd.Index(confusion.index, name="State (activity)"),
    )
    table.loc["Overall"] = [int(n), np.nan, np.nan, tp.sum() / n if n else 0.0]
    table["Number of samples"] = table["Number of samples"].astype(int)
    return table
