"""Train the activity HMM and evaluate it with recording-level cross-validation.

Run from the repository root (after ``python -m src.dataset``):  python -m src.training
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import (
    ACTIVITIES,
    DEFAULT_HMM_CONFIG,
    FEATURES_FILE,
    TEST_FEATURES_FILE,
    HMMConfig,
)
from src.dataset import ID_COLUMNS
from src.evaluation import evaluate
from src.hmm import GaussianHMM

STATE_OF = {activity: i for i, activity in enumerate(ACTIVITIES)}


@dataclass
class TrainedModel:
    """A fitted HMM with the scaler and feature columns it expects."""

    hmm: GaussianHMM
    scaler: StandardScaler
    feature_columns: list[str]
    baum_welch_history: list[float] = field(default_factory=list)


def feature_columns(table: pd.DataFrame) -> list[str]:
    return [c for c in table.columns if c not in ID_COLUMNS]


def recording_sequences(table: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    """Split a feature table into per-recording sequences in window order."""
    return [
        (name, group.sort_values("window_index"))
        for name, group in table.groupby("recording", sort=True)
    ]


def recording_folds(table: pd.DataFrame, n_folds: int) -> list[list[str]]:
    """Assign recordings to folds so every fold holds out recordings of every activity.

    Within each activity, recordings are sorted by name and dealt round-robin, so
    fold sizes per activity differ by at most one.
    """
    folds: list[list[str]] = [[] for _ in range(n_folds)]
    for _, group in table.groupby("activity", sort=True):
        for i, rec in enumerate(sorted(group["recording"].unique())):
            folds[i % n_folds].append(rec)
    return [f for f in folds if f]


def train_model(
    train: pd.DataFrame,
    config: HMMConfig = DEFAULT_HMM_CONFIG,
    baum_welch: bool = False,
) -> TrainedModel:
    """Fit the scaler and HMM on training windows.

    Parameters are first estimated from the labels (supervised). With
    ``baum_welch=True`` they are then refined by Baum-Welch on the same
    sequences, ignoring the labels.
    """
    columns = feature_columns(train)
    scaler = StandardScaler().fit(train[columns].to_numpy(dtype=float))
    sequences, states = [], []
    for _, seq in recording_sequences(train):
        sequences.append(scaler.transform(seq[columns].to_numpy(dtype=float)))
        states.append(seq["activity"].map(STATE_OF).to_numpy())

    hmm = GaussianHMM(len(ACTIVITIES), variance_floor=config.variance_floor)
    hmm.fit_supervised(sequences, states, pseudocount=config.transition_pseudocount)
    history = hmm.fit(sequences, config.max_iter, config.tol) if baum_welch else []
    return TrainedModel(hmm, scaler, columns, history)


def predict(model: TrainedModel, table: pd.DataFrame) -> pd.Series:
    """Viterbi-decode each recording separately; returns predicted activity per window."""
    predictions = []
    for _, seq in recording_sequences(table):
        X = model.scaler.transform(seq[model.feature_columns].to_numpy(dtype=float))
        path = model.hmm.predict(X)
        predictions.append(pd.Series([ACTIVITIES[s] for s in path], index=seq.index))
    return pd.concat(predictions).reindex(table.index)


def cross_validate(
    table: pd.DataFrame,
    config: HMMConfig = DEFAULT_HMM_CONFIG,
    variants: Sequence[str] = ("supervised", "baum_welch"),
) -> pd.DataFrame:
    """Out-of-fold predictions for every window, one column per training variant."""
    out = table[ID_COLUMNS].copy()
    out["fold"] = -1
    for name in variants:
        out[name] = None
    for fold, test_recordings in enumerate(recording_folds(table, config.n_folds)):
        is_test = table["recording"].isin(test_recordings)
        out.loc[is_test, "fold"] = fold
        for name in variants:
            model = train_model(table[~is_test], config, baum_welch=(name == "baum_welch"))
            out.loc[is_test, name] = predict(model, table[is_test])
    return out


def predict_unseen(
    train: pd.DataFrame,
    test: pd.DataFrame,
    config: HMMConfig = DEFAULT_HMM_CONFIG,
    variants: Sequence[str] = ("supervised", "baum_welch"),
) -> tuple[pd.DataFrame, dict[str, TrainedModel]]:
    """Train on all of ``train`` and Viterbi-decode every test recording.

    Returns the test identifier columns plus one prediction column per variant,
    and the fitted models. Transition windows are predicted too; drop them
    (see ``scored_windows``) before computing metrics.
    """
    out = test[ID_COLUMNS].copy()
    models = {}
    for name in variants:
        models[name] = train_model(train, config, baum_welch=(name == "baum_welch"))
        out[name] = predict(models[name], test)
    return out, models


def scored_windows(results: pd.DataFrame) -> pd.DataFrame:
    """Only windows that lie inside one labeled activity segment."""
    return results[results["activity"].isin(ACTIVITIES)]


def main() -> None:
    table = pd.read_csv(FEATURES_FILE)
    if TEST_FEATURES_FILE.exists():
        test = pd.read_csv(TEST_FEATURES_FILE)
        results, _ = predict_unseen(table, test)
        scored = scored_windows(results)
        print(f"=== unseen test set: {test['recording'].nunique()} recordings, "
              f"{len(scored)} scored windows ({len(results) - len(scored)} transition) ===")
        for variant in ("supervised", "baum_welch"):
            metrics = evaluate(scored["activity"], scored[variant], ACTIVITIES)
            print(f"--- {variant}: accuracy {metrics['accuracy']:.3f}")
            print(metrics["confusion"], "\n")
            print(metrics["per_class"].round(3), "\n")
    else:
        print(f"no {TEST_FEATURES_FILE.name}; run python -m src.test_set for the unseen test\n")

    results = cross_validate(table)
    n_folds = results["fold"].nunique()
    print(f"{len(table)} windows, {table['recording'].nunique()} recordings, "
          f"{n_folds}-fold cross-validation by recording\n")
    for variant in ("supervised", "baum_welch"):
        metrics = evaluate(results["activity"], results[variant], ACTIVITIES)
        per_recording = (
            results.assign(correct=results["activity"] == results[variant])
            .groupby("recording")["correct"].mean()
        )
        print(f"=== {variant} ===")
        print(f"window accuracy: {metrics['accuracy']:.3f}")
        print(metrics["confusion"], "\n")
        print(metrics["per_class"].round(3), "\n")
        imperfect = per_recording[per_recording < 1]
        print("recordings not fully correct:",
              "none" if imperfect.empty else imperfect.round(2).to_dict(), "\n")

    final = train_model(table, baum_welch=False)
    print("Transition matrix (all data, supervised):")
    print(pd.DataFrame(final.hmm.transmat, index=ACTIVITIES, columns=ACTIVITIES).round(3))


if __name__ == "__main__":
    main()
