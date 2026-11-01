"""Tests for src.training."""
import numpy as np
import pandas as pd
import pytest

from src.config import ACTIVITIES, HMMConfig
from src.training import cross_validate, predict, recording_folds, train_model

# Feature means per activity, far apart so a correct model separates them perfectly.
CENTERS = {"standing": 0.0, "walking": 4.0, "jumping": 8.0, "still": -4.0}


def make_table(n_recordings=4, n_windows=10, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for activity in ACTIVITIES:
        for r in range(n_recordings):
            for w in range(n_windows):
                rows.append({
                    "recording": f"p01_{activity}_{r + 1:02d}",
                    "participant": "p01",
                    "activity": activity,
                    "window_index": w,
                    "window_start_s": w * 0.5,
                    "f1": CENTERS[activity] + rng.normal(scale=0.5),
                    "f2": rng.normal(),
                })
    # Shuffle rows so code cannot rely on table order.
    return pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)


def test_folds_hold_out_one_recording_per_activity():
    folds = recording_folds(make_table(), n_folds=4)
    assert len(folds) == 4
    all_recs = [r for f in folds for r in f]
    assert len(all_recs) == len(set(all_recs)) == 16
    for fold in folds:
        assert sorted(r.split("_")[1] for r in fold) == sorted(ACTIVITIES)


def test_scaler_fitted_on_training_data_only():
    table = make_table()
    model = train_model(table[table["activity"] != "jumping"])
    assert model.scaler.mean_[0] == pytest.approx(0.0, abs=0.3)  # mean of -4, 0, 4


def test_predict_preserves_index_and_window_order():
    table = make_table()
    model = train_model(table)
    pred = predict(model, table)
    assert pred.index.equals(table.index)
    assert (pred == table["activity"]).all()


@pytest.mark.parametrize("variant", ["supervised", "baum_welch"])
def test_cross_validation_on_separable_data(variant):
    table = make_table()
    results = cross_validate(table, HMMConfig(n_folds=4), variants=[variant])
    assert set(results["fold"]) == {0, 1, 2, 3}
    assert (results[variant] == results["activity"]).mean() == 1.0
