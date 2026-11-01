"""Tests for src.test_set."""
import pandas as pd
import pytest

from src.config import TRANSITION_LABEL
from src.test_set import build_test_table, label_windows, load_labels
from tests.test_data_loader import write_sensor_logger_csv

SEGMENTS = pd.DataFrame({
    "recording": ["r", "r"],
    "start_s": [0.0, 5.0],
    "end_s": [4.0, 9.0],
    "activity": ["walking", "still"],
})


def test_label_windows_inside_and_across_segments():
    labels = label_windows([0.0, 1.5, 2.0, 2.5, 4.0, 5.0, 7.0, 7.5], 2.0, SEGMENTS)
    assert list(labels) == [
        "walking", "walking", "walking",  # ends at 2, 3.5, 4 -> inside [0, 4]
        TRANSITION_LABEL,                 # 2.5-4.5 crosses the end of walking
        TRANSITION_LABEL,                 # 4-6 starts in the gap
        "still", "still",                 # 5-7, 7-9
        TRANSITION_LABEL,                 # 7.5-9.5 runs past the end
    ]


def test_load_labels_rejects_bad_tables(tmp_path):
    path = tmp_path / "labels.csv"
    SEGMENTS.assign(activity=["walking", "running"]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="unknown"):
        load_labels(path)
    SEGMENTS.assign(end_s=[6.0, 9.0]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="overlapping"):
        load_labels(path)


def test_build_test_table_keeps_long_recording_untrimmed(tmp_path):
    rec = tmp_path / "p01_mixed_01"
    rec.mkdir()
    write_sensor_logger_csv(rec / "Accelerometer.csv", n=2000)  # ~20 s, over the 10 s limit
    write_sensor_logger_csv(rec / "Gyroscope.csv", n=2000)
    pd.DataFrame({
        "recording": ["p01_mixed_01"] * 2,
        "start_s": [0.0, 10.0],
        "end_s": [8.0, 19.99],
        "activity": ["walking", "jumping"],
    }).to_csv(tmp_path / "labels.csv", index=False)

    table = build_test_table(tmp_path)

    assert table["window_start_s"].iloc[0] == 0.0  # not trimmed
    counts = table["activity"].value_counts()
    assert counts["walking"] == 13  # starts 0.0 .. 6.0
    assert counts["jumping"] == 16  # starts 10.0 .. 17.5
    assert counts[TRANSITION_LABEL] == len(table) - 29
