"""Tests for src.dataset."""
import pytest

from src.dataset import ID_COLUMNS, build_feature_table, process_recording
from tests.test_data_loader import write_sensor_logger_csv


def make_recording(root, name, n_samples):
    rec = root / name
    rec.mkdir()
    write_sensor_logger_csv(rec / "Accelerometer.csv", n=n_samples)
    write_sensor_logger_csv(rec / "Gyroscope.csv", n=n_samples)
    return rec


def test_process_recording_trims_and_harmonizes(tmp_path):
    rec = make_recording(tmp_path, "p01_walking_01", n_samples=1000)  # 9.99 s at 100 Hz
    signals = process_recording(rec)
    assert signals["timestamp"].iloc[0] == 0.0
    assert signals["timestamp"].iloc[-1] == pytest.approx(7.98)
    assert len(signals) == 400


def test_process_recording_reports_errors(tmp_path):
    rec = make_recording(tmp_path, "p01_walking_01", n_samples=300)  # 1 s left after trim
    with pytest.raises(ValueError, match="duration"):
        process_recording(rec)


def test_build_feature_table_labels_and_skips(tmp_path):
    make_recording(tmp_path, "p01_walking_01", n_samples=1000)
    make_recording(tmp_path, "p02_still_01", n_samples=1000)
    make_recording(tmp_path, "p01_jumping_01", n_samples=300)  # too short after trim
    make_recording(tmp_path, "p01_running_01", n_samples=1000)  # unknown activity

    table, skipped = build_feature_table(tmp_path)

    assert list(table.columns[: len(ID_COLUMNS)]) == ID_COLUMNS
    assert set(table["recording"]) == {"p01_walking_01", "p02_still_01"}
    assert set(table["activity"]) == {"walking", "still"}
    assert set(table.loc[table["recording"] == "p02_still_01", "participant"]) == {"p02"}
    assert set(skipped) == {"p01_jumping_01", "p01_running_01"}


def test_build_feature_table_empty_root(tmp_path):
    table, skipped = build_feature_table(tmp_path)
    assert table.empty and list(table.columns) == ID_COLUMNS and skipped == {}
