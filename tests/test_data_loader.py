"""Tests for src.data_loader."""
import numpy as np
import pandas as pd
import pytest

from src.config import (
    COMBINED_CANONICAL,
    SENSOR_COLUMNS,
    SENSOR_LOGGER_ACCELEROMETER,
    TIMESTAMP_COLUMN,
    ColumnMapping,
)
from src.data_loader import (
    RecordingInfo,
    find_recordings,
    load_recording,
    load_sensor_csv,
    parse_recording_name,
)


def write_sensor_logger_csv(path, n=5, start_ns=1_700_000_000_000_000_000, step_ns=10_000_000):
    """Write a tiny CSV in Sensor Logger's layout (time in ns, z/y/x column order)."""
    time = start_ns + np.arange(n, dtype=np.int64) * step_ns
    pd.DataFrame({
        "time": time,
        "seconds_elapsed": np.arange(n) * step_ns / 1e9,
        "z": np.full(n, 3.0),
        "y": np.full(n, 2.0),
        "x": np.full(n, 1.0),
    }).to_csv(path, index=False)


class TestLoadSensorCsv:
    def test_maps_columns_and_converts_ns_to_seconds(self, tmp_path):
        path = tmp_path / "Accelerometer.csv"
        write_sensor_logger_csv(path)

        df = load_sensor_csv(path, SENSOR_LOGGER_ACCELEROMETER)

        assert list(df.columns) == [TIMESTAMP_COLUMN, "acc_x", "acc_y", "acc_z"]
        assert df["acc_x"].tolist() == [1.0] * 5
        assert df["acc_z"].tolist() == [3.0] * 5
        # Epoch-ns as float64 seconds is precise to ~0.2 us, far below one sample interval.
        np.testing.assert_allclose(np.diff(df[TIMESTAMP_COLUMN]), 0.01, atol=1e-6)

    @pytest.mark.parametrize("unit, scale", [("s", 1.0), ("ms", 1e-3), ("us", 1e-6)])
    def test_timestamp_units(self, tmp_path, unit, scale):
        path = tmp_path / "data.csv"
        pd.DataFrame({"t": [0, 1000, 2000], "a": [1.0, 2.0, 3.0]}).to_csv(path, index=False)
        mapping = ColumnMapping(timestamp="t", timestamp_unit=unit, columns={"a": "acc_x"})

        df = load_sensor_csv(path, mapping)

        np.testing.assert_allclose(df[TIMESTAMP_COLUMN], np.array([0, 1000, 2000]) * scale)

    def test_combined_six_axis_csv(self, tmp_path):
        path = tmp_path / "combined.csv"
        data = {TIMESTAMP_COLUMN: [0.0, 0.02]} | {c: [0.5, 0.6] for c in SENSOR_COLUMNS}
        pd.DataFrame(data).to_csv(path, index=False)

        df = load_sensor_csv(path, COMBINED_CANONICAL)

        assert list(df.columns) == [TIMESTAMP_COLUMN, *SENSOR_COLUMNS]

    def test_missing_column_raises(self, tmp_path):
        path = tmp_path / "bad.csv"
        pd.DataFrame({"time": [0, 1], "x": [1, 2], "y": [1, 2]}).to_csv(path, index=False)

        with pytest.raises(ValueError, match="missing columns.*'z'"):
            load_sensor_csv(path, SENSOR_LOGGER_ACCELEROMETER)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_sensor_csv(tmp_path / "nope.csv", SENSOR_LOGGER_ACCELEROMETER)

    def test_non_numeric_values_become_nan(self, tmp_path):
        path = tmp_path / "data.csv"
        pd.DataFrame({"t": [0, 1, 2], "a": ["1.0", "oops", ""]}).to_csv(path, index=False)
        mapping = ColumnMapping(timestamp="t", timestamp_unit="s", columns={"a": "acc_x"})

        df = load_sensor_csv(path, mapping)

        assert df["acc_x"].isna().tolist() == [False, True, True]


class TestColumnMapping:
    def test_rejects_unknown_unit(self):
        with pytest.raises(ValueError, match="Unsupported timestamp unit"):
            ColumnMapping(timestamp="t", timestamp_unit="minutes")

    def test_rejects_duplicate_targets(self):
        with pytest.raises(ValueError, match="Duplicate"):
            ColumnMapping(timestamp="t", timestamp_unit="s", columns={"a": "acc_x", "b": "acc_x"})


class TestRecordingNames:
    def test_parse_valid_name(self, tmp_path):
        info = parse_recording_name(tmp_path / "p01_walking_03")
        assert info == RecordingInfo(participant="p01", activity="walking", index=3)

    @pytest.mark.parametrize("name", ["p01_running_01", "p01-walking-01", "walking_01"])
    def test_parse_invalid_name(self, name):
        with pytest.raises(ValueError):
            parse_recording_name(name)

    def test_find_recordings_filters_by_pattern(self, tmp_path):
        (tmp_path / "p01_still_01").mkdir()
        (tmp_path / "p02_jumping_02").mkdir()
        (tmp_path / "scratch").mkdir()
        (tmp_path / "p01_still_02.csv").write_text("")

        found = [p.name for p in find_recordings(tmp_path)]

        assert found == ["p01_still_01", "p02_jumping_02"]

    def test_find_recordings_missing_root(self, tmp_path):
        assert find_recordings(tmp_path / "absent") == []


def test_load_recording_returns_both_streams(tmp_path):
    rec = tmp_path / "p01_standing_01"
    rec.mkdir()
    write_sensor_logger_csv(rec / "Accelerometer.csv")
    write_sensor_logger_csv(rec / "Gyroscope.csv")

    acc, gyro = load_recording(rec)

    assert list(acc.columns) == [TIMESTAMP_COLUMN, "acc_x", "acc_y", "acc_z"]
    assert list(gyro.columns) == [TIMESTAMP_COLUMN, "gyro_x", "gyro_y", "gyro_z"]
