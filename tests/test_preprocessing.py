"""Tests for src.preprocessing."""
import numpy as np
import pandas as pd
import pytest

from src.config import (
    ACCELEROMETER_COLUMNS,
    GYROSCOPE_COLUMNS,
    SENSOR_COLUMNS,
    TIMESTAMP_COLUMN,
    RecordingConfig,
)
from src.preprocessing import (
    check_duration,
    check_missing_values,
    check_sampling_rate,
    check_timestamps,
    estimate_sampling_rate,
    harmonize,
    resample,
    trim,
    validate_recording,
)

CONFIG = RecordingConfig()


def make_stream(duration_s=8.0, rate_hz=100.0, columns=ACCELEROMETER_COLUMNS, start=0.0):
    """Uniformly sampled stream where each column equals its timestamp (easy to check)."""
    ts = start + np.arange(int(round(duration_s * rate_hz)) + 1) / rate_hz
    return pd.DataFrame({TIMESTAMP_COLUMN: ts} | {c: ts.copy() for c in columns})


class TestSamplingRateAndDuration:
    def test_estimate_sampling_rate(self):
        assert estimate_sampling_rate(np.arange(0, 1, 0.01)) == pytest.approx(100.0)

    def test_estimate_sampling_rate_too_few_samples(self):
        assert np.isnan(estimate_sampling_rate([1.0]))

    def test_rate_below_target_is_error(self):
        errors = check_sampling_rate(make_stream(rate_hz=25.0), CONFIG)
        assert errors and "below the target" in errors[0]

    def test_rate_within_tolerance_passes(self):
        assert check_sampling_rate(make_stream(rate_hz=48.0), CONFIG) == []

    @pytest.mark.parametrize("duration", [1.5, 11.0])
    def test_duration_out_of_range(self, duration):
        assert check_duration(make_stream(duration_s=duration), CONFIG)

    @pytest.mark.parametrize("duration", [2.0, 3.6, 8.0, 10.0])
    def test_duration_in_range(self, duration):
        assert check_duration(make_stream(duration_s=duration), CONFIG) == []


class TestTimestamps:
    def test_clean_timestamps(self):
        assert check_timestamps(make_stream(), CONFIG) == []

    def test_missing_timestamp_column(self):
        df = make_stream().drop(columns=TIMESTAMP_COLUMN)
        assert check_timestamps(df, CONFIG) == [f"missing {TIMESTAMP_COLUMN!r} column"]

    def test_nan_timestamps(self):
        df = make_stream()
        df.loc[3, TIMESTAMP_COLUMN] = np.nan
        assert any("missing or non-numeric" in e for e in check_timestamps(df, CONFIG))

    def test_non_increasing_timestamps(self):
        df = make_stream()
        df.loc[10, TIMESTAMP_COLUMN] = df.loc[9, TIMESTAMP_COLUMN]
        assert any("not strictly increasing" in e for e in check_timestamps(df, CONFIG))

    def test_large_gap(self):
        df = make_stream()
        df = df.drop(index=range(100, 150)).reset_index(drop=True)  # 0.5 s gap
        assert any("gaps longer" in e for e in check_timestamps(df, CONFIG))


class TestMissingValues:
    def test_reports_nan_counts(self):
        df = make_stream()
        df.loc[[1, 2], "acc_y"] = np.nan
        assert check_missing_values(df, ACCELEROMETER_COLUMNS) == ["2 missing values in 'acc_y'"]

    def test_reports_absent_columns(self):
        errors = check_missing_values(make_stream(), SENSOR_COLUMNS)
        assert errors and "gyro_x" in errors[0]


class TestTrim:
    def test_removes_configured_seconds_from_each_end(self):
        out = trim(make_stream(duration_s=11.6, start=100.0), CONFIG)
        assert out[TIMESTAMP_COLUMN].iloc[0] == pytest.approx(101.0)
        assert out[TIMESTAMP_COLUMN].iloc[-1] == pytest.approx(110.6)
        assert check_duration(out, CONFIG) == []

    def test_custom_amounts(self):
        config = RecordingConfig(trim_start_s=0.5, trim_end_s=0.0)
        out = trim(make_stream(duration_s=8.0), config)
        assert out[TIMESTAMP_COLUMN].iloc[0] == pytest.approx(0.5)
        assert out[TIMESTAMP_COLUMN].iloc[-1] == pytest.approx(8.0)

    def test_keeps_rows_with_missing_timestamps(self):
        df = make_stream()
        df.loc[400, TIMESTAMP_COLUMN] = np.nan
        out = trim(df, CONFIG)
        assert out[TIMESTAMP_COLUMN].isna().sum() == 1
        assert any("missing or non-numeric" in e for e in check_timestamps(out, CONFIG))

    def test_does_not_modify_input(self):
        df = make_stream()
        trim(df, CONFIG)
        assert len(df) == 801


class TestValidateRecording:
    def test_valid_recording(self):
        report = validate_recording(make_stream())
        assert report.is_valid, report.errors
        assert report.duration_s == pytest.approx(8.0)
        assert report.sampling_rate_hz == pytest.approx(100.0)
        assert report.n_samples == 801

    def test_collects_multiple_errors(self):
        df = make_stream(duration_s=1.5, rate_hz=20.0)
        df.loc[0, "acc_x"] = np.nan
        report = validate_recording(df)
        assert not report.is_valid
        assert len(report.errors) == 3  # missing value, low rate, short duration


class TestHarmonize:
    def test_resample_uniform_grid(self):
        out = resample(make_stream(duration_s=1.0, rate_hz=100.0), 50.0, 0.0, 1.0)
        assert len(out) == 51
        np.testing.assert_allclose(np.diff(out[TIMESTAMP_COLUMN]), 0.02)
        np.testing.assert_allclose(out["acc_x"], out[TIMESTAMP_COLUMN])  # linear signal

    def test_combines_streams_at_target_rate(self):
        acc = make_stream(rate_hz=100.0, columns=ACCELEROMETER_COLUMNS, start=10.0)
        gyro = make_stream(rate_hz=200.0, columns=GYROSCOPE_COLUMNS, start=10.25)

        out = harmonize([acc, gyro], CONFIG)

        assert list(out.columns) == [TIMESTAMP_COLUMN, *SENSOR_COLUMNS]
        assert out[TIMESTAMP_COLUMN].iloc[0] == 0.0
        assert estimate_sampling_rate(out[TIMESTAMP_COLUMN]) == pytest.approx(50.0)
        # Overlap is [10.25, 18.0]; both linear signals equal absolute time.
        assert out[TIMESTAMP_COLUMN].iloc[-1] == pytest.approx(7.74)
        np.testing.assert_allclose(out["acc_x"], out[TIMESTAMP_COLUMN] + 10.25)
        np.testing.assert_allclose(out["gyro_z"], out[TIMESTAMP_COLUMN] + 10.25)
        assert validate_recording(out).is_valid

    def test_rejects_non_overlapping_streams(self):
        acc = make_stream(columns=ACCELEROMETER_COLUMNS, start=0.0)
        gyro = make_stream(columns=GYROSCOPE_COLUMNS, start=100.0)
        with pytest.raises(ValueError, match="do not overlap"):
            harmonize([acc, gyro], CONFIG)

    def test_rejects_duplicate_columns(self):
        with pytest.raises(ValueError, match="more than one stream"):
            harmonize([make_stream(), make_stream()], CONFIG)

    def test_rejects_unclean_stream(self):
        df = make_stream()
        df.loc[5, "acc_x"] = np.nan
        with pytest.raises(ValueError, match="validate_recording"):
            harmonize([df], CONFIG)
