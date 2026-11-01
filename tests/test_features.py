"""Tests for src.features."""
import numpy as np
import pandas as pd
import pytest

from src.config import SENSOR_COLUMNS, TIMESTAMP_COLUMN, FeatureConfig
from src.features import (
    add_magnitudes,
    dominant_frequency,
    extract_features,
    sliding_windows,
    spectral_energy,
    window_features,
    window_starts,
)

FS = 50.0


def make_signals(duration_s=8.0, fs=FS):
    """Harmonized-style table: acc_x is a 2 Hz sine, other channels small constants."""
    t = np.arange(int(duration_s * fs)) / fs
    data = {TIMESTAMP_COLUMN: t} | {c: np.full_like(t, 0.1) for c in SENSOR_COLUMNS}
    data["acc_x"] = np.sin(2 * np.pi * 2.0 * t)
    return pd.DataFrame(data)


class TestWindowing:
    def test_window_starts(self):
        assert window_starts(10, 4, 2).tolist() == [0, 2, 4, 6]

    def test_signal_shorter_than_window(self):
        assert window_starts(3, 4, 2).size == 0
        assert sliding_windows(np.zeros((3, 2)), 4, 2).shape == (0, 4, 2)

    def test_invalid_sizes(self):
        with pytest.raises(ValueError):
            window_starts(10, 0, 1)

    def test_sliding_windows_content(self):
        signal = np.arange(12, dtype=float).reshape(6, 2)
        windows = sliding_windows(signal, 3, 3)
        assert windows.shape == (2, 3, 2)
        np.testing.assert_array_equal(windows[1], signal[3:6])


class TestSpectralFeatures:
    def test_dominant_frequency_of_sine(self):
        t = np.arange(100) / FS
        assert dominant_frequency(np.sin(2 * np.pi * 3.0 * t), FS) == pytest.approx(3.0)

    def test_constant_signal(self):
        x = np.full(100, 5.0)
        assert dominant_frequency(x, FS) == 0.0
        assert spectral_energy(x) == pytest.approx(0.0)

    def test_spectral_energy_matches_variance(self):
        # Parseval: mean-removed power equals the variance.
        x = np.random.default_rng(0).normal(size=100)
        assert spectral_energy(x) == pytest.approx(x.var())


class TestWindowFeatures:
    def test_magnitudes(self):
        df = pd.DataFrame({c: [3.0] if c.endswith(("_x", "_y")) else [0.0] for c in SENSOR_COLUMNS})
        out = add_magnitudes(df)
        assert out["acc_mag"].iloc[0] == pytest.approx(np.hypot(3, 3))
        assert "acc_mag" not in df.columns

    def test_basic_statistics_and_correlation(self):
        a = np.array([1.0, 2.0, 3.0, 4.0])
        window = np.column_stack([a, 2 * a, -a])
        feats = window_features(window, ["acc_x", "acc_y", "acc_z"], FS)
        assert feats["acc_x_mean"] == pytest.approx(2.5)
        assert feats["acc_x_max"] == 4.0
        assert feats["corr_acc_x_acc_y"] == pytest.approx(1.0)
        assert feats["corr_acc_x_acc_z"] == pytest.approx(-1.0)
        assert feats["acc_sma"] == pytest.approx(np.mean(4 * a))
        assert "gyro_sma" not in feats  # gyroscope axes absent

    def test_constant_channel_correlation_is_zero(self):
        window = np.column_stack([np.arange(5.0), np.ones(5), np.ones(5)])
        feats = window_features(window, ["acc_x", "acc_y", "acc_z"], FS)
        assert feats["corr_acc_x_acc_y"] == 0.0


class TestExtractFeatures:
    def test_window_count_and_columns(self):
        out = extract_features(make_signals(8.0), FS, FeatureConfig(window_s=2.0, step_s=0.5))
        assert len(out) == 13  # (400 - 100) / 25 + 1
        assert out["window_index"].tolist() == list(range(13))
        np.testing.assert_allclose(out["window_start_s"], np.arange(13) * 0.5)
        assert out.shape[1] == 2 + 8 * 6 + 2 + 6
        assert not out.isna().any().any()

    def test_features_reflect_signal(self):
        out = extract_features(make_signals(), FS)
        np.testing.assert_allclose(out["acc_x_dom_freq"], 2.0)
        np.testing.assert_allclose(out["acc_y_std"], 0.0, atol=1e-12)

    def test_too_short_recording(self):
        assert extract_features(make_signals(1.0), FS).empty
