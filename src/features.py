"""Time- and frequency-domain feature extraction from signal windows."""
from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations

import numpy as np
import pandas as pd

from src.config import (
    ACCELEROMETER_COLUMNS,
    DEFAULT_FEATURE_CONFIG,
    DEFAULT_RECORDING_CONFIG,
    GYROSCOPE_COLUMNS,
    MAGNITUDE_COLUMNS,
    SENSOR_COLUMNS,
    TIMESTAMP_COLUMN,
    FeatureConfig,
)


def add_magnitudes(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with Euclidean-norm channels (``acc_mag``, ``gyro_mag``) appended."""
    out = df.copy()
    for name, axes in MAGNITUDE_COLUMNS.items():
        out[name] = np.linalg.norm(df[list(axes)].to_numpy(dtype=float), axis=1)
    return out


def window_starts(n_samples: int, window_size: int, step: int) -> np.ndarray:
    """Start indices of every full window; empty if the signal is shorter than one window."""
    if window_size <= 0 or step <= 0:
        raise ValueError("window_size and step must be positive")
    if n_samples < window_size:
        return np.empty(0, dtype=int)
    return np.arange(0, n_samples - window_size + 1, step)


def sliding_windows(signal: np.ndarray, window_size: int, step: int) -> np.ndarray:
    """Cut a ``(n_samples, n_channels)`` signal into ``(n_windows, window_size, n_channels)``."""
    signal = np.asarray(signal, dtype=float)
    starts = window_starts(len(signal), window_size, step)
    if starts.size == 0:
        return np.empty((0, window_size, signal.shape[1]))
    return np.stack([signal[s:s + window_size] for s in starts])


def dominant_frequency(x: np.ndarray, fs: float) -> float:
    """Frequency (Hz) of the largest non-DC FFT component; 0 for a constant signal."""
    spectrum = np.abs(np.fft.rfft(x - x.mean()))
    if spectrum.size < 2 or not spectrum[1:].any():
        return 0.0
    freqs = np.fft.rfftfreq(len(x), d=1.0 / fs)
    return float(freqs[1 + np.argmax(spectrum[1:])])


def spectral_energy(x: np.ndarray) -> float:
    """Mean squared FFT magnitude of the mean-removed signal (per-sample signal power)."""
    spectrum = np.fft.fft(x - x.mean())  # full spectrum so Parseval holds without bin weighting
    return float(np.sum(np.abs(spectrum) ** 2) / len(x) ** 2)


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def window_features(window: np.ndarray, channels: Sequence[str], fs: float) -> dict[str, float]:
    """Compute all features for one ``(window_size, n_channels)`` window.

    Per channel: mean, std, min, max, dominant frequency, spectral energy.
    Per sensor: signal magnitude area (SMA) and pairwise axis correlations.
    """
    col = {name: window[:, i] for i, name in enumerate(channels)}
    feats: dict[str, float] = {}
    for name, x in col.items():
        feats[f"{name}_mean"] = float(x.mean())
        feats[f"{name}_std"] = float(x.std())
        feats[f"{name}_min"] = float(x.min())
        feats[f"{name}_max"] = float(x.max())
        feats[f"{name}_dom_freq"] = dominant_frequency(x, fs)
        feats[f"{name}_energy"] = spectral_energy(x)

    for prefix, axes in (("acc", ACCELEROMETER_COLUMNS), ("gyro", GYROSCOPE_COLUMNS)):
        if not all(a in col for a in axes):
            continue
        feats[f"{prefix}_sma"] = float(np.mean(np.sum([np.abs(col[a]) for a in axes], axis=0)))
        for a, b in combinations(axes, 2):
            feats[f"corr_{a}_{b}"] = _correlation(col[a], col[b])
    return feats


def extract_features(
    df: pd.DataFrame,
    sampling_rate_hz: float = DEFAULT_RECORDING_CONFIG.target_sampling_rate_hz,
    config: FeatureConfig = DEFAULT_FEATURE_CONFIG,
) -> pd.DataFrame:
    """Turn one harmonized recording into a feature table with one row per window.

    ``df`` must be uniformly sampled at ``sampling_rate_hz`` (the output of
    ``src.preprocessing.harmonize``). Rows are in time order; ``window_start_s``
    gives each window's start relative to the recording.
    """
    signals = add_magnitudes(df)
    channels = [*SENSOR_COLUMNS, *MAGNITUDE_COLUMNS]
    window_size = int(round(config.window_s * sampling_rate_hz))
    step = int(round(config.step_s * sampling_rate_hz))

    starts = window_starts(len(signals), window_size, step)
    windows = sliding_windows(signals[channels].to_numpy(), window_size, step)
    ts = signals[TIMESTAMP_COLUMN].to_numpy(dtype=float)
    rows = [
        {"window_index": i, "window_start_s": float(ts[s])}
        | window_features(w, channels, sampling_rate_hz)
        for i, (s, w) in enumerate(zip(starts, windows))
    ]
    return pd.DataFrame(rows)
