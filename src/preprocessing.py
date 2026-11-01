"""Trimming, validation, and sampling-rate harmonization of raw sensor recordings."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import DEFAULT_RECORDING_CONFIG, TIMESTAMP_COLUMN, RecordingConfig


@dataclass
class ValidationReport:
    """Outcome of validating one sensor stream or recording."""

    n_samples: int
    duration_s: float
    sampling_rate_hz: float
    errors: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.errors


def _timestamps(df: pd.DataFrame) -> np.ndarray:
    return df[TIMESTAMP_COLUMN].to_numpy(dtype=float)


def estimate_sampling_rate(timestamps: Sequence[float] | np.ndarray) -> float:
    """Estimate the sampling rate (Hz) from the median positive sample interval.

    Returns NaN when fewer than two valid, increasing timestamps exist.
    """
    ts = np.asarray(timestamps, dtype=float)
    ts = ts[np.isfinite(ts)]
    steps = np.diff(ts)
    steps = steps[steps > 0]
    if steps.size == 0:
        return float("nan")
    return float(1.0 / np.median(steps))


def recording_duration(timestamps: Sequence[float] | np.ndarray) -> float:
    """Time span (s) between the first and last valid timestamp; NaN if empty."""
    ts = np.asarray(timestamps, dtype=float)
    ts = ts[np.isfinite(ts)]
    if ts.size == 0:
        return float("nan")
    return float(ts.max() - ts.min())


def trim(df: pd.DataFrame, config: RecordingConfig = DEFAULT_RECORDING_CONFIG) -> pd.DataFrame:
    """Drop the first ``trim_start_s`` and last ``trim_end_s`` seconds of a stream.

    Rows with missing timestamps are kept so that validation still reports them.
    """
    ts = _timestamps(df)
    finite = ts[np.isfinite(ts)]
    if finite.size == 0:
        return df.copy()
    lo = finite.min() + config.trim_start_s
    hi = finite.max() - config.trim_end_s
    keep = ~(ts < lo) & ~(ts > hi)
    return df[keep].reset_index(drop=True)


def check_timestamps(
    df: pd.DataFrame, config: RecordingConfig = DEFAULT_RECORDING_CONFIG
) -> list[str]:
    """Check timestamps are present, finite, strictly increasing, and without large gaps."""
    if TIMESTAMP_COLUMN not in df.columns:
        return [f"missing {TIMESTAMP_COLUMN!r} column"]

    errors = []
    ts = _timestamps(df)
    n_invalid = int((~np.isfinite(ts)).sum())
    if n_invalid:
        errors.append(f"{n_invalid} missing or non-numeric timestamps")

    finite = ts[np.isfinite(ts)]
    if finite.size < 2:
        errors.append("fewer than 2 valid timestamps")
        return errors

    steps = np.diff(finite)
    n_non_increasing = int((steps <= 0).sum())
    if n_non_increasing:
        errors.append(f"timestamps not strictly increasing at {n_non_increasing} positions")

    largest_gap = float(steps.max())
    if largest_gap > config.max_gap_s:
        n_gaps = int((steps > config.max_gap_s).sum())
        errors.append(
            f"{n_gaps} gaps longer than {config.max_gap_s} s (largest {largest_gap:.3f} s)"
        )
    return errors


def check_missing_values(df: pd.DataFrame, columns: Sequence[str]) -> list[str]:
    """Check the given sensor columns exist and contain no missing values."""
    absent = [c for c in columns if c not in df.columns]
    if absent:
        return [f"missing columns {absent}"]
    counts = df[list(columns)].isna().sum()
    return [f"{int(n)} missing values in {col!r}" for col, n in counts.items() if n]


def check_sampling_rate(
    df: pd.DataFrame, config: RecordingConfig = DEFAULT_RECORDING_CONFIG
) -> list[str]:
    """Check the stream is sampled at least as fast as the target rate (within tolerance)."""
    rate = estimate_sampling_rate(_timestamps(df))
    minimum = config.target_sampling_rate_hz * (1 - config.sampling_rate_tolerance)
    if np.isnan(rate):
        return ["sampling rate cannot be estimated"]
    if rate < minimum:
        return [
            f"sampling rate {rate:.1f} Hz is below the target "
            f"{config.target_sampling_rate_hz:.1f} Hz"
        ]
    return []


def check_duration(
    df: pd.DataFrame, config: RecordingConfig = DEFAULT_RECORDING_CONFIG
) -> list[str]:
    """Check the recording lasts between the configured minimum and maximum duration."""
    duration = recording_duration(_timestamps(df))
    if np.isnan(duration):
        return ["duration cannot be determined"]
    if not config.min_duration_s <= duration <= config.max_duration_s:
        return [
            f"duration {duration:.2f} s is outside "
            f"[{config.min_duration_s}, {config.max_duration_s}] s"
        ]
    return []


def validate_recording(
    df: pd.DataFrame,
    columns: Sequence[str] | None = None,
    config: RecordingConfig = DEFAULT_RECORDING_CONFIG,
) -> ValidationReport:
    """Run every check on a stream; ``columns`` defaults to all non-timestamp columns."""
    if columns is None:
        columns = [c for c in df.columns if c != TIMESTAMP_COLUMN]

    if TIMESTAMP_COLUMN not in df.columns:
        return ValidationReport(len(df), float("nan"), float("nan"),
                                check_timestamps(df, config))

    ts = _timestamps(df)
    errors = (
        check_timestamps(df, config)
        + check_missing_values(df, columns)
        + check_sampling_rate(df, config)
        + check_duration(df, config)
    )
    return ValidationReport(
        n_samples=len(df),
        duration_s=recording_duration(ts),
        sampling_rate_hz=estimate_sampling_rate(ts),
        errors=errors,
    )


def resample(df: pd.DataFrame, rate_hz: float, start: float, end: float) -> pd.DataFrame:
    """Linearly interpolate every sensor column onto a uniform grid over [start, end].

    The returned ``timestamp`` column holds the absolute grid times.
    """
    n = int(np.floor((end - start) * rate_hz + 1e-9)) + 1
    grid = start + np.arange(n) / rate_hz
    ts = _timestamps(df)
    out = {TIMESTAMP_COLUMN: grid}
    for col in df.columns:
        if col != TIMESTAMP_COLUMN:
            out[col] = np.interp(grid, ts, df[col].to_numpy(dtype=float))
    return pd.DataFrame(out)


def harmonize(
    streams: Sequence[pd.DataFrame],
    config: RecordingConfig = DEFAULT_RECORDING_CONFIG,
) -> pd.DataFrame:
    """Resample sensor streams onto one shared grid at the target sampling rate.

    Only the time span covered by every stream is kept. The returned
    ``timestamp`` column starts at 0. Streams must already pass validation
    (strictly increasing timestamps, no missing values).
    """
    if not streams:
        raise ValueError("no streams to harmonize")

    seen: set[str] = set()
    for i, df in enumerate(streams):
        if not _is_clean(df):
            raise ValueError(f"stream {i} has invalid timestamps or missing values; "
                             "run validate_recording first")
        sensor_cols = set(df.columns) - {TIMESTAMP_COLUMN}
        overlap = seen & sensor_cols
        if overlap:
            raise ValueError(f"column(s) {sorted(overlap)} appear in more than one stream")
        seen |= sensor_cols

    start = max(_timestamps(df)[0] for df in streams)
    end = min(_timestamps(df)[-1] for df in streams)
    if end <= start:
        raise ValueError("streams do not overlap in time")

    rate = config.target_sampling_rate_hz
    parts = [resample(df, rate, start, end) for df in streams]
    combined = parts[0]
    for part in parts[1:]:
        combined = pd.concat([combined, part.drop(columns=TIMESTAMP_COLUMN)], axis=1)
    combined[TIMESTAMP_COLUMN] -= start
    return combined


def _is_clean(df: pd.DataFrame) -> bool:
    """True if timestamps are finite and strictly increasing and no values are missing."""
    if TIMESTAMP_COLUMN not in df.columns:
        return False
    ts = _timestamps(df)
    return (
        ts.size >= 2
        and bool(np.isfinite(ts).all())
        and bool((np.diff(ts) > 0).all())
        and not df.isna().to_numpy().any()
    )
