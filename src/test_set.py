"""Unseen test set: long recordings from a new session, each with several activities.

Every recording folder in ``data/test/`` holds one continuous Sensor Logger
export. ``data/test/labels.csv`` lists its activity segments
(``recording, start_s, end_s, activity``), in seconds from the start of the
recording. Time between segments is handling (picking the phone up, putting
it down) and is left unlabeled.

A window gets a segment's activity only if it lies entirely inside that
segment. All other windows are labeled ``transition``: they stay in the
sequence that Viterbi decodes, but they are not scored.

Run from the repository root:  python -m src.test_set
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    ACTIVITIES,
    DEFAULT_FEATURE_CONFIG,
    DEFAULT_RECORDING_CONFIG,
    TEST_DATA_DIR,
    TEST_FEATURES_FILE,
    TEST_LABELS_FILE,
    TRANSITION_LABEL,
    FeatureConfig,
    RecordingConfig,
)
from src.dataset import process_recording
from src.features import extract_features

# Test recordings are long and are labeled from their first sample, so they are
# not trimmed and have no 10 s length limit.
TEST_RECORDING_CONFIG = replace(
    DEFAULT_RECORDING_CONFIG, trim_start_s=0.0, trim_end_s=0.0, max_duration_s=600.0
)


def load_labels(path: str | Path = TEST_LABELS_FILE) -> pd.DataFrame:
    """Read the segment table and check it is well formed."""
    labels = pd.read_csv(path)
    unknown = set(labels["activity"]) - set(ACTIVITIES)
    if unknown:
        raise ValueError(f"unknown activities in {path}: {sorted(unknown)}")
    if (labels["end_s"] <= labels["start_s"]).any():
        raise ValueError(f"segment with end_s <= start_s in {path}")
    for name, segs in labels.groupby("recording"):
        segs = segs.sort_values("start_s")
        if (segs["start_s"].to_numpy()[1:] < segs["end_s"].to_numpy()[:-1]).any():
            raise ValueError(f"overlapping segments in {name}")
    return labels


def label_windows(
    starts: np.ndarray | pd.Series,
    window_s: float,
    segments: pd.DataFrame,
) -> np.ndarray:
    """Activity of each window, or ``TRANSITION_LABEL`` if it is not inside one segment."""
    starts = np.asarray(starts, dtype=float)
    ends = starts + window_s
    out = np.full(len(starts), TRANSITION_LABEL, dtype=object)
    eps = 1e-9
    for seg in segments.itertuples():
        inside = (starts >= seg.start_s - eps) & (ends <= seg.end_s + eps)
        out[inside] = seg.activity
    return out


def build_test_table(
    root: str | Path = TEST_DATA_DIR,
    labels_path: str | Path | None = None,
    recording_config: RecordingConfig = TEST_RECORDING_CONFIG,
    feature_config: FeatureConfig = DEFAULT_FEATURE_CONFIG,
) -> pd.DataFrame:
    """Windowed features for every labeled test recording, with per-window labels."""
    root = Path(root)
    labels = load_labels(labels_path or root / TEST_LABELS_FILE.name)
    tables = []
    for name, segments in labels.groupby("recording", sort=True):
        signals = process_recording(root / name, recording_config)
        features = extract_features(
            signals, recording_config.target_sampling_rate_hz, feature_config
        )
        features.insert(0, "recording", name)
        features.insert(1, "participant", name.split("_")[0])
        features.insert(
            2, "activity",
            label_windows(features["window_start_s"], feature_config.window_s, segments),
        )
        tables.append(features)
    return pd.concat(tables, ignore_index=True)


def main() -> None:
    table = build_test_table()
    TEST_FEATURES_FILE.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(TEST_FEATURES_FILE, index=False)
    print(f"Wrote {len(table)} windows from {table['recording'].nunique()} test "
          f"recordings to {TEST_FEATURES_FILE}")
    print(pd.crosstab(table["recording"], table["activity"], margins=True))


if __name__ == "__main__":
    main()
