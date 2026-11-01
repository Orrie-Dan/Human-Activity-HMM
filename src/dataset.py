"""Build the labeled feature table from all raw recordings.

Run from the repository root:  python -m src.dataset
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import (
    DEFAULT_FEATURE_CONFIG,
    DEFAULT_RECORDING_CONFIG,
    FEATURES_FILE,
    RAW_DATA_DIR,
    FeatureConfig,
    RecordingConfig,
)
from src.data_loader import find_recordings, load_recording, parse_recording_name
from src.features import extract_features
from src.preprocessing import harmonize, trim, validate_recording

ID_COLUMNS = ["recording", "participant", "activity", "window_index", "window_start_s"]


def process_recording(
    recording_dir: str | Path,
    config: RecordingConfig = DEFAULT_RECORDING_CONFIG,
) -> pd.DataFrame:
    """Load, trim, validate, and harmonize one recording folder.

    Raises ``ValueError`` listing every validation error if a stream is invalid.
    """
    accelerometer, gyroscope = (trim(s, config) for s in load_recording(recording_dir))
    errors = [
        f"{name}: {error}"
        for name, stream in (("accelerometer", accelerometer), ("gyroscope", gyroscope))
        for error in validate_recording(stream, config=config).errors
    ]
    if errors:
        raise ValueError("; ".join(errors))
    return harmonize([accelerometer, gyroscope], config)


def build_feature_table(
    root: str | Path = RAW_DATA_DIR,
    recording_config: RecordingConfig = DEFAULT_RECORDING_CONFIG,
    feature_config: FeatureConfig = DEFAULT_FEATURE_CONFIG,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Extract windowed features for every valid recording under ``root``.

    Returns the feature table (identifier columns first) and a mapping of
    skipped recording name -> reason.
    """
    tables, skipped = [], {}
    for rec in find_recordings(root):
        try:
            info = parse_recording_name(rec)
            signals = process_recording(rec, recording_config)
        except ValueError as exc:
            skipped[rec.name] = str(exc)
            continue
        features = extract_features(
            signals, recording_config.target_sampling_rate_hz, feature_config
        )
        if features.empty:
            skipped[rec.name] = "shorter than one feature window"
            continue
        features.insert(0, "recording", rec.name)
        features.insert(1, "participant", info.participant)
        features.insert(2, "activity", info.activity)
        tables.append(features)

    table = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame(columns=ID_COLUMNS)
    return table, skipped


def main() -> None:
    table, skipped = build_feature_table()
    for name, reason in skipped.items():
        print(f"skipped {name}: {reason}")
    if table.empty:
        print(f"No usable recordings found in {RAW_DATA_DIR}")
        return
    FEATURES_FILE.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(FEATURES_FILE, index=False)
    n_features = table.shape[1] - len(ID_COLUMNS)
    print(f"Wrote {len(table)} windows x {n_features} features "
          f"from {table['recording'].nunique()} recordings to {FEATURES_FILE}")
    print(table.groupby("activity")["recording"].agg(recordings="nunique", windows="size"))


if __name__ == "__main__":
    main()
