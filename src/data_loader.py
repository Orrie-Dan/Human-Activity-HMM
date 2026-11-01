"""Loading raw sensor recordings into pandas DataFrames."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config import (
    ACCELEROMETER_FILENAME,
    ACTIVITIES,
    GYROSCOPE_FILENAME,
    RAW_DATA_DIR,
    RECORDING_NAME_PATTERN,
    SENSOR_LOGGER_ACCELEROMETER,
    SENSOR_LOGGER_GYROSCOPE,
    TIMESTAMP_COLUMN,
    TIMESTAMP_UNITS,
    ColumnMapping,
)


@dataclass(frozen=True)
class RecordingInfo:
    """Metadata parsed from a recording's folder name."""

    participant: str
    activity: str
    index: int


def load_sensor_csv(path: str | Path, mapping: ColumnMapping) -> pd.DataFrame:
    """Load one sensor CSV and convert it to the canonical schema.

    Returns a DataFrame with ``timestamp`` (seconds, float) followed by the
    mapped sensor columns, in the order given by ``mapping.columns``. Row order
    is preserved and values are not cleaned; use ``src.preprocessing`` to
    validate the result.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Sensor file not found: {path}")

    raw = pd.read_csv(path)
    required = [mapping.timestamp, *mapping.columns]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(
            f"{path.name}: missing columns {missing}; found {list(raw.columns)}"
        )

    out = pd.DataFrame()
    timestamps = pd.to_numeric(raw[mapping.timestamp], errors="coerce")
    out[TIMESTAMP_COLUMN] = timestamps * TIMESTAMP_UNITS[mapping.timestamp_unit]
    for source, target in mapping.columns.items():
        out[target] = pd.to_numeric(raw[source], errors="coerce").astype(float)
    return out


def parse_recording_name(path: str | Path) -> RecordingInfo:
    """Parse ``<participant>_<activity>_<index>`` from a recording folder or file name."""
    name = Path(path).stem
    match = re.match(RECORDING_NAME_PATTERN, name)
    if match is None:
        raise ValueError(
            f"Recording name {name!r} does not match <participant>_<activity>_<index>"
        )
    activity = match["activity"]
    if activity not in ACTIVITIES:
        raise ValueError(f"Unknown activity {activity!r}; expected one of {ACTIVITIES}")
    return RecordingInfo(match["participant"], activity, int(match["index"]))


def find_recordings(root: str | Path = RAW_DATA_DIR) -> list[Path]:
    """Return recording folders under ``root`` whose names follow the naming convention."""
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted(
        p for p in root.iterdir()
        if p.is_dir() and re.match(RECORDING_NAME_PATTERN, p.name)
    )


def load_recording(
    recording_dir: str | Path,
    accelerometer_mapping: ColumnMapping = SENSOR_LOGGER_ACCELEROMETER,
    gyroscope_mapping: ColumnMapping = SENSOR_LOGGER_GYROSCOPE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the accelerometer and gyroscope streams of one recording folder.

    The two streams are returned separately because they are usually sampled
    at different instants; ``src.preprocessing.harmonize`` aligns them.
    """
    recording_dir = Path(recording_dir)
    accelerometer = load_sensor_csv(recording_dir / ACCELEROMETER_FILENAME, accelerometer_mapping)
    gyroscope = load_sensor_csv(recording_dir / GYROSCOPE_FILENAME, gyroscope_mapping)
    return accelerometer, gyroscope
