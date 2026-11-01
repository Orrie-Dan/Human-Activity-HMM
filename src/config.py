"""Central configuration: activities, recording parameters, sensor schemas, and paths."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --- Paths -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

# --- Activities and canonical columns ----------------------------------------

ACTIVITIES: tuple[str, ...] = ("standing", "walking", "jumping", "still")

TIMESTAMP_COLUMN = "timestamp"  # seconds (float)
ACCELEROMETER_COLUMNS: tuple[str, ...] = ("acc_x", "acc_y", "acc_z")
GYROSCOPE_COLUMNS: tuple[str, ...] = ("gyro_x", "gyro_y", "gyro_z")
SENSOR_COLUMNS: tuple[str, ...] = ACCELEROMETER_COLUMNS + GYROSCOPE_COLUMNS

# Seconds per one unit of each supported raw timestamp unit.
TIMESTAMP_UNITS: dict[str, float] = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}


# --- Recording parameters ----------------------------------------------------

@dataclass(frozen=True)
class RecordingConfig:
    """Constraints every recording must satisfy before it can be used."""

    # Removed from each end before validation, to drop start/stop handling artifacts.
    trim_start_s: float = 1.0
    trim_end_s: float = 1.0
    # Allowed duration after trimming. The minimum is one 2 s feature window, so short
    # (~5 s) recordings still contribute after the 1 s trim at each end.
    min_duration_s: float = 2.0
    max_duration_s: float = 10.0
    # Common rate all streams are resampled to. Raw streams must be at least this fast.
    target_sampling_rate_hz: float = 50.0
    # Allowed relative shortfall of a raw stream's rate below the target (0.05 = 5%).
    sampling_rate_tolerance: float = 0.05
    # Any gap between consecutive samples longer than this is an error.
    max_gap_s: float = 0.1


DEFAULT_RECORDING_CONFIG = RecordingConfig()


# --- Feature extraction ------------------------------------------------------

@dataclass(frozen=True)
class FeatureConfig:
    """Sliding-window settings for feature extraction (applied to harmonized signals)."""

    # 2 s covers several steps/jumps (~2 Hz) and gives 0.5 Hz FFT resolution.
    window_s: float = 2.0
    # 0.5 s step (75% overlap) yields ~10-16 windows per recording.
    step_s: float = 0.5


DEFAULT_FEATURE_CONFIG = FeatureConfig()

# Orientation-independent magnitude channels added alongside the six axes.
MAGNITUDE_COLUMNS: dict[str, tuple[str, ...]] = {
    "acc_mag": ACCELEROMETER_COLUMNS,
    "gyro_mag": GYROSCOPE_COLUMNS,
}

FEATURES_FILE = PROCESSED_DATA_DIR / "features.csv"


# --- HMM ---------------------------------------------------------------------

@dataclass(frozen=True)
class HMMConfig:
    """Training settings for the Gaussian HMM (one hidden state per activity)."""

    # Added to every transition/start count, so transitions never seen in
    # training (each recording holds a single activity) stay possible.
    transition_pseudocount: float = 1.0
    # Lower bound on emission variances (features are standardized first).
    variance_floor: float = 1e-3
    # Baum-Welch stopping rule.
    max_iter: int = 50
    tol: float = 1e-4
    # Cross-validation folds; recordings of each activity are dealt evenly across folds.
    n_folds: int = 4


DEFAULT_HMM_CONFIG = HMMConfig()


# --- Sensor column mappings --------------------------------------------------

@dataclass(frozen=True)
class ColumnMapping:
    """Describes how a raw CSV's columns map onto the canonical schema.

    ``columns`` maps raw column name -> canonical column name.
    """

    timestamp: str
    timestamp_unit: str
    columns: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.timestamp_unit not in TIMESTAMP_UNITS:
            raise ValueError(
                f"Unsupported timestamp unit {self.timestamp_unit!r}; "
                f"expected one of {sorted(TIMESTAMP_UNITS)}"
            )
        targets = list(self.columns.values())
        if len(set(targets)) != len(targets):
            raise ValueError(f"Duplicate target columns in mapping: {targets}")
        if TIMESTAMP_COLUMN in targets:
            raise ValueError(f"{TIMESTAMP_COLUMN!r} is reserved for the timestamp")


# Sensor Logger (iOS/Android) exports one CSV per sensor with columns
# `time` (Unix epoch, nanoseconds), `seconds_elapsed`, `z`, `y`, `x`.
SENSOR_LOGGER_ACCELEROMETER = ColumnMapping(
    timestamp="time",
    timestamp_unit="ns",
    columns={"x": "acc_x", "y": "acc_y", "z": "acc_z"},
)
SENSOR_LOGGER_GYROSCOPE = ColumnMapping(
    timestamp="time",
    timestamp_unit="ns",
    columns={"x": "gyro_x", "y": "gyro_y", "z": "gyro_z"},
)

# A single CSV that already uses the canonical names (timestamp in seconds).
COMBINED_CANONICAL = ColumnMapping(
    timestamp=TIMESTAMP_COLUMN,
    timestamp_unit="s",
    columns={c: c for c in SENSOR_COLUMNS},
)

# File names inside one recording folder (Sensor Logger export layout).
ACCELEROMETER_FILENAME = "Accelerometer.csv"
GYROSCOPE_FILENAME = "Gyroscope.csv"

# Recording folders are named <participant>_<activity>_<index>, e.g. `p01_walking_03`.
RECORDING_NAME_PATTERN = r"^(?P<participant>[A-Za-z0-9]+)_(?P<activity>[a-z]+)_(?P<index>\d+)$"
