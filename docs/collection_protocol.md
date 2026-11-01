# Data Collection Protocol

This protocol describes how to record smartphone accelerometer and gyroscope data
for four activities: **standing**, **walking**, **jumping**, and **still**. Follow
it exactly so that every recording passes `src.preprocessing.validate_recording`
and can be harmonized with the others.

The parameters below come from `src/config.py`. If you change them there, update
this document too.

| Parameter | Value |
|---|---|
| Activities | `standing`, `walking`, `jumping`, `still` |
| Sensors | Accelerometer (x, y, z) + Gyroscope (x, y, z): six axes |
| Edge trim | 1 s removed from the start and the end |
| Recording length in the app | 5–10 s (assignment requirement); aim for 8–10 s |
| Accepted length after trim | 2–10 s (at least one 2 s feature window) |
| Minimum raw sampling rate | 50 Hz (record at 100 Hz if the app allows it) |
| Harmonized sampling rate | 50 Hz |
| Maximum gap between samples | 0.1 s |

## 1. Equipment and app setup

1. Use a smartphone with both an accelerometer and a gyroscope.
2. Install a sensor-logging app that exports CSV with a timestamp per sample.
   The default column mappings in `src/config.py` target **Sensor Logger**
   (iOS/Android), which exports one CSV per sensor. You can use another app if you
   add a `ColumnMapping` for its columns in `src/config.py` (see section 6).
3. In the app:
   - Enable **Accelerometer** and **Gyroscope** only. Disable the other sensors to
     keep the exports small.
   - Set the sampling rate to **100 Hz**, or the highest rate available. It must be
     at least 50 Hz.
   - Choose **CSV** as the export format.
4. Note the phone model, OS, app name and version, and sampling rate in the
   metadata log (section 5). Different phones report at different rates, which
   is why every stream is resampled to a common 50 Hz.

## 2. Phone placement

Use the **same placement for every recording** and every participant:

- Hold the phone in your right hand at waist height, screen facing up, top edge
  pointing forward.
- Grip it firmly so it doesn't slide or rotate in your hand.

For **still**, place the phone flat on a table, screen up, and don't touch it.

## 3. Activity definitions

| Activity | What to do |
|---|---|
| `standing` | Stand upright, holding the phone as above, with no deliberate movement. |
| `walking` | Walk at a normal, steady pace on level ground, holding the phone as above. |
| `jumping` | Jump in place continuously at a steady rhythm, holding the phone as above. |
| `still` | Phone lies motionless on a flat surface. No one touches it. |

## 4. Recording procedure

For each recording:

1. Get into the starting position for the activity.
2. Start the recording in the app.
3. Begin the activity right away and keep it up for the whole recording.
   Don't switch activities partway through.
4. After **8–10 seconds**, stop the recording. The assignment requires 5–10 s.
   The pipeline cuts 1 s from each end and needs at least 2 s left, so the
   pipeline accepts recordings of 4–12 s. Longer recordings give more windows:
   a 5 s recording yields 3 windows, a 10 s recording 13.
5. Export the recording and unzip it into its own folder under `data/raw/`
   (section 5).
6. Record an equal number of sessions for each activity. Check the assignment
   brief for the minimum count.

Tips:

- Tapping the screen to start and stop, or putting the phone down for `still`,
  causes movement at both ends. The 1 s trim removes most of it. Put the phone
  down before you start the recording, and keep it steady while you tap.
- If you get interrupted (stumble, drop the phone, notification pop-up), delete
  the recording and record it again.

## 5. File naming and layout

Each recording is a folder named `<participant>_<activity>_<index>`:

- `participant`: letters and digits only, e.g. `p01`
- `activity`: one of `standing`, `walking`, `jumping`, `still` (lowercase)
- `index`: a two-digit counter per participant and activity, e.g. `01`

```
data/raw/
├── p01_standing_01/
│   ├── Accelerometer.csv
│   └── Gyroscope.csv
├── p01_walking_01/
│   ├── Accelerometer.csv
│   └── Gyroscope.csv
└── metadata.csv
```

Keep a `data/raw/metadata.csv` log with one row per recording:

```
recording,participant,activity,source,phone_model,os,app,sampling_rate_hz,placement,notes
p01_walking_01,p01,walking,walking-2026-10-29_17-19-57.zip,Pixel 7,Android 14,Sensor Logger 1.x,100,right hand waist,
```

The CSV files in `data/raw/` are committed to the repository. The original
Sensor Logger zips are not (`*.zip` is ignored).

### Unseen test recordings

Test recordings go in `data/test/`, never in `data/raw/`, so they are never used
for training. Each one is a single continuous recording (about 60 s) in which you
switch between several activities. Name the folder `<participant>_mixed_<nn>` and
add one row per activity segment to `data/test/labels.csv`:

```
recording,start_s,end_s,activity
p01_mixed_01,1.0,12.5,walking
p01_mixed_01,13.5,26.3,still
```

Times are seconds from the start of the recording. Leave the handling between
activities (picking the phone up, putting it down) outside every segment. Windows
there are decoded but not scored. Then run `python -m src.test_set`.

## 6. Using a different app or CSV format

If your app exports different column names, define a mapping in `src/config.py`:

```python
MY_APP_ACCELEROMETER = ColumnMapping(
    timestamp="Time (s)",          # raw timestamp column
    timestamp_unit="s",            # one of: s, ms, us, ns
    columns={"ax": "acc_x", "ay": "acc_y", "az": "acc_z"},
)
```

If the app writes all six axes to a single CSV, map all six columns in one
`ColumnMapping` and load it with `load_sensor_csv`.

## 7. Check every recording after collecting it

Run this from the repository root (or in a notebook after the setup cell):

```python
from src.data_loader import find_recordings, load_recording
from src.preprocessing import harmonize, trim, validate_recording

for rec in find_recordings():
    acc, gyro = (trim(s) for s in load_recording(rec))
    for name, stream in (("accelerometer", acc), ("gyroscope", gyro)):
        report = validate_recording(stream)
        status = "OK " if report.is_valid else "BAD"
        print(f"{status} {rec.name} {name}: {report.duration_s:.2f} s, "
              f"{report.sampling_rate_hz:.1f} Hz {report.errors}")
```

Record again any recording that fails. Once both streams are valid,
`harmonize([acc, gyro])` returns one table at 50 Hz with the columns
`timestamp, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z`.
