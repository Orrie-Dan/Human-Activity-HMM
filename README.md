# human-activity-hmm

Human Activity Recognition (HAR) using Hidden Markov Models. The project models
sequences of activities (standing, walking, jumping, still) as hidden
states, and features extracted from sensor windows as observations. It uses
forward-backward for inference, Baum-Welch for training, and Viterbi for decoding.

> Status: complete pipeline. Data collection, feature extraction, the HMM,
> cross-validation and an unseen test set are implemented, and the notebooks walk through
> each step with plots and discussion.

## Data collection

The training set is 65 recordings (13–19 per activity, 98–137 s of
recording per activity) from one iPhone 11 Pro at 100 Hz. Recordings are about 5–10 s long and contain six sensor axes (accelerometer
and gyroscope, x/y/z) with a timestamp for every sample. Each stream has 1 s
trimmed from each end, and must then still be at least 2 s long (one feature
window). It is then validated and resampled to a common 50 Hz. See
[`docs/collection_protocol.md`](docs/collection_protocol.md) for the recording
procedure and folder layout. Recording parameters and CSV column mappings are
defined in `src/config.py`.

## Feature extraction

The harmonized signals are cut into 2 s windows with a 0.5 s step (75%
overlap). Each window yields 56 features. For each of the six axes and the
two magnitude channels (`acc_mag`, `gyro_mag`) the features are mean, std,
min, max, dominant frequency and spectral energy. On top of that, each sensor
gets a signal magnitude area and its pairwise axis correlations.

Build the labeled feature table from everything in `data/raw/`:

```bash
python -m src.dataset
```

This writes `data/processed/features.csv`, with one row per window plus its
recording, participant and activity. Features are not normalized there. Fit
any scaler on the training split only.

## Model and evaluation

`src/hmm.py` is a from-scratch Gaussian HMM with one hidden state per
activity and diagonal-covariance emissions. It implements forward-backward,
Viterbi and Baum-Welch, all in log space. Training runs in two stages:

1. **Supervised:** start probabilities, transitions, means and variances are
   estimated from the labeled windows. Transition counts get a pseudocount,
   because each recording holds a single activity, so switches between
   activities never appear in the data.
2. **Baum-Welch (optional):** EM refinement on the same sequences, without
   the labels.

Two evaluations, both reporting accuracy, the confusion matrix, and
sensitivity, specificity, precision and F1 for each class:

- **Cross-validation:** 4 folds by recording. Each fold holds out 3–5 recordings
  of every activity, and the scaler is fitted on the training recordings only.
- **Unseen test:** two ~60 s recordings from a separate session, each switching
  between several activities (`data/test/`). The model is trained on all of
  `data/raw/`, and the test windows inside a labeled segment are scored.

```bash
python -m src.test_set   # builds data/processed/test_features.csv
python -m src.training   # unseen test + cross-validation
```

| Evaluation | Supervised | + Baum-Welch |
|---|---|---|
| 4-fold CV (65 recordings, 462 windows) | 96.8% | 97.0% |
| Unseen test (2 recordings, 193 windows) | **91.2%** | 86.0% |

## Data collection videos

One video per activity (standing, walking, jumping, still), showing the full body while recording:
**[Watch the videos on Canva](https://canva.link/dd5i0wrclq0rkzu)**

Local copies are also in [`videos/`](videos/).

## Structure

```
human-activity-hmm/
├── data/
│   ├── raw/              # 65 training recordings (CSV) + metadata.csv
│   ├── test/             # 2 unseen test recordings + labels.csv (never used for training)
│   └── processed/        # Windowed features (rebuilt, not tracked)
├── docs/
│   └── collection_protocol.md
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing_features.ipynb
│   ├── 03_hmm_training.ipynb
│   └── 04_evaluation.ipynb
├── src/
│   ├── config.py         # Activities, recording parameters, column mappings, paths
│   ├── data_loader.py    # Load sensor CSVs into the canonical schema
│   ├── preprocessing.py  # Validation and sampling-rate harmonization
│   ├── features.py       # Windowing and feature extraction
│   ├── dataset.py        # Raw recordings -> labeled feature table
│   ├── hmm.py            # Gaussian HMM: forward-backward, Viterbi, Baum-Welch
│   ├── test_set.py       # Unseen test recordings -> labeled feature table
│   ├── training.py       # Training, cross-validation, unseen-test evaluation
│   ├── evaluation.py     # Accuracy, confusion matrix, per-class metrics
│   └── visualization.py  # Plots
├── videos/               # Data-collection videos, one per activity
├── tests/                # pytest suite
├── reports/figures/      # Generated figures (not tracked)
├── requirements.txt
└── pytest.ini
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run the tests from the repository root:

```bash
pytest
```

## Google Colab

Each notebook starts with a setup cell. In Colab, that cell clones this
repository, installs `requirements.txt`, and adds the repo root to `sys.path`
so `from src import ...` works. `REPO_URL` in that cell points to
<https://github.com/Orrie-Dan/Human-Activity-HMM>; change it if you fork the repo.

The recordings are in the repository, so the notebooks run without any upload.
Run them in order (01 → 04). Notebooks 03 and 04 rebuild `features.csv` themselves
if it is missing.

## Tech stack

NumPy · Pandas · SciPy · Scikit-learn · Matplotlib · pytest
