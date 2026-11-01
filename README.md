# human-activity-hmm

Human Activity Recognition (HAR) using Hidden Markov Models. The project models
sequences of activities (for example walking, standing, or sitting) as hidden
states, and features extracted from sensor windows as observations. It uses
forward-backward for inference, Baum-Welch for training, and Viterbi for decoding.

> Status: complete pipeline. Data collection, feature extraction, the HMM and
> cross-validated evaluation are implemented, and the notebooks walk through
> each step with plots and discussion.

## Data collection

Recordings are about 5–10 s long and contain six sensor axes (accelerometer
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

`python -m src.training` runs a 4-fold cross-validation by recording. Each
fold holds out one recording per activity, and the scaler is fitted on the
training recordings only. It reports accuracy, the confusion matrix, and
sensitivity, specificity, precision and F1 for each class.

## Structure

```
human-activity-hmm/
├── data/
│   ├── raw/              # Original sensor recordings (not tracked)
│   └── processed/        # Windowed features (not tracked)
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
│   ├── training.py       # Training and recording-level cross-validation
│   ├── evaluation.py     # Accuracy, confusion matrix, per-class metrics
│   └── visualization.py  # Plots
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

Raw recordings are not in git. When `data/raw/` is empty, the setup cell asks
you to upload a zip containing the `p01_<activity>_<nn>/` folders and
`metadata.csv`. Run the notebooks in order (01 → 04). Notebooks 03 and 04
rebuild `features.csv` themselves if it is missing.

## Tech stack

NumPy · Pandas · SciPy · Scikit-learn · Matplotlib · pytest
