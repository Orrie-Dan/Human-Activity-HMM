"""Plotting helpers for signals, transition matrices, and results."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from src.config import ACTIVITIES, DEFAULT_RECORDING_CONFIG, TIMESTAMP_COLUMN, RecordingConfig

# Fixed activity -> color mapping (validated categorical order; never re-assigned).
# Two of these sit below 3:1 contrast, so activity names are always shown as text too.
ACTIVITY_COLORS: dict[str, str] = dict(zip(ACTIVITIES, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]))

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
NEUTRAL = "#b5b4ad"

SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list(
    "seq_blue", ["#f0efec", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281", "#0d366b"]
)
DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "div_red_blue", ["#b3302f", "#e34948", "#f0efec", "#2a78d6", "#1c5cab"]
)


def apply_style() -> None:
    """Recessive axes and grid, readable text; call once per notebook."""
    mpl.rcParams.update({
        "figure.dpi": 110,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": NEUTRAL,
        "axes.labelcolor": TEXT_SECONDARY,
        "axes.titlecolor": TEXT_PRIMARY,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "lines.linewidth": 1.2,
    })


def plot_signals(
    recordings: Mapping[str, pd.DataFrame],
    channels: Sequence[str] = ("acc_mag", "gyro_mag"),
    labels: Mapping[str, str] | None = None,
) -> plt.Figure:
    """Small multiples: one column per recording, one row per channel.

    ``recordings`` maps recording name -> signal table; ``labels`` maps the
    name to its activity (used for color).
    """
    names = list(recordings)
    fig, axes = plt.subplots(
        len(channels), len(names), figsize=(3.2 * len(names), 2.2 * len(channels)),
        sharex=True, sharey="row", squeeze=False,
    )
    for j, name in enumerate(names):
        df = recordings[name]
        activity = (labels or {}).get(name, name)
        color = ACTIVITY_COLORS.get(activity, TEXT_SECONDARY)
        t = df[TIMESTAMP_COLUMN] - df[TIMESTAMP_COLUMN].iloc[0]
        for i, channel in enumerate(channels):
            ax = axes[i, j]
            ax.plot(t, df[channel], color=color)
            if i == 0:
                ax.set_title(f"{activity}\n{name}", fontsize=9)
            if j == 0:
                ax.set_ylabel(channel)
            if i == len(channels) - 1:
                ax.set_xlabel("time (s)")
    fig.tight_layout()
    return fig


def plot_trim(
    raw: pd.DataFrame, channel: str, title: str,
    config: RecordingConfig = DEFAULT_RECORDING_CONFIG,
) -> plt.Figure:
    """Show one raw channel with the trimmed start/end regions shaded."""
    t = raw[TIMESTAMP_COLUMN] - raw[TIMESTAMP_COLUMN].iloc[0]
    end = t.iloc[-1]
    fig, ax = plt.subplots(figsize=(8, 2.4))
    ax.plot(t, raw[channel], color=TEXT_SECONDARY)
    for lo, hi in ((0, config.trim_start_s), (end - config.trim_end_s, end)):
        ax.axvspan(lo, hi, color=GRID, alpha=0.9, lw=0)
    ax.text(config.trim_start_s / 2, 1.0, "trimmed", transform=ax.get_xaxis_transform(),
            ha="center", va="bottom", fontsize=8, color=TEXT_SECONDARY)
    ax.text(end - config.trim_end_s / 2, 1.0, "trimmed", transform=ax.get_xaxis_transform(),
            ha="center", va="bottom", fontsize=8, color=TEXT_SECONDARY)
    ax.set(title=title, xlabel="time (s)", ylabel=channel)
    fig.tight_layout()
    return fig


def plot_feature_by_activity(
    table: pd.DataFrame,
    features: Sequence[str],
    log_scale: Sequence[str] = (),
    ncols: int = 3,
) -> plt.Figure:
    """Strip plot of each feature per activity (one dot per window, bar = median)."""
    nrows = int(np.ceil(len(features) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.6 * nrows), squeeze=False)
    rng = np.random.default_rng(0)
    for ax, feature in zip(axes.flat, features):
        for x, activity in enumerate(ACTIVITIES):
            values = table.loc[table["activity"] == activity, feature].to_numpy()
            jitter = rng.uniform(-0.18, 0.18, size=len(values))
            ax.scatter(x + jitter, values, s=10, color=ACTIVITY_COLORS[activity],
                       alpha=0.75, linewidths=0)
            ax.hlines(np.median(values), x - 0.3, x + 0.3, color=TEXT_PRIMARY, lw=1.5)
        ax.set_xticks(range(len(ACTIVITIES)), ACTIVITIES)
        ax.grid(axis="x", visible=False)
        if feature in log_scale:
            ax.set_yscale("log")
        ax.set_title(feature, fontsize=9)
    for ax in axes.flat[len(features):]:
        ax.set_visible(False)
    fig.tight_layout()
    return fig


def plot_matrix(
    matrix: pd.DataFrame,
    title: str,
    fmt: str = "{:.2f}",
    cmap=SEQUENTIAL_CMAP,
    vmin: float | None = None,
    vmax: float | None = None,
    ax: plt.Axes | None = None,
    xlabel: str = "",
    ylabel: str = "",
) -> plt.Axes:
    """Annotated heatmap (transition matrices, confusion matrices, emission means)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(0.8 * matrix.shape[1] + 2, 0.6 * matrix.shape[0] + 1.2))
    values = matrix.to_numpy(dtype=float)
    image = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    norm = image.norm
    for (i, j), v in np.ndenumerate(values):
        dark_cell = cmap(norm(v))[:3] @ np.array([0.299, 0.587, 0.114]) < 0.55
        ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=8,
                color="white" if dark_cell else TEXT_PRIMARY)
    ax.set_xticks(range(matrix.shape[1]), matrix.columns, rotation=0)
    ax.set_yticks(range(matrix.shape[0]), matrix.index)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    return ax


def plot_history(history: Sequence[float], title: str = "Baum-Welch convergence") -> plt.Figure:
    """Total training log-likelihood per EM iteration."""
    fig, ax = plt.subplots(figsize=(5, 2.6))
    ax.plot(range(1, len(history) + 1), history, color=ACTIVITY_COLORS[ACTIVITIES[0]],
            lw=2, marker="o", ms=4)
    ax.set(title=title, xlabel="iteration", ylabel="log-likelihood")
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    fig.tight_layout()
    return fig


def plot_decoded(
    times: Sequence[float],
    true: Sequence[str],
    predicted: Sequence[str],
    title: str,
    ax: plt.Axes | None = None,
    legend: bool = True,
) -> plt.Axes:
    """Viterbi path over time: predicted activity per window, true activity as a band."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 2.2))
    state = {a: i for i, a in enumerate(ACTIVITIES)}
    times = np.asarray(times, dtype=float)
    # Windows whose true label is not an activity (e.g. "transition") leave a gap in the band.
    true_idx = np.array([state.get(a, np.nan) for a in true], dtype=float)
    pred_idx = np.array([state[a] for a in predicted])
    ax.step(times, true_idx, where="post", color=GRID, lw=8, solid_capstyle="butt",
            label="true", zorder=1)
    ax.step(times, pred_idx, where="post", color=NEUTRAL, lw=1, zorder=2)
    for a in ACTIVITIES:
        hit = pred_idx == state[a]
        if hit.any():
            ax.scatter(times[hit], pred_idx[hit], s=28, color=ACTIVITY_COLORS[a],
                       edgecolors="white", linewidths=1, zorder=3)
    ax.set_yticks(range(len(ACTIVITIES)), ACTIVITIES)
    ax.set_ylim(-0.6, len(ACTIVITIES) - 0.4)
    ax.set(title=title, xlabel="window start (s)")
    if legend:  # outside the plot area, right of the axes, so it never covers title or data
        ax.legend(handles=[mpl.lines.Line2D([], [], color=GRID, lw=8, label="true activity"),
                           mpl.lines.Line2D([], [], color=NEUTRAL, marker="o", ms=5,
                                            markerfacecolor=TEXT_SECONDARY, label="predicted")],
                  loc="upper left", bbox_to_anchor=(1.01, 1))
    return ax


def plot_emissions(
    means: np.ndarray,
    variances: np.ndarray,
    feature_columns: Sequence[str],
    features: Sequence[str],
    ncols: int = 3,
) -> plt.Figure:
    """Gaussian emission densities p(x_f | state) for selected features, one panel each.

    ``means``/``variances`` are the HMM's (n_states, n_features) parameters in
    standardized units. The y axis is symlog so broad and narrow states both show.
    """
    nrows = int(np.ceil(len(features) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 2.5 * nrows), squeeze=False)
    for ax, feature in zip(axes.flat, features):
        f = list(feature_columns).index(feature)
        mu, sd = means[:, f], np.sqrt(variances[:, f])
        x = np.linspace((mu - 3.5 * sd).min(), (mu + 3.5 * sd).max(), 600)
        for k, activity in enumerate(ACTIVITIES):
            pdf = np.exp(-0.5 * ((x - mu[k]) / sd[k]) ** 2) / (sd[k] * np.sqrt(2 * np.pi))
            ax.plot(x, pdf, color=ACTIVITY_COLORS[activity], lw=1.8, label=activity)
            ax.fill_between(x, pdf, color=ACTIVITY_COLORS[activity], alpha=0.12, lw=0)
        ax.set_yscale("symlog", linthresh=0.05)
        ax.set_ylim(bottom=0)
        ax.set_title(feature, fontsize=9)
        ax.set_xlabel("standardized value")
    for ax in axes.flat[len(features):]:
        ax.set_visible(False)
    axes.flat[0].set_ylabel("density (symlog)")
    axes.flat[min(len(features), ncols) - 1].legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout()
    return fig


def plot_test_timeline(
    signals: pd.DataFrame,
    segments: pd.DataFrame,
    title: str,
    channel: str = "acc_mag",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """One long test recording with its labeled activity segments shaded."""
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 2.2))
    ax.plot(signals[TIMESTAMP_COLUMN], signals[channel], color=TEXT_SECONDARY, lw=0.7)
    for seg in segments.itertuples():
        ax.axvspan(seg.start_s, seg.end_s, color=ACTIVITY_COLORS[seg.activity], alpha=0.18, lw=0)
        ax.text((seg.start_s + seg.end_s) / 2, 1.0, seg.activity, transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=8, color=TEXT_PRIMARY)
    ax.set(xlabel="time (s)", ylabel=channel)
    ax.set_title(title, pad=14)
    ax.set_xlim(signals[TIMESTAMP_COLUMN].iloc[0], signals[TIMESTAMP_COLUMN].iloc[-1])
    return ax
