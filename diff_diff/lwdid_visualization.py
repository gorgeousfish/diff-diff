"""Visualization methods for LWDiD results.

Provides plotting functions for cohort trends, event studies,
sensitivity analysis, and bootstrap distributions.

Requires matplotlib (optional dependency). If not installed,
raises VisualizationError with installation instructions.

Note
----
All plot functions return a matplotlib Figure object without closing it.
In batch/loop usage, call ``plt.close(fig)`` after saving or displaying
each figure to avoid memory accumulation.
"""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from diff_diff.lwdid_exceptions import VisualizationError


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        raise VisualizationError(
            "matplotlib is required for LWDiD visualization. "
            "Install with: pip install matplotlib"
        )


def plot_cohort_trends(
    data: pd.DataFrame,
    outcome: str,
    unit: str,
    time: str,
    treatment: str,
    cohort: Optional[str] = None,
    title: Optional[str] = None,
    figsize: tuple = (10, 6),
    show_ci: bool = True,
    ax=None,
):
    """Plot pre/post outcome trajectories by treatment group (or cohort).

    Shows average outcomes over time for treated vs control groups,
    with optional confidence intervals.
    """
    plt = _require_matplotlib()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Compute group means by time
    # Identify ever-treated units
    treated_units = data.loc[data[treatment] == 1, unit].unique()
    data = data.copy()
    data["_ever_treated"] = data[unit].isin(treated_units).astype(int)

    # Group averages
    group_means = (
        data.groupby([time, "_ever_treated"])[outcome].agg(["mean", "std", "count"]).reset_index()
    )
    group_means["se"] = group_means["std"] / np.sqrt(group_means["count"])

    for grp, label, color in [(1, "Treated", "steelblue"), (0, "Control", "coral")]:
        gdf = group_means[group_means["_ever_treated"] == grp]
        ax.plot(gdf[time], gdf["mean"], "o-", label=label, color=color)
        if show_ci:
            ax.fill_between(
                gdf[time],
                gdf["mean"] - 1.96 * gdf["se"],
                gdf["mean"] + 1.96 * gdf["se"],
                alpha=0.15,
                color=color,
            )

    # Mark treatment onset
    treated_times = data.loc[data[treatment] == 1, time]
    if len(treated_times) > 0:
        first_treat = treated_times.min()
        ax.axvline(
            first_treat - 0.5, color="gray", linestyle="--", alpha=0.7, label="Treatment onset"
        )

    ax.set_xlabel("Time")
    ax.set_ylabel(outcome)
    ax.set_title(title or "LWDiD: Cohort Trends")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig


def plot_event_study(
    period_effects: Dict[Any, Dict],
    title: Optional[str] = None,
    figsize: tuple = (10, 6),
    ax=None,
):
    """Plot event-study style graph of period-specific ATTs.

    Parameters
    ----------
    period_effects : dict
        From LWDiDResults.period_effects (period -> {att, se, ...}).
    """
    plt = _require_matplotlib()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    periods = sorted(period_effects.keys())
    atts = [period_effects[p]["att"] for p in periods]
    ses = [period_effects[p].get("se", 0) for p in periods]

    ax.errorbar(periods, atts, yerr=[1.96 * s for s in ses], fmt="o-", capsize=3, color="steelblue")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Period")
    ax.set_ylabel("ATT")
    ax.set_title(title or "LWDiD: Period-Specific Effects")
    ax.grid(True, alpha=0.3)

    return fig


def plot_sensitivity(
    sensitivity_result,
    title: Optional[str] = None,
    figsize: tuple = (10, 6),
    ax=None,
):
    """Plot sensitivity analysis results.

    Shows ATT estimates across different specifications with
    confidence bands, highlighting the baseline estimate.
    """
    plt = _require_matplotlib()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    specs = sensitivity_result.specifications
    x = range(len(specs))
    atts = [s.att for s in specs]
    ses = [s.se for s in specs]
    labels = [s.label for s in specs]

    ax.errorbar(x, atts, yerr=[1.96 * s for s in ses], fmt="o", capsize=3, color="steelblue")
    ax.axhline(
        sensitivity_result.baseline_att,
        color="red",
        linestyle="--",
        alpha=0.7,
        label="Baseline ATT",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("ATT")
    ax.set_title(
        title or f"Sensitivity Analysis (robustness: {sensitivity_result.robustness_level})"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig


def plot_bootstrap_distribution(
    t_stats: np.ndarray,
    t_observed: float,
    title: Optional[str] = None,
    figsize: tuple = (8, 5),
    ax=None,
):
    """Plot bootstrap t-statistic distribution with observed value."""
    plt = _require_matplotlib()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    ax.hist(t_stats, bins=50, density=True, alpha=0.7, color="steelblue", edgecolor="white")
    ax.axvline(t_observed, color="red", linewidth=2, label=f"t_obs = {t_observed:.3f}")
    ax.axvline(-t_observed, color="red", linewidth=2, linestyle="--", alpha=0.5)
    ax.set_xlabel("t-statistic")
    ax.set_ylabel("Density")
    ax.set_title(title or "Wild Cluster Bootstrap Distribution")
    ax.legend()

    return fig
