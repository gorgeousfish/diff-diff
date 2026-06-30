"""Parallel trends diagnostics for LWDiD.

Implements pre-treatment effect testing to validate the parallel trends
assumption required by Lee & Wooldridge (2025, 2026).

The key idea: under correct specification and parallel trends,
pre-treatment ATT estimates should be zero. Significant pre-treatment
effects indicate violation of parallel trends.

The conditional heterogeneous trends (CHT) framework allows each treatment
cohort to have its own linear trend, relaxing the standard parallel trends
assumption. Under CHT, demeaning is more efficient when parallel trends
holds, while detrending removes cohort-specific linear trends and restores
consistency when parallel trends fails.

References
----------
Lee, S. J. & Wooldridge, J. M. (2025). Section 4, Assumption 4.6 (CPTS). SSRN 4516518.
Lee, S. J. & Wooldridge, J. M. (2026). "Simple Approaches to Inference
  with Difference-in-Differences Estimators with Small Cross-Sectional
  Sample Sizes." SSRN 5325686.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from diff_diff.lwdid_exceptions import (
    DiagnosticError,
    DiagnosticWarning,
    InsufficientPrePeriodsError,
)

# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class PreTrendEstimate:
    """Pre-treatment ATT estimate for a single period.

    Stores the estimated treatment effect for a pre-treatment period,
    used for placebo tests and parallel trends assessment. Under the null
    hypothesis of parallel trends, these estimates should be statistically
    indistinguishable from zero.

    Attributes
    ----------
    period : int
        Calendar period (pseudo-post) used for this estimate.
    att : float
        Estimated average treatment effect on the treated.
    se : float
        Standard error of the ATT estimate.
    t_stat : float
        t-statistic computed as att / se.
    pvalue : float
        Two-sided p-value for testing H0: ATT = 0.
    """

    period: int
    att: float
    se: float
    t_stat: float
    pvalue: float

    @property
    def is_significant(self) -> bool:
        """Whether estimate is significant at 5% level."""
        return self.pvalue < 0.05


@dataclass
class ParallelTrendsTestResult:
    """Results from testing the parallel trends assumption.

    Aggregates pre-treatment ATT estimates and joint test statistics to
    assess whether the parallel trends assumption is likely to hold.

    Attributes
    ----------
    method : str
        Testing method used: 'placebo', 'joint_f', or 'regression'.
    test_stat : float
        Test statistic (chi-squared for joint Wald test).
    pvalue : float
        P-value for the overall test.
    decision : str
        Decision outcome: 'pass', 'fail', or 'inconclusive'.
    pre_treatment_effects : list of PreTrendEstimate
        Pre-treatment ATT estimates by period.
    n_pre_periods : int
        Total number of pre-treatment periods available.
    significance_level : float
        Significance level used for the decision rule.
    """

    method: str
    test_stat: float
    pvalue: float
    decision: str
    pre_treatment_effects: List[PreTrendEstimate]
    n_pre_periods: int
    significance_level: float

    @property
    def n_tested_periods(self) -> int:
        """Number of periods actually tested."""
        return len(self.pre_treatment_effects)

    @property
    def max_pre_att(self) -> float:
        """Maximum absolute pre-treatment ATT."""
        if not self.pre_treatment_effects:
            return np.nan
        return max(abs(e.att) for e in self.pre_treatment_effects)

    def summary(self) -> str:
        """Generate human-readable summary of test results."""
        lines = [
            "=" * 60,
            "PARALLEL TRENDS TEST",
            "=" * 60,
            "",
            f"Method: {self.method}",
            f"Test statistic: {self.test_stat:.4f}",
            f"P-value: {self.pvalue:.4f}",
            f"Decision (alpha={self.significance_level}): {self.decision.upper()}",
            "",
            f"Pre-treatment periods: {self.n_pre_periods}",
            f"Periods tested: {self.n_tested_periods}",
            "",
        ]

        if self.pre_treatment_effects:
            lines.append("Period-specific pre-treatment ATTs:")
            lines.append(f"  {'Period':<8} {'ATT':<10} {'SE':<10} {'t':<8} {'p':<8}")
            lines.append("  " + "-" * 44)
            for e in self.pre_treatment_effects:
                sig = "*" if e.pvalue < 0.05 else ""
                lines.append(
                    f"  {e.period:<8} {e.att:<10.4f} {e.se:<10.4f} "
                    f"{e.t_stat:<8.3f} {e.pvalue:<8.4f}{sig}"
                )

        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass
class CohortTrendEstimate:
    """Estimated linear trend for a cohort in pre-treatment period.

    Attributes
    ----------
    cohort : int
        Cohort identifier (first treatment period or group label).
    slope : float
        Estimated linear time trend slope.
    slope_se : float
        Standard error of the slope estimate.
    slope_pvalue : float
        Two-sided p-value for testing H0: slope = 0.
    n_units : int
        Number of units in this cohort.
    n_pre_periods : int
        Number of pre-treatment periods used.
    r_squared : float
        R-squared of the trend regression.
    """

    cohort: int
    slope: float
    slope_se: float
    slope_pvalue: float
    n_units: int
    n_pre_periods: int
    r_squared: float

    @property
    def has_significant_trend(self) -> bool:
        """Whether cohort has significant linear trend at 5%."""
        return self.slope_pvalue < 0.05


@dataclass
class HeterogeneousTrendsDiagnostics:
    """Results from diagnosing heterogeneous trends across cohorts.

    Attributes
    ----------
    cht_detected : bool
        Whether conditional heterogeneous trends are detected.
    trend_diff_pvalue : float
        P-value from testing equality of trends across groups.
    treated_slope : float
        Average pre-treatment trend slope for treated group.
    control_slope : float
        Average pre-treatment trend slope for control group.
    slope_difference : float
        Difference in slopes (treated - control).
    slope_diff_se : float
        Standard error of the slope difference.
    cohort_trends : List[CohortTrendEstimate]
        Per-cohort trend estimates.
    """

    cht_detected: bool
    trend_diff_pvalue: float
    treated_slope: float
    control_slope: float
    slope_difference: float
    slope_diff_se: float
    cohort_trends: List[CohortTrendEstimate] = field(default_factory=list)

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 60,
            "HETEROGENEOUS TRENDS DIAGNOSTICS",
            "=" * 60,
            "",
            f"CHT detected: {'YES' if self.cht_detected else 'NO'}",
            f"Trend difference p-value: {self.trend_diff_pvalue:.4f}",
            "",
            f"Treated group slope: {self.treated_slope:.6f}",
            f"Control group slope: {self.control_slope:.6f}",
            f"Difference: {self.slope_difference:.6f} (SE={self.slope_diff_se:.6f})",
            "",
        ]

        if self.cohort_trends:
            lines.append("Cohort-specific trends:")
            for ct in self.cohort_trends:
                sig = "*" if ct.has_significant_trend else ""
                lines.append(
                    f"  Cohort {ct.cohort}: slope={ct.slope:.6f} "
                    f"(SE={ct.slope_se:.6f}, p={ct.slope_pvalue:.4f}){sig}"
                )

        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass
class TransformationRecommendation:
    """Comprehensive recommendation for transformation method selection.

    Combines parallel trends test results and heterogeneous trends
    diagnostics to provide an informed recommendation on whether to
    use demean, detrend, or their seasonal variants.

    Attributes
    ----------
    recommended : str
        Primary recommendation: 'demean', 'detrend', 'demeanq', or 'detrendq'.
    confidence : str
        Confidence level: 'high', 'medium', or 'low'.
    rationale : str
        Explanation for the recommendation.
    parallel_trends_result : ParallelTrendsTestResult
        Results from the parallel trends test used for recommendation.
    alternative : str or None
        Alternative method if primary is uncertain.
    """

    recommended: str
    confidence: str
    rationale: str
    parallel_trends_result: ParallelTrendsTestResult
    alternative: Optional[str] = None

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 60,
            "TRANSFORMATION RECOMMENDATION",
            "=" * 60,
            "",
            f"Recommended: rolling='{self.recommended}'",
            f"Confidence: {self.confidence}",
            f"Rationale: {self.rationale}",
            "",
        ]

        if self.alternative:
            lines.append(f"Alternative: rolling='{self.alternative}'")
            lines.append("")

        lines.append(f"Based on parallel trends test: {self.parallel_trends_result.decision}")
        lines.append("=" * 60)
        return "\n".join(lines)


# =============================================================================
# Helper Functions
# =============================================================================


def _identify_pre_periods(data: pd.DataFrame, time: str, treatment: str, unit: str) -> tuple:
    """Identify pre-treatment periods from data.

    Returns
    -------
    tuple of (list, int)
        (pre_periods sorted, first_treat_time)
    """
    treated_times = data.loc[data[treatment] == 1, time].unique()
    if len(treated_times) == 0:
        raise DiagnosticError("No treated observations found in the data.")

    first_treat = int(min(treated_times))
    all_times = sorted(data[time].unique())
    pre_periods = [t for t in all_times if t < first_treat]

    return pre_periods, first_treat


def _estimate_group_slope(data: pd.DataFrame, outcome: str, unit: str, time: str) -> tuple:
    """Estimate average linear trend slope for a group of units.

    Uses pooled OLS: Y_it = alpha_i + beta * t + eps_it
    Returns (slope, slope_se, n_units, n_periods, r_squared).
    """
    # Demean at unit level for fixed effects, then regress on time
    units = data[unit].unique()
    n_units = len(units)

    if n_units == 0 or data.empty:
        return 0.0, np.inf, 0, 0, 0.0

    periods = sorted(data[time].unique())
    n_periods = len(periods)

    if n_periods < 2:
        return 0.0, np.inf, n_units, n_periods, 0.0

    # Pooled OLS with unit demeaning
    df = data[[unit, time, outcome]].copy()
    unit_means = df.groupby(unit)[outcome].transform("mean")
    time_means = df.groupby(unit)[time].transform("mean")
    y_dm = df[outcome] - unit_means
    t_dm = df[time].astype(float) - time_means

    # beta = sum(t_dm * y_dm) / sum(t_dm^2)
    ss_t = (t_dm**2).sum()
    if ss_t < 1e-12:
        return 0.0, np.inf, n_units, n_periods, 0.0

    slope = (t_dm * y_dm).sum() / ss_t

    # Residuals and SE
    resid = y_dm - slope * t_dm
    n_obs = len(df)
    dof = n_obs - n_units - 1  # unit FE + slope
    if dof <= 0:
        dof = 1

    sigma2 = (resid**2).sum() / dof
    slope_se = np.sqrt(sigma2 / ss_t)

    # R-squared
    ss_tot = (y_dm**2).sum()
    r_sq = 1 - (resid**2).sum() / ss_tot if ss_tot > 0 else 0.0

    return slope, slope_se, n_units, n_periods, r_sq


def _safe_lwdid_fit(
    data: pd.DataFrame,
    outcome: str,
    unit: str,
    time: str,
    treatment: str,
    rolling: str = "demean",
    vce: str = "hc1",
):
    """Safely fit LWDiD model, returning None on failure."""
    from diff_diff.lwdid import LWDiD

    try:
        est = LWDiD(rolling=rolling, vce=vce)
        result = est.fit(data, outcome=outcome, unit=unit, time=time, treatment=treatment)
        return result
    except (ValueError, np.linalg.LinAlgError, RuntimeError):
        return None


# =============================================================================
# Core Functions
# =============================================================================


def test_parallel_trends(
    data: pd.DataFrame,
    outcome: str = None,
    unit: str = None,
    time: str = None,
    treatment: str = None,
    cohort: Optional[str] = None,
    rolling: str = "demean",
    alpha: float = 0.05,
    # lwdid-py compatible aliases
    y: Optional[str] = None,
    ivar: Optional[str] = None,
    tvar: Optional[str] = None,
    d: Optional[str] = None,
    gvar: Optional[str] = None,
    **kwargs,
) -> ParallelTrendsTestResult:
    """Test the parallel trends assumption via placebo pre-treatment ATTs.

    For each pre-treatment period (except the first baseline period),
    creates a pseudo-treatment indicator and estimates a placebo ATT
    using LWDiD. A joint Wald test assesses whether all pre-treatment
    ATTs are jointly zero.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data with columns for outcome, unit, time, and treatment.
    outcome : str
        Name of the outcome variable column. (alias: y)
    unit : str
        Name of the unit identifier column. (alias: ivar)
    time : str
        Name of the time variable column. (alias: tvar)
    treatment : str
        Name of the binary treatment indicator column (D_it). (alias: d)
    cohort : str or None, optional
        Name of the cohort variable column (for staggered designs).
        If None, common timing is assumed. (alias: gvar)
    rolling : str, default 'demean'
        Transformation method to use for placebo estimation.
    alpha : float, default 0.05
        Significance level for the decision rule.

    Returns
    -------
    ParallelTrendsTestResult
        Test results including per-period estimates, joint statistic,
        and decision.

    Raises
    ------
    DiagnosticError
        If no treated observations are found.
    InsufficientPrePeriodsError
        If fewer than 2 pre-treatment periods are available.

    Notes
    -----
    Decision rule:
    - If joint p-value < alpha: 'fail' (reject parallel trends)
    - If joint p-value > 0.1: 'pass' (fail to reject)
    - Otherwise: 'inconclusive'

    The joint test is a Wald chi-squared test assuming independence of
    the per-period placebo estimates:
        chi2 = sum((ATT_s / SE_s)^2), df = number of valid estimates.

    Examples
    --------
    >>> result = test_parallel_trends(df, 'y', 'unit', 'time', 'treat')
    >>> print(result.decision)
    'pass'
    """
    # Resolve lwdid-py aliases
    outcome = outcome or y
    unit = unit or ivar
    time = time or tvar
    treatment = treatment or d
    cohort = cohort or gvar

    # Validate required params
    if outcome is None:
        raise ValueError("'outcome' (or 'y') parameter is required")
    if unit is None:
        raise ValueError("'unit' (or 'ivar') parameter is required")
    if time is None:
        raise ValueError("'time' (or 'tvar') parameter is required")
    if treatment is None:
        raise ValueError("'treatment' (or 'd') parameter is required")

    # Validate inputs
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"data must be a pandas DataFrame, got {type(data).__name__}.")
    if data.empty:
        raise DiagnosticError("data must not be empty.")
    for col_name, col_val in [
        ("outcome", outcome),
        ("unit", unit),
        ("time", time),
        ("treatment", treatment),
    ]:
        if col_val not in data.columns:
            raise DiagnosticError(
                f"Column '{col_val}' (specified as {col_name}) not found in data. "
                f"Available columns: {list(data.columns)}"
            )
    if rolling not in ("demean", "detrend", "demeanq", "detrendq"):
        raise ValueError(
            f"rolling must be one of ('demean', 'detrend', 'demeanq', 'detrendq'), "
            f"got '{rolling}'"
        )
    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    # Identify pre-treatment periods
    pre_periods, first_treat = _identify_pre_periods(data, time, treatment, unit)

    if len(pre_periods) < 2:
        raise InsufficientPrePeriodsError(
            f"Need at least 2 pre-treatment periods for parallel trends test, "
            f"got {len(pre_periods)}."
        )

    # For each pre-period (except the first which serves as baseline),
    # create a pseudo-treatment indicator and estimate placebo ATT.
    pre_effects: List[PreTrendEstimate] = []

    # Identify ever-treated units
    ever_treated = data.groupby(unit)[treatment].max() > 0
    treated_units = set(ever_treated[ever_treated].index)

    for pseudo_post_start in pre_periods[1:]:
        # Create pseudo dataset: only data up to pseudo_post_start
        sub = data[data[time] <= pseudo_post_start].copy()

        # Pseudo-treatment: treated group in pseudo-post period
        sub["_pseudo_treat"] = 0
        mask = sub[unit].isin(treated_units) & (sub[time] >= pseudo_post_start)
        sub.loc[mask, "_pseudo_treat"] = 1

        # Need at least some treated and control observations
        if sub["_pseudo_treat"].sum() == 0 or (sub["_pseudo_treat"] == 0).sum() == 0:
            continue

        # Check we have enough pre-periods for the transformation
        sub_pre_periods = sorted(sub.loc[sub["_pseudo_treat"] == 0, time].unique())
        # For detrend we need at least 2 pre-periods in the subset
        if rolling in ("detrend", "detrendq") and len(sub_pre_periods) < 2:
            continue
        if len(sub_pre_periods) < 1:
            continue

        # Fit LWDiD on this subset
        result = _safe_lwdid_fit(
            sub, outcome, unit, time, "_pseudo_treat", rolling=rolling, vce="hc1"
        )

        if result is not None and np.isfinite(result.att) and result.se > 0:
            t_stat = result.att / result.se
            pval = 2 * (1 - stats.norm.cdf(abs(t_stat)))
            pre_effects.append(
                PreTrendEstimate(
                    period=int(pseudo_post_start),
                    att=float(result.att),
                    se=float(result.se),
                    t_stat=float(t_stat),
                    pvalue=float(pval),
                )
            )

    # Joint Wald test: H0: all pre-ATTs = 0
    chi2 = np.nan
    pvalue = np.nan

    if len(pre_effects) > 0:
        atts = np.array([e.att for e in pre_effects])
        ses = np.array([e.se for e in pre_effects])

        valid = ses > 0
        if valid.any():
            chi2 = float(np.sum((atts[valid] / ses[valid]) ** 2))
            df = int(valid.sum())
            pvalue = float(1 - stats.chi2.cdf(chi2, df))

    # Decision rule
    if np.isnan(pvalue):
        decision = "inconclusive"
    elif pvalue < alpha:
        decision = "fail"
    elif pvalue > 0.1:
        decision = "pass"
    else:
        decision = "inconclusive"

    # Warn if few periods tested
    if len(pre_effects) < 2:
        warnings.warn(
            f"Only {len(pre_effects)} pre-treatment period(s) could be tested. "
            "Results may have low power.",
            DiagnosticWarning,
            stacklevel=2,
        )

    return ParallelTrendsTestResult(
        method="placebo",
        test_stat=chi2,
        pvalue=pvalue,
        decision=decision,
        pre_treatment_effects=pre_effects,
        n_pre_periods=len(pre_periods),
        significance_level=alpha,
    )


def diagnose_heterogeneous_trends(
    data: pd.DataFrame,
    outcome: str = None,
    unit: str = None,
    time: str = None,
    treatment: str = None,
    cohort: Optional[str] = None,
    alpha: float = 0.05,
    # lwdid-py compatible aliases
    y: Optional[str] = None,
    ivar: Optional[str] = None,
    tvar: Optional[str] = None,
    d: Optional[str] = None,
    gvar: Optional[str] = None,
    **kwargs,
) -> HeterogeneousTrendsDiagnostics:
    """Diagnose heterogeneous trends across treated and control groups.

    Estimates unit-level linear trends in the pre-treatment period for
    treated and control groups separately, then tests whether the average
    trend slopes differ significantly.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data.
    outcome : str
        Name of the outcome variable column. (alias: y)
    unit : str
        Name of the unit identifier column. (alias: ivar)
    time : str
        Name of the time variable column. (alias: tvar)
    treatment : str
        Name of the binary treatment indicator column. (alias: d)
    cohort : str or None, optional
        Name of the cohort variable column. (alias: gvar)
    alpha : float, default 0.05
        Significance level for detecting CHT.

    Returns
    -------
    HeterogeneousTrendsDiagnostics
        Diagnostic results including per-cohort trends and overall test.

    Raises
    ------
    DiagnosticError
        If no treated observations are found.
    InsufficientPrePeriodsError
        If fewer than 2 pre-treatment periods.

    Notes
    -----
    Under the standard parallel trends assumption, treated and control
    groups should have equal pre-treatment slopes. If slopes differ
    significantly, the conditional heterogeneous trends (CHT) assumption
    may hold, and detrending is recommended.
    """
    # Resolve lwdid-py aliases
    outcome = outcome or y
    unit = unit or ivar
    time = time or tvar
    treatment = treatment or d
    cohort = cohort or gvar

    # Validate required params
    if outcome is None:
        raise ValueError("'outcome' (or 'y') parameter is required")
    if unit is None:
        raise ValueError("'unit' (or 'ivar') parameter is required")
    if time is None:
        raise ValueError("'time' (or 'tvar') parameter is required")
    if treatment is None:
        raise ValueError("'treatment' (or 'd') parameter is required")

    # Identify pre-treatment periods
    pre_periods, first_treat = _identify_pre_periods(data, time, treatment, unit)

    if len(pre_periods) < 2:
        raise InsufficientPrePeriodsError(
            f"Need at least 2 pre-treatment periods for trend diagnosis, "
            f"got {len(pre_periods)}."
        )

    # Restrict to pre-treatment data
    pre_data = data[data[time] < first_treat].copy()

    # Identify treated vs control units
    ever_treated = data.groupby(unit)[treatment].max() > 0
    treated_units = set(ever_treated[ever_treated].index)
    control_units = set(ever_treated[~ever_treated].index)

    if not treated_units:
        raise DiagnosticError("No treated units identified.")
    if not control_units:
        raise DiagnosticError("No control units identified.")

    # Estimate slopes for each group
    treated_pre = pre_data[pre_data[unit].isin(treated_units)]
    control_pre = pre_data[pre_data[unit].isin(control_units)]

    t_slope, t_se, t_n, t_np, t_r2 = _estimate_group_slope(treated_pre, outcome, unit, time)
    c_slope, c_se, c_n, c_np, c_r2 = _estimate_group_slope(control_pre, outcome, unit, time)

    # Test for difference in slopes
    slope_diff = t_slope - c_slope
    slope_diff_se = np.sqrt(t_se**2 + c_se**2) if (t_se < np.inf and c_se < np.inf) else np.inf

    if slope_diff_se > 0 and slope_diff_se < np.inf:
        z_stat = slope_diff / slope_diff_se
        trend_diff_pvalue = float(2 * (1 - stats.norm.cdf(abs(z_stat))))
    else:
        trend_diff_pvalue = np.nan

    cht_detected = not np.isnan(trend_diff_pvalue) and trend_diff_pvalue < alpha

    # Build cohort-level trend estimates
    cohort_trends = []

    # Treated cohort estimate
    if t_n > 0 and t_se < np.inf:
        t_pval = float(2 * (1 - stats.norm.cdf(abs(t_slope / t_se)))) if t_se > 0 else np.nan
        cohort_trends.append(
            CohortTrendEstimate(
                cohort=first_treat,
                slope=float(t_slope),
                slope_se=float(t_se),
                slope_pvalue=t_pval,
                n_units=int(t_n),
                n_pre_periods=int(t_np),
                r_squared=float(t_r2),
            )
        )

    # Control cohort estimate (cohort=0 for never-treated)
    if c_n > 0 and c_se < np.inf:
        c_pval = float(2 * (1 - stats.norm.cdf(abs(c_slope / c_se)))) if c_se > 0 else np.nan
        cohort_trends.append(
            CohortTrendEstimate(
                cohort=0,
                slope=float(c_slope),
                slope_se=float(c_se),
                slope_pvalue=c_pval,
                n_units=int(c_n),
                n_pre_periods=int(c_np),
                r_squared=float(c_r2),
            )
        )

    return HeterogeneousTrendsDiagnostics(
        cht_detected=cht_detected,
        trend_diff_pvalue=float(trend_diff_pvalue) if not np.isnan(trend_diff_pvalue) else np.nan,
        treated_slope=float(t_slope),
        control_slope=float(c_slope),
        slope_difference=float(slope_diff),
        slope_diff_se=float(slope_diff_se) if slope_diff_se < np.inf else np.nan,
        cohort_trends=cohort_trends,
    )


def recommend_transformation(
    data: pd.DataFrame,
    outcome: str = None,
    unit: str = None,
    time: str = None,
    treatment: str = None,
    cohort: Optional[str] = None,
    alpha: float = 0.05,
    # lwdid-py compatible aliases
    y: Optional[str] = None,
    ivar: Optional[str] = None,
    tvar: Optional[str] = None,
    d: Optional[str] = None,
    gvar: Optional[str] = None,
    **kwargs,
) -> TransformationRecommendation:
    """Recommend the optimal transformation method based on diagnostics.

    Runs parallel trends tests with both 'demean' and 'detrend'
    transformations, then selects the most appropriate method:
    - If demean passes: recommend 'demean' (most efficient under PT)
    - If demean fails but detrend passes: recommend 'detrend'
    - If both fail: recommend 'detrendq' with low confidence

    Parameters
    ----------
    data : pd.DataFrame
        Panel data.
    outcome : str
        Name of the outcome variable column. (alias: y)
    unit : str
        Name of the unit identifier column. (alias: ivar)
    time : str
        Name of the time variable column. (alias: tvar)
    treatment : str
        Name of the binary treatment indicator column. (alias: d)
    cohort : str or None, optional
        Name of the cohort variable column. (alias: gvar)
    alpha : float, default 0.05
        Significance level for decision.

    Returns
    -------
    TransformationRecommendation
        Recommendation with rationale and supporting test results.

    Examples
    --------
    >>> rec = recommend_transformation(df, 'y', 'unit', 'time', 'treat')
    >>> print(rec.recommended)
    'demean'
    """
    # Resolve lwdid-py aliases
    outcome = outcome or y
    unit = unit or ivar
    time = time or tvar
    treatment = treatment or d
    cohort = cohort or gvar

    # Validate required params
    if outcome is None:
        raise ValueError("'outcome' (or 'y') parameter is required")
    if unit is None:
        raise ValueError("'unit' (or 'ivar') parameter is required")
    if time is None:
        raise ValueError("'time' (or 'tvar') parameter is required")
    if treatment is None:
        raise ValueError("'treatment' (or 'd') parameter is required")

    # Run parallel trends test with demean
    try:
        pt_demean = test_parallel_trends(
            data,
            outcome,
            unit,
            time,
            treatment,
            cohort=cohort,
            rolling="demean",
            alpha=alpha,
        )
    except (DiagnosticError, InsufficientPrePeriodsError):
        # If we can't even run the test, default to demean with low confidence
        pt_demean = ParallelTrendsTestResult(
            method="placebo",
            test_stat=np.nan,
            pvalue=np.nan,
            decision="inconclusive",
            pre_treatment_effects=[],
            n_pre_periods=0,
            significance_level=alpha,
        )

    # If demean passes, recommend it (most efficient)
    if pt_demean.decision == "pass":
        return TransformationRecommendation(
            recommended="demean",
            confidence="high",
            rationale=(
                "Parallel trends test passes under demeaning "
                f"(p={pt_demean.pvalue:.4f}). Demeaning is the most "
                "efficient transformation when parallel trends holds."
            ),
            parallel_trends_result=pt_demean,
            alternative=None,
        )

    # Demean failed or inconclusive: try detrend
    try:
        pt_detrend = test_parallel_trends(
            data,
            outcome,
            unit,
            time,
            treatment,
            cohort=cohort,
            rolling="detrend",
            alpha=alpha,
        )
    except (DiagnosticError, InsufficientPrePeriodsError):
        pt_detrend = ParallelTrendsTestResult(
            method="placebo",
            test_stat=np.nan,
            pvalue=np.nan,
            decision="inconclusive",
            pre_treatment_effects=[],
            n_pre_periods=0,
            significance_level=alpha,
        )

    # If detrend passes, recommend it
    if pt_detrend.decision == "pass":
        confidence = "high" if pt_demean.decision == "fail" else "medium"
        return TransformationRecommendation(
            recommended="detrend",
            confidence=confidence,
            rationale=(
                "Parallel trends test fails under demeaning "
                f"(p={pt_demean.pvalue:.4f}) but passes under detrending "
                f"(p={pt_detrend.pvalue:.4f}). This suggests "
                "cohort-specific linear trends (CHT) that detrending removes."
            ),
            parallel_trends_result=pt_detrend,
            alternative="demeanq",
        )

    # If detrend is inconclusive
    if pt_detrend.decision == "inconclusive":
        return TransformationRecommendation(
            recommended="detrend",
            confidence="medium",
            rationale=(
                "Parallel trends test is inconclusive for both demeaning and "
                "detrending. Detrending is recommended as the safer choice "
                "since it accommodates cohort-specific linear trends."
            ),
            parallel_trends_result=pt_detrend,
            alternative="detrendq",
        )

    # Both fail: recommend detrendq with low confidence
    return TransformationRecommendation(
        recommended="detrendq",
        confidence="low",
        rationale=(
            "Parallel trends test fails under both demeaning "
            f"(p={pt_demean.pvalue:.4f}) and detrending "
            f"(p={pt_detrend.pvalue:.4f}). Recommending quarterly "
            "detrending as a last resort, but results should be "
            "interpreted with caution."
        ),
        parallel_trends_result=pt_detrend,
        alternative="detrend",
    )


# =============================================================================
# Convenience / Reporting Functions
# =============================================================================


def run_full_diagnostics(
    data: pd.DataFrame,
    outcome: str,
    unit: str,
    time: str,
    treatment: str,
    cohort: Optional[str] = None,
    alpha: float = 0.05,
    verbose: bool = True,
) -> dict:
    """Run the complete diagnostic suite for parallel trends.

    Combines parallel trends testing, heterogeneous trends diagnosis,
    and transformation recommendation into a single report.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data.
    outcome : str
        Outcome variable column name.
    unit : str
        Unit identifier column name.
    time : str
        Time variable column name.
    treatment : str
        Binary treatment indicator column name.
    cohort : str or None, optional
        Cohort variable column name.
    alpha : float, default 0.05
        Significance level.
    verbose : bool, default True
        Whether to print summary to console.

    Returns
    -------
    dict
        Dictionary with keys 'parallel_trends', 'heterogeneous_trends',
        and 'recommendation'.
    """
    results = {}

    # 1. Parallel trends test
    try:
        pt_result = test_parallel_trends(
            data,
            outcome,
            unit,
            time,
            treatment,
            cohort=cohort,
            rolling="demean",
            alpha=alpha,
        )
        results["parallel_trends"] = pt_result
    except (DiagnosticError, InsufficientPrePeriodsError) as e:
        results["parallel_trends"] = None
        if verbose:
            print(f"Parallel trends test skipped: {e}")

    # 2. Heterogeneous trends diagnosis
    try:
        ht_result = diagnose_heterogeneous_trends(
            data,
            outcome,
            unit,
            time,
            treatment,
            cohort=cohort,
            alpha=alpha,
        )
        results["heterogeneous_trends"] = ht_result
    except (DiagnosticError, InsufficientPrePeriodsError) as e:
        results["heterogeneous_trends"] = None
        if verbose:
            print(f"Heterogeneous trends diagnosis skipped: {e}")

    # 3. Transformation recommendation
    try:
        rec = recommend_transformation(
            data,
            outcome,
            unit,
            time,
            treatment,
            cohort=cohort,
            alpha=alpha,
        )
        results["recommendation"] = rec
    except Exception as e:
        results["recommendation"] = None
        if verbose:
            print(f"Recommendation failed: {e}")

    # Print summary
    if verbose:
        if results.get("parallel_trends"):
            print(results["parallel_trends"].summary())
        if results.get("heterogeneous_trends"):
            print(results["heterogeneous_trends"].summary())
        if results.get("recommendation"):
            print(results["recommendation"].summary())

    return results
