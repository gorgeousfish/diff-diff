"""Sensitivity analysis for LWDiD estimator.

Assesses robustness of ATT estimates across different specifications:
- Pre-period selection sensitivity
- No-anticipation assumption sensitivity
- Comprehensive specification grid

Classification thresholds (per Lee & Wooldridge 2025 recommendations):
  sensitivity_ratio < 10%  → 'highly_robust'
  10% ≤ ratio < 25%       → 'moderately_robust'
  25% ≤ ratio < 50%       → 'sensitive'
  ratio ≥ 50%             → 'highly_sensitive'

References
----------
Lee, S. J. & Wooldridge, J. M. (2025). "A Simple Transformation Approach
  to Difference-in-Differences Estimation for Panel Data." SSRN 4516518.
Lee, S. J. & Wooldridge, J. M. (2026). "Simple Approaches to Inference
  with Difference-in-Differences Estimators with Small Cross-Sectional
  Sample Sizes." SSRN 5325686.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from diff_diff.lwdid_exceptions import (
    DiagnosticWarning,
    SensitivityWarning,
)

# =============================================================================
# Constants
# =============================================================================

_ROBUSTNESS_THRESHOLDS = {
    "highly_robust": 0.10,
    "moderately_robust": 0.25,
    "sensitive": 0.50,
}

_VALID_ROLLING = ("demean", "detrend")
_VALID_ESTIMATORS = ("ra", "ipw", "ipwra")


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class SpecificationResult:
    """Result from a single specification in sensitivity analysis.

    Attributes
    ----------
    label : str
        Human-readable label describing this specification.
    rolling : str
        Transformation method used ('demean' or 'detrend').
    estimator : str
        Estimation method used ('ra', 'ipw', 'ipwra').
    n_pre_periods : int
        Number of pre-treatment periods used. -1 if all periods used.
    att : float
        Average treatment effect on the treated.
    se : float
        Standard error of ATT.
    pvalue : float
        Two-sided p-value for testing H0: ATT = 0.
    """

    label: str
    rolling: str
    estimator: str
    n_pre_periods: int
    att: float
    se: float
    pvalue: float

    @property
    def is_significant(self) -> bool:
        """Whether estimate is significant at 5% level."""
        return self.pvalue < 0.05

    def to_dict(self) -> dict:
        """Convert to dictionary for DataFrame construction."""
        return {
            "label": self.label,
            "rolling": self.rolling,
            "estimator": self.estimator,
            "n_pre_periods": self.n_pre_periods,
            "att": self.att,
            "se": self.se,
            "pvalue": self.pvalue,
            "significant_05": self.is_significant,
        }


@dataclass
class SensitivityResult:
    """Result of comprehensive sensitivity analysis.

    Attributes
    ----------
    specifications : List[SpecificationResult]
        Results from each non-baseline specification.
    baseline_att : float
        ATT from the baseline specification.
    baseline_se : float
        Standard error from the baseline specification.
    sensitivity_ratio : float
        (max_att - min_att) / |baseline_att|, measuring estimate instability.
    robustness_level : str
        Categorical assessment: 'highly_robust', 'moderately_robust',
        'sensitive', or 'highly_sensitive'.
    n_specifications : int
        Total number of specifications tested (including baseline).
    """

    specifications: List[SpecificationResult]
    baseline_att: float
    baseline_se: float
    sensitivity_ratio: float
    robustness_level: str
    n_specifications: int

    def summary(self) -> str:
        """Return a formatted summary of sensitivity analysis results.

        Returns
        -------
        str
            Multi-line string summarizing the sensitivity analysis.
        """
        lines = [
            "=" * 60,
            "LWDiD Sensitivity Analysis Summary",
            "=" * 60,
            f"Baseline ATT:       {self.baseline_att:.6f}",
            f"Baseline SE:        {self.baseline_se:.6f}",
            f"Sensitivity Ratio:  {self.sensitivity_ratio:.4f} "
            f"({self.sensitivity_ratio * 100:.1f}%)",
            f"Robustness Level:   {self.robustness_level}",
            f"N Specifications:   {self.n_specifications}",
            "-" * 60,
        ]

        if self.specifications:
            lines.append(f"{'Label':<25} {'ATT':>10} {'SE':>10} {'p-value':>10}")
            lines.append("-" * 60)
            for spec in self.specifications:
                lines.append(
                    f"{spec.label:<25} {spec.att:>10.6f} " f"{spec.se:>10.6f} {spec.pvalue:>10.4f}"
                )
        else:
            lines.append("No alternative specifications computed.")

        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert all specification results to a DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: label, rolling, estimator,
            n_pre_periods, att, se, pvalue, significant_05.
        """
        rows = [
            {
                "label": "baseline",
                "rolling": "",
                "estimator": "",
                "n_pre_periods": -1,
                "att": self.baseline_att,
                "se": self.baseline_se,
                "pvalue": np.nan,
                "significant_05": True,
            }
        ]
        for spec in self.specifications:
            rows.append(spec.to_dict())
        return pd.DataFrame(rows)

    def __repr__(self) -> str:
        return (
            f"SensitivityResult(baseline_att={self.baseline_att:.4f}, "
            f"ratio={self.sensitivity_ratio:.4f}, "
            f"level='{self.robustness_level}', "
            f"n_specs={self.n_specifications})"
        )


# =============================================================================
# Helper Functions
# =============================================================================


def _classify_robustness(ratio: float) -> str:
    """Classify sensitivity ratio into robustness level.

    Parameters
    ----------
    ratio : float
        Sensitivity ratio (range / |baseline|).

    Returns
    -------
    str
        One of 'highly_robust', 'moderately_robust', 'sensitive',
        or 'highly_sensitive'.
    """
    if ratio < _ROBUSTNESS_THRESHOLDS["highly_robust"]:
        return "highly_robust"
    elif ratio < _ROBUSTNESS_THRESHOLDS["moderately_robust"]:
        return "moderately_robust"
    elif ratio < _ROBUSTNESS_THRESHOLDS["sensitive"]:
        return "sensitive"
    else:
        return "highly_sensitive"


def _compute_sensitivity_ratio(baseline_att: float, all_atts: List[float]) -> float:
    """Compute sensitivity ratio from ATT estimates.

    Parameters
    ----------
    baseline_att : float
        Baseline ATT estimate.
    all_atts : list of float
        All ATT estimates including baseline.

    Returns
    -------
    float
        Sensitivity ratio: (max - min) / |baseline|.
    """
    finite_atts = [a for a in all_atts if np.isfinite(a)]
    if len(finite_atts) <= 1:
        return 0.0
    if abs(baseline_att) < 1e-10:
        return 0.0
    return (max(finite_atts) - min(finite_atts)) / abs(baseline_att)


def _fit_single_spec(
    data: pd.DataFrame,
    outcome: str,
    unit: str,
    time: str,
    treatment: str,
    cohort: Optional[str],
    rolling: str,
    estimator: str,
    vce: str,
    cluster: Optional[str],
    controls: Optional[List[str]],
) -> Tuple[float, float, float]:
    """Fit a single LWDiD specification and return (att, se, pvalue).

    Returns (nan, nan, nan) if estimation fails.
    """
    from diff_diff.lwdid import LWDiD

    try:
        est = LWDiD(rolling=rolling, estimator=estimator, vce=vce)
        res = est.fit(
            data,
            outcome=outcome,
            unit=unit,
            time=time,
            treatment=treatment,
            cohort=cohort,
            cluster=cluster,
            controls=controls,
        )
        return res.att, res.se, res.p_value
    except Exception:
        return np.nan, np.nan, np.nan


def _get_pre_periods(data: pd.DataFrame, time: str, treatment: str) -> np.ndarray:
    """Identify pre-treatment periods from the data.

    Parameters
    ----------
    data : pd.DataFrame
        Panel dataset.
    time : str
        Time column name.
    treatment : str
        Treatment indicator column name.

    Returns
    -------
    np.ndarray
        Sorted array of pre-treatment period values.
    """
    all_periods = np.sort(data[time].unique())
    # Post-treatment periods are those where any unit is treated
    post_periods = data.loc[data[treatment] == 1, time].unique()
    pre_periods = np.array([p for p in all_periods if p not in post_periods])
    return np.sort(pre_periods)


# =============================================================================
# Public API: robustness_pre_periods
# =============================================================================


def robustness_pre_periods(
    data: pd.DataFrame,
    outcome: str = None,
    unit: str = None,
    time: str = None,
    treatment: str = None,
    cohort: Optional[str] = None,
    rolling: str = "demean",
    estimator: str = "ra",
    vce: str = "hc1",
    cluster: Optional[str] = None,
    controls: Optional[List[str]] = None,
    k_min: int = 2,
    k_max: Optional[int] = None,
    # lwdid-py compatible aliases
    y: Optional[str] = None,
    ivar: Optional[str] = None,
    tvar: Optional[str] = None,
    d: Optional[str] = None,
    gvar: Optional[str] = None,
    **kwargs,
) -> SensitivityResult:
    """Assess sensitivity of ATT to number of pre-treatment periods used.

    For each k in range(k_min, k_max+1), restricts the data to use only
    the last k pre-treatment periods for rolling transformation, then fits
    LWDiD and collects the ATT estimate.

    Parameters
    ----------
    data : pd.DataFrame
        Panel dataset in long format.
    outcome : str
        Outcome column name. (alias: y)
    unit : str
        Unit identifier column name. (alias: ivar)
    time : str
        Time period column name. (alias: tvar)
    treatment : str
        Binary treatment indicator column name. (alias: d)
    cohort : str, optional
        Cohort variable for staggered designs. (alias: gvar)
    rolling : str, default 'demean'
        Transformation method.
    estimator : str, default 'ra'
        Estimation method.
    vce : str, default 'hc1'
        Variance-covariance estimator.
    cluster : str, optional
        Cluster variable for standard errors.
    controls : list of str, optional
        Control variable column names.
    k_min : int, default 2
        Minimum number of pre-treatment periods to test.
    k_max : int, optional
        Maximum number of pre-treatment periods. If None, uses all available.

    Returns
    -------
    SensitivityResult
        Sensitivity analysis result with per-specification ATT estimates
        and overall robustness classification.
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

    pre_periods = _get_pre_periods(data, time, treatment)
    n_pre = len(pre_periods)

    if k_max is None:
        k_max = n_pre

    k_max = min(k_max, n_pre)
    k_min = max(k_min, 2)

    if k_min > k_max:
        warnings.warn(
            f"k_min ({k_min}) > k_max ({k_max}). "
            "Insufficient pre-treatment periods for robustness analysis.",
            DiagnosticWarning,
            stacklevel=2,
        )
        # Return degenerate result with baseline only
        att, se, pval = _fit_single_spec(
            data,
            outcome,
            unit,
            time,
            treatment,
            cohort,
            rolling,
            estimator,
            vce,
            cluster,
            controls,
        )
        return SensitivityResult(
            specifications=[],
            baseline_att=att,
            baseline_se=se,
            sensitivity_ratio=0.0,
            robustness_level="highly_robust",
            n_specifications=1,
        )

    # Baseline: use all pre-periods
    baseline_att, baseline_se, baseline_pval = _fit_single_spec(
        data,
        outcome,
        unit,
        time,
        treatment,
        cohort,
        rolling,
        estimator,
        vce,
        cluster,
        controls,
    )

    post_periods = np.sort(data.loc[data[treatment] == 1, time].unique())

    specs: List[SpecificationResult] = []

    for k in range(k_min, k_max + 1):
        if k == n_pre:
            # Same as baseline, skip
            continue

        # Keep only the last k pre-periods + all post-periods
        keep_pre = pre_periods[-k:]
        keep_periods = np.concatenate([keep_pre, post_periods])
        subset = data[data[time].isin(keep_periods)].copy()

        att, se, pval = _fit_single_spec(
            subset,
            outcome,
            unit,
            time,
            treatment,
            cohort,
            rolling,
            estimator,
            vce,
            cluster,
            controls,
        )

        specs.append(
            SpecificationResult(
                label=f"k={k}_pre_periods",
                rolling=rolling,
                estimator=estimator,
                n_pre_periods=k,
                att=att,
                se=se,
                pvalue=pval if not np.isnan(pval) else 1.0,
            )
        )

    # Compute sensitivity ratio
    all_atts = [baseline_att] + [s.att for s in specs]
    ratio = _compute_sensitivity_ratio(baseline_att, all_atts)
    level = _classify_robustness(ratio)

    if level in ("sensitive", "highly_sensitive"):
        warnings.warn(
            f"ATT estimates are {level} to pre-period selection "
            f"(ratio={ratio:.3f}). Consider investigating data structure.",
            SensitivityWarning,
            stacklevel=2,
        )

    return SensitivityResult(
        specifications=specs,
        baseline_att=baseline_att,
        baseline_se=baseline_se,
        sensitivity_ratio=ratio,
        robustness_level=level,
        n_specifications=len(specs) + 1,
    )


# =============================================================================
# Public API: sensitivity_no_anticipation
# =============================================================================


def sensitivity_no_anticipation(
    data: pd.DataFrame,
    outcome: str = None,
    unit: str = None,
    time: str = None,
    treatment: str = None,
    cohort: Optional[str] = None,
    exclude_periods: Optional[List[int]] = None,
    rolling: str = "demean",
    estimator: str = "ra",
    vce: str = "hc1",
    cluster: Optional[str] = None,
    controls: Optional[List[str]] = None,
    # lwdid-py compatible aliases
    y: Optional[str] = None,
    ivar: Optional[str] = None,
    tvar: Optional[str] = None,
    d: Optional[str] = None,
    gvar: Optional[str] = None,
    **kwargs,
) -> SensitivityResult:
    """Assess sensitivity to potential anticipation effects.

    For each n_exclude in exclude_periods, drops the last n_exclude
    pre-treatment periods and re-estimates LWDiD. If ATT changes
    substantially when excluding periods just before treatment,
    this suggests anticipation effects may be present.

    Parameters
    ----------
    data : pd.DataFrame
        Panel dataset in long format.
    outcome : str
        Outcome column name. (alias: y)
    unit : str
        Unit identifier column name. (alias: ivar)
    time : str
        Time period column name. (alias: tvar)
    treatment : str
        Binary treatment indicator column name. (alias: d)
    cohort : str, optional
        Cohort variable for staggered designs. (alias: gvar)
    exclude_periods : list of int, optional
        Number of pre-treatment periods to exclude in each test.
        Default is [1, 2, 3].
    rolling : str, default 'demean'
        Transformation method.
    estimator : str, default 'ra'
        Estimation method.
    vce : str, default 'hc1'
        Variance-covariance estimator.
    cluster : str, optional
        Cluster variable for standard errors.
    controls : list of str, optional
        Control variable column names.

    Returns
    -------
    SensitivityResult
        Sensitivity result with per-exclusion ATT estimates and
        overall robustness classification.
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

    if exclude_periods is None:
        exclude_periods = [1, 2, 3]

    pre_periods = _get_pre_periods(data, time, treatment)
    n_pre = len(pre_periods)

    # Baseline: no exclusion
    baseline_att, baseline_se, baseline_pval = _fit_single_spec(
        data,
        outcome,
        unit,
        time,
        treatment,
        cohort,
        rolling,
        estimator,
        vce,
        cluster,
        controls,
    )

    post_periods = np.sort(data.loc[data[treatment] == 1, time].unique())

    specs: List[SpecificationResult] = []

    for n_exclude in exclude_periods:
        if n_exclude >= n_pre:
            warnings.warn(
                f"Cannot exclude {n_exclude} periods with only {n_pre} "
                "pre-treatment periods. Skipping.",
                DiagnosticWarning,
                stacklevel=2,
            )
            continue

        # Exclude the last n_exclude pre-periods
        remaining_pre = pre_periods[:-n_exclude]
        keep_periods = np.concatenate([remaining_pre, post_periods])
        subset = data[data[time].isin(keep_periods)].copy()

        att, se, pval = _fit_single_spec(
            subset,
            outcome,
            unit,
            time,
            treatment,
            cohort,
            rolling,
            estimator,
            vce,
            cluster,
            controls,
        )

        specs.append(
            SpecificationResult(
                label=f"exclude_{n_exclude}_periods",
                rolling=rolling,
                estimator=estimator,
                n_pre_periods=n_pre - n_exclude,
                att=att,
                se=se,
                pvalue=pval if not np.isnan(pval) else 1.0,
            )
        )

    # Compute sensitivity ratio
    all_atts = [baseline_att] + [s.att for s in specs]
    ratio = _compute_sensitivity_ratio(baseline_att, all_atts)
    level = _classify_robustness(ratio)

    if level in ("sensitive", "highly_sensitive"):
        warnings.warn(
            f"ATT estimates are {level} to anticipation exclusions "
            f"(ratio={ratio:.3f}). Potential anticipation effects detected.",
            SensitivityWarning,
            stacklevel=2,
        )

    return SensitivityResult(
        specifications=specs,
        baseline_att=baseline_att,
        baseline_se=baseline_se,
        sensitivity_ratio=ratio,
        robustness_level=level,
        n_specifications=len(specs) + 1,
    )


# =============================================================================
# Public API: sensitivity_analysis (comprehensive)
# =============================================================================


def sensitivity_analysis(
    data: pd.DataFrame,
    outcome: str = None,
    unit: str = None,
    time: str = None,
    treatment: str = None,
    cohort: Optional[str] = None,
    vary_pre_periods: bool = True,
    vary_transformations: bool = True,
    vary_estimators: bool = False,
    rolling: str = "demean",
    estimator: str = "ra",
    vce: str = "hc1",
    cluster: Optional[str] = None,
    controls: Optional[List[str]] = None,
    k_min: int = 2,
    k_max: Optional[int] = None,
    # lwdid-py compatible aliases
    y: Optional[str] = None,
    ivar: Optional[str] = None,
    tvar: Optional[str] = None,
    d: Optional[str] = None,
    gvar: Optional[str] = None,
    **kwargs,
) -> SensitivityResult:
    """Comprehensive sensitivity analysis combining multiple specification axes.

    Builds a specification grid by varying (optionally) the pre-period
    count, transformation method, and estimator. Each specification is
    fitted independently, and the overall sensitivity ratio is computed.

    Parameters
    ----------
    data : pd.DataFrame
        Panel dataset in long format.
    outcome : str
        Outcome column name. (alias: y)
    unit : str
        Unit identifier column name. (alias: ivar)
    time : str
        Time period column name. (alias: tvar)
    treatment : str
        Binary treatment indicator column name. (alias: d)
    cohort : str, optional
        Cohort variable for staggered designs. (alias: gvar)
    vary_pre_periods : bool, default True
        Whether to vary the number of pre-treatment periods.
    vary_transformations : bool, default True
        Whether to vary the rolling transformation method.
    vary_estimators : bool, default False
        Whether to vary the estimation method. Only effective when
        controls are provided.
    rolling : str, default 'demean'
        Baseline transformation method.
    estimator : str, default 'ra'
        Baseline estimation method.
    vce : str, default 'hc1'
        Variance-covariance estimator.
    cluster : str, optional
        Cluster variable for standard errors.
    controls : list of str, optional
        Control variable column names.
    k_min : int, default 2
        Minimum number of pre-treatment periods to test.
    k_max : int, optional
        Maximum number of pre-treatment periods. If None, uses all available.

    Returns
    -------
    SensitivityResult
        Comprehensive sensitivity result with all specification ATTs
        and overall robustness classification.
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

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")

        # ---- Input validation ----
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"data must be a pandas DataFrame, got {type(data).__name__}.")
        if data.empty:
            raise ValueError("data must not be empty.")
        for col_name, col_val in [
            ("outcome", outcome),
            ("unit", unit),
            ("time", time),
            ("treatment", treatment),
        ]:
            if col_val not in data.columns:
                raise ValueError(
                    f"Column '{col_val}' (specified as {col_name}) not found in data. "
                    f"Available columns: {list(data.columns)}"
                )

        # ---- Baseline ----
        baseline_att, baseline_se, baseline_pval = _fit_single_spec(
            data,
            outcome,
            unit,
            time,
            treatment,
            cohort,
            rolling,
            estimator,
            vce,
            cluster,
            controls,
        )

        specs: List[SpecificationResult] = []

        # ---- Vary transformations ----
        if vary_transformations:
            for r in _VALID_ROLLING:
                if r == rolling:
                    continue
                att, se, pval = _fit_single_spec(
                    data,
                    outcome,
                    unit,
                    time,
                    treatment,
                    cohort,
                    r,
                    estimator,
                    vce,
                    cluster,
                    controls,
                )
                specs.append(
                    SpecificationResult(
                        label=f"{r}+{estimator}",
                        rolling=r,
                        estimator=estimator,
                        n_pre_periods=-1,
                        att=att,
                        se=se,
                        pvalue=pval if not np.isnan(pval) else 1.0,
                    )
                )

        # ---- Vary estimators ----
        if vary_estimators and controls is not None:
            for e in _VALID_ESTIMATORS:
                if e == estimator:
                    continue
                att, se, pval = _fit_single_spec(
                    data,
                    outcome,
                    unit,
                    time,
                    treatment,
                    cohort,
                    rolling,
                    e,
                    vce,
                    cluster,
                    controls,
                )
                specs.append(
                    SpecificationResult(
                        label=f"{rolling}+{e}",
                        rolling=rolling,
                        estimator=e,
                        n_pre_periods=-1,
                        att=att,
                        se=se,
                        pvalue=pval if not np.isnan(pval) else 1.0,
                    )
                )

        # ---- Vary pre-periods ----
        if vary_pre_periods:
            pre_periods = _get_pre_periods(data, time, treatment)
            n_pre = len(pre_periods)
            effective_k_max = min(k_max, n_pre) if k_max is not None else n_pre
            effective_k_min = max(k_min, 2)

            if effective_k_min <= effective_k_max:
                post_periods = np.sort(data.loc[data[treatment] == 1, time].unique())

                for k in range(effective_k_min, effective_k_max + 1):
                    if k == n_pre:
                        # Same as baseline, skip
                        continue

                    keep_pre = pre_periods[-k:]
                    keep_periods = np.concatenate([keep_pre, post_periods])
                    subset = data[data[time].isin(keep_periods)].copy()

                    att, se, pval = _fit_single_spec(
                        subset,
                        outcome,
                        unit,
                        time,
                        treatment,
                        cohort,
                        rolling,
                        estimator,
                        vce,
                        cluster,
                        controls,
                    )

                    specs.append(
                        SpecificationResult(
                            label=f"k={k}+{rolling}+{estimator}",
                            rolling=rolling,
                            estimator=estimator,
                            n_pre_periods=k,
                            att=att,
                            se=se,
                            pvalue=pval if not np.isnan(pval) else 1.0,
                        )
                    )

    # ---- Compute sensitivity ratio ----
    all_atts = [baseline_att] + [s.att for s in specs if np.isfinite(s.att)]
    ratio = _compute_sensitivity_ratio(baseline_att, all_atts)
    level = _classify_robustness(ratio)

    if level in ("sensitive", "highly_sensitive"):
        warnings.warn(
            f"ATT estimates are {level} across specifications "
            f"(ratio={ratio:.3f}). Results may not be robust.",
            SensitivityWarning,
            stacklevel=2,
        )

    return SensitivityResult(
        specifications=specs,
        baseline_att=baseline_att,
        baseline_se=baseline_se,
        sensitivity_ratio=ratio,
        robustness_level=level,
        n_specifications=len(specs) + 1,
    )
