"""LWDiD: Lee & Wooldridge (2025, 2026) rolling-transformation DiD.

Converts panel DiD into cross-sectional estimation via unit-specific
rolling transformations of the outcome variable. Supports common timing
and staggered adoption designs with RA, IPW, IPWRA, and PSM estimation.

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
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import linalg as scipy_linalg
from scipy.stats import norm as _scipy_norm

from diff_diff.linalg import solve_logit, solve_ols
from diff_diff.lwdid_results import LWDiDResults
from diff_diff.utils import safe_inference, validate_binary

_VALID_ROLLING = ("demean", "detrend", "demeanq", "detrendq")
_VALID_ESTIMATORS = ("ra", "ipw", "ipwra", "psm")
_VALID_VCE = ("classical", "hc0", "hc1", "hc2", "hc3", "hc4", "cluster")
_VALID_CONTROL_GROUPS = ("never_treated", "not_yet_treated")

# Propensity score trimming bounds for numerical stability
_PS_TRIM_LOWER = 0.01
_PS_TRIM_UPPER = 0.99


class LWDiD:
    """Lee & Wooldridge rolling-transformation DiD estimator.

    Parameters
    ----------
    rolling : {'demean', 'detrend', 'demeanq', 'detrendq'}, default 'demean'
        Unit-specific transformation method.
        'demean': subtract pre-treatment mean
        'detrend': subtract pre-treatment linear trend
        'demeanq': subtract unit-specific seasonal (quarterly) means
        'detrendq': subtract unit-specific linear trend + seasonal effects
    estimator : {'ra', 'ipw', 'ipwra', 'psm'}, default 'ra'
        Treatment effect estimation method.
        'ra': regression adjustment (OLS)
        'ipw': inverse probability weighting
        'ipwra': augmented IPW (doubly robust)
        'psm': propensity score matching (1:1 nearest-neighbor)
    vce : {'classical', 'hc0', 'hc1', 'hc2', 'hc3', 'hc4', 'cluster'}, default 'hc1'
        Variance-covariance estimator.
        'hc0': White (1980) heteroskedasticity-robust (no DOF correction)
        'hc2': leverage-corrected (u_i^2 / (1-h_ii))
        'hc4': Cribari-Neto (2004) (u_i^2 / (1-h_ii)^d_i)
    control_group : {'never_treated', 'not_yet_treated'}, default 'not_yet_treated'
        Control group definition for staggered designs.
    alpha : float, default 0.05
        Significance level for confidence intervals.
    n_bootstrap : int, default 0
        Number of bootstrap replications (0 = analytical inference).
    period_specific : bool, default False
        If True, estimate separate ATT for each post-treatment period
        (common-timing designs only). Ignored for staggered adoption
        designs (when cohort is specified); a UserWarning is emitted.
    trim_threshold : float, default 0.01
        Propensity score trimming threshold. Scores below this value
        or above (1 - trim_threshold) are clipped. Used by IPW/IPWRA/PSM.
    n_neighbors : int, default 1
        Number of nearest neighbors for PSM matching.
    caliper : float or None, default None
        Maximum allowable distance for PSM matches. Unmatched treated
        units (no control within caliper) receive NaN.
    with_replacement : bool, default True
        Whether PSM matching is done with replacement.

    Notes
    -----
    **Parameter mapping from lwdid-py to diff-diff:**

    The standalone ``lwdid-py`` package (``from lwdid import lwdid``) uses a
    functional interface with separate ``d`` (ever-treated indicator) and
    ``post`` (post-period indicator) columns.  In diff-diff, the ``treatment``
    column is the time-varying binary indicator ``D_i * post_t``—i.e., the
    product of the two lwdid-py columns.

    .. code-block:: python

        # lwdid-py (functional API):
        lwdid(data, y='y', d='d', ivar='unit', tvar='time', post='post',
              rolling='demean', estimator='ra', vce=None)

        # Equivalent in diff-diff (class-based API):
        LWDiD(rolling='demean', estimator='ra', vce='classical').fit(
            data, outcome='y', unit='unit', time='time', treatment='treat')
        # where data['treat'] == data['d'] * data['post']

    Parameter correspondence:

    =================  =================  ====================================
    lwdid-py           diff-diff          Notes
    =================  =================  ====================================
    y                  outcome            Outcome column name
    d + post           treatment          Binary D_it (ever-treated × post)
    ivar               unit               Unit identifier
    tvar               time               Time variable
    gvar               cohort             Cohort (first treatment period)
    rolling            rolling            Same values
    estimator          estimator          Same values
    vce=None           vce='classical'    Homoskedastic (OLS)
    vce='hc1'          vce='hc1'          Heteroskedasticity-robust
    vce='cluster'      vce='cluster'      Cluster-robust
    cluster_var        cluster            Cluster variable name
    controls           controls           Covariates
    control_group      control_group      Same values
    =================  =================  ====================================

    **Results mapping:**

    =================  =================  ====================================
    lwdid-py           diff-diff          Notes
    =================  =================  ====================================
    result.att         result.att         ATT point estimate
    result.se_att      result.se          Standard error
    result.t_stat      result.t_stat      t-statistic
    result.pvalue      result.p_value     p-value (note underscore)
    result.ci_lower    result.conf_int[0] CI lower bound
    result.ci_upper    result.conf_int[1] CI upper bound
    result.nobs        result.n_obs       Number of observations
    result.n_treated   result.n_treated   Treated units
    result.n_control   result.n_control   Control units
    result.vce_type    result.vce_type    VCE type
    result.cluster_var result.cluster_name Cluster variable name
    result.n_clusters  result.n_clusters  Number of clusters
    =================  =================  ====================================

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from diff_diff.lwdid import LWDiD
    >>> from diff_diff import generate_staggered_data
    >>> data = generate_staggered_data(n_units=100, n_periods=8, seed=0)
    >>> model = LWDiD(rolling='demean', estimator='ra')
    >>> result = model.fit(data, outcome='outcome', unit='unit',
    ...                    time='period', treatment='treated')
    >>> result.att != 0
    True
    """

    def __init__(
        self,
        rolling: str = "demean",
        estimator: str = "ra",
        vce: str = "hc1",
        control_group: str = "not_yet_treated",
        alpha: float = 0.05,
        n_bootstrap: int = 0,
        period_specific: bool = False,
        bootstrap_seed: Optional[int] = 42,
        # Engineering parameters:
        trim_threshold: float = 0.01,
        n_neighbors: int = 1,
        caliper: Optional[float] = None,
        with_replacement: bool = True,
        n_jobs: int = 1,
    ) -> None:
        # Validate rolling
        if rolling not in _VALID_ROLLING:
            raise ValueError(f"rolling must be one of {_VALID_ROLLING}, got '{rolling}'")
        # Validate estimator
        if estimator not in _VALID_ESTIMATORS:
            raise ValueError(f"estimator must be one of {_VALID_ESTIMATORS}, " f"got '{estimator}'")
        # Validate vce
        if vce not in _VALID_VCE:
            raise ValueError(f"vce must be one of {_VALID_VCE}, got '{vce}'")
        # Validate control_group
        if control_group not in _VALID_CONTROL_GROUPS:
            raise ValueError(
                f"control_group must be one of {_VALID_CONTROL_GROUPS}, " f"got '{control_group}'"
            )
        # Validate alpha
        if not (0 < alpha < 1):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        # Validate n_bootstrap
        if not isinstance(n_bootstrap, (int, np.integer)) or n_bootstrap < 0:
            raise ValueError(f"n_bootstrap must be a non-negative integer, " f"got {n_bootstrap}")

        self.rolling = rolling
        self.estimator = estimator
        self.vce = vce
        self.control_group = control_group
        self.alpha = alpha
        self.n_bootstrap = int(n_bootstrap)
        self.period_specific = period_specific
        self.bootstrap_seed = bootstrap_seed

        # Engineering parameters
        self.trim_threshold = float(trim_threshold)
        if not (0.0 < self.trim_threshold < 0.5):
            raise ValueError("trim_threshold must be between 0 and 0.5")
        self.n_neighbors = int(n_neighbors)
        if self.n_neighbors < 1:
            raise ValueError("n_neighbors must be >= 1")
        self.caliper = float(caliper) if caliper is not None else None
        self.with_replacement = bool(with_replacement)
        if not isinstance(n_jobs, (int, np.integer)) or n_jobs < 1:
            raise ValueError(f"n_jobs must be a positive integer, got {n_jobs}")
        self.n_jobs = int(n_jobs)

    def fit(
        self,
        data: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        treatment: str,
        cohort: Optional[str] = None,
        cluster: Optional[str] = None,
        controls: Optional[List[str]] = None,
        aggregate: Optional[str] = None,
    ) -> LWDiDResults:
        """Fit the LWDiD estimator.

        Parameters
        ----------
        data : pd.DataFrame
            Panel dataset in long format.
        outcome : str
            Column name of the outcome variable.
        unit : str
            Column name of the unit identifier.
        time : str
            Column name of the time period variable.
        treatment : str
            Column name of the binary treatment indicator (0/1).
        cohort : str, optional
            Column name of the cohort (first treatment time) variable.
            If None, assumes common timing (all treated units adopt
            treatment simultaneously).
        cluster : str, optional
            Column name for cluster-robust standard errors.
            Required when vce='cluster'.
        controls : list of str, optional
            Column names for control variables (covariates).
        aggregate : str, optional
            Aggregation method for staggered designs. If "event_study",
            computes per-relative-period WATT(r) estimates with
            Algorithm 1 multiplier bootstrap simultaneous confidence bands.

        Returns
        -------
        LWDiDResults
            Object containing ATT estimates, standard errors, and
            inference results.

        Raises
        ------
        ValueError
            If required columns are missing, treatment is not binary,
            or panel structure is invalid.
        """
        # --- Input validation ---
        df = data.copy()
        self._validate_inputs(df, outcome, unit, time, treatment, cohort, cluster, controls)

        # Validate treatment is binary
        validate_binary(df[treatment].values, treatment)

        # Validate cluster requirement
        if self.vce == "cluster" and cluster is None:
            raise ValueError("cluster column must be specified when vce='cluster'")

        # Normalize controls
        if controls is None:
            controls = []

        # Dispatch to common timing or staggered
        if cohort is None:
            return self._fit_common_timing(df, outcome, unit, time, treatment, cluster, controls)
        elif aggregate == "event_study":
            return self._fit_event_study(df, outcome, unit, time, cohort, cluster, controls)
        else:
            return self._fit_staggered(df, outcome, unit, time, cohort, cluster, controls)

    def get_transformation_diagnostics(
        self,
        data: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        treatment: str,
        cohort: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run the transformation step and return diagnostics without full estimation.

        This is useful for inspecting pre-treatment fit quality before running
        the full estimator.

        Parameters
        ----------
        data : pd.DataFrame
            Panel data.
        outcome : str
            Name of the outcome column.
        unit : str
            Name of the unit identifier column.
        time : str
            Name of the time period column.
        treatment : str
            Name of the treatment indicator column.
        cohort : str or None, default None
            Name of the cohort column (for staggered designs).

        Returns
        -------
        dict
            Transformation diagnostics (see _transform_* docstrings).
        """
        df = data.copy()

        # Determine pre-treatment mask
        if cohort is not None:
            # For staggered: use the earliest cohort's pre-period definition
            cohort_vals = df[cohort].dropna().unique()
            cohort_vals = sorted(cohort_vals)
            # Pre-treatment = before earliest cohort treatment time
            earliest_cohort = cohort_vals[0]
            pre_mask = df[time] < earliest_cohort
        else:
            # Common timing: pre-treatment periods are those where NO unit
            # is treated (same logic as _fit_common_timing)
            time_treatment = df.groupby(time)[treatment].max()
            pre_periods = time_treatment[time_treatment == 0].index.tolist()
            pre_mask = df[time].isin(pre_periods)

        # Dispatch to the appropriate transformation with diagnostics
        if self.rolling == "demean":
            _, diagnostics = self._transform_demean(
                df, outcome, unit, pre_mask, return_diagnostics=True
            )
        elif self.rolling == "detrend":
            _, diagnostics = self._transform_detrend(
                df, outcome, unit, time, pre_mask, return_diagnostics=True
            )
        elif self.rolling == "demeanq":
            _, diagnostics = self._transform_demeanq(
                df, outcome, unit, time, pre_mask, return_diagnostics=True
            )
        elif self.rolling == "detrendq":
            _, diagnostics = self._transform_detrendq(
                df, outcome, unit, time, pre_mask, return_diagnostics=True
            )
        else:
            _, diagnostics = self._transform_detrend(
                df, outcome, unit, time, pre_mask, return_diagnostics=True
            )

        return diagnostics

    def _validate_inputs(
        self,
        df: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        treatment: str,
        cohort: Optional[str],
        cluster: Optional[str],
        controls: Optional[List[str]],
    ) -> None:
        """Validate that all required columns exist and data is valid.

        Parameters
        ----------
        df : pd.DataFrame
            The input dataframe.
        outcome, unit, time, treatment : str
            Required column names.
        cohort, cluster : str or None
            Optional column names.
        controls : list of str or None
            Optional control variable column names.

        Raises
        ------
        ValueError
            If any specified column is not in the dataframe.
        """
        required_cols = [outcome, unit, time, treatment]
        if cohort is not None:
            required_cols.append(cohort)
        if cluster is not None:
            required_cols.append(cluster)
        if controls:
            required_cols.extend(controls)

        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Columns not found in data: {missing}")

        # Check for NaN in key columns
        for col in [outcome, unit, time, treatment]:
            if df[col].isna().any():
                raise ValueError(
                    f"Column '{col}' contains missing values. "
                    f"Please handle missing data before fitting."
                )

        # Check panel structure: each unit-time pair should be unique
        duplicates = df.duplicated(subset=[unit, time], keep=False)
        if duplicates.any():
            n_dup = duplicates.sum()
            raise ValueError(
                f"Panel is not balanced: {n_dup} duplicate "
                f"unit-time observations found. Each (unit, time) "
                f"pair must be unique."
            )

        # Panel balance check
        obs_per_unit = df.groupby(unit)[time].nunique()
        if obs_per_unit.nunique() > 1:
            n_short = (obs_per_unit < obs_per_unit.max()).sum()
            warnings.warn(
                f"Unbalanced panel: {n_short} of {obs_per_unit.shape[0]} units have "
                f"fewer than {obs_per_unit.max()} time periods. LWDiD assumes balanced "
                "panels for optimal performance.",
                UserWarning,
                stacklevel=2,
            )

    def _fit_common_timing(
        self,
        df: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        treatment: str,
        cluster: Optional[str],
        controls: List[str],
    ) -> LWDiDResults:
        """Estimate ATT under common treatment timing.

        All treated units adopt treatment at the same time period.

        Parameters
        ----------
        df : pd.DataFrame
            Panel data.
        outcome : str
            Outcome variable column.
        unit : str
            Unit identifier column.
        time : str
            Time period column.
        treatment : str
            Binary treatment indicator column.
        cluster : str or None
            Cluster variable for cluster-robust SEs.
        controls : list of str
            Control variable columns.

        Returns
        -------
        LWDiDResults
            Estimation results.
        """
        # Validation: treatment must be absorbing (once treated, stays treated)
        unit_treat_seq = df.sort_values(time).groupby(unit)[treatment].apply(list)
        for uid, seq in unit_treat_seq.items():
            saw_one = False
            for v in seq:
                if v == 1:
                    saw_one = True
                elif saw_one and v == 0:
                    raise ValueError(
                        f"Non-absorbing treatment detected for unit '{uid}': "
                        f"treatment switches from 1 to 0. LWDiD requires absorbing treatment."
                    )

        # Step 1: Identify pre/post periods from treatment column
        # Pre-treatment: periods where NO unit is treated
        # Post-treatment: periods where at least one unit is treated
        time_treatment = df.groupby(time)[treatment].max()
        pre_periods = time_treatment[time_treatment == 0].index.tolist()
        post_periods = time_treatment[time_treatment > 0].index.tolist()

        if len(pre_periods) == 0:
            raise ValueError(
                "No pre-treatment periods found. At least one period "
                "with all treatment=0 is required."
            )
        if len(post_periods) == 0:
            raise ValueError(
                "No post-treatment periods found. At least one period "
                "with some treatment=1 is required."
            )

        # Identify treated and control units
        unit_ever_treated = df.groupby(unit)[treatment].max()
        treated_units = unit_ever_treated[unit_ever_treated == 1].index.tolist()
        control_units = unit_ever_treated[unit_ever_treated == 0].index.tolist()
        treated_set = set(treated_units)

        if len(treated_units) == 0:
            raise ValueError("No treated units found in the data.")
        if len(control_units) == 0:
            raise ValueError(
                "No control units found. At least one never-treated " "unit is required."
            )

        # Step 2: Apply transformation
        pre_mask = df[time].isin(pre_periods)

        if self.rolling == "demean":
            df = self._transform_demean(df, outcome, unit, pre_mask)
        elif self.rolling == "detrend":
            df = self._transform_detrend(df, outcome, unit, time, pre_mask)
        elif self.rolling == "demeanq":
            df = self._transform_demeanq(df, outcome, unit, time, pre_mask)
        elif self.rolling == "detrendq":
            df = self._transform_detrendq(df, outcome, unit, time, pre_mask)
        else:
            df = self._transform_detrend(df, outcome, unit, time, pre_mask)

        # Step 3: Take post-treatment cross-section of transformed outcomes
        # Average transformed outcome over post-treatment periods per unit
        post_mask = df[time].isin(post_periods)
        post_df = df.loc[post_mask].copy()

        # Compute unit-level average of transformed outcome in post periods
        unit_post_avg = post_df.groupby(unit)["_ydot"].mean().reset_index()
        unit_post_avg.columns = [unit, "_ydot_avg"]

        # Build cross-sectional dataset
        # Take first observation per unit for controls
        cs_df = df.drop_duplicates(subset=[unit], keep="first")[[unit] + controls].copy()
        # Treatment indicator: 1 if unit is ever-treated
        cs_df["_treat"] = cs_df[unit].isin(treated_set).astype(float)
        if cluster is not None:
            # Get cluster from original data
            if cluster == unit:
                cs_df[cluster] = cs_df[unit]
            else:
                cluster_map = df.drop_duplicates(subset=[unit], keep="first").set_index(unit)[
                    cluster
                ]
                cs_df[cluster] = cs_df[unit].map(cluster_map)

        cs_df = cs_df.merge(unit_post_avg, on=unit, how="inner")

        # After merge, drop units whose transformation produced NaN
        n_before_drop = len(cs_df)
        cs_df = cs_df.dropna(subset=["_ydot_avg"])
        n_dropped = n_before_drop - len(cs_df)
        if n_dropped > 0 and len(cs_df) > 0:
            warnings.warn(
                f"LWDiD: {n_dropped} unit(s) dropped due to NaN transformed outcomes "
                f"(insufficient pre-treatment periods for '{self.rolling}' transformation).",
                UserWarning,
                stacklevel=2,
            )
        if len(cs_df) == 0:
            nan = float("nan")
            warnings.warn(
                f"All units have NaN transformed outcomes for rolling='{self.rolling}'. "
                "Likely insufficient pre-treatment periods. Cannot estimate ATT.",
                UserWarning,
                stacklevel=2,
            )
            return LWDiDResults(
                att=nan,
                se=nan,
                t_stat=nan,
                p_value=nan,
                conf_int=(nan, nan),
                n_obs=0,
                n_treated=0,
                n_control=0,
                rolling=self.rolling,
                estimator=self.estimator,
                vce_type=self.vce,
                alpha=self.alpha,
            )

        # Step 4: Estimate ATT
        y = cs_df["_ydot_avg"].values.astype(np.float64)
        treat = cs_df["_treat"].values.astype(np.float64)
        n_obs = len(y)
        n_treated = int(treat.sum())
        n_control = n_obs - n_treated

        # Guard: if transformation produced all-NaN outcomes, return NaN result
        if np.all(np.isnan(y)):
            warnings.warn(
                f"All transformed outcomes are NaN (likely insufficient "
                f"pre-treatment periods for '{self.rolling}' transformation). "
                f"Cannot estimate ATT.",
                UserWarning,
                stacklevel=2,
            )
            nan = float("nan")
            return LWDiDResults(
                att=nan,
                se=nan,
                t_stat=nan,
                p_value=nan,
                conf_int=(nan, nan),
                n_obs=n_obs,
                n_treated=n_treated,
                n_control=n_control,
                rolling=self.rolling,
                estimator=self.estimator,
                vce_type=self.vce,
                alpha=self.alpha,
            )

        # Build controls matrix
        controls_matrix = None
        if controls:
            controls_matrix = cs_df[controls].values.astype(np.float64)

        # Get cluster ids
        cluster_ids = None
        if cluster is not None and self.vce != 'cluster':
            warnings.warn(
                f"LWDiD: cluster='{cluster}' is ignored because vce='{self.vce}' "
                f"(set vce='cluster' to enable cluster-robust inference).",
                UserWarning,
                stacklevel=2,
            )
        if cluster is not None and self.vce == "cluster":
            cluster_ids = cs_df[cluster].values

        # Estimate
        att, se, coefs, vcov, n_params = self._dispatch_estimator(
            y, treat, controls_matrix, cluster_ids, n_obs
        )

        # Step 5: Compute inference
        # For RA estimator, n_params is K_controls (number of control variables).
        # Paper requires df = N - K - 2 (K = controls, 2 for intercept + treatment).
        # For IPW/IPWRA/PSM, n_params already equals effective parameter count.
        if self.estimator == "ra":
            df_dof = max(n_obs - n_params - 2, 1)
        else:
            df_dof = max(n_obs - n_params, 1)

        # Issue 3: Cluster-robust inference uses df = G-1
        if self.vce == "cluster" and cluster_ids is not None:
            df_dof = max(int(len(np.unique(cluster_ids))) - 1, 1)

        t_stat, p_value, conf_int = safe_inference(att, se, alpha=self.alpha, df=df_dof)

        # Step 5b: Period-specific effects if requested
        period_effects = None
        if self.period_specific and len(post_periods) >= 1:
            period_effects = self._estimate_period_effects(
                df,
                outcome,
                unit,
                time,
                post_periods,
                treated_set,
                controls,
                cluster,
                controls_matrix is not None,
            )

        # Step 6: Bootstrap if requested
        if self.n_bootstrap > 0:
            att, se, t_stat, p_value, conf_int = self._bootstrap(
                df,
                outcome,
                unit,
                time,
                treatment,
                cluster,
                controls,
                pre_periods,
                post_periods,
                treated_units,
                control_units,
            )

        result = LWDiDResults(
            att=att,
            se=se,
            t_stat=t_stat,
            p_value=p_value,
            conf_int=conf_int,
            n_obs=n_obs,
            n_treated=n_treated,
            n_control=n_control,
            rolling=self.rolling,
            estimator=self.estimator,
            vce_type=self.vce,
            alpha=self.alpha,
            cluster_name=cluster if self.vce == "cluster" else None,
            n_clusters=int(len(np.unique(cluster_ids))) if cluster_ids is not None else None,
            cohort_effects=None,
            period_effects=period_effects,
            params=coefs,
            vcov=vcov,
            df_inference=df_dof,
        )

        # Final safety net: warn if result has NaN ATT
        if np.isnan(result.att):
            warnings.warn(
                f"LWDiD estimation returned NaN ATT. This typically indicates "
                f"insufficient data for the '{self.rolling}' transformation or "
                f"numerical issues in estimation. Check your data structure and "
                f"consider using a simpler transformation (e.g., rolling='demean').",
                UserWarning,
                stacklevel=2,
            )

        return result

    def _fit_staggered(
        self,
        df: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        cohort: str,
        cluster: Optional[str],
        controls: List[str],
    ) -> LWDiDResults:
        """Estimate ATT under staggered treatment adoption.

        Treatment timing varies across cohorts. Estimates per-cohort
        effects and aggregates via cohort-size weighting.

        Parameters
        ----------
        df : pd.DataFrame
            Panel data.
        outcome : str
            Outcome variable column.
        unit : str
            Unit identifier column.
        time : str
            Time period column.
        cohort : str
            Cohort (first treatment time) column.
        cluster : str or None
            Cluster variable for cluster-robust SEs.
        controls : list of str
            Control variable columns.

        Returns
        -------
        LWDiDResults
            Estimation results with cohort_effects populated.
        """
        # Validation: cohort must be time-invariant within units
        varying = df.groupby(unit)[cohort].nunique()
        bad = varying[varying > 1]
        if len(bad) > 0:
            raise ValueError(
                f"Cohort must be time-invariant. Found {len(bad)} unit(s) with varying cohort."
            )

        # Warn if period_specific is requested (not supported for staggered)
        if self.period_specific:
            warnings.warn(
                "period_specific=True is not yet supported for staggered designs; "
                "this option will be ignored. Cohort-level effects are available via "
                "cohort_effects.",
                UserWarning,
                stacklevel=2,
            )

        # Step 1: Extract unique cohorts (first treatment times)
        # Cohort == 0 or NaN means never-treated
        unique_cohorts = sorted([g for g in df[cohort].unique() if g > 0 and not np.isnan(g)])

        if len(unique_cohorts) == 0:
            raise ValueError(
                "No treated cohorts found. The cohort column must "
                "contain positive values indicating first treatment time."
            )

        # Identify never-treated units (cohort == 0 or NaN)
        never_treated_mask = (df[cohort] == 0) | df[cohort].isna()
        never_treated_units = df.loc[never_treated_mask, unit].unique().tolist()

        if self.control_group == "never_treated" and len(never_treated_units) == 0:
            raise ValueError(
                "control_group='never_treated' requires at least one "
                "never-treated unit (cohort=0), but none found."
            )

        if self.control_group == "never_treated" and len(never_treated_units) < 2:
            raise ValueError(
                f"control_group='never_treated' requires at least 2 never-treated units "
                f"for valid estimation (LW 2026 p.26). Found {len(never_treated_units)}."
            )

        all_times = sorted(df[time].unique())

        # Step 2: For each cohort g, estimate per-cohort ATT
        cohort_effects: List[Dict[str, Any]] = []
        total_treated = 0

        for g in unique_cohorts:
            # Units in this cohort
            cohort_g_units = df.loc[df[cohort] == g, unit].unique().tolist()
            n_treated_g = len(cohort_g_units)

            # Determine control group for this cohort
            if self.control_group == "never_treated":
                control_units_g = never_treated_units
            else:
                # not_yet_treated: units that have not been treated by
                # time g (never-treated + later cohorts)
                control_units_g = (
                    df.loc[(df[cohort] == 0) | (df[cohort].isna()) | (df[cohort] > g), unit]
                    .unique()
                    .tolist()
                )

            if len(control_units_g) == 0:
                warnings.warn(
                    f"Cohort g={g}: no valid control units found. " f"Skipping this cohort.",
                    UserWarning,
                    stacklevel=2,
                )
                continue

            # Subset data to treated cohort g + control units
            relevant_units = cohort_g_units + control_units_g
            sub_df = df.loc[df[unit].isin(relevant_units)].copy()

            # Identify pre-treatment periods for this cohort
            pre_periods_g = [t for t in all_times if t < g]
            post_periods_g = [t for t in all_times if t >= g]

            if len(pre_periods_g) == 0:
                warnings.warn(
                    f"Cohort g={g}: no pre-treatment periods. " f"Skipping this cohort.",
                    UserWarning,
                    stacklevel=2,
                )
                continue

            if self.rolling == "detrend" and len(pre_periods_g) < 2:
                warnings.warn(
                    f"Cohort g={g}: detrend requires at least 2 "
                    f"pre-treatment periods, found {len(pre_periods_g)}. "
                    f"Skipping this cohort.",
                    UserWarning,
                    stacklevel=2,
                )
                continue

            if self.rolling == "detrendq" and len(pre_periods_g) < 2:
                warnings.warn(
                    f"Cohort g={g}: detrendq requires at least 2 "
                    f"pre-treatment periods, found {len(pre_periods_g)}. "
                    f"Skipping this cohort.",
                    UserWarning,
                    stacklevel=2,
                )
                continue

            # Apply transformation on this subset
            pre_mask_g = sub_df[time].isin(pre_periods_g)

            if self.rolling == "demean":
                sub_df = self._transform_demean(sub_df, outcome, unit, pre_mask_g)
            elif self.rolling == "detrend":
                sub_df = self._transform_detrend(sub_df, outcome, unit, time, pre_mask_g)
            elif self.rolling == "demeanq":
                sub_df = self._transform_demeanq(sub_df, outcome, unit, time, pre_mask_g)
            elif self.rolling == "detrendq":
                sub_df = self._transform_detrendq(sub_df, outcome, unit, time, pre_mask_g)
            else:
                sub_df = self._transform_detrend(sub_df, outcome, unit, time, pre_mask_g)

            # Take post-treatment cross-section
            # For treated cohort g units: keep all t >= g
            # For control units:
            #   - never_treated: keep all t >= g
            #   - not_yet_treated (cohort_i > g): keep only t < cohort_i
            if self.control_group == "not_yet_treated":
                cohort_g_set = set(cohort_g_units)
                post_mask_g = sub_df[time].isin(post_periods_g) & (  # type: ignore[union-attr, call-overload]
                    sub_df[unit].isin(cohort_g_set)  # type: ignore[union-attr, call-overload]
                    | (sub_df[cohort] == 0)  # type: ignore[call-overload]
                    | sub_df[cohort].isna()  # type: ignore[union-attr, call-overload]
                    | (sub_df[time] < sub_df[cohort])  # type: ignore[operator, call-overload]
                )
            else:
                post_mask_g = sub_df[time].isin(post_periods_g)  # type: ignore[union-attr, call-overload]

            post_sub = sub_df.loc[post_mask_g]  # type: ignore[union-attr]

            unit_post_avg_g = post_sub.groupby(unit)["_ydot"].mean().reset_index()
            unit_post_avg_g.columns = [unit, "_ydot_avg"]

            # Build cross-sectional sample
            # Treatment indicator: 1 if unit is in cohort g
            cs_g = sub_df.drop_duplicates(subset=[unit], keep="first")[[unit] + controls].copy()  # type: ignore[union-attr]
            cs_g["_treat_g"] = cs_g[unit].isin(cohort_g_units).astype(float)

            if cluster is not None:
                if cluster == unit:
                    cs_g[cluster] = cs_g[unit]
                else:
                    cluster_map_g = sub_df.drop_duplicates(subset=[unit], keep="first").set_index(  # type: ignore[union-attr]
                        unit
                    )[cluster]
                    cs_g[cluster] = cs_g[unit].map(cluster_map_g)

            cs_g = cs_g.merge(unit_post_avg_g, on=unit, how="inner")

            if cs_g.empty:
                warnings.warn(
                    f"Cohort g={g}: no valid post-treatment observations after "
                    f"control_group='{self.control_group}' filter. Skipping.",
                    UserWarning,
                    stacklevel=2,
                )
                continue

            # Estimate per-cohort ATT
            y_g = cs_g["_ydot_avg"].values.astype(np.float64)
            treat_g = cs_g["_treat_g"].values.astype(np.float64)
            n_obs_g = len(y_g)
            n_control_g = n_obs_g - n_treated_g

            controls_matrix_g = None
            if controls:
                controls_matrix_g = cs_g[controls].values.astype(np.float64)

            cluster_ids_g = None
            if cluster is not None and self.vce == "cluster":
                cluster_ids_g = cs_g[cluster].values

            att_g, se_g, coefs_g, vcov_g, n_params_g = self._dispatch_estimator(
                y_g, treat_g, controls_matrix_g, cluster_ids_g, n_obs_g
            )

            # Skip cohort if estimation failed
            if not np.isfinite(att_g):
                warnings.warn(
                    f"LWDiD: Cohort g={g} skipped — insufficient data or no valid "
                    f"control units for estimation.",
                    UserWarning,
                    stacklevel=2,
                )
                continue

            df_g = max(n_obs_g - n_params_g, 1)
            t_stat_g, p_value_g, conf_int_g = safe_inference(att_g, se_g, alpha=self.alpha, df=df_g)

            cohort_effects.append(
                {
                    "cohort": g,
                    "att": att_g,
                    "se": se_g,
                    "t_stat": t_stat_g,
                    "p_value": p_value_g,
                    "conf_int": conf_int_g,
                    "n_treated": n_treated_g,
                    "n_control": n_control_g,
                    "df": df_g,
                }
            )
            total_treated += n_treated_g

        # Step 3: Aggregate across cohorts (cohort-size weighted average)
        if len(cohort_effects) == 0:
            raise ValueError(
                "No valid cohort estimates could be computed. "
                "Check data structure and pre-treatment period "
                "availability."
            )

        # Use composite outcome regression (LW 2026 Eq 7.18/7.19) for
        # the overall ATT and SE when control_group='never_treated' and
        # estimator='ra' with classical VCE and no controls. This produces
        # the paper's OLS SE. For non-classical VCE, fall back to delta method.
        use_composite = (
            self.control_group == "never_treated"
            and self.estimator == "ra"
            and not controls
            and self.vce == "classical"
        )

        if use_composite:
            att_overall, se_overall, df_overall = self._composite_regression_aggregation(
                df, outcome, unit, time, cohort
            )
            df_overall = max(df_overall, 1)
        else:
            att_overall, se_overall = self._aggregate_cohort_effects(
                cohort_effects, total_treated
            )
            df_overall = max(sum(e["df"] for e in cohort_effects), 1)

        # Step 4: Compute overall inference
        t_stat, p_value, conf_int = safe_inference(
            att_overall, se_overall, alpha=self.alpha, df=df_overall
        )

        # Compute total n
        n_obs_total = sum(e["n_treated"] + e["n_control"] for e in cohort_effects)
        n_treated_total = sum(e["n_treated"] for e in cohort_effects)
        n_control_total = sum(e["n_control"] for e in cohort_effects)

        # Convert list of cohort dicts to dict keyed by cohort value
        cohort_effects_dict = {e["cohort"]: e for e in cohort_effects}

        # Compute cluster metadata for staggered results
        cluster_ids_full = None
        if cluster is not None and self.vce == "cluster":
            cluster_ids_full = df.drop_duplicates(subset=[unit], keep="first")[cluster].values

        result = LWDiDResults(
            att=att_overall,
            se=se_overall,
            t_stat=t_stat,
            p_value=p_value,
            conf_int=conf_int,
            n_obs=n_obs_total,
            n_treated=n_treated_total,
            n_control=n_control_total,
            rolling=self.rolling,
            estimator=self.estimator,
            vce_type=self.vce,
            alpha=self.alpha,
            cluster_name=cluster if self.vce == "cluster" else None,
            n_clusters=(
                int(len(np.unique(cluster_ids_full)))
                if cluster_ids_full is not None and self.vce == "cluster"
                else None
            ),
            cohort_effects=cohort_effects_dict,
            period_effects=None,
            overall_att={
                "att": att_overall,
                "se": se_overall,
                "t_stat": t_stat,
                "p_value": p_value,
                "conf_int": conf_int,
            },
            params=None,
            vcov=None,
            df_inference=df_overall,
        )

        # Final safety net: warn if result has NaN ATT
        if np.isnan(result.att):
            warnings.warn(
                f"LWDiD estimation returned NaN ATT. This typically indicates "
                f"insufficient data for the '{self.rolling}' transformation or "
                f"numerical issues in estimation. Check your data structure and "
                f"consider using a simpler transformation (e.g., rolling='demean').",
                UserWarning,
                stacklevel=2,
            )

        return result

    # ==================================================================
    # Event Study (Appendix D): WATT(r) + Algorithm 1 sup-t bands
    # ==================================================================

    def _fit_event_study(
        self,
        df: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        cohort: str,
        cluster: Optional[str],
        controls: List[str],
    ) -> LWDiDResults:
        """Estimate event-study WATT(r) for each relative period r.

        Implements LW 2025/2026 Appendix D: per-relative-period weighted ATT
        estimates with Algorithm 1 multiplier bootstrap simultaneous bands.
        """
        unique_cohorts = sorted(
            [g for g in df[cohort].unique() if g > 0 and not np.isnan(g)]
        )
        if len(unique_cohorts) == 0:
            raise ValueError("No treated cohorts found.")

        all_times = sorted(df[time].unique())
        t_min, t_max = all_times[0], all_times[-1]

        never_treated_mask = (df[cohort] == 0) | df[cohort].isna()
        never_treated_units = df.loc[never_treated_mask, unit].unique().tolist()

        # Anchor period exclusion
        if self.rolling in ("demean", "demeanq"):
            excluded_anchors = {-1}
        else:
            excluded_anchors = {-1, -2}

        # Precompute cohort info
        cohort_units_map = {}
        cohort_n_map = {}
        for g in unique_cohorts:
            g_units = df.loc[df[cohort] == g, unit].unique().tolist()
            cohort_units_map[g] = g_units
            cohort_n_map[g] = len(g_units)

        # Determine feasible relative periods
        all_relative_periods = set()
        for g in unique_cohorts:
            for t_val in all_times:
                r = int(t_val - g)
                if r in excluded_anchors:
                    continue
                if r < 0:
                    if self.rolling in ("demean", "demeanq") and r > -2:
                        continue
                    elif self.rolling in ("detrend", "detrendq") and r > -3:
                        continue
                all_relative_periods.add(r)

        sorted_r = sorted(all_relative_periods)

        # Unit indexing for influence functions
        all_units = df[unit].unique().tolist()
        unit_to_idx = {u: i for i, u in enumerate(all_units)}
        n_total_units = len(all_units)

        # Precompute control groups and sub-DataFrames per cohort
        # Also precompute transformed outcomes per cohort (all times at once)
        cohort_data_cache = {}  # g -> {t_val: {uid: y_dot}}
        for g in unique_cohorts:
            treated_units_g = cohort_units_map[g]
            pre_periods_g = [t for t in all_times if t < g]
            if len(pre_periods_g) == 0:
                continue
            if self.rolling in ("detrend", "detrendq") and len(pre_periods_g) < 2:
                continue

            # Determine control units for each target time
            # For efficiency, compute the superset (never_treated + all later cohorts)
            if self.control_group == "never_treated":
                control_units_g = never_treated_units
            else:
                # Will filter per time in the loop
                control_units_g = (
                    df.loc[
                        (df[cohort] == 0) | df[cohort].isna() | (df[cohort] > g),
                        unit,
                    ].unique().tolist()
                )

            if len(control_units_g) == 0:
                continue

            relevant_units = list(set(treated_units_g + control_units_g))
            sub_df = df.loc[df[unit].isin(relevant_units)].copy()

            # Precompute post-treatment transform for all post times
            # For demean: pre-mean per unit (one computation for all post times)
            cache_g = {}
            if self.rolling in ("demean", "demeanq"):
                pre_data = sub_df[sub_df[time].isin(pre_periods_g)]
                pre_means = pre_data.groupby(unit)[outcome].mean()
                for t_val in all_times:
                    r = int(t_val - g)
                    if r in excluded_anchors:
                        continue
                    if r >= 0:
                        # Post: Y_t - pre_mean
                        t_data = sub_df[sub_df[time] == t_val].set_index(unit)[outcome]
                        common = t_data.index.intersection(pre_means.index)
                        if len(common) > 0:
                            cache_g[t_val] = dict(zip(common, (t_data[common] - pre_means[common]).values))
                    elif r <= -2:
                        # Pre: Appendix D.1 forward-looking
                        t_data = sub_df[sub_df[time] == t_val].set_index(unit)[outcome]
                        future_data = sub_df[(sub_df[time] > t_val) & (sub_df[time] < g)]
                        if len(future_data) > 0:
                            future_means = future_data.groupby(unit)[outcome].mean()
                            common = t_data.index.intersection(future_means.index)
                            if len(common) > 0:
                                cache_g[t_val] = dict(zip(common, (t_data[common] - future_means[common]).values))
            else:  # detrend
                # Pre-compute per-unit trend coefficients
                pre_data = sub_df[sub_df[time].isin(pre_periods_g)]
                unit_betas = {}  # uid -> (beta0, beta1, t_mean)
                for uid, grp in pre_data.groupby(unit):
                    pre_t = grp[time].to_numpy(dtype=np.float64)
                    pre_y = grp[outcome].to_numpy(dtype=np.float64)
                    if len(pre_t) < 2:
                        continue
                    t_mean = pre_t.mean()
                    X_pre = np.column_stack([np.ones(len(pre_t)), pre_t - t_mean])
                    beta, *_ = np.linalg.lstsq(X_pre, pre_y, rcond=None)
                    unit_betas[uid] = (beta[0], beta[1], t_mean)

                for t_val in all_times:
                    r = int(t_val - g)
                    if r in excluded_anchors:
                        continue
                    if r >= 0:
                        # Post: Y_t - (alpha + beta*(t - t_mean))
                        t_data = sub_df[sub_df[time] == t_val].set_index(unit)[outcome]
                        cell = {}
                        for uid in t_data.index:
                            if uid in unit_betas:
                                b0, b1, tm = unit_betas[uid]
                                y_hat = b0 + b1 * (float(t_val) - tm)
                                cell[uid] = float(t_data[uid]) - y_hat
                        if cell:
                            cache_g[t_val] = cell
                    elif r <= -3:
                        # Pre: Appendix D.2 forward-looking detrend
                        t_data = sub_df[sub_df[time] == t_val].set_index(unit)[outcome]
                        future_data = sub_df[(sub_df[time] > t_val) & (sub_df[time] < g)]
                        if len(future_data) == 0:
                            continue
                        cell = {}
                        for uid, grp in future_data.groupby(unit):
                            if uid not in t_data.index:
                                continue
                            if len(grp) < 2:
                                continue
                            ft = grp[time].to_numpy(dtype=np.float64)
                            fy = grp[outcome].to_numpy(dtype=np.float64)
                            tm = ft.mean()
                            Xf = np.column_stack([np.ones(len(ft)), ft - tm])
                            beta, *_ = np.linalg.lstsq(Xf, fy, rcond=None)
                            y_hat_t = beta[0] + beta[1] * (float(t_val) - tm)
                            cell[uid] = float(t_data[uid]) - y_hat_t
                        if cell:
                            cache_g[t_val] = cell

            cohort_data_cache[g] = cache_g

        # Precompute unit-level controls lookup (time-invariant)
        _unit_controls_df = None
        if controls:
            _unit_controls_df = df.drop_duplicates(subset=[unit], keep="first").set_index(unit)

        # Compute WATT(r) and influence functions
        event_study_effects = {}
        if_matrix = {}  # r -> IF vector of shape (n_total_units,)

        for r in sorted_r:
            cohorts_r = []
            for g in unique_cohorts:
                t_val = g + r
                if t_val < t_min or t_val > t_max:
                    continue
                if r < 0:
                    if self.rolling in ("demean", "demeanq") and r > -2:
                        continue
                    elif self.rolling in ("detrend", "detrendq") and r > -3:
                        continue
                cohorts_r.append(g)

            if not cohorts_r:
                continue

            att_cells = []
            for g in cohorts_r:
                t_val = g + r
                treated_units_g = cohort_units_map[g]
                n_g = cohort_n_map[g]

                # Use precomputed cache
                if g not in cohort_data_cache:
                    continue
                if t_val not in cohort_data_cache[g]:
                    continue
                y_dot_at_t = cohort_data_cache[g][t_val]

                # Filter to not-yet-treated control at time t_val
                if self.control_group != "never_treated":
                    # For not_yet_treated: only keep control units with cohort > t_val
                    valid_controls = set(
                        df.loc[
                            (df[cohort] == 0) | df[cohort].isna() | (df[cohort] > t_val),
                            unit,
                        ].unique()
                    )
                    treated_set_g = set(treated_units_g)
                    y_dot_at_t = {u: v for u, v in y_dot_at_t.items()
                                  if u in treated_set_g or u in valid_controls}

                if len(y_dot_at_t) == 0:
                    continue

                # Build cross-section
                cs_units = list(y_dot_at_t.keys())
                y_vec = np.array([y_dot_at_t[u] for u in cs_units], dtype=np.float64)
                treat_vec = np.array(
                    [1.0 if u in set(treated_units_g) else 0.0 for u in cs_units],
                    dtype=np.float64,
                )

                if treat_vec.sum() == 0 or treat_vec.sum() == len(treat_vec):
                    continue

                valid_mask = np.isfinite(y_vec)
                if valid_mask.sum() < 3:
                    continue
                y_vec = y_vec[valid_mask]
                treat_vec = treat_vec[valid_mask]
                cs_units = [cs_units[i] for i in range(len(valid_mask)) if valid_mask[i]]

                controls_matrix_g = None
                if controls and _unit_controls_df is not None:
                    ctrl_vals = []
                    valid_ctrl_mask = []
                    for u in cs_units:
                        if u in _unit_controls_df.index:
                            row = _unit_controls_df.loc[u, controls]
                            vals = row.values.astype(np.float64) if hasattr(row, 'values') else np.array([float(row)])
                            if np.all(np.isfinite(vals)):
                                ctrl_vals.append(vals)
                                valid_ctrl_mask.append(True)
                            else:
                                valid_ctrl_mask.append(False)
                        else:
                            valid_ctrl_mask.append(False)
                    # Filter out units with missing controls
                    if len(ctrl_vals) < len(cs_units):
                        valid_ctrl_mask = np.array(valid_ctrl_mask)
                        y_vec = y_vec[valid_ctrl_mask]
                        treat_vec = treat_vec[valid_ctrl_mask]
                        cs_units = [cs_units[i] for i in range(len(valid_ctrl_mask)) if valid_ctrl_mask[i]]
                        if len(cs_units) < 3 or treat_vec.sum() == 0 or treat_vec.sum() == len(treat_vec):
                            continue
                    if ctrl_vals:
                        controls_matrix_g = np.array(ctrl_vals)

                att_g_r, se_g_r, coefs_g_r, vcov_g_r, n_params = self._dispatch_estimator(
                    y_vec, treat_vec, controls_matrix_g, None, len(y_vec)
                )

                if not np.isfinite(att_g_r):
                    continue

                # Influence function for ATT coefficient
                n_cs = len(y_vec)
                if controls_matrix_g is not None:
                    X_cs = np.column_stack([np.ones(n_cs), treat_vec, controls_matrix_g])
                else:
                    X_cs = np.column_stack([np.ones(n_cs), treat_vec])

                if coefs_g_r is not None:
                    resid = y_vec - X_cs @ coefs_g_r
                else:
                    resid = y_vec - (np.mean(y_vec[treat_vec == 0]) + att_g_r * treat_vec)

                try:
                    XtX_inv = np.linalg.pinv(X_cs.T @ X_cs / n_cs)
                except np.linalg.LinAlgError:
                    XtX_inv = np.eye(X_cs.shape[1])
                e_treat = np.zeros(X_cs.shape[1])
                e_treat[1] = 1.0
                bread = e_treat @ XtX_inv
                if_per_unit = (X_cs @ bread) * resid / n_cs

                att_cells.append({
                    "att": att_g_r,
                    "n_g": n_g,
                    "if_per_unit": if_per_unit,
                    "cs_units": cs_units,
                })

            if not att_cells:
                continue

            # Aggregate WATT(r)
            total_n_r = sum(c["n_g"] for c in att_cells)
            watt_r = sum(c["att"] * c["n_g"] / total_n_r for c in att_cells)

            # Combine influence functions
            if_combined = np.zeros(n_total_units)
            for cell in att_cells:
                w_g = cell["n_g"] / total_n_r
                for i, u in enumerate(cell["cs_units"]):
                    if u in unit_to_idx:
                        if_combined[unit_to_idx[u]] += w_g * cell["if_per_unit"][i]

            # SE from IF
            se_r = float(np.sqrt(np.sum(if_combined**2)))
            if se_r <= 0 or not np.isfinite(se_r):
                se_r = np.nan

            # Pointwise inference
            if np.isfinite(se_r) and se_r > 0:
                t_stat_r = watt_r / se_r
                p_value_r = float(2 * (1 - _scipy_norm.cdf(abs(t_stat_r))))
                z_crit = _scipy_norm.ppf(1 - self.alpha / 2)
                ci_r = (watt_r - z_crit * se_r, watt_r + z_crit * se_r)
            else:
                t_stat_r = np.nan
                p_value_r = np.nan
                ci_r = (np.nan, np.nan)

            event_study_effects[r] = {
                "effect": watt_r,
                "se": se_r,
                "t_stat": t_stat_r,
                "p_value": p_value_r,
                "conf_int": ci_r,
            }
            if_matrix[r] = if_combined

        # Algorithm 1: Multiplier bootstrap sup-t bands
        n_bootstrap = self.n_bootstrap
        cband_method = None
        cband_crit_value = None
        cband_n_bootstrap = None

        if n_bootstrap > 0 and len(if_matrix) > 1:
            rng = np.random.default_rng(self.bootstrap_seed)
            valid_r = [r for r in sorted(if_matrix.keys())
                       if r in event_study_effects
                       and np.isfinite(event_study_effects[r]["se"])
                       and event_study_effects[r]["se"] > 0]

            if len(valid_r) > 0:
                if_stack = np.column_stack([if_matrix[r] for r in valid_r])
                se_vec = np.array([event_study_effects[r]["se"] for r in valid_r])

                # Bootstrap replications for SE and sup-t
                boot_deltas = np.empty((n_bootstrap, len(valid_r)))
                sup_t_stats = np.empty(n_bootstrap)
                for b in range(n_bootstrap):
                    eps = rng.choice([-1.0, 1.0], size=n_total_units)
                    delta_b = eps @ if_stack
                    boot_deltas[b] = delta_b
                    t_b = np.abs(delta_b) / se_vec
                    sup_t_stats[b] = np.max(t_b)

                # Bootstrap SE (replace analytic SE)
                boot_se = np.std(boot_deltas, axis=0, ddof=1)

                # sup-t critical value
                # Recompute sup-t using bootstrap SEs
                se_vec_boot = boot_se.copy()
                se_vec_boot[se_vec_boot <= 0] = np.inf
                sup_t_stats_2 = np.empty(n_bootstrap)
                for b in range(n_bootstrap):
                    t_b2 = np.abs(boot_deltas[b]) / se_vec_boot
                    sup_t_stats_2[b] = np.max(t_b2)

                cband_crit_value = float(np.quantile(sup_t_stats_2, 1 - self.alpha))
                cband_method = "multiplier_bootstrap_sup_t"
                cband_n_bootstrap = n_bootstrap

                # Update SEs and CIs with bootstrap values
                for i_r, r in enumerate(valid_r):
                    bse = float(boot_se[i_r])
                    if bse > 0:
                        eff_val = event_study_effects[r]["effect"]
                        event_study_effects[r]["se"] = bse
                        event_study_effects[r]["t_stat"] = eff_val / bse
                        event_study_effects[r]["p_value"] = float(
                            2 * (1 - _scipy_norm.cdf(abs(eff_val / bse)))
                        )
                        z_crit = _scipy_norm.ppf(1 - self.alpha / 2)
                        event_study_effects[r]["conf_int"] = (
                            eff_val - z_crit * bse,
                            eff_val + z_crit * bse,
                        )
                        event_study_effects[r]["cband_conf_int"] = (
                            eff_val - cband_crit_value * bse,
                            eff_val + cband_crit_value * bse,
                        )

        # Overall ATT (simple average of post-treatment WATT(r))
        post_effects = {r: e for r, e in event_study_effects.items() if r >= 0}
        if post_effects:
            att_overall = float(np.mean([e["effect"] for e in post_effects.values()]))
        else:
            att_overall = np.nan
        se_overall = np.nan
        t_stat_ov, p_value_ov, conf_int_ov = np.nan, np.nan, (np.nan, np.nan)

        n_obs_total = len(all_units)
        n_treated_total = sum(cohort_n_map.values())
        n_control_total = len(never_treated_units)

        result = LWDiDResults(
            att=att_overall,
            se=se_overall,
            t_stat=t_stat_ov,
            p_value=p_value_ov,
            conf_int=conf_int_ov,
            n_obs=n_obs_total,
            n_treated=n_treated_total,
            n_control=n_control_total,
            rolling=self.rolling,
            estimator=self.estimator,
            vce_type=self.vce,
            alpha=self.alpha,
            event_study_effects=event_study_effects,
            cband_method=cband_method,
            cband_crit_value=cband_crit_value,
            cband_n_bootstrap=cband_n_bootstrap,
        )
        return result

    def _es_transform_post(self, sub_df, outcome, unit, time, pre_periods, target_time):
        """Standard rolling transformation evaluated at a specific post-treatment time."""
        pre_set = set(pre_periods)
        # Get outcome at target time for each unit
        target_data = sub_df[sub_df[time] == target_time].set_index(unit)[outcome]
        # Get pre-period data
        pre_data = sub_df[sub_df[time].isin(pre_set)]

        if self.rolling in ("demean", "demeanq"):
            pre_means = pre_data.groupby(unit)[outcome].mean()
            # Only keep units with both target and pre data
            common = target_data.index.intersection(pre_means.index)
            return dict(zip(common, (target_data[common] - pre_means[common]).values))
        else:  # detrend, detrendq
            result = {}
            pre_grouped = pre_data.groupby(unit)
            for uid, grp in pre_grouped:
                if uid not in target_data.index:
                    continue
                pre_t = grp[time].to_numpy(dtype=np.float64)
                pre_y = grp[outcome].to_numpy(dtype=np.float64)
                if len(pre_t) < 2:
                    continue
                t_mean = pre_t.mean()
                X_pre = np.column_stack([np.ones(len(pre_t)), pre_t - t_mean])
                beta, *_ = np.linalg.lstsq(X_pre, pre_y, rcond=None)
                y_hat = beta[0] + beta[1] * (float(target_time) - t_mean)
                result[uid] = float(target_data[uid]) - y_hat
            return result

    def _es_transform_pre(self, sub_df, outcome, unit, time, cohort_g, target_time):
        """Appendix D forward-looking transformation for pre-treatment periods.

        D.1 (demean): Y_dot = Y_t - mean(Y_q for q in {t+1, ..., g-1})
        D.2 (detrend): Y_dot = Y_t - fitted(Y on q for q in {t+1, ..., g-1})
        """
        # Target time outcome
        target_data = sub_df[sub_df[time] == target_time].set_index(unit)[outcome]
        # Future pre-treatment periods: q in (target_time, cohort_g)
        future_data = sub_df[(sub_df[time] > target_time) & (sub_df[time] < cohort_g)]

        if self.rolling in ("demean", "demeanq"):
            future_means = future_data.groupby(unit)[outcome].mean()
            common = target_data.index.intersection(future_means.index)
            if len(common) == 0:
                return {}
            return dict(zip(common, (target_data[common] - future_means[common]).values))
        else:  # detrend, detrendq
            result = {}
            future_grouped = future_data.groupby(unit)
            for uid, grp in future_grouped:
                if uid not in target_data.index:
                    continue
                if len(grp) < 2:
                    continue
                future_t = grp[time].to_numpy(dtype=np.float64)
                future_y = grp[outcome].to_numpy(dtype=np.float64)
                t_mean = future_t.mean()
                X_f = np.column_stack([np.ones(len(future_t)), future_t - t_mean])
                beta, *_ = np.linalg.lstsq(X_f, future_y, rcond=None)
                y_hat_t = beta[0] + beta[1] * (float(target_time) - t_mean)
                result[uid] = float(target_data[uid]) - y_hat_t
            return result

    def _aggregate_cohort_effects(
        self,
        cohort_effects: List[Dict[str, Any]],
        total_treated: int,
    ) -> Tuple[float, float]:
        """Aggregate per-cohort ATTs via cohort-size weighting (delta method).

        Parameters
        ----------
        cohort_effects : list of dict
            Per-cohort estimation results.
        total_treated : int
            Total number of treated units across all cohorts.

        Returns
        -------
        att : float
            Weighted average ATT.
        se : float
            Standard error of the weighted average.
        """
        if total_treated == 0:
            warnings.warn(
                "Staggered aggregation: total treated count is 0. "
                "Cannot compute weighted ATT. Returning NaN.",
                UserWarning,
                stacklevel=2,
            )
            return np.nan, np.nan

        # Cohort-size weights
        weights = np.array([e["n_treated"] / total_treated for e in cohort_effects])
        atts = np.array([e["att"] for e in cohort_effects])
        ses = np.array([e["se"] for e in cohort_effects])

        # Weighted average ATT
        att = float(np.sum(weights * atts))

        # SE via delta method (assuming independence across cohorts)
        # Var(weighted_avg) = sum(w_g^2 * se_g^2)
        valid_ses = np.isfinite(ses) & (ses > 0)
        if valid_ses.all():
            var_att = float(np.sum(weights**2 * ses**2))
            se = float(np.sqrt(var_att))
        else:
            se = np.nan

        return att, se

    def _composite_regression_aggregation(
        self,
        df: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        cohort: str,
    ) -> Tuple[float, float, int]:
        """Compute tau_omega via composite outcome regression (LW 2026 Eq 7.18/7.19).

        For staggered designs, constructs a composite outcome vector:
        - Treated units in cohort g: use their cohort's transformed outcome
        - Never-treated units: weighted average of all cohort transformations
        Then runs a single cross-sectional OLS: y_composite ~ [1, D_ever_treated]

        Parameters
        ----------
        df : pd.DataFrame
            Full panel data.
        outcome : str
            Outcome variable column.
        unit : str
            Unit identifier column.
        time : str
            Time period column.
        cohort : str
            Cohort (first treatment time) column.

        Returns
        -------
        att : float
            ATT from composite regression coefficient on D.
        se : float
            Classical OLS SE from composite regression.
        dof : int
            Degrees of freedom (n_units - 2).
        """
        # Step 1: Identify cohorts and unit membership
        fy = df.groupby(unit)[cohort].first()
        cohorts = sorted([g for g in fy.unique() if g > 0 and not np.isnan(g)])
        n_treat = int((fy > 0).sum())

        if n_treat == 0:
            return np.nan, np.nan, 0

        # Step 2: For each cohort g, compute per-unit post-average transformed outcome
        # using cohort g's pre-period for ALL units
        ydot_by_cohort: Dict[Any, pd.Series] = {}
        for g in cohorts:
            # pre_mask: periods < g (i.e., time <= g-1)
            pre_mask_g = df[time] < g
            post_mask_g = df[time] >= g

            # Apply transformation to full dataset
            if self.rolling in ("demean", "demeanq"):
                df_transformed = self._transform_demean(df, outcome, unit, pre_mask_g)
            elif self.rolling in ("detrend", "detrendq"):
                df_transformed = self._transform_detrend(df, outcome, unit, time, pre_mask_g)
            else:
                df_transformed = self._transform_demean(df, outcome, unit, pre_mask_g)

            # Per-unit average of transformed outcome in post-periods (>= g)
            post_data = df_transformed.loc[post_mask_g]  # type: ignore[union-attr]
            unit_avg_g = post_data.groupby(unit)["_ydot"].mean()
            ydot_by_cohort[g] = unit_avg_g

        # Step 3: Assemble composite outcome vector
        all_units = fy.index
        n_units = len(all_units)
        y_composite = np.empty(n_units, dtype=np.float64)
        d_ever_treated = np.empty(n_units, dtype=np.float64)

        # Compute cohort sizes for weights
        cohort_sizes = {g: int((fy == g).sum()) for g in cohorts}

        for i, u in enumerate(all_units):
            g_u = fy[u]
            if g_u > 0:  # Treated unit
                y_composite[i] = ydot_by_cohort[g_u].get(u, np.nan)
                d_ever_treated[i] = 1.0
            else:  # Never-treated (control) unit
                weighted_sum = 0.0
                for g in cohorts:
                    w_g = cohort_sizes[g] / n_treat
                    weighted_sum += w_g * ydot_by_cohort[g].get(u, 0.0)
                y_composite[i] = weighted_sum
                d_ever_treated[i] = 0.0

        # Step 4: Single OLS regression y_composite ~ [1, D]
        # Drop any NaN observations
        valid = np.isfinite(y_composite)
        y_valid = y_composite[valid]
        d_valid = d_ever_treated[valid]
        n = len(y_valid)

        if n < 3:
            return np.nan, np.nan, 0

        X = np.column_stack([np.ones(n, dtype=np.float64), d_valid])
        beta, *_ = np.linalg.lstsq(X, y_valid, rcond=None)
        resid = y_valid - X @ beta
        k = 2
        dof = n - k
        sigma2 = float(resid @ resid) / dof
        XtX_inv = np.linalg.inv(X.T @ X)
        cov = sigma2 * XtX_inv

        att = float(beta[1])
        se = float(np.sqrt(cov[1, 1]))

        return att, se, dof

    def _transform_demean(
        self,
        df: pd.DataFrame,
        outcome_col: str,
        unit_col: str,
        pre_mask: Union[pd.Series, np.ndarray],
        return_diagnostics: bool = False,
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, Any]]]:
        """Apply unit-specific demeaning transformation.

        For each unit, compute the mean of the outcome in pre-treatment
        periods, then subtract that mean from ALL periods (pre and post).

        Parameters
        ----------
        df : pd.DataFrame
            Panel data.
        outcome_col : str
            Name of the outcome column.
        unit_col : str
            Name of the unit identifier column.
        pre_mask : Series or ndarray of bool
            Boolean mask indicating pre-treatment observations.
        return_diagnostics : bool, default False
            If True, return (df, diagnostics) tuple instead of just df.

        Returns
        -------
        pd.DataFrame or (pd.DataFrame, dict)
            Input data with '_ydot' column containing demeaned outcomes.
            If return_diagnostics=True, also returns diagnostics dict.
        """
        df = df.copy()

        # Compute pre-treatment mean for each unit
        pre_df = df.loc[pre_mask, [unit_col, outcome_col]]
        pre_means = pre_df.groupby(unit_col)[outcome_col].mean()

        # Collect per-unit diagnostics if requested
        per_unit: Dict[Any, Dict[str, Any]] = {}
        if return_diagnostics:
            pre_stds = pre_df.groupby(unit_col)[outcome_col].std()
            pre_counts = pre_df.groupby(unit_col)[outcome_col].count()
            post_mask_inv = ~pre_mask
            post_df = df.loc[post_mask_inv, [unit_col, outcome_col]]
            post_counts = post_df.groupby(unit_col)[outcome_col].count()
            all_units = df[unit_col].unique()
            for uid in all_units:
                has_pre = uid in pre_means.index
                info: Dict[str, Any] = {
                    "pre_mean": float(pre_means[uid]) if has_pre else float("nan"),
                    "pre_n_periods": int(pre_counts.get(uid, 0)),
                    "pre_std": float(pre_stds.get(uid, float("nan"))),
                    "post_n_periods": int(post_counts.get(uid, 0)),
                    "valid": has_pre,
                }
                per_unit[uid] = info

        # Map pre-means back to all observations
        unit_means = df[unit_col].map(pre_means)

        # Check for units with no pre-treatment obs (shouldn't happen
        # after validation, but guard defensively)
        no_pre = unit_means.isna()
        if no_pre.any():
            n_missing = df.loc[no_pre, unit_col].nunique()
            warnings.warn(
                f"{n_missing} unit(s) have no pre-treatment observations. "
                f"Their transformed outcomes will be NaN.",
                UserWarning,
                stacklevel=2,
            )

        # Subtract pre-treatment mean from all periods
        df["_ydot"] = df[outcome_col].values - unit_means.values

        if return_diagnostics:
            valid_units = [uid for uid, info in per_unit.items() if info["valid"]]
            n_valid = len(valid_units)
            n_total = len(per_unit)
            pre_period_counts = [per_unit[uid]["pre_n_periods"] for uid in valid_units]
            diagnostics: Dict[str, Any] = {
                "method": "demean",
                "description": "\u0232_{i,pre} subtracted from all periods (Procedure 2.1, Eq 2.12)",
                "per_unit": per_unit,
                "summary": {
                    "n_units_total": n_total,
                    "n_units_valid": n_valid,
                    "n_units_dropped": n_total - n_valid,
                    "mean_pre_periods": (
                        float(np.mean(pre_period_counts)) if pre_period_counts else 0.0
                    ),
                    "min_pre_periods": int(np.min(pre_period_counts)) if pre_period_counts else 0,
                    "max_pre_periods": int(np.max(pre_period_counts)) if pre_period_counts else 0,
                },
            }
            return df, diagnostics

        return df

    def _transform_detrend(
        self,
        df: pd.DataFrame,
        outcome_col: str,
        unit_col: str,
        time_col: str,
        pre_mask: Union[pd.Series, np.ndarray],
        return_diagnostics: bool = False,
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, Any]]]:
        """Apply unit-specific linear detrending transformation.

        For each unit, fit y = alpha + beta*t on pre-treatment periods
        using scipy.linalg.lstsq, then subtract the fitted trend from
        ALL periods.

        Parameters
        ----------
        df : pd.DataFrame
            Panel data.
        outcome_col : str
            Name of the outcome column.
        unit_col : str
            Name of the unit identifier column.
        time_col : str
            Name of the time period column.
        pre_mask : Series or ndarray of bool
            Boolean mask indicating pre-treatment observations.
        return_diagnostics : bool, default False
            If True, return (df, diagnostics) tuple instead of just df.

        Returns
        -------
        pd.DataFrame or (pd.DataFrame, dict)
            Input data with '_ydot' column containing detrended outcomes.
            If return_diagnostics=True, also returns diagnostics dict.
        """
        df = df.copy()
        df["_ydot"] = np.nan

        # Pre-extract numpy arrays to avoid repeated df.loc[] overhead
        unit_arr = df[unit_col].values
        time_arr = df[time_col].values.astype(np.float64)
        y_arr = df[outcome_col].values.astype(np.float64)
        pre_arr = pre_mask.values if hasattr(pre_mask, "values") else np.asarray(pre_mask)

        units = df[unit_col].unique()
        per_unit: Dict[Any, Dict[str, Any]] = {}
        ydot_out = np.full(len(df), np.nan)

        for uid in units:
            mask_u = unit_arr == uid
            idx_u = np.where(mask_u)[0]
            t_u = time_arr[idx_u]
            y_u = y_arr[idx_u]
            pre_u = pre_arr[idx_u]

            # Pre-treatment data for this unit
            pre_sel = pre_u.astype(bool)
            n_pre = int(pre_sel.sum())

            if n_pre < 2:
                warnings.warn(
                    f"Unit {uid}: detrend requires at least 2 "
                    f"pre-treatment periods, found {n_pre}. "
                    f"Transformed outcome set to NaN.",
                    UserWarning,
                    stacklevel=2,
                )
                if return_diagnostics:
                    per_unit[uid] = {
                        "alpha": float("nan"),
                        "beta": float("nan"),
                        "pre_n_periods": n_pre,
                        "residual_std": float("nan"),
                        "r_squared": float("nan"),
                        "valid": False,
                    }
                continue

            # Extract pre-treatment time and outcome
            t_pre = t_u[pre_sel]
            y_pre = y_u[pre_sel]

            # Center time for numerical stability
            t_mean = t_pre.mean()
            t_pre_centered = t_pre - t_mean

            # Build design matrix [intercept, centered_time]
            X_pre = np.column_stack(
                [
                    np.ones(n_pre, dtype=np.float64),
                    t_pre_centered,
                ]
            )

            # Solve via scipy.linalg.lstsq
            result = scipy_linalg.lstsq(X_pre, y_pre, cond=None)
            coefs = result[0]  # [alpha, beta]

            # Check for valid coefficients
            if not np.all(np.isfinite(coefs)):
                warnings.warn(
                    f"Unit {uid}: detrending produced non-finite "
                    f"coefficients. Transformed outcome set to NaN.",
                    UserWarning,
                    stacklevel=2,
                )
                if return_diagnostics:
                    per_unit[uid] = {
                        "alpha": float("nan"),
                        "beta": float("nan"),
                        "pre_n_periods": n_pre,
                        "residual_std": float("nan"),
                        "r_squared": float("nan"),
                        "valid": False,
                    }
                continue

            # Predict on ALL periods for this unit (using same centering)
            t_all_centered = t_u - t_mean
            y_hat = coefs[0] + coefs[1] * t_all_centered

            # Residuals = outcome - fitted trend
            ydot_out[idx_u] = y_u - y_hat

            # Collect diagnostics for this unit
            if return_diagnostics:
                y_hat_pre = X_pre @ coefs
                residuals_pre = y_pre - y_hat_pre
                residual_std = float(np.std(residuals_pre, ddof=2)) if n_pre > 2 else float("nan")
                ss_res = float(np.sum(residuals_pre**2))
                ss_tot = float(np.sum((y_pre - y_pre.mean()) ** 2))
                r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
                per_unit[uid] = {
                    "alpha": float(coefs[0]),
                    "beta": float(coefs[1]),
                    "pre_n_periods": n_pre,
                    "residual_std": residual_std,
                    "r_squared": r_squared,
                    "valid": True,
                }

        df["_ydot"] = ydot_out

        if return_diagnostics:
            valid_units = [uid for uid, info in per_unit.items() if info["valid"]]
            n_valid = len(valid_units)
            n_total = len(per_unit)
            betas = [per_unit[uid]["beta"] for uid in valid_units]
            r2s = [
                per_unit[uid]["r_squared"]
                for uid in valid_units
                if np.isfinite(per_unit[uid]["r_squared"])
            ]
            diagnostics: Dict[str, Any] = {
                "method": "detrend",
                "description": "Y_{it} - (\u03b1\u0302_i + \u03b2\u0302_i * t) based on pre-treatment OLS (Procedure 3.1)",
                "per_unit": per_unit,
                "summary": {
                    "n_units_total": n_total,
                    "n_units_valid": n_valid,
                    "n_units_dropped": n_total - n_valid,
                    "mean_beta": float(np.mean(betas)) if betas else float("nan"),
                    "std_beta": float(np.std(betas)) if betas else float("nan"),
                    "mean_r_squared": float(np.mean(r2s)) if r2s else float("nan"),
                },
            }
            return df, diagnostics

        return df

    def _transform_demeanq(
        self,
        df: pd.DataFrame,
        outcome_col: str,
        unit_col: str,
        time_col: str,
        pre_mask: Union[pd.Series, np.ndarray],
        return_diagnostics: bool = False,
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, Any]]]:
        """Apply unit-specific seasonal (quarterly) demeaning transformation.

        For each unit, fit Y on [1, Q2, Q3, Q4] dummies using pre-treatment
        periods only, then subtract fitted values from ALL periods.
        Quarter is determined by time_col % 4.

        Parameters
        ----------
        df : pd.DataFrame
            Panel data.
        outcome_col : str
            Name of the outcome column.
        unit_col : str
            Name of the unit identifier column.
        time_col : str
            Name of the time period column.
        pre_mask : Series or ndarray of bool
            Boolean mask indicating pre-treatment observations.
        return_diagnostics : bool, default False
            If True, return (df, diagnostics) tuple instead of just df.

        Returns
        -------
        pd.DataFrame or (pd.DataFrame, dict)
            Input data with '_ydot' column containing seasonally-demeaned outcomes.
            If return_diagnostics=True, also returns diagnostics dict.
        """
        df = df.copy()
        df["_ydot"] = np.nan

        # Determine quarter from time column (0-indexed modulo 4 → 1-4)
        t_series = df[time_col]
        if pd.api.types.is_datetime64_any_dtype(t_series):
            quarters = t_series.dt.quarter.to_numpy()
        elif hasattr(t_series.iloc[0], "quarter"):
            quarters = np.array([v.quarter for v in t_series])
        else:
            t_vals = t_series.to_numpy()
            quarters = (t_vals.astype(np.int64) - 1) % 4 + 1

        # Pre-extract numpy arrays to avoid repeated df.loc[] overhead
        unit_arr = df[unit_col].values
        y_arr = df[outcome_col].values.astype(np.float64)
        pre_arr = pre_mask.values if hasattr(pre_mask, "values") else np.asarray(pre_mask)

        units = df[unit_col].unique()
        per_unit: Dict[Any, Dict[str, Any]] = {}
        ydot_out = np.full(len(df), np.nan)

        for uid in units:
            mask_u = unit_arr == uid
            idx_u = np.where(mask_u)[0]
            y_u = y_arr[idx_u]
            q_u = quarters[idx_u]
            pre_u = pre_arr[idx_u].astype(bool)

            # Pre-treatment data for this unit
            n_pre = int(pre_u.sum())

            # Need at least as many pre-obs as parameters (intercept + up to 3 dummies)
            q_pre = q_u[pre_u]
            observed_seasons = sorted(np.unique(q_pre))
            n_params = len(observed_seasons)  # intercept + (n_seasons - 1) dummies

            if n_pre < n_params:
                warnings.warn(
                    f"Unit {uid}: demeanq requires at least as many pre-treatment "
                    f"observations as seasonal parameters ({n_params}), "
                    f"found {n_pre}. Transformed outcome set to NaN.",
                    UserWarning,
                    stacklevel=2,
                )
                if return_diagnostics:
                    per_unit[uid] = {
                        "intercept": float("nan"),
                        "seasonal_effects": {},
                        "pre_n_periods": n_pre,
                        "valid": False,
                    }
                continue

            # Build seasonal dummy design matrix for pre-treatment
            y_pre = y_u[pre_u]

            # Create dummies: drop first category (reference)
            X_pre_parts = [np.ones(n_pre, dtype=np.float64)]
            for s in observed_seasons[1:]:
                X_pre_parts.append((q_pre == s).astype(np.float64))
            X_pre = np.column_stack(X_pre_parts)

            # Solve via scipy.linalg.lstsq
            result = scipy_linalg.lstsq(X_pre, y_pre, cond=None)
            coefs = result[0]

            if not np.all(np.isfinite(coefs)):
                warnings.warn(
                    f"Unit {uid}: demeanq produced non-finite "
                    f"coefficients. Transformed outcome set to NaN.",
                    UserWarning,
                    stacklevel=2,
                )
                if return_diagnostics:
                    per_unit[uid] = {
                        "intercept": float("nan"),
                        "seasonal_effects": {},
                        "pre_n_periods": n_pre,
                        "valid": False,
                    }
                continue

            # Predict on ALL periods for this unit
            n_all = len(q_u)
            X_all_parts = [np.ones(n_all, dtype=np.float64)]
            for s in observed_seasons[1:]:
                X_all_parts.append((q_u == s).astype(np.float64))
            X_all = np.column_stack(X_all_parts)
            y_hat = X_all @ coefs

            # Residuals
            ydot_out[idx_u] = y_u - y_hat

            # Collect diagnostics for this unit
            if return_diagnostics:
                seasonal_effects = {
                    int(s): float(coefs[idx + 1]) for idx, s in enumerate(observed_seasons[1:])
                }
                per_unit[uid] = {
                    "intercept": float(coefs[0]),
                    "seasonal_effects": seasonal_effects,
                    "pre_n_periods": n_pre,
                    "valid": True,
                }

        df["_ydot"] = ydot_out

        if return_diagnostics:
            valid_units = [uid for uid, info in per_unit.items() if info["valid"]]
            n_valid = len(valid_units)
            n_total = len(per_unit)
            diagnostics: Dict[str, Any] = {
                "method": "demeanq",
                "description": "Remove unit-specific seasonal (quarterly) fixed effects from pre-treatment",
                "per_unit": per_unit,
                "summary": {
                    "n_units_total": n_total,
                    "n_units_valid": n_valid,
                    "n_units_dropped": n_total - n_valid,
                },
            }
            return df, diagnostics

        return df

    def _transform_detrendq(
        self,
        df: pd.DataFrame,
        outcome_col: str,
        unit_col: str,
        time_col: str,
        pre_mask: Union[pd.Series, np.ndarray],
        return_diagnostics: bool = False,
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, Any]]]:
        """Apply unit-specific linear detrending with seasonal adjustment.

        For each unit, fit Y on [1, t, Q2, Q3, Q4] using pre-treatment
        periods only, then subtract fitted values from ALL periods.
        Quarter is determined by time_col % 4.

        Parameters
        ----------
        df : pd.DataFrame
            Panel data.
        outcome_col : str
            Name of the outcome column.
        unit_col : str
            Name of the unit identifier column.
        time_col : str
            Name of the time period column.
        pre_mask : Series or ndarray of bool
            Boolean mask indicating pre-treatment observations.
        return_diagnostics : bool, default False
            If True, return (df, diagnostics) tuple instead of just df.

        Returns
        -------
        pd.DataFrame or (pd.DataFrame, dict)
            Input data with '_ydot' column containing detrended+seasonally-adjusted outcomes.
            If return_diagnostics=True, also returns diagnostics dict.
        """
        df = df.copy()
        df["_ydot"] = np.nan

        # Determine quarter from time column
        t_series = df[time_col]
        if pd.api.types.is_datetime64_any_dtype(t_series):
            quarters = t_series.dt.quarter.to_numpy()
        elif hasattr(t_series.iloc[0], "quarter"):
            quarters = np.array([v.quarter for v in t_series])
        else:
            t_vals = t_series.to_numpy()
            quarters = (t_vals.astype(np.int64) - 1) % 4 + 1

        # Pre-extract numpy arrays to avoid repeated df.loc[] overhead
        unit_arr = df[unit_col].values
        time_arr = df[time_col].values.astype(np.float64)
        y_arr = df[outcome_col].values.astype(np.float64)
        pre_arr = pre_mask.values if hasattr(pre_mask, "values") else np.asarray(pre_mask)

        units = df[unit_col].unique()
        per_unit: Dict[Any, Dict[str, Any]] = {}
        ydot_out = np.full(len(df), np.nan)

        for uid in units:
            mask_u = unit_arr == uid
            idx_u = np.where(mask_u)[0]
            t_u = time_arr[idx_u]
            y_u = y_arr[idx_u]
            q_u = quarters[idx_u]
            pre_u = pre_arr[idx_u].astype(bool)

            # Pre-treatment data for this unit
            n_pre = int(pre_u.sum())

            if n_pre < 2:
                warnings.warn(
                    f"Unit {uid}: detrendq requires at least 2 "
                    f"pre-treatment periods, found {n_pre}. "
                    f"Transformed outcome set to NaN.",
                    UserWarning,
                    stacklevel=2,
                )
                if return_diagnostics:
                    per_unit[uid] = {
                        "alpha": float("nan"),
                        "beta": float("nan"),
                        "seasonal_effects": {},
                        "pre_n_periods": n_pre,
                        "valid": False,
                    }
                continue

            # Check seasonal parameters
            q_pre = q_u[pre_u]
            t_pre = t_u[pre_u]
            observed_seasons = sorted(np.unique(q_pre))
            # Parameters: intercept + slope + (n_seasons - 1) dummies
            n_params = 1 + len(observed_seasons)

            y_pre = y_u[pre_u]

            # Center time for numerical stability
            t_mean = t_pre.mean()
            t_pre_centered = t_pre - t_mean

            # If insufficient obs for full model, fall back to detrend-only
            use_seasonal = n_pre >= n_params
            if use_seasonal:
                # Build design matrix: [1, t_centered, Q2, Q3, Q4]
                X_pre_parts = [
                    np.ones(n_pre, dtype=np.float64),
                    t_pre_centered,
                ]
                for s in observed_seasons[1:]:
                    X_pre_parts.append((q_pre == s).astype(np.float64))
            else:
                # Fallback: detrend only (intercept + slope)
                X_pre_parts = [
                    np.ones(n_pre, dtype=np.float64),
                    t_pre_centered,
                ]
            X_pre = np.column_stack(X_pre_parts)

            # Solve via scipy.linalg.lstsq
            result = scipy_linalg.lstsq(X_pre, y_pre, cond=None)
            coefs = result[0]

            if not np.all(np.isfinite(coefs)):
                warnings.warn(
                    f"Unit {uid}: detrendq produced non-finite "
                    f"coefficients. Transformed outcome set to NaN.",
                    UserWarning,
                    stacklevel=2,
                )
                if return_diagnostics:
                    per_unit[uid] = {
                        "alpha": float("nan"),
                        "beta": float("nan"),
                        "seasonal_effects": {},
                        "pre_n_periods": n_pre,
                        "valid": False,
                    }
                continue

            # Predict on ALL periods for this unit
            t_all_centered = t_u - t_mean
            n_all = len(t_u)

            X_all_parts = [
                np.ones(n_all, dtype=np.float64),
                t_all_centered,
            ]
            if use_seasonal:
                for s in observed_seasons[1:]:
                    X_all_parts.append((q_u == s).astype(np.float64))
            X_all = np.column_stack(X_all_parts)
            y_hat = X_all @ coefs

            # Residuals
            ydot_out[idx_u] = y_u - y_hat

            # Collect diagnostics for this unit
            if return_diagnostics:
                if use_seasonal:
                    seasonal_effects = {
                        int(s): float(coefs[idx + 2]) for idx, s in enumerate(observed_seasons[1:])
                    }
                else:
                    seasonal_effects = {}
                per_unit[uid] = {
                    "alpha": float(coefs[0]),
                    "beta": float(coefs[1]),
                    "seasonal_effects": seasonal_effects,
                    "pre_n_periods": n_pre,
                    "valid": True,
                }

        df["_ydot"] = ydot_out

        if return_diagnostics:
            valid_units = [uid for uid, info in per_unit.items() if info["valid"]]
            n_valid = len(valid_units)
            n_total = len(per_unit)
            diagnostics: Dict[str, Any] = {
                "method": "detrendq",
                "description": "Remove unit-specific trend + seasonal effects (\u03b1\u0302_i + \u03b2\u0302_i*t + \u03a3\u03b3\u0302_q*Q_q)",
                "per_unit": per_unit,
                "summary": {
                    "n_units_total": n_total,
                    "n_units_valid": n_valid,
                    "n_units_dropped": n_total - n_valid,
                },
            }
            return df, diagnostics

        return df

    def _compute_hc4_vcov(
        self,
        X: np.ndarray,
        y: np.ndarray,
        coefs: np.ndarray,
    ) -> np.ndarray:
        """Compute HC4 heteroskedasticity-consistent covariance matrix.

        Implements the HC4 estimator of Cribari-Neto (2004), which uses an
        adaptive leverage-based exponent to downweight high-leverage observations
        more aggressively than HC3.

        Mathematical formula:
            V̂_HC4 = (X'X)^{-1} · M · (X'X)^{-1}

        where the meat matrix M is:
            M = X' · diag(ê_i² / (1 - h_ii)^δ_i) · X

        and the adaptive exponent δ_i is:
            δ_i = min(4, n · h_ii / p)

        Here h_ii are the diagonal elements of the hat matrix H = X(X'X)⁻¹X'.

        Compared to HC3 (which uses fixed exponent 2), HC4 adapts the
        downweighting strength based on each observation's relative leverage
        (h_ii compared to average leverage p/n).

        Parameters
        ----------
        X : np.ndarray of shape (n, p)
            Design matrix.
        y : np.ndarray of shape (n,)
            Response variable.
        coefs : np.ndarray of shape (p,)
            OLS coefficient estimates.

        Returns
        -------
        np.ndarray of shape (p, p)
            HC4 variance-covariance matrix of the coefficient estimates.

        References
        ----------
        Cribari-Neto, F. (2004). "Asymptotic inference under
            heteroskedasticity of unknown form." Computational Statistics
            & Data Analysis, 45(2), 215-233.
        """
        n, k = X.shape
        residuals = y - X @ coefs

        # Compute (X'X)^{-1}
        XtX = X.T @ X
        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            # Fall back to pseudo-inverse
            XtX_inv = np.linalg.pinv(XtX)

        # Compute hat matrix diagonals: h_ii = x_i' (X'X)^{-1} x_i
        # Efficient computation: H_diag = row_sum(X @ (X'X)^{-1} * X)
        h_diag = np.sum((X @ XtX_inv) * X, axis=1)
        h_diag = np.clip(h_diag, 0.0, 1.0 - 1e-10)

        # HC4 exponent: δ_i = min(4, n * h_ii / p)
        # Since sum(h_ii) = p, h_bar = p/n, so h_ii/h_bar = n*h_ii/p
        h_bar = h_diag.sum() / n  # = p/n (average leverage)
        d = np.minimum(4.0, h_diag / h_bar)

        # Adjusted residuals: e_i^2 / (1 - h_ii)^d_i
        adj_resid_sq = residuals**2 / (1.0 - h_diag) ** d

        # Meat: X' diag(adj_resid_sq) X
        meat = (X.T * adj_resid_sq) @ X

        # Sandwich: (X'X)^{-1} meat (X'X)^{-1}
        vcov = XtX_inv @ meat @ XtX_inv
        return vcov

    def _dispatch_estimator(
        self,
        y: np.ndarray,
        treatment: np.ndarray,
        controls_matrix: Optional[np.ndarray],
        cluster_ids: Optional[np.ndarray],
        n_obs: int,
    ) -> Tuple[float, float, Optional[np.ndarray], Optional[np.ndarray], int]:
        """Dispatch estimation to the appropriate method based on self.estimator.

        This is the central routing function that maps the user's estimator choice
        to the corresponding implementation. After unit-specific rolling transformation
        converts the panel into a cross-sectional dataset, this method applies the
        chosen treatment-effect estimator to obtain the ATT.

        Corresponds to Step 2 of the Lee & Wooldridge (2025, 2026) procedure:
        after computing \u1e8e_{ir} (transformed outcome), apply RA/IPW/IPWRA/PSM
        to the cross-section {(\u1e8e_{ir}, D_i, X_i)}.

        Parameters
        ----------
        y : np.ndarray of shape (n,)
            Transformed outcome variable (\u1e8e_{ir} in paper notation).
            This is the post-transformation average residual for each unit.
        treatment : np.ndarray of shape (n,)
            Binary treatment indicator (D_i). 1 = treated, 0 = control.
        controls_matrix : np.ndarray of shape (n, K) or None
            Covariate matrix (X_i). None if no controls specified.
            Used for regression adjustment, propensity score, and matching.
        cluster_ids : np.ndarray of shape (n,) or None
            Cluster identifiers for cluster-robust variance estimation.
            None if vce != 'cluster'.
        n_obs : int
            Number of cross-sectional observations (units).

        Returns
        -------
        tuple of (att, se, coefs, vcov, K_controls)
            att : float
                Estimated average treatment effect on the treated (\u03c4\u0302 in paper).
            se : float
                Standard error of the ATT estimate.
            coefs : np.ndarray or None
                Full coefficient vector from the regression (RA/IPW paths).
                None for PSM.
            vcov : np.ndarray or None
                Variance-covariance matrix of coefficients.
                None for PSM.
            K_controls : int
                Number of control variables (K), used for degrees of freedom
                computation: df = N - K - 2 (per paper Section 2.4).

        Raises
        ------
        ValueError
            If self.estimator is not in {'ra', 'ipw', 'ipwra', 'psm'}.
            (Should not occur if __init__ validation passed.)

        Notes
        -----
        Routing logic:
        - 'ra'    \u2192 _estimate_ra(): OLS of \u1e8e on [1, D, X, D*(X-X\u0304\u2081)]
                    per Equation 3.3 in Lee & Wooldridge (2025)
        - 'ipw'   \u2192 _estimate_ipw(): Inverse probability weighting via
                    logit propensity score, Hajek-style normalization
        - 'ipwra' \u2192 _estimate_ipwra(): Doubly-robust augmented IPW
                    combining outcome model and propensity weighting
        - 'psm'   \u2192 _estimate_psm(): Nearest-neighbor propensity score
                    matching (1:n with optional caliper)

        When controls_matrix is None, IPW/IPWRA/PSM fall back to RA
        (simple difference in means) with a warning.

        The VCE type (self.vce) determines which variance estimator is used:
        - 'classical': homoskedastic OLS variance
        - 'hc1': HC1 (White) heteroskedasticity-robust
        - 'hc3': HC3 (leverage-adjusted, computed post-hoc)
        - 'hc4': HC4 (alternative leverage adjustment)
        - 'cluster': cluster-robust (Liang-Zeger sandwich)

        References
        ----------
        Lee, S. & Wooldridge, J. M. (2025). "A Simple Transformation Approach
            to Difference-in-Differences Estimation for Panel Data."
            Procedure 3.1, Equation 3.3.
        Lee, S. & Wooldridge, J. M. (2026). "Simple Difference-in-Differences
            Estimation in Panel Data." Procedure 2.1.
        """
        if self.estimator == "ra":
            return self._estimate_ra(y, treatment, controls_matrix, cluster_ids, n_obs)
        elif self.estimator == "ipw":
            return self._estimate_ipw(y, treatment, controls_matrix, cluster_ids, n_obs)
        elif self.estimator == "psm":
            return self._estimate_psm(y, treatment, controls_matrix, cluster_ids, n_obs)
        else:  # ipwra
            return self._estimate_ipwra(y, treatment, controls_matrix, cluster_ids, n_obs)

    def _estimate_ra(
        self,
        y: np.ndarray,
        treatment: np.ndarray,
        controls_matrix: Optional[np.ndarray],
        cluster_ids: Optional[np.ndarray],
        n_obs: int,
    ) -> Tuple[float, float, Optional[np.ndarray], Optional[np.ndarray], int]:
        """Estimate ATT via regression adjustment (OLS).

        Fits y = alpha + tau*D + X*beta + D*(X - X_bar_1)*gamma + epsilon
        and returns tau as the ATT estimate (LW2025 Equation 3.3).

        The interaction term D*(X - X_bar_1) allows covariate effects to
        differ between treated and control groups. It is only included when
        both N_treated > K+1 and N_control > K+1.

        Parameters
        ----------
        y : ndarray of shape (n,)
            Transformed outcome.
        treatment : ndarray of shape (n,)
            Binary treatment indicator.
        controls_matrix : ndarray of shape (n, p) or None
            Control variables.
        cluster_ids : ndarray of shape (n,) or None
            Cluster identifiers for cluster-robust SEs.
        n_obs : int
            Number of observations.

        Returns
        -------
        att : float
            Treatment effect coefficient.
        se : float
            Standard error of treatment coefficient.
        coefs : ndarray
            Full coefficient vector.
        vcov : ndarray or None
            Variance-covariance matrix.
        n_params : int
            Number of parameters in the regression.
        """
        # Build design matrix: [intercept, treatment, controls, interaction]
        parts = [np.ones((n_obs, 1)), treatment.reshape(-1, 1)]
        if controls_matrix is not None:
            parts.append(controls_matrix)
            # Add D*(X - X_bar_1) interaction term when sample sizes permit
            # (LW2025 Eq 3.3: requires N_0 > K+1 and N_1 > K+1)
            K = controls_matrix.shape[1]
            treated_mask = treatment == 1
            n_treated = int(treated_mask.sum())
            n_control = n_obs - n_treated
            if n_treated > K + 1 and n_control > K + 1:
                X_bar_1 = controls_matrix[treated_mask].mean(axis=0)
                interaction = treatment.reshape(-1, 1) * (controls_matrix - X_bar_1)
                parts.append(interaction)
        X = np.hstack(parts)
        n_params = X.shape[1]

        # Determine vcov_type for solve_ols
        vcov_type = self._resolve_vcov_type()

        # Call solve_ols
        coefs, residuals, vcov = solve_ols(
            X,
            y,
            cluster_ids=cluster_ids,
            return_vcov=True,
            vcov_type=vcov_type,
        )

        # Post-hoc VCE corrections for HC0, HC3, and HC4
        if self.vce == "hc0" and vcov is not None:
            # HC0 = HC1 without the n/(n-k) DOF adjustment
            # solve_ols HC1 applies factor n/(n-k), so undo it
            vcov = vcov * (n_obs - n_params) / n_obs
        elif self.vce == "hc3" and vcov is not None:
            vcov = self._compute_hc3_vcov(X, y, coefs)
        elif self.vce == "hc4" and vcov is not None:
            vcov = self._compute_hc4_vcov(X, y, coefs)

        # ATT = coefficient on treatment (index 1)
        att = float(coefs[1])
        # SE from vcov diagonal
        if vcov is not None and np.isfinite(vcov[1, 1]):
            se = float(np.sqrt(max(vcov[1, 1], 0.0)))
        else:
            se = np.nan

        # Return effective K (number of control variables) for df computation.
        # Paper requires df = N - K - 2, where K = number of controls.
        K_controls = controls_matrix.shape[1] if controls_matrix is not None else 0
        return att, se, coefs, vcov, K_controls

    def _estimate_ipw(
        self,
        y: np.ndarray,
        treatment: np.ndarray,
        controls_matrix: Optional[np.ndarray],
        cluster_ids: Optional[np.ndarray],
        n_obs: int,
    ) -> Tuple[float, float, Optional[np.ndarray], Optional[np.ndarray], int]:
        """Estimate ATT via inverse probability weighting.

        Uses propensity scores to reweight control observations.

        Parameters
        ----------
        y : ndarray of shape (n,)
            Transformed outcome.
        treatment : ndarray of shape (n,)
            Binary treatment indicator.
        controls_matrix : ndarray of shape (n, p) or None
            Covariates for propensity score model.
        cluster_ids : ndarray of shape (n,) or None
            Cluster identifiers.
        n_obs : int
            Number of observations.

        Returns
        -------
        att : float
            IPW-estimated ATT.
        se : float
            Standard error.
        coefs : ndarray or None
            Not returned for IPW (None).
        vcov : ndarray or None
            Not returned for IPW (None).
        n_params : int
            Number of parameters in the underlying regression.
        """
        if controls_matrix is None or controls_matrix.shape[1] == 0:
            # Without covariates, IPW reduces to simple difference
            # in means (propensity score is constant)
            warnings.warn(
                "IPW without control variables reduces to a simple "
                "difference in means. Consider using estimator='ra'.",
                UserWarning,
                stacklevel=2,
            )
            return self._estimate_ra(
                y, treatment, None, cluster_ids, n_obs
            )  # returns 5-tuple including n_params

        # Step 1: Estimate propensity score via logit
        # solve_logit adds intercept automatically
        coefs_logit, probs = solve_logit(controls_matrix, treatment)

        # Convergence check: coefficients must be finite
        if not np.all(np.isfinite(coefs_logit)):
            warnings.warn(
                "Logistic regression did not converge (non-finite coefficients). "
                "Falling back to RA estimation. Consider standardizing controls.",
                UserWarning,
                stacklevel=2,
            )
            return self._estimate_ra(y, treatment, controls_matrix, cluster_ids, n_obs)

        # Convergence check: complete/quasi-complete separation
        if np.any(probs < 1e-8) or np.any(probs > 1 - 1e-8):
            warnings.warn(
                "Possible complete separation detected in propensity score model. "
                "Some predicted probabilities are near 0 or 1. "
                "Results may be unreliable.",
                UserWarning,
                stacklevel=2,
            )

        # Step 2: Trim propensity scores to [trim_threshold, 1 - trim_threshold]
        trim_lo, trim_hi = self.trim_threshold, 1.0 - self.trim_threshold
        n_trimmed = int((probs < trim_lo).sum() + (probs > trim_hi).sum())
        if n_trimmed > 0:
            warnings.warn(
                f"LWDiD: {n_trimmed} observation(s) had propensity scores trimmed "
                f"to [{self.trim_threshold:.3f}, {1-self.trim_threshold:.3f}].",
                UserWarning,
                stacklevel=2,
            )
        probs = np.clip(probs, trim_lo, trim_hi)

        # Step 3: Compute IPW weights
        # For treated: weight = 1
        # For control: weight = p(x) / (1 - p(x))
        # Normalized so control weights sum to n_treated
        ipw_weights = np.where(
            treatment == 1,
            1.0,
            probs / (1.0 - probs),
        )

        # Normalize weights: treated get weight 1/n_treated,
        # control weights normalized to sum to 1
        treat_mask = treatment == 1
        ctrl_mask = treatment == 0

        w_ctrl_sum = ipw_weights[ctrl_mask].sum()
        if w_ctrl_sum <= 0:
            warnings.warn(
                "IPW control weights sum to zero. Falling back to " "unweighted RA estimation.",
                UserWarning,
                stacklevel=2,
            )
            return self._estimate_ra(y, treatment, controls_matrix, cluster_ids, n_obs)

        # Hajek-style ATT estimator
        att_treated = y[treat_mask].mean()
        att_control = np.sum(ipw_weights[ctrl_mask] * y[ctrl_mask]) / w_ctrl_sum
        att = float(att_treated - att_control)

        # Step 4: Compute SE via semiparametric influence function
        # Follows Lunceford & Davidian (2004), matching Stata lwdid and lwdid-py.
        # The full IF consists of the Hajek main term plus a propensity score
        # estimation uncertainty correction.
        n_treated_f = float(treat_mask.sum())
        p_bar = n_treated_f / n_obs  # P(D=1) estimate

        # --- Hajek influence function (main term) ---
        w_ctrl = ipw_weights[ctrl_mask]  # p/(1-p) for controls

        psi_ht = np.zeros(n_obs)
        psi_ht[treat_mask] = (y[treat_mask] - att_treated) / p_bar
        psi_ht[ctrl_mask] = -w_ctrl * (y[ctrl_mask] - att_control) / p_bar

        # --- Propensity score estimation uncertainty correction ---
        # Design matrix with intercept (solve_logit adds intercept internally,
        # so we reconstruct it here for the IF computation).
        X_ps = np.column_stack([np.ones(n_obs), controls_matrix])

        # Logit score: S_i = (D_i - p_i) * X_i
        S_gamma = (treatment - probs)[:, np.newaxis] * X_ps

        # Logit Hessian: H = -(1/n) * X' diag(p*(1-p)) X
        W_ps = probs * (1 - probs)
        H_gamma = -(X_ps.T * W_ps) @ X_ps / n_obs
        try:
            H_gamma_inv = np.linalg.inv(H_gamma)
        except np.linalg.LinAlgError:
            H_gamma_inv = np.linalg.pinv(H_gamma)

        # Sensitivity: dATT/dgamma
        # dw/dgamma_i = w_i * X_i (logit chain rule)
        # dATT/dgamma = -(1/w_sum) * sum_ctrl(w_i * X_i * (Y_i - mu_0))
        # The (Y_i - mu_0) centering comes from the quotient rule for the
        # Hajek estimator (d/dgamma of Sigma(wY)/Sigma(w)) and ensures
        # translation invariance of the resulting SE.
        dw_dgamma_ctrl = w_ctrl[:, np.newaxis] * X_ps[ctrl_mask]
        Y_ctrl_centered = (y[ctrl_mask] - att_control)
        dATT_dgamma = -(dw_dgamma_ctrl * Y_ctrl_centered[:, np.newaxis]).sum(axis=0) / (n_obs * p_bar)

        # PS adjustment: psi_adj_i = (S_i @ H^{-1}) @ dATT_dgamma
        ps_adjustment = (S_gamma @ H_gamma_inv.T) @ dATT_dgamma

        # Full IF = main term - PS correction
        psi_full = psi_ht - ps_adjustment

        # --- Variance estimation ---
        if cluster_ids is not None and self.vce == "cluster":
            cluster_df = pd.DataFrame({"psi": psi_full, "cluster": cluster_ids})
            cluster_sums = cluster_df.groupby("cluster")["psi"].sum().values
            n_clusters = len(cluster_sums)
            if n_clusters <= 1:
                warnings.warn(
                    "Only 1 cluster found; falling back to non-clustered "
                    "variance for IPW influence function.",
                    UserWarning,
                    stacklevel=2,
                )
                var_att = float(np.var(psi_full, ddof=1) / n_obs)
            else:
                var_att = float(
                    (n_clusters / (n_clusters - 1)) * np.sum(cluster_sums**2) / n_obs**2
                )
        else:
            var_att = float(np.var(psi_full, ddof=1) / n_obs)

        se = float(np.sqrt(max(var_att, 0.0)))

        # n_params: intercept + controls (propensity model)
        n_params = 1 + controls_matrix.shape[1]
        return att, se, None, None, n_params

    def _estimate_psm(
        self,
        y: np.ndarray,
        treatment: np.ndarray,
        controls_matrix: Optional[np.ndarray],
        cluster_ids: Optional[np.ndarray],
        n_obs: int,
    ) -> Tuple[float, float, Optional[np.ndarray], Optional[np.ndarray], int]:
        """Estimate ATT via propensity score matching.

        For each treated unit, find the nearest control unit by propensity
        score (1:1 nearest-neighbor matching with replacement), then compute
        ATT as the average difference between treated and matched control.

        Parameters
        ----------
        y : ndarray of shape (n,)
            Transformed outcome.
        treatment : ndarray of shape (n,)
            Binary treatment indicator.
        controls_matrix : ndarray of shape (n, p) or None
            Covariates for propensity score model.
        cluster_ids : ndarray of shape (n,) or None
            Cluster identifiers.
        n_obs : int
            Number of observations.

        Returns
        -------
        att : float
            PSM-estimated ATT.
        se : float
            Standard error (simple matching SE).
        coefs : ndarray or None
            Not returned for PSM (None).
        vcov : ndarray or None
            Not returned for PSM (None).
        n_params : int
            Effective number of parameters.
        """
        if controls_matrix is None or controls_matrix.shape[1] == 0:
            # Without covariates, PSM reduces to simple difference in means
            warnings.warn(
                "PSM without control variables reduces to a simple "
                "difference in means. Consider using estimator='ra'.",
                UserWarning,
                stacklevel=2,
            )
            return self._estimate_ra(y, treatment, None, cluster_ids, n_obs)

        treat_mask = treatment == 1
        ctrl_mask = treatment == 0
        n_treated = int(treat_mask.sum())
        n_control = int(ctrl_mask.sum())

        if n_treated == 0 or n_control == 0:
            warnings.warn(
                "PSM estimation failed: no treated or no control units available. "
                "Returning NaN results.",
                UserWarning,
                stacklevel=2,
            )
            return np.nan, np.nan, None, None, 2

        # Step 1: Estimate propensity score via logit
        coefs_logit, probs = solve_logit(controls_matrix, treatment)

        # Convergence check: coefficients must be finite
        if not np.all(np.isfinite(coefs_logit)):
            warnings.warn(
                "Logistic regression did not converge (non-finite coefficients). "
                "Falling back to RA estimation. Consider standardizing controls.",
                UserWarning,
                stacklevel=2,
            )
            return self._estimate_ra(y, treatment, controls_matrix, cluster_ids, n_obs)

        # Convergence check: complete/quasi-complete separation
        if np.any(probs < 1e-8) or np.any(probs > 1 - 1e-8):
            warnings.warn(
                "Possible complete separation detected in propensity score model. "
                "Some predicted probabilities are near 0 or 1. "
                "Results may be unreliable.",
                UserWarning,
                stacklevel=2,
            )

        # Step 2: Trim propensity scores to [trim_threshold, 1 - trim_threshold]
        trim_lo, trim_hi = self.trim_threshold, 1.0 - self.trim_threshold
        n_trimmed = int((probs < trim_lo).sum() + (probs > trim_hi).sum())
        if n_trimmed > 0:
            warnings.warn(
                f"LWDiD: {n_trimmed} observation(s) had propensity scores trimmed "
                f"to [{self.trim_threshold:.3f}, {1-self.trim_threshold:.3f}].",
                UserWarning,
                stacklevel=2,
            )
        probs = np.clip(probs, trim_lo, trim_hi)

        # Step 3: Nearest-neighbor matching (with replacement)
        p_treated = probs[treat_mask]
        p_control = probs[ctrl_mask]
        y_treated = y[treat_mask]
        y_control = y[ctrl_mask]

        # For each treated unit, find n_neighbors nearest controls
        matched_y_control = np.empty(n_treated)
        available_mask = np.ones(n_control, dtype=bool)

        for i in range(n_treated):
            valid_control_idx = np.where(available_mask)[0]
            if len(valid_control_idx) == 0:
                matched_y_control[i] = np.nan
                continue

            distances = np.abs(p_treated[i] - p_control[valid_control_idx])

            if self.caliper is not None:
                within_caliper = distances <= self.caliper
                if not within_caliper.any():
                    matched_y_control[i] = np.nan
                    continue
                distances = np.where(within_caliper, distances, np.inf)

            nearest_local = np.argsort(distances)[: self.n_neighbors]
            nearest_global = valid_control_idx[nearest_local]
            matched_y_control[i] = y_control[nearest_global].mean()

            if not self.with_replacement:
                available_mask[nearest_global] = False

        # Step 4: Compute ATT = mean(Y_treated - Y_matched_control)
        # Exclude NaN matches (from caliper)
        valid_matches = np.isfinite(matched_y_control)
        n_unmatched = int(np.isnan(matched_y_control).sum())
        if n_unmatched > 0:
            warnings.warn(
                f"LWDiD PSM: {n_unmatched} treated unit(s) could not be matched "
                f"within caliper={self.caliper}. ATT computed from {n_treated - n_unmatched} matches.",
                UserWarning,
                stacklevel=2,
            )
        if not valid_matches.any():
            warnings.warn(
                "PSM estimation failed: no valid matches found (all exceeded caliper). "
                "Returning NaN results.",
                UserWarning,
                stacklevel=2,
            )
            return np.nan, np.nan, None, None, 2
        diffs = y_treated[valid_matches] - matched_y_control[valid_matches]
        att = float(np.mean(diffs))

        # Step 5: Compute SE
        # Simple matching SE: SE = sqrt(Var(diffs) / N_treated)
        n_matched = int(valid_matches.sum())
        if n_matched > 1:
            var_diffs = float(np.var(diffs, ddof=1))
            se = float(np.sqrt(var_diffs / n_matched))
        else:
            se = np.nan

        # Effective n_params: intercept + controls (for propensity model)
        n_params = 1 + controls_matrix.shape[1]
        return att, se, None, None, n_params

    def _estimate_ipwra(
        self,
        y: np.ndarray,
        treatment: np.ndarray,
        controls_matrix: Optional[np.ndarray],
        cluster_ids: Optional[np.ndarray],
        n_obs: int,
    ) -> Tuple[float, float, Optional[np.ndarray], Optional[np.ndarray], int]:
        """Estimate ATT via augmented IPW (doubly robust).

        Combines regression adjustment with inverse probability weighting
        for double robustness.

        Parameters
        ----------
        y : ndarray of shape (n,)
            Transformed outcome.
        treatment : ndarray of shape (n,)
            Binary treatment indicator.
        controls_matrix : ndarray of shape (n, p) or None
            Covariates.
        cluster_ids : ndarray of shape (n,) or None
            Cluster identifiers.
        n_obs : int
            Number of observations.

        Returns
        -------
        att : float
            Doubly-robust ATT estimate.
        se : float
            Standard error.
        coefs : ndarray or None
            Not returned for IPWRA (None).
        vcov : ndarray or None
            Not returned for IPWRA (None).
        n_params : int
            Effective number of parameters for df computation.
        """
        if controls_matrix is None or controls_matrix.shape[1] == 0:
            # Without covariates, IPWRA reduces to RA
            return self._estimate_ra(
                y, treatment, None, cluster_ids, n_obs
            )  # returns 5-tuple including n_params

        treat_mask = treatment == 1
        ctrl_mask = treatment == 0
        n_treated = int(treat_mask.sum())
        n_control = int(ctrl_mask.sum())

        # Step 1: Get propensity scores
        coefs_logit, probs = solve_logit(controls_matrix, treatment)

        # Convergence check: coefficients must be finite
        if not np.all(np.isfinite(coefs_logit)):
            warnings.warn(
                "Logistic regression did not converge (non-finite coefficients). "
                "Falling back to RA estimation. Consider standardizing controls.",
                UserWarning,
                stacklevel=2,
            )
            return self._estimate_ra(y, treatment, controls_matrix, cluster_ids, n_obs)

        # Convergence check: complete/quasi-complete separation
        if np.any(probs < 1e-8) or np.any(probs > 1 - 1e-8):
            warnings.warn(
                "Possible complete separation detected in propensity score model. "
                "Some predicted probabilities are near 0 or 1. "
                "Results may be unreliable.",
                UserWarning,
                stacklevel=2,
            )

        trim_lo_ipwra, trim_hi_ipwra = self.trim_threshold, 1.0 - self.trim_threshold
        n_trimmed_ipwra = int((probs < trim_lo_ipwra).sum() + (probs > trim_hi_ipwra).sum())
        if n_trimmed_ipwra > 0:
            warnings.warn(
                f"LWDiD: {n_trimmed_ipwra} observation(s) had propensity scores trimmed "
                f"to [{self.trim_threshold:.3f}, {1-self.trim_threshold:.3f}].",
                UserWarning,
                stacklevel=2,
            )
        probs = np.clip(probs, self.trim_threshold, 1.0 - self.trim_threshold)

        # Step 2: Fit outcome model on control units only using WLS with IPW weights
        # This matches the Stata/lwdid-py reference: outcome model is fitted on
        # controls with weights w_i = p(X_i)/(1-p(X_i)) to target ATT.
        X_ctrl = np.column_stack([np.ones(n_control), controls_matrix[ctrl_mask]])
        y_ctrl = y[ctrl_mask]

        # IPW weights for control units
        ipw_ctrl = probs[ctrl_mask] / (1.0 - probs[ctrl_mask])
        ipw_ctrl_sum = ipw_ctrl.sum()

        if ipw_ctrl_sum <= 0:
            # Fall back to RA if IPW weights degenerate
            return self._estimate_ra(
                y, treatment, controls_matrix, cluster_ids, n_obs
            )  # returns 5-tuple including n_params

        # WLS via sqrt(w) transformation: beta = (X'WX)^{-1} X'WY
        sqrt_w = np.sqrt(ipw_ctrl)
        X_ctrl_w = X_ctrl * sqrt_w[:, np.newaxis]
        y_ctrl_w = y_ctrl * sqrt_w
        try:
            XtWX_inv = np.linalg.inv(X_ctrl_w.T @ X_ctrl_w)
            coefs_outcome = XtWX_inv @ (X_ctrl_w.T @ y_ctrl_w)
        except np.linalg.LinAlgError:
            XtWX_inv = np.linalg.pinv(X_ctrl_w.T @ X_ctrl_w)
            coefs_outcome = XtWX_inv @ (X_ctrl_w.T @ y_ctrl_w)

        # Predict counterfactual for all units
        X_all = np.column_stack([np.ones(n_obs), controls_matrix])
        mu_0 = X_all @ coefs_outcome

        # Step 3: Compute AIPW/IPWRA estimator (Hajek normalization)
        # ATT = mean_{D=1}(Y - mu_0) - sum_{D=0}[w*(Y-mu_0)] / sum_{D=0}(w)
        resid = y - mu_0
        resid_ctrl = resid[ctrl_mask]

        # Treated component
        att_treated_part = resid[treat_mask].mean()

        # Control component (Hajek: divide by sum of weights)
        weights_sum = ipw_ctrl_sum
        att_ctrl_part = np.sum(ipw_ctrl * resid_ctrl) / weights_sum

        att = float(att_treated_part - att_ctrl_part)

        # Step 4: Compute SE via full semiparametric influence function
        # The IPWRA IF consists of 3 components (Cattaneo 2010, Lunceford & Davidian 2004):
        # 1. Hajek main term (plug-in IF)
        # 2. Propensity score estimation uncertainty correction
        # 3. Outcome model estimation uncertainty correction
        n_treated_f = float(n_treated)
        p_bar = n_treated_f / n_obs  # P(D=1) estimate

        # Control term (Hajek weighted mean of control residuals)
        control_term = att_ctrl_part  # = sum(w*resid_C) / sum(w)

        # ================================================================
        # Component 1: Hajek influence function (main term)
        # Hajek linearization for ATT = mean_T(resid) - sum_C(w*resid)/sum_C(w)
        # ================================================================
        psi = np.zeros(n_obs)
        psi[treat_mask] = (resid[treat_mask] - att) / p_bar
        psi[ctrl_mask] = -ipw_ctrl * (resid_ctrl - control_term) / weights_sum * n_obs

        # ================================================================
        # Component 2: Propensity score estimation uncertainty correction
        # S_gamma_i = (D_i - p_i) * X_i (logit score)
        # H_gamma = -(1/n) * X' diag(p*(1-p)) X (logit Hessian)
        # dATT/dgamma = -sum_C[dw/dgamma * (resid - B)] / sum_C(w)
        # ================================================================
        X_ps = np.column_stack([np.ones(n_obs), controls_matrix])

        # Logit score
        S_gamma = (treatment - probs)[:, np.newaxis] * X_ps

        # Logit Hessian
        W_ps = probs * (1 - probs)
        H_gamma = -(X_ps.T * W_ps) @ X_ps / n_obs
        try:
            H_gamma_inv = np.linalg.inv(H_gamma)
        except np.linalg.LinAlgError:
            H_gamma_inv = np.linalg.pinv(H_gamma)

        # Sensitivity of ATT to propensity score parameters
        # dw/dgamma_i = w_i * X_i; chain through the Hajek control term
        r_minus_B = resid_ctrl - control_term
        dw_dgamma_ctrl = ipw_ctrl[:, np.newaxis] * X_ps[ctrl_mask]
        dATT_dgamma = -(dw_dgamma_ctrl * r_minus_B[:, np.newaxis]).sum(axis=0) / weights_sum

        # PS adjustment
        ps_adjustment = (S_gamma @ H_gamma_inv.T) @ dATT_dgamma

        # ================================================================
        # Component 3: Outcome model estimation uncertainty correction
        # The outcome model is WLS fitted on controls with IPW weights:
        #   E[Y|X, D=0] fitted by WLS with w_i = p/(1-p).
        # S_beta_i = w_i * resid_i * X_i * I(D_i=0) (WLS score)
        # H_beta = -(1/n) * X_ctrl' diag(w) X_ctrl (WLS Hessian)
        # dATT/dbeta = -mean_T(X_i) + sum_C(w_i*X_i) / sum_C(w)
        # ================================================================
        X_om = np.column_stack([np.ones(n_obs), controls_matrix])
        X_ctrl_om = X_om[ctrl_mask]

        # WLS score (nonzero only for control units)
        S_beta = np.zeros((n_obs, X_om.shape[1]))
        S_beta[ctrl_mask] = ipw_ctrl[:, np.newaxis] * resid_ctrl[:, np.newaxis] * X_ctrl_om

        # WLS Hessian: H_beta = -(1/n) * X_ctrl' diag(w) X_ctrl
        H_beta = -(X_ctrl_om.T * ipw_ctrl) @ X_ctrl_om / n_obs
        try:
            H_beta_inv = np.linalg.inv(H_beta)
        except np.linalg.LinAlgError:
            H_beta_inv = np.linalg.pinv(H_beta)

        # Sensitivity of ATT to outcome model parameters
        # dATT/dbeta = -mean_T(X_i) + weighted_mean_C(X_i)
        X_bar_treated = X_om[treat_mask].mean(axis=0)
        X_bar_ctrl_w = (ipw_ctrl[:, np.newaxis] * X_ctrl_om).sum(axis=0) / weights_sum
        dATT_dbeta = -X_bar_treated + X_bar_ctrl_w

        # Outcome model adjustment
        om_adjustment = (S_beta @ H_beta_inv.T) @ dATT_dbeta

        # ================================================================
        # Combine: full IF = main - PS correction - outcome correction
        # ================================================================
        psi_full = psi - ps_adjustment - om_adjustment

        # --- Variance estimation ---
        if cluster_ids is not None and self.vce == "cluster":
            # Cluster-robust: sum phi within clusters, then outer product
            cluster_df = pd.DataFrame({"psi": psi_full, "cluster": cluster_ids})
            cluster_sums = cluster_df.groupby("cluster")["psi"].sum().values
            n_clusters = len(cluster_sums)
            if n_clusters <= 1:
                warnings.warn(
                    "Only 1 cluster found; falling back to non-clustered "
                    "variance for IPWRA influence function.",
                    UserWarning,
                    stacklevel=2,
                )
                var_att = float(np.var(psi_full, ddof=1) / n_obs)
            else:
                var_att = float(
                    (n_clusters / (n_clusters - 1)) * np.sum(cluster_sums**2) / n_obs**2
                )
        else:
            var_att = float(np.var(psi_full, ddof=1) / n_obs)

        se = float(np.sqrt(max(var_att, 0.0)))

        # Effective n_params: intercept + treatment + controls (outcome model)
        # + propensity score parameters
        K = controls_matrix.shape[1]
        n_params = 2 + K
        return att, se, None, None, n_params

    def _resolve_vcov_type(self) -> str:
        """Map the user-facing vce parameter to solve_ols vcov_type.

        Returns
        -------
        str
            The vcov_type string compatible with solve_ols.
        """
        mapping = {
            "classical": "classical",
            "hc0": "hc1",  # HC0 = HC1 without DOF adjustment; corrected post-hoc
            "hc1": "hc1",
            "hc2": "hc2",
            "hc3": "hc1",  # HC3 computed post-hoc via hat diagonals
            "hc4": "hc1",  # HC4 computed post-hoc via hat diagonals
            "cluster": "hc1",  # cluster-robust uses hc1 with cluster_ids
        }
        return mapping[self.vce]

    def _compute_hc3_vcov(
        self,
        X: np.ndarray,
        y: np.ndarray,
        coefs: np.ndarray,
    ) -> np.ndarray:
        """Compute HC3 variance-covariance matrix.

        HC3 uses leverage-based adjustment: e_i^2 / (1 - h_ii)^2.
        More conservative than HC1 for small samples.

        Parameters
        ----------
        X : ndarray of shape (n, k)
            Design matrix.
        y : ndarray of shape (n,)
            Outcome vector.
        coefs : ndarray of shape (k,)
            OLS coefficient estimates.

        Returns
        -------
        ndarray of shape (k, k)
            HC3 variance-covariance matrix.
        """
        n, k = X.shape
        residuals = y - X @ coefs

        # Compute (X'X)^{-1}
        XtX = X.T @ X
        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            XtX_inv = np.linalg.pinv(XtX)

        # Compute hat matrix diagonals: h_ii = x_i' (X'X)^{-1} x_i
        h_diag = np.sum((X @ XtX_inv) * X, axis=1)
        h_diag = np.clip(h_diag, 0.0, 1.0 - 1e-10)

        # HC3: e_i^2 / (1 - h_ii)^2
        adj_resid_sq = residuals**2 / (1.0 - h_diag) ** 2

        # Meat: X' diag(adj_resid_sq) X
        meat = (X.T * adj_resid_sq) @ X

        # Sandwich: (X'X)^{-1} meat (X'X)^{-1}
        vcov = XtX_inv @ meat @ XtX_inv
        return vcov

    def _bootstrap(
        self,
        df: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        treatment: str,
        cluster: Optional[str],
        controls: List[str],
        pre_periods: List[Any],
        post_periods: List[Any],
        treated_units: List[Any],
        control_units: List[Any],
    ) -> Tuple[float, float, float, float, Tuple[float, float]]:
        """Compute bootstrap standard errors.

        Uses unit-level block bootstrap for panel data.

        Parameters
        ----------
        df : pd.DataFrame
            Full panel data.
        outcome : str
            Outcome column name.
        unit : str
            Unit identifier column name.
        time : str
            Time period column name.
        treatment : str
            Treatment indicator column name.
        cluster : str or None
            Cluster column name.
        controls : list of str
            Control variable column names.
        pre_periods : list
            Pre-treatment period values.
        post_periods : list
            Post-treatment period values.
        treated_units : list
            Treated unit identifiers.
        control_units : list
            Control unit identifiers.

        Returns
        -------
        att : float
            Point estimate from full sample.
        se : float
            Bootstrap standard error.
        t_stat : float
            t-statistic.
        p_value : float
            Two-sided p-value.
        conf_int : tuple of float
            Confidence interval (lower, upper).
        """
        # Full-sample estimate
        treated_set = set(treated_units)
        pre_mask = df[time].isin(pre_periods)
        if self.rolling == "demean":
            df_t = self._transform_demean(df, outcome, unit, pre_mask)
        elif self.rolling == "detrend":
            df_t = self._transform_detrend(df, outcome, unit, time, pre_mask)
        elif self.rolling == "demeanq":
            df_t = self._transform_demeanq(df, outcome, unit, time, pre_mask)
        elif self.rolling == "detrendq":
            df_t = self._transform_detrendq(df, outcome, unit, time, pre_mask)
        else:
            df_t = self._transform_detrend(df, outcome, unit, time, pre_mask)

        post_mask = df_t[time].isin(post_periods)  # type: ignore[union-attr, call-overload]
        post_df = df_t.loc[post_mask]  # type: ignore[union-attr]
        unit_post_avg = post_df.groupby(unit)["_ydot"].mean()

        cs_df = df.drop_duplicates(subset=[unit], keep="first")[[unit] + controls].copy()
        cs_df["_treat"] = cs_df[unit].isin(treated_set).astype(float)
        cs_df["_ydot_avg"] = cs_df[unit].map(unit_post_avg)
        cs_df = cs_df.dropna(subset=["_ydot_avg"])

        y_full = cs_df["_ydot_avg"].values.astype(np.float64)
        treat_full = cs_df["_treat"].values.astype(np.float64)
        controls_mat = cs_df[controls].values.astype(np.float64) if controls else None

        att_full, _, _, _, n_params_full = self._dispatch_estimator(
            y_full, treat_full, controls_mat, None, len(y_full)
        )

        # Bootstrap replications (unit-level block bootstrap)
        treated_arr = np.array(treated_units)
        control_arr = np.array(control_units)
        n_treated = len(treated_arr)
        n_control = len(control_arr)
        n_units = n_treated + n_control
        unit_counts = df.groupby(unit).size().to_dict()

        if self.n_jobs == 1:
            # --- Serial path (original implementation, unchanged) ---
            rng = np.random.default_rng(seed=self.bootstrap_seed)
            boot_atts = np.empty(self.n_bootstrap)
            for b in range(self.n_bootstrap):
                # Resample treated and control units SEPARATELY to preserve proportions
                boot_treated = rng.choice(treated_arr, size=n_treated, replace=True)
                boot_control = rng.choice(control_arr, size=n_control, replace=True)
                boot_units = np.concatenate([boot_treated, boot_control])

                # Build bootstrap sample (all periods for resampled units)
                boot_indices = []
                for i, u in enumerate(boot_units):
                    idx = df.index[df[unit] == u].tolist()
                    boot_indices.extend(idx)

                boot_df = df.iloc[boot_indices].copy()
                # Assign new unit IDs to handle duplicates (dict lookup, no sort needed)
                repeat_counts = [unit_counts[u] for u in boot_units]
                boot_df["_boot_unit"] = np.repeat(np.arange(n_units), repeat_counts)

                # Treatment indicator from group membership (not from raw data column)
                boot_treat_vec = np.array([1.0] * n_treated + [0.0] * n_control, dtype=np.float64)

                # Apply transformation
                pre_mask_b = boot_df[time].isin(pre_periods)
                if self.rolling == "demean":
                    boot_df = self._transform_demean(boot_df, outcome, "_boot_unit", pre_mask_b)
                elif self.rolling == "detrend":
                    boot_df = self._transform_detrend(
                        boot_df, outcome, "_boot_unit", time, pre_mask_b
                    )
                elif self.rolling == "demeanq":
                    boot_df = self._transform_demeanq(
                        boot_df, outcome, "_boot_unit", time, pre_mask_b
                    )
                elif self.rolling == "detrendq":
                    boot_df = self._transform_detrendq(
                        boot_df, outcome, "_boot_unit", time, pre_mask_b
                    )
                else:
                    boot_df = self._transform_detrend(
                        boot_df, outcome, "_boot_unit", time, pre_mask_b
                    )

                # Cross-sectional estimate
                post_mask_b = boot_df[time].isin(post_periods)  # type: ignore[union-attr, call-overload]
                post_b = boot_df.loc[post_mask_b]  # type: ignore[union-attr]
                unit_avg_b = post_b.groupby("_boot_unit")["_ydot"].mean()

                cs_b = boot_df.drop_duplicates(subset=["_boot_unit"], keep="first")[  # type: ignore[union-attr]
                    ["_boot_unit"]
                ].copy()
                if controls:
                    for c in controls:
                        cs_b[c] = boot_df.drop_duplicates(subset=["_boot_unit"], keep="first")[  # type: ignore[union-attr]
                            c
                        ].values

                # Map treatment status from group membership
                boot_treat_map = dict(zip(range(n_units), boot_treat_vec))
                cs_b["_treat"] = cs_b["_boot_unit"].map(boot_treat_map)
                cs_b["_ydot_avg"] = cs_b["_boot_unit"].map(unit_avg_b)
                cs_b = cs_b.dropna(subset=["_ydot_avg"])

                if len(cs_b) < 3:
                    boot_atts[b] = np.nan
                    continue

                y_b = cs_b["_ydot_avg"].values.astype(np.float64)
                treat_b = cs_b["_treat"].values.astype(np.float64)
                ctrl_b = cs_b[controls].values.astype(np.float64) if controls else None

                try:
                    att_b, _, _, _, _ = self._dispatch_estimator(
                        y_b, treat_b, ctrl_b, None, len(y_b)
                    )
                    boot_atts[b] = att_b
                except (np.linalg.LinAlgError, ValueError):
                    boot_atts[b] = np.nan
        else:
            # --- Parallel path (n_jobs > 1) ---
            from concurrent.futures import ThreadPoolExecutor

            warnings.warn(
                "Parallel bootstrap (n_jobs > 1) is experimental. "
                "ThreadPoolExecutor is used; speedup depends on "
                "GIL-releasing operations in numpy/scipy.",
                UserWarning,
                stacklevel=2,
            )

            # Pre-generate all bootstrap unit samples with deterministic seeds
            boot_unit_samples = []
            for b in range(self.n_bootstrap):
                rng_b = np.random.default_rng(seed=(self.bootstrap_seed or 0) + b)
                boot_treated = rng_b.choice(treated_arr, size=n_treated, replace=True)
                boot_control = rng_b.choice(control_arr, size=n_control, replace=True)
                boot_unit_samples.append(np.concatenate([boot_treated, boot_control]))

            def _run_replicate(b: int) -> float:
                """Execute a single bootstrap replicate."""
                boot_units = boot_unit_samples[b]

                # Build bootstrap sample (all periods for resampled units)
                boot_indices = []
                for u in boot_units:
                    idx = df.index[df[unit] == u].tolist()
                    boot_indices.extend(idx)

                boot_df = df.iloc[boot_indices].copy()
                repeat_counts = [unit_counts[u] for u in boot_units]
                boot_df["_boot_unit"] = np.repeat(np.arange(n_units), repeat_counts)

                boot_treat_vec = np.array([1.0] * n_treated + [0.0] * n_control, dtype=np.float64)

                # Apply transformation
                pre_mask_b = boot_df[time].isin(pre_periods)
                if self.rolling == "demean":
                    boot_df = self._transform_demean(boot_df, outcome, "_boot_unit", pre_mask_b)
                elif self.rolling == "detrend":
                    boot_df = self._transform_detrend(
                        boot_df, outcome, "_boot_unit", time, pre_mask_b
                    )
                elif self.rolling == "demeanq":
                    boot_df = self._transform_demeanq(
                        boot_df, outcome, "_boot_unit", time, pre_mask_b
                    )
                elif self.rolling == "detrendq":
                    boot_df = self._transform_detrendq(
                        boot_df, outcome, "_boot_unit", time, pre_mask_b
                    )
                else:
                    boot_df = self._transform_detrend(
                        boot_df, outcome, "_boot_unit", time, pre_mask_b
                    )

                # Cross-sectional estimate
                post_mask_b = boot_df[time].isin(post_periods)  # type: ignore[union-attr, call-overload]
                post_b = boot_df.loc[post_mask_b]  # type: ignore[union-attr]
                unit_avg_b = post_b.groupby("_boot_unit")["_ydot"].mean()

                cs_b = boot_df.drop_duplicates(subset=["_boot_unit"], keep="first")[  # type: ignore[union-attr]
                    ["_boot_unit"]
                ].copy()
                if controls:
                    for c in controls:
                        cs_b[c] = boot_df.drop_duplicates(subset=["_boot_unit"], keep="first")[  # type: ignore[union-attr]
                            c
                        ].values

                boot_treat_map = dict(zip(range(n_units), boot_treat_vec))
                cs_b["_treat"] = cs_b["_boot_unit"].map(boot_treat_map)
                cs_b["_ydot_avg"] = cs_b["_boot_unit"].map(unit_avg_b)
                cs_b = cs_b.dropna(subset=["_ydot_avg"])

                if len(cs_b) < 3:
                    return np.nan

                y_b = cs_b["_ydot_avg"].values.astype(np.float64)
                treat_b = cs_b["_treat"].values.astype(np.float64)
                ctrl_b = cs_b[controls].values.astype(np.float64) if controls else None

                try:
                    att_b, _, _, _, _ = self._dispatch_estimator(
                        y_b, treat_b, ctrl_b, None, len(y_b)
                    )
                    return att_b
                except (np.linalg.LinAlgError, ValueError):
                    return np.nan

            with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                boot_atts = np.array(list(executor.map(_run_replicate, range(self.n_bootstrap))))

        # Compute bootstrap SE
        n_failed = int(np.isnan(boot_atts).sum())
        if n_failed > 0:
            warnings.warn(
                f"LWDiD bootstrap: {n_failed}/{self.n_bootstrap} replication(s) failed "
                f"(returned NaN). Results based on {self.n_bootstrap - n_failed} valid replications.",
                UserWarning,
                stacklevel=2,
            )
        valid_boots = boot_atts[np.isfinite(boot_atts)]
        if len(valid_boots) < 2:
            se = np.nan
        else:
            se = float(np.std(valid_boots, ddof=1))

        t_stat, p_value, conf_int = safe_inference(
            att_full, se, alpha=self.alpha, df=max(len(y_full) - n_params_full, 1)
        )

        return att_full, se, t_stat, p_value, conf_int

    def _estimate_period_effects(
        self,
        df: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        post_periods: List[Any],
        treated_set: set,
        controls: List[str],
        cluster: Optional[str],
        has_controls: bool,
    ) -> Dict[Any, Dict]:
        """Estimate separate ATT for each post-treatment period.

        For each post-period t, takes the cross-section of transformed
        outcomes at time t and estimates ATT on that single period.

        Parameters
        ----------
        df : pd.DataFrame
            Panel data with '_ydot' column already computed.
        outcome : str
            Outcome variable column name.
        unit : str
            Unit identifier column.
        time : str
            Time period column.
        post_periods : list
            List of post-treatment period values.
        treated_set : set
            Set of treated unit identifiers.
        controls : list of str
            Control variable columns.
        cluster : str or None
            Cluster variable name.
        has_controls : bool
            Whether controls were provided.

        Returns
        -------
        dict
            Mapping from period to dict with 'att', 'se', 't_stat',
            'p_value', 'conf_int', 'n_obs'.
        """
        period_effects: Dict[Any, Dict] = {}

        for t in sorted(post_periods):
            # Take cross-section at period t
            t_mask = df[time] == t
            t_df = df.loc[t_mask].copy()

            if len(t_df) == 0:
                continue

            y_t = t_df["_ydot"].values.astype(np.float64)
            treat_t = t_df[unit].isin(treated_set).astype(float).values

            # Filter NaN before estimation
            finite_mask = np.isfinite(y_t)
            if not finite_mask.any():
                period_effects[t] = {
                    "att": np.nan,
                    "se": np.nan,
                    "t_stat": np.nan,
                    "p_value": np.nan,
                    "conf_int": (np.nan, np.nan),
                    "n_obs": len(y_t),
                }
                continue
            y_t = y_t[finite_mask]
            treat_t = treat_t[finite_mask]

            n_obs_t = len(y_t)
            n_treated_t = int(treat_t.sum())

            if n_treated_t == 0 or n_treated_t == n_obs_t:
                continue

            controls_matrix_t = None
            if has_controls and controls:
                controls_matrix_t = t_df[controls].values.astype(np.float64)
                controls_matrix_t = controls_matrix_t[finite_mask]

            cluster_ids_t = None
            if cluster is not None and self.vce == "cluster":
                cluster_ids_t = t_df[cluster].values
                cluster_ids_t = cluster_ids_t[finite_mask]

            try:
                att_t, se_t, _, _, n_params_t = self._dispatch_estimator(
                    y_t, treat_t, controls_matrix_t, cluster_ids_t, n_obs_t
                )
                df_t = max(n_obs_t - n_params_t, 1)
                t_stat_t, p_value_t, conf_int_t = safe_inference(
                    att_t, se_t, alpha=self.alpha, df=df_t
                )
                period_effects[t] = {
                    "att": att_t,
                    "se": se_t,
                    "t_stat": t_stat_t,
                    "p_value": p_value_t,
                    "conf_int": conf_int_t,
                    "n_obs": n_obs_t,
                }
            except (np.linalg.LinAlgError, ValueError):
                period_effects[t] = {
                    "att": np.nan,
                    "se": np.nan,
                    "t_stat": np.nan,
                    "p_value": np.nan,
                    "conf_int": (np.nan, np.nan),
                    "n_obs": n_obs_t,
                }

        return period_effects if period_effects else None

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default True
            If True, return parameters for sub-objects. Not used here
            but included for sklearn compatibility.

        Returns
        -------
        dict
            Parameter names mapped to their values.
        """
        return {
            "rolling": self.rolling,
            "estimator": self.estimator,
            "vce": self.vce,
            "control_group": self.control_group,
            "alpha": self.alpha,
            "n_bootstrap": self.n_bootstrap,
            "period_specific": self.period_specific,
            "bootstrap_seed": self.bootstrap_seed,
            "trim_threshold": self.trim_threshold,
            "n_neighbors": self.n_neighbors,
            "caliper": self.caliper,
            "with_replacement": self.with_replacement,
            "n_jobs": self.n_jobs,
        }

    def set_params(self, **params: Any) -> LWDiD:
        """Set parameters on this estimator.

        Parameters
        ----------
        **params : dict
            Estimator parameters to update.

        Returns
        -------
        self
            The estimator instance.

        Raises
        ------
        ValueError
            If any parameter name is invalid or value is out of range.
        """
        valid_params = self.get_params()
        # Phase 1: Validate all key names BEFORE any state change
        for key in params:
            if key not in valid_params:
                raise ValueError(
                    f"Invalid parameter '{key}' for LWDiD. "
                    f"Valid parameters: {list(valid_params.keys())}"
                )

        # Phase 2: Save old values and apply new ones
        old_values = {key: getattr(self, key) for key in params}
        for key, value in params.items():
            setattr(self, key, value)

        # Re-validate after setting; rollback on failure
        try:
            if self.rolling not in _VALID_ROLLING:
                raise ValueError(f"rolling must be one of {_VALID_ROLLING}, got '{self.rolling}'")
            if self.estimator not in _VALID_ESTIMATORS:
                raise ValueError(
                    f"estimator must be one of {_VALID_ESTIMATORS}, " f"got '{self.estimator}'"
                )
            if self.vce not in _VALID_VCE:
                raise ValueError(f"vce must be one of {_VALID_VCE}, got '{self.vce}'")
            if self.control_group not in _VALID_CONTROL_GROUPS:
                raise ValueError(
                    f"control_group must be one of "
                    f"{_VALID_CONTROL_GROUPS}, got '{self.control_group}'"
                )
            if not (0 < self.alpha < 1):
                raise ValueError(f"alpha must be in (0, 1), got {self.alpha}")
            if not isinstance(self.n_bootstrap, (int, np.integer)) or self.n_bootstrap < 0:
                raise ValueError(
                    f"n_bootstrap must be a non-negative integer, " f"got {self.n_bootstrap}"
                )
            if not (0.0 < self.trim_threshold < 0.5):
                raise ValueError(f"trim_threshold must be in (0, 0.5), got {self.trim_threshold}")
            if self.n_neighbors < 1:
                raise ValueError(f"n_neighbors must be >= 1, got {self.n_neighbors}")
            if not isinstance(self.n_jobs, (int, np.integer)) or self.n_jobs < 1:
                raise ValueError(f"n_jobs must be a positive integer, got {self.n_jobs}")
        except (ValueError, TypeError):
            # Rollback to old values on validation failure
            for key, old_val in old_values.items():
                setattr(self, key, old_val)
            raise

        return self

    def __repr__(self) -> str:
        """Return string representation of the estimator."""
        params = self.get_params()
        params_str = ", ".join(f"{k}={v!r}" for k, v in params.items())
        return f"LWDiD({params_str})"


def lwdid(
    data,
    y="y",
    d="d",
    ivar="unit",
    tvar="time",
    post="post",
    gvar=None,
    rolling="demean",
    estimator="ra",
    vce=None,
    controls=None,
    control_group="not_yet_treated",
    cluster=None,
    alpha=0.05,
    n_bootstrap=0,
    **kwargs,
) -> "LWDiDResults":
    """Functional interface to LWDiD (compatible with lwdid-py calling convention).

    This is a convenience wrapper that maps lwdid-py parameter names to
    diff-diff's LWDiD class interface, enabling near-zero-effort migration
    from lwdid-py code.

    Parameters
    ----------
    data : pd.DataFrame
        Panel dataset.
    y : str
        Outcome column name.
    d : str
        Ever-treated indicator column name (1 for treated units, 0 for control).
    ivar : str
        Unit identifier column name.
    tvar : str
        Time variable column name.
    post : str
        Post-treatment period indicator column name (for common timing).
    gvar : str or None
        Cohort variable for staggered adoption (NaN or 0 = never-treated).
    rolling : str
        Transformation method ('demean', 'detrend', 'demeanq', 'detrendq').
    estimator : str
        Estimation method ('ra', 'ipw', 'ipwra', 'psm').
    vce : str or None
        Variance estimator (None='classical', 'hc1', 'hc3', 'cluster', etc.).
    controls : list of str or None
        Control variable column names.
    control_group : str
        Control group strategy ('never_treated' or 'not_yet_treated').
    cluster : str or None
        Cluster variable column name.
    alpha : float
        Significance level.
    n_bootstrap : int
        Number of bootstrap replications (0 = analytical inference).

    Returns
    -------
    LWDiDResults
        Estimation results.

    Examples
    --------
    >>> from diff_diff import lwdid
    >>> result = lwdid(df, y='outcome', d='treated', ivar='id', tvar='year', post='post_period')
    >>> print(result.att, result.se)
    """

    # Handle lwdid-py parameter alias: cluster_var -> cluster
    if cluster is None and "cluster_var" in kwargs:
        cluster = kwargs.pop("cluster_var")

    # Reject unknown keyword arguments
    _valid_lwdid_params = set(LWDiD().get_params().keys())
    unknown = {k for k in kwargs if k not in _valid_lwdid_params}
    if unknown:
        raise ValueError(
            f"lwdid() received unexpected keyword arguments: {sorted(unknown)}. "
            f"Check parameter names — did you mean one of: "
            f"{sorted(_valid_lwdid_params)}?"
        )

    # Map VCE (handle lwdid-py aliases)
    _vce_aliases = {"robust": "hc1", "ols": "classical", None: "classical"}
    vce_dd = _vce_aliases.get(vce, vce) if vce in _vce_aliases else vce

    # Build treatment column: d * post for common timing
    if gvar is None:
        # Common timing: treatment = ever_treated * post_period
        treatment_col = "_lwdid_treat"
        data = data.copy()
        if post is None or post not in data.columns:
            # If no post column, infer from treatment timing:
            # post = 1 for periods where any unit is treated
            # Requires 'd' to be ever-treated and some way to identify post
            raise ValueError(
                f"For common-timing designs (gvar=None), a 'post' column is required. "
                f"Either provide post='{post}' column in the data, or use "
                f"gvar for staggered designs. "
                f"Available columns: {list(data.columns)}"
            )
        data[treatment_col] = (data[d].astype(int) * data[post].astype(int)).astype(int)
    else:
        # Staggered: treatment derived from cohort timing
        treatment_col = "_lwdid_treat"
        data = data.copy()
        cohort_vals = data[gvar].fillna(0)
        data[treatment_col] = ((cohort_vals > 0) & (data[tvar] >= cohort_vals)).astype(int)

    # Instantiate and fit
    est = LWDiD(
        rolling=rolling,
        estimator=estimator,
        vce=vce_dd,
        control_group=control_group,
        alpha=alpha,
        n_bootstrap=n_bootstrap,
        **{k: v for k, v in kwargs.items() if k in LWDiD().get_params()},
    )

    cohort_col = gvar if gvar is not None else None

    return est.fit(
        data,
        outcome=y,
        unit=ivar,
        time=tvar,
        treatment=treatment_col,
        cohort=cohort_col,
        controls=controls,
        cluster=cluster,
    )


def validate_staggered_data(data, unit, time, cohort) -> Dict[str, Any]:
    """Validate panel data structure for staggered DiD estimation.

    Checks:
    - Panel is complete (all unit×time combinations exist)
    - Cohort is time-invariant within units
    - At least one never-treated group exists (cohort==0)
    - No missing values in key columns

    Parameters
    ----------
    data : pd.DataFrame
        Panel dataset.
    unit : str
        Unit identifier column name.
    time : str
        Time period column name.
    cohort : str
        Cohort column name (0 or NaN = never-treated).

    Returns
    -------
    dict
        Validation results with keys: 'valid', 'warnings', 'errors',
        'n_units', 'n_periods', 'n_cohorts', 'n_never_treated'.

    Raises
    ------
    ValueError
        If data structure is fundamentally invalid.
    """

    df = data.copy()

    results: dict[str, Any] = {"valid": True, "warnings": [], "errors": []}

    # Check required columns exist
    for col in [unit, time, cohort]:
        if col not in df.columns:
            results["valid"] = False
            results["errors"].append(f"Column '{col}' not found in data")
            return results

    # Check cohort time-invariance
    cohort_per_unit = df.groupby(unit)[cohort].nunique()
    varying = cohort_per_unit[cohort_per_unit > 1]
    if len(varying) > 0:
        results["valid"] = False
        results["errors"].append(f"{len(varying)} units have time-varying cohort values")

    # Check for never-treated
    never_treated = df[df[cohort] == 0][unit].nunique()
    if never_treated == 0:
        results["warnings"].append("No never-treated units found (cohort==0)")

    # Check panel balance
    n_units = df[unit].nunique()
    n_times = df[time].nunique()
    expected_rows = n_units * n_times
    if len(df) != expected_rows:
        results["warnings"].append(f"Unbalanced panel: {len(df)} rows vs {expected_rows} expected")

    # Check missing values
    for col in [unit, time, cohort]:
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            results["warnings"].append(f"{n_missing} missing values in '{col}'")

    results["n_units"] = n_units
    results["n_periods"] = n_times
    results["n_cohorts"] = df[df[cohort] > 0][cohort].nunique()
    results["n_never_treated"] = never_treated

    return results


def is_never_treated(data, unit, cohort) -> np.ndarray:
    """Identify never-treated units in staggered design.

    Parameters
    ----------
    data : pd.DataFrame
        Panel dataset.
    unit : str
        Unit identifier column name.
    cohort : str
        Cohort column name (0 = never treated).

    Returns
    -------
    np.ndarray of bool
        True for never-treated units (one entry per unique unit).
    """
    unit_cohort = data.groupby(unit)[cohort].first()
    return np.array((unit_cohort == 0) | unit_cohort.isna())
