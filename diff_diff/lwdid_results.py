"""Results class for the LWDiD (Lee & Wooldridge 2025, 2026) estimator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class LWDiDResults:
    """Results from LWDiD.fit().

    Follows the diff-diff standard results interface. Holds the headline ATT
    estimate and inference for the common-timing case, or per-cohort effects
    and an overall weighted ATT for the staggered case.

    Parameters
    ----------
    att : float
        Average treatment effect on the treated.
    se : float
        Standard error of the ATT estimate.
    t_stat : float
        t-statistic (att / se).
    p_value : float
        Two-sided p-value.
    conf_int : tuple of float
        (lower, upper) confidence interval at level ``1 - alpha``.
    n_obs : int
        Total observations used in estimation.
    n_treated : int
        Number of treated units.
    n_control : int
        Number of control units.
    rolling : str
        Transformation method used ('demean', 'detrend', 'demeanq', or 'detrendq').
    estimator : str
        Estimation method ('ra', 'ipw', 'ipwra', or 'psm').
    vce_type : str
        Variance estimator ('classical', 'hc0', 'hc1', 'hc2', 'hc3', 'hc4', or 'cluster').
    alpha : float
        Significance level used for confidence intervals.
    df_inference : int or None
        Degrees of freedom used for t-distribution inference.
    cluster_name : str or None
        Name of the cluster variable, if clustered.
    n_clusters : int or None
        Number of clusters, if clustered.
    cohort_effects : dict or None
        Per-cohort ATT results for staggered designs.
    overall_att : dict or None
        Weighted overall ATT across cohorts for staggered designs.
    period_effects : dict or None
        Per-period ATT results for common-timing designs with period_specific=True.
    params : ndarray or None
        All coefficient estimates from the regression.
    bse : ndarray or None
        All standard errors from the regression.
    vcov : ndarray or None
        Variance-covariance matrix.
    """

    # ------------------------------------------------------------------ #
    # Core inference fields                                               #
    # ------------------------------------------------------------------ #
    att: float
    se: float
    t_stat: float
    p_value: float
    conf_int: Tuple[float, float]

    # ------------------------------------------------------------------ #
    # Sample information                                                  #
    # ------------------------------------------------------------------ #
    n_obs: int
    n_treated: int
    n_control: int

    # ------------------------------------------------------------------ #
    # Method metadata                                                     #
    # ------------------------------------------------------------------ #
    rolling: str
    estimator: str
    vce_type: str
    alpha: float
    df_inference: Optional[int] = None
    cluster_name: Optional[str] = None
    n_clusters: Optional[int] = None

    # ------------------------------------------------------------------ #
    # Staggered-specific (optional)                                       #
    # ------------------------------------------------------------------ #
    cohort_effects: Optional[Dict[Any, Dict]] = field(default=None, repr=False)
    overall_att: Optional[Dict] = field(default=None, repr=False)

    # ------------------------------------------------------------------ #
    # Period-specific effects (optional)                                  #
    # ------------------------------------------------------------------ #
    period_effects: Optional[Dict[Any, Dict]] = field(default=None, repr=False)

    # ------------------------------------------------------------------ #
    # Full regression output (optional)                                   #
    # ------------------------------------------------------------------ #
    params: Optional[np.ndarray] = field(default=None, repr=False)
    bse: Optional[np.ndarray] = field(default=None, repr=False)
    vcov: Optional[np.ndarray] = field(default=None, repr=False)

    # ------------------------------------------------------------------ #
    # Cached RI/WCB results (optional)                                    #
    # ------------------------------------------------------------------ #
    _ri_result: Optional[Any] = field(default=None, repr=False)
    _wcb_result: Optional[Any] = field(default=None, repr=False)

    # ------------------------------------------------------------------ #
    # Properties                                                          #
    # ------------------------------------------------------------------ #
    @property
    def pvalue(self) -> float:
        """Alias for p_value (diff-diff API convention)."""
        return self.p_value

    @property
    def ci(self) -> Tuple[float, float]:
        """Alias for conf_int (diff-diff API convention)."""
        return self.conf_int

    @property
    def is_staggered(self) -> bool:
        """Whether this result comes from a staggered adoption design."""
        return self.cohort_effects is not None

    @property
    def has_period_effects(self) -> bool:
        """Whether period-specific effects are available."""
        return self.period_effects is not None and len(self.period_effects) > 0

    # ------------------------------------------------------------------ #
    # Serialization                                                       #
    # ------------------------------------------------------------------ #
    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            For common timing: a single-row DataFrame (plus period rows if available).
            For staggered: one row per cohort plus an "Overall" row.
        """
        if not self.is_staggered:
            rows: List[Dict[str, Any]] = [
                {
                    "term": "ATT",
                    "att": self.att,
                    "se": self.se,
                    "t_stat": self.t_stat,
                    "p_value": self.p_value,
                    "ci_lower": self.conf_int[0],
                    "ci_upper": self.conf_int[1],
                    "n_obs": self.n_obs,
                    "n_treated": self.n_treated,
                    "n_control": self.n_control,
                    "rolling": self.rolling,
                    "estimator": self.estimator,
                    "vce_type": self.vce_type,
                }
            ]
            if self.has_period_effects:
                for period, eff in sorted(self.period_effects.items()):
                    ci = eff.get("conf_int", (np.nan, np.nan))
                    rows.append(
                        {
                            "term": f"Period {period}",
                            "att": eff.get("att", np.nan),
                            "se": eff.get("se", np.nan),
                            "t_stat": eff.get("t_stat", np.nan),
                            "p_value": eff.get("p_value", np.nan),
                            "ci_lower": ci[0] if ci else np.nan,
                            "ci_upper": ci[1] if ci else np.nan,
                            "n_obs": eff.get("n_obs", 0),
                            "n_treated": None,
                            "n_control": None,
                            "rolling": self.rolling,
                            "estimator": self.estimator,
                            "vce_type": self.vce_type,
                        }
                    )
            return pd.DataFrame(rows)

        rows: List[Dict[str, Any]] = []
        for cohort, eff in self.cohort_effects.items():
            ci = eff.get("conf_int", (np.nan, np.nan))
            n_t = eff.get("n_treated", 0)
            n_c = eff.get("n_control", 0)
            rows.append(
                {
                    "cohort": cohort,
                    "att": eff.get("att", np.nan),
                    "se": eff.get("se", np.nan),
                    "t_stat": eff.get("t_stat", np.nan),
                    "p_value": eff.get("p_value", np.nan),
                    "ci_lower": ci[0] if ci else np.nan,
                    "ci_upper": ci[1] if ci else np.nan,
                    "n_treated": n_t,
                    "n_control": n_c,
                }
            )
        # Append overall row
        rows.append(
            {
                "cohort": "Overall",
                "att": self.att,
                "se": self.se,
                "t_stat": self.t_stat,
                "p_value": self.p_value,
                "ci_lower": self.conf_int[0],
                "ci_upper": self.conf_int[1],
                "n_treated": self.n_treated,
                "n_control": self.n_control,
            }
        )
        return pd.DataFrame(rows)

    def to_dict(self) -> Dict[str, Any]:
        """Convert results to a JSON-serializable dictionary.

        Returns
        -------
        dict
            All scalar results and metadata. Arrays are converted to lists.
        """
        result: Dict[str, Any] = {
            "att": self.att,
            "se": self.se,
            "t_stat": self.t_stat,
            "p_value": self.p_value,
            "conf_int_lower": self.conf_int[0],
            "conf_int_upper": self.conf_int[1],
            "n_obs": self.n_obs,
            "n_treated": self.n_treated,
            "n_control": self.n_control,
            "rolling": self.rolling,
            "estimator": self.estimator,
            "vce_type": self.vce_type,
            "alpha": self.alpha,
        }
        if self.cluster_name is not None:
            result["cluster_name"] = self.cluster_name
        if self.n_clusters is not None:
            result["n_clusters"] = self.n_clusters
        if self.cohort_effects is not None:
            result["cohort_effects"] = {str(k): v for k, v in self.cohort_effects.items()}
        if self.overall_att is not None:
            result["overall_att"] = self.overall_att
        if self.params is not None:
            result["params"] = self.params.tolist()
        if self.bse is not None:
            result["bse"] = self.bse.tolist()
        if self.period_effects is not None:
            result["period_effects"] = {str(k): v for k, v in self.period_effects.items()}
        return result

    # ------------------------------------------------------------------ #
    # Aggregation                                                         #
    # ------------------------------------------------------------------ #
    def to_csv(self, path: str) -> None:
        """Export results to CSV file.

        Parameters
        ----------
        path : str
            File path for the CSV output.
        """
        self.to_dataframe().to_csv(path, index=False)

    def to_latex(self, path: Optional[str] = None) -> str:
        """Export results as LaTeX table.

        Parameters
        ----------
        path : str or None, default None
            If provided, write LaTeX to this file path.

        Returns
        -------
        str
            LaTeX table string.
        """
        df = self.to_dataframe()
        latex_str = df.to_latex(index=False, float_format="%.4f")
        if path is not None:
            with open(path, "w") as f:
                f.write(latex_str)
        return latex_str

    def aggregate(self, by: str = "overall") -> LWDiDResults:
        """Aggregate staggered cohort effects to a single ATT.

        Parameters
        ----------
        by : str, default 'overall'
            Aggregation method. Currently supports 'overall' (weighted average
            across cohorts using cohort sample sizes).

        Returns
        -------
        LWDiDResults
            New results object with the aggregated ATT.

        Raises
        ------
        ValueError
            If called on non-staggered results or with an unsupported method.
        """
        if not self.is_staggered:
            raise ValueError(
                "aggregate() is only available for staggered results " "with cohort_effects."
            )
        if by != "overall":
            raise ValueError(f"Unsupported aggregation method: {by!r}")

        cohorts = self.cohort_effects
        atts = []
        weights = []
        for cohort, eff in cohorts.items():
            att_c = eff.get("att", np.nan)
            n_c = eff.get("n_treated", 1)
            if not np.isnan(att_c):
                atts.append(att_c)
                weights.append(n_c)

        if not atts:
            return LWDiDResults(
                att=np.nan,
                se=np.nan,
                t_stat=np.nan,
                p_value=np.nan,
                conf_int=(np.nan, np.nan),
                n_obs=self.n_obs,
                n_treated=self.n_treated,
                n_control=self.n_control,
                rolling=self.rolling,
                estimator=self.estimator,
                vce_type=self.vce_type,
                alpha=self.alpha,
                cluster_name=self.cluster_name,
                n_clusters=self.n_clusters,
            )

        w = np.array(weights, dtype=float)
        w = w / w.sum()
        a = np.array(atts, dtype=float)
        agg_att = float(np.dot(w, a))

        # Aggregate SEs via delta method (independence across cohorts)
        # Exclude cohorts with NaN or non-positive SE from aggregation
        valid_mask = []
        for i, (cohort, eff) in enumerate(
            (c, e) for c, e in cohorts.items() if not np.isnan(e.get("att", np.nan))
        ):
            se_c = eff.get("se", np.nan)
            valid_mask.append(np.isfinite(se_c) and se_c > 0)

        valid_mask = np.array(valid_mask, dtype=bool)
        if not valid_mask.any():
            agg_se = np.nan
            agg_t = np.nan
            agg_p = np.nan
            agg_ci = (np.nan, np.nan)
        else:
            # Re-normalize weights for valid SEs only
            ses = []
            for cohort, eff in cohorts.items():
                se_c = eff.get("se", np.nan)
                if not np.isnan(eff.get("att", np.nan)):
                    ses.append(se_c)
            se_arr = np.array(ses, dtype=float)

            # Only use valid (finite, positive) SEs
            w_valid = w[valid_mask]
            w_valid = w_valid / w_valid.sum()
            se_valid = se_arr[valid_mask]
            agg_se = float(np.sqrt(np.dot(w_valid**2, se_valid**2)))

            # Use safe_inference with t-distribution for proper inference
            from diff_diff.utils import safe_inference

            # Use sum of cluster counts or residual df for aggregation
            _agg_df = (
                max(int(valid_mask.sum()) - 1, 1)
                if self.n_clusters is None
                else max(self.n_clusters - 1, 1)
            )
            agg_t, agg_p, agg_ci = safe_inference(agg_att, agg_se, alpha=self.alpha, df=_agg_df)

        return LWDiDResults(
            att=agg_att,
            se=agg_se,
            t_stat=agg_t,
            p_value=agg_p,
            conf_int=agg_ci,
            n_obs=self.n_obs,
            n_treated=self.n_treated,
            n_control=self.n_control,
            rolling=self.rolling,
            estimator=self.estimator,
            vce_type=self.vce_type,
            alpha=self.alpha,
            cluster_name=self.cluster_name,
            n_clusters=self.n_clusters,
            cohort_effects=self.cohort_effects,
            overall_att={
                "att": agg_att,
                "se": agg_se,
                "t_stat": agg_t,
                "p_value": agg_p,
                "conf_int": agg_ci,
            },
        )

    # ------------------------------------------------------------------ #
    # Text summary                                                        #
    # ------------------------------------------------------------------ #
    def summary(self) -> str:
        """Formatted text summary of results.

        Returns
        -------
        str
            Human-readable summary table.
        """
        from diff_diff.results import _format_vcov_label, _get_significance_stars

        ci_pct = int(round((1 - self.alpha) * 100))
        width = 88
        bar = "=" * width
        dash = "-" * width

        def _fmt(x: Any, nd: int = 4) -> str:
            try:
                xf = float(x)
            except (TypeError, ValueError):
                return ""
            return "" if np.isnan(xf) else f"{xf:.{nd}f}"

        lines: List[str] = [
            bar,
            "Lee & Wooldridge DiD (LWDiD) Results".center(width),
            bar,
            f"Observations: {self.n_obs}    "
            f"Treated units: {self.n_treated}    "
            f"Control units: {self.n_control}",
            f"Rolling: {self.rolling}    "
            f"Estimator: {self.estimator}    "
            f"Alpha: {self.alpha}",
        ]

        # Variance label
        vcov_label = _format_vcov_label(
            self.vce_type,
            cluster_name=self.cluster_name,
            n_clusters=self.n_clusters,
            n_obs=self.n_obs,
        )
        if vcov_label:
            lines.append(f"Std. errors: {vcov_label}")

        # Header for results table
        header = (
            f"{'':>12}  {'Estimate':>10}  {'Std.Err':>10}  {'t':>8}  "
            f"{'P>|t|':>8}  [{ci_pct}% Conf. Int.]"
        )

        # Main ATT row
        lines.append("")
        if self.is_staggered:
            lines.append("Cohort-level effects:")
            lines.append(dash)
            lines.append(header)
            lines.append(dash)
            for cohort, eff in self.cohort_effects.items():
                ci = eff.get("conf_int", (np.nan, np.nan))
                p = eff.get("p_value", np.nan)
                stars = "" if np.isnan(p) else _get_significance_stars(float(p))
                label = f"G={cohort}"
                lines.append(
                    f"{label:>12}  {_fmt(eff.get('att')):>10}  "
                    f"{_fmt(eff.get('se')):>10}  "
                    f"{_fmt(eff.get('t_stat'), 2):>8}  "
                    f"{_fmt(p, 3):>8}  "
                    f"[{_fmt(ci[0]):>9}, {_fmt(ci[1]):>9}] {stars}"
                )
            lines.append(dash)
            # Overall ATT
            stars = _get_significance_stars(self.p_value) if not np.isnan(self.p_value) else ""
            lines.append(
                f"{'Overall ATT':>12}  {_fmt(self.att):>10}  "
                f"{_fmt(self.se):>10}  "
                f"{_fmt(self.t_stat, 2):>8}  "
                f"{_fmt(self.p_value, 3):>8}  "
                f"[{_fmt(self.conf_int[0]):>9}, {_fmt(self.conf_int[1]):>9}] {stars}"
            )
        else:
            lines.append("ATT estimate:")
            lines.append(dash)
            lines.append(header)
            lines.append(dash)
            stars = _get_significance_stars(self.p_value) if not np.isnan(self.p_value) else ""
            lines.append(
                f"{'ATT':>12}  {_fmt(self.att):>10}  "
                f"{_fmt(self.se):>10}  "
                f"{_fmt(self.t_stat, 2):>8}  "
                f"{_fmt(self.p_value, 3):>8}  "
                f"[{_fmt(self.conf_int[0]):>9}, {_fmt(self.conf_int[1]):>9}] {stars}"
            )
            # Period-specific effects
            if self.has_period_effects:
                lines.append("")
                lines.append("Period-specific effects:")
                lines.append(dash)
                lines.append(header)
                lines.append(dash)
                for period, eff in sorted(self.period_effects.items()):
                    ci = eff.get("conf_int", (np.nan, np.nan))
                    p = eff.get("p_value", np.nan)
                    stars_p = "" if np.isnan(p) else _get_significance_stars(float(p))
                    label = f"t={period}"
                    lines.append(
                        f"{label:>12}  {_fmt(eff.get('att')):>10}  "
                        f"{_fmt(eff.get('se')):>10}  "
                        f"{_fmt(eff.get('t_stat'), 2):>8}  "
                        f"{_fmt(p, 3):>8}  "
                        f"[{_fmt(ci[0]):>9}, {_fmt(ci[1]):>9}] {stars_p}"
                    )

        lines.append(bar)
        lines.append("Signif. codes: *** p<0.001, ** p<0.01, * p<0.05")
        return "\n".join(lines)

    def print_summary(self) -> None:
        """Print the formatted summary to stdout."""
        print(self.summary())

    # ================================================================
    # Advanced inference and diagnostics (delegate to standalone modules)
    # ================================================================

    @property
    def ri_pvalue(self):
        """Randomization inference p-value (None if not computed)."""
        if self._ri_result is not None:
            return self._ri_result.pvalue
        return None

    @property
    def bootstrap_pvalue(self):
        """Wild cluster bootstrap p-value (None if not computed)."""
        if self._wcb_result is not None:
            return self._wcb_result.pvalue
        return None

    def wild_cluster_bootstrap(
        self,
        y,
        treatment,
        cluster_ids,
        controls=None,
        n_reps=999,
        weight_type="rademacher",
        seed=None,
    ):
        """Run wild cluster bootstrap inference on the fitted results.

        Delegates to diff_diff.lwdid_wild_bootstrap.wild_cluster_bootstrap().
        Result is cached and accessible via the `bootstrap_pvalue` property.
        """
        from diff_diff.lwdid_wild_bootstrap import wild_cluster_bootstrap as _wcb

        result = _wcb(
            y,
            treatment,
            cluster_ids,
            controls=controls,
            n_reps=n_reps,
            weight_type=weight_type,
            seed=seed,
        )
        object.__setattr__(self, "_wcb_result", result)
        return result

    def randomization_test(
        self, y, treatment, controls=None, n_reps=1000, method="permutation", seed=None
    ):
        """Run Fisher randomization inference on the fitted results.

        Delegates to diff_diff.lwdid_randomization.randomization_inference().
        Result is cached and accessible via the `ri_pvalue` property.
        """
        from diff_diff.lwdid_randomization import randomization_inference as _ri

        result = _ri(y, treatment, controls=controls, n_reps=n_reps, method=method, seed=seed)
        object.__setattr__(self, "_ri_result", result)
        return result

    # ------------------------------------------------------------------ #
    # Repr                                                                #
    # ------------------------------------------------------------------ #
    def __repr__(self) -> str:
        cluster = f", cluster={self.cluster_name}, G={self.n_clusters}" if self.cluster_name else ""
        att_s = "nan" if np.isnan(self.att) else f"{self.att:.4f}"
        se_s = "nan" if np.isnan(self.se) else f"{self.se:.4f}"
        stag = ", staggered=True" if self.is_staggered else ""
        return (
            f"LWDiDResults("
            f"ATT={att_s}, SE={se_s}, "
            f"rolling={self.rolling!r}, estimator={self.estimator!r}, "
            f"vce={self.vce_type!r}{cluster}{stag})"
        )
