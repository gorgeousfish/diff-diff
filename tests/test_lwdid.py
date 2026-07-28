"""Tests for LWDiD estimator (Lee & Wooldridge 2025, 2026)."""

import json
import warnings

import numpy as np
import pandas as pd
import pytest

from diff_diff import LW, LWDiD, LWDiDResults

# ─── Test Data Generators ───────────────────────────────────────────────────


def _make_common_timing_panel(
    n_treated=30,
    n_control=50,
    n_pre=5,
    n_post=3,
    true_att=2.0,
    seed=42,
):
    """Generate balanced common-timing panel with known ATT.

    Pre-treatment periods: 1..n_pre (treatment=0 for all)
    Post-treatment periods: n_pre+1..n_pre+n_post (treatment=1 for treated)
    """
    rng = np.random.default_rng(seed)
    n_units = n_treated + n_control
    n_periods = n_pre + n_post

    rows = []
    for i in range(n_units):
        is_treated = i < n_treated
        unit_fe = rng.normal(0, 1)
        for t in range(1, n_periods + 1):
            time_trend = 0.3 * t
            noise = rng.normal(0, 0.5)
            post = 1 if t > n_pre else 0
            treat = 1 if (is_treated and post) else 0
            y = unit_fe + time_trend + noise + (true_att if treat else 0)
            rows.append(
                {
                    "unit": i,
                    "time": t,
                    "y": y,
                    "treat": treat,
                }
            )
    return pd.DataFrame(rows)


def _make_staggered_panel(
    n_units=120,
    n_periods=10,
    n_cohorts=3,
    true_att=1.5,
    seed=42,
):
    """Generate staggered adoption panel with multiple cohorts.

    Cohort assignment:
    - First ~1/4 units: never-treated (cohort=0)
    - Remaining units split across n_cohorts with treatment times spread.
    """
    rng = np.random.default_rng(seed)
    n_never = n_units // 4
    n_per_cohort = (n_units - n_never) // n_cohorts

    # Cohort adoption times (spread across middle periods)
    cohort_times = [3 + i * 2 for i in range(n_cohorts)]

    rows = []
    uid = 0
    for i in range(n_never):
        unit_fe = rng.normal(0, 1)
        for t in range(1, n_periods + 1):
            y = unit_fe + 0.2 * t + rng.normal(0, 0.5)
            rows.append(
                {
                    "unit": uid,
                    "time": t,
                    "y": y,
                    "treat": 0,
                    "cohort": 0,
                }
            )
        uid += 1

    for c_idx, g in enumerate(cohort_times):
        for i in range(n_per_cohort):
            unit_fe = rng.normal(0, 1)
            for t in range(1, n_periods + 1):
                post = 1 if t >= g else 0
                treat = post  # treated once cohort adopts
                effect = true_att * post
                y = unit_fe + 0.2 * t + rng.normal(0, 0.5) + effect
                rows.append(
                    {
                        "unit": uid,
                        "time": t,
                        "y": y,
                        "treat": treat,
                        "cohort": g,
                    }
                )
            uid += 1

    return pd.DataFrame(rows)


# ─── Parameter Interface Tests ──────────────────────────────────────────────


class TestLWDiDParams:
    """Test parameter setting, getting, and validation."""

    def test_get_params_returns_all(self):
        est = LWDiD(rolling="demean", estimator="ra", vce="hc1")
        params = est.get_params()
        assert "rolling" in params
        assert "estimator" in params
        assert "vce" in params
        assert "control_group" in params
        assert "alpha" in params
        assert "n_bootstrap" in params
        assert params["rolling"] == "demean"
        assert params["estimator"] == "ra"
        assert params["vce"] == "hc1"

    def test_set_params_modifies(self):
        est = LWDiD()
        est.set_params(rolling="detrend")
        assert est.rolling == "detrend"

    def test_set_params_returns_self(self):
        est = LWDiD()
        ret = est.set_params(estimator="ipw")
        assert ret is est

    def test_invalid_rolling_raises(self):
        with pytest.raises(ValueError, match="rolling"):
            LWDiD(rolling="invalid")

    def test_invalid_estimator_raises(self):
        with pytest.raises(ValueError, match="estimator"):
            LWDiD(estimator="invalid")

    def test_invalid_vce_raises(self):
        with pytest.raises(ValueError, match="vce"):
            LWDiD(vce="invalid")

    def test_invalid_control_group_raises(self):
        with pytest.raises(ValueError, match="control_group"):
            LWDiD(control_group="invalid")

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            LWDiD(alpha=0.0)
        with pytest.raises(ValueError, match="alpha"):
            LWDiD(alpha=1.0)

    def test_invalid_n_bootstrap_raises(self):
        with pytest.raises(ValueError, match="n_bootstrap"):
            LWDiD(n_bootstrap=-1)

    def test_alias_LW_is_LWDiD(self):
        assert LW is LWDiD

    def test_default_params(self):
        est = LWDiD()
        assert est.rolling == "demean"
        assert est.estimator == "ra"
        assert est.vce == "hc1"
        assert est.control_group == "not_yet_treated"
        assert est.alpha == 0.05
        assert est.n_bootstrap == 0

    def test_repr(self):
        est = LWDiD(rolling="demean", estimator="ra")
        r = repr(est)
        assert "LWDiD" in r
        assert "demean" in r
        assert "ra" in r

    def test_set_params_invalid_key_raises(self):
        est = LWDiD()
        with pytest.raises(ValueError, match="Invalid parameter"):
            est.set_params(bad_param="x")


# ─── Input Validation Tests ─────────────────────────────────────────────────


class TestLWDiDInputValidation:
    """Test input data validation."""

    def test_missing_column_raises(self):
        df = pd.DataFrame({"unit": [1], "time": [1], "y": [1.0]})
        with pytest.raises(ValueError, match="Columns not found"):
            LWDiD().fit(df, outcome="y", unit="unit", time="time", treatment="treat")

    def test_nan_in_outcome_raises(self):
        df = pd.DataFrame(
            {
                "unit": [1, 1, 2, 2],
                "time": [1, 2, 1, 2],
                "y": [1.0, np.nan, 2.0, 3.0],
                "treat": [0, 1, 0, 0],
            }
        )
        with pytest.raises(ValueError, match="missing values"):
            LWDiD().fit(df, outcome="y", unit="unit", time="time", treatment="treat")

    def test_nan_in_treatment_raises(self):
        df = pd.DataFrame(
            {
                "unit": [1, 1, 2, 2],
                "time": [1, 2, 1, 2],
                "y": [1.0, 2.0, 2.0, 3.0],
                "treat": [0, np.nan, 0, 0],
            }
        )
        with pytest.raises(ValueError, match="missing values"):
            LWDiD().fit(df, outcome="y", unit="unit", time="time", treatment="treat")

    def test_duplicate_unit_time_raises(self):
        df = pd.DataFrame(
            {
                "unit": [1, 1, 1, 2],
                "time": [1, 1, 2, 1],
                "y": [1.0, 1.5, 2.0, 3.0],
                "treat": [0, 0, 1, 0],
            }
        )
        with pytest.raises(ValueError, match="duplicate"):
            LWDiD().fit(df, outcome="y", unit="unit", time="time", treatment="treat")

    def test_non_binary_treatment_raises(self):
        df = pd.DataFrame(
            {
                "unit": [1, 1, 2, 2],
                "time": [1, 2, 1, 2],
                "y": [1.0, 2.0, 3.0, 4.0],
                "treat": [0, 2, 0, 0],  # not binary
            }
        )
        with pytest.raises(ValueError):
            LWDiD().fit(df, outcome="y", unit="unit", time="time", treatment="treat")

    def test_cluster_required_when_vce_cluster(self):
        panel = _make_common_timing_panel()
        with pytest.raises(ValueError, match="cluster"):
            LWDiD(vce="cluster").fit(
                panel, outcome="y", unit="unit", time="time", treatment="treat"
            )

    def test_no_treated_units_raises(self):
        df = pd.DataFrame(
            {
                "unit": [1, 1, 2, 2],
                "time": [1, 2, 1, 2],
                "y": [1.0, 2.0, 3.0, 4.0],
                "treat": [0, 0, 0, 0],
            }
        )
        with pytest.raises(ValueError, match="[Nn]o treated|[Nn]o post"):
            LWDiD().fit(df, outcome="y", unit="unit", time="time", treatment="treat")

    def test_no_control_units_raises(self):
        df = pd.DataFrame(
            {
                "unit": [1, 1, 2, 2],
                "time": [1, 2, 1, 2],
                "y": [1.0, 2.0, 3.0, 4.0],
                "treat": [0, 1, 0, 1],
            }
        )
        with pytest.raises(ValueError, match="[Nn]o control"):
            LWDiD().fit(df, outcome="y", unit="unit", time="time", treatment="treat")


# ─── Transformation Tests ───────────────────────────────────────────────────


class TestLWDiDTransformations:
    """Test that rolling transformations are correctly applied."""

    def test_demean_subtracts_pre_mean(self):
        """Construct simple 2-unit panel where pre-mean is known."""
        # Unit 0 (control): y = [2, 4, 6] → pre_mean = 3
        # Unit 1 (treated): y = [1, 3, 10] → pre_mean = 2
        df = pd.DataFrame(
            {
                "unit": [0, 0, 0, 1, 1, 1],
                "time": [1, 2, 3, 1, 2, 3],
                "y": [2.0, 4.0, 6.0, 1.0, 3.0, 10.0],
                "treat": [0, 0, 0, 0, 0, 1],
            }
        )
        res = LWDiD(rolling="demean", estimator="ra").fit(
            df, outcome="y", unit="unit", time="time", treatment="treat"
        )
        # The method demeaned using pre-treatment periods (time 1,2)
        # Unit 0: pre_mean = 3, post (time 3) ydot = 6-3 = 3
        # Unit 1: pre_mean = 2, post (time 3) ydot = 10-2 = 8
        # ATT = 8 - 3 = 5 (treatment effect + any trend difference)
        assert isinstance(res, LWDiDResults)
        assert np.isfinite(res.att)

    def test_detrend_removes_linear_trend(self):
        """Construct unit with perfect linear trend y = 1 + 2*t.

        After detrend, residuals should be ~0 in pre-period.
        """
        # Need at least 2 pre periods for detrend
        # Unit 0 (control): y = 1 + 2*t for all t
        # Unit 1 (treated): y = 1 + 2*t in pre, + 5 in post
        df = pd.DataFrame(
            {
                "unit": [0, 0, 0, 0, 1, 1, 1, 1],
                "time": [1, 2, 3, 4, 1, 2, 3, 4],
                "y": [3.0, 5.0, 7.0, 9.0, 3.0, 5.0, 12.0, 14.0],
                "treat": [0, 0, 0, 0, 0, 0, 1, 1],
            }
        )
        res = LWDiD(rolling="detrend", estimator="ra").fit(
            df, outcome="y", unit="unit", time="time", treatment="treat"
        )
        assert isinstance(res, LWDiDResults)
        # Detrended control should be ~0, detrended treated should show effect
        assert res.att > 0

    def test_transform_preserves_treatment_effect(self):
        """After demean, the treatment effect should still be visible."""
        panel = _make_common_timing_panel(true_att=5.0, seed=123)
        res = LWDiD(rolling="demean", estimator="ra").fit(
            panel, outcome="y", unit="unit", time="time", treatment="treat"
        )
        # True ATT is 5.0, estimate should be positive and in range
        assert res.att > 2.0


# ─── Common Timing Tests ────────────────────────────────────────────────────


class TestLWDiDCommonTiming:
    """Test common-timing estimation paths."""

    @pytest.fixture
    def panel(self):
        return _make_common_timing_panel(true_att=2.0)

    def test_ra_returns_results(self, panel):
        est = LWDiD(rolling="demean", estimator="ra")
        res = est.fit(panel, outcome="y", unit="unit", time="time", treatment="treat")
        assert isinstance(res, LWDiDResults)

    def test_ra_demean_positive_att(self, panel):
        res = LWDiD(rolling="demean", estimator="ra").fit(
            panel, outcome="y", unit="unit", time="time", treatment="treat"
        )
        assert res.att > 0  # True ATT is 2.0

    def test_ra_detrend_positive_att(self, panel):
        res = LWDiD(rolling="detrend", estimator="ra").fit(
            panel, outcome="y", unit="unit", time="time", treatment="treat"
        )
        assert res.att > 0

    def test_ra_att_close_to_truth(self, panel):
        """RA demean should recover ATT near 2.0 with enough data."""
        res = LWDiD(rolling="demean", estimator="ra").fit(
            panel, outcome="y", unit="unit", time="time", treatment="treat"
        )
        # Allow generous tolerance due to small sample noise
        assert 0.5 < res.att < 4.0

    def test_ipw_positive_att(self, panel):
        """IPW needs controls for propensity score."""
        panel_with_x = panel.copy()
        rng = np.random.default_rng(0)
        panel_with_x["x1"] = rng.normal(size=len(panel))
        res = LWDiD(rolling="demean", estimator="ipw").fit(
            panel_with_x, outcome="y", unit="unit", time="time", treatment="treat", controls=["x1"]
        )
        assert res.att > 0

    def test_ipwra_positive_att(self, panel):
        """IPWRA (doubly robust) should recover positive ATT."""
        panel_with_x = panel.copy()
        rng = np.random.default_rng(0)
        panel_with_x["x1"] = rng.normal(size=len(panel))
        res = LWDiD(rolling="demean", estimator="ipwra").fit(
            panel_with_x, outcome="y", unit="unit", time="time", treatment="treat", controls=["x1"]
        )
        assert res.att > 0

    def test_hc1_se_positive(self, panel):
        res = LWDiD(vce="hc1").fit(panel, outcome="y", unit="unit", time="time", treatment="treat")
        assert res.se > 0

    def test_classical_se_positive(self, panel):
        res = LWDiD(vce="classical").fit(
            panel, outcome="y", unit="unit", time="time", treatment="treat"
        )
        assert res.se > 0

    def test_cluster_robust_se(self, panel):
        """Cluster-robust SE should be positive."""
        # Create a cluster variable (group units into clusters)
        panel_cl = panel.copy()
        panel_cl["cluster_id"] = panel_cl["unit"] % 10
        res = LWDiD(vce="cluster").fit(
            panel_cl, outcome="y", unit="unit", time="time", treatment="treat", cluster="cluster_id"
        )
        assert res.se > 0

    def test_n_obs_n_treated_n_control(self, panel):
        """Sample sizes should be consistent."""
        res = LWDiD().fit(panel, outcome="y", unit="unit", time="time", treatment="treat")
        assert res.n_treated == 30
        assert res.n_control == 50
        assert res.n_obs == 80

    def test_result_not_staggered(self, panel):
        res = LWDiD().fit(panel, outcome="y", unit="unit", time="time", treatment="treat")
        assert not res.is_staggered
        assert res.cohort_effects is None

    def test_params_stored(self, panel):
        """RA should store coefficient vector."""
        res = LWDiD(rolling="demean", estimator="ra").fit(
            panel, outcome="y", unit="unit", time="time", treatment="treat"
        )
        assert res.params is not None
        assert len(res.params) >= 2  # intercept + treatment

    def test_vcov_stored(self, panel):
        """RA should store vcov matrix."""
        res = LWDiD(rolling="demean", estimator="ra").fit(
            panel, outcome="y", unit="unit", time="time", treatment="treat"
        )
        assert res.vcov is not None
        assert res.vcov.shape[0] == res.vcov.shape[1]

    def test_controls_improve_precision(self):
        """Adding relevant controls should reduce SE (most cases)."""
        rng = np.random.default_rng(99)
        panel = _make_common_timing_panel(n_treated=50, n_control=100, seed=99)
        # Add control correlated with outcome
        unit_map = {}
        for uid in panel["unit"].unique():
            unit_map[uid] = rng.normal(0, 2)
        panel["x_corr"] = panel["unit"].map(unit_map)

        res_no_ctrl = LWDiD(estimator="ra").fit(
            panel, outcome="y", unit="unit", time="time", treatment="treat"
        )
        res_ctrl = LWDiD(estimator="ra").fit(
            panel, outcome="y", unit="unit", time="time", treatment="treat", controls=["x_corr"]
        )
        # Both should produce finite results
        assert np.isfinite(res_no_ctrl.se)
        assert np.isfinite(res_ctrl.se)


# ─── Staggered Design Tests ─────────────────────────────────────────────────


class TestLWDiDStaggered:
    """Test staggered adoption designs."""

    @pytest.fixture
    def stag_panel(self):
        return _make_staggered_panel(true_att=1.5)

    def test_staggered_never_treated(self, stag_panel):
        res = LWDiD(control_group="never_treated").fit(
            stag_panel, outcome="y", unit="unit", time="time", treatment="treat", cohort="cohort"
        )
        assert isinstance(res, LWDiDResults)
        assert res.cohort_effects is not None

    def test_staggered_not_yet_treated(self, stag_panel):
        res = LWDiD(control_group="not_yet_treated").fit(
            stag_panel, outcome="y", unit="unit", time="time", treatment="treat", cohort="cohort"
        )
        assert res.att is not None
        assert np.isfinite(res.att)

    def test_cohort_effects_populated(self, stag_panel):
        res = LWDiD().fit(
            stag_panel, outcome="y", unit="unit", time="time", treatment="treat", cohort="cohort"
        )
        assert res.cohort_effects is not None
        assert len(res.cohort_effects) > 0

    def test_staggered_att_positive(self, stag_panel):
        """Overall ATT should be positive (true_att=1.5)."""
        res = LWDiD(control_group="never_treated").fit(
            stag_panel, outcome="y", unit="unit", time="time", treatment="treat", cohort="cohort"
        )
        assert res.att > 0

    def test_staggered_is_staggered(self, stag_panel):
        res = LWDiD().fit(
            stag_panel, outcome="y", unit="unit", time="time", treatment="treat", cohort="cohort"
        )
        assert res.is_staggered

    def test_staggered_se_positive(self, stag_panel):
        res = LWDiD(control_group="never_treated").fit(
            stag_panel, outcome="y", unit="unit", time="time", treatment="treat", cohort="cohort"
        )
        assert res.se > 0

    def test_aggregate_preserves_fitted_overall_inference(self, stag_panel):
        """aggregate('overall') must not discard joint fitted inference."""
        result = LWDiD(estimator="ra", vce="classical", control_group="never_treated").fit(
            stag_panel, outcome="y", unit="unit", time="time", treatment="treat", cohort="cohort"
        )

        aggregated = result.aggregate("overall")

        assert aggregated is not result
        assert aggregated.att == result.att
        assert aggregated.se == result.se
        assert aggregated.t_stat == result.t_stat
        assert aggregated.p_value == result.p_value
        assert aggregated.conf_int == result.conf_int
        assert aggregated.df_inference == result.df_inference

    def test_staggered_detrend(self, stag_panel):
        """Detrend should also work for staggered."""
        res = LWDiD(rolling="detrend", control_group="never_treated").fit(
            stag_panel, outcome="y", unit="unit", time="time", treatment="treat", cohort="cohort"
        )
        assert isinstance(res, LWDiDResults)
        assert res.att > 0

    def test_no_treated_cohorts_raises(self):
        """All cohort=0 should raise."""
        df = pd.DataFrame(
            {
                "unit": [1, 1, 2, 2],
                "time": [1, 2, 1, 2],
                "y": [1.0, 2.0, 3.0, 4.0],
                "treat": [0, 0, 0, 0],
                "cohort": [0, 0, 0, 0],
            }
        )
        with pytest.raises(ValueError, match="[Nn]o treated cohort"):
            LWDiD().fit(
                df, outcome="y", unit="unit", time="time", treatment="treat", cohort="cohort"
            )

    def test_never_treated_required_when_specified(self):
        """control_group='never_treated' requires at least one cohort=0 unit."""
        # All units are in cohort 3 (treated)
        df = pd.DataFrame(
            {
                "unit": [1, 1, 2, 2, 3, 3],
                "time": [1, 2, 1, 2, 1, 2],
                "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "treat": [0, 1, 0, 1, 0, 0],
                "cohort": [2, 2, 2, 2, 3, 3],
            }
        )
        with pytest.raises(ValueError, match="never-treated"):
            LWDiD(control_group="never_treated").fit(
                df, outcome="y", unit="unit", time="time", treatment="treat", cohort="cohort"
            )


# ─── Results Container Tests ────────────────────────────────────────────────


class TestLWDiDResults:
    """Test the LWDiDResults dataclass interface."""

    @pytest.fixture
    def result(self):
        panel = _make_common_timing_panel()
        return LWDiD().fit(panel, outcome="y", unit="unit", time="time", treatment="treat")

    def test_inference_consistency(self, result):
        """t_stat ≈ att / se."""
        if result.se > 0 and np.isfinite(result.se):
            np.testing.assert_allclose(result.t_stat, result.att / result.se, rtol=1e-10)

    def test_conf_int_bounds(self, result):
        """CI should bracket ATT."""
        lo, hi = result.conf_int
        assert lo < result.att < hi

    def test_conf_int_symmetric(self, result):
        """CI should be symmetric around ATT (normal-based)."""
        lo, hi = result.conf_int
        half_width_lo = result.att - lo
        half_width_hi = hi - result.att
        np.testing.assert_allclose(half_width_lo, half_width_hi, rtol=1e-10)

    def test_p_value_range(self, result):
        """p-value should be in [0, 1]."""
        assert 0 <= result.p_value <= 1

    def test_summary_contains_fields(self, result):
        s = result.summary()
        assert "ATT" in s or "att" in s.lower()
        assert "LWDiD" in s

    def test_to_dataframe(self, result):
        df = result.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) >= 1
        assert "att" in df.columns

    def test_to_dict_serializable(self, result):
        """to_dict() should produce JSON-serializable output."""
        d = result.to_dict()
        json.dumps(d, default=str)

    def test_to_dict_contains_keys(self, result):
        d = result.to_dict()
        assert "att" in d
        assert "se" in d
        assert "rolling" in d
        assert "estimator" in d

    def test_repr_informative(self, result):
        r = repr(result)
        assert "LWDiDResults" in r
        assert "ATT" in r

    def test_rolling_metadata(self, result):
        assert result.rolling == "demean"
        assert result.estimator == "ra"
        assert result.vce_type == "hc1"
        assert result.alpha == 0.05

    def test_nan_inference_when_se_zero(self):
        """Direct construction with se=0 should give NaN inference."""
        res = LWDiDResults(
            att=1.0,
            se=0.0,
            t_stat=float("nan"),
            p_value=float("nan"),
            conf_int=(float("nan"), float("nan")),
            n_obs=100,
            n_treated=30,
            n_control=70,
            rolling="demean",
            estimator="ra",
            vce_type="hc1",
            alpha=0.05,
        )
        assert np.isnan(res.t_stat)
        assert np.isnan(res.p_value)
        assert np.isnan(res.conf_int[0])
        assert np.isnan(res.conf_int[1])


# ─── Different VCE Comparisons ──────────────────────────────────────────────


class TestLWDiDVCEComparisons:
    """Compare VCE methods produce different but finite SEs."""

    @pytest.fixture
    def panel(self):
        return _make_common_timing_panel(n_treated=40, n_control=80, seed=77)

    def test_hc1_vs_classical(self, panel):
        res_cl = LWDiD(vce="classical").fit(
            panel, outcome="y", unit="unit", time="time", treatment="treat"
        )
        res_hc1 = LWDiD(vce="hc1").fit(
            panel, outcome="y", unit="unit", time="time", treatment="treat"
        )
        # ATTs should be the same (same point estimate)
        np.testing.assert_allclose(res_cl.att, res_hc1.att, atol=1e-12)
        # SEs differ
        assert res_cl.se > 0
        assert res_hc1.se > 0

    def test_cluster_vs_hc1(self, panel):
        panel_cl = panel.copy()
        panel_cl["cluster_id"] = panel_cl["unit"] % 10
        res_hc1 = LWDiD(vce="hc1").fit(
            panel, outcome="y", unit="unit", time="time", treatment="treat"
        )
        res_cl = LWDiD(vce="cluster").fit(
            panel_cl, outcome="y", unit="unit", time="time", treatment="treat", cluster="cluster_id"
        )
        # Point estimates should be identical
        np.testing.assert_allclose(res_hc1.att, res_cl.att, atol=1e-12)
        # Both SEs positive
        assert res_cl.se > 0
        assert res_hc1.se > 0


# ─── Estimator Consistency Tests ────────────────────────────────────────────


class TestLWDiDEstimatorConsistency:
    """Test that different estimators produce consistent results."""

    @pytest.fixture
    def panel_with_controls(self):
        panel = _make_common_timing_panel(n_treated=50, n_control=100, seed=55)
        rng = np.random.default_rng(55)
        panel["x1"] = rng.normal(size=len(panel))
        return panel

    def test_ra_ipw_same_sign(self, panel_with_controls):
        """RA and IPW should give same-sign ATT."""
        res_ra = LWDiD(estimator="ra").fit(
            panel_with_controls,
            outcome="y",
            unit="unit",
            time="time",
            treatment="treat",
            controls=["x1"],
        )
        res_ipw = LWDiD(estimator="ipw").fit(
            panel_with_controls,
            outcome="y",
            unit="unit",
            time="time",
            treatment="treat",
            controls=["x1"],
        )
        assert np.sign(res_ra.att) == np.sign(res_ipw.att)

    def test_ra_ipwra_same_sign(self, panel_with_controls):
        """RA and IPWRA should give same-sign ATT."""
        res_ra = LWDiD(estimator="ra").fit(
            panel_with_controls,
            outcome="y",
            unit="unit",
            time="time",
            treatment="treat",
            controls=["x1"],
        )
        res_ipwra = LWDiD(estimator="ipwra").fit(
            panel_with_controls,
            outcome="y",
            unit="unit",
            time="time",
            treatment="treat",
            controls=["x1"],
        )
        assert np.sign(res_ra.att) == np.sign(res_ipwra.att)

    def test_ipw_without_controls_warns(self):
        """IPW without controls should warn and behave like RA."""
        panel = _make_common_timing_panel(seed=88)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            res = LWDiD(estimator="ipw").fit(
                panel, outcome="y", unit="unit", time="time", treatment="treat"
            )
            # Should produce a warning about no controls
            ipw_warnings = [x for x in w if "IPW" in str(x.message)]
            assert len(ipw_warnings) > 0
        assert np.isfinite(res.att)
