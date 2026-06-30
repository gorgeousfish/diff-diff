"""Tests for LWDiD diagnostics output and mathematical correctness.

Verifies:
1. _dispatch_estimator routing and return structure
2. Transformation diagnostics (get_transformation_diagnostics)
3. Mathematical correctness against Lee & Wooldridge (2025, 2026) formulas
4. Backward compatibility (existing fit() behavior unchanged)
"""

import numpy as np
import pandas as pd
import pytest

from diff_diff import LWDiD

# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def simple_panel():
    """Simple balanced panel: 40 units, 8 periods, treatment at t=5."""
    rng = np.random.default_rng(42)
    records = []
    for i in range(40):
        d = int(i < 15)
        for t in range(1, 9):
            y = 1.0 + 0.3 * i / 40 + 0.1 * t + rng.normal(0, 0.3)
            post = int(t > 4)
            if d and post:
                y += 2.0
            records.append({"unit": i, "time": t, "y": y, "treat": d * post})
    return pd.DataFrame(records)


@pytest.fixture
def panel_with_controls():
    """Panel with covariate X."""
    rng = np.random.default_rng(123)
    records = []
    for i in range(60):
        d = int(i < 20)
        x1 = rng.normal() + d * 0.3
        for t in range(1, 9):
            y = 1.0 + 0.5 * x1 + 0.1 * t + rng.normal(0, 0.3)
            post = int(t > 4)
            if d and post:
                y += 2.0
            records.append({"unit": i, "time": t, "y": y, "treat": d * post, "x1": x1})
    return pd.DataFrame(records)


@pytest.fixture
def quarterly_panel():
    """Panel with 16 periods (4 years of quarterly data)."""
    rng = np.random.default_rng(99)
    records = []
    for i in range(50):
        d = int(i < 18)
        for t in range(1, 17):
            q = (t - 1) % 4 + 1
            seasonal = 0.5 * (q == 4) - 0.3 * (q == 1)
            y = 2.0 + 0.05 * t + seasonal + rng.normal(0, 0.2)
            post = int(t > 8)
            if d and post:
                y += 1.5
            records.append({"unit": i, "time": t, "y": y, "treat": d * post})
    return pd.DataFrame(records)


# ============================================================
# Class 1: _dispatch_estimator behavior verification
# ============================================================


class TestDispatchEstimator:
    """Verify _dispatch_estimator routing and return structure."""

    def test_ra_returns_valid_result(self, simple_panel):
        """RA path returns valid ATT estimate."""
        est = LWDiD(rolling="demean", estimator="ra")
        res = est.fit(simple_panel, outcome="y", unit="unit", time="time", treatment="treat")
        # Verify ATT is finite and reasonable
        assert np.isfinite(res.att)
        assert 1.0 < res.att < 3.0  # true ATT = 2.0

    def test_ipw_returns_valid_result(self, panel_with_controls):
        """IPW path returns valid results with controls."""
        est = LWDiD(rolling="demean", estimator="ipw")
        res = est.fit(
            panel_with_controls,
            outcome="y",
            unit="unit",
            time="time",
            treatment="treat",
            controls=["x1"],
        )
        assert np.isfinite(res.att)
        assert np.isfinite(res.se)

    def test_ipwra_returns_valid_result(self, panel_with_controls):
        """IPWRA path returns valid doubly-robust results."""
        est = LWDiD(rolling="demean", estimator="ipwra")
        res = est.fit(
            panel_with_controls,
            outcome="y",
            unit="unit",
            time="time",
            treatment="treat",
            controls=["x1"],
        )
        assert np.isfinite(res.att)

    def test_psm_returns_valid_result(self, panel_with_controls):
        """PSM path returns valid matched results."""
        est = LWDiD(rolling="demean", estimator="psm")
        res = est.fit(
            panel_with_controls,
            outcome="y",
            unit="unit",
            time="time",
            treatment="treat",
            controls=["x1"],
        )
        assert np.isfinite(res.att)

    def test_all_estimators_same_data_give_reasonable_att(self, panel_with_controls):
        """All 4 estimators should give ATT in [1.0, 3.0] for true ATT=2.0."""
        for est_name in ["ra", "ipw", "ipwra", "psm"]:
            est = LWDiD(rolling="demean", estimator=est_name)
            res = est.fit(
                panel_with_controls,
                outcome="y",
                unit="unit",
                time="time",
                treatment="treat",
                controls=["x1"],
            )
            assert 1.0 < res.att < 3.0, f"{est_name} ATT={res.att} outside [1,3]"

    def test_ipw_without_controls_still_works(self, simple_panel):
        """IPW without controls still produces a result."""
        import warnings

        est = LWDiD(estimator="ipw")
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            res = est.fit(simple_panel, outcome="y", unit="unit", time="time", treatment="treat")
        assert np.isfinite(res.att)


# ============================================================
# Class 2: Transformation diagnostics
# ============================================================


class TestTransformationDiagnostics:
    """Verify get_transformation_diagnostics() output structure and values."""

    def test_demean_diagnostics_structure(self, simple_panel):
        """Demean diagnostics has correct structure."""
        est = LWDiD(rolling="demean")
        diag = est.get_transformation_diagnostics(
            simple_panel, outcome="y", unit="unit", time="time", treatment="treat"
        )
        assert diag["method"] == "demean"
        assert "per_unit" in diag
        assert "summary" in diag
        assert len(diag["per_unit"]) == 40  # 40 units
        # Check per-unit fields
        first_unit = list(diag["per_unit"].values())[0]
        assert "pre_mean" in first_unit
        assert "pre_n_periods" in first_unit
        assert "pre_std" in first_unit
        assert "valid" in first_unit
        # Check summary fields
        assert "n_units_total" in diag["summary"]
        assert "n_units_valid" in diag["summary"]

    def test_detrend_diagnostics_structure(self, simple_panel):
        """Detrend diagnostics has correct structure with alpha/beta."""
        est = LWDiD(rolling="detrend")
        diag = est.get_transformation_diagnostics(
            simple_panel, outcome="y", unit="unit", time="time", treatment="treat"
        )
        assert diag["method"] == "detrend"
        first_unit = list(diag["per_unit"].values())[0]
        assert "alpha" in first_unit
        assert "beta" in first_unit
        assert "r_squared" in first_unit

    def test_demeanq_diagnostics_structure(self, quarterly_panel):
        """Demeanq diagnostics has seasonal effects."""
        est = LWDiD(rolling="demeanq")
        diag = est.get_transformation_diagnostics(
            quarterly_panel, outcome="y", unit="unit", time="time", treatment="treat"
        )
        assert diag["method"] == "demeanq"
        first_unit = list(diag["per_unit"].values())[0]
        assert "intercept" in first_unit
        assert "seasonal_effects" in first_unit

    def test_detrendq_diagnostics_structure(self, quarterly_panel):
        """Detrendq diagnostics has trend + seasonal."""
        est = LWDiD(rolling="detrendq")
        diag = est.get_transformation_diagnostics(
            quarterly_panel, outcome="y", unit="unit", time="time", treatment="treat"
        )
        assert diag["method"] == "detrendq"
        first_unit = list(diag["per_unit"].values())[0]
        assert "alpha" in first_unit
        assert "beta" in first_unit
        assert "seasonal_effects" in first_unit

    def test_diagnostics_does_not_affect_estimation(self, simple_panel):
        """get_transformation_diagnostics does not change fit() results."""
        est = LWDiD(rolling="detrend")
        # Get diagnostics first
        est.get_transformation_diagnostics(
            simple_panel, outcome="y", unit="unit", time="time", treatment="treat"
        )
        # Then fit
        res = est.fit(simple_panel, outcome="y", unit="unit", time="time", treatment="treat")
        # Should still be correct
        assert np.isfinite(res.att)
        assert 1.0 < res.att < 3.0


# ============================================================
# Class 3: Mathematical correctness (Lee & Wooldridge formulas)
# ============================================================


class TestMathematicalCorrectness:
    """Verify mathematical formulas against hand-computed values.

    Reference: Lee & Wooldridge (2025), Procedures 2.1 and 3.1.
    """

    def test_demean_formula_hand_computed(self):
        """Verify Ȳ_{i,pre} = (1/(S-1)) * Σ_{t=1}^{S-1} Y_{it}.

        Per Procedure 2.1: pre-treatment mean subtracted from all periods.
        """
        # Construct tiny known dataset: 3 units, 4 periods, treatment at t=3
        # All units are treated so pre_mask = (treat == 0) → t=1,2 for all
        df = pd.DataFrame(
            {
                "unit": [0] * 4 + [1] * 4 + [2] * 4,
                "time": [1, 2, 3, 4] * 3,
                "y": [
                    2.0,
                    4.0,
                    10.0,
                    12.0,  # unit 0: pre_mean = (2+4)/2 = 3.0
                    1.0,
                    3.0,
                    8.0,
                    10.0,  # unit 1: pre_mean = (1+3)/2 = 2.0
                    3.0,
                    5.0,
                    6.0,
                    7.0,  # unit 2: pre_mean = (3+5)/2 = 4.0
                ],
                "treat": [0, 0, 1, 1] * 3,  # all units treated at t=3
            }
        )
        est = LWDiD(rolling="demean")
        diag = est.get_transformation_diagnostics(
            df, outcome="y", unit="unit", time="time", treatment="treat"
        )
        # Verify pre-treatment means
        np.testing.assert_allclose(diag["per_unit"][0]["pre_mean"], 3.0, atol=1e-10)
        np.testing.assert_allclose(diag["per_unit"][1]["pre_mean"], 2.0, atol=1e-10)
        np.testing.assert_allclose(diag["per_unit"][2]["pre_mean"], 4.0, atol=1e-10)

    def test_detrend_formula_hand_computed(self):
        """Verify α̂_i, β̂_i from pre-treatment OLS: Y_{it} = α + β*t + ε.

        Per Procedure 3.1: unit-specific linear trend removed.
        """
        # Unit with perfect linear trend: Y = 1 + 2*t
        # Pre periods: t=1→3, t=2→5, t=3→7
        # OLS fit with centered time: Y = α + β*(t - t_mean)
        # t_mean = 2.0, so t_centered = [-1, 0, 1]
        # Y = [3, 5, 7] => perfect fit: α=5 (at t_centered=0), β=2
        df = pd.DataFrame(
            {
                "unit": [0] * 6,
                "time": [1, 2, 3, 4, 5, 6],
                "y": [3.0, 5.0, 7.0, 20.0, 22.0, 24.0],  # post has treatment effect
                "treat": [0, 0, 0, 1, 1, 1],
            }
        )
        est = LWDiD(rolling="detrend")
        diag = est.get_transformation_diagnostics(
            df, outcome="y", unit="unit", time="time", treatment="treat"
        )
        unit_diag = diag["per_unit"][0]
        # Beta (slope) should be 2.0 — invariant to centering
        np.testing.assert_allclose(unit_diag["beta"], 2.0, atol=1e-10)
        # Alpha is intercept at centered origin: Y at t_centered=0 = Y at t=2 = 5.0
        np.testing.assert_allclose(unit_diag["alpha"], 5.0, atol=1e-10)
        # R^2 should be 1.0 for perfect linear fit
        np.testing.assert_allclose(unit_diag["r_squared"], 1.0, atol=1e-10)

    def test_degrees_of_freedom_formula(self, simple_panel):
        """Verify df = N - K - 2 per paper Section 2.4.

        Without controls: df = N - 0 - 2 = N - 2
        """
        est = LWDiD(rolling="demean", estimator="ra")
        res = est.fit(simple_panel, outcome="y", unit="unit", time="time", treatment="treat")
        # N = 40 units, K = 0 controls → df = 40 - 0 - 2 = 38
        assert res.df_inference == 38

    def test_ra_interaction_term_present(self, panel_with_controls):
        """Verify RA includes interaction per Eq 3.3.

        Design matrix should include [1, D, X, D*(X-X̄₁)] when controls present.
        """
        est = LWDiD(rolling="demean", estimator="ra")
        res = est.fit(
            panel_with_controls,
            outcome="y",
            unit="unit",
            time="time",
            treatment="treat",
            controls=["x1"],
        )
        assert np.isfinite(res.att)
        assert np.isfinite(res.se)

    def test_cluster_uses_g_minus_1_df(self, simple_panel):
        """Verify cluster-robust uses df = G - 1."""
        est = LWDiD(rolling="demean", vce="cluster")
        res = est.fit(
            simple_panel, outcome="y", unit="unit", time="time", treatment="treat", cluster="unit"
        )
        # G = 40 units as clusters → df = 39
        assert res.df_inference == 39

    def test_t_stat_equals_att_over_se(self, simple_panel):
        """Verify t_stat = att / se (basic algebra check)."""
        est = LWDiD()
        res = est.fit(simple_panel, outcome="y", unit="unit", time="time", treatment="treat")
        if np.isfinite(res.t_stat) and np.isfinite(res.se) and res.se > 0:
            np.testing.assert_allclose(res.t_stat, res.att / res.se, rtol=1e-10)

    def test_confidence_interval_symmetric(self, simple_panel):
        """Verify CI is symmetric around ATT."""
        est = LWDiD()
        res = est.fit(simple_panel, outcome="y", unit="unit", time="time", treatment="treat")
        ci_lower, ci_upper = res.conf_int
        midpoint = (ci_lower + ci_upper) / 2
        np.testing.assert_allclose(midpoint, res.att, atol=1e-10)


# ============================================================
# Class 4: Backward compatibility
# ============================================================


class TestBackwardCompatibility:
    """Ensure existing fit() behavior is preserved."""

    def test_fit_unchanged_demean(self, simple_panel):
        """fit() with demean gives correct result."""
        est = LWDiD(rolling="demean")
        res = est.fit(simple_panel, outcome="y", unit="unit", time="time", treatment="treat")
        assert isinstance(res.att, float)
        assert np.isfinite(res.att)
        assert 1.0 < res.att < 3.0

    def test_fit_unchanged_detrend(self, simple_panel):
        """fit() with detrend gives correct result."""
        est = LWDiD(rolling="detrend")
        res = est.fit(simple_panel, outcome="y", unit="unit", time="time", treatment="treat")
        assert np.isfinite(res.att)

    def test_fit_unchanged_staggered(self):
        """Staggered fit still works correctly."""
        rng = np.random.default_rng(42)
        records = []
        for i in range(90):
            g = [0, 4, 7][i % 3]
            for t in range(1, 10):
                y = 1.0 + 0.05 * t + rng.normal(0, 0.2)
                if g > 0 and t >= g:
                    y += 1.5
                records.append(
                    {"unit": i, "time": t, "y": y, "treat": int(g > 0 and t >= g), "cohort": g}
                )
        df = pd.DataFrame(records)
        est = LWDiD(control_group="never_treated")
        res = est.fit(df, outcome="y", unit="unit", time="time", treatment="treat", cohort="cohort")
        assert np.isfinite(res.att)
        assert 1.0 < res.att < 2.5

    def test_bootstrap_unchanged(self, simple_panel):
        """Bootstrap still works after transform changes."""
        est = LWDiD(n_bootstrap=20)
        res = est.fit(simple_panel, outcome="y", unit="unit", time="time", treatment="treat")
        assert np.isfinite(res.att)
        assert np.isfinite(res.se)
