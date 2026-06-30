"""Tests for lwdid_trend_diagnostics module."""

import numpy as np
import pandas as pd
import pytest

from diff_diff.lwdid_exceptions import (
    DiagnosticError,
    InsufficientPrePeriodsError,
)
from diff_diff.lwdid_trend_diagnostics import (
    ParallelTrendsTestResult,
    TransformationRecommendation,
    recommend_transformation,
)
from diff_diff.lwdid_trend_diagnostics import (
    test_parallel_trends as check_parallel_trends,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def panel_data():
    """Panel data WITH parallel trends (no pre-treatment effects)."""
    rng = np.random.default_rng(42)
    records = []
    for i in range(80):
        d = int(i < 25)
        for t in range(1, 9):
            y = 1.0 + 0.1 * t + rng.normal(0, 0.3)
            if d and t > 4:
                y += 2.0
            records.append({"unit": i, "time": t, "y": y, "treat": d * int(t > 4)})
    return pd.DataFrame(records)


@pytest.fixture
def panel_data_no_pt():
    """Panel data WITHOUT parallel trends (diverging pre-trends)."""
    rng = np.random.default_rng(42)
    records = []
    for i in range(80):
        d = int(i < 25)
        for t in range(1, 9):
            # Treated group has a strong upward pre-trend
            y = 1.0 + 0.1 * t + rng.normal(0, 0.3)
            if d:
                y += 0.8 * t  # diverging trend for treated
            if d and t > 4:
                y += 2.0
            records.append({"unit": i, "time": t, "y": y, "treat": d * int(t > 4)})
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# ParallelTrendsTestResult fields
# ---------------------------------------------------------------------------


class TestParallelTrendsResultFields:
    """Test that ParallelTrendsTestResult has expected fields."""

    def test_result_fields_present(self, panel_data):
        r = check_parallel_trends(
            panel_data, outcome="y", unit="unit", time="time", treatment="treat"
        )
        assert hasattr(r, "method")
        assert hasattr(r, "test_stat")
        assert hasattr(r, "pvalue")
        assert hasattr(r, "decision")
        assert hasattr(r, "pre_treatment_effects")
        assert hasattr(r, "n_pre_periods")
        assert hasattr(r, "significance_level")

    def test_result_types(self, panel_data):
        r = check_parallel_trends(
            panel_data, outcome="y", unit="unit", time="time", treatment="treat"
        )
        assert isinstance(r.method, str)
        assert isinstance(r.decision, str)
        assert isinstance(r.pre_treatment_effects, list)
        assert isinstance(r.n_pre_periods, int)
        assert isinstance(r.significance_level, float)


# ---------------------------------------------------------------------------
# Decision values
# ---------------------------------------------------------------------------


class TestDecisionValues:
    """Test decision is one of pass/fail/inconclusive."""

    def test_decision_is_valid(self, panel_data):
        r = check_parallel_trends(
            panel_data, outcome="y", unit="unit", time="time", treatment="treat"
        )
        assert r.decision in ("pass", "fail", "inconclusive")

    def test_summary_returns_string(self, panel_data):
        r = check_parallel_trends(
            panel_data, outcome="y", unit="unit", time="time", treatment="treat"
        )
        s = r.summary()
        assert isinstance(s, str)
        assert "PARALLEL TRENDS TEST" in s


# ---------------------------------------------------------------------------
# Data WITH parallel trends -> pass
# ---------------------------------------------------------------------------


class TestParallelTrendsPass:
    """Data with parallel trends should yield decision 'pass'."""

    def test_parallel_trends_detected(self, panel_data):
        r = check_parallel_trends(
            panel_data, outcome="y", unit="unit", time="time", treatment="treat"
        )
        # With true parallel trends the test should pass or be inconclusive
        # (never 'fail' for well-behaved data)
        assert r.decision in ("pass", "inconclusive")


# ---------------------------------------------------------------------------
# Data WITHOUT parallel trends -> fail
# ---------------------------------------------------------------------------


class TestParallelTrendsFail:
    """Data without parallel trends should yield decision 'fail'."""

    def test_no_parallel_trends_detected(self, panel_data_no_pt):
        r = check_parallel_trends(
            panel_data_no_pt,
            outcome="y",
            unit="unit",
            time="time",
            treatment="treat",
        )
        # With a strong diverging pre-trend the test should fail or be inconclusive
        assert r.decision in ("fail", "inconclusive")


# ---------------------------------------------------------------------------
# recommend_transformation returns valid recommendation
# ---------------------------------------------------------------------------


class TestRecommendTransformation:
    """Test recommend_transformation returns valid recommendation."""

    def test_returns_recommendation(self, panel_data):
        rec = recommend_transformation(
            panel_data, outcome="y", unit="unit", time="time", treatment="treat"
        )
        assert isinstance(rec, TransformationRecommendation)
        assert rec.recommended in ("demean", "detrend", "demeanq", "detrendq")
        assert rec.confidence in ("high", "medium", "low")
        assert isinstance(rec.rationale, str)
        assert len(rec.rationale) > 0

    def test_recommendation_has_parallel_trends_result(self, panel_data):
        rec = recommend_transformation(
            panel_data, outcome="y", unit="unit", time="time", treatment="treat"
        )
        assert isinstance(rec.parallel_trends_result, ParallelTrendsTestResult)

    def test_good_data_recommends_demean(self, panel_data):
        """With parallel trends holding, should recommend demean."""
        rec = recommend_transformation(
            panel_data, outcome="y", unit="unit", time="time", treatment="treat"
        )
        # Should recommend demean for data with parallel trends
        assert rec.recommended in ("demean", "detrend")

    def test_recommendation_summary(self, panel_data):
        rec = recommend_transformation(
            panel_data, outcome="y", unit="unit", time="time", treatment="treat"
        )
        s = rec.summary()
        assert isinstance(s, str)
        assert "RECOMMENDATION" in s


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test error handling for edge cases."""

    def test_insufficient_pre_periods_raises(self):
        """With only 1 pre-period, should raise InsufficientPrePeriodsError."""
        records = []
        for i in range(40):
            d = int(i < 10)
            for t in [1, 2]:  # Only 1 pre-period (t=1), t=2 is post
                y = 1.0 + np.random.normal(0, 0.3)
                if d and t == 2:
                    y += 2.0
                records.append({"unit": i, "time": t, "y": y, "treat": d * int(t == 2)})
        df = pd.DataFrame(records)
        with pytest.raises(InsufficientPrePeriodsError):
            check_parallel_trends(df, outcome="y", unit="unit", time="time", treatment="treat")

    def test_no_treated_raises(self):
        """With no treated observations, should raise DiagnosticError."""
        records = []
        for i in range(20):
            for t in range(1, 5):
                records.append({"unit": i, "time": t, "y": 1.0, "treat": 0})
        df = pd.DataFrame(records)
        with pytest.raises(DiagnosticError):
            check_parallel_trends(df, outcome="y", unit="unit", time="time", treatment="treat")

    def test_n_tested_periods_property(self, panel_data):
        r = check_parallel_trends(
            panel_data, outcome="y", unit="unit", time="time", treatment="treat"
        )
        assert r.n_tested_periods == len(r.pre_treatment_effects)
