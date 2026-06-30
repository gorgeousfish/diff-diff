"""Tests for lwdid_sensitivity module."""

import numpy as np
import pandas as pd
import pytest

from diff_diff.lwdid_sensitivity import (
    _classify_robustness,
    _compute_sensitivity_ratio,
    robustness_pre_periods,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def panel_data():
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


# ---------------------------------------------------------------------------
# SensitivityResult fields
# ---------------------------------------------------------------------------


class TestSensitivityResultFields:
    """Test SensitivityResult dataclass has all expected fields."""

    def test_result_fields_present(self, panel_data):
        r = robustness_pre_periods(
            panel_data,
            outcome="y",
            unit="unit",
            time="time",
            treatment="treat",
        )
        assert hasattr(r, "specifications")
        assert hasattr(r, "baseline_att")
        assert hasattr(r, "baseline_se")
        assert hasattr(r, "sensitivity_ratio")
        assert hasattr(r, "robustness_level")
        assert hasattr(r, "n_specifications")

    def test_result_types(self, panel_data):
        r = robustness_pre_periods(
            panel_data,
            outcome="y",
            unit="unit",
            time="time",
            treatment="treat",
        )
        assert isinstance(r.specifications, list)
        assert isinstance(r.baseline_att, float)
        assert isinstance(r.baseline_se, float)
        assert isinstance(r.sensitivity_ratio, float)
        assert isinstance(r.robustness_level, str)
        assert isinstance(r.n_specifications, int)


# ---------------------------------------------------------------------------
# Robustness level valid
# ---------------------------------------------------------------------------


class TestRobustnessLevel:
    """Test robustness_level is a valid classification."""

    VALID_LEVELS = {"highly_robust", "moderately_robust", "sensitive", "highly_sensitive"}

    def test_robustness_level_valid(self, panel_data):
        r = robustness_pre_periods(
            panel_data,
            outcome="y",
            unit="unit",
            time="time",
            treatment="treat",
        )
        assert r.robustness_level in self.VALID_LEVELS

    def test_classify_robustness_helper(self):
        assert _classify_robustness(0.05) == "highly_robust"
        assert _classify_robustness(0.15) == "moderately_robust"
        assert _classify_robustness(0.35) == "sensitive"
        assert _classify_robustness(0.60) == "highly_sensitive"


# ---------------------------------------------------------------------------
# Sensitivity ratio non-negative
# ---------------------------------------------------------------------------


class TestSensitivityRatio:
    """Test sensitivity_ratio is non-negative."""

    def test_ratio_non_negative(self, panel_data):
        r = robustness_pre_periods(
            panel_data,
            outcome="y",
            unit="unit",
            time="time",
            treatment="treat",
        )
        assert r.sensitivity_ratio >= 0.0

    def test_compute_sensitivity_ratio_helper(self):
        assert _compute_sensitivity_ratio(2.0, [2.0, 2.1, 1.9]) == pytest.approx(0.1)
        assert _compute_sensitivity_ratio(2.0, [2.0]) == 0.0
        # Near-zero baseline
        assert _compute_sensitivity_ratio(1e-15, [1e-15, 0.5]) == 0.0


# ---------------------------------------------------------------------------
# Specifications list populated
# ---------------------------------------------------------------------------


class TestSpecifications:
    """Test specifications list is populated."""

    def test_specs_populated(self, panel_data):
        r = robustness_pre_periods(
            panel_data,
            outcome="y",
            unit="unit",
            time="time",
            treatment="treat",
        )
        # Should have at least 1 specification
        assert len(r.specifications) >= 1
        assert r.n_specifications >= 2  # baseline + at least 1 alternative

    def test_spec_has_expected_attributes(self, panel_data):
        r = robustness_pre_periods(
            panel_data,
            outcome="y",
            unit="unit",
            time="time",
            treatment="treat",
        )
        if r.specifications:
            spec = r.specifications[0]
            assert hasattr(spec, "label")
            assert hasattr(spec, "rolling")
            assert hasattr(spec, "estimator")
            assert hasattr(spec, "att")
            assert hasattr(spec, "se")
            assert hasattr(spec, "pvalue")


# ---------------------------------------------------------------------------
# to_dataframe()
# ---------------------------------------------------------------------------


class TestToDataframe:
    """Test to_dataframe() returns a DataFrame."""

    def test_to_dataframe_returns_df(self, panel_data):
        r = robustness_pre_periods(
            panel_data,
            outcome="y",
            unit="unit",
            time="time",
            treatment="treat",
        )
        df = r.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) >= 1
        assert "att" in df.columns
        assert "label" in df.columns

    def test_summary_returns_string(self, panel_data):
        r = robustness_pre_periods(
            panel_data,
            outcome="y",
            unit="unit",
            time="time",
            treatment="treat",
        )
        s = r.summary()
        assert isinstance(s, str)
        assert "Sensitivity" in s
