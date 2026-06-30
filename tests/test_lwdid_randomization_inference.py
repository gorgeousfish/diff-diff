"""Tests for lwdid_randomization module."""

import numpy as np
import pytest

from diff_diff.lwdid_exceptions import RandomizationError
from diff_diff.lwdid_randomization import (
    randomization_inference,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cross_section_data():
    rng = np.random.default_rng(42)
    n = 100
    y = np.concatenate([rng.normal(2, 0.5, 30), rng.normal(0, 0.5, 70)])
    treatment = np.array([1.0] * 30 + [0.0] * 70)
    cluster_ids = np.repeat(np.arange(20), 5)
    controls = rng.normal(0, 1, (n, 2))
    return y, treatment, cluster_ids, controls


# ---------------------------------------------------------------------------
# Result fields
# ---------------------------------------------------------------------------


class TestRandomizationResultFields:
    """Test that RandomizationResult has all expected fields."""

    def test_result_fields_present(self, cross_section_data):
        y, treatment, _, _ = cross_section_data
        r = randomization_inference(y, treatment, n_reps=200, seed=0)
        assert hasattr(r, "pvalue")
        assert hasattr(r, "att_observed")
        assert hasattr(r, "att_distribution")
        assert hasattr(r, "n_reps")
        assert hasattr(r, "n_valid")
        assert hasattr(r, "n_failed")
        assert hasattr(r, "failure_rate")
        assert hasattr(r, "method")
        assert hasattr(r, "seed")

    def test_result_types(self, cross_section_data):
        y, treatment, _, _ = cross_section_data
        r = randomization_inference(y, treatment, n_reps=200, seed=0)
        assert isinstance(r.pvalue, float)
        assert isinstance(r.att_observed, float)
        assert isinstance(r.att_distribution, np.ndarray)
        assert isinstance(r.n_reps, int)
        assert isinstance(r.n_valid, int)
        assert isinstance(r.n_failed, int)
        assert isinstance(r.failure_rate, float)
        assert isinstance(r.method, str)


# ---------------------------------------------------------------------------
# Permutation preserves N_treated
# ---------------------------------------------------------------------------


class TestPermutationPreservation:
    """Permutation should preserve number of treated units."""

    def test_permutation_preserves_n_treated(self, cross_section_data):
        y, treatment, _, _ = cross_section_data
        r = randomization_inference(y, treatment, method="permutation", n_reps=500, seed=0)
        # With permutation, no draws are degenerate
        assert r.n_failed == 0
        assert r.failure_rate == 0.0

    def test_bootstrap_may_not_preserve(self, cross_section_data):
        y, treatment, _, _ = cross_section_data
        # Bootstrap may produce degenerate draws but should not necessarily
        r = randomization_inference(y, treatment, method="bootstrap", n_reps=500, seed=0)
        # n_failed may be >= 0 (not guaranteed to be zero)
        assert r.n_failed >= 0


# ---------------------------------------------------------------------------
# P-value properties
# ---------------------------------------------------------------------------


class TestPValueProperties:
    """Test p-value is in valid range."""

    def test_pvalue_in_0_1_permutation(self, cross_section_data):
        y, treatment, _, _ = cross_section_data
        r = randomization_inference(y, treatment, method="permutation", n_reps=500, seed=42)
        assert 0.0 <= r.pvalue <= 1.0

    def test_pvalue_in_0_1_bootstrap(self, cross_section_data):
        y, treatment, _, _ = cross_section_data
        r = randomization_inference(y, treatment, method="bootstrap", n_reps=500, seed=42)
        assert 0.0 <= r.pvalue <= 1.0

    def test_clear_treatment_effect_detected(self, cross_section_data):
        """With a clear treatment effect, p-value should be small."""
        y, treatment, _, _ = cross_section_data
        r = randomization_inference(y, treatment, method="permutation", n_reps=999, seed=0)
        assert r.pvalue < 0.05


# ---------------------------------------------------------------------------
# With and without controls
# ---------------------------------------------------------------------------


class TestControls:
    """Test with and without control variables."""

    def test_without_controls(self, cross_section_data):
        y, treatment, _, _ = cross_section_data
        r = randomization_inference(y, treatment, n_reps=200, seed=0)
        assert np.isfinite(r.att_observed)
        assert r.n_valid > 0

    def test_with_controls(self, cross_section_data):
        y, treatment, _, controls = cross_section_data
        r = randomization_inference(y, treatment, controls=controls, n_reps=200, seed=0)
        assert np.isfinite(r.att_observed)
        assert r.n_valid > 0


# ---------------------------------------------------------------------------
# Degenerate data handling
# ---------------------------------------------------------------------------


class TestDegenerateData:
    """Test handling of degenerate inputs."""

    def test_all_treated_raises(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        treatment = np.array([1.0, 1.0, 1.0, 1.0])
        with pytest.raises(RandomizationError):
            randomization_inference(y, treatment, n_reps=100)

    def test_all_control_raises(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        treatment = np.array([0.0, 0.0, 0.0, 0.0])
        with pytest.raises(RandomizationError):
            randomization_inference(y, treatment, n_reps=100)

    def test_too_small_sample_raises(self):
        y = np.array([1.0, 2.0])
        treatment = np.array([1.0, 0.0])
        with pytest.raises(RandomizationError):
            randomization_inference(y, treatment, n_reps=100)

    def test_invalid_method_raises(self, cross_section_data):
        y, treatment, _, _ = cross_section_data
        with pytest.raises(RandomizationError):
            randomization_inference(y, treatment, method="invalid", n_reps=100)


# ---------------------------------------------------------------------------
# Seed reproducibility
# ---------------------------------------------------------------------------


class TestSeedReproducibility:
    """Test that seed produces reproducible results."""

    def test_same_seed_same_result(self, cross_section_data):
        y, treatment, _, _ = cross_section_data
        r1 = randomization_inference(y, treatment, n_reps=200, seed=123)
        r2 = randomization_inference(y, treatment, n_reps=200, seed=123)
        assert r1.pvalue == r2.pvalue
        np.testing.assert_array_equal(r1.att_distribution, r2.att_distribution)

    def test_different_seed_different_result(self, cross_section_data):
        y, treatment, _, _ = cross_section_data
        r1 = randomization_inference(y, treatment, n_reps=200, seed=1)
        r2 = randomization_inference(y, treatment, n_reps=200, seed=2)
        # Distributions should differ (extremely unlikely to be equal)
        assert not np.array_equal(r1.att_distribution, r2.att_distribution)
