"""Exception and warning classes for LWDiD advanced inference and diagnostics.

These are used by the wild cluster bootstrap, randomization inference,
trend diagnostics, sensitivity analysis, and visualization modules.
"""

# ============================================================
# Base classes
# ============================================================


class LWDIDError(Exception):
    """Base exception for all LWDiD errors."""

    pass


class LWDIDWarning(UserWarning):
    """Base warning for all LWDiD warnings."""

    pass


# ============================================================
# Inference errors
# ============================================================


class LWDIDInferenceError(LWDIDError):
    """Raised when inference computation fails.

    Common causes: singular matrices, non-convergence of optimization,
    insufficient observations for requested inference method.
    """

    pass


class BootstrapConvergenceError(LWDIDInferenceError):
    """Raised when bootstrap fails to converge or produces degenerate results."""

    pass


class RandomizationError(LWDIDInferenceError):
    """Raised when randomization inference encounters an unrecoverable error.

    Common causes: all permutations produce degenerate treatment assignments,
    insufficient variation in treatment variable.
    """

    pass


# ============================================================
# Diagnostic errors
# ============================================================


class DiagnosticError(LWDIDError):
    """Raised when a diagnostic computation cannot be completed."""

    pass


class InsufficientPrePeriodsError(DiagnosticError):
    """Raised when there are too few pre-treatment periods for diagnostics."""

    pass


# ============================================================
# Visualization errors
# ============================================================


class VisualizationError(LWDIDError):
    """Raised when visualization cannot be produced.

    Most commonly due to matplotlib not being installed.
    Install with: pip install matplotlib
    """

    pass


# ============================================================
# Warnings
# ============================================================


class NumericalWarning(LWDIDWarning):
    """Warning for numerical stability issues.

    Issued when computations involve near-singular matrices,
    extreme condition numbers, or potential loss of precision.
    """

    pass


class RandomizationWarning(LWDIDWarning):
    """Warning for randomization inference quality issues.

    Issued when a high proportion of randomization draws produce
    degenerate results (all-treated or all-control assignments).
    """

    pass


class DiagnosticWarning(LWDIDWarning):
    """Warning when diagnostic results may be unreliable.

    Issued when sample sizes are small, pre-periods are few,
    or test power is likely insufficient.
    """

    pass


class SensitivityWarning(LWDIDWarning):
    """Warning for sensitivity analysis concerns.

    Issued when results appear highly sensitive to specification choices.
    """

    pass


class VisualizationWarning(LWDIDWarning):
    """Warning for non-critical visualization issues."""

    pass
