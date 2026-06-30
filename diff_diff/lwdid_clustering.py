"""Clustering diagnostics for LWDiD.

Provides tools to diagnose appropriate clustering level and
check consistency across different clustering strategies.
"""

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from diff_diff.lwdid_exceptions import DiagnosticWarning
from diff_diff.lwdid_wild_bootstrap import wild_cluster_bootstrap


@dataclass
class ClusteringDiagnostics:
    """Result of clustering level diagnosis."""

    level: str
    se: float
    pvalue: float
    n_clusters: int
    att: float


@dataclass
class ClusteringRecommendation:
    """Recommendation for clustering level."""

    recommended_level: str
    confidence: str  # 'high', 'medium', 'low'
    rationale: str
    diagnostics: List[ClusteringDiagnostics]


def diagnose_clustering(
    y: np.ndarray,
    treatment: np.ndarray,
    candidate_cluster_vars: Dict[str, np.ndarray],
    controls: Optional[np.ndarray] = None,
    n_reps: int = 999,
    seed: Optional[int] = None,
) -> List[ClusteringDiagnostics]:
    """Diagnose clustering at multiple levels.

    Runs wild cluster bootstrap at each candidate clustering level
    and reports SE, p-value, and number of clusters.

    Parameters
    ----------
    y : ndarray (n,)
        Transformed outcome.
    treatment : ndarray (n,)
        Binary treatment indicator.
    candidate_cluster_vars : dict
        Mapping of level_name -> cluster_ids array.
        E.g., {'unit': unit_ids, 'state': state_ids, 'region': region_ids}
    controls : ndarray (n, K) or None
        Control variables.
    n_reps : int
        Bootstrap replications per level.
    seed : int or None
        Random seed.

    Returns
    -------
    List[ClusteringDiagnostics]
        One entry per candidate level, sorted by n_clusters ascending.
    """
    results = []
    for level_name, cluster_ids in candidate_cluster_vars.items():
        cluster_ids = np.asarray(cluster_ids)
        n_clusters = len(np.unique(cluster_ids))
        if n_clusters < 2:
            warnings.warn(
                f"Clustering level '{level_name}' has only {n_clusters} cluster(s); skipping.",
                DiagnosticWarning,
                stacklevel=2,
            )
            continue
        try:
            wb = wild_cluster_bootstrap(
                y,
                treatment,
                cluster_ids,
                controls=controls,
                n_reps=n_reps,
                seed=seed,
            )
            results.append(
                ClusteringDiagnostics(
                    level=level_name,
                    se=wb.se_bootstrap,
                    pvalue=wb.pvalue,
                    n_clusters=n_clusters,
                    att=wb.att,
                )
            )
        except Exception as e:
            warnings.warn(
                f"Clustering level '{level_name}' failed: {e}",
                DiagnosticWarning,
                stacklevel=2,
            )
    results.sort(key=lambda x: x.n_clusters)
    return results


def diagnose_clustering_from_data(
    data,
    outcome,
    unit,
    time,
    treatment,
    candidate_levels=None,
    **kwargs,
):
    """lwdid-py compatible wrapper for diagnose_clustering.

    Accepts a DataFrame with column names (matching lwdid-py's signature)
    and delegates to the array-based diagnose_clustering().

    Parameters
    ----------
    data : pd.DataFrame
        Panel dataset.
    outcome : str
        Outcome column name.
    unit : str
        Unit identifier column name.
    time : str
        Time period column name.
    treatment : str
        Binary treatment indicator column name.
    candidate_levels : list of str or None
        Column names to evaluate as clustering levels.
        If None, defaults to [unit].
    **kwargs
        Additional arguments passed to diagnose_clustering()
        (n_reps, seed, controls column names via 'control_cols').

    Returns
    -------
    List[ClusteringDiagnostics]
        One entry per candidate level, sorted by n_clusters ascending.
    """

    if candidate_levels is None:
        candidate_levels = [unit]

    # Extract arrays
    y_arr = data[outcome].values
    treat_arr = data[treatment].values

    # Build candidate_cluster_vars dict
    candidate_cluster_vars = {}
    for level in candidate_levels:
        if level not in data.columns:
            raise ValueError(f"Column '{level}' not found in data")
        candidate_cluster_vars[level] = data[level].values

    # Extract controls if specified
    controls = None
    control_cols = kwargs.pop("control_cols", None)
    if control_cols is not None:
        controls = data[control_cols].values

    return diagnose_clustering(
        y=y_arr,
        treatment=treat_arr,
        candidate_cluster_vars=candidate_cluster_vars,
        controls=controls,
        **kwargs,
    )


def recommend_clustering_level(
    diagnostics: List[ClusteringDiagnostics],
) -> ClusteringRecommendation:
    """Recommend clustering level based on diagnostics.

    Rule of thumb (Cameron & Miller 2015):
    - Use the highest level of clustering that still has enough clusters (G >= 20)
    - If all levels have G < 20, use the one with most clusters
    - Flag if results are sensitive to clustering level choice

    Parameters
    ----------
    diagnostics : list of ClusteringDiagnostics
        Output from diagnose_clustering().

    Returns
    -------
    ClusteringRecommendation
    """
    if not diagnostics:
        return ClusteringRecommendation(
            recommended_level="none",
            confidence="low",
            rationale="No valid clustering levels available.",
            diagnostics=[],
        )

    # Prefer levels with G >= 20
    large_enough = [d for d in diagnostics if d.n_clusters >= 20]

    if large_enough:
        # Among those with enough clusters, pick the coarsest (fewest clusters)
        # as it's more conservative
        recommended = large_enough[0]  # sorted ascending by n_clusters
        confidence = "high"
        rationale = (
            f"Level '{recommended.level}' has {recommended.n_clusters} clusters (>= 20) "
            f"and is the most conservative valid option."
        )
    else:
        # All have < 20 clusters; pick the one with most
        recommended = diagnostics[-1]
        confidence = "low"
        rationale = (
            f"All clustering levels have < 20 clusters. "
            f"'{recommended.level}' ({recommended.n_clusters} clusters) is the best available, "
            f"but inference may be unreliable. Consider wild bootstrap with Webb weights."
        )

    # Check sensitivity: are SEs consistent across levels?
    ses = [d.se for d in diagnostics if d.se > 0]
    if len(ses) >= 2:
        se_ratio = max(ses) / min(ses)
        if se_ratio > 2.0:
            confidence = "low"
            rationale += (
                f" WARNING: SE varies {se_ratio:.1f}x across levels — "
                f"results are sensitive to clustering choice."
            )

    return ClusteringRecommendation(
        recommended_level=recommended.level,
        confidence=confidence,
        rationale=rationale,
        diagnostics=diagnostics,
    )
