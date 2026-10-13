"""
debt_cycle_scoring
===================

Reference implementation of the scoring math described in
`debt-cycle-framework-design-doc.md`, section 3A.

This package is deliberately decoupled from data ingestion (FRED/BIS
API pulls). It takes a plain pandas DataFrame of already-fetched raw
metric time series and turns it into stage probabilities. That keeps
the scoring math independently testable and independently backtestable
(section 7 of the design doc), before any API/ingestion code exists.

Pipeline, matching the design doc section numbers:

    raw metrics
        -> normalize.py        (3A.1: percentile rank / shrinkage z-score,
                                 winsorize, direction-encode)
        -> aggregate.py         (3A.2: PCA per category)
        -> stages.py            (3A.3: Mahalanobis analog matching ->
                                 stage probability distribution)
        -> explain.py           (3A.5: per-category distance decomposition)
"""

from .normalize import (
    winsorize,
    expanding_percentile_rank,
    shrinkage_zscore,
    direction_encode,
)
from .aggregate import pca_category_score, CategoryScoreResult
from .stages import fit_archetypes, stage_probabilities, mahalanobis_distance, STAGES
from .explain import decompose_distance

__all__ = [
    "winsorize",
    "expanding_percentile_rank",
    "shrinkage_zscore",
    "direction_encode",
    "pca_category_score",
    "CategoryScoreResult",
    "fit_archetypes",
    "stage_probabilities",
    "mahalanobis_distance",
    "decompose_distance",
    "STAGES",
]
