"""
Explainability decomposition -- design doc section 3A.5.

Breaks a stage's Mahalanobis distance down into per-category contributions,
so a stage-read is never a black box: every probability shown in the app
should be traceable back to which category(ies) are driving it (e.g. "this
quarter looks close to the historical Bubble archetype mainly because of
Debt Burden & Asset Prices, not because of Central Bank Stance").
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from .stages import StageArchetype


def decompose_distance(
    x: np.ndarray,
    archetype: StageArchetype,
    category_names: List[str],
) -> pd.Series:
    """
    Mahalanobis distance^2 = diff @ inv_cov @ diff.T. Expanding that
    quadratic form gives a symmetric matrix of per-pair contributions
    (contrib[i,j] == contrib[j,i], since both diff⊗diff and inv_cov are
    symmetric). Taking each category's ROW SUM already attributes every
    off-diagonal pair evenly between the two categories involved: pair
    (i,j) contributes contrib[i,j] to category i's row sum and the equal
    value contrib[j,i] to category j's row sum, so the pair's total
    contribution (contrib[i,j] + contrib[j,i]) is split evenly without
    any extra division -- and the row sums total exactly the full squared
    distance.

    A large positive value for a category means that category's score is
    a long way (in whitened, covariance-adjusted terms) from where this
    stage's historical archetype usually sits -- i.e. it's evidence AGAINST
    this stage. Sort descending to see what's most inconsistent with this
    stage; sort ascending to see what best matches it.
    """
    diff = x - archetype.centroid
    contrib_matrix = np.outer(diff, diff) * archetype.inv_cov
    per_category = contrib_matrix.sum(axis=1)
    return pd.Series(per_category, index=category_names, name="distance_contribution").sort_values(
        ascending=False
    )
