"""
Category aggregation -- design doc section 3A.2.

Turns a set of correlated, direction-encoded [0,1] metric scores within
one category (e.g. "Debt Burden & Growth") into a single category score,
using PCA rather than hand-picked weights, so the weighting reflects the
metrics' actual covariance structure and stays reportable via loadings
("this category's score is currently driven mostly by X and Y").
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class CategoryScoreResult:
    score: pd.Series                  # category score in [0, 1], PC1 rescaled
    loadings: pd.Series                # each metric's loading on PC1 -- the explainability hook
    explained_variance_ratio: float    # how "unidimensional" this category's signal actually is


def pca_category_score(metric_scores: pd.DataFrame) -> CategoryScoreResult:
    """
    metric_scores: DataFrame of direction-encoded [0,1] metric scores for
    ONE category, columns = metrics, rows = time (already normalized via
    normalize.py). Rows with any NaN are dropped before fitting PC1, so a
    metric with a lot of missing history simply contributes less overall
    rather than crashing the whole category.

    A low `explained_variance_ratio` (e.g. well under ~50%) is itself a
    useful diagnostic: it means this category isn't really one coherent
    signal and might be worth splitting into two sub-scores rather than
    forced into a single number -- flag it in the methodology page rather
    than silently trusting PC1.
    """
    clean = metric_scores.dropna()
    if clean.shape[0] < 2 or clean.shape[1] < 1:
        raise ValueError("Not enough overlapping, non-NaN data to fit a category PCA")

    columns = clean.columns
    X = clean.values
    Xc = X - X.mean(axis=0)

    # SVD-based PCA -- avoids adding scikit-learn as a dependency for one matrix factorization.
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    pc1 = Vt[0]
    projection = Xc @ pc1

    # PCA's sign is arbitrary (PC1 and -PC1 explain identical variance), so
    # orient it deliberately: higher projection should mean "more of this
    # category's metrics are reading high" (i.e. more late-cycle/stressed),
    # not the reverse -- otherwise the category score could come out inverted
    # depending on floating-point happenstance.
    row_mean = Xc.mean(axis=1)
    if np.corrcoef(projection, row_mean)[0, 1] < 0:
        pc1 = -pc1
        projection = -projection

    explained_variance_ratio = float((S[0] ** 2) / (S ** 2).sum())

    p_min, p_max = projection.min(), projection.max()
    if p_max > p_min:
        scaled = (projection - p_min) / (p_max - p_min)
    else:
        scaled = np.zeros_like(projection)

    score = pd.Series(np.nan, index=metric_scores.index, name="category_score")
    score.loc[clean.index] = scaled

    loadings = pd.Series(pc1, index=columns, name="pc1_loading")

    return CategoryScoreResult(
        score=score,
        loadings=loadings,
        explained_variance_ratio=explained_variance_ratio,
    )
