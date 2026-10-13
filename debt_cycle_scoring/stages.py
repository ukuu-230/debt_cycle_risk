"""
Stage classification via Mahalanobis-distance analog matching --
design doc section 3A.3, Method 2 (the recommended starting point,
ahead of layering in an HMM later).

Given a set of historical HIGH-CONFIDENCE-labeled quarters (from the
independent labeling pass, section 3B -- NOT Dalio's own case studies,
per the circularity problem discussed there), fit a centroid + covariance
for each long-term-cycle stage in category-score space. For any new
observation, compute the Mahalanobis distance (covariance-aware, unlike
plain Euclidean distance) to every stage's centroid, and convert those
distances into a probability distribution via softmax over negative
distance -- directly implementing the "score stages, don't declare one"
principle from section 3.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

STAGES: List[str] = [
    "early",
    "bubble",
    "top",
    "deleveraging_beautiful",
    "deleveraging_ugly",
    "reflation",
]


@dataclass
class StageArchetype:
    stage: str
    centroid: np.ndarray
    inv_cov: np.ndarray  # inverse (or pseudo-inverse) covariance, pre-computed for speed
    n_observations: int  # how many labeled quarters this archetype was fit on -- a real
    # confidence signal in its own right (e.g. "reflation" may have far fewer
    # labeled historical quarters than "bubble" or "top")


def fit_archetypes(
    category_scores: pd.DataFrame,
    labels: pd.Series,
    shrinkage: float = 0.1,
) -> Dict[str, StageArchetype]:
    """
    category_scores: DataFrame of PCA-aggregated category scores (output of
        aggregate.py), rows = time. Caller must pre-filter to HIGH-CONFIDENCE
        labeled quarters only (section 3B.4) before calling this -- this
        function doesn't know about label confidence tiers itself.
    labels: Series aligned to category_scores.index, one value per row,
        drawn from STAGES.
    shrinkage: how much to shrink each stage's sample covariance toward a
        diagonal (Ledoit-Wolf style) target. This matters a lot here because
        some stages will have very few labeled historical quarters -- a raw
        sample covariance from a handful of points is unreliable and can
        make Mahalanobis distance behave erratically.
    """
    archetypes: Dict[str, StageArchetype] = {}
    global_var = category_scores.var(numeric_only=True).values
    global_mean = category_scores.mean(numeric_only=True).values

    for stage in labels.dropna().unique():
        rows = category_scores.loc[labels == stage].dropna()

        if len(rows) < 2:
            # Too few labeled examples to fit any covariance at all -- fall
            # back to the global variance so the archetype still exists
            # (with a very wide, low-confidence distance scale) rather than
            # silently vanishing from the model. n_observations records this
            # so the UI/backtest can flag it rather than hide it.
            centroid = rows.mean(axis=0).values if len(rows) else global_mean
            cov = np.diag(global_var)
        else:
            centroid = rows.mean(axis=0).values
            sample_cov = np.atleast_2d(np.cov(rows.values, rowvar=False))
            # np.cov collapses to a 0-d scalar (not a 1x1 matrix) when there's
            # only a single category/feature column -- np.atleast_2d guards
            # against that edge case, which otherwise crashes np.diag() below.
            # (Caught by the labeling/demo.py integration demo, which fits
            # archetypes on a single "debt_burden" category.)
            target = np.diag(np.diag(sample_cov))
            cov = (1 - shrinkage) * sample_cov + shrinkage * target

        cov = cov + np.eye(cov.shape[0]) * 1e-6  # regularize before inverting
        inv_cov = np.linalg.pinv(cov)

        archetypes[stage] = StageArchetype(
            stage=stage,
            centroid=centroid,
            inv_cov=inv_cov,
            n_observations=len(rows),
        )
    return archetypes


def mahalanobis_distance(x: np.ndarray, archetype: StageArchetype) -> float:
    diff = x - archetype.centroid
    return float(np.sqrt(diff @ archetype.inv_cov @ diff.T))


def stage_probabilities(
    x: np.ndarray,
    archetypes: Dict[str, StageArchetype],
    temperature: float = 1.0,
) -> Dict[str, float]:
    """
    Convert per-stage Mahalanobis distances into a probability distribution
    via softmax over negative distance.

    `temperature` controls how sharply peaked the output is: higher
    temperature spreads probability out more (a deliberately more
    "uncertain" read). Section 7's calibration test (Brier score /
    reliability diagram) is exactly what should be used to tune this
    parameter -- don't hand-pick it before that backtest exists.
    """
    stages = list(archetypes.keys())
    distances = np.array([mahalanobis_distance(x, archetypes[s]) for s in stages])
    neg_scaled = -distances / temperature
    neg_scaled -= neg_scaled.max()  # numerical stability, doesn't change the result
    weights = np.exp(neg_scaled)
    probs = weights / weights.sum()
    return dict(zip(stages, probs))
