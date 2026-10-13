"""
Scoring engine orchestrator -- the integration layer that has never
existed before. Runs the full pipeline end to end:

    raw quarterly panel (from ingestion)
      -> normalize each scored metric (percentile rank + direction encode)
      -> aggregate metrics into per-category PCA scores
      -> fit Mahalanobis stage archetypes on the TRAINING-ELIGIBLE labels
         (from the labeling pipeline's training mask)
      -> score EVERY quarter into a stage-probability distribution
      -> attach a 0-100 risk score per quarter

Country-aware and pooled from day one: the panel carries a `country`
column (just "US" for now). Archetypes are fit over pooled rows
regardless of country, so adding countries later is adding rows.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..normalize import winsorize, expanding_percentile_rank, direction_encode
from ..aggregate import pca_category_score
from ..stages import STAGES
from .risk_score import compute_risk_score


# Stages the scorer will actually try to fit. "top" is deliberately
# excluded per the labeling decision (no clean independent training
# signal) -- kept here as a single source of truth so the engine and any
# report agree on which stages are modellable.
TRAINABLE_STAGES: List[str] = [s for s in STAGES if s != "top"]

# Below this many training examples, a stage's archetype covariance is
# too thin to trust; the engine still fits it (so the stage can be
# predicted at all) but flags it so the report can caveat it rather than
# present a fragile probability as if it were solid. bubble=3 on US data
# is exactly this case.
LOW_CONFIDENCE_ARCHETYPE_THRESHOLD = 8

# A quarter scored on fewer than this many categories is not scored at
# all (returns NaN). A 1-category Mahalanobis "distance" is just a
# z-score and gives false precision; requiring >=2 keeps the distance
# genuinely multivariate.
MIN_CATEGORIES_TO_SCORE = 2

# Ledoit-Wolf-style shrinkage toward a diagonal target, applied when
# (re-)building a per-stage covariance. Matches the default in
# stages.fit_archetypes so behavior is consistent.
_SHRINKAGE = 0.1


def _mahalanobis_on_subset(
    x: np.ndarray,
    stage_rows: np.ndarray,
    global_var: np.ndarray,
) -> float:
    """
    Mahalanobis distance from point x to the centroid of stage_rows,
    using ONLY the categories present in x/stage_rows (both already
    sliced to the same subset of columns by the caller).

    Rebuilds and inverts the covariance on the fly for this specific
    subset of categories -- which is why the engine keeps the raw
    per-stage training rows rather than a single pre-inverted matrix:
    you cannot correctly slice an already-inverted covariance down to a
    sub-dimension, you have to slice the raw covariance and re-invert.
    """
    centroid = stage_rows.mean(axis=0)
    if stage_rows.shape[0] < 2:
        cov = np.diag(global_var)
    else:
        sample_cov = np.atleast_2d(np.cov(stage_rows, rowvar=False))
        target = np.diag(np.diag(sample_cov))
        cov = (1 - _SHRINKAGE) * sample_cov + _SHRINKAGE * target
    cov = cov + np.eye(cov.shape[0]) * 1e-6
    inv_cov = np.linalg.pinv(cov)
    diff = x - centroid
    return float(np.sqrt(diff @ inv_cov @ diff.T))


@dataclass
class ScorerResult:
    stage_probabilities: pd.DataFrame       # index=quarter, columns=stages, values=prob
    top_stage: pd.Series                     # index=quarter, most-likely stage
    risk_score: pd.Series                    # index=quarter, 0-100
    risk_stage_component: pd.Series          # index=quarter, the stage part of the risk score
    risk_gap_component: pd.Series            # index=quarter, the gap part
    categories_used: pd.Series               # index=quarter, how many categories fed each read (evidence depth)
    evidence_confidence: pd.Series           # index=quarter, categories_used / total categories, in (0,1]
    category_scores: pd.DataFrame            # the PCA category scores that fed the classifier
    archetype_n: Dict[str, int]              # stage -> #training examples it was fit on
    low_confidence_stages: List[str]         # stages whose archetype is below the trust threshold
    dropped_stages: List[str]                # stages requested for training but lost to history burn-in
    pca_explained_variance: Dict[str, float] # category -> PC1 explained-variance ratio (a data-quality read)
    availability_patterns: Dict[str, int]    # reserved for per-pattern reporting; empty for now


def prepare_scored_panel(
    scored_metrics: pd.DataFrame,
    metric_specs: Dict[str, object],
    country: str = "US",
    min_periods: int = 20,
) -> "tuple[pd.DataFrame, Dict[str, float]]":
    """
    Normalize each scored metric (winsorize -> expanding percentile rank
    -> direction encode) and aggregate into per-category PCA scores.

    metric_specs: name -> spec object exposing `.category` and
        `.higher_is_later_cycle` (works for both MetricSpec from the
        catalog and DerivedMetricSpec from the derived-metrics layer,
        since both expose those attributes).

    Returns (category_scores, pca_explained_variance). category_scores is
    indexed by quarter with one column per category; it carries no country
    column itself (that's attached by the caller when pooling), but the
    country arg is accepted now so the signature is stable once pooling
    lands.
    """
    # 1. normalize each metric
    normalized_by_category: Dict[str, pd.DataFrame] = {}
    for name in scored_metrics.columns:
        spec = metric_specs.get(name)
        if spec is None or getattr(spec, "higher_is_later_cycle", None) is None:
            # not a scorable metric (reference/label series slipped in) -- skip
            continue
        raw = scored_metrics[name].dropna()
        if len(raw) < min_periods + 1:
            continue
        pct = expanding_percentile_rank(winsorize(raw), min_periods=min_periods)
        encoded = direction_encode(pct, higher_is_later_cycle=spec.higher_is_later_cycle)
        category = spec.category
        normalized_by_category.setdefault(category, pd.DataFrame(index=scored_metrics.index))
        normalized_by_category[category][name] = encoded

    # 2. aggregate each category via PCA
    category_scores = pd.DataFrame(index=scored_metrics.index)
    pca_explained_variance: Dict[str, float] = {}
    for category, metrics_df in normalized_by_category.items():
        # need at least one column with enough overlapping data
        if metrics_df.dropna(how="all").shape[0] < 2 or metrics_df.shape[1] < 1:
            continue
        try:
            result = pca_category_score(metrics_df)
        except ValueError:
            # not enough overlapping non-NaN data to fit this category's PCA
            continue
        category_scores[category] = result.score
        pca_explained_variance[category] = result.explained_variance_ratio

    return category_scores, pca_explained_variance


def fit_and_score(
    scored_metrics: pd.DataFrame,
    metric_specs: Dict[str, object],
    stage_labels: pd.Series,
    training_mask: pd.Series,
    credit_gap: Optional[pd.Series] = None,
    country: str = "US",
    temperature: float = 0.5,
    risk_stage_weight: float = 0.6,
) -> ScorerResult:
    """
    The full fit-and-score run.

    scored_metrics / metric_specs: the raw quarterly panel and specs
        (from ingestion + derived-metrics layers).
    stage_labels: descriptive stage label per quarter (from
        stage_labels_from_consensus).
    training_mask: boolean per quarter -- True where the label may be
        used as training ground truth (from the labeling training mask).
        ONLY these quarters are used to fit archetypes.
    credit_gap: per-quarter raw credit-to-GDP gap for the risk score's
        gap component. Optional; risk falls back to stage-only if absent.
    country: stored for pooling; single-country for now.
    temperature / risk_stage_weight: passed through to the softmax and
        the risk blend respectively.
    """
    category_scores, pca_ev = prepare_scored_panel(
        scored_metrics, metric_specs, country=country
    )
    all_categories = list(category_scores.columns)
    if not all_categories:
        raise ValueError("No categories could be built from the panel -- cannot score.")

    # A quarter is SCORABLE if it has at least MIN_CATEGORIES_TO_SCORE
    # non-NaN categories -- NOT all of them. This is the core of variable-
    # dimensionality scoring: one late-starting category (e.g. asset_prices
    # only fully present from ~2020 because of SP500's rolling window) must
    # not veto 55 years of otherwise-scorable history.
    n_cats_per_quarter = category_scores.notna().sum(axis=1)
    scorable_index = category_scores.index[n_cats_per_quarter >= MIN_CATEGORIES_TO_SCORE]
    if len(scorable_index) == 0:
        raise ValueError(
            f"No quarter has at least {MIN_CATEGORIES_TO_SCORE} categories -- cannot score."
        )

    # Global per-category variance (over all scorable quarters), used as a
    # fallback covariance when a stage has too few training rows to
    # estimate its own.
    global_var_by_cat = category_scores.loc[scorable_index].var(numeric_only=True)

    # --- gather per-stage training rows (NOT a pre-inverted covariance) ---
    train_idx_full = training_mask.reindex(category_scores.index, fill_value=False)
    train_labels_full = stage_labels.reindex(category_scores.index)
    stage_training_rows: Dict[str, pd.DataFrame] = {}
    for stage in TRAINABLE_STAGES:
        stage_mask = train_idx_full & (train_labels_full == stage)
        rows = category_scores.loc[stage_mask]
        # a training row is only usable for a given category if that
        # category is non-NaN in it; we keep the full rows here and slice
        # per-category at scoring time.
        rows = rows.loc[rows.index.intersection(scorable_index)]
        if len(rows) > 0:
            stage_training_rows[stage] = rows

    if not stage_training_rows:
        raise ValueError("No training-eligible quarters after alignment -- cannot fit archetypes.")

    archetype_n = {stage: len(rows) for stage, rows in stage_training_rows.items()}
    low_confidence_stages = [
        stage for stage, n in archetype_n.items() if n < LOW_CONFIDENCE_ARCHETYPE_THRESHOLD
    ]

    original_requested = set(
        stage_labels[training_mask.reindex(stage_labels.index, fill_value=False)].dropna()
    )
    requested_trainable = {s for s in original_requested if s in TRAINABLE_STAGES}
    dropped_stages = sorted(requested_trainable - set(stage_training_rows.keys()))

    # --- score EVERY scorable quarter on ITS available categories ---
    prob_rows = {}
    n_cats_used = {}
    for period in scorable_index:
        row = category_scores.loc[period]
        present = [c for c in all_categories if pd.notna(row[c])]
        if len(present) < MIN_CATEGORIES_TO_SCORE:
            continue
        x = row[present].values.astype(float)
        n_cats_used[period] = len(present)

        stage_dists = {}
        for stage, rows in stage_training_rows.items():
            # restrict this stage's training rows to the SAME categories
            # present this quarter, keeping only training rows that have
            # all of them.
            sub = rows[present].dropna()
            if len(sub) < 1:
                continue  # this stage can't be scored in this subspace
            gv = global_var_by_cat[present].values.astype(float)
            stage_dists[stage] = _mahalanobis_on_subset(x, sub.values.astype(float), gv)

        if not stage_dists:
            continue
        # softmax over negative distance (same as stages.stage_probabilities)
        stages_here = list(stage_dists.keys())
        dists = np.array([stage_dists[s] for s in stages_here])
        neg = -dists / temperature
        neg -= neg.max()
        w = np.exp(neg)
        probs = w / w.sum()
        prob_rows[period] = dict(zip(stages_here, probs))

    if not prob_rows:
        raise ValueError("No quarter could be scored after subspace alignment.")

    prob_df = pd.DataFrame.from_dict(prob_rows, orient="index")
    prob_df = prob_df.reindex(columns=[s for s in STAGES if s in prob_df.columns])
    prob_df = prob_df.fillna(0.0)  # a stage absent from a quarter's subspace = 0 prob that quarter

    top_stage = prob_df.idxmax(axis=1)
    n_categories = pd.Series(n_cats_used)
    total_categories = len(all_categories)
    # Evidence confidence in (0,1]: a quarter scored on 2 of 4 categories
    # is a genuinely thinner read than one scored on all 4. This is an
    # explicit design decision -- thin-evidence quarters get their risk
    # score pulled toward a neutral baseline so a 2-category read never
    # looks as authoritative as a 4-category one.
    evidence_confidence = (n_categories / total_categories).clip(upper=1.0)

    # --- risk score per quarter (then evidence-adjusted) ---
    NEUTRAL_RISK = 50.0  # the baseline a zero-evidence read would collapse toward
    risk_vals, stage_comp, gap_comp = {}, {}, {}
    for period in prob_df.index:
        probs = prob_df.loc[period].to_dict()
        gap_val = None
        if credit_gap is not None and period in credit_gap.index:
            gap_val = credit_gap.loc[period]
        rr = compute_risk_score(probs, gap_val, stage_weight=risk_stage_weight)
        # pull the raw risk toward NEUTRAL_RISK in proportion to how little
        # evidence backed it: full evidence (conf=1) leaves it unchanged; a
        # 2-of-4 read (conf=0.5) moves it halfway to neutral.
        conf = float(evidence_confidence.get(period, 1.0))
        adjusted = conf * rr.risk_score + (1.0 - conf) * NEUTRAL_RISK
        risk_vals[period] = round(adjusted, 1)
        stage_comp[period] = rr.stage_component
        gap_comp[period] = rr.gap_component

    return ScorerResult(
        stage_probabilities=prob_df,
        top_stage=top_stage,
        risk_score=pd.Series(risk_vals),
        risk_stage_component=pd.Series(stage_comp),
        risk_gap_component=pd.Series(gap_comp),
        categories_used=n_categories,
        evidence_confidence=evidence_confidence,
        category_scores=category_scores.loc[prob_df.index],
        archetype_n=archetype_n,
        low_confidence_stages=low_confidence_stages,
        dropped_stages=dropped_stages,
        pca_explained_variance=pca_ev,
        availability_patterns={},
    )
