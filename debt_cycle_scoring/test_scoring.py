"""
Lightweight correctness tests for the scoring math -- plain asserts,
no external test framework dependency. Run with:
    python3 -m debt_cycle_scoring.test_scoring

These check specific numerical properties claimed in the design doc
(no-look-ahead, direction consistency, distance = 0 at a stage's own
centroid, probabilities summing to 1) rather than just "does it run".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .normalize import (
    winsorize,
    expanding_percentile_rank,
    shrinkage_zscore,
    direction_encode,
)
from .aggregate import pca_category_score
from .stages import fit_archetypes, stage_probabilities, mahalanobis_distance
from .explain import decompose_distance


def test_winsorize_clips_outliers():
    s = pd.Series([1, 2, 3, 4, 5, 1000])
    w = winsorize(s, lower_pct=0.0, upper_pct=0.8)
    assert w.max() < 1000, "winsorize should clip the extreme outlier"
    assert w.iloc[0] == 1, "winsorize should leave in-range values untouched"


def test_percentile_rank_has_no_lookahead():
    # A strictly increasing series: at any point T, the percentile of the
    # value at T against all history up to and including T should be 1.0
    # (it's the max seen so far), NOT influenced by later, larger values.
    s = pd.Series(range(1, 51), dtype=float)
    ranks = expanding_percentile_rank(s, min_periods=5)
    # From min_periods onward, every point is the running max -> percentile 1.0
    assert (ranks.dropna() == 1.0).all(), (
        "a monotonically increasing series should rank at the 100th percentile "
        "at every point, since each value is the max of its own history"
    )
    assert ranks.iloc[:5].isna().all(), "should be NaN before min_periods trailing observations exist"


def test_percentile_rank_matches_known_case():
    # [10, 20, 30, 40, 50]: value 30 (index 2) with history [10, 20, 30] ->
    # 3 of 3 <= 30 -> percentile 1.0 (current point counts as "≤ itself" by
    # construction of the window in this implementation).
    s = pd.Series([10.0, 20.0, 30.0, 20.0, 10.0])
    ranks = expanding_percentile_rank(s, min_periods=2)
    # At index 3 (value 20), history is [10, 20, 30]; 2 of 3 <= 20 -> 0.667
    assert abs(ranks.iloc[3] - (2 / 3)) < 1e-9, f"expected 0.667, got {ranks.iloc[3]}"
    # At index 4 (value 10), history is [10, 20, 30, 20]; 1 of 4 <= 10 -> 0.25
    assert abs(ranks.iloc[4] - 0.25) < 1e-9, f"expected 0.25, got {ranks.iloc[4]}"


def test_direction_encode_flips_correctly():
    p = pd.Series([0.2, 0.8])
    assert (direction_encode(p, higher_is_later_cycle=True) == p).all()
    flipped = direction_encode(p, higher_is_later_cycle=False)
    assert np.allclose(flipped.values, [0.8, 0.2])


def test_shrinkage_zscore_converges_to_own_history_over_time():
    rng = np.random.default_rng(0)
    s = pd.Series(rng.normal(5, 1, 200))
    result = shrinkage_zscore(s, prior_mean=0.0, prior_std=10.0, full_credibility_n=20)
    early_weight = result.weight_on_own_history.iloc[5]
    late_weight = result.weight_on_own_history.iloc[-1]
    assert early_weight < late_weight, "credibility weight on own history should grow with sample size"
    assert late_weight > 0.9, "with 200 observations vs. full_credibility_n=20, own history should dominate"


def test_pca_category_score_is_bounded_and_oriented():
    idx = pd.RangeIndex(50)
    # two perfectly correlated metrics rising over time
    metric_a = pd.Series(np.linspace(0, 1, 50), index=idx)
    metric_b = metric_a * 0.9 + 0.05
    df = pd.DataFrame({"a": metric_a, "b": metric_b})
    result = pca_category_score(df)
    assert result.score.min() >= -1e-9 and result.score.max() <= 1 + 1e-9, "category score should be in [0,1]"
    assert result.explained_variance_ratio > 0.99, "two near-identical metrics should be ~100% one-dimensional"
    # oriented correctly: score should rise as the underlying metrics rise
    assert result.score.iloc[-1] > result.score.iloc[0]


def test_mahalanobis_distance_zero_at_own_centroid():
    idx = pd.RangeIndex(20)
    scores = pd.DataFrame(
        {"cat1": np.linspace(0, 1, 20), "cat2": np.linspace(1, 0, 20)}, index=idx
    )
    labels = pd.Series(["bubble"] * 20, index=idx)
    archetypes = fit_archetypes(scores, labels)
    centroid = archetypes["bubble"].centroid
    d = mahalanobis_distance(centroid, archetypes["bubble"])
    assert d < 1e-6, "distance from a stage's own centroid to itself should be ~0"


def test_fit_archetypes_single_category_does_not_crash():
    # Regression test: np.cov on a single-column array returns a 0-d
    # scalar rather than a 1x1 matrix, which used to crash np.diag()
    # inside fit_archetypes. Caught by the labeling/demo.py integration
    # demo (which fits archetypes on a single "debt_burden" category),
    # not by any test that only exercised the multi-category case.
    idx = pd.RangeIndex(20)
    scores = pd.DataFrame({"only_category": np.linspace(0, 1, 20)}, index=idx)
    labels = pd.Series(["early"] * 10 + ["bubble"] * 10, index=idx)
    archetypes = fit_archetypes(scores, labels)  # should not raise
    assert set(archetypes.keys()) == {"early", "bubble"}
    probs = stage_probabilities(np.array([0.9]), archetypes)
    assert abs(sum(probs.values()) - 1.0) < 1e-9


def test_stage_probabilities_sum_to_one():
    idx = pd.RangeIndex(30)
    scores = pd.DataFrame(
        {"cat1": np.concatenate([np.zeros(15), np.ones(15)]), "cat2": np.linspace(0, 1, 30)},
        index=idx,
    )
    labels = pd.Series(["early"] * 15 + ["bubble"] * 15, index=idx)
    archetypes = fit_archetypes(scores, labels)
    probs = stage_probabilities(np.array([0.5, 0.5]), archetypes, temperature=1.0)
    assert abs(sum(probs.values()) - 1.0) < 1e-9, "stage probabilities must sum to 1"
    assert all(p >= 0 for p in probs.values()), "no probability should be negative"


def test_decomposition_sums_to_total_distance_squared():
    idx = pd.RangeIndex(20)
    scores = pd.DataFrame(
        {"cat1": np.linspace(0, 1, 20), "cat2": np.linspace(1, 0, 20)}, index=idx
    )
    labels = pd.Series(["bubble"] * 10 + ["top"] * 10, index=idx)
    archetypes = fit_archetypes(scores, labels)
    x = np.array([0.9, 0.1])
    decomposition = decompose_distance(x, archetypes["top"], ["cat1", "cat2"])
    total_from_decomp = decomposition.sum()
    total_direct = mahalanobis_distance(x, archetypes["top"]) ** 2
    assert abs(total_from_decomp - total_direct) < 1e-9, (
        "per-category contributions should sum exactly back to the total squared distance"
    )


def run_all():
    tests = [obj for name, obj in list(globals().items()) if name.startswith("test_")]
    failures = []
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failures.append(t.__name__)
            print(f"FAIL  {t.__name__}: {e}")
    print()
    if failures:
        print(f"{len(failures)} of {len(tests)} tests FAILED: {failures}")
        raise SystemExit(1)
    print(f"All {len(tests)} tests passed.")


if __name__ == "__main__":
    run_all()
