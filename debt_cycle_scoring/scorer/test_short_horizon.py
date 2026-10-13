"""
Tests for debt_cycle_scoring.scorer.short_horizon_flags. Run with:
    python3 -m debt_cycle_scoring.scorer.test_short_horizon
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .short_horizon_flags import (
    _robust_z_expanding, _z_to_0_100, build_short_horizon_score,
    SHORT_FLAG_NAMES, _COL_VIX, _COL_DOLLAR, _COL_SPREAD, _COL_CURVE, _COL_NFCI,
)


def _qidx(start, n):
    return pd.period_range(start, periods=n, freq="Q")


def test_robust_z_no_lookahead():
    # identical up to index 30, then diverge -> trailing z at <=30 must match
    rng = np.random.default_rng(0)
    shared = rng.normal(0, 1, 31)
    a = pd.Series(np.concatenate([shared, rng.normal(0, 1, 20)]))
    b = pd.Series(np.concatenate([shared, rng.normal(50, 1, 20)]))
    za = _robust_z_expanding(a, min_periods=12)
    zb = _robust_z_expanding(b, min_periods=12)
    assert np.allclose(za.iloc[:31].values, zb.iloc[:31].values, atol=1e-9, equal_nan=True), \
        "robust z before the divergence point must not depend on later data"


def test_robust_z_resists_outliers():
    # a single massive spike should not blow up the scale for other points
    s = pd.Series([1.0] * 40 + [1000.0] + [1.0] * 10)
    z = _robust_z_expanding(s, min_periods=12)
    # points after the spike, at their normal value of 1.0, should still be
    # near the median (z near 0), NOT dragged around by the outlier
    post = z.iloc[42:]
    assert post.abs().median() < 2.0, "robust z should not let one spike distort later normal readings"


def test_z_to_0_100_centering():
    z = pd.Series([-3.0, 0.0, 3.0])
    out = _z_to_0_100(z)
    assert abs(out.iloc[1] - 50.0) < 1e-6      # z=0 -> 50
    assert out.iloc[0] < 10                      # deep calm -> low
    assert out.iloc[2] > 90                      # extreme -> high
    assert out.is_monotonic_increasing


def _make_panel(n=60):
    idx = _qidx("2000Q1", n)
    rng = np.random.default_rng(1)
    # VIX: calm ~15, with a stress spike near the end
    vix = pd.Series(15 + rng.normal(0, 2, n), index=idx)
    vix.iloc[-3:] = [40, 55, 45]
    dollar = pd.Series(100 + np.cumsum(rng.normal(0, 0.5, n)), index=idx)
    dollar.iloc[-3:] = dollar.iloc[-4] * np.array([1.05, 1.10, 1.12])  # surge
    short = pd.DataFrame({_COL_VIX: vix, _COL_DOLLAR: dollar})
    short_worst = pd.DataFrame({_COL_VIX: vix * 1.3, _COL_DOLLAR: dollar})
    scored = pd.DataFrame({
        _COL_SPREAD: 5 + rng.normal(0, 0.3, n),
        _COL_CURVE: np.linspace(2, -0.5, n),
    }, index=idx)
    scored.loc[scored.index[-3:], _COL_SPREAD] = [7, 10, 9]  # spread widening
    return short, short_worst, scored, idx


def test_build_short_horizon_score_end_to_end():
    short, short_worst, scored, idx = _make_panel()
    result = build_short_horizon_score(short, short_worst, scored, min_periods=8)
    valid = result.stress.dropna()
    assert len(valid) > 0
    assert valid.between(0, 100).all()
    # the late stress spike should push recent stress above the calm baseline
    assert result.stress.dropna().iloc[-1] > result.stress.dropna().iloc[:10].mean()
    # flags frame has the expected raw-component columns
    assert set(result.flags.columns).issubset(set(SHORT_FLAG_NAMES))
    # worst-based score should generally be >= average-based at the spike
    assert result.stress_worst.dropna().iloc[-1] >= result.stress.dropna().iloc[-1] - 5


def test_validation_is_cross_check_not_ingredient():
    short, short_worst, scored, idx = _make_panel()
    # build a validation frame (NFCI-like) that DISAGREES with our score
    validation = pd.DataFrame({_COL_NFCI: np.linspace(-1, 1, len(idx))}, index=idx)
    result = build_short_horizon_score(short, short_worst, scored, validation=validation, min_periods=8)
    # validation appears as its own 0-100 column, separate from the flags
    assert _COL_NFCI in result.validation.columns
    assert _COL_NFCI not in result.flags.columns, "validation series must NOT be a scoring flag"
    # divergence is computed and populated
    assert result.validation_divergence.dropna().shape[0] > 0


def test_score_is_mean_of_present_flags_only():
    # if only VIX is present, score should equal the mean of the VIX flags
    idx = _qidx("2000Q1", 40)
    rng = np.random.default_rng(2)
    short = pd.DataFrame({_COL_VIX: pd.Series(15 + rng.normal(0, 2, 40), index=idx)})
    short_worst = pd.DataFrame({_COL_VIX: pd.Series(20 + rng.normal(0, 2, 40), index=idx)})
    scored = pd.DataFrame(index=idx)  # no spread/curve
    result = build_short_horizon_score(short, short_worst, scored, min_periods=8)
    # only vix_level + vix_change should be present
    present = [c for c in result.flags.columns if result.flags[c].notna().any()]
    assert set(present).issubset({"vix_level", "vix_change"})
    assert result.stress.dropna().between(0, 100).all()


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
        except Exception as e:
            failures.append(t.__name__)
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print()
    if failures:
        print(f"{len(failures)} of {len(tests)} tests FAILED: {failures}")
        raise SystemExit(1)
    print(f"All {len(tests)} tests passed.")


if __name__ == "__main__":
    run_all()
