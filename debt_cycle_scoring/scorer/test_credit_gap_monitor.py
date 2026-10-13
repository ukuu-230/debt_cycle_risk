"""
Tests for scorer.credit_gap_monitor. Run:
    python3 -m debt_cycle_scoring.scorer.test_credit_gap_monitor
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from debt_cycle_scoring.scorer.credit_gap_monitor import (
    build_credit_gap_monitor, CREDIT_RATIO_SERIES, _basel_flag,
)
from debt_cycle_scoring.labeling.credit_gap import BASEL_LOWER_THRESHOLD, BASEL_UPPER_THRESHOLD


def _qidx(start, n):
    return pd.period_range(start, periods=n, freq="Q")


def test_all_four_countries_have_verified_series():
    assert set(CREDIT_RATIO_SERIES) == {"US", "Euro", "UK", "Japan"}
    for cc, sid in CREDIT_RATIO_SERIES.items():
        assert sid.endswith("PAM770A"), f"{cc} should use a BIS credit-to-GDP series"


def test_basel_flag_thresholds():
    assert _basel_flag(12.0) == "max-warning"      # >=10
    assert _basel_flag(5.0) == "watch"             # 2-10
    assert _basel_flag(0.0) == "neutral"
    assert _basel_flag(-5.0) == "below-trend"      # <=-2
    assert _basel_flag(float("nan")) == "n/a"


def test_monitor_computes_gaps_per_country():
    # two countries, one building credit, one flat
    idx = _qidx("1990Q1", 80)
    building = pd.Series(np.concatenate([np.full(40, 100.0), 100 + np.arange(40) * 1.5]), index=idx)
    flat = pd.Series(100.0 + np.random.default_rng(0).normal(0, 0.2, 80), index=idx)
    res = build_credit_gap_monitor({"BuildLand": building, "FlatLand": flat}, min_periods=30)
    assert "BuildLand" in res.countries and "FlatLand" in res.countries
    # the building country should have a clearly higher latest gap
    assert res.countries["BuildLand"].latest_gap > res.countries["FlatLand"].latest_gap
    # and reach at least the watch flag
    assert res.countries["BuildLand"].latest_flag in ("watch", "max-warning")


def test_monitor_ranking_orders_by_latest_gap():
    idx = _qidx("1990Q1", 80)
    hi = pd.Series(np.concatenate([np.full(40, 100.0), 100 + np.arange(40) * 2.0]), index=idx)
    mid = pd.Series(np.concatenate([np.full(40, 100.0), 100 + np.arange(40) * 0.8]), index=idx)
    lo = pd.Series(100.0 + np.random.default_rng(1).normal(0, 0.2, 80), index=idx)
    res = build_credit_gap_monitor({"Lo": lo, "Hi": hi, "Mid": mid}, min_periods=30)
    ranking = res.ranking()
    names = [r[0] for r in ranking]
    assert names[0] == "Hi", f"highest gap should rank first, got {names}"
    assert names.index("Hi") < names.index("Mid") < names.index("Lo")
    # ranking rows now carry a 4th element: confidence
    assert len(ranking[0]) == 4
    assert ranking[0][3] in ("high", "moderate", "low")


def test_confidence_tier_reflects_history_length():
    from debt_cycle_scoring.scorer.credit_gap_monitor import _gap_confidence
    assert _gap_confidence(200) == "high"
    assert _gap_confidence(80) == "moderate"
    assert _gap_confidence(50) == "low"
    # a short-history country (like euro ~69 usable quarters) should be moderate,
    # a very short one low -- both flagged, never silently presented as solid
    idx = _qidx("2009Q1", 70)   # ~euro-length after burn-in
    ratio = pd.Series(np.linspace(140, 160, 70), index=idx)
    res = build_credit_gap_monitor({"ShortLand": ratio}, min_periods=30)
    assert res.countries["ShortLand"].confidence in ("low", "moderate")


def test_monitor_gap_panel_is_wide_and_comparable():
    idx = _qidx("1990Q1", 80)
    a = pd.Series(np.linspace(100, 160, 80), index=idx)
    b = pd.Series(np.linspace(100, 120, 80), index=idx)
    res = build_credit_gap_monitor({"A": a, "B": b}, min_periods=30)
    assert set(res.gap_panel.columns) == {"A", "B"}
    # panel index is quarterly and aligned
    assert isinstance(res.gap_panel.index, pd.PeriodIndex)


def test_monitor_skips_too_short_series():
    idx = _qidx("2020Q1", 10)  # only 10 quarters, < min_periods
    short = pd.Series(np.linspace(100, 110, 10), index=idx)
    res = build_credit_gap_monitor({"Short": short}, min_periods=40)
    assert "Short" not in res.countries, "series too short for the HP burn-in should be skipped"


def run_all():
    tests = [obj for name, obj in list(globals().items()) if name.startswith("test_")]
    failures = []
    for t in tests:
        try:
            t(); print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failures.append(t.__name__); print(f"FAIL  {t.__name__}: {e}")
        except Exception as e:
            failures.append(t.__name__); print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print()
    if failures:
        print(f"{len(failures)} of {len(tests)} tests FAILED: {failures}"); raise SystemExit(1)
    print(f"All {len(tests)} tests passed.")


if __name__ == "__main__":
    run_all()
