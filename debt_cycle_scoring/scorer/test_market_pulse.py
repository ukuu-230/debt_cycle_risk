"""
Tests for scorer.market_pulse. Run:
    python -m debt_cycle_scoring.scorer.test_market_pulse
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from debt_cycle_scoring.scorer.market_pulse import build_market_pulse


def _daily(values, end="2026-09-30"):
    idx = pd.bdate_range(end=end, periods=len(values))
    return pd.Series(values, index=idx, dtype=float)


def test_rate_changes_are_in_basis_points():
    s = _daily(np.r_[np.full(40, 4.00), 4.10])      # +10bp on the last day
    item = {p.sid: p for p in build_market_pulse({"DGS10": s})}["DGS10"]
    assert abs(item.chg_1d - 10.0) < 1e-6 and item.chg_unit == "bp"


def test_wider_spread_is_toward_risk_and_steeper_curve_is_not():
    widen = _daily(np.linspace(3.0, 4.0, 60))         # HY OAS widening
    steepen = _daily(np.linspace(0.0, 1.0, 60))       # curve steepening
    p = {x.sid: x for x in build_market_pulse({"BAA10Y": widen, "T10Y2Y": steepen})}
    assert p["BAA10Y"].toward_risk_1m is True
    assert p["T10Y2Y"].toward_risk_1m is False


def test_breakeven_is_two_sided():
    s = _daily(np.linspace(2.0, 3.0, 60))
    assert build_market_pulse({"T10YIE": s})[0].toward_risk_1m is None


def test_sp500_shown_as_drawdown_from_52w_high():
    s = _daily(np.r_[np.full(100, 100.0), 90.0])      # 10% below the high
    item = build_market_pulse({"SP500": s})[0]
    assert abs(item.latest - (-10.0)) < 1e-6 and item.level_close == 90.0
    assert item.toward_risk_1m is True                # deeper drawdown = risk


def test_holiday_gaps_and_short_series_handled():
    s = _daily(np.linspace(1, 2, 60)); s.iloc[-3] = np.nan   # FRED "." holiday
    assert len(build_market_pulse({"VIXCLS": s})) == 1
    assert build_market_pulse({"VIXCLS": _daily(np.ones(10))}) == []   # too short -> skipped
    assert build_market_pulse({}) == []                                 # missing -> skipped


def test_percentile_uses_trailing_window_only():
    s = _daily(np.r_[np.full(300, 1.0), 5.0])
    assert build_market_pulse({"VIXCLS": s})[0].pct_10y == 100


def test_short_history_is_reported_not_presented_as_10y():
    s = _daily(np.linspace(1, 2, 3 * 252))           # ~3 years, like FRED's ICE series
    assert build_market_pulse({"VIXCLS": s})[0].pct_years < 3.5


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
