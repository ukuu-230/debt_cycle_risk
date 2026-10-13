"""
Tests for debt_cycle_scoring.scorer.risk_flags. Run with:
    python3 -m debt_cycle_scoring.scorer.test_risk_flags
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .risk_flags import (
    flag_credit_gap, flag_debt_service_ratio, flag_yield_curve, flag_credit_spread,
    flag_m2_velocity, flag_long_bond_level, flag_debasement,
    regime_from_gap, compute_composite_risk, RESEARCH_WEIGHTS, FLAG_NAMES,
)


# ---- individual flags ------------------------------------------------

def test_credit_gap_flag_monotonic_and_anchored():
    assert flag_credit_gap(-5.0) == 0.0
    assert flag_credit_gap(10.0) == 100.0
    assert flag_credit_gap(2.0) < flag_credit_gap(6.0) < flag_credit_gap(10.0)
    # at the lower Basel threshold it should be modest (~20)
    assert 15 <= flag_credit_gap(2.0) <= 25


def test_yield_curve_flag_inversion_is_high_risk():
    assert flag_yield_curve(2.5) == 0.0          # steep = no risk
    assert flag_yield_curve(0.0) == 60.0          # flat = elevated
    assert flag_yield_curve(-1.0) == 100.0        # deep inversion = max
    assert flag_yield_curve(-0.5) > flag_yield_curve(0.5)


def test_m2_velocity_flag_is_inverted():
    # LOW velocity = high risk
    assert flag_m2_velocity(0.1) > flag_m2_velocity(0.9)
    assert flag_m2_velocity(0.0) == 100.0
    assert flag_m2_velocity(1.0) == 0.0


def test_debasement_flag_is_two_tailed():
    # Risk lowest at the ~2% target, rising for BOTH high inflation
    # expectations (debasement) and low/negative ones (deflation risk).
    assert flag_debasement(2.0) == 0.0                      # at target -> no risk
    assert flag_debasement(3.5) == 60.0                     # +1.5 from target -> 60
    assert flag_debasement(0.5) == 60.0                     # -1.5 from target -> 60 (symmetric)
    assert flag_debasement(5.0) == 100.0                    # +3 -> max (runaway inflation)
    assert flag_debasement(-1.0) == 100.0                   # -3 -> max (deflation)
    # both directions increase risk away from target
    assert flag_debasement(4.0) > flag_debasement(3.0)
    assert flag_debasement(1.0) > flag_debasement(1.8)


def test_percentile_flags_scale_to_100():
    assert flag_debt_service_ratio(0.5) == 50.0
    assert flag_credit_spread(0.9) == 90.0
    assert flag_long_bond_level(0.0) == 0.0


def test_all_flags_return_nan_on_nan_input():
    for f in [flag_credit_gap, flag_debt_service_ratio, flag_yield_curve,
              flag_credit_spread, flag_m2_velocity, flag_long_bond_level, flag_debasement]:
        assert np.isnan(f(np.nan))


# ---- regime selection ------------------------------------------------

def test_regime_from_gap():
    assert regime_from_gap(-1.0) == "calm"
    assert regime_from_gap(1.9) == "calm"
    assert regime_from_gap(2.0) == "building"
    assert regime_from_gap(9.9) == "building"
    assert regime_from_gap(10.0) == "elevated"
    assert regime_from_gap(np.nan) == "building"  # neutral default


def test_research_weights_sum_to_one_per_regime():
    for regime, w in RESEARCH_WEIGHTS.items():
        assert abs(sum(w.values()) - 1.0) < 1e-9, f"{regime} weights must sum to 1"
        assert set(w.keys()) == set(FLAG_NAMES), f"{regime} must weight every flag"


def test_building_regime_weights_gap_heaviest():
    # per BIS WP421: credit gap is the top long-horizon signal in build-up
    w = RESEARCH_WEIGHTS["building"]
    assert w["credit_gap"] == max(w.values())


def test_elevated_regime_shifts_toward_short_horizon_flags():
    # DSR + curve should gain weight relative to the building regime
    building, elevated = RESEARCH_WEIGHTS["building"], RESEARCH_WEIGHTS["elevated"]
    assert elevated["debt_service_ratio"] > building["debt_service_ratio"]
    assert elevated["yield_curve"] > building["yield_curve"]
    assert elevated["credit_gap"] < building["credit_gap"]


# ---- composite -------------------------------------------------------

def test_composite_all_flags_high_gives_high_risk():
    flags = {name: 90.0 for name in FLAG_NAMES}
    result = compute_composite_risk(flags, gap=8.0)
    assert result.research_weighted > 80
    assert result.equal_weighted == 90.0
    assert result.n_flags_present == len(FLAG_NAMES)


def test_composite_all_flags_low_gives_low_risk():
    flags = {name: 5.0 for name in FLAG_NAMES}
    result = compute_composite_risk(flags, gap=-1.0)
    assert result.research_weighted < 15
    assert result.regime == "calm"


def test_composite_drops_missing_flags_and_renormalizes():
    # only two flags present -> weights renormalize over those two
    flags = {"credit_gap": 80.0, "yield_curve": 40.0}
    result = compute_composite_risk(flags, gap=6.0)
    assert result.n_flags_present == 2
    # research-weighted must lie between the two flag values
    assert 40.0 <= result.research_weighted <= 80.0
    # equal-weighted is the simple mean
    assert result.equal_weighted == 60.0


def test_composite_divergence_is_reported():
    # a regime that weights one high flag heavily should diverge from equal
    flags = {"credit_gap": 100.0, "debasement": 0.0}
    result = compute_composite_risk(flags, gap=6.0)  # building: gap weighted >> debasement
    assert result.divergence > 0
    # research-weighted should be pulled toward the heavily-weighted gap (high)
    assert result.research_weighted > result.equal_weighted


def test_composite_all_missing_returns_nan():
    result = compute_composite_risk({"credit_gap": np.nan}, gap=np.nan)
    assert np.isnan(result.research_weighted)
    assert result.n_flags_present == 0


def test_composite_below_min_flags_suppresses_score_but_reports_count():
    # A single-flag "composite" is not a composite -- it must return NaN
    # (no false-authority score) while still reporting n_flags_present=1.
    # This is the fix for deep-history quarters (e.g. euro pre-2008, US
    # earliest years) scoring on one lonely flag and pinning at extremes.
    one_flag = {"long_bond_level": 95.0}
    r = compute_composite_risk(one_flag, gap=np.nan, min_flags=2)
    assert np.isnan(r.research_weighted), "1 flag with min_flags=2 must not produce a score"
    assert r.n_flags_present == 1, "flag count still reported for transparency"
    # with min_flags=1 the same input DOES score
    r1 = compute_composite_risk(one_flag, gap=np.nan, min_flags=1)
    assert not np.isnan(r1.research_weighted)


def test_composite_contributions_sum_to_research_weighted():
    flags = {"credit_gap": 70.0, "yield_curve": 50.0, "credit_spread": 30.0}
    result = compute_composite_risk(flags, gap=6.0)
    assert abs(sum(result.flag_contributions.values()) - result.research_weighted) < 0.2


def test_build_risk_series_end_to_end():
    import pandas as pd
    from .risk_flags import build_risk_series, FLAG_NAMES
    idx = pd.period_range("2000Q1", periods=60, freq="Q")
    rng = np.random.default_rng(0)
    scored = pd.DataFrame({
        "Household debt service ratio": np.linspace(10, 13, 60) + rng.normal(0, 0.1, 60),
        "ICE BofA US High Yield OAS (credit spread)": 5 + rng.normal(0, 0.5, 60),
        "10y-2y Treasury spread": np.linspace(2, -0.5, 60),
        "M2 velocity": np.linspace(2.0, 1.1, 60),
    }, index=idx)
    reference = pd.DataFrame({
        "10-year Treasury yield (level)": np.linspace(6, 4, 60),
        "10-year breakeven inflation expectations": np.linspace(1.5, 2.5, 60),
    }, index=idx)
    credit_gap = pd.Series(np.linspace(-4, 12, 60), index=idx)

    result = build_risk_series(scored, reference, credit_gap, min_periods=8)
    # risk is populated in [0,100] once enough history exists
    valid = result.risk.dropna()
    assert len(valid) > 0
    assert valid.between(0, 100).all()
    # equal-weighted also present
    assert result.risk_equal.dropna().between(0, 100).all()
    # flags frame has all flag columns
    assert list(result.flags.columns) == FLAG_NAMES
    # regime shifts to 'elevated' once the gap crosses 10pp near the end
    assert (result.regime == "elevated").any()
    # as the gap rises and curve inverts, late-period risk should exceed early
    assert result.risk.dropna().iloc[-1] > result.risk.dropna().iloc[0]


def _q(start, n):
    return pd.period_range(start, periods=n, freq="Q")


def test_publication_lag_retimes_quarterly_macro_inputs():
    from .risk_flags import build_risk_series
    idx = _q("1990Q1", 60)
    scored = pd.DataFrame({"10y-2y Treasury spread": np.full(60, 1.0)}, index=idx)
    gap = pd.Series(np.nan, index=idx); gap.iloc[40] = 12.0          # one gap reading
    r = build_risk_series(scored, pd.DataFrame(index=idx), gap, min_periods=8)
    # reference quarter 40 is only PUBLIC two quarters later
    assert np.isnan(r.flags["credit_gap"].iloc[40])
    assert r.flags["credit_gap"].iloc[42] == 100.0
    r0 = build_risk_series(scored, pd.DataFrame(index=idx), gap, min_periods=8, publication_lags={})
    assert r0.flags["credit_gap"].iloc[40] == 100.0                   # pre-audit alignment still available


def test_m2_velocity_flag_ignores_a_pure_downtrend():
    """A noisy structural decline must NOT read as permanent alarm (the pre-audit
    bug): on the change, the late readings should centre near 50, whereas a
    LEVEL percentile of the same series would sit near 100."""
    from .risk_flags import build_risk_series
    idx = _q("1980Q1", 160)
    rng = np.random.default_rng(7)
    vel = pd.Series(2.2 * np.exp(np.cumsum(-0.01 + rng.normal(0, 0.01, 160))), index=idx)
    scored = pd.DataFrame({"M2 velocity": vel, "10y-2y Treasury spread": 1.0}, index=idx)
    r = build_risk_series(scored, pd.DataFrame(index=idx), pd.Series(dtype=float),
                          min_periods=8, publication_lags={})
    late = r.flags["m2_velocity"].iloc[-60:]
    assert 30 <= late.mean() <= 70, f"downtrend biased the flag: mean {late.mean():.0f}"


def test_m2_velocity_flag_fires_on_a_sharp_fall():
    from .risk_flags import build_risk_series
    idx = _q("1980Q1", 120)
    v = np.full(120, 2.0); v[110:] = 1.6                              # sudden 20% drop
    scored = pd.DataFrame({"M2 velocity": v, "10y-2y Treasury spread": 1.0}, index=idx)
    r = build_risk_series(scored, pd.DataFrame(index=idx), pd.Series(dtype=float),
                          min_periods=8, publication_lags={})
    assert r.flags["m2_velocity"].iloc[111] >= 90


def test_long_bond_uses_trailing_window_not_full_history():
    """Yields rising for 20 years then flat for 15: an expanding percentile stays
    pinned high; the trailing window must come back down."""
    from .risk_flags import build_risk_series
    idx = _q("1962Q1", 140)
    y = np.r_[np.linspace(4, 15, 80), np.full(60, 9.0)]
    ref = pd.DataFrame({"10-year Treasury yield (level)": y}, index=idx)
    scored = pd.DataFrame({"10y-2y Treasury spread": np.full(140, 1.0)}, index=idx)
    r = build_risk_series(scored, ref, pd.Series(dtype=float), min_periods=8, publication_lags={})
    assert r.flags["long_bond_level"].iloc[-1] <= 50


def test_curve_prefers_long_history_10y_3m():
    from .risk_flags import build_risk_series
    idx = _q("1960Q1", 40)
    ref = pd.DataFrame({"10-year Treasury yield (monthly, long history)": np.full(40, 4.0),
                        "3-month Treasury bill (monthly, long history)": np.full(40, 5.0)}, index=idx)
    scored = pd.DataFrame({"10y-2y Treasury spread": np.full(40, 2.5)}, index=idx)
    r = build_risk_series(scored, ref, pd.Series(dtype=float), min_periods=8, publication_lags={})
    assert r.curve_source.startswith("10y-3m")
    assert r.flags["yield_curve"].iloc[-1] == 100.0                   # -1.0pp inversion, not the +2.5 10y-2y


def test_curve_fills_latest_quarter_from_daily_10y_3m_only():
    """Reviewer D2: the monthly pair posts late; the current quarter must fall
    back to daily 10y-3m -- and never to 10y-2y, a different spread."""
    from .risk_flags import build_risk_series
    idx = _q("2020Q1", 24)
    gs10 = np.full(24, 4.0); tb3 = np.full(24, 3.0); gs10[-1] = np.nan; tb3[-1] = np.nan
    ref = pd.DataFrame({"10-year Treasury yield (monthly, long history)": gs10,
                        "3-month Treasury bill (monthly, long history)": tb3}, index=idx)
    scored = pd.DataFrame({"10y-3m Treasury spread": np.full(24, -1.0),
                           "10y-2y Treasury spread": np.full(24, 3.0)}, index=idx)
    r = build_risk_series(scored, ref, pd.Series(dtype=float), min_periods=8, publication_lags={})
    assert abs(r.flags["yield_curve"].iloc[-2] - 36.0) < 1e-9             # monthly pair: +1.0pp
    assert r.flags["yield_curve"].iloc[-1] == 100.0                       # filled from daily 10y-3m
    no_daily = scored.drop(columns=["10y-3m Treasury spread"])
    r2 = build_risk_series(no_daily, ref, pd.Series(dtype=float), min_periods=8, publication_lags={})
    assert np.isnan(r2.flags["yield_curve"].iloc[-1])                     # NOT filled from 10y-2y


def test_latest_published_gap_carries_forward_so_regime_does_not_default():
    """Live-run bug: the newest quarter lost the (lagged) credit gap, so the regime
    fell back to its 'building' default. The last PUBLISHED gap must carry forward."""
    from .risk_flags import build_risk_series, CARRY_FORWARD_QUARTERS
    idx = _q("2000Q1", 60)
    scored = pd.DataFrame({"10y-2y Treasury spread": np.full(60, 1.0)}, index=idx)
    gap = pd.Series(-11.5, index=idx[:57])                  # BIS data ends 3 quarters early
    r = build_risk_series(scored, pd.DataFrame(index=idx), gap, min_periods=8)
    assert r.regime.iloc[-1] == "calm" and r.gap_used.iloc[-1] == -11.5
    # but never carried indefinitely
    gap_old = pd.Series(-11.5, index=idx[:40])
    r2 = build_risk_series(scored, pd.DataFrame(index=idx), gap_old, min_periods=8)
    assert np.isnan(r2.gap_used.iloc[40 + 2 + CARRY_FORWARD_QUARTERS])


def test_credit_spread_prefers_long_history_baa_over_short_hy_oas():
    from .risk_flags import build_risk_series
    idx = _q("1960Q1", 268)
    rng = np.random.default_rng(1)
    ref = pd.DataFrame({"Moody's Baa corporate yield (monthly)": 6 + rng.normal(0, 0.5, 268),
                        "10-year Treasury yield (monthly, long history)": np.full(268, 4.0),
                        "3-month Treasury bill (monthly, long history)": np.full(268, 3.0)}, index=idx)
    oas = pd.Series(np.nan, index=idx); oas.iloc[-12:] = 3.0      # FRED's ~3 years of ICE data
    scored = pd.DataFrame({"ICE BofA US High Yield OAS (credit spread)": oas}, index=idx)
    r = build_risk_series(scored, ref, pd.Series(dtype=float), min_periods=20, publication_lags={})
    assert r.spread_source.startswith("Baa")
    assert r.context_flags["credit_spread"].notna().sum() > 200


def test_headline_skips_a_new_quarter_missing_flags():
    """Live run: a 2-day-old quarter with 5 of 7 flags became the headline."""
    from .risk_flags import RiskSeriesResult, headline_quarter
    idx = _q("2024Q1", 12)
    n = pd.Series([7] * 11 + [5], index=idx)
    risk = pd.Series(30.0, index=idx)
    rr = RiskSeriesResult(risk=risk, risk_equal=risk, divergence=risk * 0, regime=risk.astype(str),
                          flags=pd.DataFrame(index=idx), n_flags=n)
    q, req = headline_quarter(rr, 4)
    assert str(q) == "2026Q3" and req == 7


def test_spec_v2_spread_is_context_only_and_v1_is_reproducible():
    from .risk_flags import (build_risk_series, composite_spec_v1, FLAG_NAMES,
                             SPEC_V1_WEIGHTS, RESEARCH_WEIGHTS)
    assert "credit_spread" not in FLAG_NAMES
    for reg in RESEARCH_WEIGHTS:                  # v2 = v1 renormalized, nothing re-chosen
        tot = 1 - SPEC_V1_WEIGHTS[reg]["credit_spread"]
        for k, w in RESEARCH_WEIGHTS[reg].items():
            assert abs(w - SPEC_V1_WEIGHTS[reg][k] / tot) < 1e-3
    idx = _q("1980Q1", 80)
    ref = pd.DataFrame({"Moody's Baa corporate yield (monthly)": np.linspace(6, 9, 80),
                        "10-year Treasury yield (monthly, long history)": np.full(80, 4.0),
                        "3-month Treasury bill (monthly, long history)": np.full(80, 3.0)}, index=idx)
    r = build_risk_series(pd.DataFrame(index=idx), ref, pd.Series(dtype=float), min_periods=8,
                          publication_lags={})
    assert "credit_spread" not in r.flags.columns
    v1 = composite_spec_v1(r)
    assert (v1.dropna() != r.risk.reindex(v1.dropna().index)).any()   # v1 really includes the spread


def test_percentile_gives_ties_half_credit():
    from .risk_flags import _expanding_percentile
    idx = _q("2000Q1", 30)
    flat = _expanding_percentile(pd.Series(5.0, index=idx), min_periods=5)
    assert abs(flat.dropna().iloc[-1] - 0.5) < 1e-12


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
