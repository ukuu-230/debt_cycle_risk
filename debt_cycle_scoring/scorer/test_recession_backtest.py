"""
Tests for debt_cycle_scoring.scorer.recession_backtest. Run with:
    python3 -m debt_cycle_scoring.scorer.test_recession_backtest
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .recession_backtest import (
    run_recession_backtest, _recession_onset_quarters, _auc, DEFAULT_HORIZONS,
)


def _qidx(start, n):
    return pd.period_range(start, periods=n, freq="Q")


def test_auc_perfect_and_null():
    # positives all higher than negatives -> AUC 1.0
    assert _auc(np.array([80, 90, 70]), np.array([10, 20, 30])) == 1.0
    # positives all lower -> AUC 0.0
    assert _auc(np.array([10, 20]), np.array([80, 90])) == 0.0
    # interleaved -> ~0.5
    a = _auc(np.array([50, 50]), np.array([50, 50]))
    assert abs(a - 0.5) < 1e-9


def test_recession_onset_detection():
    idx = _qidx("2000Q1", 12)
    # recession quarters 4-6 and 9-10
    vals = [0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0]
    usrec = pd.Series(vals, index=idx)
    onsets = _recession_onset_quarters(usrec)
    assert onsets == [idx[4], idx[9]], f"got {onsets}"


def test_risk_that_leads_recession_scores_high_auc():
    # Build 80 quarters with two recessions, and risk that SPIKES in the
    # 4 quarters before each onset -> the 4Q-horizon AUC should be high.
    idx = _qidx("1990Q1", 80)
    usrec = pd.Series(0, index=idx)
    onsets_pos = [30, 60]
    for o in onsets_pos:
        usrec.iloc[o:o+4] = 1  # 4-quarter recessions
    risk = pd.Series(20.0, index=idx)  # calm baseline
    for o in onsets_pos:
        risk.iloc[o-4:o] = 85.0  # risk elevated in the 4 quarters before onset
    result = run_recession_backtest(risk, usrec, horizons=[4])
    h = result.horizons[0]
    assert h.auc > 0.9, f"risk that clearly leads should give high AUC, got {h.auc}"
    assert h.mean_risk_before_recession > h.mean_risk_otherwise
    assert result.n_recessions_in_sample == 2


def test_random_risk_scores_near_half_auc():
    idx = _qidx("1990Q1", 120)
    usrec = pd.Series(0, index=idx)
    for o in [40, 80]:
        usrec.iloc[o:o+3] = 1
    rng = np.random.default_rng(0)
    risk = pd.Series(rng.uniform(0, 100, 120), index=idx)  # random, no relationship
    result = run_recession_backtest(risk, usrec, horizons=[4])
    h = result.horizons[0]
    # random risk should be near 0.5 (wide tolerance given small samples)
    assert 0.25 < h.auc < 0.75, f"random risk AUC should be near 0.5, got {h.auc}"


def test_in_recession_quarters_excluded_from_cases():
    idx = _qidx("2000Q1", 20)
    usrec = pd.Series(0, index=idx)
    usrec.iloc[10:14] = 1
    risk = pd.Series(50.0, index=idx)
    result = run_recession_backtest(risk, usrec, horizons=[1])
    # 4 quarters are in-recession and must be excluded; the latest quarters are
    # right-censored: the 1Q window must end on or before the last USREC quarter
    # minus the NBER dating lag
    from .recession_backtest import NBER_DATING_LAG
    assert result.horizons[0].n_cases == 20 - 4 - (1 + NBER_DATING_LAG)


def test_multiple_horizons_all_reported():
    idx = _qidx("1990Q1", 80)
    usrec = pd.Series(0, index=idx)
    usrec.iloc[40:44] = 1
    risk = pd.Series(np.linspace(10, 60, 80), index=idx)
    result = run_recession_backtest(risk, usrec)  # default horizons
    assert [h.horizon for h in result.horizons] == DEFAULT_HORIZONS


def test_lead_profile_populated():
    idx = _qidx("1990Q1", 60)
    usrec = pd.Series(0, index=idx)
    usrec.iloc[40:43] = 1
    risk = pd.Series(np.arange(60, dtype=float), index=idx)
    result = run_recession_backtest(risk, usrec, horizons=[4])
    # risk_at_each_onset should have the onset row and -0Q..-8Q columns
    assert result.risk_at_each_onset.shape[0] == 1
    row = result.risk_at_each_onset.iloc[0]
    # -0Q is the onset quarter's risk (index 40 -> value 40.0)
    assert row["-0Q"] == 40.0
    assert row["-4Q"] == 36.0


def test_accepts_already_quarterly_usrec():
    # Regression: the live panel hands usrec in as an ALREADY-quarterly
    # PeriodIndex series (not monthly). run_recession_backtest must accept
    # that directly -- an earlier run_scorer.py block wrongly called
    # .to_period on it and crashed with AttributeError.
    idx = _qidx("1990Q1", 80)
    usrec = pd.Series(0, index=idx)   # already quarterly PeriodIndex
    usrec.iloc[40:44] = 1
    risk = pd.Series(np.linspace(10, 60, 80), index=idx)
    result = run_recession_backtest(risk, usrec)  # must not raise
    assert result.n_recessions_in_sample == 1
    assert len(result.horizons) == 3


def test_recent_quarters_are_right_censored_not_negatives():
    # 40 quarters, no recession at all, observed through the last quarter.
    # At H=8 the last 8 quarters' outcome windows run past the data, so they
    # must be EXCLUDED -- not counted as "no recession followed".
    idx = _qidx("2000Q1", 40)
    usrec = pd.Series(0, index=idx); usrec.iloc[20:22] = 1
    risk = pd.Series(np.linspace(0, 100, 40), index=idx)
    r = run_recession_backtest(risk, usrec, horizons=[8])
    from .recession_backtest import NBER_DATING_LAG
    in_rec = 2
    assert r.horizons[0].n_cases == 40 - in_rec - (8 + NBER_DATING_LAG)


def test_known_positive_kept_even_if_window_runs_past_data():
    """Reviewer D1: a window that already contains an onset is a KNOWN positive."""
    from .recession_backtest import _split_cases, _recession_onset_quarters
    idx = _qidx("2000Q1", 12)
    usrec = pd.Series(0, index=idx); usrec.iloc[11] = 1          # onset in the final quarter
    score = pd.Series(np.arange(12.0), index=idx)
    pos, neg = _split_cases(score, usrec, _recession_onset_quarters(usrec), 4, last_observed=idx[11])
    assert sorted(pos) == [7.0, 8.0, 9.0, 10.0]                   # all 4 quarters before the onset


def test_sample_starting_mid_recession_has_no_false_onset():
    from .recession_backtest import _recession_onset_quarters
    idx = _qidx("2008Q1", 12)
    usrec = pd.Series(0, index=idx); usrec.iloc[0:6] = 1
    assert _recession_onset_quarters(usrec) == []


def test_flag_attribution_identifies_the_carrying_flag():
    from .risk_flags import RiskSeriesResult, FLAG_NAMES
    from .recession_backtest import flag_attribution
    idx = _qidx("1980Q1", 120)
    usrec = pd.Series(0, index=idx)
    rng = np.random.default_rng(3)
    flags = pd.DataFrame(rng.uniform(20, 40, (120, len(FLAG_NAMES))), index=idx, columns=FLAG_NAMES)
    for start in (20, 50, 80):                       # three recessions
        usrec.iloc[start:start + 3] = 1
        flags.iloc[start - 6:start, FLAG_NAMES.index("yield_curve")] = 95   # curve leads them
    from .risk_flags import compute_composite_risk
    risk = pd.Series({q: compute_composite_risk(flags.loc[q].to_dict(), 0.0).research_weighted for q in idx})
    rr = RiskSeriesResult(risk=risk, risk_equal=risk, divergence=risk * 0, regime=risk.astype(str),
                          flags=flags, n_flags=pd.Series(7, index=idx),
                          gap_used=pd.Series(0.0, index=idx))
    att = flag_attribution(rr, usrec, horizon=8)
    assert att["delta"].idxmax() == "yield_curve"
    assert att.loc["yield_curve", "standalone_auc"] > 0.7


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
