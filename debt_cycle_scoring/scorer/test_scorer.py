"""
Tests for debt_cycle_scoring.scorer -- the risk score and the end-to-end
engine. Run with:
    python3 -m debt_cycle_scoring.scorer.test_scorer
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .risk_score import compute_risk_score, STAGE_RISK_WEIGHTS, _gap_to_0_100
from .engine import fit_and_score, prepare_scored_panel, TRAINABLE_STAGES


def _qidx(start: str, n: int) -> pd.PeriodIndex:
    return pd.period_range(start, periods=n, freq="Q")


@dataclass
class FakeSpec:
    category: str
    higher_is_later_cycle: object  # bool or None


# ---- risk_score.py ---------------------------------------------------

def test_gap_to_0_100_clips_and_interpolates():
    assert _gap_to_0_100(-5.0) == 0.0
    assert _gap_to_0_100(15.0) == 100.0
    assert _gap_to_0_100(5.0) == 50.0          # midpoint of [-5, 15]
    assert _gap_to_0_100(-100.0) == 0.0        # clipped
    assert _gap_to_0_100(100.0) == 100.0       # clipped


def test_risk_score_pure_bubble_is_high():
    probs = {"bubble": 1.0}
    result = compute_risk_score(probs, credit_gap=12.0, stage_weight=0.6)
    # stage component == bubble weight; gap 12 -> high; blend should be high
    assert result.stage_component == STAGE_RISK_WEIGHTS["bubble"]
    assert result.risk_score > 75


def test_risk_score_pure_early_is_low():
    probs = {"early": 1.0}
    result = compute_risk_score(probs, credit_gap=-2.0, stage_weight=0.6)
    assert result.stage_component == STAGE_RISK_WEIGHTS["early"]
    assert result.risk_score < 30


def test_risk_score_falls_back_to_stage_only_when_gap_missing():
    probs = {"bubble": 0.5, "early": 0.5}
    result = compute_risk_score(probs, credit_gap=None, stage_weight=0.6)
    expected_stage = 0.5 * STAGE_RISK_WEIGHTS["bubble"] + 0.5 * STAGE_RISK_WEIGHTS["early"]
    assert abs(result.stage_component - expected_stage) < 1e-9
    assert result.risk_score == round(expected_stage, 1)
    assert result.stage_weight_used == 1.0
    assert np.isnan(result.gap_component)


def test_risk_score_contributions_sum_to_stage_component():
    probs = {"bubble": 0.3, "early": 0.4, "reflation": 0.3}
    result = compute_risk_score(probs, credit_gap=5.0)
    assert abs(sum(result.contributions.values()) - result.stage_component) < 1e-9


# ---- engine.py -------------------------------------------------------

def _make_panel(n=60):
    idx = _qidx("2000Q1", n)
    rng = np.random.default_rng(0)
    # two debt_burden metrics that rise over time, one cb_stance metric
    scored = pd.DataFrame(
        {
            "debtA": np.linspace(0, 10, n) + rng.normal(0, 0.2, n),
            "debtB": np.linspace(0, 8, n) + rng.normal(0, 0.2, n),
            "rate": np.concatenate([np.linspace(0, 5, n // 2), np.linspace(5, 1, n - n // 2)]),
        },
        index=idx,
    )
    specs = {
        "debtA": FakeSpec("debt_burden", True),
        "debtB": FakeSpec("debt_burden", True),
        "rate": FakeSpec("cb_stance", True),
    }
    return scored, specs, idx


def test_prepare_scored_panel_produces_category_scores():
    scored, specs, idx = _make_panel()
    category_scores, pca_ev = prepare_scored_panel(scored, specs)
    assert "debt_burden" in category_scores.columns
    assert "cb_stance" in category_scores.columns
    # PC1 explained-variance ratio should be reported per category
    assert "debt_burden" in pca_ev and 0 <= pca_ev["debt_burden"] <= 1


def test_prepare_scored_panel_skips_reference_metrics():
    scored, specs, idx = _make_panel()
    # add a reference-style metric with higher_is_later_cycle=None
    scored["gdp"] = np.linspace(100, 200, len(idx))
    specs["gdp"] = FakeSpec("reference", None)
    category_scores, _ = prepare_scored_panel(scored, specs)
    assert "reference" not in category_scores.columns, "reference metrics must not become a scored category"


def test_fit_and_score_end_to_end():
    scored, specs, idx = _make_panel(n=80)
    # build simple labels: first third early, middle bubble, last third reflation
    n = len(idx)
    labels = pd.Series("early", index=idx)
    labels.iloc[n // 3 : 2 * n // 3] = "bubble"
    labels.iloc[2 * n // 3 :] = "reflation"
    # training mask: mark a handful of each as trainable, all placed AFTER
    # the ~20-quarter percentile burn-in so none get dropped for lack of
    # history. "early" examples go at quarters 22-27 (past burn-in but
    # still in the early-labeled region).
    mask = pd.Series(False, index=idx)
    mask.iloc[22:28] = True                       # early (past burn-in)
    mask.iloc[n // 3 + 2 : n // 3 + 8] = True     # bubble
    mask.iloc[2 * n // 3 + 2 : 2 * n // 3 + 8] = True  # reflation
    credit_gap = pd.Series(np.linspace(-3, 12, n), index=idx)

    result = fit_and_score(scored, specs, labels, mask, credit_gap=credit_gap)

    # every scorable quarter gets a probability distribution summing to ~1
    row_sums = result.stage_probabilities.sum(axis=1)
    assert np.allclose(row_sums.values, 1.0, atol=1e-6)
    # risk score is populated and in range
    assert result.risk_score.between(0, 100).all()
    # archetypes were fit for the trained stages, including early
    assert set(result.archetype_n.keys()).issubset(set(TRAINABLE_STAGES))
    assert "early" in result.archetype_n
    assert not result.dropped_stages, f"no stage should have been dropped, got {result.dropped_stages}"


def test_fit_and_score_reports_dropped_stage_from_burn_in():
    # Deliberately place a stage's only training examples INSIDE the
    # percentile burn-in window so they produce NaN category scores and
    # drop out -- the engine should report this via dropped_stages rather
    # than silently omitting the stage. This is the real-data risk: some
    # "early" examples cluster in the earliest history and get eaten by
    # the burn-in.
    scored, specs, idx = _make_panel(n=80)
    n = len(idx)
    labels = pd.Series("early", index=idx)
    labels.iloc[n // 2 :] = "reflation"
    mask = pd.Series(False, index=idx)
    mask.iloc[2:8] = True                       # early -- INSIDE burn-in, will be dropped
    mask.iloc[n // 2 + 2 : n // 2 + 8] = True   # reflation -- fine
    result = fit_and_score(scored, specs, labels, mask)
    assert "early" in result.dropped_stages, "early (trained only inside burn-in) should be reported as dropped"
    assert "early" not in result.archetype_n


def test_fit_and_score_flags_low_confidence_stage():
    scored, specs, idx = _make_panel()
    n = len(idx)
    labels = pd.Series("early", index=idx)
    labels.iloc[n // 2 :] = "bubble"
    mask = pd.Series(False, index=idx)
    mask.iloc[2:12] = True                    # 10 early examples
    mask.iloc[n // 2 + 1 : n // 2 + 4] = True  # only 3 bubble examples -> low confidence
    result = fit_and_score(scored, specs, labels, mask)
    assert "bubble" in result.low_confidence_stages, "a 3-example stage should be flagged low-confidence"
    assert "early" not in result.low_confidence_stages


def test_fit_and_score_excludes_top_from_training():
    scored, specs, idx = _make_panel(n=80)
    n = len(idx)
    labels = pd.Series("early", index=idx)
    labels.iloc[n // 2 :] = "top"   # deliberately try to train a top
    mask = pd.Series(False, index=idx)
    mask.iloc[22:28] = True                    # early, past burn-in -> survives
    mask.iloc[n // 2 + 1 : n // 2 + 6] = True  # top -> must be excluded
    result = fit_and_score(scored, specs, labels, mask)
    assert "top" not in result.archetype_n, "top must never get an archetype even if labeled+masked"
    assert "early" in result.archetype_n, "the non-top trainable stage should still fit"


def test_ragged_history_late_category_does_not_veto_earlier_quarters():
    # Regression test for the real crash: one category (here "asset_prices")
    # only has data late in the sample, while others span the whole period.
    # Under the old all-categories-required rule this collapsed the scorable
    # window to near-nothing and starved training ("No training-eligible
    # quarters after alignment"). Variable-dimensionality scoring must
    # instead score the earlier quarters on whatever categories they DO
    # have.
    idx = _qidx("1990Q1", 100)
    rng = np.random.default_rng(5)
    scored = pd.DataFrame(
        {
            "debtA": np.linspace(0, 10, 100) + rng.normal(0, 0.2, 100),
            "debtB": np.linspace(0, 8, 100) + rng.normal(0, 0.2, 100),
            "rate": np.linspace(0, 5, 100) + rng.normal(0, 0.2, 100),
            "equity": np.linspace(0, 6, 100) + rng.normal(0, 0.2, 100),
        },
        index=idx,
    )
    # equity (asset_prices) only exists for the last 40 quarters -- enough
    # to clear the percentile-rank burn-in (min_periods+1) so it becomes a
    # real category, but still absent for the first 60.
    scored.loc[idx[:60], "equity"] = np.nan
    specs = {
        "debtA": FakeSpec("debt_burden", True),
        "debtB": FakeSpec("debt_burden", True),
        "rate": FakeSpec("cb_stance", True),
        "equity": FakeSpec("asset_prices", True),
    }
    labels = pd.Series("early", index=idx)
    labels.iloc[60:] = "reflation"
    mask = pd.Series(False, index=idx)
    mask.iloc[30:40] = True   # early, well before equity exists
    mask.iloc[60:70] = True   # reflation
    credit_gap = pd.Series(np.linspace(-3, 10, 100), index=idx)

    result = fit_and_score(scored, specs, labels, mask, credit_gap=credit_gap)

    # quarters BEFORE equity exists must still be scored (on the other 2 cats)
    early_period = idx[35]
    assert early_period in result.stage_probabilities.index, "pre-asset-price quarters must still be scored"
    assert result.categories_used.loc[early_period] >= 2
    # late quarters get MORE categories
    late_period = idx[95]
    assert result.categories_used.loc[late_period] > result.categories_used.loc[early_period], (
        "later quarters should be scored on more categories than earlier ones"
    )
    # probabilities still valid everywhere
    assert np.allclose(result.stage_probabilities.sum(axis=1).values, 1.0, atol=1e-6)


def test_quarter_with_too_few_categories_is_not_scored():
    # A quarter with fewer than MIN_CATEGORIES_TO_SCORE categories should
    # be left unscored rather than given a false-precision 1-D "distance".
    idx = _qidx("1990Q1", 60)
    rng = np.random.default_rng(6)
    scored = pd.DataFrame(
        {
            "debtA": np.linspace(0, 10, 60) + rng.normal(0, 0.2, 60),
            "rate": np.linspace(0, 5, 60) + rng.normal(0, 0.2, 60),
        },
        index=idx,
    )
    # leave only ONE category (debtA) present for the first 30 quarters
    scored.loc[idx[:30], "rate"] = np.nan
    specs = {"debtA": FakeSpec("debt_burden", True), "rate": FakeSpec("cb_stance", True)}
    labels = pd.Series("early", index=idx)
    labels.iloc[40:] = "reflation"
    mask = pd.Series(False, index=idx)
    mask.iloc[32:38] = True
    mask.iloc[45:52] = True
    result = fit_and_score(scored, specs, labels, mask)
    # a single-category quarter (first 30) must not be scored
    one_cat_period = idx[10]
    assert one_cat_period not in result.stage_probabilities.index, (
        "a quarter with only one category must not be scored"
    )


def test_thin_evidence_quarter_risk_pulled_toward_neutral():
    # Our explicit decision: a quarter scored on fewer categories is a
    # thinner read, so its risk score is pulled toward the neutral 50
    # baseline in proportion to how little evidence backed it. Here we
    # build a panel where early quarters have 2 categories and late
    # quarters have 3, and confirm evidence_confidence is reported and
    # < 1 for the thin quarters.
    idx = _qidx("2000Q1", 100)
    rng = np.random.default_rng(1)
    scored = pd.DataFrame(
        {
            "debtA": np.linspace(0, 10, 100) + rng.normal(0, 0.2, 100),
            "rate": np.linspace(0, 5, 100) + rng.normal(0, 0.2, 100),
            "equity": pd.Series(np.linspace(0, 9, 100), index=idx).where(idx >= pd.Period("2015Q1", freq="Q")),
        },
        index=idx,
    )
    specs = {
        "debtA": FakeSpec("debt_burden", True),
        "rate": FakeSpec("cb_stance", True),
        "equity": FakeSpec("asset_prices", True),
    }
    labels = pd.Series("early", index=idx)
    labels.iloc[60:] = "reflation"
    mask = pd.Series(False, index=idx)
    mask.iloc[30:40] = True
    mask.iloc[60:70] = True
    result = fit_and_score(scored, specs, labels, mask)

    early_period = idx[35]   # only debt_burden + cb_stance available
    late_period = idx[95]    # all three available
    assert result.evidence_confidence.loc[early_period] < result.evidence_confidence.loc[late_period]
    assert result.evidence_confidence.loc[late_period] <= 1.0
    assert 0 < result.evidence_confidence.loc[early_period] < 1.0


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
