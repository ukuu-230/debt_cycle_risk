"""
Tests for debt_cycle_scoring.labeling -- section 3B's independent
labeling pass. Run with:
    python3 -m debt_cycle_scoring.labeling.test_labeling
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .nber import extract_recession_turning_points
from .hp_filter import hp_filter_two_sided, one_sided_hp_trend
from .credit_gap import compute_credit_gap, BASEL_LOWER_THRESHOLD, BASEL_UPPER_THRESHOLD
from .turning_points import bry_boschan_turning_points
from .consensus import build_consensus_labels, stage_labels_from_consensus


def _qidx(start: str, n: int) -> pd.PeriodIndex:
    return pd.period_range(start, periods=n, freq="Q")


# ---- nber.py ---------------------------------------------------------

def test_extract_recession_turning_points_basic():
    idx = _qidx("2005Q1", 20)
    # recession from index 10 to 12 inclusive (3 quarters)
    values = [0] * 10 + [1, 1, 1] + [0] * 7
    usrec = pd.Series(values, index=idx)
    events = extract_recession_turning_points(usrec)
    assert events.loc[idx[9], "event"] == "peak", "peak should be the quarter BEFORE recession starts"
    assert events.loc[idx[12], "event"] == "trough", "trough should be the LAST recession quarter"
    assert len(events) == 2


def test_extract_recession_turning_points_multiple_recessions():
    idx = _qidx("2000Q1", 30)
    values = [0] * 5 + [1, 1] + [0] * 10 + [1, 1, 1] + [0] * 10
    usrec = pd.Series(values, index=idx)
    events = extract_recession_turning_points(usrec)
    assert (events["event"] == "peak").sum() == 2
    assert (events["event"] == "trough").sum() == 2


# ---- hp_filter.py ------------------------------------------------------

def test_hp_filter_two_sided_constant_series_stays_constant():
    s = pd.Series([5.0] * 30)
    trend = hp_filter_two_sided(s, lamb=1600)
    assert np.allclose(trend.values, 5.0, atol=1e-6)


def test_one_sided_hp_trend_has_no_lookahead():
    # Build two series that are IDENTICAL up through quarter 39, then
    # diverge sharply afterward. The one-sided trend at quarter 39 must
    # be identical in both cases -- if it isn't, the filter is using
    # future information.
    rng = np.random.default_rng(1)
    shared = np.cumsum(rng.normal(0, 1, 40))
    tail_a = shared[-1] + np.cumsum(rng.normal(0, 1, 20))
    tail_b = shared[-1] + np.cumsum(rng.normal(20, 1, 20))  # big divergent shock

    series_a = pd.Series(np.concatenate([shared, tail_a]))
    series_b = pd.Series(np.concatenate([shared, tail_b]))

    trend_a = one_sided_hp_trend(series_a, lamb=400_000, min_periods=20)
    trend_b = one_sided_hp_trend(series_b, lamb=400_000, min_periods=20)

    assert np.allclose(
        trend_a.iloc[:40].values, trend_b.iloc[:40].values, atol=1e-8, equal_nan=True
    ), ("one-sided trend for quarters 0-39 must not depend on data from quarter 40 onward")


def test_one_sided_hp_trend_nan_before_min_periods():
    s = pd.Series(np.arange(50, dtype=float))
    trend = one_sided_hp_trend(s, min_periods=20)
    assert trend.iloc[:19].isna().all()
    assert trend.iloc[19:].notna().all()


# ---- credit_gap.py -------------------------------------------------------

def test_compute_credit_gap_flags_thresholds_correctly():
    idx = _qidx("1990Q1", 75)
    # flat ratio for a long burn-in, then a sharp, sustained run-up that
    # should breach both Basel III thresholds well before the series ends
    ratio = np.concatenate([np.full(40, 100.0), 100 + np.arange(35) * 1.5])
    s = pd.Series(ratio, index=idx)
    result = compute_credit_gap(s, min_periods=30)

    late_gap = result["gap"].iloc[-1]
    assert late_gap > BASEL_UPPER_THRESHOLD, "a sustained sharp run-up should breach the upper threshold eventually"
    assert result["flag"].iloc[-1] == "top_warning"
    assert result["flag"].iloc[35] == "early", "flat early period should read as 'early'"


def test_declining_from_peak_requires_a_real_prior_peak():
    idx = _qidx("1990Q1", 60)
    # flat and low throughout -- noise wobbling near zero should NOT
    # register as "declining from peak", since there was never a real
    # peak (bubble-building territory) to decline from
    rng = np.random.default_rng(2)
    ratio = 100 + np.cumsum(rng.normal(0, 0.1, 60))
    s = pd.Series(ratio, index=idx)
    result = compute_credit_gap(s, min_periods=30, decline_lookback=12)
    assert not result["declining_from_peak"].any(), (
        "small noise fluctuations with no real bubble-building peak should never flag as 'declining from peak'"
    )


def test_declining_from_peak_fires_after_a_real_runup_and_retreat():
    idx = _qidx("1990Q1", 80)
    # flat burn-in, sharp run-up into bubble territory, then a sharp fall
    ratio = np.concatenate(
        [np.full(40, 100.0), 100 + np.arange(20) * 1.5, 130 - np.arange(20) * 2.0]
    )
    s = pd.Series(ratio, index=idx)
    result = compute_credit_gap(s, min_periods=30, decline_lookback=12, decline_threshold=3.0)
    # somewhere in the retreat phase (last 20 quarters), declining_from_peak should fire
    assert result["declining_from_peak"].iloc[-10:].any(), (
        "a sharp retreat from real bubble-building territory should eventually flag as 'declining from peak'"
    )


# ---- turning_points.py ----------------------------------------------------

def test_bry_boschan_finds_obvious_peak_and_trough():
    idx = _qidx("2000Q1", 20)
    # up, peak at index 9, down, trough at index 15, up again
    values = list(range(10)) + list(range(9, 3, -1)) + list(range(4, 8))
    s = pd.Series(values[:20], index=idx)
    result = bry_boschan_turning_points(s, window=2, min_phase_quarters=2)
    assert "peak" in result["event"].values
    assert "trough" in result["event"].values
    # events must alternate
    events = result["event"].tolist()
    for a, b in zip(events, events[1:]):
        assert a != b, "consecutive turning points must alternate between peak and trough"


def test_bry_boschan_enforces_min_phase_length():
    idx = _qidx("2000Q1", 10)
    # a single-quarter wiggle: should NOT register as a valid peak+trough
    # pair given a min_phase_quarters requirement larger than 1
    values = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2]
    s = pd.Series(values, index=idx)
    result = bry_boschan_turning_points(s, window=1, min_phase_quarters=3)
    # with a strict min phase length, most single-quarter wiggles should be pruned
    assert len(result) < len(values)


def test_bry_boschan_pruning_never_breaks_alternation():
    # Regression test: an earlier version of the pruning step could drop
    # an interior turning point and leave two same-type points adjacent
    # to each other (e.g. pruning a peak between two troughs left
    # trough-trough back to back) -- this was caught by the integration
    # demo (labeling/demo.py), not by the simpler tests above, since it
    # only shows up with a longer, noisier series and a phase-length
    # requirement that forces at least one interior point to be dropped.
    idx = _qidx("1995Q1", 100)
    rng = np.random.default_rng(4)
    values = pd.Series(np.cumsum(rng.normal(0, 1, 100)), index=idx)
    result = bry_boschan_turning_points(values, window=2, min_phase_quarters=4)
    events = result["event"].tolist()
    for a, b in zip(events, events[1:]):
        assert a != b, f"found two consecutive '{a}' events -- alternation was broken by pruning"


def test_bry_boschan_default_smooth_window_matches_unsmoothed_behavior():
    # smooth_window=1 (the default) must behave EXACTLY like the original
    # unsmoothed function -- this is a backward-compatibility guard for
    # every other test/caller that doesn't pass smooth_window explicitly.
    idx = _qidx("2000Q1", 20)
    values = list(range(10)) + list(range(9, 3, -1)) + list(range(4, 8))
    s = pd.Series(values[:20], index=idx)
    default_result = bry_boschan_turning_points(s, window=2, min_phase_quarters=2)
    explicit_result = bry_boschan_turning_points(s, window=2, min_phase_quarters=2, smooth_window=1)
    pd.testing.assert_frame_equal(default_result, explicit_result)


def test_bry_boschan_smoothing_removes_a_spurious_short_lived_notch():
    # Regression test for a real finding: running this against real BIS
    # credit-to-GDP data produced a spurious mechanical "peak" right at
    # the genuine 2009Q2 NBER trough, caused by short-term noise the
    # unsmoothed detector is too sensitive to. This builds a small,
    # controlled version of that pattern -- a brief one-quarter notch
    # sitting on top of an otherwise smooth trend -- and confirms
    # smoothing removes the spurious extra turning point the notch alone
    # would otherwise create.
    idx = _qidx("2000Q1", 30)
    trend = np.linspace(100, 130, 30)
    notch = np.zeros(30)
    notch[15] = -4.0  # one-quarter dip, the kind of noise causing false extrema
    values = trend + notch
    s = pd.Series(values, index=idx)

    unsmoothed = bry_boschan_turning_points(s, window=2, min_phase_quarters=2, smooth_window=1)
    smoothed = bry_boschan_turning_points(s, window=2, min_phase_quarters=2, smooth_window=5)

    assert len(smoothed) < len(unsmoothed), (
        "smoothing should remove at least the spurious turning point pair "
        "the single-quarter notch creates in the unsmoothed version"
    )


# ---- consensus.py -----------------------------------------------------

def test_consensus_high_confidence_when_all_sources_agree():
    idx = _qidx("2005Q1", 8)
    nber_events = pd.DataFrame({"event": ["peak"]}, index=[idx[4]])
    credit_gap_df = pd.DataFrame({"flag": ["top_warning"] * 8}, index=idx)
    mechanical_events = pd.DataFrame({"event": ["peak"]}, index=[idx[4]])

    consensus = build_consensus_labels(idx, nber_events, credit_gap_df, mechanical_events, agreement_window=2)
    assert consensus.loc[idx[4], "tier"] == "high_confidence"
    assert consensus.loc[idx[4], "label"] == "top"


def test_consensus_contested_when_recession_without_credit_buildup():
    # mimics the 2020-COVID-style negative case from section 3B.4: NBER
    # flags a recession, but the credit gap never built up beforehand.
    idx = _qidx("2019Q1", 6)
    nber_events = pd.DataFrame({"event": ["peak"]}, index=[idx[3]])
    credit_gap_df = pd.DataFrame({"flag": ["early"] * 6}, index=idx)
    mechanical_events = pd.DataFrame(columns=["event"])

    consensus = build_consensus_labels(idx, nber_events, credit_gap_df, mechanical_events, agreement_window=1)
    assert consensus.loc[idx[3], "tier"] == "contested"
    assert consensus.loc[idx[3], "label"] == "non_debt_recession"


def test_consensus_silent_when_nothing_flags():
    idx = _qidx("2000Q1", 5)
    empty = pd.DataFrame(columns=["event"])
    credit_gap_df = pd.DataFrame({"flag": ["early"] * 5}, index=idx)
    consensus = build_consensus_labels(idx, empty, credit_gap_df, empty)
    assert (consensus["tier"] == "silent").all()


def test_contested_overrides_coincidental_mechanical_agreement():
    # Regression test for a real bug found running this against live FRED
    # data: an NBER recession with a credit gap explicitly reading "early"
    # (no debt buildup at all) must be CONTESTED even if the mechanical
    # detector happens to also fire nearby by coincidence -- NBER + a
    # coincidental mechanical turning point is NOT enough to overrule an
    # explicit "no credit buildup" reading from the credit gap. An earlier
    # version got this backwards (checked top_votes >= 2 before checking
    # for the contested pattern), which misclassified the real 2020Q1
    # COVID recession as a high-confidence debt-cycle "top".
    idx = _qidx("2019Q1", 6)
    nber_events = pd.DataFrame({"event": ["peak"]}, index=[idx[3]])
    credit_gap_df = pd.DataFrame(
        {"flag": ["early"] * 6, "declining_from_peak": [False] * 6}, index=idx
    )
    # mechanical detector ALSO fires here by coincidence -- this is the
    # part that broke the old ordering
    mechanical_events = pd.DataFrame({"event": ["peak"]}, index=[idx[3]])

    consensus = build_consensus_labels(idx, nber_events, credit_gap_df, mechanical_events, agreement_window=2)
    assert consensus.loc[idx[3], "tier"] == "contested", (
        "an explicit 'early' credit gap must override NBER+mechanical agreement, not the other way around"
    )
    assert consensus.loc[idx[3], "label"] == "non_debt_recession"


def test_trough_gets_credit_gap_vote_not_just_nber_and_mechanical():
    # A trough should be reachable via NBER + a declining credit gap alone,
    # with NO mechanical corroboration needed -- mirroring how a top can
    # be reached via NBER + gap_building alone. Before this fix, troughs
    # had no credit-side vote at all.
    idx = _qidx("2008Q1", 6)
    nber_events = pd.DataFrame({"event": ["trough"]}, index=[idx[3]])
    credit_gap_df = pd.DataFrame(
        {"flag": ["bubble_building"] * 6, "declining_from_peak": [True] * 6}, index=idx
    )
    mechanical_events = pd.DataFrame(columns=["event"])  # no mechanical support at all

    consensus = build_consensus_labels(idx, nber_events, credit_gap_df, mechanical_events, agreement_window=2)
    assert consensus.loc[idx[3], "tier"] == "high_confidence"
    assert consensus.loc[idx[3], "label"] == "trough"


def test_conflicting_tier_when_top_and_trough_evidence_both_fire():
    # If top evidence AND trough evidence both independently reach
    # high-confidence levels in the same quarter, that's a genuine
    # conflict between sources and should be surfaced as its own tier,
    # not silently resolved by whichever check happens to run first
    # (that was the exact SHAPE of the contested-ordering bug -- this
    # test guards against the same mistake recurring for top/trough).
    idx = _qidx("2009Q1", 4)
    nber_events = pd.DataFrame({"event": ["peak"]}, index=[idx[1]])
    credit_gap_df = pd.DataFrame(
        {"flag": ["bubble_building"] * 4, "declining_from_peak": [True] * 4}, index=idx
    )
    mechanical_events = pd.DataFrame({"event": ["trough"]}, index=[idx[1]])

    consensus = build_consensus_labels(idx, nber_events, credit_gap_df, mechanical_events, agreement_window=1)
    row = consensus.loc[idx[1]]
    # nber_top=True, gap_building=True -> top_votes=2 (nber_top, gap_building)
    # nber_trough=False, gap_declining=True, mech_trough=True -> trough_votes=2
    assert row["tier"] == "conflicting"
    assert row["label"] is None


def test_stage_labels_from_consensus_full_cycle():
    idx = _qidx("2000Q1", 20)
    # Build a consensus table by hand: high-confidence trough at index 0,
    # high-confidence top at index 12.
    rows = []
    for i, period in enumerate(idx):
        if i == 0:
            tier, label, gap_flag = "high_confidence", "trough", "early"
        elif i == 12:
            tier, label, gap_flag = "high_confidence", "top", "top_warning"
        elif 6 <= i < 12:
            tier, label, gap_flag = "silent", None, "bubble_building"
        else:
            tier, label, gap_flag = "silent", None, "early"
        rows.append({"period": period, "tier": tier, "label": label, "credit_gap_flag": gap_flag})
    consensus = pd.DataFrame(rows).set_index("period")

    # Real GDP growth: positive throughout the post-top window -> should
    # classify the deleveraging as "beautiful".
    real_gdp_growth = pd.Series(0.5, index=idx)

    labels = stage_labels_from_consensus(consensus, real_gdp_qoq_growth=real_gdp_growth, reflation_window=4)

    assert labels.iloc[0:4].eq("reflation").all(), "first reflation_window quarters after a trough should be 'reflation'"
    assert labels.iloc[6:12].eq("bubble").all(), "quarters with a building credit gap before the top should be 'bubble'"
    assert labels.iloc[12:].eq("deleveraging_beautiful").all(), (
        "quarters after a confirmed top with non-negative real GDP growth should be 'deleveraging_beautiful'"
    )


def test_stage_labels_ugly_deleveraging_with_negative_growth():
    idx = _qidx("2008Q1", 6)
    rows = [{"period": idx[0], "tier": "high_confidence", "label": "top", "credit_gap_flag": "top_warning"}]
    rows += [{"period": p, "tier": "silent", "label": None, "credit_gap_flag": "early"} for p in idx[1:]]
    consensus = pd.DataFrame(rows).set_index("period")

    real_gdp_growth = pd.Series(-1.0, index=idx)  # sustained contraction
    labels = stage_labels_from_consensus(consensus, real_gdp_qoq_growth=real_gdp_growth)
    assert (labels == "deleveraging_ugly").all()


def test_bubble_and_early_become_trainable_with_independent_gap_agreement():
    # The strict rule: a "bubble" quarter is trainable only when the
    # credit gap independently reads top_warning, and an "early" quarter
    # only when the gap independently reads early. This is what gives the
    # scorer training examples of early/bubble, which the confirmed-
    # turning-point-only rule structurally never produced.
    idx = _qidx("2000Q1", 16)
    rows = []
    for i, p in enumerate(idx):
        if i == 0:
            rows.append({"period": p, "tier": "high_confidence", "label": "trough", "credit_gap_flag": "early"})
        elif i == 10:
            rows.append({"period": p, "tier": "high_confidence", "label": "top", "credit_gap_flag": "top_warning"})
        else:
            rows.append({"period": p, "tier": "silent", "label": None, "credit_gap_flag": "early"})
    consensus = pd.DataFrame(rows).set_index("period")
    # make the pre-top run-up read as a genuine bubble via the gap flag
    for i in range(6, 10):
        consensus.iloc[i, consensus.columns.get_loc("credit_gap_flag")] = "top_warning"

    real_gdp_growth = pd.Series(0.5, index=idx)
    labels, train_ok = stage_labels_from_consensus(
        consensus, real_gdp_qoq_growth=real_gdp_growth, return_training_mask=True
    )

    # a quarter labeled bubble WITH top_warning gap agreement -> trainable
    bubble_quarters = [p for p in idx if labels[p] == "bubble" and consensus.loc[p, "credit_gap_flag"] == "top_warning"]
    assert bubble_quarters, "test setup should produce at least one gap-agreeing bubble quarter"
    assert all(train_ok[p] for p in bubble_quarters), "bubble quarters with top_warning agreement should be trainable"

    # an early quarter with early gap agreement -> trainable
    early_quarters = [p for p in idx if labels[p] == "early" and consensus.loc[p, "credit_gap_flag"] == "early"]
    assert early_quarters, "test setup should produce at least one gap-agreeing early quarter"
    assert all(train_ok[p] for p in early_quarters), "early quarters with early gap agreement should be trainable"


def test_bubble_without_gap_agreement_is_not_trainable():
    # Strictness check: a quarter that INHERITS the "bubble" label from
    # the walk but whose credit gap only reads "bubble_building" (not the
    # stronger top_warning) must NOT be trainable -- that's the whole
    # point of the strict variant.
    idx = _qidx("2000Q1", 12)
    rows = []
    for i, p in enumerate(idx):
        if i == 0:
            rows.append({"period": p, "tier": "high_confidence", "label": "trough", "credit_gap_flag": "early"})
        else:
            # after the reflation window, an inherited "bubble" label with
            # only bubble_building (weaker) gap support
            rows.append({"period": p, "tier": "silent", "label": None, "credit_gap_flag": "bubble_building"})
    consensus = pd.DataFrame(rows).set_index("period")
    real_gdp_growth = pd.Series(0.5, index=idx)
    labels, train_ok = stage_labels_from_consensus(
        consensus, real_gdp_qoq_growth=real_gdp_growth, return_training_mask=True
    )
    inherited_bubble = [p for p in idx if labels[p] == "bubble"]
    assert inherited_bubble, "test setup should produce inherited bubble quarters"
    assert not any(train_ok[p] for p in inherited_bubble), (
        "bubble quarters with only bubble_building (not top_warning) gap support must NOT be trainable under the strict rule"
    )


def test_top_label_is_never_trainable():
    # "top" is deliberately excluded from training entirely. Even if a
    # quarter somehow ends up labeled "top", the guard must keep it out.
    idx = _qidx("2000Q1", 4)
    rows = [
        {"period": idx[0], "tier": "high_confidence", "label": "top", "credit_gap_flag": "top_warning"},
        {"period": idx[1], "tier": "silent", "label": None, "credit_gap_flag": "top_warning"},
        {"period": idx[2], "tier": "silent", "label": None, "credit_gap_flag": "early"},
        {"period": idx[3], "tier": "silent", "label": None, "credit_gap_flag": "early"},
    ]
    consensus = pd.DataFrame(rows).set_index("period")
    real_gdp_growth = pd.Series(0.5, index=idx)
    labels, train_ok = stage_labels_from_consensus(
        consensus, real_gdp_qoq_growth=real_gdp_growth, return_training_mask=True
    )
    top_quarters = [p for p in idx if labels[p] == "top"]
    assert not any(train_ok[p] for p in top_quarters), "no 'top'-labeled quarter may ever be trainable"


def test_conflicting_quarter_excluded_from_training_mask_but_keeps_display_label():
    # Regression test for a real, subtle bug found looking closely at how
    # conflicting quarters flow downstream: a conflicting quarter (sources
    # disagree) still INHERITS a descriptive stage label from the last
    # confirmed event, which is fine for a display timeline -- but it must
    # NOT be handed to the scorer as ground-truth training data. Before
    # return_training_mask existed, 2009Q2-style conflicting quarters
    # silently trained the model as confirmed "deleveraging_beautiful".
    idx = _qidx("2007Q1", 12)
    rows = []
    for i, p in enumerate(idx):
        if i == 3:
            rows.append({"period": p, "tier": "high_confidence", "label": "top", "credit_gap_flag": "top_warning"})
        elif i == 9:
            # a conflicting quarter well after the top -- will inherit
            # "deleveraging_beautiful" as its descriptive label
            rows.append({"period": p, "tier": "conflicting", "label": None, "credit_gap_flag": "bubble_building"})
        else:
            rows.append({"period": p, "tier": "silent", "label": None, "credit_gap_flag": "early"})
    consensus = pd.DataFrame(rows).set_index("period")
    real_gdp_growth = pd.Series(0.5, index=idx)

    labels, train_ok = stage_labels_from_consensus(
        consensus, real_gdp_qoq_growth=real_gdp_growth, return_training_mask=True
    )

    conflicting_period = idx[9]
    # it DOES get a descriptive label (for the timeline)
    assert labels[conflicting_period] == "deleveraging_beautiful"
    # but it is NOT training-eligible
    assert not train_ok[conflicting_period], "a conflicting quarter must be excluded from training anchors"
    # the actual high-confidence top IS training-eligible
    assert train_ok[idx[3]], "a high-confidence quarter should be training-eligible"
    # a plain silent quarter that merely inherited a label is also excluded
    assert not train_ok[idx[7]], "a silent quarter that only inherited a label must not be training-eligible"


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
