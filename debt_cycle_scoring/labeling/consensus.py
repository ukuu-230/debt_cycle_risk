"""
Consensus combiner -- design doc section 3B.4.

Combines the three independent label sources (NBER turning points, the
BIS-style credit-to-GDP gap, and Bry-Boschan mechanical turning points)
into confidence-tiered labels, then (in `stage_labels_from_consensus`)
walks those high-confidence events chronologically to assign a full
STAGES-vocabulary label (matching `stages.STAGES` in the parent package)
to each quarter -- this becomes the ground truth `fit_archetypes()`
should train on, replacing the hand-built synthetic labels used in the
earlier demos.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd


def _quarters_between(a: pd.Period, b: pd.Period) -> int:
    return abs(a.ordinal - b.ordinal)


def build_consensus_labels(
    quarterly_index: pd.PeriodIndex,
    nber_events: pd.DataFrame,
    credit_gap_df: pd.DataFrame,
    mechanical_events: pd.DataFrame,
    agreement_window: int = 4,
) -> pd.DataFrame:
    """
    Per-quarter multi-source agreement table, directly implementing the
    section 3B.4 confidence tiers:
      - "high_confidence": >=2 of {NBER, credit-gap, mechanical} agree
        there's a top within `agreement_window` quarters of this one.
      - "contested": NBER flags a recession here but the credit gap never
        built up beforehand -- the "non-debt-cycle recession" case (1970s
        oil shocks, 2020 COVID) that a working model should NOT treat as
        debt-cycle evidence.
      - "single_source": only one source flags a turning point nearby.
      - "silent": no source flags anything nearby.

    nber_events / mechanical_events: output of extract_recession_turning_points()
        / bry_boschan_turning_points() -- indexed by period, column "event"
        in {"peak", "trough"}.
    credit_gap_df: output of compute_credit_gap() -- indexed by period,
        column "flag" in {"early", "bubble_building", "top_warning", None}.
    """
    nber_peaks = nber_events.index[nber_events["event"] == "peak"] if not nber_events.empty else pd.Index([])
    nber_troughs = nber_events.index[nber_events["event"] == "trough"] if not nber_events.empty else pd.Index([])
    mech_peaks = (
        mechanical_events.index[mechanical_events["event"] == "peak"]
        if not mechanical_events.empty else pd.Index([])
    )
    mech_troughs = (
        mechanical_events.index[mechanical_events["event"] == "trough"]
        if not mechanical_events.empty else pd.Index([])
    )

    def near(target_index: pd.Index, period: pd.Period) -> bool:
        return any(_quarters_between(period, t) <= agreement_window for t in target_index)

    rows = []
    for period in quarterly_index:
        nber_top = near(nber_peaks, period)
        nber_trough = near(nber_troughs, period)
        mech_top = near(mech_peaks, period)
        mech_trough = near(mech_troughs, period)
        gap_flag = credit_gap_df["flag"].get(period) if period in credit_gap_df.index else None
        gap_building = gap_flag in ("bubble_building", "top_warning")
        gap_declining = (
            bool(credit_gap_df["declining_from_peak"].get(period, False))
            if "declining_from_peak" in credit_gap_df.columns and period in credit_gap_df.index
            else False
        )

        top_votes = sum([nber_top, gap_building, mech_top])
        trough_votes = sum([nber_trough, gap_declining, mech_trough])

        # Priority, in order:
        #   1. Contested pattern overrides everything: an NBER recession
        #      with an explicitly non-building credit gap is the
        #      "non-debt-cycle recession" case (1970s oil shocks, 2020
        #      COVID) regardless of what the other sources say. Checking
        #      this FIRST (not after the vote-count checks) was itself a
        #      bug fix -- confirmed against live FRED data, where 2020Q1
        #      was misclassified as a high-confidence "top" before this
        #      check was moved to the front.
        #   2. If top AND trough evidence both independently reach
        #      high-confidence levels in the SAME quarter, that's a real
        #      conflict between sources, not something to silently
        #      resolve by whichever branch happens to be checked first
        #      (that was the exact shape of bug #1 -- not repeating it
        #      here for the top/trough case). Surface it as its own tier
        #      rather than picking a side.
        #   3. Otherwise, ordinary vote counting.
        if nber_top and not gap_building:
            tier, label = "contested", "non_debt_recession"
        elif top_votes >= 2 and trough_votes >= 2:
            tier, label = "conflicting", None
        elif top_votes >= 2:
            tier, label = "high_confidence", "top"
        elif trough_votes >= 2:
            tier, label = "high_confidence", "trough"
        elif top_votes == 1 and trough_votes == 1:
            tier, label = "single_source", None
        elif top_votes == 1:
            tier, label = "single_source", "top_candidate"
        elif trough_votes == 1:
            tier, label = "single_source", "trough_candidate"
        else:
            tier, label = "silent", None

        rows.append(
            {
                "period": period,
                "nber_top": nber_top,
                "nber_trough": nber_trough,
                "credit_gap_flag": gap_flag,
                "credit_gap_declining": gap_declining,
                "mechanical_top": mech_top,
                "mechanical_trough": mech_trough,
                "tier": tier,
                "label": label,
            }
        )

    return pd.DataFrame(rows).set_index("period")


def stage_labels_from_consensus(
    consensus: pd.DataFrame,
    real_gdp_qoq_growth: Optional[pd.Series] = None,
    reflation_window: int = 4,
    return_training_mask: bool = False,
):
    """
    Walk the HIGH-CONFIDENCE top/trough events in `consensus` (output of
    build_consensus_labels) in chronological order and assign a full
    STAGES-vocabulary label to every quarter. Quarters before the first
    high-confidence event are left unlabeled (NaN) -- there isn't enough
    confirmed history yet to say anything, which is preferable to
    guessing.

    Rules (documented here since this is a genuine design decision, not
    something spelled out metric-by-metric in the design doc):
      - The `reflation_window` quarters right after a confirmed trough
        are labeled "reflation".
      - After that, until the next confirmed top: "bubble" if the credit
        gap says bubble_building/top_warning that quarter, else "early".
      - Every quarter after a confirmed top, until the next confirmed
        trough, is "deleveraging_beautiful" or "deleveraging_ugly",
        determined by average real GDP q/q growth over that span if
        `real_gdp_qoq_growth` is supplied. WITHOUT real GDP data, every
        such quarter defaults to "deleveraging_beautiful" -- an
        explicit, documented placeholder, not a real determination.
        Pass real_gdp_qoq_growth in any real use of this function.

    IMPORTANT distinction between the DESCRIPTIVE label and TRAINING
    eligibility. The stage label above is assigned to every quarter for
    display/timeline purposes -- it answers "given the last confirmed
    turning point, what stage does this quarter fall in." But some of
    those quarters sit in periods the three sources actively DISAGREED
    about (tier="conflicting") or that were flagged as non-debt-cycle
    recessions (tier="contested"). Those quarters must NOT be fed to the
    scorer as ground-truth training anchors just because they inherit a
    descriptive label from the preceding confirmed event -- doing so
    would train the model on exactly the periods we're least sure about.
    This was a real, subtle bug: before `return_training_mask` existed,
    a conflicting quarter like 2009Q2 (GFC trough sitting on a still-
    elevated credit gap) silently inherited "deleveraging_beautiful"
    from the 2007-08 top and would have been used as training ground
    truth as if confirmed.

    When return_training_mask=True, returns (labels, train_ok) where
    train_ok is a boolean Series marking quarters whose stage label is
    corroborated by an independent signal and may be used as scorer
    ground truth: confirmed turning-point quarters (high_confidence tier)
    for deleveraging/reflation, plus quarters where the credit-gap flag
    independently agrees with an inherited "bubble" or "early" label
    (STRICT rule). "top" is deliberately never trainable (no clean
    independent signal distinct from late-bubble). Everything
    conflicting/contested/silent/single-source, and any label merely
    inherited without independent corroboration, is excluded. Feed only
    labels[train_ok] to the scorer; show all of `labels` in the timeline.
    """
    high_conf = consensus[consensus["tier"] == "high_confidence"]
    tops = sorted(high_conf.index[high_conf["label"] == "top"])
    troughs = sorted(high_conf.index[high_conf["label"] == "trough"])
    events = sorted([(p, "top") for p in tops] + [(p, "trough") for p in troughs])

    labels = pd.Series(index=consensus.index, dtype=object)

    for period in consensus.index:
        past = [e for e in events if e[0] <= period]
        if not past:
            continue
        last_event_period, last_event_kind = past[-1]

        if last_event_kind == "top":
            if real_gdp_qoq_growth is not None:
                window_growth = real_gdp_qoq_growth.loc[last_event_period:period]
                labels[period] = (
                    "deleveraging_beautiful" if window_growth.mean() >= 0 else "deleveraging_ugly"
                )
            else:
                labels[period] = "deleveraging_beautiful"
        else:  # last confirmed event was a trough
            distance = _quarters_between(period, last_event_period)
            gap_flag = consensus.loc[period, "credit_gap_flag"]
            if distance <= reflation_window:
                labels[period] = "reflation"
            elif gap_flag in ("bubble_building", "top_warning"):
                labels[period] = "bubble"
            else:
                labels[period] = "early"

    if return_training_mask:
        # A quarter is training-eligible when its stage label is
        # corroborated by an INDEPENDENT signal, not merely inherited by
        # walking forward from the last confirmed turning point. Without
        # this, only the stages that immediately follow a turning point
        # (deleveraging after a top, reflation after a trough) ever became
        # trainable -- leaving early/bubble/top with zero training
        # examples, since those stages never coincide with a
        # high-confidence turning-point quarter. See the design
        # discussion: the fix is to let each stage earn trainability from
        # its own explicit evidence.
        #
        # Rules (STRICT variant -- require the independent credit-gap
        # signal to agree with the inherited label, producing fewer but
        # cleaner examples):
        #   - deleveraging_*/reflation: trainable when the quarter's OWN
        #     tier is high_confidence (a directly confirmed turning point),
        #     exactly as before.
        #   - bubble: trainable when the credit gap is at top_warning
        #     (BIS's >=10pp maximum-warning level -- the strongest
        #     independent "this is genuinely bubbly" statement the data
        #     makes) AND the descriptive label is "bubble" AND the tier
        #     is not contested/conflicting.
        #   - early: trainable when the credit gap flag is "early" (below
        #     the 2pp lower threshold) AND the descriptive label is
        #     "early" AND the tier is not contested/conflicting. Requiring
        #     the credit-gap signal to AGREE with the inherited label is
        #     what makes this "clear" rather than merely inherited.
        #   - top: DELIBERATELY never trainable. A "top" plateau has no
        #     clean independent signal distinct from late-bubble, so
        #     rather than invent one, the scorer simply won't have a top
        #     archetype -- at prediction time a genuine top reads as
        #     high-bubble / early-deleveraging, which is roughly how a top
        #     actually looks. This is an honest gap, documented not hidden.
        tier = consensus["tier"]
        gap = consensus["credit_gap_flag"]
        not_disputed = ~tier.isin(["contested", "conflicting"])

        confirmed_turning_point = (tier == "high_confidence") & labels.notna()
        bubble_trainable = (labels == "bubble") & (gap == "top_warning") & not_disputed
        early_trainable = (labels == "early") & (gap == "early") & not_disputed

        train_ok = confirmed_turning_point | bubble_trainable | early_trainable
        # Guard: a top label must never be trainable, even if some future
        # edit to the rules above accidentally lets one through.
        train_ok = train_ok & (labels != "top")
        return labels, train_ok

    return labels
