"""
Integration demo: wires the section 3B independent labeling pass into
the scoring engine, replacing the hand-built synthetic labels used in
debt_cycle_scoring/demo.py.

Still no live network call (this sandbox has none) -- instead, this
builds a synthetic panel deliberately shaped like a real bubble -> top
-> deleveraging -> reflation episode (credit-to-GDP ratio, an NBER-style
recession indicator, and real GDP growth all constructed consistently
with each other), then runs the REAL labeling functions (not a stand-in)
against that synthetic panel: NBER turning-point extraction, the
one-sided HP filter / credit gap, Bry-Boschan mechanical turning points,
and the section 3B.4 consensus combiner.

The point of this demo is to prove the labeling pipeline itself works
end-to-end and produces sensible, HIGH-CONFIDENCE labels precisely where
a human looking at the constructed episode would expect them -- before
ever pointing it at real FRED/BIS data.

Run with: python3 -m debt_cycle_scoring.labeling.demo
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .nber import extract_recession_turning_points
from .credit_gap import compute_credit_gap
from .turning_points import bry_boschan_turning_points
from .consensus import build_consensus_labels, stage_labels_from_consensus
from ..stages import fit_archetypes, stage_probabilities, STAGES
from ..normalize import direction_encode, expanding_percentile_rank, winsorize
from ..aggregate import pca_category_score


def build_synthetic_episode(n_quarters: int = 100, seed: int = 3):
    """
    A hand-built episode: early cycle (0-59) -> bubble (60-79, credit-to-GDP
    ratio ramps up) -> NBER recession (78-83, overlapping the bubble's end,
    the way a real top-into-recession transition does) -> deleveraging
    with positive real growth, i.e. "beautiful" (84-95) -> reflation (96+).
    """
    rng = np.random.default_rng(seed)
    idx = pd.period_range("1995Q1", periods=n_quarters, freq="Q")
    t = np.arange(n_quarters)

    credit_to_gdp = np.piecewise(
        t.astype(float),
        [t < 60, (t >= 60) & (t < 80), (t >= 80) & (t < 96), t >= 96],
        [
            lambda t: 100 + 0.05 * t,
            lambda t: 103 + 1.2 * (t - 60),
            lambda t: 127 - 0.8 * (t - 80),
            lambda t: 114.2 + 0.1 * (t - 96),
        ],
    ) + rng.normal(0, 0.3, n_quarters)

    usrec = np.zeros(n_quarters, dtype=int)
    usrec[78:84] = 1  # a 6-quarter recession right at the top of the bubble

    real_gdp_growth = np.where(
        (t >= 78) & (t < 96), rng.normal(0.3, 0.2, n_quarters), rng.normal(2.0, 0.3, n_quarters)
    )  # modest but POSITIVE growth even during the downturn -> should read "beautiful"

    return (
        pd.Series(credit_to_gdp, index=idx, name="credit_to_gdp"),
        pd.Series(usrec, index=idx, name="usrec"),
        pd.Series(real_gdp_growth, index=idx, name="real_gdp_qoq_growth"),
    )


def run_demo() -> None:
    credit_to_gdp, usrec, real_gdp_growth = build_synthetic_episode()

    # --- Source A: NBER turning points ---
    nber_events = extract_recession_turning_points(usrec)
    print("NBER turning points:")
    print(nber_events.to_string())
    print()

    # --- Source B: BIS-style credit-to-GDP gap ---
    gap_df = compute_credit_gap(credit_to_gdp, min_periods=30)
    print("Credit gap around the bubble peak:")
    print(gap_df.loc["2010Q1":"2011Q4", ["ratio", "trend", "gap", "flag"]].to_string())
    print()

    # --- Source C: mechanical turning points on the same credit series ---
    mechanical_events = bry_boschan_turning_points(credit_to_gdp, window=2, min_phase_quarters=4)
    print("Mechanical (Bry-Boschan) turning points:")
    print(mechanical_events.to_string())
    print()

    # --- Section 3B.4: consensus combiner ---
    consensus = build_consensus_labels(
        credit_to_gdp.index, nber_events, gap_df, mechanical_events, agreement_window=4
    )
    print("Consensus tiers around the top (should be high_confidence, since this")
    print("synthetic episode DOES have credit-gap buildup before the recession):")
    top_period = nber_events.index[nber_events["event"] == "peak"][0]
    window = pd.period_range(top_period - 4, top_period + 4, freq="Q")
    print(
        consensus.loc[window, ["nber_top", "credit_gap_flag", "mechanical_top", "tier", "label"]].to_string()
    )
    print()

    stage_labels = stage_labels_from_consensus(consensus, real_gdp_qoq_growth=real_gdp_growth)
    print("Resulting STAGES-vocabulary labels (value counts):")
    print(stage_labels.value_counts(dropna=False).to_string())
    print()

    # --- Feed the REAL labels into the scoring engine, replacing the ---
    # --- hand-built synthetic labels used in the earlier demo.py ---
    normalized = pd.DataFrame(index=credit_to_gdp.index)
    pct = expanding_percentile_rank(winsorize(credit_to_gdp), min_periods=20)
    normalized["credit_to_gdp"] = direction_encode(pct, higher_is_later_cycle=True)
    category_scores = pd.DataFrame({"debt_burden": pca_category_score(normalized).score}).dropna()

    labeled = stage_labels.reindex(category_scores.index).dropna()
    training_scores = category_scores.loc[labeled.index]

    archetypes = fit_archetypes(training_scores, labeled)
    print("Archetypes fit on REAL (labeling-pipeline-derived) labels:")
    for stage, arche in archetypes.items():
        print(f"  {stage}: n={arche.n_observations}")
    print()

    example_period = pd.Period("2010Q3", freq="Q")
    x = category_scores.loc[example_period].values
    probs = stage_probabilities(x, archetypes, temperature=0.5)
    print(f"Stage read for {example_period} (mid-bubble, per the synthetic construction):")
    for stage, p in sorted(probs.items(), key=lambda kv: -kv[1]):
        print(f"  {stage}: {p:.0%}")


if __name__ == "__main__":
    run_demo()
