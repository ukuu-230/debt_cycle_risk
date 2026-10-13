"""
End-to-end demo of the scoring pipeline on SYNTHETIC data.

This does not pull real FRED/BIS data (that's the ingestion layer, not
built yet -- see the design doc's Phase 1 plan). It exists to prove the
normalize -> aggregate -> classify -> explain pipeline runs correctly and
behaves sensibly on a hand-constructed sequence that mimics a real
bubble -> top -> deleveraging episode, before any real data is wired in.

Run with: python3 -m debt_cycle_scoring.demo
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .normalize import expanding_percentile_rank, winsorize, direction_encode
from .aggregate import pca_category_score
from .stages import fit_archetypes, stage_probabilities, STAGES
from .explain import decompose_distance


def make_synthetic_history(n_quarters: int = 160, seed: int = 7) -> pd.DataFrame:
    """
    Build a synthetic quarterly panel with two categories:
      - debt_burden: two correlated metrics that ramp up into a bubble
        (quarters ~100-130), then collapse (deleveraging, ~130-145),
        then recover (reflation, ~145+)
      - cb_stance: a metric that tightens (rises) into the top and eases
        (falls) during deleveraging -- deliberately out of phase with
        debt_burden so the two categories aren't just restating each other.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_quarters)

    # Debt burden & growth category: two correlated underlying metrics
    bubble_path = np.piecewise(
        t.astype(float),
        [t < 100, (t >= 100) & (t < 130), (t >= 130) & (t < 145), t >= 145],
        [
            lambda t: 0.3 + 0.001 * t,
            lambda t: 3.3 + 0.04 * (t - 100),
            lambda t: 4.5 - 0.09 * (t - 130),
            lambda t: 3.15 + 0.01 * (t - 145),
        ],
    )
    debt_gdp = bubble_path + rng.normal(0, 0.05, n_quarters)
    credit_growth = bubble_path * 0.8 + rng.normal(0, 0.08, n_quarters)

    # Central bank stance: tightens into the top, eases during deleveraging
    cb_path = np.piecewise(
        t.astype(float),
        [t < 115, (t >= 115) & (t < 132), t >= 132],
        [lambda t: 0.2, lambda t: 0.2 + 0.15 * (t - 115), lambda t: 2.75 - 0.05 * (t - 132)],
    )
    real_rate = cb_path + rng.normal(0, 0.05, n_quarters)

    idx = pd.period_range("1985Q1", periods=n_quarters, freq="Q")
    return pd.DataFrame(
        {"debt_gdp": debt_gdp, "credit_growth": credit_growth, "real_rate": real_rate},
        index=idx,
    )


def build_synthetic_labels(index: pd.PeriodIndex) -> pd.Series:
    """
    Hand-labeled ground truth matching how make_synthetic_history was
    constructed, standing in for the section 3B consensus-label output
    until real NBER/BIS/Bry-Boschan labels are wired in.
    """
    t = np.arange(len(index))
    labels = np.select(
        [t < 100, (t >= 100) & (t < 128), (t >= 128) & (t < 132), (t >= 132) & (t < 145), t >= 145],
        ["early", "bubble", "top", "deleveraging_beautiful", "reflation"],
        default="early",
    )
    return pd.Series(labels, index=index)


def run_demo() -> None:
    raw = make_synthetic_history()
    labels = build_synthetic_labels(raw.index)

    # --- Step 1: normalize (section 3A.1) ---
    normalized = pd.DataFrame(index=raw.index)
    normalized["debt_gdp"] = direction_encode(
        expanding_percentile_rank(winsorize(raw["debt_gdp"])), higher_is_later_cycle=True
    )
    normalized["credit_growth"] = direction_encode(
        expanding_percentile_rank(winsorize(raw["credit_growth"])), higher_is_later_cycle=True
    )
    normalized["real_rate"] = direction_encode(
        expanding_percentile_rank(winsorize(raw["real_rate"])), higher_is_later_cycle=True
    )

    # --- Step 2: aggregate into categories (section 3A.2) ---
    debt_burden_result = pca_category_score(normalized[["debt_gdp", "credit_growth"]])
    cb_stance_result = pca_category_score(normalized[["real_rate"]])

    category_scores = pd.DataFrame(
        {
            "debt_burden": debt_burden_result.score,
            "cb_stance": cb_stance_result.score,
        }
    ).dropna()

    print("Debt Burden category -- PC1 explained variance ratio: "
          f"{debt_burden_result.explained_variance_ratio:.2f}")
    print("Debt Burden category -- PC1 loadings:")
    print(debt_burden_result.loadings.to_string())
    print()

    # --- Step 3: fit stage archetypes on the (synthetic) high-confidence
    # labels, then classify a few example quarters (section 3A.3) ---
    aligned_labels = labels.reindex(category_scores.index)
    archetypes = fit_archetypes(category_scores, aligned_labels)

    print("Archetypes fit (stage: n_observations):")
    for stage, arche in archetypes.items():
        print(f"  {stage}: n={arche.n_observations}")
    print()

    example_quarters = ["1993Q1", "2010Q1", "2013Q4", "2016Q4", "2018Q1"]  # early / bubble / top / delev / reflation
    for q in example_quarters:
        period = pd.Period(q, freq="Q")
        if period not in category_scores.index:
            continue
        x = category_scores.loc[period].values
        probs = stage_probabilities(x, archetypes, temperature=0.5)
        top_stage = max(probs, key=probs.get)
        print(f"{q} (true label: {aligned_labels.get(period, 'n/a')}) -> "
              f"predicted {top_stage} ({probs[top_stage]:.0%})")
        print("  full distribution: " + ", ".join(f"{s}={p:.0%}" for s, p in probs.items()))

        # --- Step 4: explain the top prediction (section 3A.5) ---
        decomposition = decompose_distance(x, archetypes[top_stage], list(category_scores.columns))
        print("  distance decomposition (higher = more inconsistent with this stage):")
        print("    " + decomposition.to_string().replace("\n", "\n    "))
        print()


if __name__ == "__main__":
    run_demo()
