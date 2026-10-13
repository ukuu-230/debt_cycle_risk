"""
Risk score -- the 0-100 number the user asked for, alongside the stage
probability distribution.

Deliberately TRANSPARENT rather than a black box: the score is a simple,
inspectable weighted combination of two things the rest of the system
already produces, so for any quarter you can see exactly why it scored
70 vs 30:

  1. The stage-probability distribution (from the Mahalanobis scorer).
     Each stage carries a fixed risk weight reflecting where in Dalio's
     cycle it sits -- a confirmed bubble/top is high risk, early-cycle
     is low risk, deleveraging is elevated-but-declining, reflation is
     low. The stage component is the probability-weighted average of
     those per-stage weights.
  2. The raw credit-to-GDP gap level (from the labeling pipeline's
     Source B). This is included directly, NOT only via the stage, so
     that a dangerously high credit gap raises the risk score even in a
     quarter the stage classifier is uncertain about -- the gap is the
     single most load-bearing indicator in Dalio's framework and
     shouldn't be fully laundered through the stage probabilities.

The two components are blended with an explicit, adjustable weight so
the blend is a visible knob, not a hidden constant.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd

# Per-stage risk weights on a 0-100 scale. These are a judgment call,
# stated explicitly so they can be argued with and tuned rather than
# buried. Ordering reflects Dalio's cycle: risk builds through bubble to
# a peak at the top, stays elevated but declining through deleveraging,
# and is lowest in early/reflation.
STAGE_RISK_WEIGHTS: Dict[str, float] = {
    "early": 15.0,
    "bubble": 80.0,
    "top": 95.0,
    "deleveraging_ugly": 85.0,
    "deleveraging_beautiful": 55.0,
    "reflation": 25.0,
}


@dataclass
class RiskScoreResult:
    risk_score: float                       # final 0-100
    stage_component: float                  # 0-100, the probability-weighted stage risk
    gap_component: float                    # 0-100, the credit-gap contribution
    stage_weight_used: float                # blend weight actually applied to the stage component
    contributions: Dict[str, float] = field(default_factory=dict)  # per-stage prob*weight, for explainability


def _gap_to_0_100(gap: float, gap_floor: float = -5.0, gap_ceiling: float = 15.0) -> float:
    """
    Map a raw credit-to-GDP gap (percentage points) onto 0-100 by linear
    interpolation between an explicit floor and ceiling, then clip.

    The ceiling (15pp) sits above BIS's 10pp maximum-warning threshold so
    the very worst historical gaps land near but not pinned at 100; the
    floor (-5pp) reflects a meaningfully below-trend credit level. Both
    are stated, adjustable constants -- not fitted, since with so few
    real bubble episodes any "fit" would be curve-fitting to one or two
    events.
    """
    if gap is None or (isinstance(gap, float) and np.isnan(gap)):
        return np.nan
    scaled = (gap - gap_floor) / (gap_ceiling - gap_floor) * 100.0
    return float(np.clip(scaled, 0.0, 100.0))


def compute_risk_score(
    stage_probabilities: Dict[str, float],
    credit_gap: Optional[float],
    stage_weight: float = 0.6,
) -> RiskScoreResult:
    """
    stage_probabilities: dict stage -> probability (should sum to ~1),
        as produced by stages.stage_probabilities.
    credit_gap: the raw credit-to-GDP gap in percentage points for this
        quarter (from the labeling pipeline). May be None/NaN early in a
        country's history before the gap is defined -- in that case the
        score falls back to the stage component alone.
    stage_weight: how much of the final score comes from the stage
        component vs. the raw gap component. 0.6 = 60% stage / 40% gap by
        default. Exposed as an argument so the blend is a visible knob.

    Returns a RiskScoreResult with the final score AND both components
    broken out, so the number is always explainable.
    """
    contributions = {
        stage: prob * STAGE_RISK_WEIGHTS.get(stage, 0.0)
        for stage, prob in stage_probabilities.items()
    }
    stage_component = float(sum(contributions.values()))

    gap_component = _gap_to_0_100(credit_gap)

    if gap_component is None or np.isnan(gap_component):
        # No usable gap -> fall back to the stage component alone rather
        # than dragging the score toward zero with a missing input.
        return RiskScoreResult(
            risk_score=round(stage_component, 1),
            stage_component=round(stage_component, 1),
            gap_component=float("nan"),
            stage_weight_used=1.0,
            contributions=contributions,
        )

    risk = stage_weight * stage_component + (1.0 - stage_weight) * gap_component
    return RiskScoreResult(
        risk_score=round(float(risk), 1),
        stage_component=round(stage_component, 1),
        gap_component=round(gap_component, 1),
        stage_weight_used=stage_weight,
        contributions=contributions,
    )
