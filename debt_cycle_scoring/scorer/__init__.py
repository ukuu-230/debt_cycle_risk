"""
debt_cycle_scoring.scorer
=========================

The end-to-end scoring layer that ties everything else together and was
the whole point of the project: takes the normalized/aggregated metric
panel plus the training-eligible labels from the labeling pipeline, fits
the Mahalanobis stage archetypes, and produces -- for every quarter --
a stage-probability distribution AND a 0-100 risk score, plus the
combined report (current read, recent trajectory, full history).

Design decisions baked in here (see project memory / design doc):
  - POOLED, COUNTRY-AWARE from day one: every row carries a `country`
    column (just "US" for now). Archetypes are fit over pooled quarters
    regardless of country, so adding more countries later is loading
    rows, not rewriting. No country-specific branching anywhere.
  - Mahalanobis analog matching (not a heavier model), sized to the
    small real training set (~99 US quarters) to avoid overfitting.
  - Honest about data scarcity: the bubble stage has very few training
    examples, so its archetype confidence is surfaced, not hidden.
"""
from .risk_score import compute_risk_score, RiskScoreResult, STAGE_RISK_WEIGHTS
from .engine import ScorerResult, fit_and_score, prepare_scored_panel
from .validation import (
    walk_forward_predict,
    calibration_report,
    WalkForwardResult,
    CalibrationResult,
)
from .risk_flags import (
    build_risk_series,
    build_country_risk_series,
    compute_composite_risk,
    RiskSeriesResult,
    CompositeRiskResult,
    FLAG_NAMES,
    RESEARCH_WEIGHTS,
)
from .short_horizon_flags import (
    build_short_horizon_score,
    ShortHorizonResult,
    SHORT_FLAG_NAMES,
)
from .recession_backtest import (
    run_recession_backtest,
    RecessionBacktestResult,
    HorizonResult,
)

__all__ = [
    "compute_risk_score",
    "RiskScoreResult",
    "STAGE_RISK_WEIGHTS",
    "ScorerResult",
    "fit_and_score",
    "prepare_scored_panel",
    "walk_forward_predict",
    "calibration_report",
    "WalkForwardResult",
    "CalibrationResult",
    "build_risk_series",
    "build_country_risk_series",
    "compute_composite_risk",
    "RiskSeriesResult",
    "CompositeRiskResult",
    "FLAG_NAMES",
    "RESEARCH_WEIGHTS",
    "build_short_horizon_score",
    "ShortHorizonResult",
    "SHORT_FLAG_NAMES",
    "run_recession_backtest",
    "RecessionBacktestResult",
    "HorizonResult",
]
