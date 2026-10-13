"""
Composite risk score from independent flags -- scorer/risk_flags.py

This replaces the stage-classifier-dependent risk score (which the
walk-forward backtest showed does not beat baseline). The risk read is
now a composite of independently-validated risk flags, following the
method real financial-stress indices use (Fed / OFR): normalize each
flag to a common 0-100 risk scale, orient so higher = more risk, then
combine. No dependence on the stage classifier.

WHY NOT PCA to combine the flags: PCA's PC1 captures maximum-variance,
not maximum-risk, and can be hijacked by one noisy high-variance flag
(e.g. gold). Composite-index practice normalizes and weights instead,
which is robust, transparent, and un-hijackable.

WHY NOT weights fit to our own data: with ~2-3 US crises, any weights
tuned to outcomes overfit to 2008 specifically. Instead we use FIXED,
horizon-conditional weights grounded in EXTERNAL published research
(BIS EWI studies, Fed/OFR stress indices, Adrian-Estrella-Shin on the
curve). Key research findings the weights encode:
  - BIS WP421: the credit-to-GDP gap is the best EWI at LONGER horizons;
    the debt service ratio (DSR) dominates at SHORTER horizons.
  - Yield-curve inversion leads recessions by ~6-24 months (shorter
    horizon than the gap).
  - Credit spreads are largely COINCIDENT (spike as stress hits).
  - No single indicator suffices; combine to cut false positives.
Because "which flag matters most" depends on horizon, and the credit
gap itself tells us roughly where in the build-up we are, the weights
are conditional on the gap regime -- NOT on any flag's current value
(weighting by "whichever reads highest" would just be a risk maximizer,
not a gauge).

The equal-weight composite is ALWAYS computed alongside the research-
weighted one. If they diverge sharply, that divergence is itself
diagnostic and is surfaced, not hidden.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..labeling.credit_gap import BASEL_LOWER_THRESHOLD, BASEL_UPPER_THRESHOLD


# ---------------------------------------------------------------------------
# Flag definitions. Each flag maps some raw input(s) to a 0-100 risk value
# where higher = more risk. Kept as small pure functions so each is
# independently testable and its logic is fully visible.
# ---------------------------------------------------------------------------

def _clip01_to_100(x: float) -> float:
    return float(np.clip(x, 0.0, 100.0))


def flag_credit_gap(gap: float) -> float:
    """
    Credit-to-GDP gap (pp) -> 0-100. Anchored on Basel III's OWN published
    thresholds (2pp lower / 10pp upper), not invented cutoffs: at/below the
    lower threshold ~ low risk, at/above the upper threshold ~ max risk,
    linear between. Below the lower threshold scales down toward 0.
    """
    if gap is None or np.isnan(gap):
        return np.nan
    lo, hi = BASEL_LOWER_THRESHOLD, BASEL_UPPER_THRESHOLD
    if gap <= lo:
        # below the activation threshold: scale 0 at gap=-5, ~20 at gap=lo
        return _clip01_to_100((gap - (-5.0)) / (lo - (-5.0)) * 20.0)
    return _clip01_to_100(20.0 + (gap - lo) / (hi - lo) * 80.0)


def flag_debt_service_ratio(dsr_percentile: float) -> float:
    """
    DSR has no universal absolute threshold, so it enters as a percentile
    of its own history (0-1) -> 0-100. BIS WP421: DSR dominates at short
    horizons, which the WEIGHTING (below) reflects; the flag value itself
    is just "how elevated vs history".
    """
    if dsr_percentile is None or np.isnan(dsr_percentile):
        return np.nan
    return _clip01_to_100(dsr_percentile * 100.0)


def flag_yield_curve(spread_10y_minus_short: float) -> float:
    """
    Term spread (pp) -> 0-100. Inversion (negative spread) is the risk
    signal. Maps: spread >= +2.5 -> ~0 risk; spread = 0 (flat) -> ~60;
    spread <= -1.0 (deep inversion) -> 100. Piecewise-linear, thresholds
    from the standard recession-signal literature (inversion, not level).
    """
    if spread_10y_minus_short is None or np.isnan(spread_10y_minus_short):
        return np.nan
    s = spread_10y_minus_short
    if s >= 2.5:
        return 0.0
    if s >= 0.0:
        return _clip01_to_100((2.5 - s) / 2.5 * 60.0)          # 0 at +2.5 -> 60 at 0
    return _clip01_to_100(60.0 + (0.0 - s) / 1.0 * 40.0)        # 60 at 0 -> 100 at -1.0


def flag_credit_spread(spread_percentile: float) -> float:
    """
    High-yield OAS as a percentile of its own history -> 0-100. Spreads
    are largely coincident, so like DSR the flag is "how wide vs history";
    the coincident nature is reflected in weighting, not the value.
    """
    if spread_percentile is None or np.isnan(spread_percentile):
        return np.nan
    return _clip01_to_100(spread_percentile * 100.0)


def flag_m2_velocity(velocity_percentile: float) -> float:
    """
    Percentile of M2 velocity's year-on-year % change -> 0-100, INVERTED:
    FALLING velocity is the stress signal (easing not reaching the real
    economy, MP2->MP3). So risk = (1 - percentile) * 100. (Fed the change,
    not the level: the level trends down structurally, so a level
    percentile is a trend detector, not a risk signal.)
    """
    if velocity_percentile is None or np.isnan(velocity_percentile):
        return np.nan
    return _clip01_to_100((1.0 - velocity_percentile) * 100.0)


def flag_long_bond_level(yield_percentile: float) -> float:
    """
    Long-bond yield LEVEL percentile within a trailing 10-year window ->
    0-100. Interpreted as a mild debt-servicing-cost stress: long rates
    unusually high versus the recent regime raise the cost of rolling debt. Weak flag (weighted low); included for completeness
    per the user's request for the long-bond level.
    """
    if yield_percentile is None or np.isnan(yield_percentile):
        return np.nan
    return _clip01_to_100(yield_percentile * 100.0)


def flag_debasement(breakeven_inflation_pct: float, target: float = 2.0) -> float:
    """
    'Monetary debasement / deflation stress' from 10y breakeven inflation
    expectations (T10YIE). Replaces the former gold flag: FRED deleted ALL
    LBMA gold price series on 2022-01-31 (ICE Benchmark Administration data
    removed), so no daily gold price is available on FRED. Breakeven
    inflation is a cleaner debasement signal anyway -- it isolates expected
    fiat purchasing-power erosion, whereas gold was contaminated by real-
    rate and safe-haven noise.

    TWO-TAILED, per Dalio's framework: BOTH runaway inflation expectations
    (currency debasement) AND very low/negative expectations (deflationary
    debt-spiral risk) are danger signals. Risk is LOWEST when breakevens
    sit near the Fed's ~2% target and rises as they deviate in EITHER
    direction:
      - at target (2.0%)         -> ~0 risk
      - +/-1.5pp from target     -> ~60 risk
      - +/-3pp or more from target-> ~100 risk (either 5%+ inflation
                                     expectations OR deflation ~ -1%)

    Note T10YIE only starts 2003, so this flag is NaN before then --
    acceptable since the short-horizon score is about recent stress and
    the structural composite renormalizes over available flags.
    """
    if breakeven_inflation_pct is None or np.isnan(breakeven_inflation_pct):
        return np.nan
    deviation = abs(breakeven_inflation_pct - target)
    if deviation <= 1.5:
        return _clip01_to_100(deviation / 1.5 * 60.0)          # 0 at target -> 60 at +/-1.5
    return _clip01_to_100(60.0 + (deviation - 1.5) / 1.5 * 40.0)  # 60 -> 100 at +/-3.0


# ---------------------------------------------------------------------------
# Horizon-conditional weights, grounded in external research (see module
# docstring). Regime is set by the credit gap (where in the build-up we
# are), NOT by any flag's current value.
#
# Three regimes:
#   - "calm"    (gap < 2pp): no build-up. Coincident/short flags (curve,
#                spreads) matter more for catching an unexpected shock;
#                the gap itself carries little.
#   - "building"(2 <= gap < 10): the classic build-up. The GAP is the
#                dominant long-horizon signal (BIS WP421), DSR next.
#   - "elevated"(gap >= 10): late build-up / imminent. Short-horizon
#                flags (DSR, curve, spreads) gain weight as the trigger
#                nears; the gap is already maxed and adds less at the margin.
#
# Weights within each regime sum to 1 across the flags present. Flags
# missing for a quarter are dropped and the remainder renormalized.
# ---------------------------------------------------------------------------

# Specification v1 (through 2026-10-02): seven flags including the credit-
# spread LEVEL. Kept so the before/after of the v2 change can always be shown.
SPEC_V1_FLAG_NAMES: List[str] = [
    "credit_gap", "debt_service_ratio", "yield_curve",
    "credit_spread", "m2_velocity", "long_bond_level", "debasement",
]
SPEC_V1_WEIGHTS: Dict[str, Dict[str, float]] = {
    "calm": {
        "credit_gap": 0.10, "debt_service_ratio": 0.15, "yield_curve": 0.25,
        "credit_spread": 0.25, "m2_velocity": 0.10, "long_bond_level": 0.05,
        "debasement": 0.10,
    },
    "building": {
        "credit_gap": 0.32, "debt_service_ratio": 0.20, "yield_curve": 0.15,
        "credit_spread": 0.13, "m2_velocity": 0.08, "long_bond_level": 0.04,
        "debasement": 0.08,
    },
    "elevated": {
        "credit_gap": 0.20, "debt_service_ratio": 0.24, "yield_curve": 0.22,
        "credit_spread": 0.16, "m2_velocity": 0.08, "long_bond_level": 0.04,
        "debasement": 0.06,
    },
}

# Specification v2 (2026-10-05, user decision): the credit-spread LEVEL is
# removed from the STRUCTURAL composite and lives in the short-horizon stress
# gauge (level + widening). Reason, stated in the original design notes before
# any data was seen: spreads are largely COINCIDENT -- wide in and after
# recessions, tight before them -- so they measure stress that has arrived,
# not stress that is building. The live run confirmed it (standalone 8Q AUC
# 0.40; it also pulled 2006-07 down). This change was made AFTER seeing that
# result, so any AUC gain from it is NOT validation; v1 numbers are reported
# alongside. The remaining weights are the v1 weights renormalized -- no
# weight was re-chosen.
FLAG_NAMES: List[str] = [
    "credit_gap", "debt_service_ratio", "yield_curve",
    "m2_velocity", "long_bond_level", "debasement",
]
CONTEXT_FLAG_NAMES: List[str] = ["credit_spread"]   # computed and shown, never weighted


def _renormalize_without(weights: Dict[str, Dict[str, float]], drop: str) -> Dict[str, Dict[str, float]]:
    out = {}
    for regime, w in weights.items():
        kept = {k: v for k, v in w.items() if k != drop}
        tot = sum(kept.values())
        out[regime] = {k: v / tot for k, v in kept.items()}
    return out


RESEARCH_WEIGHTS: Dict[str, Dict[str, float]] = _renormalize_without(SPEC_V1_WEIGHTS, "credit_spread")


def regime_from_gap(gap: Optional[float]) -> str:
    if gap is None or np.isnan(gap):
        return "building"  # neutral default when the gap itself is unavailable
    if gap < BASEL_LOWER_THRESHOLD:
        return "calm"
    if gap < BASEL_UPPER_THRESHOLD:
        return "building"
    return "elevated"


@dataclass
class CompositeRiskResult:
    research_weighted: float             # 0-100, horizon-conditional research weights
    equal_weighted: float                # 0-100, equal weights (sanity check)
    divergence: float                     # |research - equal|, surfaced not hidden
    regime: str                           # which weight set was used
    flag_values: Dict[str, float] = field(default_factory=dict)   # each flag's 0-100 value
    flag_contributions: Dict[str, float] = field(default_factory=dict)  # weight*value under research weights
    n_flags_present: int = 0


def compute_composite_risk(flag_values: Dict[str, float], gap: Optional[float],
                           min_flags: int = 2,
                           weights: Optional[Dict[str, Dict[str, float]]] = None) -> CompositeRiskResult:
    """
    flag_values: {flag_name: 0-100 risk value}. Any flag missing/NaN is
        dropped and the remaining weights renormalized.
    gap: the raw credit-gap (pp), used ONLY to pick the regime weight set.
    min_flags: a quarter scored on FEWER than this many present flags is
        NOT given a composite score (returns NaN), because a "composite"
        resting on one flag is just that flag wearing a risk label -- it
        produces false-authority readings (e.g. a lone long-bond flag
        pinning at 100 during a rate spike, decades before the credit-gap
        anchor even exists). n_flags_present is still reported for
        transparency. Default 2. The flag COUNT is always returned even
        when the score is suppressed.

    Returns both the research-weighted and equal-weighted composites plus
    their divergence and the full per-flag breakdown.
    """
    present = {k: v for k, v in flag_values.items() if v is not None and not (isinstance(v, float) and np.isnan(v))}
    if len(present) < min_flags:
        # too little evidence to form a composite -- report the count, not a score
        return CompositeRiskResult(
            research_weighted=np.nan, equal_weighted=np.nan, divergence=np.nan,
            regime=regime_from_gap(gap),
            flag_values={k: round(v, 1) for k, v in present.items()},
            flag_contributions={}, n_flags_present=len(present),
        )

    regime = regime_from_gap(gap)
    W = RESEARCH_WEIGHTS if weights is None else weights
    raw_w = {k: W[regime].get(k, 0.0) for k in present}
    total_w = sum(raw_w.values())
    if total_w <= 0:
        # no research weight for the present flags -> fall back to equal
        norm_w = {k: 1.0 / len(present) for k in present}
    else:
        norm_w = {k: w / total_w for k, w in raw_w.items()}

    research_weighted = sum(norm_w[k] * present[k] for k in present)
    equal_weighted = float(np.mean(list(present.values())))
    contributions = {k: norm_w[k] * present[k] for k in present}

    return CompositeRiskResult(
        research_weighted=round(float(research_weighted), 1),
        equal_weighted=round(equal_weighted, 1),
        divergence=round(abs(research_weighted - equal_weighted), 1),
        regime=regime,
        flag_values={k: round(v, 1) for k, v in present.items()},
        flag_contributions={k: round(v, 1) for k, v in contributions.items()},
        n_flags_present=len(present),
    )


# ---------------------------------------------------------------------------
# Builder: turn a live panel + credit gap into a per-quarter composite risk
# series. This is the wiring layer between ingestion/labeling and the pure
# flag functions above. Kept here (not in engine.py) so the entire
# composite-risk path lives in one file.
# ---------------------------------------------------------------------------

# Panel column names the flags draw from (exact names from the catalog).
_COL_DSR = "Household debt service ratio"
_COL_SPREAD = "ICE BofA US High Yield OAS (credit spread)"       # fallback only (short history on FRED)
_COL_BAA = "Moody's Baa corporate yield (monthly)"
_COL_BAA10Y_DAILY = "Moody's Baa - 10y Treasury spread (daily)"
_COL_CURVE = "10y-2y Treasury spread"                       # fallback only (starts 1976)
_COL_CURVE_3M_DAILY = "10y-3m Treasury spread"              # fallback (starts 1982)
_COL_GS10 = "10-year Treasury yield (monthly, long history)"
_COL_TB3MS = "3-month Treasury bill (monthly, long history)"
_COL_VELOCITY = "M2 velocity"
_COL_LONGBOND = "10-year Treasury yield (level)"
_COL_BREAKEVEN = "10-year breakeven inflation expectations"

# Quarters between a reference quarter and the end of the quarter in which
# its value is actually PUBLISHED. The composite for quarter q may only use
# what was public by the end of q, so quarterly macro inputs are shifted
# forward by these lags. Conservative (rounded up) on purpose:
#   - BIS credit-to-GDP: quarterly releases ~2 quarters after the reference quarter
#   - Fed household debt service ratio: ~10-12 weeks after quarter-end, can slip
#   - M2 velocity: needs the GDP advance estimate, ~4 weeks after quarter-end
# Daily market series (curve, spreads, breakevens, yields) have no lag.
# Before the statistics audit these were aligned by REFERENCE quarter, which
# let the backtest use data before it existed and inflated measured lead time.
PUBLICATION_LAGS: Dict[str, int] = {"credit_gap": 2, "debt_service_ratio": 2, "m2_velocity": 1}

# Long-bond percentile window. An EXPANDING percentile of a trending series
# measures the trend, not the signal: the 10y yield rose almost monotonically
# 1962-81, so its expanding percentile sat near 100 through the whole 1970s.
# A trailing 10-year window reads "high relative to the recent regime".
LONG_BOND_WINDOW = 40


def _midrank(hist: np.ndarray, cur: float) -> float:
    """
    Mid-rank percentile in [0,1]: share of prior values strictly below `cur`,
    plus HALF the share exactly equal. Ties get half credit, so a flat
    series reads 0.5, not 1.0. (The pre-audit '<=' convention read every
    tie as the top of the distribution, biasing flags upward.)
    """
    hist = np.asarray(hist, dtype=float)
    return float((hist < cur).mean() + 0.5 * (hist == cur).mean())


def _expanding_percentile(series: pd.Series, min_periods: int = 20) -> pd.Series:
    """
    Backward-looking expanding percentile rank in [0,1]: mid-rank of the
    current value among PRIOR observations (ties get half credit; the
    current point is excluded from its own reference set). No look-ahead. Only appropriate for roughly
    stationary inputs -- see LONG_BOND_WINDOW and the M2 velocity note.
    """
    def _pct_of_last(window: np.ndarray) -> float:
        *hist, cur = window
        hist = np.asarray(hist)
        if len(hist) == 0:
            return np.nan
        return _midrank(hist, cur)
    return series.expanding(min_periods=min_periods + 1).apply(_pct_of_last, raw=True)


def _rolling_percentile(series: pd.Series, window: int, min_periods: int = 20) -> pd.Series:
    """Same as _expanding_percentile but against only the trailing `window` prior points."""
    def _pct_of_last(w: np.ndarray) -> float:
        *hist, cur = w
        hist = np.asarray(hist)
        return _midrank(hist, cur) if len(hist) else np.nan
    return series.rolling(window + 1, min_periods=min_periods + 1).apply(_pct_of_last, raw=True)


def _full_quarter_index(*objs) -> pd.PeriodIndex:
    """Contiguous quarterly index spanning every non-empty input (so shifts are by real quarters)."""
    lo, hi = None, None
    for o in objs:
        if o is None or len(o.index) == 0 or not isinstance(o.index, pd.PeriodIndex):
            continue
        if o is None or len(o.index) == 0:
            continue
        lo = o.index.min() if lo is None else min(lo, o.index.min())
        hi = o.index.max() if hi is None else max(hi, o.index.max())
    if lo is None:
        return pd.PeriodIndex([], freq="Q")
    return pd.period_range(lo, hi, freq="Q")


# After re-timing, the latest PUBLISHED value is carried forward for up to this
# many quarters -- exactly what a real-time reader has in hand while waiting for
# the next release. Without it the newest quarter lost the credit gap and the
# regime silently fell back to its "building" default (seen on the live run).
CARRY_FORWARD_QUARTERS = 2


def _published(series: pd.Series, lag: int, idx: pd.PeriodIndex) -> pd.Series:
    """Re-time a reference-quarter series to the quarter its value became public,
    then carry the latest published value forward (no look-ahead: only past releases)."""
    s = series.reindex(idx).shift(lag) if lag else series.reindex(idx)
    return s.ffill(limit=CARRY_FORWARD_QUARTERS)


def select_curve(scored_metrics: pd.DataFrame, reference_series: pd.DataFrame,
                 idx: pd.PeriodIndex):
    """
    10y-3m term spread, quarterly mean of monthly GS10 - TB3MS (from 1953),
    the spread the curve's research weight is based on. Falls back to the
    daily 10y-3m (1982+) and then 10y-2y (1976+) if the monthly pair is
    unavailable. Returns (spread Series, source label).
    """
    if _COL_GS10 in reference_series.columns and _COL_TB3MS in reference_series.columns:
        sp = (reference_series[_COL_GS10] - reference_series[_COL_TB3MS]).dropna()
        if len(sp):
            sp = sp.reindex(idx)
            label = "10y-3m (GS10 - TB3MS, monthly avg)"
            # Per-quarter fallback to the SAME spread from the daily series: the
            # monthly pair posts after month-end, so the current quarter would
            # otherwise lose its curve for weeks. Never fill from 10y-2y (a
            # different spread with a different level).
            if _COL_CURVE_3M_DAILY in scored_metrics.columns:
                filled = sp.combine_first(scored_metrics[_COL_CURVE_3M_DAILY].reindex(idx))
                if filled.notna().sum() > sp.notna().sum():
                    label += ", latest from daily T10Y3M"
                sp = filled
            return sp, label
    for name, label in ((_COL_CURVE_3M_DAILY, "10y-3m (daily T10Y3M)"), (_COL_CURVE, "10y-2y (T10Y2Y)")):
        if name in scored_metrics.columns and scored_metrics[name].notna().any():
            return scored_metrics[name].reindex(idx), label
    return pd.Series(np.nan, index=idx), "unavailable"


def select_spread(scored_metrics: pd.DataFrame, reference_series: pd.DataFrame,
                  idx: pd.PeriodIndex):
    """
    Credit spread for the composite: Moody's Baa minus the 10y Treasury
    (monthly BAA - GS10, quarterly mean, from 1919), with the current
    quarter filled from the daily BAA10Y. Falls back to HY OAS only if the
    Moody's series are unavailable. Returns (spread Series, source label).
    """
    ref = reference_series
    if _COL_BAA in ref.columns and _COL_GS10 in ref.columns:
        sp = (ref[_COL_BAA] - ref[_COL_GS10]).dropna()
        if len(sp):
            sp = sp.reindex(idx)
            label = "Baa - 10y (Moody's, monthly avg)"
            if _COL_BAA10Y_DAILY in ref.columns:
                filled = sp.combine_first(ref[_COL_BAA10Y_DAILY].reindex(idx))
                if filled.notna().sum() > sp.notna().sum():
                    label += ", latest from daily BAA10Y"
                sp = filled
            return sp, label
    if _COL_BAA10Y_DAILY in ref.columns and ref[_COL_BAA10Y_DAILY].notna().any():
        return ref[_COL_BAA10Y_DAILY].reindex(idx), "Baa - 10y (daily BAA10Y)"
    if _COL_SPREAD in scored_metrics.columns and scored_metrics[_COL_SPREAD].notna().any():
        return scored_metrics[_COL_SPREAD].reindex(idx), "HY OAS (ICE, short history)"
    return pd.Series(np.nan, index=idx), "unavailable"


def headline_quarter(risk_result, min_flags: int = 4, lookback: int = 8):
    """
    Latest quarter scored on the FULL recent flag set: n_flags must reach the
    most flags present in any of the last `lookback` quarters (and at least
    `min_flags`). A brand-new quarter scored on fewer flags -- because monthly
    or daily inputs haven't posted yet -- is provisional, not the headline.
    Returns (headline_quarter, required_flag_count).
    """
    n = risk_result.n_flags[risk_result.risk.notna()]
    if n.empty:
        return None, min_flags
    required = max(min_flags, int(n.tail(lookback).max()))
    ok = n[n >= required]
    return (ok.index.max() if len(ok) else n.index.max()), required


@dataclass
class RiskSeriesResult:
    risk: pd.Series                  # research-weighted composite, 0-100, per quarter
    risk_equal: pd.Series            # equal-weighted composite, per quarter
    divergence: pd.Series            # |research - equal| per quarter
    regime: pd.Series                # regime label per quarter
    flags: pd.DataFrame              # per-quarter 0-100 value of each flag (transparency)
    n_flags: pd.Series               # how many flags were present each quarter
    gap_used: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))  # publication-aligned gap (sets regime)
    curve_source: str = ""
    spread_source: str = ""
    context_flags: pd.DataFrame = field(default_factory=pd.DataFrame)   # computed, never weighted
    publication_lags: Dict[str, int] = field(default_factory=dict)


def build_risk_series(
    scored_metrics: pd.DataFrame,
    reference_series: pd.DataFrame,
    credit_gap: pd.Series,
    min_periods: int = 20,
    publication_lags: Optional[Dict[str, int]] = None,
) -> RiskSeriesResult:
    """
    Compute the composite risk score for every quarter from a live panel.

    Input transformations (each chosen so the input is roughly STATIONARY
    before it is ranked -- ranking a trending series measures the trend):
      - credit gap: one-sided HP gap (already detrended), Basel anchors
      - DSR, credit spread (Moody's Baa - 10y): expanding percentile of the level
      - yield curve: 10y-3m spread level via the flag's own anchors
      - M2 velocity: expanding percentile of its YEAR-ON-YEAR % CHANGE,
        inverted. The level fell structurally after 1997, so a level
        percentile read "alarm" continuously from 2008 regardless of
        conditions; the change captures Dalio's actual signal (velocity
        FALLING as easing stops transmitting).
      - long-bond yield: percentile within a trailing 10-year window
      - breakeven inflation: two-tailed distance from 2% via flag anchors

    publication_lags: {flag: quarters} -- defaults to PUBLICATION_LAGS.
        Pass {} to align by reference quarter (the pre-audit behaviour).
    """
    lags = PUBLICATION_LAGS if publication_lags is None else publication_lags
    idx = _full_quarter_index(scored_metrics, reference_series)
    if len(idx) == 0:
        idx = scored_metrics.index

    def col(df: pd.DataFrame, name: str) -> pd.Series:
        return df[name].dropna() if name in df.columns else pd.Series(dtype=float)

    dsr_pct = _published(_expanding_percentile(col(scored_metrics, _COL_DSR), min_periods),
                         lags.get("debt_service_ratio", 0), idx)
    spread, spread_source = select_spread(scored_metrics, reference_series, idx)
    spread_pct = _expanding_percentile(spread.dropna(), min_periods).reindex(idx)
    vel = col(scored_metrics, _COL_VELOCITY)
    # year-on-year change by QUARTER (reindex first), not by row position
    vel_yoy = (vel.reindex(pd.period_range(vel.index.min(), vel.index.max(), freq="Q"))
               .pct_change(4, fill_method=None).dropna() * 100.0) if len(vel) else vel
    vel_pct = _published(_expanding_percentile(vel_yoy, min_periods), lags.get("m2_velocity", 0), idx)
    lb_pct = _rolling_percentile(col(reference_series, _COL_LONGBOND), LONG_BOND_WINDOW, min_periods).reindex(idx)

    curve, curve_source = select_curve(scored_metrics, reference_series, idx)
    breakeven = col(reference_series, _COL_BREAKEVEN).reindex(idx)
    gap = _published(credit_gap.dropna() if credit_gap is not None else pd.Series(dtype=float),
                     lags.get("credit_gap", 0), idx)

    rows = {}
    flag_rows = {}
    context_rows = {}
    for q in idx:
        g = gap.get(q, np.nan)
        fv = {
            "credit_gap": flag_credit_gap(g),
            "debt_service_ratio": flag_debt_service_ratio(dsr_pct.get(q, np.nan)),
            "yield_curve": flag_yield_curve(curve.get(q, np.nan)),
            "m2_velocity": flag_m2_velocity(vel_pct.get(q, np.nan)),
            "long_bond_level": flag_long_bond_level(lb_pct.get(q, np.nan)),
            "debasement": flag_debasement(breakeven.get(q, np.nan)),
        }
        res = compute_composite_risk(fv, g)
        rows[q] = res
        flag_rows[q] = res.flag_values
        context_rows[q] = {"credit_spread": flag_credit_spread(spread_pct.get(q, np.nan))}

    risk = pd.Series({q: r.research_weighted for q, r in rows.items()})
    risk_equal = pd.Series({q: r.equal_weighted for q, r in rows.items()})
    divergence = pd.Series({q: r.divergence for q, r in rows.items()})
    regime = pd.Series({q: r.regime for q, r in rows.items()})
    n_flags = pd.Series({q: r.n_flags_present for q, r in rows.items()})
    flags = pd.DataFrame.from_dict(flag_rows, orient="index").reindex(index=list(rows.keys()), columns=FLAG_NAMES)

    return RiskSeriesResult(
        risk=risk, risk_equal=risk_equal, divergence=divergence,
        regime=regime, flags=flags, n_flags=n_flags,
        gap_used=gap, curve_source=curve_source, spread_source=spread_source,
        context_flags=pd.DataFrame.from_dict(context_rows, orient="index").reindex(
            index=list(rows.keys()), columns=CONTEXT_FLAG_NAMES),
        publication_lags=dict(lags),
    )


def build_country_risk_series(
    inputs: Dict[str, pd.Series],
    credit_gap: pd.Series,
    min_periods: int = 20,
) -> RiskSeriesResult:
    """
    Multi-country composite risk. Same math as build_risk_series, but takes
    a flat, country-agnostic dict of already-quarterly input Series keyed by
    a canonical name, instead of the US-specific panel frames. Any input
    that is missing or None simply produces a NaN flag, and
    compute_composite_risk renormalizes the research weights over whatever
    flags are present -- so a country with 5 of 7 flags is scored honestly
    on those 5, with n_flags reporting the count.

    Expected keys in `inputs` (any may be absent):
      dsr, credit_spread, curve, m2_velocity, long_bond, breakeven
    plus `credit_gap` passed separately (the gap, not the ratio).

    The yield-curve input `curve` is expected pre-computed (10y - short)
    by the caller, since which short rate to use is country-specific; if a
    country has no short rate, the caller omits `curve` and the flag drops.
    """
    # union index across whatever inputs exist, plus the gap
    present = [s for s in list(inputs.values()) + [credit_gap] if s is not None and len(s)]
    if not present:
        raise ValueError("build_country_risk_series: no usable inputs for this country")
    idx = present[0].index
    for s in present[1:]:
        idx = idx.union(s.index)
    idx = idx.sort_values()

    def series(name: str) -> pd.Series:
        s = inputs.get(name)
        return s.reindex(idx) if s is not None else pd.Series(np.nan, index=idx)

    # Same input transforms as build_risk_series (statistics audit): velocity
    # as yoy change, long bond in a trailing window, quarterly macro re-timed
    # to publication. Kept in step so the two paths are comparable.
    full = pd.period_range(idx.min(), idx.max(), freq="Q")
    lags = PUBLICATION_LAGS
    dsr_pct = _published(_expanding_percentile(series("dsr").dropna(), min_periods),
                         lags["debt_service_ratio"], full).reindex(idx)
    spread_pct = _expanding_percentile(series("credit_spread").dropna(), min_periods).reindex(idx)
    v = series("m2_velocity").reindex(full)
    vel_pct = _published(_expanding_percentile((v.pct_change(4, fill_method=None) * 100.0).dropna(), min_periods),
                         lags["m2_velocity"], full).reindex(idx)
    lb_pct = _rolling_percentile(series("long_bond").dropna(), LONG_BOND_WINDOW, min_periods).reindex(idx)
    curve = series("curve")
    breakeven = series("breakeven")
    gap = (_published(credit_gap.dropna(), lags["credit_gap"], full).reindex(idx)
           if credit_gap is not None and len(credit_gap.dropna()) else pd.Series(np.nan, index=idx))

    rows, flag_rows = {}, {}
    for q in idx:
        fv = {
            "credit_gap": flag_credit_gap(gap.get(q, np.nan)),
            "debt_service_ratio": flag_debt_service_ratio(dsr_pct.get(q, np.nan)),
            "yield_curve": flag_yield_curve(curve.get(q, np.nan)),
            "m2_velocity": flag_m2_velocity(vel_pct.get(q, np.nan)),
            "long_bond_level": flag_long_bond_level(lb_pct.get(q, np.nan)),
            "debasement": flag_debasement(breakeven.get(q, np.nan)),
        }
        res = compute_composite_risk(fv, gap.get(q, np.nan))
        rows[q] = res
        flag_rows[q] = res.flag_values

    return RiskSeriesResult(
        risk=pd.Series({q: r.research_weighted for q, r in rows.items()}),
        risk_equal=pd.Series({q: r.equal_weighted for q, r in rows.items()}),
        divergence=pd.Series({q: r.divergence for q, r in rows.items()}),
        regime=pd.Series({q: r.regime for q, r in rows.items()}),
        flags=pd.DataFrame.from_dict(flag_rows, orient="index").reindex(index=list(rows.keys()), columns=FLAG_NAMES),
        n_flags=pd.Series({q: r.n_flags_present for q, r in rows.items()}),
    )


def composite_spec_v1(risk_result: RiskSeriesResult) -> pd.Series:
    """
    The research-weighted composite under SPECIFICATION v1 (credit-spread
    level included, original weights), rebuilt from the stored flags. Used
    only to report the before/after of the v2 change -- never as the headline.
    """
    flags = risk_result.flags
    ctx = risk_result.context_flags
    gap = risk_result.gap_used
    out = {}
    for q in flags.index:
        fv = {k: v for k, v in flags.loc[q].items() if pd.notna(v)}
        if q in ctx.index and "credit_spread" in ctx.columns and pd.notna(ctx.loc[q, "credit_spread"]):
            fv["credit_spread"] = float(ctx.loc[q, "credit_spread"])
        out[q] = compute_composite_risk(fv, gap.get(q, np.nan) if len(gap) else np.nan,
                                        weights=SPEC_V1_WEIGHTS).research_weighted
    return pd.Series(out)
