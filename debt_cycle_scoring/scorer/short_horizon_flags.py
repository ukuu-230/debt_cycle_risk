"""
Short-horizon market-stress sub-score -- scorer/short_horizon_flags.py

A SEPARATE, fast-moving stress gauge shown ALONGSIDE (never blended into)
the slow structural composite in risk_flags.py. The structural composite
measures years-long credit-cycle buildup (and correctly flagged 2006-07);
this measures acute market stress happening now. Two numbers, two
meanings -- kept apart on purpose.

Quant-finance choices, made explicit so they can be checked:

1. NO LOOK-AHEAD: every normalization uses only trailing data (expanding
   window), same discipline as the structural side.

2. ROBUST normalization (median + MAD, not mean + std): daily market
   series have fat tails; a single 2008-style spike would wreck a
   mean/std scale and distort every other quarter's reading. Median and
   MAD (median absolute deviation) are outlier-robust. The robust
   z-score is z = (x - median) / (1.4826 * MAD)  [the 1.4826 makes MAD a
   consistent estimator of std for normal data], then mapped to 0-100 via
   a logistic squash so extreme spikes saturate near 100 instead of
   running away.

3. LEVEL vs CHANGE separation: VIX and the dollar enter partly as levels
   and partly as rate-of-change, because acute stress often shows in the
   MOVE (a fast VIX spike, a sharp dollar surge) more than the level.
   Spread WIDENING (change in HY OAS) and curve CHANGE are pure
   change-based flags. Mixing level and change blindly is a common error;
   here each is chosen deliberately per flag.

4. BUILD FROM RAW COMPONENTS, validate against pro composites: NFCI and
   STLFSI are themselves 100+-series composites that already contain VIX
   and spreads. Blending them in with VIX/spreads would double- or
   triple-count volatility. So the sub-score is built ONLY from raw
   components (VIX, dollar, spread-change, curve-change), and NFCI/STLFSI
   are reported as independent external CROSS-CHECKS -- if our sub-score
   and NFCI disagree sharply, that's a signal to investigate, not a
   number to average in.

5. Both quarter-AVERAGE (the score) and quarter-WORST (peak stress that
   quarter, shown as a flag) are surfaced, per the aggregation decision.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# Panel column names (exact, from the catalog).
_COL_VIX = "VIX equity volatility"
_COL_DOLLAR = "Broad nominal US dollar index"
_COL_SPREAD = "ICE BofA US High Yield OAS (credit spread)"   # from scored_metrics
_COL_CURVE = "10y-2y Treasury spread"                          # from scored_metrics
_COL_NFCI = "Chicago Fed NFCI (validation cross-check)"
_COL_STLFSI = "St Louis Fed Financial Stress Index (validation cross-check)"

SHORT_FLAG_NAMES: List[str] = [
    "vix_level", "vix_change", "dollar_surge", "spread_level", "spread_widening", "curve_change",
]


def _robust_z_expanding(series: pd.Series, min_periods: int = 12) -> pd.Series:
    """
    Backward-looking robust z-score: at each point, standardize against the
    median and MAD of ONLY the PRIOR history (expanding window, current
    point excluded so a spike cannot dampen its own z-score). MAD is
    scaled by 1.4826 to be a consistent std estimator under normality.
    Returns NaN until min_periods trailing points exist. No look-ahead.
    """
    vals = series.values
    n = len(vals)
    out = np.full(n, np.nan)
    for t in range(min_periods, n):
        if np.isnan(vals[t]):
            continue
        hist = vals[:t]                      # PRIOR points only: the current value is not
        hist = hist[~np.isnan(hist)]         # part of its own reference distribution
        if len(hist) < min_periods:
            continue
        med = np.median(hist)
        mad = np.median(np.abs(hist - med))
        scale = 1.4826 * mad if mad > 1e-9 else (np.std(hist) if np.std(hist) > 1e-9 else 1.0)
        out[t] = (vals[t] - med) / scale
    return pd.Series(out, index=series.index)


def _z_to_0_100(z: pd.Series) -> pd.Series:
    """
    Map a robust z-score to 0-100 via a logistic squash centered at z=0,
    so 'normal' (z=0) -> 50, elevated (z=+2) -> ~88, extreme (z>=+3)
    saturates near 100, and calm (z<=-2) -> low. The logistic keeps fat-
    tail spikes from running off-scale while preserving ordering.
    Only the UPSIDE is risk for these flags (high VIX, wide spreads, etc.),
    so negative z (calm) correctly maps below 50.
    """
    return 100.0 / (1.0 + np.exp(-1.0 * z))


@dataclass
class ShortHorizonResult:
    stress: pd.Series                    # 0-100 short-horizon stress sub-score (quarter-average based)
    stress_worst: pd.Series              # 0-100 based on quarter-WORST inputs (peak stress that quarter)
    flags: pd.DataFrame                  # per-quarter 0-100 value of each raw-component flag
    n_flags: pd.Series                   # flags present each quarter
    validation: pd.DataFrame = field(default_factory=pd.DataFrame)  # NFCI/STLFSI as 0-100, external cross-checks
    validation_divergence: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))


def _compute_flags(vix, dollar, spread, curve, min_periods) -> pd.DataFrame:
    """Build the five raw-component flags (each 0-100) from quarterly inputs."""
    idx = vix.index if vix is not None else (dollar.index if dollar is not None else None)

    flags = pd.DataFrame(index=idx)

    if vix is not None:
        flags["vix_level"] = _z_to_0_100(_robust_z_expanding(vix, min_periods))
        flags["vix_change"] = _z_to_0_100(_robust_z_expanding(vix.diff(), min_periods))
    if dollar is not None:
        # dollar SURGE: YoY % change, only the upside is risk
        dollar_yoy = dollar.pct_change(4, fill_method=None) * 100.0
        flags["dollar_surge"] = _z_to_0_100(_robust_z_expanding(dollar_yoy, min_periods))
    if spread is not None:
        # spread LEVEL: wide credit spreads = acute stress that has ARRIVED. Moved
        # here from the structural composite in spec v2 (spreads are coincident).
        flags["spread_level"] = _z_to_0_100(_robust_z_expanding(spread, min_periods))
        # spread WIDENING: quarter-over-quarter change in the credit spread
        flags["spread_widening"] = _z_to_0_100(_robust_z_expanding(spread.diff(), min_periods))
    if curve is not None:
        # curve CHANGE: a rapid drop (flattening/inverting) is the stress
        # signal, so risk rises as the change goes NEGATIVE -> negate first
        flags["curve_change"] = _z_to_0_100(_robust_z_expanding(-curve.diff(), min_periods))

    return flags.reindex(columns=[c for c in SHORT_FLAG_NAMES if c in flags.columns])


def build_short_horizon_score(
    short_horizon: pd.DataFrame,
    short_horizon_worst: pd.DataFrame,
    scored_metrics: pd.DataFrame,
    validation: Optional[pd.DataFrame] = None,
    min_periods: int = 12,
    reference_series: Optional[pd.DataFrame] = None,
) -> ShortHorizonResult:
    """
    short_horizon / short_horizon_worst: quarter-average / quarter-worst
        frames of the short-horizon series (VIX, dollar) from the panel.
    scored_metrics: source of the credit spread and curve (already there).
    validation: NFCI/STLFSI frame, used only as an external cross-check.

    Returns the 0-100 short-horizon stress sub-score (average- and worst-
    based), the per-flag breakdown, and the validation cross-check.
    """
    def get(df, col):
        return df[col] if (df is not None and col in df.columns) else None

    # Credit spread: the same long-history Moody's Baa - 10y series the
    # composite uses (falls back to HY OAS inside select_spread).
    spread_in = get(scored_metrics, _COL_SPREAD)
    if reference_series is not None and len(scored_metrics.index):
        from .risk_flags import select_spread
        sp, _src = select_spread(scored_metrics, reference_series, scored_metrics.index)
        if sp.notna().any():
            spread_in = sp

    # --- average-based score (the headline sub-score) ---
    flags_avg = _compute_flags(
        get(short_horizon, _COL_VIX), get(short_horizon, _COL_DOLLAR),
        spread_in, get(scored_metrics, _COL_CURVE),
        min_periods,
    )
    stress = flags_avg.mean(axis=1, skipna=True)   # equal-weight across present raw components
    n_flags = flags_avg.notna().sum(axis=1)

    # --- worst-based score (peak stress that quarter) ---
    flags_worst = _compute_flags(
        get(short_horizon_worst, _COL_VIX), get(short_horizon_worst, _COL_DOLLAR),
        spread_in, get(scored_metrics, _COL_CURVE),
        min_periods,
    )
    stress_worst = flags_worst.mean(axis=1, skipna=True)

    # --- external validation cross-checks (NOT ingredients) ---
    val_0_100 = pd.DataFrame(index=flags_avg.index)
    divergence = pd.Series(dtype=float, index=flags_avg.index)
    if validation is not None and not validation.empty:
        for col in validation.columns:
            val_0_100[col] = _z_to_0_100(_robust_z_expanding(validation[col], min_periods)).reindex(flags_avg.index)
        # divergence = |our score - mean of the pro composites|, where both exist
        pro_mean = val_0_100.mean(axis=1, skipna=True)
        divergence = (stress - pro_mean).abs()

    return ShortHorizonResult(
        stress=stress.round(1),
        stress_worst=stress_worst.round(1),
        flags=flags_avg.round(1),
        n_flags=n_flags,
        validation=val_0_100.round(1),
        validation_divergence=divergence.round(1),
    )
