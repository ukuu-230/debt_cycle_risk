"""
Feature normalization -- design doc section 3A.1.

Two normalization modes, used situationally:
- `expanding_percentile_rank`: default, for metrics with long history.
  Makes no distributional assumption and only ever looks backward in
  time, which is what makes a walk-forward backtest honest.
- `shrinkage_zscore`: for metrics with short history, where a raw
  percentile would be too noisy (too few points to rank against).

Plus two supporting steps that have to happen before either of the above:
`winsorize` (clip outliers before ranking) and `direction_encode` (make
every metric's score mean the same thing: 1.0 = latest-cycle/most
stressed, 0.0 = earliest-cycle, regardless of the metric's raw sign).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def winsorize(series: pd.Series, lower_pct: float = 0.01, upper_pct: float = 0.99) -> pd.Series:
    """
    Clip a series to its own [lower_pct, upper_pct] quantiles, computed once
    over the full available history. This keeps a single crisis-quarter
    outlier (e.g. a 2020-style GDP print) from distorting the historical
    distribution used to rank every other quarter.

    Note: for a true walk-forward backtest this should be refit using only
    trailing history at each step, same as the percentile rank itself --
    this simple version is for exploratory/offline use.
    """
    lo, hi = series.quantile(lower_pct), series.quantile(upper_pct)
    return series.clip(lower=lo, upper=hi)


def expanding_percentile_rank(series: pd.Series, min_periods: int = 20) -> pd.Series:
    """
    Rank each observation against ONLY the history available up to and
    including that point (expanding window, empirical CDF). Returns a
    value in [0, 1]; NaN until `min_periods` trailing observations exist.

    This is the core no-look-ahead guarantee the whole backtest (section 7)
    depends on: a quarter's percentile never uses data from its own future.
    """

    def _pct_of_last(window: np.ndarray) -> float:
        *history, current = window
        history = np.asarray(history)
        if len(history) == 0:
            return np.nan
        return float((history <= current).mean())

    return series.expanding(min_periods=min_periods + 1).apply(_pct_of_last, raw=True)


@dataclass
class ShrinkageZScoreResult:
    z: pd.Series
    weight_on_own_history: pd.Series  # diagnostic: how much the shrinkage trusted own data vs. the prior


def shrinkage_zscore(
    series: pd.Series,
    prior_mean: float,
    prior_std: float,
    full_credibility_n: int = 40,
) -> ShrinkageZScoreResult:
    """
    For a metric with short history: compute a z-score against the metric's
    own (short, expanding) history, then shrink it toward a prior
    distribution (e.g. a pooled cross-country or cross-metric-family
    distribution supplied by the caller) using a simple empirical-Bayes
    credibility weight that rises with sample size.

    w = n / (n + full_credibility_n)
    shrunk_z = w * own_z + (1 - w) * prior_z

    At n = full_credibility_n, own data and the prior get equal weight;
    as n grows, w -> 1 and the metric's own history dominates.
    """
    n = series.expanding().count()
    own_mean = series.expanding().mean()
    own_std = series.expanding().std()
    own_z = (series - own_mean) / own_std.replace(0, np.nan)
    prior_z = (series - prior_mean) / prior_std

    w = n / (n + full_credibility_n)
    shrunk_z = w * own_z.fillna(prior_z) + (1 - w) * prior_z
    return ShrinkageZScoreResult(z=shrunk_z, weight_on_own_history=w)


def direction_encode(percentile: pd.Series, higher_is_later_cycle: bool) -> pd.Series:
    """
    Flip a [0, 1] percentile so that, after this step, every metric agrees
    on direction: 1.0 always means "most late-cycle / most stressed",
    0.0 always means "most early-cycle". Without this, combining metrics
    would silently cancel out metrics that move in opposite directions
    for the same underlying reason.

    Example: credit spreads COMPRESSING is a late-bubble signal, so that
    metric needs higher_is_later_cycle=False (a raw high spread reading
    means early-cycle/stressed-but-not-bubbly, not late-bubble).
    """
    return percentile if higher_is_later_cycle else 1.0 - percentile
