"""
Hodrick-Prescott filter, including the ONE-SIDED (backward-looking-only)
version -- design doc section 3B.2, matching the BIS's own published
methodology for the credit-to-GDP gap (one-sided HP filter, lambda =
400,000 on quarterly data, confirmed against BIS's own published
documentation while researching the design doc).

The two-sided filter (`hp_filter_two_sided`) is the textbook Hodrick-
Prescott (1997) trend extraction: it uses the WHOLE sample, including
future observations relative to any given point, so it's only valid for
retrospective/offline analysis. `one_sided_hp_trend` wraps it to produce
a real-time-safe trend, which is what the credit-gap computation and the
Phase 1 walk-forward backtest (design doc section 7) both actually need.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve


def hp_filter_two_sided(series: pd.Series, lamb: float = 1600.0) -> pd.Series:
    """
    Standard two-sided HP trend: solves (I + lamb * D'D) trend = y, where
    D is the second-difference operator, via a sparse linear solve.
    lamb=1600 is the traditional quarterly-data default from Hodrick &
    Prescott (1997); the BIS credit-gap methodology instead uses
    lamb=400,000 (see `credit_gap.py`), deliberately much smoother, since
    it's meant to track a multi-decade credit cycle, not ordinary
    business-cycle fluctuations.
    """
    y = np.asarray(series.values, dtype=float)
    n = len(y)
    if n < 3:
        return pd.Series(y, index=series.index)

    # Second-difference operator D: (n-2) x n
    D = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n - 2, n))
    I = sparse.identity(n)
    A = (I + lamb * (D.T @ D)).tocsc()
    trend = spsolve(A, y)
    return pd.Series(trend, index=series.index)


def one_sided_hp_trend(series: pd.Series, lamb: float = 400_000.0, min_periods: int = 40) -> pd.Series:
    """
    Real-time / one-sided HP trend: for each quarter t, refit the
    two-sided HP filter using ONLY the data available up to and including
    t, and keep only the trend value AT t. This is what makes the
    resulting credit-gap number something the app could actually have
    computed in real time historically -- no observation ever uses data
    from its own future, matching the no-look-ahead requirement from
    section 3A.1.

    This is O(n^2) (refits the whole filter at every step), which is
    fine for a few hundred quarters of quarterly macro data and NOT fine
    for anything higher-frequency -- don't reuse this for daily series.

    Returns NaN for the first `min_periods` quarters, since a HP trend
    fit on very little data is unstable and shouldn't be trusted (BIS's
    own guidance recommends at least ~10 years / 40 quarters before
    relying on the credit gap -- see the design doc's section 3B notes).
    """
    if series.isna().any():
        # The HP solve has no notion of a missing observation: one NaN inside
        # the window turns every later trend value into NaN, silently. Callers
        # must pass a gap-free series (leading/trailing NaN should be dropped).
        inner = series.loc[series.first_valid_index():series.last_valid_index()] if series.notna().any() else series
        if inner.isna().any():
            raise ValueError("one_sided_hp_trend: series has interior missing values; "
                             "interpolate or drop them explicitly before filtering")
        series = series.dropna()
    values = series.values
    n = len(values)
    trend_at_t = np.full(n, np.nan)

    for t in range(min_periods - 1, n):
        window = pd.Series(values[: t + 1])
        trend_series = hp_filter_two_sided(window, lamb=lamb)
        trend_at_t[t] = trend_series.iloc[-1]

    return pd.Series(trend_at_t, index=series.index)
