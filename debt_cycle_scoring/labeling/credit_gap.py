"""
Source B -- BIS-style credit-to-GDP gap, design doc section 3B.2.

Applies the one-sided HP filter (lambda=400,000) to a credit-to-GDP
ratio series and flags quarters using Basel III's own published
reference thresholds (2 percentage points / 10 percentage points),
confirmed against BIS/BCBS/IMF documentation -- not thresholds invented
for this project.
"""
from __future__ import annotations

import pandas as pd

from .hp_filter import one_sided_hp_trend

BASEL_LOWER_THRESHOLD = 2.0   # percentage points -- buffer starts activating
BASEL_UPPER_THRESHOLD = 10.0  # percentage points -- maximum warning level


def compute_credit_gap(
    credit_to_gdp_ratio: pd.Series,
    lamb: float = 400_000.0,
    min_periods: int = 40,
    decline_threshold: float = 3.0,
    decline_lookback: int = 12,
) -> pd.DataFrame:
    """
    credit_to_gdp_ratio: quarterly series, e.g. FRED's QUSPAM770A (BIS's
    own "Total Credit to Private Non-Financial Sector... as % of GDP"
    for the US -- see ingestion/catalog.py).

    Returns a DataFrame with the ratio, its one-sided HP trend, the gap
    (ratio - trend, in percentage points), a `flag` column, and a
    `declining_from_peak` column:
      - "early": gap below the lower Basel III threshold
      - "bubble_building": gap between the lower and upper thresholds
      - "top_warning": gap at or above the upper threshold -- BIS's own
        "maximum buffer" line, historically associated with most major
        banking crises in BIS's cross-country evaluation, though BIS's
        own guidance is explicit that this should breach roughly 2-3
        YEARS before a crisis, not coincide with one (see design doc
        section 3B.2) -- treat this as a leading indicator, not a
        coincident "crisis is happening now" flag.
      - `declining_from_peak` (bool): True when the gap has fallen by at
        least `decline_threshold` percentage points from its trailing
        `decline_lookback`-quarter rolling max, AND that rolling max
        itself reached bubble-building territory (>= BASEL_LOWER_THRESHOLD)
        -- i.e. there was a real elevated peak to be declining FROM, not
        just noise wobbling near zero. This exists specifically to give
        troughs an independent credit-based vote in consensus.py, mirroring
        the role `gap_building` (rising into bubble/top territory) already
        plays for tops. Without this, trough detection in consensus.py
        relied entirely on NBER+mechanical agreement with no credit-side
        corroboration at all -- an asymmetry that was NOT the direct cause
        of the 2009Q2 real-data misclassification (that came from the
        one-sided filter's inherent lag keeping `flag` at "bubble_building"
        for a few quarters after the true peak, combined with a spurious
        mechanical peak), but is a related structural gap worth closing
        while addressing that finding.
    """
    trend = one_sided_hp_trend(credit_to_gdp_ratio, lamb=lamb, min_periods=min_periods)
    gap = credit_to_gdp_ratio - trend

    flag = pd.Series("early", index=gap.index, dtype=object)
    flag[gap >= BASEL_LOWER_THRESHOLD] = "bubble_building"
    flag[gap >= BASEL_UPPER_THRESHOLD] = "top_warning"
    flag[gap.isna()] = None  # don't silently label the min_periods burn-in window

    rolling_max = gap.rolling(window=decline_lookback, min_periods=1).max()
    declining_from_peak = (rolling_max >= BASEL_LOWER_THRESHOLD) & (
        (rolling_max - gap) >= decline_threshold
    )
    declining_from_peak = declining_from_peak.where(gap.notna(), False)

    return pd.DataFrame(
        {
            "ratio": credit_to_gdp_ratio,
            "trend": trend,
            "gap": gap,
            "flag": flag,
            "declining_from_peak": declining_from_peak,
        }
    )
