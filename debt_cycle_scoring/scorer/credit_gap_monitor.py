"""
Cross-country credit-gap monitor -- scorer/credit_gap_monitor.py

The quant-honest core of the multi-country work. Per-country COMPOSITES
proved mostly redundant (long-bond and spreads co-move globally and were
already in the US model; and non-US scores can't be recession-backtested).
The ONE genuinely non-redundant, country-specific signal is the BIS
credit-to-GDP GAP: credit cycles are national, and the cross-country
SPREAD of gaps -- who is building credit faster, and where the US sits
relative to peers -- is information a US-only model cannot see.

So this module does one thing well: compute each economy's credit-to-GDP
gap (same one-sided HP filter, lambda=400000, min_periods=40, no
look-ahead as the US labeling pipeline) and present them side by side
against Basel III's own thresholds (2pp watch / 10pp max-warning).

All four credit-ratio series verified live on FRED:
  US   QUSPAM770A (1952-)   Euro QXMPAM770A (1999-)
  UK   QGBPAM770A (1963-)   Japan QJPPAM770A (1964-)
These are BIS series with identical methodology, so they ARE
cross-country comparable -- unlike the composite's other flags.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..labeling.credit_gap import (
    compute_credit_gap, BASEL_LOWER_THRESHOLD, BASEL_UPPER_THRESHOLD,
)

# Country -> BIS credit-to-GDP RATIO series on FRED (all verified live).
CREDIT_RATIO_SERIES: Dict[str, str] = {
    "US": "QUSPAM770A",
    "Euro": "QXMPAM770A",
    "UK": "QGBPAM770A",
    "Japan": "QJPPAM770A",
}


def _basel_flag(gap: float) -> str:
    if gap is None or np.isnan(gap):
        return "n/a"
    if gap >= BASEL_UPPER_THRESHOLD:
        return "max-warning"     # >=10pp: BIS's crisis-associated level
    if gap >= BASEL_LOWER_THRESHOLD:
        return "watch"           # 2-10pp: buffer building
    if gap <= -BASEL_LOWER_THRESHOLD:
        return "below-trend"     # credit unusually low vs trend (post-deleveraging)
    return "neutral"


@dataclass
class CountryGap:
    country: str
    gap: pd.Series               # credit-to-GDP gap (pp), quarterly, NaN during HP burn-in
    ratio: pd.Series             # the raw credit-to-GDP ratio (pp of GDP)
    latest_gap: float
    latest_flag: str
    latest_quarter: Optional[pd.Period]
    n_quarters: int              # usable (post-burn-in) gap quarters
    confidence: str              # "high" / "moderate" / "low" -- how stable the HP trend is


# A one-sided HP trend needs a good deal of history to stabilize. With few
# quarters the trend is still "catching up" and the gap magnitude is soft
# (directional sign is fine, exact value less so). Thresholds are on the
# USABLE post-burn-in quarter count.
_CONF_HIGH = 120     # ~30 years of gap history -> trend well-established
_CONF_MODERATE = 60  # ~15 years -> usable but treat magnitude with some caution


def _gap_confidence(n_quarters: int) -> str:
    if n_quarters >= _CONF_HIGH:
        return "high"
    if n_quarters >= _CONF_MODERATE:
        return "moderate"
    return "low"


@dataclass
class CreditGapMonitorResult:
    countries: Dict[str, CountryGap] = field(default_factory=dict)
    # wide frame: index=quarter, columns=country, values=gap -- for the
    # cross-country comparison and any charting.
    gap_panel: pd.DataFrame = field(default_factory=pd.DataFrame)

    def ranking(self) -> List[tuple]:
        """Countries ranked by latest gap, highest (most credit build-up) first.
        Each row: (country, latest_gap, latest_flag, confidence)."""
        rows = [(c.country, c.latest_gap, c.latest_flag, c.confidence)
                for c in self.countries.values() if not np.isnan(c.latest_gap)]
        return sorted(rows, key=lambda r: -r[1])


def build_credit_gap_monitor(
    ratios: Dict[str, pd.Series],
    lamb: float = 400_000.0,
    min_periods: int = 40,
) -> CreditGapMonitorResult:
    """
    ratios: {country: quarterly credit-to-GDP ratio Series}. Each is run
        through the same one-sided HP filter the US pipeline uses, so the
        gaps are computed identically and are genuinely comparable.
    Returns per-country gaps + a wide cross-country gap panel. Each country
    carries a `confidence` tier reflecting how much post-burn-in history
    its HP trend rests on -- a short series (e.g. euro area, ~69q) gives a
    directionally-valid but magnitude-soft gap and is flagged "low".
    """
    countries: Dict[str, CountryGap] = {}
    gap_cols = {}
    for country, ratio in ratios.items():
        r = ratio.dropna()
        if len(r) < min_periods + 1:
            continue
        gap = compute_credit_gap(r, lamb=lamb, min_periods=min_periods)["gap"]
        valid = gap.dropna()
        latest_q = valid.index.max() if len(valid) else None
        latest_gap = float(valid.loc[latest_q]) if latest_q is not None else np.nan
        n_usable = int(valid.shape[0])
        countries[country] = CountryGap(
            country=country, gap=gap, ratio=r,
            latest_gap=round(latest_gap, 1) if not np.isnan(latest_gap) else np.nan,
            latest_flag=_basel_flag(latest_gap),
            latest_quarter=latest_q, n_quarters=n_usable,
            confidence=_gap_confidence(n_usable),
        )
        gap_cols[country] = gap

    gap_panel = pd.DataFrame(gap_cols).sort_index() if gap_cols else pd.DataFrame()
    return CreditGapMonitorResult(countries=countries, gap_panel=gap_panel)
