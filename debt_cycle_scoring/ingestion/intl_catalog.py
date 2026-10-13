"""
International (multi-country) flag catalog -- ingestion/intl_catalog.py

Maps each composite-risk FLAG to its FRED series ID per country, so the
risk score can be computed for economies beyond the US. Kept SEPARATE from
the main US catalog.py so the existing US path is untouched (no regression
risk): the US catalog still drives the full US pipeline (labeling, stages,
short-horizon); this module only feeds the multi-country COMPOSITE RISK
score via build_country_risk_series in scorer/risk_flags.py.

Design decisions (from the multi-country expansion discussion):
  - Eurozone first (strongest non-US FRED coverage), then UK + Japan.
  - Flags DROP per country when unavailable; the composite renormalizes
    over present flags and reports the flag count. No fabricated proxies.
  - China deliberately excluded (smoothed/unreliable official data,
    state-directed non-market credit cycle) -- can be added later, heavily
    caveated, if ever.
  - Every series ID here was verified to EXIST on FRED before being added
    (learned the hard way from the Wilshire and gold removals). Where a
    flag has no clean FRED series for a country, it is set to None -- the
    flag simply won't contribute for that country, honestly.

Series notes:
  - credit_gap / debt_service_ratio: BIS series on FRED, near-identical
    methodology across countries (the two most research-validated flags,
    present for all four economies). credit_gap here is the RATIO series
    (…PAM770A); the gap itself is computed by labeling/credit_gap.py's
    one-sided HP filter, same as the US.
  - long_bond_level / yield_curve: OECD 10-year yields on FRED
    (IRLTLT01<cc>M156N), MONTHLY. The curve needs a short rate too; where
    no clean short rate exists, yield_curve is None for now.
  - credit_spread: only US and Euro area have a clean HY OAS on FRED.
  - m2_velocity / debasement: US-specific for now (others need derivation
    or lack clean series); None elsewhere -> flags drop.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class CountryFlagSeries:
    country: str
    # flag_name -> FRED series ID, or None if that flag is unavailable here.
    credit_gap_ratio: Optional[str]      # BIS credit-to-GDP RATIO (gap computed downstream)
    debt_service_ratio: Optional[str]
    long_bond_10y: Optional[str]
    short_rate: Optional[str]            # for the yield-curve flag (10y - short); None -> curve drops
    credit_spread: Optional[str]
    m2_velocity: Optional[str]
    breakeven_inflation: Optional[str]


# All IDs below were verified to exist on FRED (see module docstring).
INTL_CATALOG: Dict[str, CountryFlagSeries] = {
    "US": CountryFlagSeries(
        country="US",
        credit_gap_ratio="QUSPAM770A",
        debt_service_ratio="TDSP",              # US household DSR (already in US catalog)
        long_bond_10y="DGS10",
        short_rate="DGS2",                      # 2y for the 10y-2y curve
        credit_spread="BAMLH0A0HYM2",
        m2_velocity="M2V",
        breakeven_inflation="T10YIE",
    ),
    "Euro": CountryFlagSeries(
        country="Euro",
        credit_gap_ratio="QXMPAM770A",          # VERIFIED live: euro area BIS credit-to-GDP, 1999Q1-2025Q4
        debt_service_ratio=None,                # FRED has NO non-US DSR (QXMDSR 404s). BIS publishes it
                                                #   only on its own portal, never imported to FRED. Honest None.
        long_bond_10y="IRLTLT01EZM156N",        # VERIFIED: OECD euro area 10y, monthly
        short_rate=None,                        # no clean single euro-area short rate on FRED -> curve drops
        credit_spread="BAMLHE00EHYIOAS",        # ICE BofA Euro HY OAS
        m2_velocity=None,                       # derive later; drop for now
        breakeven_inflation=None,               # weak coverage; drop for now
    ),
    # UK and Japan are staged but commented until their IDs are individually
    # verified live (Eurozone-first per the plan). Uncomment after checking
    # each ID against FRED, exactly as the US/Euro ones were.
    # "UK": CountryFlagSeries(
    #     country="UK",
    #     credit_gap_ratio="QGBPAM770A",        # verified pattern-match; confirm live
    #     debt_service_ratio="QGBDSR",
    #     long_bond_10y="IRLTLT01GBM156N",
    #     short_rate=None,
    #     credit_spread=None,
    #     m2_velocity=None,
    #     breakeven_inflation=None,
    # ),
    # "Japan": CountryFlagSeries(
    #     country="Japan",
    #     credit_gap_ratio="QJPPAM770A",
    #     debt_service_ratio="QJPDSR",
    #     long_bond_10y="IRLTLT01JPM156N",
    #     short_rate=None,
    #     credit_spread=None,
    #     m2_velocity=None,
    #     breakeven_inflation=None,
    # ),
}


def country_series_ids(country: str) -> Dict[str, Optional[str]]:
    """All FRED series IDs needed for `country`'s risk flags, flag -> id|None."""
    c = INTL_CATALOG[country]
    return {
        "credit_gap_ratio": c.credit_gap_ratio,
        "debt_service_ratio": c.debt_service_ratio,
        "long_bond_10y": c.long_bond_10y,
        "short_rate": c.short_rate,
        "credit_spread": c.credit_spread,
        "m2_velocity": c.m2_velocity,
        "breakeven_inflation": c.breakeven_inflation,
    }


def available_countries() -> list:
    return list(INTL_CATALOG.keys())
