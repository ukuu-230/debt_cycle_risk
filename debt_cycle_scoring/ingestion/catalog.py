"""
FRED metric catalog -- maps the metrics from design doc section 2 to actual
FRED series IDs, plus the metadata normalize.py/aggregate.py need: which
category each metric belongs to, its native frequency (for resampling to
the quarterly cadence the scoring engine and backtest use), and its
direction (does a HIGHER raw reading mean more late-cycle/stressed, or the
opposite -- see normalize.direction_encode).

A few metrics are marked category="reference": these aren't scored
directly, they're raw building blocks (e.g. nominal GDP) needed to build
a RATIO metric (e.g. Wilshire 5000 / GDP) in panel.py, rather than being
percentile-ranked on their own.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class MetricSpec:
    name: str                      # human-readable name, matches design doc section 2 tables
    series_id: str                 # FRED series ID
    category: str                  # one of the section-2 categories, or "reference"
    native_frequency: str          # "D", "W", "M", "Q", "A" as FRED reports it
    resample_method: str           # "last", "mean", "sum", "ffill" -- how to collapse to quarterly
    higher_is_later_cycle: Optional[bool]  # None for "reference" series that aren't scored directly
    short_history: bool = False    # True if this series has < ~15 years of history (see normalize.shrinkage_zscore)
    notes: str = ""


FRED_CATALOG = [
    # --- 2.1 Debt burden & growth -----------------------------------------
    MetricSpec(
        "Household debt/GDP", "HDTGPDUSQ163N", "debt_burden", "Q", "last",
        higher_is_later_cycle=True,
    ),
    MetricSpec(
        "Nonfinancial corporate debt (level)", "NCBDBIQ027S", "debt_burden", "Q", "last",
        higher_is_later_cycle=True,
    ),
    MetricSpec(
        "Federal debt/GDP", "GFDEGDQ188S", "debt_burden", "Q", "last",
        higher_is_later_cycle=True,
    ),
    MetricSpec(
        "Household debt service ratio", "TDSP", "debt_burden", "Q", "last",
        higher_is_later_cycle=True,
    ),
    MetricSpec(
        "Bank credit, all commercial banks", "TOTBKCR", "debt_burden", "W", "mean",
        higher_is_later_cycle=True,
        notes="Used to derive real/nominal credit GROWTH rate in panel.py, not scored as a raw level.",
    ),

    # --- Composite risk-flag inputs (design: scorer/risk_flags.py) --------
    # These feed the independent-flag composite risk score, NOT the stage
    # classifier's category scores. Several are also useful as category
    # metrics, but their primary role now is the risk composite.
    MetricSpec(
        "M2 velocity", "M2V", "cb_stance", "Q", "last",
        higher_is_later_cycle=False,
        notes=(
            "Velocity of M2 money stock. FALLING velocity signals monetary "
            "easing not reaching the real economy (Dalio's MP2->MP3 "
            "transition), so lower = more stressed -> higher_is_later_cycle=False. "
            "Primary use: the 'M2 velocity' risk flag in scorer/risk_flags.py."
        ),
    ),
    MetricSpec(
        "M2 money supply", "M2SL", "reference", "M", "last",
        higher_is_later_cycle=None,
        notes=(
            "M2 money stock level. Reference series -- used for growth-rate "
            "computation, not scored raw. Definition changed May 2020 (a real "
            "discontinuity to flag if used for a long-history growth series)."
        ),
    ),
    MetricSpec(
        "10-year Treasury yield (monthly, long history)", "GS10", "reference", "M", "mean",
        higher_is_later_cycle=None,
        notes=(
            "Monthly 10y constant-maturity yield from 1953. Paired with TB3MS to build the "
            "10y-3m term spread used by the yield-curve flag -- the spread used by "
            "Estrella-Mishkin and the NY Fed recession model, which the curve weight cites. "
            "Quarterly MEAN, matching the NY Fed's averaging. TB3MS is quoted on a discount "
            "basis, GS10 bond-equivalent; the gap is ~10-20bp at high rates, small vs the "
            "flag's 0/+2.5pp anchors."
        ),
    ),
    MetricSpec(
        "3-month Treasury bill (monthly, long history)", "TB3MS", "reference", "M", "mean",
        higher_is_later_cycle=None,
        notes="Secondary-market 3m bill rate from 1934. Short leg of the 10y-3m term spread.",
    ),
    MetricSpec(
        "10-year Treasury yield (level)", "DGS10", "reference", "D", "last",
        higher_is_later_cycle=None,
        notes=(
            "Long-bond yield LEVEL (the curve SPREAD is already covered by "
            "T10Y2Y/T10Y3M). Reference series feeding the 'long-bond level' "
            "risk flag in scorer/risk_flags.py rather than a category score."
        ),
    ),
    MetricSpec(
        "10-year breakeven inflation expectations", "T10YIE", "reference", "D", "last",
        higher_is_later_cycle=None,
        short_history=True,
        notes=(
            "Market-implied 10y inflation expectations. Only starts 2003 "
            "(short history). Feeds the TWO-TAILED 'monetary debasement / "
            "deflation' risk flag in scorer/risk_flags.py (replacing the "
            "former gold flag after FRED deleted all LBMA gold series in "
            "2022): risk rises for BOTH high expectations (debasement) and "
            "very low/negative ones (deflation), lowest near the ~2% target."
        ),
    ),

    # (Gold price removed: FRED deleted ALL LBMA gold price series on
    # 2022-01-31 when ICE Benchmark Administration data was pulled from
    # FRED. The 'monetary debasement' signal gold was proxying is now
    # captured by the two-tailed breakeven-inflation flag (T10YIE) in
    # scorer/risk_flags.py -- a cleaner debasement measure that isolates
    # expected fiat purchasing-power erosion without gold's real-rate and
    # safe-haven noise, and that won't vanish from FRED.)

    # --- Short-horizon stress inputs (scorer/short_horizon_flags.py) ------
    # Fast daily/weekly market-stress series feeding the SEPARATE short-
    # horizon sub-score -- NOT the slow structural composite, and NOT the
    # stage classifier. category="short_horizon" keeps them from ever
    # leaking into the structural scorer's category PCA (which only
    # consumes debt_burden/asset_prices/cb_stance/deleveraging_signals).
    MetricSpec(
        "VIX equity volatility", "VIXCLS", "short_horizon", "D", "mean",
        higher_is_later_cycle=None,
        notes="Daily since 1990. Raw component of the short-horizon sub-score (level).",
    ),
    MetricSpec(
        "Broad nominal US dollar index", "DTWEXBGS", "short_horizon", "D", "mean",
        higher_is_later_cycle=None,
        notes=(
            "Daily. Short-horizon sub-score, RATE-OF-CHANGE (a sharp dollar "
            "SPIKE is the risk-off/funding-stress signal, not the level). "
            "Caveat: for the US, dollar strength is ambiguous (safe-haven vs "
            "economic strength) -- cleaner risk signal for non-US economies."
        ),
    ),
    MetricSpec(
        "Chicago Fed NFCI (validation cross-check)", "NFCI", "short_horizon_validation", "W", "mean",
        higher_is_later_cycle=None,
        notes=(
            "Weekly back to 1971. A pro-built 105-series financial-conditions "
            "index. Used ONLY as an external cross-check against our own "
            "raw-component sub-score, NOT as an ingredient (it already "
            "contains VIX/spreads -- including it would double-count)."
        ),
    ),
    MetricSpec(
        "St Louis Fed Financial Stress Index (validation cross-check)", "STLFSI4", "short_horizon_validation", "W", "mean",
        higher_is_later_cycle=None,
        notes="Weekly since 1993. Second external cross-check, same rationale as NFCI.",
    ),

    # --- 2.2 Asset prices & leverage-financed speculation ------------------
    MetricSpec(
        "S&P 500", "SP500", "asset_prices", "D", "last",
        higher_is_later_cycle=True,
        short_history=True,
        notes=(
            "Originally specced as the Wilshire 5000 (broader market coverage), "
            "but Wilshire discontinued distributing its indexes to FRED "
            "following its 2021 rebrand to 'FT Wilshire' -- confirmed live "
            "against a real API call: both WILL5000PRFC and WILL5000INDFC "
            "return 'series does not exist' as of this catalog's last check. "
            "S&P 500 is FRED's standard replacement (per FRED's own public "
            "user datalists), but per FRED's licensing agreement with S&P "
            "Dow Jones Indices, SP500 carries only a ROLLING 10-YEAR window "
            "of daily history, not full history back to the 1960s-70s like "
            "the metric this was meant to replace -- hence short_history=True, "
            "which routes it through normalize.py's shrinkage_zscore path "
            "instead of a full-sample percentile rank."
        ),
    ),
    MetricSpec(
        "Household net worth", "TNWBSHNO", "asset_prices", "Q", "last",
        higher_is_later_cycle=True,
        notes="Divided by disposable personal income (DPI) in panel.py.",
    ),
    MetricSpec(
        "Case-Shiller US National Home Price Index", "CSUSHPISA", "asset_prices", "M", "last",
        higher_is_later_cycle=True,
    ),
    MetricSpec(
        "Moody's Baa corporate yield (monthly)", "BAA", "reference", "M", "mean",
        higher_is_later_cycle=None,
        notes=(
            "Seasoned Baa corporate bond yield, monthly from 1919. Baa minus GS10 is the "
            "composite's credit-spread input: long history (all 8 recessions), not ICE-"
            "licensed. Replaced HY OAS as primary after the live run showed the OAS flag "
            "with 0 usable quarters (FRED serves only recent ICE history)."
        ),
    ),
    MetricSpec(
        "Moody's Baa - 10y Treasury spread (daily)", "BAA10Y", "reference", "D", "mean",
        higher_is_later_cycle=None,
        notes="Daily Baa-10y spread (1986+). Fills the current quarter before the monthly BAA posts.",
    ),
    MetricSpec(
        "ICE BofA US High Yield OAS (credit spread)", "BAMLH0A0HYM2", "asset_prices", "D", "mean",
        higher_is_later_cycle=False,
        short_history=True,
        notes="Spread COMPRESSING is the late-bubble signal, hence higher_is_later_cycle=False.",
    ),

    # --- 2.3 Central bank stance & the short-term cycle ---------------------
    MetricSpec(
        "Effective federal funds rate", "FEDFUNDS", "cb_stance", "M", "mean",
        higher_is_later_cycle=True,
        notes="Combined with CPI YoY in panel.py to derive the REAL policy rate.",
    ),
    MetricSpec(
        "10y-2y Treasury spread", "T10Y2Y", "cb_stance", "D", "last",
        higher_is_later_cycle=False,
        notes="Inversion (negative) is the late-cycle/top signal, hence higher_is_later_cycle=False.",
    ),
    MetricSpec(
        "10y-3m Treasury spread", "T10Y3M", "cb_stance", "D", "last",
        higher_is_later_cycle=False,
    ),
    MetricSpec(
        "Fed total assets (balance sheet)", "WALCL", "cb_stance", "W", "last",
        higher_is_later_cycle=True,
        short_history=True,
        notes="QE intensity / MP2-MP3 classification input; divided by GDP in panel.py.",
    ),
    MetricSpec(
        "CPI, all urban consumers", "CPIAUCSL", "reference", "M", "last",
        higher_is_later_cycle=None,
        notes="Reference series: used to compute YoY inflation and the real policy rate, not scored raw.",
    ),
    MetricSpec(
        "Core PCE price index", "PCEPILFE", "reference", "M", "last",
        higher_is_later_cycle=None,
        notes="Reference series, same role as CPIAUCSL but the Fed's preferred inflation gauge.",
    ),
    MetricSpec(
        "Civilian unemployment rate", "UNRATE", "cb_stance", "M", "mean",
        higher_is_later_cycle=False,
        notes="LOW unemployment (below the natural rate) is the overheating/late-cycle signal.",
    ),
    MetricSpec(
        "Disposable personal income", "DSPI", "reference", "M", "last", higher_is_later_cycle=None,
        notes="Reference series -- denominator for the net-worth/income ratio.",
    ),
    MetricSpec(
        "Median household income (nominal)", "MEHOINUSA646N", "reference", "A", "ffill",
        higher_is_later_cycle=None,
        notes=(
            "US Census Bureau, nominal (not inflation-adjusted) dollars, "
            "annual, back to 1984. Deliberately the NOMINAL series "
            "(MEHOINUSA646N), not the more commonly cited real/inflation-"
            "adjusted MEHOINUSA672N -- CSUSHPISA (Case-Shiller) is itself a "
            "nominal price index, so pairing it with a real income series "
            "would introduce a spurious inflation-adjustment mismatch into "
            "the home-price/income ratio over time. Forward-filled to "
            "quarterly like the other annual series."
        ),
    ),
    MetricSpec(
        "Real GDP", "GDPC1", "reference", "Q", "last", higher_is_later_cycle=None,
        notes="Reference series for output gap and real-vs-nominal growth comparisons.",
    ),
    MetricSpec(
        "Nominal GDP", "GDP", "reference", "Q", "last", higher_is_later_cycle=None,
        notes="Reference series -- denominator for most of the debt/GDP and asset-price/GDP ratios.",
    ),

    # --- 2.4 Deleveraging-in-progress signals -------------------------------
    MetricSpec(
        "Real broad effective exchange rate (US)", "RBUSBIS", "deleveraging_signals", "M", "last",
        higher_is_later_cycle=False,
        short_history=True,
        notes="A sharp DECLINE signals the money-printing/currency-depreciation lever is dominant.",
    ),
    MetricSpec(
        "Federal surplus/deficit as % of GDP", "FYFSGDA188S", "deleveraging_signals", "A", "ffill",
        higher_is_later_cycle=False,
        notes="Annual frequency -- forward-filled to quarterly. More negative (bigger deficit) = more stimulus lever in use.",
    ),

    # --- BIS credit-to-GDP gap input (design doc section 3B, Source B) -----
    MetricSpec(
        "US Credit-to-GDP ratio (BIS, via FRED)", "QUSPAM770A", "label_source", "Q", "last",
        higher_is_later_cycle=None,
        notes=(
            "BIS 'Total Credit to Private Non-Financial Sector, Adjusted for "
            "Breaks, for United States', as % of GDP. Feeds the section 3B "
            "credit-to-GDP GAP computation (one-sided HP filter, section "
            "3B in labeling/credit_gap.py) -- not scored directly as a "
            "category metric itself."
        ),
    ),

    # --- NBER short-term cycle labels (design doc section 3B, Source A) ----
    MetricSpec(
        "NBER recession indicator", "USREC", "label_source", "M", "max",
        higher_is_later_cycle=None,
        notes=(
            "Not a scored metric -- feeds the section 3B independent labeling pass and the "
            "recession backtest. Collapsed with MAX: a quarter counts as recession if ANY "
            "month in it was (e.g. 2020Q2, recession through April). 'last' would mark only "
            "the final month and drop such quarters -- fixed after the statistics audit."
        ),
    ),
]
