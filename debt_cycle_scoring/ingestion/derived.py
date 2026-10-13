"""
Derived ("ratio") metrics -- the layer explicitly called out as missing
in the ingestion README: turns raw ingested series into the actual ratio
metrics the design doc's section 2 describes (Wilshire/GDP, real policy
rate, etc.), which are not raw FRED series themselves.

Kept as its own module, separate from both `panel.py` (raw ingestion)
and the scoring engine (`normalize.py`/`aggregate.py`), so each of the
three stays independently testable: panel.py doesn't need to know what
ratios will be built from it, and the scoring engine doesn't need to
know where its input metrics came from.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .panel import QuarterlyPanel


@dataclass(frozen=True)
class DerivedMetricSpec:
    name: str
    category: str
    higher_is_later_cycle: bool
    formula: str    # human-readable, shown in the methodology page per design doc section 3
    caveat: str = ""  # honest methodological caveats -- see notes below on WALCL/GDP and Wilshire/GDP


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """
    Aligns two series on their shared index and divides, turning a
    zero or NaN denominator into NaN rather than +/-inf -- an ordinary
    division-by-zero silently producing inf would otherwise poison every
    downstream percentile rank once it shows up in normalize.py.
    """
    aligned_num, aligned_den = numerator.align(denominator, join="inner")
    ratio = aligned_num / aligned_den.replace(0, np.nan)
    return ratio.replace([np.inf, -np.inf], np.nan)


def compute_derived_metrics(panel: QuarterlyPanel) -> Tuple[pd.DataFrame, Dict[str, DerivedMetricSpec]]:
    """
    Computes every ratio metric described in design doc section 2 that
    isn't itself a raw FRED series, using whatever raw/reference series
    `panel` already has. Silently skips a ratio (with nothing added) if
    one of its required inputs isn't present in the panel, rather than
    raising -- lets this run against a partial catalog during development.

    Returns (derived_df, specs) where derived_df has one column per
    successfully computed ratio and specs maps each name to a
    DerivedMetricSpec (category + direction, matching MetricSpec's role
    in catalog.py so normalize.py/aggregate.py can consume either kind
    of metric identically).
    """
    scored = panel.scored_metrics
    ref = panel.reference_series

    derived: Dict[str, pd.Series] = {}
    specs: Dict[str, DerivedMetricSpec] = {}

    def add(name: str, series: pd.Series, spec: DerivedMetricSpec) -> None:
        derived[name] = series
        specs[name] = spec

    # --- S&P 500 / nominal GDP ("Buffett indicator" proxy) -----------------
    if "S&P 500" in scored.columns and "Nominal GDP" in ref.columns:
        add(
            "S&P 500 / GDP (Buffett indicator proxy)",
            _safe_ratio(scored["S&P 500"], ref["Nominal GDP"]),
            DerivedMetricSpec(
                "S&P 500 / GDP (Buffett indicator proxy)", "asset_prices", higher_is_later_cycle=True,
                formula="S&P 500 / Nominal GDP",
                caveat=(
                    "Originally specced as Wilshire 5000/GDP for broader market "
                    "coverage; swapped to S&P 500 after confirming (via a real "
                    "FRED API call) that Wilshire discontinued distributing its "
                    "indexes to FRED following its 2021 'FT Wilshire' rebrand. "
                    "Two separate caveats stack here: (1) neither index is total "
                    "market cap in dollars, so this ratio is only meaningful as "
                    "a RELATIVE historical percentile, same as the original "
                    "Wilshire version; (2) FRED's SP500 series carries only a "
                    "rolling 10-year window per its licensing terms, so the "
                    "'historical percentile' this feeds into is only ever "
                    "against the last 10 years, not the multi-decade history "
                    "the design doc's asset-price category otherwise assumes."
                ),
            ),
        )

    # --- Household net worth / disposable personal income ------------------
    if "Household net worth" in scored.columns and "Disposable personal income" in ref.columns:
        add(
            "Household net worth / DPI",
            _safe_ratio(scored["Household net worth"], ref["Disposable personal income"]),
            DerivedMetricSpec(
                "Household net worth / DPI", "asset_prices", higher_is_later_cycle=True,
                formula="Household net worth / Disposable personal income",
            ),
        )

    # --- Real federal funds rate --------------------------------------------
    if "Effective federal funds rate" in scored.columns and "CPI, all urban consumers" in ref.columns:
        cpi_yoy = ref["CPI, all urban consumers"].pct_change(4) * 100  # 4 quarters = 1 year
        real_rate = (scored["Effective federal funds rate"] - cpi_yoy).dropna()
        add(
            "Real federal funds rate",
            real_rate,
            DerivedMetricSpec(
                "Real federal funds rate", "cb_stance", higher_is_later_cycle=True,
                formula="Effective federal funds rate - YoY CPI inflation",
                caveat=(
                    "Treating a HIGHER real rate as more late-cycle is a "
                    "simplification (it captures 'policy has tightened a lot,' "
                    "which is a reasonable top signal) -- it doesn't capture "
                    "the opposite late-cycle pattern where a very NEGATIVE real "
                    "rate signals cheap-money bubble conditions before a top. "
                    "This metric alone can't distinguish those two regimes; "
                    "the yield-curve and credit-spread metrics are meant to "
                    "help cover that gap when combined via PCA."
                ),
            ),
        )

    # --- Fed balance sheet / nominal GDP ------------------------------------
    if "Fed total assets (balance sheet)" in scored.columns and "Nominal GDP" in ref.columns:
        add(
            "Fed balance sheet / GDP",
            _safe_ratio(scored["Fed total assets (balance sheet)"], ref["Nominal GDP"]),
            DerivedMetricSpec(
                "Fed balance sheet / GDP", "cb_stance", higher_is_later_cycle=True,
                formula="Fed total assets / Nominal GDP",
                caveat=(
                    "Direction is genuinely context-dependent, not a clean "
                    "'higher = later cycle' relationship: a large balance sheet "
                    "mostly reflects being in or just past a QE-driven crisis "
                    "response (2009-2014, 2020-2021), which is a specific "
                    "MP2/MP3 regime rather than a uniform 'top' signal at every "
                    "level. Treated as higher_is_later_cycle=True here as a "
                    "rough default; worth revisiting once the section 7 "
                    "backtest shows how this metric actually behaves historically."
                ),
            ),
        )

    # --- Home price / income ------------------------------------------------
    if "Case-Shiller US National Home Price Index" in scored.columns and "Median household income (nominal)" in ref.columns:
        add(
            "Home price / income",
            _safe_ratio(scored["Case-Shiller US National Home Price Index"], ref["Median household income (nominal)"]),
            DerivedMetricSpec(
                "Home price / income", "asset_prices", higher_is_later_cycle=True,
                formula="Case-Shiller US National Home Price Index / Median household income (nominal, MEHOINUSA646N)",
                caveat=(
                    "Uses the NOMINAL median household income series "
                    "deliberately, matched to CSUSHPISA's own nominal terms -- "
                    "using the more commonly-cited real/inflation-adjusted "
                    "median income series (MEHOINUSA672N) instead would "
                    "introduce a spurious inflation-adjustment mismatch into "
                    "this ratio's trend over time. Income data is annual "
                    "(forward-filled to quarterly), so within-year moves in "
                    "this ratio are driven entirely by the home-price side."
                ),
            ),
        )

    return pd.DataFrame(derived), specs


def merge_derived_metrics(panel: QuarterlyPanel) -> QuarterlyPanel:
    """
    Computes every derived ratio and returns a NEW QuarterlyPanel with
    them appended to `scored_metrics` and `metric_specs` -- ready to feed
    straight into normalize.py exactly like any raw FRED-sourced metric,
    since DerivedMetricSpec exposes the same category/higher_is_later_cycle
    fields as MetricSpec.
    """
    derived_df, derived_specs = compute_derived_metrics(panel)

    combined_scored = pd.concat([panel.scored_metrics, derived_df], axis=1)
    combined_specs = dict(panel.metric_specs)
    combined_specs.update(derived_specs)

    return QuarterlyPanel(
        scored_metrics=combined_scored,
        reference_series=panel.reference_series,
        label_sources=panel.label_sources,
        short_horizon=panel.short_horizon,
        short_horizon_worst=panel.short_horizon_worst,
        short_horizon_validation=panel.short_horizon_validation,
        metric_specs=combined_specs,
    )
