"""
Assemble a single quarterly panel from the FRED catalog -- the ingestion
output that normalize.py/aggregate.py (the scoring engine) consume.

Kept deliberately dumb: this resamples and aligns raw series onto a
common quarterly PeriodIndex. It does NOT compute ratios (Wilshire/GDP,
real policy rate, etc.) or run any of the normalize.py/aggregate.py
scoring steps -- those stay in the scoring engine, so the ingestion layer
can be tested and changed independently of the scoring math, per the
architecture in the design doc's section 5.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from .catalog import FRED_CATALOG, MetricSpec
from .fred_client import FredClient


def resample_to_quarterly(series: pd.Series, method: str) -> pd.Series:
    """
    Collapse a series at its native frequency down to one value per
    calendar quarter, indexed by a pandas PeriodIndex(freq="Q") to match
    the cadence used throughout the scoring engine and the design doc's
    Phase 1 backtest plan (section 7).
    """
    quarterly = series.copy()
    quarterly.index = quarterly.index.to_period("Q")
    grouped = quarterly.groupby(level=0)

    if method == "last":
        return grouped.last()
    if method == "mean":
        return grouped.mean()
    if method == "sum":
        return grouped.sum()
    if method == "max":
        return grouped.max()
    if method == "min":
        return grouped.min()
    if method == "ffill":
        # For annual/sparse series: take the last value observed within
        # the quarter if there is one, else this quarter is left NaN here
        # and should be forward-filled ACROSS quarters by the caller
        # (kept as a separate step so it's obvious in the panel where data
        # is genuinely observed vs. carried forward).
        return grouped.last()
    raise ValueError(f"Unknown resample_method: {method}")


@dataclass
class QuarterlyPanel:
    scored_metrics: pd.DataFrame           # columns = metric names with a defined direction, ready for normalize.py
    reference_series: pd.DataFrame          # category == "reference": raw series used to build ratios
    label_sources: pd.DataFrame             # category == "label_source": e.g. NBER USREC, feeds section 3B
    short_horizon: pd.DataFrame = field(default_factory=pd.DataFrame)   # quarter-AVERAGE of short-horizon stress series
    short_horizon_worst: pd.DataFrame = field(default_factory=pd.DataFrame)  # quarter-WORST (max) of the same
    short_horizon_validation: pd.DataFrame = field(default_factory=pd.DataFrame)  # NFCI/STLFSI cross-checks (quarter-avg)
    metric_specs: Dict[str, MetricSpec] = field(default_factory=dict)  # name -> spec, for direction/category lookups


def build_quarterly_panel(
    client: FredClient,
    catalog: Optional[List[MetricSpec]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    force_refresh: bool = False,
) -> QuarterlyPanel:
    catalog = catalog if catalog is not None else FRED_CATALOG

    series_ids = [spec.series_id for spec in catalog]
    fetched = client.fetch_many(
        series_ids, start_date=start_date, end_date=end_date, force_refresh=force_refresh
    )

    scored_cols: Dict[str, pd.Series] = {}
    reference_cols: Dict[str, pd.Series] = {}
    label_cols: Dict[str, pd.Series] = {}
    short_cols: Dict[str, pd.Series] = {}
    short_worst_cols: Dict[str, pd.Series] = {}
    short_val_cols: Dict[str, pd.Series] = {}
    specs_by_name: Dict[str, MetricSpec] = {}

    for spec in catalog:
        if spec.series_id not in fetched:
            # series was skipped by fetch_many (unavailable/discontinued) --
            # omit it; downstream code tolerates missing columns.
            continue
        raw = fetched[spec.series_id].series
        if spec.native_frequency == "A" and spec.resample_method == "ffill":
            quarterly = resample_to_quarterly(raw, "last")
            # Now forward-fill ACROSS quarters, since an annual series only
            # has one true observation per year -- the other 3 quarters
            # should carry the last known annual value forward, not be NaN.
            full_index = pd.period_range(quarterly.index.min(), quarterly.index.max(), freq="Q")
            quarterly = quarterly.reindex(full_index).ffill()
        else:
            quarterly = resample_to_quarterly(raw, spec.resample_method)

        specs_by_name[spec.name] = spec
        if spec.category == "reference":
            reference_cols[spec.name] = quarterly
        elif spec.category == "label_source":
            label_cols[spec.name] = quarterly
        elif spec.category == "short_horizon":
            # quarter-average (the score input) AND quarter-worst (peak
            # stress that quarter), per the "both" aggregation decision.
            short_cols[spec.name] = quarterly
            short_worst_cols[spec.name] = resample_to_quarterly(raw, "max")
        elif spec.category == "short_horizon_validation":
            short_val_cols[spec.name] = quarterly
        else:
            if spec.higher_is_later_cycle is None:
                raise ValueError(
                    f"'{spec.name}' has category='{spec.category}' (meant to be scored) but "
                    "higher_is_later_cycle=None (meant not to be scored directly). "
                    "This metric should probably be category='reference' instead -- "
                    "fix the catalog entry rather than silently passing it to the scoring engine."
                )
            scored_cols[spec.name] = quarterly

    return QuarterlyPanel(
        scored_metrics=pd.DataFrame(scored_cols),
        reference_series=pd.DataFrame(reference_cols),
        label_sources=pd.DataFrame(label_cols),
        short_horizon=pd.DataFrame(short_cols),
        short_horizon_worst=pd.DataFrame(short_worst_cols),
        short_horizon_validation=pd.DataFrame(short_val_cols),
        metric_specs=specs_by_name,
    )
