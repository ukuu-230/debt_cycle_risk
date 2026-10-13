"""
debt_cycle_scoring.ingestion
============================

FRED data ingestion -- design doc section 4 ("Data Source Map by Country").

This is deliberately a separate subpackage from the scoring math
(normalize/aggregate/stages/explain), matching the architecture in the
design doc's section 5: an ingestion layer that produces a plain
DataFrame, feeding a calculation engine that doesn't need to know where
its input came from.
"""
from .fred_client import FredClient, FredSeriesResult
from .catalog import FRED_CATALOG, MetricSpec
from .panel import build_quarterly_panel
from .derived import compute_derived_metrics, merge_derived_metrics, DerivedMetricSpec

__all__ = [
    "FredClient",
    "FredSeriesResult",
    "FRED_CATALOG",
    "MetricSpec",
    "build_quarterly_panel",
    "compute_derived_metrics",
    "merge_derived_metrics",
    "DerivedMetricSpec",
]
