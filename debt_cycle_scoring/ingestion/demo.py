"""
Integration demo: proves the ingestion layer's output (a QuarterlyPanel)
plugs directly into the scoring engine built earlier (normalize.py /
aggregate.py), with no real network call -- a fake transport generates
FRED-shaped synthetic data for every series in the catalog.

This is still not real data (no live API key or network access in this
sandbox), but it validates the SHAPE of the handoff between the two
pieces of the system that were built in separate steps: ingestion
(this subpackage) and scoring (the parent package).

Run with: python3 -m debt_cycle_scoring.ingestion.demo
"""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from .catalog import FRED_CATALOG
from .fred_client import FredClient
from .panel import build_quarterly_panel
from .derived import merge_derived_metrics
from ..normalize import direction_encode, expanding_percentile_rank, winsorize
from ..aggregate import pca_category_score


def _synthetic_observations(series_id: str, native_frequency: str, seed: int) -> list:
    freq_map = {"D": "D", "W": "W", "M": "MS", "Q": "QS", "A": "YS"}
    dates = pd.date_range("1985-01-01", "2019-12-31", freq=freq_map[native_frequency])
    rng = np.random.default_rng(abs(hash(series_id)) % (2**32))
    # random walk with drift, just enough structure to have a non-degenerate PCA later
    values = np.cumsum(rng.normal(0.01, 0.5, len(dates))) + 10
    return [{"date": d.strftime("%Y-%m-%d"), "value": f"{v:.4f}"} for d, v in zip(dates, values)]


def make_synthetic_fred_transport():
    cache = {}

    def transport(url, params):
        series_id = params["series_id"]
        if series_id not in cache:
            spec = next(s for s in FRED_CATALOG if s.series_id == series_id)
            cache[series_id] = _synthetic_observations(series_id, spec.native_frequency, seed=hash(series_id))
        return 200, json.dumps({"observations": cache[series_id]})

    return transport


def run_demo() -> None:
    with TemporaryDirectory() as tmp:
        client = FredClient(
            api_key="synthetic_demo_key",  # any non-None value routes through the JSON path
            transport=make_synthetic_fred_transport(),
            cache_dir=Path(tmp),
        )

        panel = build_quarterly_panel(client, start_date="1985-01-01", end_date="2019-12-31")
        panel = merge_derived_metrics(panel)

        print(f"Scored metrics columns (raw + derived): {list(panel.scored_metrics.columns)}")
        print(f"Reference series columns: {list(panel.reference_series.columns)}")
        print(f"Label source columns: {list(panel.label_sources.columns)}")
        print()

        # Take the asset_prices category -- this now includes BOTH raw
        # ingested metrics (Wilshire level, home price index) AND the
        # derived ratios (Wilshire/GDP, home price/DPI) computed by
        # derived.py, proving the two layers combine into one category
        # exactly as normalize.py/aggregate.py expect.
        asset_price_names = [
            name for name, spec in panel.metric_specs.items() if spec.category == "asset_prices"
        ]
        print(f"Asset-price category now includes: {asset_price_names}")
        print()

        # Take the debt_burden category, run it through the scoring engine's
        # normalize -> aggregate steps, exactly the handoff the real pipeline
        # will use once this is pointed at live data.
        debt_burden_names = [
            name for name, spec in panel.metric_specs.items() if spec.category == "debt_burden"
        ]
        normalized = pd.DataFrame(index=panel.scored_metrics.index)
        for name in debt_burden_names:
            spec = panel.metric_specs[name]
            raw = panel.scored_metrics[name]
            pct = expanding_percentile_rank(winsorize(raw.dropna()), min_periods=8)
            normalized[name] = direction_encode(pct, higher_is_later_cycle=spec.higher_is_later_cycle)

        result = pca_category_score(normalized)
        print("Debt Burden category score (tail):")
        print(result.score.dropna().tail(8).to_string())
        print()
        print(f"PC1 explained variance ratio: {result.explained_variance_ratio:.2f}")
        print("PC1 loadings:")
        print(result.loadings.to_string())


if __name__ == "__main__":
    run_demo()
