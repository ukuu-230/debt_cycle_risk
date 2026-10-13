"""
Tests for debt_cycle_scoring.ingestion.derived -- the ratio-computation
layer. Run with:
    python3 -m debt_cycle_scoring.ingestion.test_derived
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .derived import _safe_ratio, compute_derived_metrics, merge_derived_metrics
from .panel import QuarterlyPanel


def _qidx(start: str, n: int) -> pd.PeriodIndex:
    return pd.period_range(start, periods=n, freq="Q")


def test_safe_ratio_handles_zero_denominator():
    idx = _qidx("2000Q1", 5)
    num = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0], index=idx)
    den = pd.Series([2.0, 0.0, 5.0, 0.0, 10.0], index=idx)
    ratio = _safe_ratio(num, den)
    assert ratio.iloc[0] == 5.0
    assert pd.isna(ratio.iloc[1]), "division by zero should become NaN, not inf"
    assert pd.isna(ratio.iloc[3])
    assert ratio.iloc[4] == 5.0


def test_safe_ratio_aligns_mismatched_indices():
    num = pd.Series([10.0, 20.0, 30.0], index=_qidx("2000Q1", 3))
    den = pd.Series([2.0, 5.0], index=_qidx("2000Q2", 2))  # overlaps num on Q2 and Q3
    ratio = _safe_ratio(num, den)
    assert len(ratio) == 2, "inner join should keep only the overlapping quarters (Q2, Q3)"
    assert ratio.loc[pd.Period("2000Q2", freq="Q")] == 20.0 / 2.0
    assert ratio.loc[pd.Period("2000Q3", freq="Q")] == 30.0 / 5.0


def _make_test_panel() -> QuarterlyPanel:
    idx = _qidx("2000Q1", 12)
    scored = pd.DataFrame(
        {
            "S&P 500": np.linspace(1000, 2000, 12),
            "Household net worth": np.linspace(50, 100, 12),
            "Effective federal funds rate": np.linspace(1, 5, 12),
            "Fed total assets (balance sheet)": np.linspace(500, 900, 12),
            "Case-Shiller US National Home Price Index": np.linspace(150, 220, 12),
        },
        index=idx,
    )
    reference = pd.DataFrame(
        {
            "Nominal GDP": np.linspace(10000, 12000, 12),
            "Disposable personal income": np.linspace(8000, 9500, 12),
            "Median household income (nominal)": np.linspace(60000, 68000, 12),
            "CPI, all urban consumers": np.linspace(200, 215, 12),
        },
        index=idx,
    )
    labels = pd.DataFrame({"NBER recession indicator": [0] * 12}, index=idx)
    return QuarterlyPanel(
        scored_metrics=scored, reference_series=reference, label_sources=labels, metric_specs={}
    )


def test_compute_derived_metrics_produces_all_expected_ratios():
    panel = _make_test_panel()
    derived_df, specs = compute_derived_metrics(panel)

    expected = {
        "S&P 500 / GDP (Buffett indicator proxy)",
        "Household net worth / DPI",
        "Real federal funds rate",
        "Fed balance sheet / GDP",
        "Home price / income",
    }
    assert expected.issubset(set(derived_df.columns))
    assert expected.issubset(set(specs.keys()))
    for name in expected:
        assert specs[name].higher_is_later_cycle is True
        assert specs[name].category in ("asset_prices", "cb_stance")


def test_compute_derived_metrics_skips_missing_inputs_gracefully():
    panel = _make_test_panel()
    # drop a required input for one ratio
    panel.scored_metrics = panel.scored_metrics.drop(columns=["Household net worth"])
    derived_df, specs = compute_derived_metrics(panel)
    assert "Household net worth / DPI" not in derived_df.columns
    assert "Household net worth / DPI" not in specs
    # the other ratios should still be computed fine
    assert "S&P 500 / GDP (Buffett indicator proxy)" in derived_df.columns


def test_real_fed_funds_rate_matches_manual_calculation():
    panel = _make_test_panel()
    derived_df, _ = compute_derived_metrics(panel)
    cpi = panel.reference_series["CPI, all urban consumers"]
    cpi_yoy = cpi.pct_change(4) * 100
    expected = (panel.scored_metrics["Effective federal funds rate"] - cpi_yoy).dropna()
    actual = derived_df["Real federal funds rate"].dropna()
    pd.testing.assert_series_equal(actual.sort_index(), expected.sort_index(), check_names=False)


def test_merge_derived_metrics_extends_panel_without_losing_raw_columns():
    panel = _make_test_panel()
    merged = merge_derived_metrics(panel)

    # original raw columns should still be present
    for col in panel.scored_metrics.columns:
        assert col in merged.scored_metrics.columns
    # new derived columns should be present
    assert "S&P 500 / GDP (Buffett indicator proxy)" in merged.scored_metrics.columns
    # metric_specs should have entries for the new derived columns
    assert "S&P 500 / GDP (Buffett indicator proxy)" in merged.metric_specs
    assert merged.metric_specs["S&P 500 / GDP (Buffett indicator proxy)"].category == "asset_prices"
    # reference/label_sources should pass through unchanged
    pd.testing.assert_frame_equal(merged.reference_series, panel.reference_series)
    pd.testing.assert_frame_equal(merged.label_sources, panel.label_sources)


def test_home_price_ratio_uses_nominal_median_income_not_dpi():
    # Regression/documentation test: home price/income must use the
    # NOMINAL median household income series, matched to CSUSHPISA's own
    # nominal terms -- not disposable personal income (a different
    # concept entirely) and not the real/inflation-adjusted median income
    # series (which would introduce a spurious inflation mismatch since
    # CSUSHPISA is nominal). See derived.py's caveat text for the reasoning.
    panel = _make_test_panel()
    derived_df, specs = compute_derived_metrics(panel)
    assert "Home price / income" in derived_df.columns
    assert "Median household income" in specs["Home price / income"].formula
    assert "Disposable personal income" not in specs["Home price / income"].formula


def test_merge_derived_metrics_preserves_short_horizon_frames():
    # Regression test for a real runtime bug: merge_derived_metrics rebuilds
    # the QuarterlyPanel, and an earlier version forgot to carry the
    # short_horizon / short_horizon_worst / short_horizon_validation frames
    # forward, so they silently reset to empty. That made the entire short-
    # horizon sub-score vanish in run_scorer.py even though the panel built
    # them correctly. Unit tests missed it because they built panels without
    # short-horizon data -- this test supplies it explicitly.
    panel = _make_test_panel()
    idx = panel.scored_metrics.index
    panel.short_horizon = pd.DataFrame({"VIX equity volatility": range(len(idx))}, index=idx)
    panel.short_horizon_worst = pd.DataFrame({"VIX equity volatility": range(len(idx))}, index=idx)
    panel.short_horizon_validation = pd.DataFrame({"Chicago Fed NFCI (validation cross-check)": range(len(idx))}, index=idx)

    merged = merge_derived_metrics(panel)
    assert "VIX equity volatility" in merged.short_horizon.columns, "short_horizon must survive merge"
    assert "VIX equity volatility" in merged.short_horizon_worst.columns, "short_horizon_worst must survive merge"
    assert "Chicago Fed NFCI (validation cross-check)" in merged.short_horizon_validation.columns, \
        "short_horizon_validation must survive merge"


def run_all():
    tests = [obj for name, obj in list(globals().items()) if name.startswith("test_")]
    failures = []
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failures.append(t.__name__)
            print(f"FAIL  {t.__name__}: {e}")
        except Exception as e:
            failures.append(t.__name__)
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print()
    if failures:
        print(f"{len(failures)} of {len(tests)} tests FAILED: {failures}")
        raise SystemExit(1)
    print(f"All {len(tests)} tests passed.")


if __name__ == "__main__":
    run_all()
