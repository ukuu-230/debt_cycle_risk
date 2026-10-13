"""
Tests for the multi-country expansion:
  - ingestion.intl_catalog (per-country flag series map)
  - scorer.build_country_risk_series (country-agnostic composite builder)
Run with:
    python3 -m debt_cycle_scoring.scorer.test_multicountry
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from debt_cycle_scoring.ingestion.intl_catalog import (
    INTL_CATALOG, country_series_ids, available_countries,
)
from debt_cycle_scoring.scorer import build_country_risk_series, FLAG_NAMES


def _qidx(start, n):
    return pd.period_range(start, periods=n, freq="Q")


# ---- intl_catalog ----------------------------------------------------

def test_us_and_euro_present():
    assert "US" in available_countries()
    assert "Euro" in available_countries()


def test_us_has_all_flag_series():
    ids = country_series_ids("US")
    # US should have every flag series populated (it's the full-coverage case)
    for k in ["credit_gap_ratio", "debt_service_ratio", "long_bond_10y",
              "short_rate", "credit_spread", "m2_velocity", "breakeven_inflation"]:
        assert ids[k] is not None, f"US missing {k}"


def test_euro_drops_some_flags_honestly():
    ids = country_series_ids("Euro")
    # Euro HAS the credit-gap anchor + long bond + spread
    assert ids["credit_gap_ratio"] == "QXMPAM770A"
    assert ids["long_bond_10y"] == "IRLTLT01EZM156N"
    assert ids["credit_spread"] == "BAMLHE00EHYIOAS"
    # Euro DROPS DSR (FRED has no non-US DSR), m2_velocity, breakeven,
    # and short_rate (curve) -- all honest None, no fabricated proxies
    assert ids["debt_service_ratio"] is None
    assert ids["m2_velocity"] is None
    assert ids["breakeven_inflation"] is None
    assert ids["short_rate"] is None


def test_china_not_included():
    assert "China" not in available_countries(), "China deliberately excluded per assessment"


# ---- build_country_risk_series --------------------------------------

def _euro_like_inputs(n=60):
    """Euro-like: 4 flags present (gap, dsr, long_bond, spread), 3 absent."""
    idx = _qidx("2000Q1", n)
    rng = np.random.default_rng(0)
    gap = pd.Series(np.linspace(-4, 12, n), index=idx)   # rising into bubble territory
    inputs = {
        "dsr": pd.Series(np.linspace(8, 12, n) + rng.normal(0, 0.1, n), index=idx),
        "long_bond": pd.Series(np.linspace(5, 3, n), index=idx),
        "credit_spread": pd.Series(4 + rng.normal(0, 0.3, n), index=idx),
        # no curve, no m2_velocity, no breakeven -> those flags drop
    }
    return inputs, gap, idx


def test_country_builder_runs_with_partial_flags():
    inputs, gap, idx = _euro_like_inputs()
    result = build_country_risk_series(inputs, gap, min_periods=8)
    valid = result.risk.dropna()
    assert len(valid) > 0
    assert valid.between(0, 100).all()
    # only the present flags should ever be counted; max here is 4
    # (credit_gap, dsr, credit_spread, long_bond_level)
    assert result.n_flags.max() <= 4
    assert result.n_flags.max() >= 3


def test_country_builder_drops_absent_flags_from_frame():
    inputs, gap, idx = _euro_like_inputs()
    result = build_country_risk_series(inputs, gap, min_periods=8)
    last = result.flags.dropna(how="all").index.max()
    present = result.flags.loc[last].dropna().index.tolist()
    # curve / m2_velocity / debasement were not supplied -> absent
    assert "yield_curve" not in present
    assert "m2_velocity" not in present
    assert "debasement" not in present
    # the supplied ones are present
    assert "credit_gap" in present
    assert "debt_service_ratio" in present


def test_country_builder_rising_gap_raises_risk():
    inputs, gap, idx = _euro_like_inputs()
    result = build_country_risk_series(inputs, gap, min_periods=8)
    v = result.risk.dropna()
    # gap rises from -4 to 12 over the sample -> late risk should exceed early
    assert v.iloc[-1] > v.iloc[0]


def test_country_builder_flag_columns_are_canonical():
    inputs, gap, idx = _euro_like_inputs()
    result = build_country_risk_series(inputs, gap, min_periods=8)
    assert list(result.flags.columns) == FLAG_NAMES


def test_country_builder_raises_on_empty():
    idx = _qidx("2000Q1", 10)
    try:
        build_country_risk_series({}, pd.Series(dtype=float))
        assert False, "expected ValueError on no inputs"
    except ValueError:
        pass


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
