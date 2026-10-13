"""
Tests for scorer.fiscal_panel and the dashboard exporter's render step. Run:
    python -m debt_cycle_scoring.scorer.test_fiscal_panel
"""
from __future__ import annotations

import json
import tempfile
import types
from pathlib import Path

import numpy as np
import pandas as pd

from debt_cycle_scoring.scorer.fiscal_panel import build_fiscal_panel, expanding_percentile


def _panel():
    qi = pd.period_range("1970Q1", "2025Q4", freq="Q")
    return types.SimpleNamespace(
        scored_metrics=pd.DataFrame({"Household net worth / DPI": np.linspace(4, 8, len(qi))}, index=qi),
        reference_series=pd.DataFrame(index=qi))


def _raw():
    d = pd.date_range("1970-01-01", "2025-12-01", freq="QS")
    a = pd.date_range("1970-01-01", "2025-01-01", freq="YS")
    return {
        "GFDEGDQ188S": pd.Series(np.linspace(35, 120, len(d)), index=d),
        "FYFSGDA188S": pd.Series(np.full(len(a), -3.0), index=a),
        "A091RC1Q027SBEA": pd.Series(np.linspace(10, 1000, len(d)), index=d),
        "FGRECPT": pd.Series(np.linspace(200, 5000, len(d)), index=d),
    }


def test_expanding_percentile_has_no_lookahead():
    s = pd.Series(np.arange(60, dtype=float), index=pd.period_range("2000Q1", periods=60, freq="Q"))
    p = expanding_percentile(s)
    s2 = s.copy(); s2.iloc[-1] = -999            # changing the future...
    p2 = expanding_percentile(s2)
    assert p.iloc[:-1].equals(p2.iloc[:-1])       # ...never changes the past


def test_record_high_reads_100th_percentile():
    m = {x.key: x for x in build_fiscal_panel(_raw(), _panel())}
    assert m["fed_debt"].percentile == 100
    assert m["fed_debt"].latest == 120.0 and m["fed_debt"].unit == "% GDP"


def test_deficit_is_shown_positive_and_not_extended_into_future():
    m = {x.key: x for x in build_fiscal_panel(_raw(), _panel())}
    assert m["deficit"].latest == 3.0
    assert pd.Period(m["deficit"].as_of, "Q") <= pd.Timestamp.today().to_period("Q")


def test_missing_series_are_skipped_not_fatal():
    raw = _raw(); del raw["FGRECPT"]              # interest/receipts can't be built
    keys = {x.key for x in build_fiscal_panel(raw, _panel())}
    assert "interest" not in keys and "fed_debt" in keys


def test_index_ratios_hide_raw_value():
    m = {x.key: x for x in build_fiscal_panel(_raw(), _panel())}
    assert m["net_worth"].unit == "" and np.isnan(m["net_worth"].latest)


def test_every_metric_is_in_a_dalio_group_and_oriented():
    from debt_cycle_scoring.scorer.fiscal_panel import _SPECS, GROUPS
    assert len(_SPECS) >= 15
    for sp in _SPECS:
        assert sp.group in GROUPS and sp.risk_dir in (-1, 0, 1)


def test_dollar_level_is_divided_by_gdp_but_a_ratio_is_not():
    from debt_cycle_scoring.scorer.fiscal_panel import _per_gdp
    d = pd.date_range("1970-01-01", periods=200, freq="QS")
    gdp = pd.Series(np.linspace(1000, 30000, 200), index=d)          # $bn
    level = gdp * 1000 * 0.5                                          # $mn, constant 50% of GDP
    ratio = pd.Series(np.full(200, 45.0), index=d)                    # already % of GDP
    r1 = _per_gdp({"X": level, "GDP": gdp}, "X")
    r2 = _per_gdp({"Y": ratio, "GDP": gdp}, "Y")
    assert r1.std() < 1e-9            # level / GDP -> flat ratio, the trend is removed
    assert (r2 == 45.0).all()         # ratio left untouched


def test_low_is_risk_metrics_are_flipped():
    d = pd.date_range("1960-01-01", periods=240, freq="MS")
    raw = {"FEDFUNDS": pd.Series(np.r_[np.linspace(8, 1, 200), np.full(40, 0.1)], index=d)}
    m = {x.key: x for x in build_fiscal_panel(raw, _panel())}
    ff = m["fed_funds"]
    assert ff.percentile <= 10 and ff.risk_percentile >= 90  # rate at its low = little room to cut (ties: mid-rank)


def test_implausible_units_hide_the_raw_value():
    d = pd.date_range("1970-01-01", periods=200, freq="QS")
    raw = {"FDHBFIN": pd.Series(np.full(200, 7000.0), index=d),      # $bn
           "GFDEBTN": pd.Series(np.full(200, 30000.0), index=d)}     # wrong scale on purpose
    m = {x.key: x for x in build_fiscal_panel(raw, _panel())}
    assert m["foreign_share"].unit == "" and np.isnan(m["foreign_share"].latest)


def test_exporter_render_injects_json_safely():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import export_dashboard as E
    if not E.TEMPLATE.exists():
        return                                       # template not shipped alongside -- skip
    data = E.clean({"headline": {"q": "2025Q4", "risk": 40, "note": "</script><b>x"}, "v": float("nan")})
    assert data["v"] is None                         # NaN -> null, valid JSON
    out = E.render(data, Path(tempfile.mkdtemp()))
    html = out.read_text(encoding="utf-8")
    assert E.PLACEHOLDER not in html
    assert "</script><b>x" not in html               # can't break out of the script tag
    # plain-English layer is inlined, not left as placeholders
    assert "__EXPLAIN_JS__" not in html and "__GUIDE_JS__" not in html
    assert "const EXPLAIN" in html and "SCENARIOS" in html
    json.loads((out.parent / "data.json").read_text())


def run_all():
    tests = [obj for name, obj in list(globals().items()) if name.startswith("test_")]
    failures = []
    for t in tests:
        try:
            t(); print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failures.append(t.__name__); print(f"FAIL  {t.__name__}: {e}")
        except Exception as e:
            failures.append(t.__name__); print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print()
    if failures:
        print(f"{len(failures)} of {len(tests)} tests FAILED: {failures}"); raise SystemExit(1)
    print(f"All {len(tests)} tests passed.")


if __name__ == "__main__":
    run_all()
