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
