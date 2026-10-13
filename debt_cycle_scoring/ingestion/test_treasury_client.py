"""
Tests for ingestion.treasury_client. Fake responses copy the live layouts
checked on 2026-10-13. Run:
    python -m debt_cycle_scoring.ingestion.test_treasury_client
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from debt_cycle_scoring.ingestion.treasury_client import (
    TreasuryClient, TreasuryDataError, parse_tic_table5, fetch_treasury_extras,
    EP_AVG_RATES, EP_MSPD_1, TIC_TABLE5_URL,
)

TIC_TEXT = (
    "Table 5: Major Foreign Holders of Treasury Securities\n"
    "Holdings at end of time period\nBillions of dollars\n"
    "Link: https://ticdata.treasury.gov/resource-center/data-chart-center/tic/Documents/slt_table5.txt\n\n"
    "Country\t2026-07\t2026-06\t2026-05\t2026-04\t2026-03\t2026-02\t2026-01\t2025-12\t2025-11\t2025-10\t2025-09\t2025-08\t2025-07\n"
    "Japan\t1103.9\t1116.7\t1143.1\t1209.9\t1191.6\t1239.3\t1225.3\t1185.5\t1202.7\t1200.0\t1189.3\t1183.9\t1155.4\n"
    "United Kingdom\t998.3\t939.9\t948.6\t937.5\t927.1\t897.3\t879.8\t863.1\t879.8\t875.4\t862.5\t901.3\t895.9\n"
    "China, Mainland\t618.0\t633.4\t659.3\t651.1\t653.3\t694.2\t695.3\t684.4\t683.9\t687.7\t699.2\t699.7\t695.6\n"
    "Grand Total\t9248.1\t9298.5\t9368.5\t9350.1\t9352.4\t9487.3\t9289.1\t9269.5\t9349.6\t9230.5\t9237.5\t9248.2\t9109.5\n"
    "Of Which: Foreign Official\t3773.1\t3778.1\t3845.9\t3904.5\t3903.0\t4009.6\t3941.8\t3877.7\t3912.3\t3866.4\t3872.6\t3919.8\t3886.5\n"
    "Of Which: Foreign Official Treasury Bills\t354.4\t360.6\t396.2\t457.3\t443.7\t453.3\t407.8\t388.5\t385.6\t386.0\t371.3\t411.6\t410.0\n"
)


def _avg_rates_payload(page, pages):
    dates = pd.date_range("2001-01-31", periods=300, freq="ME")
    chunk = np.array_split(np.arange(len(dates)), pages)[page - 1]
    data = []
    for i in chunk:
        d = dates[i].strftime("%Y-%m-%d")
        data.append({"record_date": d, "security_type_desc": "Marketable",
                     "security_desc": "Total Marketable", "avg_interest_rate_amt": f"{2 + i / 300:.3f}"})
        data.append({"record_date": d, "security_type_desc": "Marketable",
                     "security_desc": "Treasury Bills", "avg_interest_rate_amt": "9.999"})
    return json.dumps({"data": data, "meta": {"total-pages": pages}})


def _mspd_payload():
    data = []
    for d in pd.date_range("2001-01-31", periods=300, freq="ME"):
        ds = d.strftime("%Y-%m-%d")
        data += [
            {"record_date": ds, "security_type_desc": "Marketable", "security_class_desc": "Bills", "total_mil_amt": "7118145.3141"},
            {"record_date": ds, "security_type_desc": "Marketable", "security_class_desc": "Notes", "total_mil_amt": "16281507.3155"},
            {"record_date": ds, "security_type_desc": "Total Marketable", "security_class_desc": "_", "total_mil_amt": "31834878.4434023"},
        ]
    return json.dumps({"data": data, "meta": {"total-pages": 1}})


def _transport(fail=()):
    calls = []

    def t(url, params):
        calls.append(url)
        for f in fail:
            if f in url:
                return 503, "down"
        if url.endswith(EP_AVG_RATES):
            return 200, _avg_rates_payload(int(params["page[number]"]), 2)
        if url.endswith(EP_MSPD_1):
            return 200, _mspd_payload()
        if url == TIC_TABLE5_URL:
            return 200, TIC_TEXT
        return 404, "nope"
    t.calls = calls
    return t


def _client(**kw):
    return TreasuryClient(transport=_transport(**kw), cache_dir=Path(tempfile.mkdtemp()), sleep_fn=lambda s: None)


def test_avg_rate_reads_total_marketable_across_pages():
    s = _client().avg_rate_marketable()
    assert len(s) == 300                              # both pages, one row per month
    assert s.max() < 3.1                              # 'Treasury Bills' rows (9.999) excluded


def test_bills_share_matches_live_example():
    s = _client().bills_share()
    assert abs(s.iloc[-1] - 22.36) < 0.01             # 7,118,145 / 31,834,878 = 22.36% (Sep 2026)


def test_tic_table5_parses_live_layout():
    t = parse_tic_table5(TIC_TEXT)
    assert t.index[-1] == pd.Timestamp("2026-07-31") and len(t) == 13
    assert t["Japan"].iloc[-1] == 1103.9 and t["China, Mainland"].iloc[0] == 695.6


def test_tic_layout_change_raises_clear_error():
    try:
        parse_tic_table5("totally different layout")
        assert False, "should raise"
    except TreasuryDataError as exc:
        assert "Country" in str(exc)


def test_extras_build_expected_series():
    out = fetch_treasury_extras(_client(), log=lambda m: None)
    assert set(out) == {"TSY_AVG_RATE_MKT", "TSY_BILLS_SHARE", "TIC_JP_CN", "TIC_OFFICIAL"}
    assert abs(out["TIC_JP_CN"].iloc[-1] - (1103.9 + 618.0)) < 1e-9
    assert out["TIC_OFFICIAL"].iloc[-1] == 3773.1     # the total row, not the bills sub-row


def test_one_source_down_does_not_block_others():
    msgs = []
    out = fetch_treasury_extras(_client(fail=("ticdata",)), log=msgs.append)
    assert "TSY_AVG_RATE_MKT" in out and "TIC_JP_CN" not in out
    assert any("TIC" in m for m in msgs)


def test_responses_are_cached():
    c = _client()
    c.avg_rate_marketable(); n = len(c.transport.calls)
    c.avg_rate_marketable()
    assert len(c.transport.calls) == n                # second call served from disk


def test_panel_rows_from_treasury_data():
    import types
    from debt_cycle_scoring.scorer.fiscal_panel import build_fiscal_panel
    extras = fetch_treasury_extras(_client(), log=lambda m: None)
    d = pd.date_range("2001-01-01", periods=300, freq="MS")
    raw = dict(extras, DGS10=pd.Series(np.linspace(4.5, 4.5, 300), index=d))
    qi = pd.period_range("2001Q1", periods=100, freq="Q")
    panel = types.SimpleNamespace(scored_metrics=pd.DataFrame(index=qi), reference_series=pd.DataFrame(index=qi))
    m = {x.key: x for x in build_fiscal_panel(raw, panel)}
    assert m["refi_gap"].unit == "pp" and m["refi_gap"].latest > 0          # 4.5% market vs ~3% average
    jc = m["tic_jp_cn"]
    assert jc.mode == "chg12m" and jc.percentile is None
    assert abs(jc.latest - ((1103.9 + 618.0) / (1155.4 + 695.6) - 1) * 100) < 0.1
    assert jc.toward_risk is True                                            # holdings fell = risky way
    assert abs(m["bills_share"].latest - 22.4) < 0.05


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
