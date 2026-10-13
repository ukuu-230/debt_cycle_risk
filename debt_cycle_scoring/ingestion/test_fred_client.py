"""
Tests for the FRED ingestion client. No real network calls -- a fake
transport returns hand-built responses shaped like FRED's documented
JSON and CSV formats, so parsing, caching, retries, and panel assembly
are all verified without needing outbound network access.

Run with: python3 -m debt_cycle_scoring.ingestion.test_fred_client
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pandas as pd

from .catalog import MetricSpec
from .fred_client import FredAPIError, FredClient, FredSeriesNotFoundError
from .panel import build_quarterly_panel, resample_to_quarterly


# ---- fake transports -----------------------------------------------------

def make_json_transport(observations_by_series, status_overrides=None):
    """
    Builds a fake transport matching the shape of FRED's real JSON API
    response: {"observations": [{"date": "...", "value": "..."}, ...]}.
    `status_overrides` lets a test simulate a transient 500 then success.
    """
    call_count = {"n": 0}
    status_overrides = status_overrides or {}

    def transport(url, params):
        call_count["n"] += 1
        n = call_count["n"]
        if n in status_overrides:
            return status_overrides[n], "server error"
        series_id = params["series_id"]
        obs = observations_by_series[series_id]
        body = json.dumps({"observations": obs})
        return 200, body

    transport.call_count = call_count
    return transport


def make_csv_transport(csv_by_series):
    def transport(url, params):
        series_id = params["id"]
        return 200, csv_by_series[series_id]

    return transport


def make_404_transport():
    def transport(url, params):
        return 404, "not found"

    return transport


# ---- tests ----------------------------------------------------------------

def test_json_parsing_handles_missing_values():
    obs = [
        {"date": "2020-01-01", "value": "1.5"},
        {"date": "2020-04-01", "value": "."},  # FRED's own missing-value marker
        {"date": "2020-07-01", "value": "2.0"},
    ]
    transport = make_json_transport({"TESTID": obs})
    with tempfile.TemporaryDirectory() as tmp:
        client = FredClient(api_key="fake_key", transport=transport, cache_dir=Path(tmp))
        result = client.fetch_series("TESTID")
        assert result.source == "api"
        assert len(result.series) == 3
        assert pd.isna(result.series.iloc[1]), "FRED's '.' missing-value marker should become NaN"
        assert result.series.iloc[0] == 1.5
        assert result.series.iloc[2] == 2.0


def test_csv_fallback_used_when_no_api_key():
    csv_body = "DATE,TESTID\n2020-01-01,1.5\n2020-04-01,.\n2020-07-01,2.0\n"
    transport = make_csv_transport({"TESTID": csv_body})
    with tempfile.TemporaryDirectory() as tmp:
        client = FredClient(api_key=None, transport=transport, cache_dir=Path(tmp))
        result = client.fetch_series("TESTID")
        assert result.source == "csv"
        assert len(result.series) == 3
        assert pd.isna(result.series.iloc[1])


def test_cache_avoids_second_transport_call():
    obs = [{"date": "2020-01-01", "value": "1.0"}]
    transport = make_json_transport({"TESTID": obs})
    with tempfile.TemporaryDirectory() as tmp:
        client = FredClient(api_key="fake_key", transport=transport, cache_dir=Path(tmp))
        client.fetch_series("TESTID")
        client.fetch_series("TESTID")  # should hit cache, not transport again
        assert transport.call_count["n"] == 1, "second fetch should be served from cache"


def test_force_refresh_bypasses_cache():
    obs = [{"date": "2020-01-01", "value": "1.0"}]
    transport = make_json_transport({"TESTID": obs})
    with tempfile.TemporaryDirectory() as tmp:
        client = FredClient(api_key="fake_key", transport=transport, cache_dir=Path(tmp))
        client.fetch_series("TESTID")
        client.fetch_series("TESTID", force_refresh=True)
        assert transport.call_count["n"] == 2


def test_retry_recovers_from_transient_500():
    obs = [{"date": "2020-01-01", "value": "1.0"}]
    # first call returns 500, second call (retry) succeeds
    transport = make_json_transport({"TESTID": obs}, status_overrides={1: 500})
    with tempfile.TemporaryDirectory() as tmp:
        client = FredClient(
            api_key="fake_key", transport=transport, cache_dir=Path(tmp),
            sleep_fn=lambda seconds: None,  # don't actually sleep in tests
        )
        result = client.fetch_series("TESTID")
        assert len(result.series) == 1
        assert transport.call_count["n"] == 2, "should have retried once after the 500"


def test_404_raises_not_found_without_retrying():
    transport = make_404_transport()
    with tempfile.TemporaryDirectory() as tmp:
        client = FredClient(
            api_key="fake_key", transport=transport, cache_dir=Path(tmp),
            sleep_fn=lambda seconds: None,
        )
        try:
            client.fetch_series("NOPE")
            assert False, "expected FredSeriesNotFoundError"
        except FredSeriesNotFoundError:
            pass


def test_fetch_many_skips_missing_series_by_default():
    # A 400 "does not exist" on one series should NOT crash a multi-series
    # fetch -- it should be skipped with a warning (FRED discontinues/
    # renames series periodically: Wilshire, gold). The returned dict
    # simply omits the missing one.
    good_obs = [{"date": "2020-01-01", "value": "1.0"}]

    def transport(url, params):
        sid = params["series_id"]
        if sid == "BADSERIES":
            return 400, '{"error_code":400,"error_message":"Bad Request. The series does not exist."}'
        return 200, json.dumps({"observations": good_obs})

    with tempfile.TemporaryDirectory() as tmp:
        client = FredClient(api_key="fake_key", transport=transport, cache_dir=Path(tmp),
                            sleep_fn=lambda s: None)
        results = client.fetch_many(["GOODONE", "BADSERIES", "GOODTWO"])
        assert "GOODONE" in results and "GOODTWO" in results
        assert "BADSERIES" not in results, "missing series should be skipped, not crash"


def test_fetch_many_can_still_hard_fail_on_missing():
    def transport(url, params):
        return 404, "not found"
    with tempfile.TemporaryDirectory() as tmp:
        client = FredClient(api_key="fake_key", transport=transport, cache_dir=Path(tmp),
                            sleep_fn=lambda s: None)
        try:
            client.fetch_many(["X"], skip_missing=False)
            assert False, "expected an error with skip_missing=False"
        except FredAPIError:
            pass


def test_repeated_500_eventually_raises():
    transport = lambda url, params: (500, "server error")
    with tempfile.TemporaryDirectory() as tmp:
        client = FredClient(
            api_key="fake_key", transport=transport, cache_dir=Path(tmp),
            max_retries=2, sleep_fn=lambda seconds: None,
        )
        try:
            client.fetch_series("TESTID")
            assert False, "expected FredAPIError after exhausting retries"
        except FredAPIError:
            pass


def test_api_key_never_leaks_into_error_messages():
    # Regression test: an earlier version embedded the raw params dict
    # (including the API key) directly into every exception message --
    # meaning any error a user pasted somewhere (a terminal, a chat, a
    # bug report) leaked their key in plain text. This checks the key is
    # redacted across every error path: repeated 5xx, a 4xx auth
    # rejection, and a 404.
    secret_key = "totally_secret_key_12345"

    with tempfile.TemporaryDirectory() as tmp:
        transport_500 = lambda url, params: (500, "server error")
        client = FredClient(
            api_key=secret_key, transport=transport_500, cache_dir=Path(tmp),
            max_retries=2, sleep_fn=lambda seconds: None,
        )
        try:
            client.fetch_series("TESTID")
            assert False, "expected an error"
        except FredAPIError as e:
            assert secret_key not in str(e), "API key leaked into the repeated-500 error message"

    with tempfile.TemporaryDirectory() as tmp:
        transport_401 = lambda url, params: (401, "invalid api_key")
        client = FredClient(
            api_key=secret_key, transport=transport_401, cache_dir=Path(tmp),
            sleep_fn=lambda seconds: None,
        )
        try:
            client.fetch_series("TESTID")
            assert False, "expected an error"
        except FredAPIError as e:
            assert secret_key not in str(e), "API key leaked into the 401 error message"

    with tempfile.TemporaryDirectory() as tmp:
        transport_404 = make_404_transport()
        client = FredClient(
            api_key=secret_key, transport=transport_404, cache_dir=Path(tmp),
            sleep_fn=lambda seconds: None,
        )
        try:
            client.fetch_series("TESTID")
            assert False, "expected an error"
        except FredSeriesNotFoundError as e:
            assert secret_key not in str(e), "API key leaked into the 404 error message"


def test_resample_to_quarterly_last_and_mean():
    idx = pd.date_range("2020-01-01", periods=6, freq="MS")
    s = pd.Series([1, 2, 3, 4, 5, 6], index=idx)
    last_result = resample_to_quarterly(s, "last")
    mean_result = resample_to_quarterly(s, "mean")
    q1 = pd.Period("2020Q1", freq="Q")
    assert last_result[q1] == 3, "last of Jan/Feb/Mar values [1,2,3] should be 3"
    assert mean_result[q1] == 2, "mean of [1,2,3] should be 2"


def test_build_quarterly_panel_separates_categories_correctly():
    catalog = [
        MetricSpec("Scored Metric", "SCORED", "debt_burden", "M", "mean", higher_is_later_cycle=True),
        MetricSpec("Reference Metric", "REF", "reference", "Q", "last", higher_is_later_cycle=None),
        MetricSpec("Label Metric", "LABEL", "label_source", "M", "last", higher_is_later_cycle=None),
    ]
    obs_by_series = {
        "SCORED": [{"date": "2020-01-01", "value": "1.0"}, {"date": "2020-02-01", "value": "2.0"}],
        "REF": [{"date": "2020-01-01", "value": "100.0"}],
        "LABEL": [{"date": "2020-01-01", "value": "0"}],
    }
    transport = make_json_transport(obs_by_series)
    with tempfile.TemporaryDirectory() as tmp:
        client = FredClient(api_key="fake_key", transport=transport, cache_dir=Path(tmp))
        panel = build_quarterly_panel(client, catalog=catalog)

        assert "Scored Metric" in panel.scored_metrics.columns
        assert "Reference Metric" in panel.reference_series.columns
        assert "Label Metric" in panel.label_sources.columns
        assert "Reference Metric" not in panel.scored_metrics.columns
        assert panel.metric_specs["Scored Metric"].higher_is_later_cycle is True


def test_annual_series_forward_fills_across_quarters():
    catalog = [
        MetricSpec("Annual Metric", "ANNUAL", "deleveraging_signals", "A", "ffill", higher_is_later_cycle=False),
    ]
    obs = [
        {"date": "2020-01-01", "value": "-3.0"},
        {"date": "2021-01-01", "value": "-5.0"},
    ]
    transport = make_json_transport({"ANNUAL": obs})
    with tempfile.TemporaryDirectory() as tmp:
        client = FredClient(api_key="fake_key", transport=transport, cache_dir=Path(tmp))
        panel = build_quarterly_panel(client, catalog=catalog)
        series = panel.scored_metrics["Annual Metric"]
        # All four quarters of 2020 should carry the -3.0 annual reading forward
        assert (series.loc["2020Q1":"2020Q4"] == -3.0).all()
        assert series.loc["2021Q1"] == -5.0


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
