"""
FRED ingestion client -- design doc section 4.

Supports two modes:
  - "api" mode (preferred): the official api.stlouisfed.org/fred/series/observations
    endpoint, JSON, requires a free API key (https://fredaccount.stlouisfed.org/apikeys).
    Gives clean structured data and room to add vintage/real-time-start
    queries later (relevant to the ALFRED point raised in the design doc's
    section 7.2 backtest caveat).
  - "csv" mode (no key required): the public fredgraph.csv download. Useful
    for quick prototyping without registering a key, but more fragile --
    it's not a documented/versioned API contract the way the JSON endpoint
    is, just a stable-in-practice public download URL.

Network access is injected via a `transport` callable rather than calling
`requests` directly inside this class, specifically so the parsing/caching/
retry logic can be fully unit-tested offline (see test_fred_client.py) --
this sandbox itself has no outbound network access, so that's not a nice-to-have
here, it's the only way this code gets tested before you run it somewhere
with real network access.
"""
from __future__ import annotations

import hashlib
import io
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Tuple

import pandas as pd

Transport = Callable[[str, Dict[str, str]], Tuple[int, str]]


def _redact(params: Dict[str, str]) -> Dict[str, str]:
    """
    Never let an API key end up in an exception message, a log line, or a
    terminal paste-into-chat -- all real things that happen. Used
    everywhere `params` gets embedded into error text below.
    """
    return {k: ("***REDACTED***" if k == "api_key" else v) for k, v in params.items()}


class FredAPIError(RuntimeError):
    """Raised when FRED returns an error status this client can't recover from."""


class FredSeriesNotFoundError(FredAPIError):
    """Raised specifically for a 404 / 'series does not exist' response."""


def _is_missing_series_error(exc: Exception) -> bool:
    """
    True if this error looks like a 'series does not exist' response (a 404,
    or a 400 whose body says the series doesn't exist), as opposed to a
    transient/auth/other error we should NOT silently skip.
    """
    if isinstance(exc, FredSeriesNotFoundError):
        return True
    msg = str(exc).lower()
    return ("does not exist" in msg) or ("404" in msg)


@dataclass
class FredSeriesResult:
    series_id: str
    series: pd.Series          # float values, indexed by pd.Timestamp (daily resolution as FRED reports it)
    source: str                 # "api" | "csv" | "cache"
    fetched_at: pd.Timestamp


def _default_requests_transport(url: str, params: Dict[str, str]) -> Tuple[int, str]:
    """
    Real HTTP transport using `requests`. This is the default used when no
    transport is injected -- swap in a fake for tests, or leave this as-is
    once you're running with network access and (optionally) an API key.
    """
    import requests  # imported lazily so the module has no hard dependency for offline/test use

    resp = requests.get(url, params=params, timeout=30)
    return resp.status_code, resp.text


class FredClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        transport: Optional[Transport] = None,
        cache_dir: Optional[Path] = None,
        cache_ttl_hours: float = 24.0,
        max_retries: int = 3,
        backoff_base_seconds: float = 1.0,
        sleep_fn: Callable[[float], None] = time.sleep,
    ):
        """
        api_key: FRED API key. If None, falls back to the no-key CSV endpoint.
        transport: injectable (url, params) -> (status_code, body_text) callable.
            Defaults to a real `requests`-based call. Tests pass a fake here.
        cache_dir: where to store cached responses on disk. Defaults to
            ~/.cache/debt_cycle_scoring/fred -- caching matters a lot for
            this project specifically, since the Phase 1 backtest (design
            doc section 7) will re-request the same historical series many
            times across walk-forward steps.
        cache_ttl_hours: how long a cached response is considered fresh.
            Historical data revises occasionally (see the ALFRED/vintage
            caveat in the design doc), so this shouldn't be set to "forever"
            even for old-looking data.
        max_retries / backoff_base_seconds: simple exponential backoff on
            transient (5xx or transport-level) failures.
        sleep_fn: injectable so tests don't actually sleep during retry tests.
        """
        self.api_key = api_key
        self.transport = transport or _default_requests_transport
        self.cache_dir = cache_dir or (Path.home() / ".cache" / "debt_cycle_scoring" / "fred")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_ttl_hours = cache_ttl_hours
        self.max_retries = max_retries
        self.backoff_base_seconds = backoff_base_seconds
        self.sleep_fn = sleep_fn

    # ---- public API ---------------------------------------------------

    def fetch_series(
        self,
        series_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        force_refresh: bool = False,
    ) -> FredSeriesResult:
        cache_path = self._cache_path(series_id, start_date, end_date)

        if not force_refresh:
            cached = self._read_cache(cache_path)
            if cached is not None:
                return cached

        if self.api_key:
            series = self._fetch_via_api(series_id, start_date, end_date)
            source = "api"
        else:
            series = self._fetch_via_csv(series_id, start_date, end_date)
            source = "csv"

        result = FredSeriesResult(
            series_id=series_id, series=series, source=source, fetched_at=pd.Timestamp.utcnow()
        )
        self._write_cache(cache_path, result)
        return result

    def fetch_many(
        self,
        series_ids: Iterable[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        force_refresh: bool = False,
        inter_request_delay_seconds: float = 0.0,
        skip_missing: bool = True,
    ) -> Dict[str, FredSeriesResult]:
        """
        Fetch several series in sequence. `inter_request_delay_seconds` is
        there for real live use against FRED's rate limits (120 requests/min
        on the official API) -- leave at 0 for cached/test runs.

        skip_missing (default True): if a series returns a 400/404 "does not
        exist" (FRED discontinues or renames series periodically -- this has
        happened with Wilshire and gold already), log a warning and SKIP it
        rather than crashing the entire run. The returned dict simply omits
        that series; downstream code already tolerates missing columns
        (the composite renormalizes over available flags, categories drop
        out gracefully). Set False to restore hard-fail behavior.
        """
        import sys as _sys
        results: Dict[str, FredSeriesResult] = {}
        ids = list(series_ids)
        for i, series_id in enumerate(ids):
            try:
                results[series_id] = self.fetch_series(
                    series_id, start_date=start_date, end_date=end_date, force_refresh=force_refresh
                )
            except (FredSeriesNotFoundError, FredAPIError) as exc:
                if skip_missing and _is_missing_series_error(exc):
                    print(
                        f"WARNING: FRED series '{series_id}' unavailable "
                        f"(likely discontinued/renamed) -- skipping it. "
                        f"Downstream code tolerates the omission.",
                        file=_sys.stderr,
                    )
                    continue
                raise
            if inter_request_delay_seconds and i < len(ids) - 1:
                self.sleep_fn(inter_request_delay_seconds)
        return results

    # ---- fetch implementations -----------------------------------------

    def _fetch_via_api(
        self, series_id: str, start_date: Optional[str], end_date: Optional[str]
    ) -> pd.Series:
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
        }
        if start_date:
            params["observation_start"] = start_date
        if end_date:
            params["observation_end"] = end_date

        status, body = self._request_with_retry(
            "https://api.stlouisfed.org/fred/series/observations", params
        )
        payload = json.loads(body)
        observations = payload.get("observations", [])
        dates, values = [], []
        for obs in observations:
            dates.append(obs["date"])
            raw_value = obs.get("value", ".")
            values.append(float("nan") if raw_value in (".", None, "") else float(raw_value))
        series = pd.Series(values, index=pd.to_datetime(dates), name=series_id).sort_index()
        return series

    def _fetch_via_csv(
        self, series_id: str, start_date: Optional[str], end_date: Optional[str]
    ) -> pd.Series:
        params = {"id": series_id}
        # fredgraph.csv accepts custom-observation-start/end-date params in
        # practice (cosd/coed); this isn't a versioned contract the way the
        # JSON API is, so treat it as best-effort and verify against a live
        # response before relying on it for a real backtest date range.
        if start_date:
            params["cosd"] = start_date
        if end_date:
            params["coed"] = end_date

        status, body = self._request_with_retry(
            "https://fred.stlouisfed.org/graph/fredgraph.csv", params
        )
        df = pd.read_csv(io.StringIO(body))
        # fredgraph.csv's value column is named after the series ID itself.
        date_col = df.columns[0]
        value_col = series_id if series_id in df.columns else df.columns[1]
        df[value_col] = pd.to_numeric(df[value_col], errors="coerce")  # "." -> NaN
        series = pd.Series(
            df[value_col].values, index=pd.to_datetime(df[date_col]), name=series_id
        ).sort_index()
        return series

    def _request_with_retry(self, url: str, params: Dict[str, str]) -> Tuple[int, str]:
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                status, body = self.transport(url, params)
            except Exception as exc:  # transport-level failure (timeout, DNS, etc.)
                last_error = exc
                self._backoff(attempt)
                continue

            if status == 404:
                raise FredSeriesNotFoundError(
                    f"FRED series not found (404) for params={_redact(params)}"
                )
            if status >= 500:
                last_error = FredAPIError(f"FRED returned {status} for params={_redact(params)}")
                self._backoff(attempt)
                continue
            if status >= 400:
                # 4xx other than 404 (e.g. bad/missing API key) -- not worth retrying,
                # since retrying an auth error just wastes the rate-limit budget.
                raise FredAPIError(f"FRED returned {status} for params={_redact(params)}: {body[:300]}")

            return status, body

        raise FredAPIError(
            f"FRED request failed after {self.max_retries} attempts for params={_redact(params)}"
        ) from last_error

    def _backoff(self, attempt: int) -> None:
        self.sleep_fn(self.backoff_base_seconds * (2**attempt))

    # ---- caching ---------------------------------------------------------

    def _cache_path(self, series_id: str, start_date: Optional[str], end_date: Optional[str]) -> Path:
        mode = "api" if self.api_key else "csv"
        key = f"{series_id}|{start_date}|{end_date}|{mode}"
        digest = hashlib.sha256(key.encode()).hexdigest()[:16]
        return self.cache_dir / f"{series_id}_{digest}.json"

    def _read_cache(self, path: Path) -> Optional[FredSeriesResult]:
        if not path.exists():
            return None
        payload = json.loads(path.read_text())
        fetched_at = pd.Timestamp(payload["fetched_at"])
        age_hours = (pd.Timestamp.utcnow() - fetched_at).total_seconds() / 3600
        if age_hours > self.cache_ttl_hours:
            return None
        series = pd.Series(
            payload["values"], index=pd.to_datetime(payload["dates"]), name=payload["series_id"]
        )
        return FredSeriesResult(
            series_id=payload["series_id"], series=series, source="cache", fetched_at=fetched_at
        )

    def _write_cache(self, path: Path, result: FredSeriesResult) -> None:
        payload = {
            "series_id": result.series_id,
            "dates": [d.strftime("%Y-%m-%d") for d in result.series.index],
            "values": [None if pd.isna(v) else v for v in result.series.values],
            "fetched_at": result.fetched_at.isoformat(),
        }
        path.write_text(json.dumps(payload))
