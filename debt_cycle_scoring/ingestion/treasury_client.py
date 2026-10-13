"""
US Treasury data client -- ingestion/treasury_client.py

Free, keyless official sources for the parts of Dalio's bond-market framework
that FRED does not carry:

  * Fiscal Data API (api.fiscaldata.treasury.gov)
      - Average Interest Rates on U.S. Treasury Securities  (monthly, 2001+)
        -> the average rate the government pays on its EXISTING debt
      - Monthly Statement of the Public Debt, table 1 (MSPD) (monthly, 2001+)
        -> marketable debt outstanding by type (bills, notes, bonds, TIPS, FRNs)
  * Treasury International Capital (TIC) Table 5, "Major Foreign Holders of
    Treasury Securities" (ticdata.treasury.gov) -- tab-separated text, the
    latest 13 months only.

Same design as FredClient: injectable transport (tests pass a fake), disk
cache, retries with backoff, and every public fetch raises only
TreasuryDataError so callers can skip a source cleanly.

Field names and layouts were checked against the live endpoints on
2026-10-13; if Treasury changes them, the parser raises a clear error and the
dashboard drops just those rows.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

FISCAL_BASE = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/"
EP_AVG_RATES = "v2/accounting/od/avg_interest_rates"
EP_MSPD_1 = "v1/debt/mspd/mspd_table_1"
TIC_TABLE5_URL = "https://ticdata.treasury.gov/resource-center/data-chart-center/tic/Documents/slt_table5.txt"

Transport = Callable[[str, Dict[str, str]], Tuple[int, str]]
_UA = "debt-cycle-monitor/1.0 (research dashboard; github.com)"


class TreasuryDataError(RuntimeError):
    pass


def _default_transport(url: str, params: Dict[str, str]) -> Tuple[int, str]:
    import requests
    r = requests.get(url, params=params, timeout=45, headers={"User-Agent": _UA})
    return r.status_code, r.text


class TreasuryClient:
    def __init__(self, transport: Optional[Transport] = None, cache_dir: Optional[Path] = None,
                 cache_ttl_hours: float = 12.0, max_retries: int = 3,
                 sleep_fn: Callable[[float], None] = time.sleep):
        self.transport = transport or _default_transport
        self.cache_dir = cache_dir or (Path.home() / ".cache" / "debt_cycle_scoring" / "treasury")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_ttl_hours = cache_ttl_hours
        self.max_retries = max_retries
        self.sleep_fn = sleep_fn

    # ------------------------------------------------------------ plumbing
    def _get(self, url: str, params: Dict[str, str]) -> str:
        key = (url + "?" + "&".join(f"{k}={v}" for k, v in sorted(params.items())))
        cache = self.cache_dir / (hashlib.sha1(key.encode("utf-8")).hexdigest() + ".txt")
        if cache.exists() and (time.time() - cache.stat().st_mtime) < self.cache_ttl_hours * 3600:
            return cache.read_text(encoding="utf-8")
        last = None
        for attempt in range(self.max_retries):
            try:
                status, body = self.transport(url, params)
            except Exception as exc:              # network-level failure
                last = f"{type(exc).__name__}: {exc}"
            else:
                if status == 200:
                    cache.write_text(body, encoding="utf-8")
                    return body
                last = f"HTTP {status}"
                if 400 <= status < 500 and status != 429:
                    break                          # not retryable
            self.sleep_fn(1.5 * (2 ** attempt))
        raise TreasuryDataError(f"{url} failed: {last}")

    def _fiscal(self, endpoint: str, fields: List[str], filters: str = "") -> pd.DataFrame:
        """All pages of a Fiscal Data endpoint, oldest first."""
        rows, page = [], 1
        while True:
            params = {"fields": ",".join(fields), "sort": "record_date",
                      "page[size]": "10000", "page[number]": str(page)}
            if filters:
                params["filter"] = filters
            body = self._get(FISCAL_BASE + endpoint, params)
            try:
                payload = json.loads(body)
                rows.extend(payload["data"])
                total_pages = int(payload.get("meta", {}).get("total-pages", 1))
            except (ValueError, KeyError, TypeError) as exc:
                raise TreasuryDataError(f"{endpoint}: unexpected response layout ({exc})")
            if page >= total_pages:
                break
            page += 1
        if not rows:
            raise TreasuryDataError(f"{endpoint}: no rows returned")
        df = pd.DataFrame(rows)
        missing = [f for f in fields if f not in df.columns]
        if missing:
            raise TreasuryDataError(f"{endpoint}: fields missing from response: {missing}")
        df["record_date"] = pd.to_datetime(df["record_date"])
        return df

    # ------------------------------------------------------------ public series
    def avg_rate_marketable(self, start: str = "2001-01-01") -> pd.Series:
        """Monthly average interest rate (%) on ALL outstanding marketable Treasury debt."""
        df = self._fiscal(EP_AVG_RATES, ["record_date", "security_type_desc", "security_desc",
                                         "avg_interest_rate_amt"], f"record_date:gte:{start}")
        m = df[(df["security_type_desc"] == "Marketable") & (df["security_desc"] == "Total Marketable")]
        if m.empty:
            raise TreasuryDataError("avg_interest_rates: no 'Total Marketable' rows")
        s = pd.to_numeric(m.set_index("record_date")["avg_interest_rate_amt"], errors="coerce")
        return s[~s.index.duplicated(keep="last")].sort_index().dropna()

    def marketable_by_type(self, start: str = "2001-01-01") -> pd.DataFrame:
        """Monthly marketable debt outstanding ($mn) by class, plus a 'Total' column."""
        df = self._fiscal(EP_MSPD_1, ["record_date", "security_type_desc", "security_class_desc",
                                      "total_mil_amt"], f"record_date:gte:{start}")
        df["total_mil_amt"] = pd.to_numeric(df["total_mil_amt"], errors="coerce")
        mk = df[df["security_type_desc"] == "Marketable"]
        wide = mk.pivot_table(index="record_date", columns="security_class_desc",
                              values="total_mil_amt", aggfunc="last")
        tot = df[df["security_type_desc"] == "Total Marketable"].set_index("record_date")["total_mil_amt"]
        wide["Total"] = tot[~tot.index.duplicated(keep="last")]
        if "Bills" not in wide.columns or wide["Total"].dropna().empty:
            raise TreasuryDataError("mspd_table_1: 'Bills' or 'Total Marketable' rows missing")
        return wide.sort_index()

    def bills_share(self, start: str = "2001-01-01") -> pd.Series:
        """Bills as % of marketable debt outstanding (monthly)."""
        w = self.marketable_by_type(start)
        return (w["Bills"] / w["Total"] * 100.0).dropna()

    def tic_major_holders(self) -> pd.DataFrame:
        """TIC Table 5 (latest 13 months), $bn: index = month-end, columns = country/row label."""
        body = self._get(TIC_TABLE5_URL, {})
        return parse_tic_table5(body)


def parse_tic_table5(text: str) -> pd.DataFrame:
    """
    Parse TIC Table 5 text: a header line starting 'Country' followed by
    tab-separated YYYY-MM labels, then one tab-separated row per holder.
    """
    lines = [ln.rstrip("\r") for ln in text.splitlines()]
    try:
        hdr_i = next(i for i, ln in enumerate(lines) if ln.split("\t")[0].strip().lower() == "country")
    except StopIteration:
        raise TreasuryDataError("TIC table 5: header row 'Country' not found (layout changed?)")
    months = [c.strip() for c in lines[hdr_i].split("\t")[1:] if c.strip()]
    try:
        idx = [pd.Period(m, "M").to_timestamp("M") for m in months]
    except Exception:
        raise TreasuryDataError(f"TIC table 5: unexpected month labels {months[:3]}")
    data: Dict[str, List[float]] = {}
    for ln in lines[hdr_i + 1:]:
        parts = ln.split("\t")
        if len(parts) < 2 or not parts[0].strip():
            continue
        vals = []
        for v in parts[1:len(months) + 1]:
            try:
                vals.append(float(v.replace(",", "")))
            except ValueError:
                vals.append(float("nan"))
        if len(vals) == len(months):
            data[parts[0].strip()] = vals
    if not data:
        raise TreasuryDataError("TIC table 5: no data rows parsed")
    return pd.DataFrame(data, index=idx).sort_index()


def fetch_treasury_extras(client: TreasuryClient, log: Callable[[str], None] = print) -> Dict[str, pd.Series]:
    """
    Everything the Dalio context panel needs from Treasury, as monthly Series
    keyed by pseudo-IDs. Any source that fails is skipped with a warning.
    """
    out: Dict[str, pd.Series] = {}
    try:
        out["TSY_AVG_RATE_MKT"] = client.avg_rate_marketable()
    except TreasuryDataError as exc:
        log(f"  WARNING: Treasury average-rate data unavailable -- skipped ({exc})")
    try:
        out["TSY_BILLS_SHARE"] = client.bills_share()
    except TreasuryDataError as exc:
        log(f"  WARNING: Treasury debt-by-type data unavailable -- skipped ({exc})")
    try:
        tic = client.tic_major_holders()
        jc = [c for c in tic.columns if c.lower().startswith("japan") or c.lower().startswith("china, mainland")]
        if len(jc) == 2:
            out["TIC_JP_CN"] = tic[jc].sum(axis=1)
        off = [c for c in tic.columns if "foreign official" in c.lower() and "bill" not in c.lower()
               and "bond" not in c.lower()]
        if off:
            out["TIC_OFFICIAL"] = tic[off[0]]
        if "TIC_JP_CN" not in out and "TIC_OFFICIAL" not in out:
            log("  WARNING: TIC table 5 parsed but Japan/China/official rows not found -- skipped")
    except TreasuryDataError as exc:
        log(f"  WARNING: TIC foreign-holder data unavailable -- skipped ({exc})")
    return out
