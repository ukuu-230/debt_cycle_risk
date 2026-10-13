#!/usr/bin/env python3
"""
check_sources.py -- confirm the free US Treasury sources respond and parse.

    python check_sources.py

No API key needed. Prints the latest values from each source, or the exact
error if Treasury has changed a layout. Run this if the dashboard's Treasury
rows go missing.
"""
from debt_cycle_scoring.ingestion.treasury_client import TreasuryClient, TreasuryDataError

c = TreasuryClient(cache_ttl_hours=0)
checks = [
    ("Average rate on marketable debt (%)", lambda: c.avg_rate_marketable("2024-01-01")),
    ("Bills share of marketable debt (%)", lambda: c.bills_share("2024-01-01")),
]
for name, fn in checks:
    try:
        s = fn()
        print(f"OK    {name}: {s.index[-1].date()} = {s.iloc[-1]:.2f}  ({len(s)} months)")
    except TreasuryDataError as exc:
        print(f"FAIL  {name}: {exc}")
try:
    t = c.tic_major_holders()
    print(f"OK    TIC Table 5: {t.index[0].date()} to {t.index[-1].date()}, rows: {', '.join(list(t.columns)[:6])} ...")
    for row in ("Japan", "China, Mainland", "Grand Total"):
        if row in t.columns:
            print(f"        {row:<16} latest ${t[row].iloc[-1]:,.1f}bn  (12-mo change {t[row].iloc[-1] - t[row].iloc[0]:+,.1f}bn)")
except TreasuryDataError as exc:
    print(f"FAIL  TIC Table 5: {exc}")
