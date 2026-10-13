#!/usr/bin/env python3
"""
Cross-country credit-gap monitor -- run_credit_gaps.py

Fetches the BIS credit-to-GDP ratio for the US, Euro area, UK, and Japan
from FRED, computes each one's gap (same one-sided HP filter as the US
pipeline), and prints them side by side. This is the quant-honest form of
the multi-country work: the credit gap is the ONE cross-country signal
that carries information a US-only model lacks (credit cycles are
national), while avoiding the redundant globally-correlated flags a full
per-country composite was padding itself with.

Setup: FRED_API_KEY env var; requests/pandas/numpy/scipy installed.

Run:
    python run_credit_gaps.py
    python run_credit_gaps.py --start 1960-01-01 --out credit_gaps.csv
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd


def _to_quarterly(series: pd.Series) -> pd.Series:
    q = series.copy()
    q.index = q.index.to_period("Q")
    return q.groupby(level=0).last()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="1960-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--out", default=None, help="Optional CSV path for the wide gap panel")
    args = parser.parse_args()

    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        print("ERROR: FRED_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    try:
        from debt_cycle_scoring.ingestion import FredClient
        from debt_cycle_scoring.scorer.credit_gap_monitor import (
            build_credit_gap_monitor, CREDIT_RATIO_SERIES,
        )
    except ImportError as exc:
        print(f"ERROR: could not import debt_cycle_scoring ({exc}). Run from the project root.",
              file=sys.stderr)
        sys.exit(1)

    print(f"Fetching BIS credit-to-GDP ratios for {', '.join(CREDIT_RATIO_SERIES)} ...")
    client = FredClient(api_key=api_key)
    ratios = {}
    for country, sid in CREDIT_RATIO_SERIES.items():
        try:
            s = client.fetch_series(sid, start_date=args.start, end_date=args.end).series
            ratios[country] = _to_quarterly(s)
            print(f"  {country:6s} {sid}: {ratios[country].dropna().shape[0]} quarters")
        except Exception as exc:
            print(f"  WARNING: {country} ({sid}) failed to load -- skipping. ({type(exc).__name__})",
                  file=sys.stderr)

    if not ratios:
        print("ERROR: no credit-ratio data fetched.", file=sys.stderr)
        sys.exit(1)

    res = build_credit_gap_monitor(ratios)

    print("\n" + "=" * 70)
    print("CROSS-COUNTRY CREDIT-TO-GDP GAP  (latest reading per economy)")
    print("=" * 70)
    print("(gap = credit/GDP minus its one-sided HP trend, in percentage points.")
    print(" Basel III: >=2pp watch, >=10pp max-warning [crisis-associated],")
    print(" <=-2pp below-trend [post-deleveraging]. This is THE non-redundant")
    print(" cross-country signal -- credit cycles are national.)\n")
    print(f"   {'Country':<8}{'latest gap':<13}{'flag':<14}{'confidence':<12}{'as of':<9}{'quarters'}")
    for country, gap_val, flag, conf in res.ranking():
        c = res.countries[country]
        print(f"   {country:<8}{gap_val:<13}{flag:<14}{conf:<12}{str(c.latest_quarter):<9}{c.n_quarters}")

    print("\n   Reading: countries are ranked by current credit build-up. A")
    print("   positive/high gap = credit running above trend (risk building);")
    print("   negative = credit below trend (deleveraged). Compare where the")
    print("   US sits RELATIVE to peers -- that relative position is the signal.")
    print("   CONFIDENCE reflects how much history the HP trend rests on: 'low'")
    print("   (e.g. euro, ~69q) means the gap's DIRECTION is valid but its exact")
    print("   MAGNITUDE is soft -- don't over-read a low-confidence number.")

    # recent trajectory of the cross-country spread
    print("\n" + "=" * 70)
    print("RECENT GAP TRAJECTORY (last 8 quarters, pp)")
    print("=" * 70)
    if not res.gap_panel.empty:
        print(res.gap_panel.tail(8).round(1).to_string())

    if args.out:
        res.gap_panel.to_csv(args.out)
        print(f"\nSaved cross-country gap panel to {args.out}")


if __name__ == "__main__":
    main()
