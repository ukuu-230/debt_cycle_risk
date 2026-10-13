#!/usr/bin/env python3
"""
Run the debt_cycle_scoring ingestion pipeline against REAL FRED data.

This CANNOT be run inside the sandbox that built it (no outbound network
access there — confirmed via the egress proxy's own `x-deny-reason:
host_not_allowed` response). Run this on your own machine instead.

Setup:
    pip install requests pandas numpy scipy
    export FRED_API_KEY="your key here"        # macOS/Linux
    set FRED_API_KEY=your key here              # Windows (cmd)

    # Put the debt_cycle_scoring/ package (everything downloaded from
    # this conversation) on your PYTHONPATH, or run this script from the
    # parent directory that contains debt_cycle_scoring/.

Run:
    python3 run_live.py
    python3 run_live.py --start 1990-01-01 --out panel.csv

NEVER hardcode the key in this file or any file you commit/share --
reading it from the environment is deliberate, not a style choice: a key
typed directly into a script has a way of ending up committed to git or
pasted somewhere it shouldn't be.
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="1962-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD), default = today")
    parser.add_argument("--out", default=None, help="Optional CSV path to save the scored-metrics panel")
    args = parser.parse_args()

    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        print(
            "ERROR: FRED_API_KEY environment variable is not set.\n"
            "Get a free key at https://fredaccount.stlouisfed.org/apikeys and run:\n"
            "  export FRED_API_KEY='your key here'   (macOS/Linux)\n"
            "  set FRED_API_KEY=your key here          (Windows cmd)",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        from debt_cycle_scoring.ingestion import FredClient, build_quarterly_panel, merge_derived_metrics
    except ImportError:
        print(
            "ERROR: could not import debt_cycle_scoring. Make sure the package "
            "directory is on your PYTHONPATH or you're running this from its "
            "parent directory.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Fetching FRED data from {args.start} to {args.end or 'today'} ...")
    client = FredClient(api_key=api_key)

    try:
        panel = build_quarterly_panel(client, start_date=args.start, end_date=args.end)
    except Exception as exc:
        print(f"ERROR while fetching from FRED: {type(exc).__name__}: {exc}", file=sys.stderr)
        # The wrapper error above is often generic ("failed after 3 attempts") --
        # the ACTUAL cause (a connection error, SSL issue, real auth rejection,
        # missing `requests` package, etc.) is chained onto it. Print that too,
        # since without it there's no way to tell what actually went wrong.
        cause = exc.__cause__
        depth = 0
        while cause is not None and depth < 5:
            print(f"  caused by: {type(cause).__name__}: {cause}", file=sys.stderr)
            cause = cause.__cause__
            depth += 1
        print(
            "If this is an auth error, double check FRED_API_KEY is a valid, "
            "currently-active key (32 lowercase hex characters).",
            file=sys.stderr,
        )
        sys.exit(1)

    panel = merge_derived_metrics(panel)

    print()
    print(f"Scored metrics: {panel.scored_metrics.shape[0]} quarters x {panel.scored_metrics.shape[1]} metrics")
    print(f"  Date range: {panel.scored_metrics.index.min()} to {panel.scored_metrics.index.max()}")
    print(f"  Columns: {list(panel.scored_metrics.columns)}")
    print()
    print(f"Reference series: {list(panel.reference_series.columns)}")
    print(f"Label sources: {list(panel.label_sources.columns)}")
    print()

    print("Most recent quarter, all scored metrics:")
    print(panel.scored_metrics.tail(1).T.to_string())
    print()

    missing_pct = panel.scored_metrics.isna().mean().sort_values(ascending=False)
    print("Missing-data check (% NaN per metric, top 5 -- expect some NaN early")
    print("in the sample for shorter-history series like credit spreads, but")
    print("investigate anything unexpectedly high):")
    print(missing_pct.head(5).to_string())

    if args.out:
        panel.scored_metrics.to_csv(args.out)
        print(f"\nSaved scored-metrics panel to {args.out}")


if __name__ == "__main__":
    main()
