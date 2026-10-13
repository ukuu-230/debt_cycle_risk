#!/usr/bin/env python3
"""
Multi-country composite risk read -- run_country.py

Fetches a single country's risk-flag inputs from FRED, computes its
credit-to-GDP gap, and prints the composite risk read for that country
using the SAME research-weighted flag methodology as the US, with flags
dropping out honestly wherever that country lacks a clean series.

Setup: same as run_scorer.py (FRED_API_KEY env var; requests/pandas/
numpy/scipy installed).

Run:
    python run_country.py --country Euro
    python run_country.py --country Euro --start 1970-01-01 --out euro_risk.csv
    python run_country.py --list          # show available countries
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd


def _to_quarterly(series: pd.Series, how: str = "last") -> pd.Series:
    q = series.copy()
    q.index = q.index.to_period("Q")
    g = q.groupby(level=0)
    return g.last() if how == "last" else g.mean()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--country", default="Euro", help="Country key (see --list)")
    parser.add_argument("--start", default="1970-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--out", default=None, help="Optional CSV path for the per-quarter risk")
    parser.add_argument("--min-headline-flags", type=int, default=3,
                        help="Min flags for the headline quarter (default 3; non-US has fewer flags)")
    parser.add_argument("--list", action="store_true", help="List available countries and exit")
    args = parser.parse_args()

    try:
        from debt_cycle_scoring.ingestion.intl_catalog import (
            INTL_CATALOG, country_series_ids, available_countries,
        )
        from debt_cycle_scoring.ingestion import FredClient
        from debt_cycle_scoring.labeling import compute_credit_gap
        from debt_cycle_scoring.scorer import build_country_risk_series, FLAG_NAMES
    except ImportError as exc:
        print(f"ERROR: could not import debt_cycle_scoring ({exc}). Run from the project root.",
              file=sys.stderr)
        sys.exit(1)

    if args.list:
        print("Available countries:")
        for c in available_countries():
            ids = country_series_ids(c)
            present = [k for k, v in ids.items() if v is not None]
            print(f"   {c:8s} {len(present)} flag-inputs available: {', '.join(present)}")
        return

    if args.country not in available_countries():
        print(f"ERROR: unknown country '{args.country}'. Available: {available_countries()}",
              file=sys.stderr)
        sys.exit(1)

    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        print("ERROR: FRED_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    ids = country_series_ids(args.country)
    ids_to_fetch = {k: v for k, v in ids.items() if v is not None}
    print(f"Fetching {args.country} data from FRED "
          f"({len(ids_to_fetch)} series: {', '.join(ids_to_fetch.values())}) ...")

    client = FredClient(api_key=api_key)
    fetched = {}
    for flag_key, series_id in ids_to_fetch.items():
        try:
            fetched[flag_key] = client.fetch_series(series_id, start_date=args.start, end_date=args.end).series
        except Exception as exc:
            print(f"  WARNING: {series_id} ({flag_key}) unavailable -- skipping. ({type(exc).__name__})",
                  file=sys.stderr)

    # credit gap from the ratio series (one-sided HP filter, same as US)
    credit_gap = None
    if "credit_gap_ratio" in fetched:
        ratio_q = _to_quarterly(fetched["credit_gap_ratio"]).dropna()
        credit_gap = compute_credit_gap(ratio_q, min_periods=40)["gap"]
        n_gap = int(credit_gap.notna().sum()) if credit_gap is not None else 0
        if n_gap == 0:
            print("  WARNING: credit-gap ratio fetched but produced 0 usable gap quarters "
                  "(needs 40+ quarters of history for the HP-filter burn-in).", file=sys.stderr)
        else:
            print(f"  credit-gap flag: {n_gap} quarters available "
                  f"(from {credit_gap.dropna().index.min()})")
    else:
        # This is the most important flag -- say so loudly if it's absent.
        print("  WARNING: credit-to-GDP gap (the top-weighted flag) did NOT load for this "
              "country -- the composite will rest on weaker flags only.", file=sys.stderr)

    # assemble the country-agnostic input dict (quarterly)
    inputs = {}
    if "debt_service_ratio" in fetched:
        inputs["dsr"] = _to_quarterly(fetched["debt_service_ratio"])
    if "credit_spread" in fetched:
        inputs["credit_spread"] = _to_quarterly(fetched["credit_spread"], how="mean")
    if "m2_velocity" in fetched:
        inputs["m2_velocity"] = _to_quarterly(fetched["m2_velocity"])
    if "long_bond_10y" in fetched:
        inputs["long_bond"] = _to_quarterly(fetched["long_bond_10y"])
    if "breakeven_inflation" in fetched:
        inputs["breakeven"] = _to_quarterly(fetched["breakeven_inflation"])
    # yield curve = 10y - short, only if BOTH present
    if "long_bond_10y" in fetched and "short_rate" in fetched:
        lb = _to_quarterly(fetched["long_bond_10y"])
        sr = _to_quarterly(fetched["short_rate"])
        inputs["curve"] = (lb - sr).dropna()

    if credit_gap is None and not inputs:
        print("ERROR: no usable data fetched for this country.", file=sys.stderr)
        sys.exit(1)

    result = build_country_risk_series(inputs, credit_gap if credit_gap is not None else pd.Series(dtype=float))

    # headline: latest quarter clearing the min-flags bar
    valid = result.risk.dropna()
    eligible = [q for q in result.n_flags[result.n_flags >= args.min_headline_flags].index if q in valid.index]
    headline = max(eligible) if eligible else valid.index.max()

    print("\n" + "=" * 70)
    print(f"{args.country.upper()} COMPOSITE RISK READ  ({headline})   "
          f"[latest quarter with >= {args.min_headline_flags} flags]")
    print("=" * 70)
    print(f"\n   RISK SCORE: {result.risk.loc[headline]:.0f} / 100   "
          f"(research-weighted, regime: {result.regime.loc[headline]})")
    print(f"   equal-weight cross-check: {result.risk_equal.loc[headline]:.0f} / 100   "
          f"(divergence {result.divergence.loc[headline]:.0f})")
    print(f"\n   Flags feeding this read ({int(result.n_flags.loc[headline])} of 7 present):")
    labels = {
        "credit_gap": "Credit-to-GDP gap (BIS)", "debt_service_ratio": "Debt service ratio (BIS)",
        "yield_curve": "Yield curve", "credit_spread": "Credit spreads",
        "m2_velocity": "M2 velocity", "long_bond_level": "Long-bond level",
        "debasement": "Debasement/deflation",
    }
    for flag, val in result.flags.loc[headline].dropna().sort_values(ascending=False).items():
        print(f"      {labels.get(flag, flag):32s} {val:5.0f}")
    absent = [labels.get(f, f) for f in FLAG_NAMES if pd.isna(result.flags.loc[headline].get(f))]
    if absent:
        print(f"\n   Flags NOT available for {args.country} (dropped, composite renormalized):")
        print(f"      {', '.join(absent)}")

    print("\n" + "=" * 70)
    print("RISK TRAJECTORY (last 12 quarters)")
    print("=" * 70)
    traj = pd.DataFrame({"risk": result.risk, "equal_wt": result.risk_equal,
                         "regime": result.regime, "n_flags": result.n_flags}).tail(12)
    print(traj.to_string())

    print("\n" + "=" * 70)
    print("FULL-HISTORY RISK SUMMARY")
    print("=" * 70)
    print(f"   mean {valid.mean():.0f}, median {valid.median():.0f}, "
          f"min {valid.min():.0f}, max {valid.max():.0f}")
    print("   highest-risk quarters on record:")
    for q, v in valid.sort_values(ascending=False).head(5).items():
        print(f"      {q}   {v:.0f}")

    if args.out:
        out = pd.DataFrame({"risk_score": result.risk, "risk_equal_weight": result.risk_equal,
                            "regime": result.regime, "n_flags": result.n_flags})
        for flag in FLAG_NAMES:
            if flag in result.flags.columns:
                out[f"flag_{flag}"] = result.flags[flag]
        out.to_csv(args.out)
        print(f"\nSaved {args.country} per-quarter risk to {args.out}")


if __name__ == "__main__":
    main()
