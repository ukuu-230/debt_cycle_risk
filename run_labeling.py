#!/usr/bin/env python3
"""
Run the section 3B independent labeling pipeline against REAL FRED data --
the first time NBER extraction, the BIS-style credit-to-GDP gap, and
Bry-Boschan mechanical turning points get tested against actual US
history rather than the synthetic episode in labeling/demo.py.

Setup: same as run_live.py (FRED_API_KEY environment variable, requests/
pandas/numpy/scipy installed).

Run:
    python run_labeling.py
    python run_labeling.py --start 1962-01-01 --out labels.csv
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
    parser.add_argument("--out", default=None, help="Optional CSV path to save the consensus + stage labels")
    args = parser.parse_args()

    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        print(
            "ERROR: FRED_API_KEY environment variable is not set.\n"
            "Get a free key at https://fredaccount.stlouisfed.org/apikeys",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        from debt_cycle_scoring.ingestion import FredClient, build_quarterly_panel, merge_derived_metrics
        from debt_cycle_scoring.labeling import (
            extract_recession_turning_points,
            compute_credit_gap,
            bry_boschan_turning_points,
            build_consensus_labels,
        )
        from debt_cycle_scoring.labeling.consensus import stage_labels_from_consensus
    except ImportError as exc:
        print(
            f"ERROR: could not import debt_cycle_scoring ({exc}). Make sure the "
            "package directory is on your PYTHONPATH or you're running this "
            "from its parent directory.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Fetching FRED data from {args.start} to {args.end or 'today'} ...")
    client = FredClient(api_key=api_key)

    try:
        panel = build_quarterly_panel(client, start_date=args.start, end_date=args.end)
    except Exception as exc:
        print(f"ERROR while fetching from FRED: {type(exc).__name__}: {exc}", file=sys.stderr)
        cause = exc.__cause__
        depth = 0
        while cause is not None and depth < 5:
            print(f"  caused by: {type(cause).__name__}: {cause}", file=sys.stderr)
            cause = cause.__cause__
            depth += 1
        sys.exit(1)

    panel = merge_derived_metrics(panel)

    if "NBER recession indicator" not in panel.label_sources.columns:
        print("ERROR: 'NBER recession indicator' not found in panel.label_sources.", file=sys.stderr)
        sys.exit(1)
    if "US Credit-to-GDP ratio (BIS, via FRED)" not in panel.label_sources.columns:
        print("ERROR: BIS credit-to-GDP series not found in panel.label_sources.", file=sys.stderr)
        sys.exit(1)

    usrec = panel.label_sources["NBER recession indicator"]
    credit_ratio = panel.label_sources["US Credit-to-GDP ratio (BIS, via FRED)"].dropna()

    print(f"\nUSREC covers {usrec.index.min()} to {usrec.index.max()}")
    print(f"BIS credit-to-GDP ratio covers {credit_ratio.index.min()} to {credit_ratio.index.max()}")

    # --- Source A ---
    print("\n=== Source A: NBER turning points ===")
    nber_events = extract_recession_turning_points(usrec)
    print(f"{len(nber_events)} turning points found (peaks + troughs).")
    print(nber_events.to_string())

    # --- Source B ---
    print("\n=== Source B: BIS-style credit-to-GDP gap ===")
    print("(min_periods=40, i.e. ~10 years of trailing data required before this reports anything --")
    print(" matches BIS's own stated rule of thumb, see design doc section 3A.6 / labeling/hp_filter.py)")
    gap_df = compute_credit_gap(credit_ratio, min_periods=40)
    print("\nMost recent 8 quarters:")
    print(gap_df.tail(8)[["ratio", "trend", "gap", "flag", "declining_from_peak"]].to_string())

    # --- Source C ---
    print("\n=== Source C: Bry-Boschan mechanical turning points (on the credit-to-GDP ratio) ===")
    print("(smooth_window=5: light smoothing before detection, added after finding the unsmoothed")
    print(" version fired a spurious 'peak' right at the real 2009Q2 trough on real data)")
    mechanical_events = bry_boschan_turning_points(credit_ratio, window=2, min_phase_quarters=4, smooth_window=5)
    print(f"{len(mechanical_events)} turning points found.")
    print(mechanical_events.to_string())

    # --- Section 3B.4 consensus ---
    print("\n=== Section 3B.4: consensus combiner ===")
    consensus = build_consensus_labels(
        panel.scored_metrics.index, nber_events, gap_df, mechanical_events, agreement_window=4
    )

    high_conf = consensus[consensus["tier"] == "high_confidence"]
    print(f"\nHigh-confidence events: {len(high_conf)}")
    if len(high_conf):
        print(high_conf[["label"]].to_string())

    contested = consensus[consensus["tier"] == "contested"]
    print(f"\nContested (recession WITHOUT prior credit-gap buildup -- i.e. likely non-debt-cycle) periods: {len(contested)}")
    if len(contested):
        print(contested[["label"]].to_string())

    # --- Sanity check against episodes we know the answer to ---
    print("\n=== Sanity check against known history ===")
    print("(this is the real test -- does the pipeline's read match what actually happened)")
    checkpoints = {
        "2001Q1": "dot-com bust -- weak credit-cycle signature expected; may not reach high-confidence",
        "2007Q4": "official NBER peak before the GFC -- expect a high-confidence top nearby",
        "2009Q2": "official NBER trough after the GFC -- expect a trough signal nearby",
        "2020Q1": "COVID recession peak -- expect CONTESTED (non-debt-cycle), not a 'top'",
    }
    for q, description in checkpoints.items():
        period = pd.Period(q, freq="Q")
        if period in consensus.index:
            row = consensus.loc[period]
            print(f"  {q} ({description})")
            print(
                f"       -> tier={row['tier']}, label={row['label']}, "
                f"credit_gap_flag={row['credit_gap_flag']}, declining={row['credit_gap_declining']}"
            )
        else:
            print(f"  {q}: not in range of fetched data")

    # --- Full STAGES-vocabulary labels ---
    if "Real GDP" in panel.reference_series.columns:
        real_gdp_growth = panel.reference_series["Real GDP"].pct_change(fill_method=None) * 100
        stage_labels, train_ok = stage_labels_from_consensus(
            consensus, real_gdp_qoq_growth=real_gdp_growth, return_training_mask=True
        )
        print("\n=== Resulting STAGES-vocabulary label distribution (ALL quarters, for the timeline) ===")
        print(stage_labels.value_counts(dropna=False).to_string())

        print("\n=== Training-eligible subset (high-confidence quarters ONLY -- what the scorer will learn from) ===")
        print("(conflicting/contested/single-source/silent quarters are shown in the timeline above")
        print(" but deliberately EXCLUDED here, so the scorer never trains on periods the sources")
        print(" disagreed about -- e.g. the 2009Q2 GFC-trough conflict)")
        print(f"Training-eligible quarters: {int(train_ok.sum())} of {len(stage_labels)} total")
        print(stage_labels[train_ok].value_counts(dropna=False).to_string())
    else:
        stage_labels = None
        train_ok = None
        print("\n(Real GDP not found in reference_series -- skipping STAGES label assignment)")

    if args.out:
        out_df = consensus.copy()
        if stage_labels is not None:
            out_df["stage_label"] = stage_labels
            out_df["training_eligible"] = train_ok
        out_df.to_csv(args.out)
        print(f"\nSaved consensus + stage labels (with training_eligible flag) to {args.out}")


if __name__ == "__main__":
    main()
