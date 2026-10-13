#!/usr/bin/env python3
"""
Run the FULL debt-cycle scorer against real FRED data and print the
combined report: current stage read + risk score, recent trajectory,
and the full historical stage/risk timeline.

This is the end-to-end product: ingestion -> derived metrics -> labeling
-> training mask -> Mahalanobis scorer -> stage probabilities + 0-100
risk score.

Setup: same as run_live.py (FRED_API_KEY env var; requests/pandas/numpy/
scipy installed).

Run:
    python run_scorer.py
    python run_scorer.py --start 1962-01-01 --out scores.csv
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="1962-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--out", default=None, help="Optional CSV path for the full per-quarter scores")
    parser.add_argument("--trajectory-quarters", type=int, default=12, help="How many recent quarters to show")
    parser.add_argument("--min-headline-flags", type=int, default=4,
                        help="Minimum flags a quarter needs to be eligible as the headline read "
                             "(recent quarters with fewer flags are shown as provisional, not headlined)")
    parser.add_argument("--backtest", action="store_true",
                        help="Also run the walk-forward out-of-sample backtest + calibration report (slower)")
    args = parser.parse_args()

    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        print("ERROR: FRED_API_KEY environment variable is not set.", file=sys.stderr)
        print("Get a free key at https://fredaccount.stlouisfed.org/apikeys", file=sys.stderr)
        sys.exit(1)

    try:
        from debt_cycle_scoring.ingestion import FredClient, build_quarterly_panel, merge_derived_metrics
        from debt_cycle_scoring.labeling import (
            extract_recession_turning_points, compute_credit_gap,
            bry_boschan_turning_points, build_consensus_labels,
        )
        from debt_cycle_scoring.labeling.consensus import stage_labels_from_consensus
        from debt_cycle_scoring.scorer import fit_and_score
    except ImportError as exc:
        print(f"ERROR: could not import debt_cycle_scoring ({exc}).", file=sys.stderr)
        print("Run this from the project root (the folder CONTAINING debt_cycle_scoring).", file=sys.stderr)
        sys.exit(1)

    # --- 1. ingest ---
    print(f"Fetching FRED data {args.start} to {args.end or 'today'} ...")
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

    # --- 2. labeling ---
    usrec = panel.label_sources["NBER recession indicator"]
    credit_ratio = panel.label_sources["US Credit-to-GDP ratio (BIS, via FRED)"].dropna()
    nber_events = extract_recession_turning_points(usrec)
    gap_df = compute_credit_gap(credit_ratio, min_periods=40)
    mechanical_events = bry_boschan_turning_points(credit_ratio, window=2, min_phase_quarters=4, smooth_window=5)
    consensus = build_consensus_labels(panel.scored_metrics.index, nber_events, gap_df, mechanical_events, agreement_window=4)
    real_gdp_growth = panel.reference_series["Real GDP"].pct_change(fill_method=None) * 100
    stage_labels, train_ok = stage_labels_from_consensus(
        consensus, real_gdp_qoq_growth=real_gdp_growth, return_training_mask=True
    )

    # --- 3. compute the COMPOSITE RISK SCORE (the headline product) ---
    from debt_cycle_scoring.scorer import build_risk_series, FLAG_NAMES
    credit_gap_series = gap_df["gap"]
    risk = build_risk_series(
        panel.scored_metrics, panel.reference_series, credit_gap_series,
    )

    # --- 3b. stage classifier (SECONDARY context only -- backtest showed it
    #         does not beat baseline, so it is a footnote, not the product) ---
    stage_result = None
    try:
        stage_result = fit_and_score(
            panel.scored_metrics, panel.metric_specs, stage_labels, train_ok,
            credit_gap=credit_gap_series, country="US",
        )
    except Exception as exc:
        print(f"(stage classifier context unavailable: {exc})", file=sys.stderr)

    # --- 4. HEADLINE: latest WELL-EVIDENCED composite risk read ---
    # Recent quarters lose flags to reporting lag (quarterly series like the
    # BIS credit gap, DSR, and M2 velocity publish 1-2 quarters behind the
    # daily curve/spread/breakeven series). The most RECENT quarter is therefore
    # systematically the WORST-evidenced -- so the headline uses the latest
    # quarter that clears --min-headline-flags, and shows the bleeding-edge
    # quarters separately as provisional.
    valid_risk = risk.risk.dropna()
    from debt_cycle_scoring.scorer.risk_flags import headline_quarter
    headline_q, required_flags = headline_quarter(risk, args.min_headline_flags)
    if headline_q is None:
        headline_q, required_flags = valid_risk.index.max(), args.min_headline_flags

    latest_q = valid_risk.index.max()

    print("\n" + "=" * 70)
    print(f"COMPOSITE RISK READ  ({headline_q})   [latest quarter with >= "
          f"{required_flags} flags = full recent set]")
    print("=" * 70)
    print(f"\n   Curve input: {risk.curve_source}.  Spread input: {risk.spread_source}.  Publication lags applied (quarters): "
          f"{', '.join(f'{k} {v}' for k, v in risk.publication_lags.items()) or 'none'}")
    print(f"\n   RISK SCORE: {risk.risk.loc[headline_q]:.0f} / 100   "
          f"(research-weighted, regime: {risk.regime.loc[headline_q]})")
    print(f"   equal-weight cross-check: {risk.risk_equal.loc[headline_q]:.0f} / 100   "
          f"(divergence {risk.divergence.loc[headline_q]:.0f})")
    if risk.divergence.loc[headline_q] > 15:
        print("   !! research- and equal-weighted composites diverge notably -- the")
        print("      weighting is doing real work here; inspect the per-flag breakdown.")
    print(f"\n   Flags feeding this read ({int(risk.n_flags.loc[headline_q])} present, 0-100 each, higher = more risk):")
    flag_labels = {
        "credit_gap": "Credit-to-GDP gap (BIS)",
        "debt_service_ratio": "Debt service ratio",
        "yield_curve": "Yield curve (10y-3m)",
        "credit_spread": "Credit spread (Baa-10y)",
        "m2_velocity": "M2 velocity yoy (falling=risk)",
        "long_bond_level": "Long-bond yield vs 10y window",
        "debasement": "Debasement/deflation (breakeven infl.)",
    }
    headline_flags = risk.flags.loc[headline_q].dropna().sort_values(ascending=False)
    for flag, val in headline_flags.items():
        print(f"      {flag_labels.get(flag, flag):40s} {val:5.0f}")

    # --- input coverage: catches a series FRED has truncated or delisted ---
    print(f"\n   Input coverage (quarters where each flag has a value):")
    cs = risk.context_flags["credit_spread"].dropna() if "credit_spread" in risk.context_flags.columns else pd.Series(dtype=float)
    print(f"      Credit spread (context only, not weighted): {risk.spread_source}, "
          f"{len(cs)} quarters" + (f", latest {cs.iloc[-1]:.0f}" if len(cs) else ""))
    for flag in FLAG_NAMES:
        fs = risk.flags[flag].dropna() if flag in risk.flags.columns else pd.Series(dtype=float)
        if len(fs):
            short = "   <-- SHORT: check the FRED series" if len(fs) < 80 else ""
            print(f"      {flag_labels.get(flag, flag):40s} {str(fs.index.min()):>7} to {str(fs.index.max()):<7} n={len(fs):<4}{short}")
        else:
            print(f"      {flag_labels.get(flag, flag):40s}    NO DATA   <-- flag absent from every quarter")

    if headline_q != latest_q:
        print(f"\n   PROVISIONAL (more recent, but thin evidence -- quarterly series like the")
        print(f"   credit gap/DSR/M2V lag the daily flags, so these rest on fewer inputs):")
        prov = risk.risk.loc[[q for q in risk.risk.index if q > headline_q]].dropna()
        for q in prov.index:
            print(f"      {q}   risk {prov.loc[q]:.0f}   ({int(risk.n_flags.loc[q])} flags: "
                  f"{', '.join(risk.flags.loc[q].dropna().index)})")

    # --- 5. risk trajectory (the thing that actually matters) ---
    print("\n" + "=" * 70)
    print(f"RISK TRAJECTORY  (last {args.trajectory_quarters} quarters)")
    print("=" * 70)
    print("(quarterly macro inputs are shown in the quarter they were PUBLISHED, so")
    print(" recent quarters normally carry the full flag set; compare equal n_flags)")
    traj = pd.DataFrame({
        "risk": risk.risk,
        "equal_wt": risk.risk_equal,
        "regime": risk.regime,
        "n_flags": risk.n_flags,
    }).tail(args.trajectory_quarters)
    print(traj.to_string())

    # --- 5b. SHORT-HORIZON market-stress sub-score (SEPARATE from the ---
    #         slow structural risk above -- fast/acute stress, not buildup) ---
    from debt_cycle_scoring.scorer import build_short_horizon_score
    # Diagnostic: show which short-horizon INPUTS actually loaded, so a
    # missing VIX/dollar (e.g. skipped at fetch, or catalog not updated) is
    # visible rather than silently leaving the sub-score on 2 flags.
    _sh_inputs = {
        "VIX (VIXCLS)": "VIX equity volatility" in panel.short_horizon.columns,
        "Dollar (DTWEXBGS)": "Broad nominal US dollar index" in panel.short_horizon.columns,
        "Credit spread (in scored)": "ICE BofA US High Yield OAS (credit spread)" in panel.scored_metrics.columns,
        "Curve (in scored)": "10y-2y Treasury spread" in panel.scored_metrics.columns,
        "NFCI (validation)": "Chicago Fed NFCI (validation cross-check)" in panel.short_horizon_validation.columns,
    }
    _missing = [k for k, present in _sh_inputs.items() if not present]
    if _missing:
        print(f"\n(note: short-horizon inputs NOT loaded: {', '.join(_missing)} -- "
              f"if VIX/Dollar are here, check they fetched and the catalog is updated)",
              file=sys.stderr)

    sh = build_short_horizon_score(
        panel.short_horizon, panel.short_horizon_worst,
        panel.scored_metrics, validation=panel.short_horizon_validation,
        reference_series=panel.reference_series,
    )
    sh_valid = sh.stress.dropna()
    if len(sh_valid) > 0:
        sh_latest = sh_valid.index.max()
        print("\n" + "=" * 70)
        print(f"SHORT-HORIZON MARKET-STRESS SUB-SCORE  ({sh_latest})")
        print("=" * 70)
        print("(SEPARATE gauge -- fast/acute market stress, NOT the slow structural")
        print(" credit-cycle risk above. Built from raw components with robust")
        print(" median/MAD normalization; NFCI/STLFSI used only as external cross-checks.)")
        print(f"\n   STRESS: {sh.stress.loc[sh_latest]:.0f} / 100   "
              f"(quarter-average; peak-that-quarter: {sh.stress_worst.loc[sh_latest]:.0f})")
        sh_flags = sh.flags.loc[sh_latest].dropna().sort_values(ascending=False)
        sh_labels = {
            "vix_level": "VIX level", "vix_change": "VIX spike (change)",
            "dollar_surge": "Dollar surge (YoY)", "spread_level": "Credit-spread level (Baa-10y)",
            "spread_widening": "Credit-spread widening",
            "curve_change": "Curve flattening (change)",
        }
        print(f"\n   Component flags (0-100, higher = more stress):")
        for flag, val in sh_flags.items():
            print(f"      {sh_labels.get(flag, flag):32s} {val:5.0f}")
        if not sh.validation.empty and sh_latest in sh.validation.index:
            vrow = sh.validation.loc[sh_latest].dropna()
            if len(vrow):
                print(f"\n   External cross-checks (pro composites, NOT blended in):")
                for name, val in vrow.items():
                    short_name = name.split("(")[0].strip()
                    print(f"      {short_name:40s} {val:5.0f}")
                div = sh.validation_divergence.get(sh_latest, float('nan'))
                if not pd.isna(div):
                    print(f"      divergence from our sub-score: {div:.0f}"
                          + ("   (LARGE -- investigate)" if div > 25 else "   (in agreement)"))


    print("\n" + "=" * 70)
    print("FULL-HISTORY RISK SUMMARY")
    print("=" * 70)
    print(f"   mean {valid_risk.mean():.0f}, median {valid_risk.median():.0f}, "
          f"min {valid_risk.min():.0f}, max {valid_risk.max():.0f}")
    # Rank only quarters with a comparable flag set: a 1960s quarter scored on
    # 3-4 flags is not comparable with a 2007 quarter scored on 7.
    comparable = valid_risk[risk.n_flags.reindex(valid_risk.index) >= 5]
    print(f"   highest-risk quarters on record (quarters with >= 5 flags, n={len(comparable)}):")
    for q, v in comparable.sort_values(ascending=False).head(5).items():
        print(f"      {q}   {v:.0f}   ({int(risk.n_flags.loc[q])} flags)")

    # --- 7. stage read: SECONDARY CONTEXT ONLY (footnote) ---
    if stage_result is not None and headline_q in stage_result.top_stage.index:
        print("\n" + "-" * 70)
        print("stage context (SECONDARY -- backtest showed the stage classifier does")
        print("not beat a base-rate baseline, so treat this as loose color, NOT the")
        print("risk signal above):")
        st = stage_result.top_stage.loc[headline_q]
        print(f"   nearest stage archetype: {st}")
        print("   ('top' is intentionally not modellable; a real top reads as high-bubble)")


    # --- 7b. optional walk-forward backtest + calibration ---
    if args.backtest:
        from debt_cycle_scoring.scorer.validation import walk_forward_predict, calibration_report
        print("\n" + "=" * 70)
        print("WALK-FORWARD OUT-OF-SAMPLE BACKTEST")
        print("=" * 70)
        print("(refits using ONLY data before each quarter, predicts that quarter unseen,")
        print(" rolls forward -- this is the honest test that the model isn't circular)")
        try:
            wf = walk_forward_predict(
                panel.scored_metrics, panel.metric_specs, stage_labels, train_ok,
                credit_gap=credit_gap_series, min_train_per_start=15, refit_every=2,
            )
            cal = calibration_report(wf, n_bins=5)
            print(f"\nOut-of-sample quarters predicted: {len(wf.predictions)} "
                  f"(from {wf.first_scored})")
            print(f"Evaluated against finalized labels: {cal.n_evaluated} quarters")
            print(f"\nTop-1 accuracy: {cal.top1_accuracy:.1%}")
            print("Per-stage accuracy:")
            for stage, acc in sorted(cal.per_stage_accuracy.items(), key=lambda kv: -kv[1]):
                print(f"   {stage:24s} {acc:.1%}")
            print(f"\nMulticlass Brier score: {cal.brier_score:.3f}  "
                  f"(baseline / always-predict-base-rate: {cal.baseline_brier:.3f})")
            if cal.brier_score < cal.baseline_brier:
                print("   -> model beats the base-rate baseline (its probabilities add information)")
            else:
                print("   -> model does NOT beat the base-rate baseline (probabilities may not add"
                      " information -- treat stage calls with heavy caution)")
            print("\nReliability table (well-calibrated = mean_predicted ~ observed_freq):")
            print(cal.reliability_table.to_string(index=False))
        except ValueError as exc:
            print(f"\nBacktest could not run: {exc}")

        # --- composite-risk recession-predictiveness backtest ---
        # THE question the tool exists to answer: does high COMPOSITE RISK
        # precede recessions? (The block above tests the demoted stage
        # classifier; this tests the actual risk product.)
        from debt_cycle_scoring.scorer import run_recession_backtest
        print("\n" + "=" * 70)
        print("COMPOSITE-RISK RECESSION-PREDICTIVENESS BACKTEST")
        print("=" * 70)
        print("(does a HIGH composite risk reading precede an NBER recession onset?")
        print(" AUC 0.5 = no signal, 1.0 = perfect; tested at 1Q / 4Q / 8Q lead)")
        try:
            # usrec from label_sources is ALREADY quarterly (the panel
            # resampled it). Only convert if it's still at a finer frequency.
            usrec_q = usrec.copy()
            if not (isinstance(usrec_q.index, pd.PeriodIndex) and usrec_q.index.freqstr.startswith("Q")):
                usrec_q.index = usrec_q.index.to_period("Q")
                usrec_q = usrec_q.groupby(level=0).max()   # any recession month -> recession quarter
            rb = run_recession_backtest(risk.risk, usrec_q)
            print(f"\nRecessions in sample: {rb.n_recessions_in_sample}  "
                  f"({', '.join(rb.recession_onsets)})")
            print(f"\n{'Horizon':<10}{'AUC':<8}{'risk before':<14}{'risk otherwise':<16}{'# preceding'}")
            for h in rb.horizons:
                print(f"{str(h.horizon)+'Q':<10}{h.auc:<8}{h.mean_risk_before_recession:<14}"
                      f"{h.mean_risk_otherwise:<16}{h.n_positive}")
            print("\n  Interpretation: AUC clearly above 0.5 at a horizon means high risk")
            print("  tends to lead recessions at that horizon. " + rb.horizons[0].auc_note)
            print("\nRisk level in the quarters BEFORE each recession onset (lead profile):")
            print(rb.risk_at_each_onset.to_string())

            # --- robustness: is the AUC real, or a composition / one-flag artefact? ---
            from debt_cycle_scoring.scorer.recession_backtest import auc_at_horizon, flag_attribution
            print("\n" + "=" * 70)
            print("BACKTEST ROBUSTNESS  (descriptive -- never used to re-weight)")
            print("=" * 70)
            print("AUC on subsets (1Q / 4Q / 8Q):")
            subsets = [
                ("all scored quarters", risk.risk),
                (f">= {args.min_headline_flags} flags (consistent composition)",
                 risk.risk[risk.n_flags >= args.min_headline_flags]),
                ("1985 onward (post-Volcker)", risk.risk[risk.risk.index >= pd.Period("1985Q1", "Q")]),
            ]
            for label, sub in subsets:
                aucs = [auc_at_horizon(sub, usrec_q, h) for h in (1, 4, 8)]
                print(f"   {label:<42}" + "  ".join("  n/a" if pd.isna(a) else f"{a:5.2f}" for a in aucs))

            from debt_cycle_scoring.scorer.risk_flags import composite_spec_v1
            v1 = composite_spec_v1(risk)
            v1_aucs = [auc_at_horizon(v1, usrec_q, h) for h in (1, 4, 8)]
            print(f"\n   Spec v1 (credit spread IN the composite)    " +
                  "  ".join("  n/a" if pd.isna(a) else f"{a:5.2f}" for a in v1_aucs))
            print("   Spec v2 (current) removed the spread level after the live run showed it is")
            print("   coincident. The v2 gain over v1 was found AFTER seeing the data: v1 is the")
            print("   clean validated number; v2 is the better-designed gauge. Report both.")
            print("\nWhich flags carry the 8Q AUC:")
            att = flag_attribution(risk, usrec_q, horizon=8)
            print(att.to_string())
            print("\n  delta > 0: removing the flag lowers the AUC (it helps). A flag with")
            print("  delta < 0 is diluting the signal; a result resting on ONE flag is")
            print("  fragile. With ~8 recessions, |delta| < ~0.03 is noise.")
        except Exception as exc:
            print(f"\nRecession backtest could not run: {type(exc).__name__}: {exc}")

    # --- 8. save ---
    if args.out:
        out = pd.DataFrame({
            "risk_score": risk.risk,
            "risk_equal_weight": risk.risk_equal,
            "divergence": risk.divergence,
            "regime": risk.regime,
            "n_flags": risk.n_flags,
            "credit_gap_published": risk.gap_used.reindex(risk.risk.index),
        })
        # per-flag 0-100 values for full transparency
        for flag in FLAG_NAMES:
            if flag in risk.flags.columns:
                out[f"flag_{flag}"] = risk.flags[flag]
        # stage context as a trailing column (secondary)
        if stage_result is not None:
            out["stage_context"] = stage_result.top_stage.reindex(out.index)
        out.to_csv(args.out)
        print(f"\nSaved per-quarter composite risk (+ per-flag breakdown) to {args.out}")


if __name__ == "__main__":
    main()
