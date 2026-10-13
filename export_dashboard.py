#!/usr/bin/env python3
"""
Dashboard exporter -- export_dashboard.py

Runs the full pipeline against live FRED data and writes a self-contained
dashboard:

    site/index.html   the Debt-Cycle Risk Monitor, with fresh data baked in
    site/data.json    the same numbers as plain JSON (for inspection/history)

Every number on the page is computed here -- nothing is hand-entered. Used
by the weekly GitHub Actions workflow, and runnable locally:

    $env:FRED_API_KEY="<key>"
    python export_dashboard.py            # then open site/index.html

The key is read ONLY from the FRED_API_KEY environment variable and is never
written into any output.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
TEMPLATE = ROOT / "dashboard" / "template.html"
PLACEHOLDER = "/*__DASHBOARD_DATA__*/null"

HEADLINE_MIN_FLAGS = 4            # same floor as run_scorer.py's headline
EXOGENOUS_ONSET_YEARS = {2001, 2020}   # dot-com + COVID: not debt-cycle recessions

FLAG_LABELS = {
    "credit_gap": "Credit-to-GDP gap (BIS)",
    "debt_service_ratio": "Debt service ratio",
    "yield_curve": "Yield curve (10y−3m)",
    "credit_spread": "Credit spread (Baa − 10y)",
    "m2_velocity": "M2 velocity, yoy change (inverted)",
    "long_bond_level": "Long-bond yield (vs trailing 10y)",
    "debasement": "Debasement / deflation (BEI)",
}
COUNTRY_META = {   # display name, colour token
    "US": ("United States", "--c-us"),
    "Japan": ("Japan", "--c-jp"),
    "UK": ("United Kingdom", "--c-uk"),
    "Euro": ("Euro area", "--c-eu"),
}


def state_of(v: float) -> str:
    return "good" if v < 33 else ("watch" if v < 67 else "alarm")


def clean(o):
    """JSON-safe: NaN/inf -> None, numpy -> python, periods -> str."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if (math.isnan(f) or math.isinf(f)) else round(f, 3)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, pd.Period):
        return str(o)
    return o


def to_quarterly(s: pd.Series, how: str = "last") -> pd.Series:
    q = s.dropna().copy()
    if not isinstance(q.index, pd.PeriodIndex):
        q.index = pd.DatetimeIndex(q.index).to_period("Q")
    g = q.groupby(level=0)
    return g.max() if how == "max" else g.last()


# ----------------------------------------------------------------------
def build_data(client) -> Dict:
    from debt_cycle_scoring.ingestion import build_quarterly_panel, merge_derived_metrics
    from debt_cycle_scoring.labeling import compute_credit_gap
    from debt_cycle_scoring.scorer import build_risk_series, run_recession_backtest
    from debt_cycle_scoring.scorer.risk_flags import RESEARCH_WEIGHTS
    from debt_cycle_scoring.scorer.short_horizon_flags import build_short_horizon_score
    from debt_cycle_scoring.scorer.credit_gap_monitor import (
        build_credit_gap_monitor, CREDIT_RATIO_SERIES)
    from debt_cycle_scoring.scorer.fiscal_panel import build_fiscal_panel, FISCAL_SERIES
    from debt_cycle_scoring.scorer.market_pulse import build_market_pulse, PULSE_SERIES

    log = lambda m: print(m, file=sys.stderr)

    # --- 1. panel + composite -------------------------------------------
    log("Building panel ...")
    panel = merge_derived_metrics(build_quarterly_panel(client, start_date="1960-01-01"))
    credit_ratio = panel.label_sources["US Credit-to-GDP ratio (BIS, via FRED)"].dropna()
    gap = compute_credit_gap(credit_ratio, min_periods=40)["gap"]
    risk = build_risk_series(panel.scored_metrics, panel.reference_series, gap)

    valid = risk.risk.dropna()
    from debt_cycle_scoring.scorer.risk_flags import headline_quarter
    hq, required_flags = headline_quarter(risk, HEADLINE_MIN_FLAGS)
    if hq is None:
        hq, required_flags = valid.index.max(), HEADLINE_MIN_FLAGS

    regime = risk.regime.loc[hq]
    gap_at_hq = risk.gap_used.get(hq, float("nan"))          # publication-aligned gap that set the regime
    present = risk.flags.loc[hq].dropna()
    raw_w = {k: RESEARCH_WEIGHTS[regime].get(k, 0.0) for k in present.index}
    tw = sum(raw_w.values()) or 1.0
    flags = []
    for k, v in present.items():
        w = raw_w[k] / tw
        flags.append(dict(key=k, name=FLAG_LABELS.get(k, k), v=round(float(v)),
                          wt=round(w, 3), contrib=round(w * float(v), 1), st=state_of(float(v))))
    missing = [FLAG_LABELS[k] for k in FLAG_LABELS if k not in present.index]

    provisional = [dict(q=str(q), risk=round(float(risk.risk.loc[q])), n=int(risk.n_flags.loc[q]))
                   for q in valid.index if q > hq]

    # --- 2. history + recession shading ---------------------------------
    usrec_q = panel.label_sources["NBER recession indicator"]
    if not isinstance(usrec_q.index, pd.PeriodIndex):
        usrec_q = to_quarterly(usrec_q, "max")
    usrec_q = usrec_q.fillna(0).astype(int)
    hist = [dict(q=str(q), r=float(v), rec=int(usrec_q.get(q, 0)))
            for q, v in risk.risk.items() if not pd.isna(v)]

    # --- 3. recession backtest ------------------------------------------
    rb = run_recession_backtest(risk.risk, usrec_q)
    from debt_cycle_scoring.scorer.recession_backtest import auc_at_horizon, flag_attribution
    subsets = [
        ("All scored quarters", risk.risk),
        (f"≥ {HEADLINE_MIN_FLAGS} flags present", risk.risk[risk.n_flags >= HEADLINE_MIN_FLAGS]),
        ("1985 onward", risk.risk[risk.risk.index >= pd.Period("1985Q1", "Q")]),
    ]
    robustness = [dict(label=lab, auc=[auc_at_horizon(sub, usrec_q, h) for h in (1, 4, 8)]) for lab, sub in subsets]
    from debt_cycle_scoring.scorer.risk_flags import composite_spec_v1
    robustness.append(dict(label="Spec v1: credit spread in composite",
                           auc=[auc_at_horizon(composite_spec_v1(risk), usrec_q, h) for h in (1, 4, 8)],
                           v1=True))
    att = flag_attribution(risk, usrec_q, horizon=8)
    attribution = [dict(key=k, name=FLAG_LABELS.get(k, k), **{c: att.loc[k, c] for c in att.columns}) for k in att.index]
    auc = [dict(h=h.horizon, auc=h.auc, n_pos=h.n_positive,
                before=h.mean_risk_before_recession, other=h.mean_risk_otherwise) for h in rb.horizons]
    cols = [f"-{k}Q" for k in range(8, -1, -1)]          # -8Q ... -0Q (left to right)
    prof = rb.risk_at_each_onset.reindex(columns=cols)
    exo_idx = [o for o in prof.index if int(o[:4]) in EXOGENOUS_ONSET_YEARS]
    cred_idx = [o for o in prof.index if o not in exo_idx]
    lead = dict(
        credit=prof.loc[cred_idx].mean(axis=0, skipna=True).tolist() if cred_idx else [],
        exog=prof.loc[exo_idx].mean(axis=0, skipna=True).tolist() if exo_idx else [],
        credit_onsets=cred_idx, exog_onsets=exo_idx,
        per_onset={o: prof.loc[o].tolist() for o in prof.index},
    )

    # --- 4. short-horizon stress ----------------------------------------
    short = None
    try:
        sh = build_short_horizon_score(panel.short_horizon, panel.short_horizon_worst,
                                       panel.scored_metrics, validation=panel.short_horizon_validation,
                                       reference_series=panel.reference_series)
        s_valid = sh.stress.dropna()
        if len(s_valid):
            sq = s_valid.index.max()
            nfci = None
            if not sh.validation.empty and sq in sh.validation.index:
                row = sh.validation.loc[sq].dropna()
                nfci = float(row.iloc[0]) if len(row) else None
            short = dict(q=str(sq), stress=round(float(sh.stress.loc[sq])),
                         worst=round(float(sh.stress_worst.loc[sq])), nfci=nfci)
    except Exception as exc:
        log(f"(short-horizon unavailable: {exc})")

    # --- 5. cross-country credit gaps -----------------------------------
    log("Fetching cross-country credit ratios ...")
    ratios = {}
    for c, sid in CREDIT_RATIO_SERIES.items():
        try:
            ratios[c] = to_quarterly(client.fetch_series(sid, start_date="1960-01-01").series)
        except Exception as exc:
            log(f"  WARNING: {c} ({sid}) unavailable -- skipped ({type(exc).__name__})")
    gaps = None
    if ratios:
        mon = build_credit_gap_monitor(ratios)
        gp = mon.gap_panel.dropna(how="all").tail(12)
        meta = []
        for c, g, flag, conf in mon.ranking():
            cg = mon.countries[c]
            gs = cg.gap.dropna()
            d2y = float(gs.iloc[-1] - gs.iloc[-9]) if len(gs) >= 9 else float("nan")
            name, col = COUNTRY_META.get(c, (c, "--ink-2"))
            meta.append(dict(key=c, c=name, col=col, gap=round(float(g), 1), state=flag,
                             nq=cg.n_quarters, conf=conf, q=str(cg.latest_quarter), d2y=round(d2y, 1)))
        gaps = dict(q=[str(q) for q in gp.index],
                    series={c: gp[c].tolist() for c in gp.columns},
                    meta=meta)

    # --- 6. fiscal & valuation panel ------------------------------------
    log("Fetching fiscal series ...")
    raw = {}
    for sid in FISCAL_SERIES:
        try:
            raw[sid] = client.fetch_series(sid, start_date="1960-01-01").series
        except Exception as exc:
            log(f"  WARNING: {sid} unavailable -- skipped ({type(exc).__name__})")
    # free US Treasury sources FRED doesn't carry (avg rate on existing debt,
    # debt by type, TIC foreign holders); each is skipped cleanly if unavailable
    try:
        from debt_cycle_scoring.ingestion.treasury_client import TreasuryClient, fetch_treasury_extras
        log("Fetching US Treasury data (Fiscal Data API, TIC) ...")
        raw.update(fetch_treasury_extras(TreasuryClient(), log))
    except Exception as exc:
        log(f"  WARNING: Treasury sources skipped ({type(exc).__name__}: {exc})")
    fiscal = [m.__dict__ for m in build_fiscal_panel(raw, panel)]

    # --- 7. daily market pulse (always re-downloaded: never a cached close) --
    log("Fetching daily market series ...")
    start_d = (pd.Timestamp.today() - pd.DateOffset(years=11)).strftime("%Y-%m-%d")
    daily = {}
    for sid in PULSE_SERIES:
        try:
            daily[sid] = client.fetch_series(sid, start_date=start_d, force_refresh=True).series
        except Exception as exc:
            log(f"  WARNING: {sid} unavailable -- skipped ({type(exc).__name__})")
    pulse = [p.__dict__ for p in build_market_pulse(daily)]

    return clean(dict(
        generated=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        headline=dict(q=str(hq), risk=round(float(risk.risk.loc[hq])),
                      equal=round(float(risk.risk_equal.loc[hq])),
                      divergence=round(float(risk.divergence.loc[hq])),
                      regime=regime, gap=gap_at_hq, n_flags=int(risk.n_flags.loc[hq]),
                      min_flags=required_flags),
        flags=sorted(flags, key=lambda f: -f["contrib"]),
        missing_flags=missing,
        provisional=provisional,
        history=hist,
        auc=auc, n_recessions=rb.n_recessions_in_sample, onsets=rb.recession_onsets,
        robustness=robustness, attribution=attribution,
        method=dict(curve=risk.curve_source, spread=risk.spread_source, lags=risk.publication_lags),
        context=dict(credit_spread=(lambda c: None if c.empty else round(float(c.iloc[-1])))(
            risk.context_flags["credit_spread"].dropna()) if "credit_spread" in risk.context_flags.columns else None),
        lead=lead,
        short=short,
        gaps=gaps,
        fiscal=fiscal,
        pulse=pulse,
    ))


def render(data: Dict, out_dir: Path) -> Path:
    html = TEMPLATE.read_text(encoding="utf-8")
    if PLACEHOLDER not in html:
        raise RuntimeError(f"placeholder {PLACEHOLDER!r} missing from {TEMPLATE}")
    blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "data.json").write_text(json.dumps(data, indent=1), encoding="utf-8")
    html = html.replace(PLACEHOLDER, blob)
    # plain-English layer: explanation library + scenario engine, inlined so the
    # page stays a single self-contained file
    for marker, fname in (("/*__EXPLAIN_JS__*/", "explain.js"), ("/*__GUIDE_JS__*/", "guide.js")):
        src = TEMPLATE.parent / fname
        if marker in html and src.exists():
            html = html.replace(marker, src.read_text(encoding="utf-8").replace("</script", "<\\/script"))
    out = out_dir / "index.html"
    out.write_text(html, encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="site")
    args = ap.parse_args()
    key = os.environ.get("FRED_API_KEY")
    if not key:
        print("ERROR: FRED_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)
    from debt_cycle_scoring.ingestion import FredClient
    data = build_data(FredClient(api_key=key))
    out = render(data, ROOT / args.out)
    h = data["headline"]
    print(f"Wrote {out}  --  {h['q']}: risk {h['risk']}/100, regime {h['regime']}")


if __name__ == "__main__":
    main()
