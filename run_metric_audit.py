#!/usr/bin/env python3
"""
Metric-gap audit -- run_metric_audit.py

Question: is there a useful metric the composite risk score leaves out?

For every CANDIDATE metric (fetched-but-unscored ones, plus a few new FRED
series), this measures, on live data:

  1. history      usable quarters (short history -> can't be trusted)
  2. AUC 4Q/8Q    standalone recession-lead AUC, same method as the
                  composite backtest (expanding percentile -> no look-ahead)
  3. max |rho|    highest Spearman correlation with any of the 7 existing
                  flags (redundancy: >=0.6 means it mostly repeats one)
  4. dAUC 8Q      incremental: equal-weight composite AUC WITH the
                  candidate minus WITHOUT it, on the same quarters
  5. now          the candidate's current percentile (0-100): what it would
                  be saying today. Trending levels (debt/GDP, valuations,
                  term premium...) are ranked within a trailing 10-year
                  window; mean-reverting series against full history.

Each candidate is shifted by its PUBLICATION LAG, so it is judged only on
what was public at the time -- the same discipline as the composite.

Each candidate's RISK DIRECTION is fixed in advance from theory (never
flipped to fit the data -- that would be snooping on ~8 recessions).

VERDICT rules are also pre-specified:
  ADD          AUC8 >= 0.60, max|rho| < 0.60, >= 120 quarters, dAUC >= +0.02
  REDUNDANT    max|rho| >= 0.60 (repeats an existing flag)
  CONTEXT      Dalio-central but NOT a recession-lead signal (the NBER
               backtest target is structurally blind to it) -> show on the
               dashboard as its own panel, don't blend into the composite
  SHORT        < 80 usable quarters -> can't judge
  WEAK         everything else

Honest caveat printed with the results: ~8 non-independent US recessions.
A dAUC under ~0.03 is inside the noise. This ranks candidates; it does not
prove any of them.

Run (from the folder CONTAINING debt_cycle_scoring/):
    $env:FRED_API_KEY="<key>"
    python run_metric_audit.py
    python run_metric_audit.py --out metric_audit.csv
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def to_quarterly(s: pd.Series, how: str = "mean") -> pd.Series:
    s = s.dropna()
    if s.empty:
        return s
    if isinstance(s.index, pd.PeriodIndex):
        q = s.copy()
        q.index = q.index.asfreq("Q")
    else:
        q = s.copy()
        q.index = pd.DatetimeIndex(q.index).to_period("Q")
    g = q.groupby(level=0)
    return g.mean() if how == "mean" else g.last()


def rolling_pct(s: pd.Series, window: int = 40, min_periods: int = 20) -> pd.Series:
    """Mid-rank percentile (0-100) vs the trailing `window` PRIOR quarters only."""
    def _p(w: np.ndarray) -> float:
        *hist, cur = w
        h = np.asarray(hist, dtype=float)
        return float((h < cur).mean() + 0.5 * (h == cur).mean()) if len(h) else np.nan
    return s.dropna().rolling(window + 1, min_periods=min_periods + 1).apply(_p, raw=True) * 100.0


def expanding_pct(s: pd.Series, min_periods: int = 20) -> pd.Series:
    """Backward-looking percentile rank (0-100) -- identical logic to the scorer."""
    def _p(w: np.ndarray) -> float:
        *hist, cur = w
        h = np.asarray(hist, dtype=float)
        return float((h < cur).mean() + 0.5 * (h == cur).mean()) if len(h) else np.nan   # mid-rank
    return s.dropna().expanding(min_periods=min_periods + 1).apply(_p, raw=True) * 100.0


def auc_lead(score: pd.Series, usrec_q: pd.Series, horizon: int) -> float:
    """Same case definition as the composite backtest, including right-censoring."""
    from debt_cycle_scoring.scorer.recession_backtest import auc_at_horizon
    return auc_at_horizon(score, usrec_q, horizon)


# ----------------------------------------------------------------------
# candidates
# ----------------------------------------------------------------------
@dataclass
class Candidate:
    name: str
    dalio_role: str
    context_only: bool                     # theory says: not a recession-lead signal
    build: Callable[[Dict[str, pd.Series], "object"], Optional[pd.Series]]
    note: str = ""
    lag: int = 1      # quarters from reference period to publication (see composite PUBLICATION_LAGS)
    trending: bool = False   # trending LEVEL -> rank within a trailing 10y window, not full history


def _get(raw: Dict[str, pd.Series], sid: str) -> Optional[pd.Series]:
    s = raw.get(sid)
    return None if s is None or s.dropna().empty else s


def _panel_col(panel, name: str) -> Optional[pd.Series]:
    for df in (panel.scored_metrics, panel.reference_series):
        if name in df.columns:
            return df[name]
    return None


def _yoy(s: pd.Series) -> pd.Series:
    return s.pct_change(4, fill_method=None) * 100.0


def _fiscal_interest_burden(raw, panel):
    i, r = _get(raw, "A091RC1Q027SBEA"), _get(raw, "FGRECPT")
    if i is None or r is None:
        return None
    return (to_quarterly(i, "last") / to_quarterly(r, "last")) * 100.0


def _r_minus_g(raw, panel):
    y, gdp = _get(raw, "DGS10"), _get(raw, "GDP")
    if y is None or gdp is None:
        return None
    return to_quarterly(y, "mean") - _yoy(to_quarterly(gdp, "last"))


def _deficit(raw, panel):
    d = _get(raw, "FYFSGDA188S")          # annual, surplus(+)/deficit(-) % GDP
    if d is None:
        return None
    q = to_quarterly(d, "last")
    full = pd.period_range(q.index.min(), q.index.max() + 3, freq="Q")
    return -q.reindex(full).ffill()        # risk = bigger deficit


def _hh_debt_long(raw, panel):
    """Z.1 household debt (CMDEBT, from 1945) / nominal GDP, 3yr change.
    Replaces the IMF series (HDTGPDUSQ163N) that only starts in 2005."""
    d, gdp = _get(raw, "CMDEBT"), _get(raw, "GDP")
    if d is None or gdp is None:
        return None
    return (to_quarterly(d, "last") / to_quarterly(gdp, "last") * 100.0).diff(12)


def _corp_debt_ratio_chg(raw, panel):
    """NCBDBIQ027S is a Z.1 dollar LEVEL, not a ratio: divide by GDP before the
    3-yr change (fixed 2026-10-13; earlier audits measured nominal debt growth)."""
    from debt_cycle_scoring.scorer.fiscal_panel import _per_gdp, _change
    return _change(_per_gdp(raw, "NCBDBIQ027S"), 12)


def _chg12(sid: str):
    def f(raw, panel):
        s = _get(raw, sid)
        return None if s is None else to_quarterly(s, "last").diff(12)
    return f


def _level(sid: str, how: str = "last", sign: float = 1.0):
    def f(raw, panel):
        s = _get(raw, sid)
        return None if s is None else sign * to_quarterly(s, how)
    return f


def _yoy_of(sid: str):
    def f(raw, panel):
        s = _get(raw, sid)
        return None if s is None else _yoy(to_quarterly(s, "last"))
    return f


def _ratio_to_gdp(sid: str, sign: float = 1.0):
    def f(raw, panel):
        a, gdp = _get(raw, sid), _get(raw, "GDP")
        if a is None or gdp is None:
            return None
        return sign * (to_quarterly(a, "last") / to_quarterly(gdp, "last")).dropna()
    return f


def _foreign_share(raw, panel):
    fh, tot = _get(raw, "FDHBFIN"), _get(raw, "GFDEBTN")    # $bn / $mn
    if fh is None or tot is None:
        return None
    share = (to_quarterly(fh, "last") * 1000.0 / to_quarterly(tot, "last") * 100.0).dropna()
    return -share.diff(8)                                    # risk = foreign share FALLING over 2y


def _from_panel(name: str, transform: Optional[Callable] = None):
    def f(raw, panel):
        s = _panel_col(panel, name)
        if s is None:
            return None
        return transform(s) if transform else s
    return f


CANDIDATES: List[Candidate] = [
    # --- fiscal / sovereign leg: the one Dalio now emphasizes most for the US
    Candidate("Federal interest / receipts", "sovereign debt squeeze (How Countries Go Broke)",
              True, _fiscal_interest_burden, "new series"),
    Candidate("r - g (10y yield - nominal GDP growth)", "deleveraging math: r>g = debt snowballs",
              False, _r_minus_g, "new series (GDP)"),
    Candidate("Federal debt / GDP", "sovereign debt stock", True, _level("GFDEGDQ188S"),
              "fetched, unscored"),
    Candidate("Federal deficit / GDP", "sovereign debt flow", True, _deficit, "fetched, unscored"),
    Candidate("10y term premium (Kim-Wright)", "bond buyers demanding compensation",
              True, _level("THREEFYTP10", "mean"), "new series"),
    # --- sector credit (private leg split out of the total credit gap)
    Candidate("Household debt/GDP, 3yr change", "household leverage build (Mian-Sufi-Verner)",
              False, _chg12("HDTGPDUSQ163N"), "fetched, unscored"),
    Candidate("Household debt/GDP (Z.1, long), 3yr chg", "household leverage build, full history",
              False, _hh_debt_long, "new series (CMDEBT)"),
    Candidate("Corporate debt/GDP, 3yr change", "corporate leverage build",
              False, _corp_debt_ratio_chg, "fetched, unscored"),
    Candidate("Bank lending standards (SLOOS C&I)", "credit supply tightening",
              False, _level("DRTSCILM", "last"), "new series (from 1990)"),
    # --- monetary stance (MP1/MP2/MP3)
    Candidate("Real fed funds rate", "MP1 tightness", False,
              _from_panel("Real federal funds rate"), "derived, unscored"),
    Candidate("Fed balance sheet / GDP, 1yr change", "MP2 (QE) stage signal", True,
              _from_panel("Fed balance sheet / GDP", lambda s: s.diff(4)), "derived, unscored"),
    Candidate("Core PCE inflation (yoy)", "inflation constraint on easing", False,
              _yoy_of("PCEPILFE"), "fetched, unscored"),
    # --- bubble gauges
    Candidate("Household net worth / income", "asset bubble (wealth vs income)", False,
              _from_panel("Household net worth / DPI"), "derived, unscored"),
    Candidate("Home price / income", "housing bubble", False,
              _from_panel("Home price / income"), "derived, unscored"),
    Candidate("Real broad dollar (yoy)", "currency pressure / reserve demand", True,
              _yoy_of("RBUSBIS"), "fetched, unscored"),
    # --- added in the second coverage review
    Candidate("Corporate equities / GDP (Z.1, long)", "bubble gauge #1: prices vs traditional measures",
              False, _ratio_to_gdp("NCBEILQ027S"), "new series (Buffett, from 1945)"),
    Candidate("Building permits (yoy)", "housing leads the cycle (Leamer)", False,
              lambda raw, p: (None if _get(raw, "PERMIT") is None
                              else -_yoy(to_quarterly(_get(raw, "PERMIT"), "mean"))), "new series"),
    Candidate("Policy-rate headroom (fed funds, inverted)", "MP1 exhaustion: little room to cut", True,
              _level("FEDFUNDS", "mean", sign=-1.0), "fetched, unscored", lag=0),
    Candidate("Foreign share of federal debt, 2y fall", "dependence on foreign creditors", True,
              _foreign_share, "new series"),
    Candidate("Current-account deficit / GDP", "external financing need / currency risk", True,
              _ratio_to_gdp("IEABC", sign=-1.0), "new series"),
]

# Publication lags (quarters) where they differ from the default of 1.
# Quarter-end daily/market data: 0. Annual series dated at year start: long.
_LAGS = {
    "10y term premium (Kim-Wright)": 0,
    "Real fed funds rate": 1,
    "Federal deficit / GDP": 3,          # fiscal-year data published ~Oct, dated Jan
    "Home price / income": 6,             # annual median income published ~Sep of year+1
    "Household debt/GDP, 3yr change": 2,  # IMF series
    "Bank lending standards (SLOOS C&I)": 0,
    "Fed balance sheet / GDP, 1yr change": 1,
}
_TRENDING = {
    "Federal interest / receipts", "Federal debt / GDP", "10y term premium (Kim-Wright)",
    "Household net worth / income", "Home price / income", "Corporate equities / GDP (Z.1, long)",
    "Policy-rate headroom (fed funds, inverted)",
}
for _c in CANDIDATES:
    if _c.name in _LAGS:
        _c.lag = _LAGS[_c.name]
    _c.trending = _c.name in _TRENDING

NEW_SERIES = ["NCBEILQ027S", "PERMIT", "FDHBFIN", "GFDEBTN", "IEABC", "FEDFUNDS", "CMDEBT", "A091RC1Q027SBEA", "FGRECPT", "GDP", "THREEFYTP10", "DRTSCILM"]
RAW_SERIES = NEW_SERIES + ["DGS10", "FYFSGDA188S", "GFDEGDQ188S", "HDTGPDUSQ163N",
                           "NCBDBIQ027S", "PCEPILFE", "RBUSBIS"]


def verdict(auc8, rho, nq, dauc, context_only) -> str:
    if nq < 80 or np.isnan(auc8):
        return "SHORT"
    if not np.isnan(rho) and rho >= 0.60:
        return "REDUNDANT"
    if auc8 >= 0.60 and nq >= 120 and not np.isnan(dauc) and dauc >= 0.02:
        return "ADD"
    if context_only:
        return "CONTEXT"
    return "WEAK"


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def run_audit(panel, raw: Dict[str, pd.Series], risk_flags: pd.DataFrame,
              usrec_q: pd.Series) -> pd.DataFrame:
    base_flags = risk_flags.copy()
    rows = []
    for c in CANDIDATES:
        try:
            s = c.build(raw, panel)
        except Exception as exc:                       # never let one candidate kill the audit
            print(f"  WARNING: {c.name} failed to build ({type(exc).__name__}: {exc})", file=sys.stderr)
            s = None
        if s is not None and not s.dropna().empty and c.lag:
            s = s.dropna()
            full = pd.period_range(s.index.min(), s.index.max() + c.lag, freq="Q")
            s = s.reindex(full).shift(c.lag)        # judged only on what was public at the time
        if s is None or s.dropna().empty:
            rows.append(dict(candidate=c.name, dalio_role=c.dalio_role, source=c.note,
                             quarters=0, auc_4q=np.nan, auc_8q=np.nan, max_rho=np.nan,
                             rho_with="", d_auc_8q=np.nan, now_pct=np.nan, verdict="UNAVAILABLE"))
            continue

        # 0-100, higher = more risk, no look-ahead. Trending LEVELS are ranked within
        # a trailing 10y window: a full-history rank of a trending series measures
        # the trend (it would sit near 100 for decades), not the risk signal.
        pct = rolling_pct(s) if c.trending else expanding_pct(s)
        nq = int(pct.dropna().shape[0])
        a4, a8 = auc_lead(pct, usrec_q, 4), auc_lead(pct, usrec_q, 8)

        # redundancy vs each existing flag, on overlapping quarters
        best_rho, best_with = np.nan, ""
        for f in base_flags.columns:
            both = pd.concat([pct, base_flags[f]], axis=1).dropna()
            if len(both) >= 40 and both.iloc[:, 0].nunique() > 1 and both.iloc[:, 1].nunique() > 1:
                rho = abs(both.iloc[:, 0].corr(both.iloc[:, 1], method="spearman"))
                if np.isnan(rho):
                    continue
                if np.isnan(best_rho) or rho > best_rho:
                    best_rho, best_with = rho, f

        # incremental: equal-weight composite with vs without, same quarters
        common = base_flags.index.intersection(pct.dropna().index)
        without = base_flags.loc[common].mean(axis=1, skipna=True)
        with_c = pd.concat([base_flags.loc[common], pct.reindex(common).rename("cand")], axis=1).mean(axis=1)
        enough = base_flags.loc[common].notna().sum(axis=1) >= 2
        d8 = auc_lead(with_c[enough], usrec_q, 8) - auc_lead(without[enough], usrec_q, 8)

        now = pct.dropna().iloc[-1] if nq else np.nan
        rows.append(dict(
            candidate=c.name, dalio_role=c.dalio_role, source=c.note, quarters=nq,
            auc_4q=round(a4, 2), auc_8q=round(a8, 2),
            max_rho=round(best_rho, 2) if not np.isnan(best_rho) else np.nan, rho_with=best_with,
            d_auc_8q=round(d8, 3) if not np.isnan(d8) else np.nan,
            now_pct=round(now) if not np.isnan(now) else np.nan,
            verdict=verdict(a8, best_rho, nq, d8, c.context_only),
        ))
    order = {"ADD": 0, "CONTEXT": 1, "WEAK": 2, "REDUNDANT": 3, "SHORT": 4, "UNAVAILABLE": 5}
    df = pd.DataFrame(rows)
    return df.sort_values(["verdict", "auc_8q"], key=lambda col: col.map(order) if col.name == "verdict" else -col.fillna(0))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="1962-01-01")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        print("ERROR: FRED_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    from debt_cycle_scoring.ingestion import FredClient, build_quarterly_panel, merge_derived_metrics
    from debt_cycle_scoring.labeling import compute_credit_gap
    from debt_cycle_scoring.scorer import build_risk_series

    client = FredClient(api_key=api_key)
    print("Building the baseline panel + composite (same as run_scorer.py) ...")
    panel = merge_derived_metrics(build_quarterly_panel(client, start_date=args.start))
    credit_ratio = panel.label_sources["US Credit-to-GDP ratio (BIS, via FRED)"].dropna()
    gap = compute_credit_gap(credit_ratio, min_periods=40)["gap"]
    base = build_risk_series(panel.scored_metrics, panel.reference_series, gap)

    usrec_q = panel.label_sources["NBER recession indicator"].copy()
    if not isinstance(usrec_q.index, pd.PeriodIndex):
        q = usrec_q.dropna().copy(); q.index = pd.DatetimeIndex(q.index).to_period("Q")
        usrec_q = q.groupby(level=0).max()             # any recession month -> recession quarter
    usrec_q = usrec_q.fillna(0).astype(int)

    print("Fetching candidate series (missing IDs are skipped and reported) ...")
    raw: Dict[str, pd.Series] = {}
    for sid in RAW_SERIES:
        try:
            raw[sid] = client.fetch_series(sid, start_date=args.start).series
        except Exception as exc:
            print(f"  WARNING: {sid} unavailable ({type(exc).__name__}) -- skipping", file=sys.stderr)

    base_auc8 = auc_lead(base.risk, usrec_q, 8)
    df = run_audit(panel, raw, base.flags, usrec_q)

    pd.set_option("display.width", 200, "display.max_colwidth", 44)
    print("\n" + "=" * 100)
    print(f"METRIC-GAP AUDIT   (baseline research composite AUC 8Q = {base_auc8:.2f})")
    print("=" * 100)
    print(df[["candidate", "quarters", "auc_4q", "auc_8q", "max_rho", "rho_with",
              "d_auc_8q", "now_pct", "verdict"]].to_string(index=False))
    print("\nReading it:")
    print("  ADD       -> leads recessions, not redundant, improves the composite: candidate 8th flag")
    print("  CONTEXT   -> Dalio-central but the NBER-recession backtest can't validate it")
    print("               (the US has had no fiscal/currency crisis in the FRED era).")
    print("               Show it as a separate panel -- do NOT blend it in.")
    print("  REDUNDANT -> mostly repeats the flag named in rho_with")
    print("  now_pct   -> where the metric sits today (100 = most extreme); trending levels vs")
    print("               their trailing 10 years, others vs full history")
    print("  Caveat: ~8 non-independent recessions. dAUC < ~0.03 is noise. Ranking, not proof.")

    if args.out:
        df.to_csv(args.out, index=False)
        print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
