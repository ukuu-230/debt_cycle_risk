"""
Dalio context panel -- scorer/fiscal_panel.py

The parts of Dalio's framework that the composite risk score cannot carry,
grouped the way he groups them:

  * Sovereign debt & financing   (How Countries Go Broke)
  * Bubble gauges                (Big Debt Crises: the bubble stage)
  * Private leverage & credit    (the credit cycle beyond the BIS gap)
  * Monetary policy capacity     (MP1 -> MP2 -> MP3)

Why these are DISPLAYED, never SCORED:
  * The BIS credit-to-GDP gap covers the PRIVATE non-financial sector only.
  * The composite is validated against NBER recessions; the US has had no
    fiscal or currency crisis in the FRED era, and the metric-gap audit
    found none of these improves the recession score (federal debt/GDP and
    the deficit even score BELOW 0.5 -- they rise after recessions).
  * So each is shown as a backward-looking percentile against its own FULL
    history ("is this historically unusual?"), with no panel composite --
    any weight would be a guess.

Selection rule: metrics are included because Dalio's framework names them,
NOT because they read extreme today (that would be selection bias). Housing
permits, for example, test well but are not a Dalio indicator, so they are
not here.

Units: a raw value is shown only where the FRED units are unambiguous
(% of GDP, %, pp). Ratios built from Z.1 / balance-sheet dollar LEVELS are
shown as percentiles only -- the percentile does not depend on units.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

# FRED series this panel fetches directly (beyond the main catalog).
FISCAL_SERIES: List[str] = [
    "GFDEGDQ188S",       # federal debt, % GDP (Q)
    "FYFSGDA188S",       # federal surplus(+)/deficit(-), % GDP (A)
    "A091RC1Q027SBEA",   # federal interest payments, $bn SAAR (Q)
    "FGRECPT",           # federal current receipts, $bn SAAR (Q)
    "THREEFYTP10",       # Kim-Wright 10y term premium, pp (D)
    "GDP",               # nominal GDP, $bn SAAR (Q)
    "DGS10",             # 10y Treasury yield (D)
    "FDHBFIN",           # federal debt held by foreign investors, $bn (Q)
    "GFDEBTN",           # total public debt, $mn (Q)
    "IEABC",             # current-account balance (Q, level)
    "RBUSBIS",           # real broad effective exchange rate (M)
    "NCBEILQ027S",       # nonfinancial corporate equities, level (Q, Z.1)
    "CMDEBT",            # household debt, level (Q, Z.1)
    "NCBDBIQ027S",       # nonfinancial corporate debt, level (Q, Z.1)
    "DRTSCILM",          # SLOOS: net % banks tightening C&I standards (Q)
    "FEDFUNDS",          # effective fed funds rate (M)
    "PCEPILFE",          # core PCE price index (M)
    "WALCL",             # Fed total assets, level (W)
    "WSHOBL",            # Fed holdings of Treasury bills, $mn (W)
]
# Added by ingestion/treasury_client.fetch_treasury_extras (not FRED):
#   TSY_AVG_RATE_MKT  average rate on outstanding marketable Treasury debt, % (M)
#   TSY_BILLS_SHARE   bills as % of marketable debt outstanding (M)
#   TIC_JP_CN         Japan + China (mainland) Treasury holdings, $bn (M, latest 13 months)
#   TIC_OFFICIAL      foreign official Treasury holdings, $bn (M, latest 13 months)

GROUPS = [
    "Sovereign debt & financing",
    "Bubble gauges",
    "Private leverage & credit",
    "Monetary policy capacity (MP1→MP3)",
]


def to_quarterly(s: pd.Series, how: str = "last") -> pd.Series:
    s = s.dropna()
    if s.empty:
        return s
    q = s.copy()
    q.index = (q.index.asfreq("Q") if isinstance(q.index, pd.PeriodIndex)
               else pd.DatetimeIndex(q.index).to_period("Q"))
    g = q.groupby(level=0)
    return g.mean() if how == "mean" else g.last()


def expanding_percentile(s: pd.Series, min_periods: int = 20) -> pd.Series:
    """Backward-looking mid-rank percentile (0-100) of each value vs all PRIOR values."""
    def _p(w: np.ndarray) -> float:
        *hist, cur = w
        h = np.asarray(hist, dtype=float)
        return float((h < cur).mean() + 0.5 * (h == cur).mean()) if len(h) else np.nan
    return s.dropna().expanding(min_periods=min_periods + 1).apply(_p, raw=True) * 100.0


def _q(raw: Dict[str, pd.Series], sid: str, how: str = "last") -> Optional[pd.Series]:
    s = raw.get(sid)
    return None if s is None or s.dropna().empty else to_quarterly(s, how)


def _yoy(s: pd.Series) -> pd.Series:
    full = pd.period_range(s.index.min(), s.index.max(), freq="Q")
    return s.reindex(full).pct_change(4, fill_method=None) * 100.0


def _per_gdp(raw, sid: str) -> Optional[pd.Series]:
    """
    Series relative to nominal GDP. A dollar LEVEL (Z.1 / balance-sheet series
    are millions; median magnitude in the thousands+) is divided by GDP; a
    series that is already a ratio (median magnitude < 1000) is returned as is.
    Scale is irrelevant because only the percentile is displayed.
    """
    s, gdp = _q(raw, sid), _q(raw, "GDP")
    if s is None:
        return None
    if s.abs().median() < 1000:
        return s
    if gdp is None:
        return None
    return (s / gdp).dropna()


def _change(s: Optional[pd.Series], quarters: int) -> Optional[pd.Series]:
    if s is None:
        return None
    full = pd.period_range(s.index.min(), s.index.max(), freq="Q")
    return s.reindex(full).diff(quarters).dropna()


# ---------------------------------------------------------------- builders
def _deficit(raw, panel):
    d = _q(raw, "FYFSGDA188S")
    if d is None:
        return None
    # annual value carried through its year, but never past the current quarter
    end = min(d.index.max() + 3, pd.Timestamp.today().to_period("Q"))
    full = pd.period_range(d.index.min(), max(end, d.index.max()), freq="Q")
    return -d.reindex(full).ffill()     # show the deficit as a positive number


def _interest_receipts(raw, panel):
    i, r = _q(raw, "A091RC1Q027SBEA"), _q(raw, "FGRECPT")
    if i is None or r is None:
        return None
    return (i / r * 100.0).dropna()


def _r_minus_g(raw, panel):
    y, gdp = _q(raw, "DGS10", "mean"), _q(raw, "GDP")
    if y is None or gdp is None:
        return None
    return (y - _yoy(gdp)).dropna()


def _foreign_share(raw, panel):
    fh, tot = _q(raw, "FDHBFIN"), _q(raw, "GFDEBTN")    # $bn / $mn
    if fh is None or tot is None:
        return None
    return (fh * 1000.0 / tot * 100.0).dropna()


def _real_policy_rate(raw, panel):
    ff, pce = _q(raw, "FEDFUNDS", "mean"), _q(raw, "PCEPILFE")
    if ff is None or pce is None:
        return None
    return (ff - _yoy(pce)).dropna()


def _refi_gap(raw, panel):
    """10y market yield minus the average rate on existing marketable debt."""
    y, avg = _q(raw, "DGS10", "mean"), _q(raw, "TSY_AVG_RATE_MKT")
    if y is None or avg is None:
        return None
    return (y - avg).dropna()


def _fed_bills_chg(raw, panel):
    b = raw.get("WSHOBL")
    if b is None or b.dropna().empty:
        return None
    return _change(to_quarterly(b) / 1000.0, 4)          # $bn change over 4 quarters


def _monthly(raw, sid):
    s = raw.get(sid)
    return None if s is None or s.dropna().empty else s.dropna().sort_index()


def _panel_col(name: str):
    def f(raw, panel):
        for df in (panel.scored_metrics, panel.reference_series):
            if name in df.columns:
                return df[name].dropna()
        return None
    return f


@dataclass
class _Spec:
    key: str
    group: str
    name: str
    why: str
    unit: str            # "" -> raw value not shown (units ambiguous); percentile only
    risk_dir: int        # +1 high = stretched, -1 low = stretched, 0 two-sided / context
    build: Callable
    plausible: Optional[tuple] = None   # (lo, hi): hide the raw value if outside (unit guard)
    mode: str = "pct"    # "pct": percentile vs history; "chg12m": 12-month % change (short series)


_G0, _G1, _G2, _G3 = GROUPS

_SPECS: List[_Spec] = [
    # --- Sovereign debt & financing
    _Spec("interest", _G0, "Federal interest / receipts", "Dalio’s preferred gauge: how much of the government’s income its debt costs to service",
          "% receipts", +1, _interest_receipts, (0, 60)),
    _Spec("fed_debt", _G0, "Federal debt / GDP", "Debt stock; Dalio calls it weak on its own, but it is the fuel for the debt-service squeeze",
          "% GDP", +1, lambda raw, p: _q(raw, "GFDEGDQ188S"), (10, 300)),
    _Spec("deficit", _G0, "Federal deficit / GDP", "New borrowing each year; large deficits outside recession are structural",
          "% GDP", +1, _deficit, (-10, 35)),
    _Spec("r_minus_g", _G0, "r − g (10y yield − nominal GDP growth)", "If r > g, debt/GDP grows on its own: Dalio's deleveraging math",
          "pp", +1, _r_minus_g, (-30, 30)),
    _Spec("term_premium", _G0, "10y term premium (Kim-Wright)", "What buyers demand to hold long debt: the price of supply outrunning demand",
          "pp", +1, lambda raw, p: _q(raw, "THREEFYTP10", "mean"), (-5, 10)),
    _Spec("foreign_share", _G0, "Foreign share of federal debt", "Reliance on foreign creditors; a falling share means weaker outside demand",
          "% of debt", -1, _foreign_share, (5, 70)),
    _Spec("refi_gap", _G0, "Refinancing gap (10y yield − avg rate on existing debt)", "Positive: the interest bill keeps rising as old, cheaper debt is refinanced",
          "pp", +1, _refi_gap, (-8, 8)),
    _Spec("bills_share", _G0, "Bills share of marketable debt", "Reliance on short-term funding; Treasury’s advisers suggest 15–20%",
          "% of debt", +1, lambda raw, p: _q(raw, "TSY_BILLS_SHARE"), (3, 70)),
    _Spec("tic_jp_cn", _G0, "Japan + China Treasury holdings", "The two largest foreign holders; Dalio flags both as likely to reduce",
          "% 12-mo", -1, lambda raw, p: _monthly(raw, "TIC_JP_CN"), (-60, 60), "chg12m"),
    _Spec("tic_official", _G0, "Foreign central-bank Treasury holdings", "Official foreign demand for US government debt",
          "% 12-mo", -1, lambda raw, p: _monthly(raw, "TIC_OFFICIAL"), (-60, 60), "chg12m"),
    _Spec("current_account", _G0, "Current-account balance / GDP", "External borrowing need; deficits must be financed by foreigners",
          "", -1, lambda raw, p: _per_gdp(raw, "IEABC")),
    _Spec("real_dollar", _G0, "Real broad dollar (yoy)", "Reserve-currency strength; a weakening dollar is Dalio's debasement channel",
          "% yoy", 0, lambda raw, p: (None if _q(raw, "RBUSBIS", "mean") is None
                                      else _yoy(_q(raw, "RBUSBIS", "mean")).dropna()), (-40, 40)),
    # --- Bubble gauges
    _Spec("equities_gdp", _G1, "Corporate equities / GDP (Z.1)", "Prices vs traditional measures, Dalio's first bubble test",
          "", +1, lambda raw, p: _per_gdp(raw, "NCBEILQ027S")),
    _Spec("net_worth", _G1, "Household net worth / income", "Asset prices vs the income that must service them",
          "", +1, _panel_col("Household net worth / DPI")),
    _Spec("home_price", _G1, "Home price / income", "Housing valuation vs household income",
          "", +1, _panel_col("Home price / income")),
    # --- Private leverage & credit
    _Spec("hh_debt_3y", _G2, "Household debt / GDP, 3-yr change", "Pace of household leverage build-up",
          "", +1, lambda raw, p: _change(_per_gdp(raw, "CMDEBT"), 12)),
    _Spec("corp_debt_3y", _G2, "Corporate debt / GDP, 3-yr change", "Pace of corporate leverage build-up",
          "", +1, lambda raw, p: _change(_per_gdp(raw, "NCBDBIQ027S"), 12)),
    _Spec("sloos", _G2, "Banks tightening loan standards (SLOOS)", "Credit supply: tightening is how a credit boom turns",
          "% net", +1, lambda raw, p: _q(raw, "DRTSCILM"), (-100, 100)),
    # --- Monetary policy capacity
    _Spec("fed_funds", _G3, "Fed funds rate", "Room to cut: near zero means MP1 (rate cuts) is exhausted",
          "%", -1, lambda raw, p: _q(raw, "FEDFUNDS", "mean"), (-1, 25)),
    _Spec("real_policy_rate", _G3, "Real policy rate (fed funds − core PCE)", "How tight policy is after inflation",
          "pp", 0, _real_policy_rate, (-20, 25)),
    _Spec("core_pce", _G3, "Core PCE inflation (yoy)", "Inflation limits how far the Fed can ease",
          "% yoy", +1, lambda raw, p: (None if _q(raw, "PCEPILFE") is None
                                       else _yoy(_q(raw, "PCEPILFE")).dropna()), (-5, 20)),
    _Spec("fed_bills", _G3, "Fed Treasury-bill holdings, 12-mo change", "Dalio’s watch item: bill buying broadening into money printing",
          "$bn", 0, _fed_bills_chg, (-3000, 3000)),
    _Spec("fed_bs_1y", _G3, "Fed balance sheet / GDP, 1-yr change", "Money printing (MP2): expands in crises, shrinks in tightening",
          "", 0, lambda raw, p: _change(_per_gdp(raw, "WALCL"), 4)),
]


@dataclass
class FiscalMetric:
    key: str
    group: str
    name: str
    why: str
    unit: str                        # "" -> raw value not shown, percentile only
    risk_dir: int                    # +1 / -1 / 0 (see _Spec)
    latest: float                    # latest raw value (NaN if unit hidden)
    percentile: float                # 0-100 percentile of the raw value vs own history
    risk_percentile: Optional[float]  # oriented so high = stretched; None if two-sided or short
    as_of: str
    n_quarters: int
    history: List[float] = field(default_factory=list)       # last 40q raw, for a sparkline
    history_q: List[str] = field(default_factory=list)
    mode: str = "pct"
    toward_risk: Optional[bool] = None   # chg12m rows: did the last 12 months move the risky way?


def build_fiscal_panel(raw: Dict[str, pd.Series], panel) -> List[FiscalMetric]:
    out: List[FiscalMetric] = []
    for sp in _SPECS:
        try:
            s = sp.build(raw, panel)
        except Exception:
            s = None
        if s is None or s.dropna().empty:
            continue
        s = s.dropna()
        if sp.mode == "chg12m":
            # Short published history (e.g. TIC: latest 13 months): report the
            # 12-month change; there is no history to rank against.
            if len(s) < 13:
                continue
            chg = float(s.iloc[-1] / s.iloc[-13] - 1.0) * 100.0
            ok = sp.plausible is None or sp.plausible[0] <= chg <= sp.plausible[1]
            out.append(FiscalMetric(
                key=sp.key, group=sp.group, name=sp.name, why=sp.why,
                unit=sp.unit if ok else "", risk_dir=sp.risk_dir,
                latest=round(chg, 1) if ok else float("nan"),
                percentile=None, risk_percentile=None,
                as_of=str(s.index[-1].to_period("M")) if hasattr(s.index[-1], "to_period") else str(s.index[-1]),
                n_quarters=int(len(s)),
                history=[round(float(v), 1) for v in s.tail(13).values],
                history_q=[str(i)[:7] for i in s.tail(13).index],
                mode="chg12m",
                toward_risk=(None if chg == 0 else bool((chg > 0) == (sp.risk_dir > 0))),
            ))
            continue
        pct = expanding_percentile(s)
        if pct.dropna().empty:
            continue
        p = float(pct.dropna().iloc[-1])
        latest = float(s.iloc[-1])
        unit = sp.unit
        if unit and sp.plausible and not (sp.plausible[0] <= latest <= sp.plausible[1]):
            unit = ""                                  # units don't look right: show percentile only
        tail = s.tail(40)
        out.append(FiscalMetric(
            key=sp.key, group=sp.group, name=sp.name, why=sp.why, unit=unit, risk_dir=sp.risk_dir,
            latest=round(latest, 1) if unit else float("nan"),
            percentile=round(p),
            risk_percentile=(round(p) if sp.risk_dir > 0 else round(100 - p) if sp.risk_dir < 0 else None),
            as_of=str(s.index[-1]),
            n_quarters=int(s.shape[0]),
            history=[round(float(v), 4) for v in tail.values],
            history_q=[str(q) for q in tail.index],
        ))
    return out
