"""
Fiscal & valuation context panel -- scorer/fiscal_panel.py

The government-debt and asset-valuation side of Dalio's framework, which the
composite risk score structurally cannot see:

  * The BIS credit-to-GDP gap covers the PRIVATE non-financial sector only.
  * The composite is validated against NBER recessions, and the US has had no
    fiscal or currency crisis in the FRED era -- so these metrics cannot be
    validated (the metric-gap audit found debt/GDP and the deficit score
    AUC ~0.3: deficits balloon AFTER recessions, so blending them in would
    actively damage the recession signal).

Therefore: this panel is DISPLAYED, never SCORED. Each metric is shown as a
backward-looking expanding percentile against its own history (no
look-ahead), plus its latest raw value where the units are meaningful. There
is deliberately no panel composite -- any weight would be a guess.

The metric set is chosen from theory (Dalio, *How Countries Go Broke*), NOT
by which metrics happen to read extreme today -- choosing the extreme ones
would be selection bias.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

# FRED series this panel needs beyond the main catalog (fetch these too).
FISCAL_SERIES: List[str] = [
    "GFDEGDQ188S",       # federal debt, % GDP (Q)
    "FYFSGDA188S",       # federal surplus(+)/deficit(-), % GDP (A)
    "A091RC1Q027SBEA",   # federal interest payments, $bn SAAR (Q)
    "FGRECPT",           # federal current receipts, $bn SAAR (Q)
    "THREEFYTP10",       # Kim-Wright 10y term premium, pp (D)
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
    """Backward-looking percentile (0-100) of each value vs all PRIOR values."""
    def _p(w: np.ndarray) -> float:
        *hist, cur = w
        h = np.asarray(hist, dtype=float)
        return float((h < cur).mean() + 0.5 * (h == cur).mean()) if len(h) else np.nan   # mid-rank
    return s.dropna().expanding(min_periods=min_periods + 1).apply(_p, raw=True) * 100.0


@dataclass
class FiscalMetric:
    key: str
    name: str
    why: str                         # the Dalio mechanism, one line
    unit: str                        # "" -> raw value not meaningful, show percentile only
    latest: float                    # latest raw value (NaN if unit == "")
    percentile: float                # 0-100 vs own history
    as_of: str
    n_quarters: int
    history: List[float] = field(default_factory=list)       # last 40q raw, for a sparkline
    history_q: List[str] = field(default_factory=list)


def _q(raw: Dict[str, pd.Series], sid: str, how: str = "last") -> Optional[pd.Series]:
    s = raw.get(sid)
    return None if s is None or s.dropna().empty else to_quarterly(s, how)


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


def _panel_col(name: str):
    def f(raw, panel):
        for df in (panel.scored_metrics, panel.reference_series):
            if name in df.columns:
                return df[name].dropna()
        return None
    return f


_SPECS: List[tuple] = [
    # key, name, why, unit, builder
    ("fed_debt", "Federal debt / GDP", "Sovereign debt stock -- the leg the private credit gap cannot see",
     "% GDP", lambda raw, p: _q(raw, "GFDEGDQ188S")),
    ("deficit", "Federal deficit / GDP", "Annual new borrowing; large deficits outside recession = structural",
     "% GDP", _deficit),
    ("interest", "Federal interest / receipts", "Debt-service squeeze on the sovereign (How Countries Go Broke)",
     "% receipts", _interest_receipts),
    ("term_premium", "10y term premium (Kim-Wright)", "Compensation bond buyers demand to hold duration",
     "pp", lambda raw, p: _q(raw, "THREEFYTP10", "mean")),
    ("net_worth", "Household net worth / income", "Asset prices vs the income that must service them",
     "", _panel_col("Household net worth / DPI")),
    ("home_price", "Home price / income", "Housing valuation vs household income",
     "", _panel_col("Home price / income")),
]


def build_fiscal_panel(raw: Dict[str, pd.Series], panel) -> List[FiscalMetric]:
    out: List[FiscalMetric] = []
    for key, name, why, unit, builder in _SPECS:
        try:
            s = builder(raw, panel)
        except Exception:
            s = None
        if s is None or s.dropna().empty:
            continue
        s = s.dropna()
        pct = expanding_percentile(s)
        if pct.dropna().empty:
            continue
        tail = s.tail(40)
        out.append(FiscalMetric(
            key=key, name=name, why=why, unit=unit,
            latest=round(float(s.iloc[-1]), 1) if unit else float("nan"),
            percentile=round(float(pct.dropna().iloc[-1])),
            as_of=str(s.index[-1]),
            n_quarters=int(s.shape[0]),
            history=[round(float(v), 3) for v in tail.values],
            history_q=[str(q) for q in tail.index],
        ))
    return out
