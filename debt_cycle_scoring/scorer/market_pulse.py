"""
Daily market pulse -- scorer/market_pulse.py

The fastest-moving bond and equity readings, at the frequency FRED actually
publishes them (once per business day, after the close). This is a DISPLAY
panel: it is never blended into the quarterly structural composite, which
would let day-to-day market noise contaminate the credit-cycle signal.

For each series:
  * latest close and its date (series post on different lags -- shown, not hidden)
  * change over 1 business day, 1 week (5 obs), 1 month (21 obs)
  * percentile of the latest level vs the trailing 10 years of daily closes
  * which direction counts as MORE risk, fixed from theory in advance, so the
    page can say whether the last month's move was toward or away from risk
  * a 90-business-day sparkline
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# sid, name, unit, change unit, risk direction (+1 higher = risk, -1 lower = risk, 0 two-sided), transform
PULSE_SPECS = [
    ("DGS10", "10y Treasury yield", "%", "bp", +1, "level"),
    ("T10Y2Y", "10y − 2y spread", "pp", "bp", -1, "level"),
    ("T10Y3M", "10y − 3m spread", "pp", "bp", -1, "level"),
    ("BAA10Y", "Baa corporate − 10y Treasury spread", "pp", "bp", +1, "level"),
    ("T10YIE", "10y breakeven inflation", "%", "bp", 0, "level"),
    ("VIXCLS", "VIX equity volatility", "", "pts", +1, "level"),
    ("SP500", "S&P 500 drawdown from 52-week high", "%", "pp", -1, "drawdown"),
    ("DTWEXBGS", "Broad trade-weighted dollar", "", "%", +1, "level"),
]
PULSE_SERIES: List[str] = [s[0] for s in PULSE_SPECS]

WINDOW_YEARS = 10
SPARK_DAYS = 90


@dataclass
class PulseItem:
    sid: str
    name: str
    unit: str
    chg_unit: str
    risk_dir: int
    latest: float
    as_of: str
    chg_1d: float
    chg_1w: float
    chg_1m: float
    pct_10y: float                       # 0-100 percentile of latest level vs trailing 10y
    toward_risk_1m: Optional[bool]       # None for two-sided series
    pct_years: float = 10.0              # years of history the percentile actually rests on
    spark: List[float] = field(default_factory=list)
    level_close: Optional[float] = None  # for drawdown: the index level itself


def _chg(s: pd.Series, n: int, chg_unit: str) -> float:
    if len(s) <= n:
        return float("nan")
    a, b = float(s.iloc[-1]), float(s.iloc[-1 - n])
    if chg_unit == "bp":
        return (a - b) * 100.0
    if chg_unit == "%":
        return (a / b - 1.0) * 100.0 if b else float("nan")
    return a - b


def build_market_pulse(raw: Dict[str, pd.Series]) -> List[PulseItem]:
    out: List[PulseItem] = []
    for sid, name, unit, chg_unit, risk_dir, kind in PULSE_SPECS:
        s = raw.get(sid)
        if s is None:
            continue
        s = s.dropna().sort_index()
        if len(s) < 30:
            continue
        level_close = None
        if kind == "drawdown":
            level_close = float(s.iloc[-1])
            peak = s.rolling(252, min_periods=20).max()
            s = ((s / peak - 1.0) * 100.0).dropna()
        cutoff = s.index[-1] - pd.DateOffset(years=WINDOW_YEARS)
        hist = s[s.index >= cutoff]
        prior, cur = hist.iloc[:-1], hist.iloc[-1]      # mid-rank vs PRIOR closes in the window
        pct = float(((prior < cur).mean() + 0.5 * (prior == cur).mean()) * 100.0) if len(prior) else float("nan")
        c1m = _chg(s, 21, chg_unit)
        toward = None if risk_dir == 0 or np.isnan(c1m) else bool(np.sign(c1m) == risk_dir and c1m != 0)
        out.append(PulseItem(
            sid=sid, name=name, unit=unit, chg_unit=chg_unit, risk_dir=risk_dir,
            latest=round(float(s.iloc[-1]), 2), as_of=s.index[-1].strftime("%Y-%m-%d"),
            chg_1d=round(_chg(s, 1, chg_unit), 2), chg_1w=round(_chg(s, 5, chg_unit), 2),
            chg_1m=round(c1m, 2), pct_10y=round(pct), toward_risk_1m=toward,
            pct_years=round((hist.index[-1] - hist.index[0]).days / 365.25, 1),
            spark=[round(float(v), 3) for v in s.tail(SPARK_DAYS).values],
            level_close=round(level_close, 2) if level_close is not None else None,
        ))
    return out
