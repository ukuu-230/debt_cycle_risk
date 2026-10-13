"""
Source A -- NBER short-term cycle turning points, design doc section 3B.1.

Turns the raw USREC 0/1 recession-indicator series (already ingestible via
the FRED catalog -- see ingestion/catalog.py) into explicit peak/trough
turning-point dates, following NBER's own dating convention: the peak is
the LAST quarter of expansion (the quarter just before recession begins),
and the trough is the LAST quarter of the recession (the quarter just
before expansion resumes).
"""
from __future__ import annotations

import pandas as pd


def extract_recession_turning_points(usrec: pd.Series) -> pd.DataFrame:
    """
    usrec: a quarterly series of 0/1 values (1 = in recession that
    quarter), indexed by a pandas PeriodIndex(freq="Q") -- i.e. already
    resampled the way ingestion/panel.py produces it.

    Returns a DataFrame indexed by period with one column, `event`,
    valued "peak" or "trough".
    """
    s = usrec.dropna().astype(int)
    diffs = s.diff()
    idx = s.index

    events = []
    for i in range(1, len(s)):
        if diffs.iloc[i] == 1:
            # transition 0 -> 1: recession starts THIS quarter, so the
            # PEAK (last expansion quarter) is the one before it.
            events.append((idx[i - 1], "peak"))
        elif diffs.iloc[i] == -1:
            # transition 1 -> 0: recession ended, meaning the TROUGH
            # (last recession quarter) is the one before this one.
            events.append((idx[i - 1], "trough"))

    result = pd.DataFrame(events, columns=["period", "event"]).set_index("period")
    return result
