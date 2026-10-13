"""
Source C -- mechanical turning-point detection, design doc section 3B.3.

A simplified implementation of the Bry-Boschan (1971) turning-point
logic, adapted to quarterly data the way Harding & Pagan's "BBQ"
algorithm does: find local extrema within a window, then prune them so
peaks and troughs strictly alternate and no phase (peak-to-trough or
trough-to-peak) is shorter than a minimum length.

This is the most independent of the three label sources, since it
applies a fixed, published, purely mechanical rule directly to whatever
series is passed in (e.g. the credit-to-GDP ratio, or real GDP) -- no
institution's crisis judgment is involved at all, unlike NBER's dating
committee or the BIS's choice of HP filter parameters.

Simplifications vs. the full academic Bry-Boschan/BBQ specification:
enforces alternation and a minimum PHASE length, but does not enforce
a separate minimum full-CYCLE length or the original algorithm's
additional endpoint-handling rules. Good enough as an independent
cross-check per section 3B; would need hardening before being treated
as a citable replication of the academic algorithm.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd


def _find_local_extrema(values: np.ndarray, window: int) -> Tuple[List[int], List[int]]:
    peaks, troughs = [], []
    n = len(values)
    for t in range(window, n - window):
        segment = values[t - window : t + window + 1]
        center = window
        if values[t] == segment.max() and np.argmax(segment) == center:
            peaks.append(t)
        if values[t] == segment.min() and np.argmin(segment) == center:
            troughs.append(t)
    return peaks, troughs


def bry_boschan_turning_points(
    series: pd.Series,
    window: int = 2,
    min_phase_quarters: int = 2,
    smooth_window: int = 1,
) -> pd.DataFrame:
    """
    series: quarterly series (e.g. credit-to-GDP ratio, real GDP level).
    window: how many quarters on each side to compare when looking for a
        local max/min. 2 is the standard quarterly-data choice.
    min_phase_quarters: minimum distance between consecutive turning
        points. Candidate turning points closer together than this are
        dropped (the "too choppy to be a real cycle phase" filter).
    smooth_window: size of a CENTERED moving average applied to the
        series before searching for local extrema (1 = no smoothing,
        the original behavior of this function). The classic
        Bry-Boschan (1971) methodology smooths the series first
        specifically so short-term noise doesn't register as a false
        turning point -- omitted from the first version of this
        function, and confirmed to matter running this against real
        BIS credit-to-GDP data: without smoothing, a spurious
        mechanical "peak" fired right at the genuine 2009Q2 NBER
        trough, only correctly overridden because consensus.py's
        `conflicting` tier caught the resulting disagreement rather
        than mislabeling it outright. A smooth_window of 3-5 quarters
        is a reasonable starting point for a series like the
        credit-to-GDP ratio; treat this as a real parameter to tune
        against your own data, not a validated default.

    Returns a DataFrame indexed by period with one column, `event`
    ("peak" or "trough"), guaranteed to alternate. Turning points are
    reported at their ORIGINAL (unsmoothed) period -- smoothing only
    affects which points get DETECTED as extrema, not the timestamp
    reported for them.
    """
    working = series.dropna()
    if smooth_window > 1:
        smoothed_values = working.rolling(window=smooth_window, center=True, min_periods=1).mean().values
    else:
        smoothed_values = working.values

    values = smoothed_values  # used for both detection and "which is more extreme" comparisons below
    index = working.index
    peaks_idx, troughs_idx = _find_local_extrema(values, window)

    candidates = sorted([(i, "peak") for i in peaks_idx] + [(i, "trough") for i in troughs_idx])
    if not candidates:
        return pd.DataFrame(columns=["event"]).rename_axis("period")

    # Iterate merge-same-type-adjacent + prune-too-close-in-time to a fixed
    # point. A single pass of "prune, then done" is NOT safe: dropping an
    # interior turning point can leave two same-type points adjacent to
    # each other on either side of the gap it left behind (e.g. pruning a
    # peak between two troughs leaves trough-trough adjacent), so
    # alternation has to be re-checked after every prune, not just once.
    current: List[Tuple[int, str]] = candidates
    changed = True
    while changed:
        changed = False

        merged: List[Tuple[int, str]] = [current[0]]
        for i, kind in current[1:]:
            prev_i, prev_kind = merged[-1]
            if kind == prev_kind:
                changed = True
                if kind == "peak" and values[i] > values[prev_i]:
                    merged[-1] = (i, kind)
                elif kind == "trough" and values[i] < values[prev_i]:
                    merged[-1] = (i, kind)
                # else: drop the new candidate, keep the existing (more extreme) one
            else:
                merged.append((i, kind))

        pruned: List[Tuple[int, str]] = [merged[0]]
        for i, kind in merged[1:]:
            prev_i, _ = pruned[-1]
            if i - prev_i < min_phase_quarters:
                changed = True
                continue
            pruned.append((i, kind))

        current = pruned

    result = pd.DataFrame(current, columns=["position", "event"])
    result["period"] = index[result["position"]]
    return result.set_index("period")[["event"]]
