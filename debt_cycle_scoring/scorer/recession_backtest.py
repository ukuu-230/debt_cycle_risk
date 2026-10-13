"""
Composite-risk recession-predictiveness backtest -- scorer/recession_backtest.py

Tests the question the whole tool exists to answer: does a HIGH composite
risk reading actually PRECEDE a recession? Uses NBER recession onsets
(from USREC, the same authoritative source the labeling pipeline uses) as
the outcome, at multiple lead horizons (1Q / 4Q / 8Q ahead), because the
credit-gap-driven signal is theorized to lead by YEARS, not quarters --
so we need to see at what horizon (if any) the signal actually lives.

HONESTY CONSTRAINTS baked in (stated, not hidden):
  1. Few events. There are only ~7-8 NBER recessions in the post-1962
     sample, and they are not independent (a recession spans several
     quarters). So this is reported as DESCRIPTIVE evidence -- do high-
     risk quarters cluster ahead of recessions? -- plus a rank statistic
     (AUC) with an explicit "small-sample, interpret with caution" label.
     No p-value theater that implies certainty the data can't support.
  2. Not circular. The composite risk is NOT fit to recessions; its flag
     thresholds come from external research. USREC is finalized NBER
     data. So risk-vs-recession is a fair out-of-sample-style check.
     (The credit GAP does inform both the risk score and, separately, the
     labeling pipeline -- but recession ONSET here is raw NBER, not our
     labels, so there is no label leakage.)
  3. Overlap caveat. AUC treats each quarter as a case; with multi-quarter
     recessions and autocorrelated risk, the effective sample is far
     smaller than the row count. The report says so.

Outcome definition: for horizon H, a quarter is a POSITIVE case if an
NBER recession BEGINS within the next H quarters (strictly after the
current quarter, through +H). Quarters already IN a recession are
excluded from the "predict the onset" test -- you're testing foresight,
not coincidence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


DEFAULT_HORIZONS = [1, 4, 8]


@dataclass
class HorizonResult:
    horizon: int
    n_cases: int                     # quarters evaluated (not in-recession, risk present)
    n_positive: int                  # of those, how many preceded a recession onset within H
    mean_risk_before_recession: float
    mean_risk_otherwise: float
    auc: float                       # rank discrimination: P(risk higher before a recession)
    auc_note: str


@dataclass
class RecessionBacktestResult:
    horizons: List[HorizonResult]
    recession_onsets: List[str]      # the onset quarters used, for transparency
    n_recessions_in_sample: int
    risk_at_each_onset: pd.DataFrame  # risk 0-8Q BEFORE each onset, so you can eyeball the lead profile


def _recession_onset_quarters(usrec_quarterly: pd.Series) -> List[pd.Period]:
    """
    Quarters where a recession BEGINS: USREC goes 0 -> 1. USREC is monthly
    0/1; caller passes it already resampled to quarterly (max, so any
    recession month in the quarter marks it 1).
    """
    s = usrec_quarterly.dropna().astype(int)
    onsets = []
    # Start from the FIRST observed state, not 0: a sample that begins mid-
    # recession must not record a spurious onset in its first quarter.
    prev = int(s.iloc[0]) if len(s) else 0
    for period, val in s.items():
        if val == 1 and prev == 0:
            onsets.append(period)
        prev = val
    return onsets


def _auc(scores_pos: np.ndarray, scores_neg: np.ndarray) -> float:
    """
    Area under ROC via the Mann-Whitney U relation: the probability that a
    randomly chosen positive case has a higher risk than a randomly chosen
    negative one. 0.5 = no discrimination, 1.0 = perfect. Computed directly
    (no sklearn dependency).
    """
    n_pos, n_neg = len(scores_pos), len(scores_neg)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    all_scores = np.concatenate([scores_pos, scores_neg])
    ranks = pd.Series(all_scores).rank().values
    rank_pos_sum = ranks[:n_pos].sum()
    u = rank_pos_sum - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def _split_cases(score: pd.Series, usrec: pd.Series, onsets: List[pd.Period], H: int,
                 last_observed: Optional[pd.Period] = None):
    """
    Positive = an NBER onset falls in (q, q+H]; quarters already in recession
    are excluded (foresight, not coincidence). A quarter with NO onset in its
    window is only a valid negative if the whole window (q, q+H] has been
    observed; otherwise its outcome is unknown (right-censored) and it is
    dropped. A window that already CONTAINS an onset is a known positive even
    if it runs past the data. Returns (pos_scores, neg_scores).
    """
    pos, neg = [], []
    onset_ords = [o.ordinal for o in onsets]
    last_ord = last_observed.ordinal if last_observed is not None else None
    for q, r in score.items():
        if pd.isna(r):
            continue
        u = usrec.get(q)
        if pd.notna(u) and u == 1:
            continue
        hit = any(q.ordinal + 1 <= o <= q.ordinal + H for o in onset_ords)
        if not hit and last_ord is not None and q.ordinal + H > last_ord:
            continue          # no onset seen yet AND window not fully observed -> outcome unknown
        (pos if hit else neg).append(r)
    return pos, neg


# NBER announces business-cycle peaks 6-21 months after the fact, while FRED
# fills USREC with 0 right up to the current month. So the most recent
# quarters' "no recession" is provisional, not observed. Treat the last
# NBER_DATING_LAG quarters as unobserved when censoring negatives.
NBER_DATING_LAG = 4


def _last_observed(usrec_quarterly: pd.Series, dating_lag: int = NBER_DATING_LAG) -> Optional[pd.Period]:
    s = usrec_quarterly.dropna()
    return (s.index.max() - dating_lag) if len(s) else None


def auc_at_horizon(score: pd.Series, usrec_quarterly: pd.Series, horizon: int) -> float:
    """Recession-lead AUC of any 0-100 score at one horizon (same case definition as the backtest)."""
    score = score.dropna().sort_index()
    usrec = usrec_quarterly.reindex(score.index)
    onsets = _recession_onset_quarters(usrec_quarterly)
    pos, neg = _split_cases(score, usrec, onsets, horizon, _last_observed(usrec_quarterly))
    return _auc(np.array(pos), np.array(neg))


def flag_attribution(risk_result, usrec_quarterly: pd.Series, horizon: int = 8) -> pd.DataFrame:
    """
    Which flags carry the composite's recession-lead AUC?

      standalone_auc  the flag alone (on the quarters where it exists)
      drop_one_auc    research composite rebuilt WITHOUT this flag, on the
                      same quarters as `composite_auc` for a fair comparison
      delta           composite_auc - drop_one_auc (>0: the flag helps)

    DESCRIPTIVE ONLY. Re-weighting flags from this table would fit the
    weights to ~8 recessions -- exactly the overfitting the fixed
    research weights exist to avoid. Use it to spot a flag that is
    actively hurting, or a result that rests on one flag.
    """
    from .risk_flags import compute_composite_risk
    flags = risk_result.flags
    gap = risk_result.gap_used if len(risk_result.gap_used) else pd.Series(dtype=float)
    # one pass over quarters: plain dicts of the present flags (fast, no per-flag .loc)
    present_by_q = {q: {k: float(v) for k, v in row.items() if pd.notna(v)}
                    for q, row in zip(flags.index, flags.to_dict("records"))}
    gap_by_q = {q: gap.get(q, np.nan) for q in flags.index}
    full = pd.Series({q: compute_composite_risk(fv, gap_by_q[q]).research_weighted
                      for q, fv in present_by_q.items()})      # rebuilt from the same stored flags
    rows = []
    for f in flags.columns:
        dropped = pd.Series({q: compute_composite_risk({k: v for k, v in fv.items() if k != f},
                                                       gap_by_q[q]).research_weighted
                             for q, fv in present_by_q.items()})
        common = full.dropna().index.intersection(dropped.dropna().index)
        base = auc_at_horizon(full.reindex(common), usrec_quarterly, horizon)
        d1 = auc_at_horizon(dropped.reindex(common), usrec_quarterly, horizon)
        rows.append(dict(
            flag=f, quarters=int(flags[f].notna().sum()),
            standalone_auc=round(auc_at_horizon(flags[f], usrec_quarterly, horizon), 3),
            composite_auc=round(base, 3), drop_one_auc=round(d1, 3),
            delta=round(base - d1, 3),
        ))
    return pd.DataFrame(rows).set_index("flag")


def run_recession_backtest(
    risk: pd.Series,
    usrec_quarterly: pd.Series,
    horizons: Optional[List[int]] = None,
) -> RecessionBacktestResult:
    """
    risk: composite risk series (0-100), indexed by quarterly PeriodIndex.
        Pass the research-weighted composite from build_risk_series.
    usrec_quarterly: NBER recession 0/1 indicator resampled to quarterly
        (quarter = 1 if any month in it was a recession month).
    horizons: lead horizons in quarters to test (default 1, 4, 8).
    """
    horizons = horizons or DEFAULT_HORIZONS
    risk = risk.dropna().sort_index()
    usrec = usrec_quarterly.reindex(risk.index)  # align to scored quarters

    onsets = _recession_onset_quarters(usrec_quarterly)
    onsets_in_range = [o for o in onsets if risk.index.min() <= o <= risk.index.max()]

    horizon_results: List[HorizonResult] = []
    for H in horizons:
        pos_scores, neg_scores = _split_cases(risk, usrec, onsets, H, _last_observed(usrec_quarterly))
        pos_arr, neg_arr = np.array(pos_scores), np.array(neg_scores)
        auc = _auc(pos_arr, neg_arr)
        eff_events = len([o for o in onsets_in_range])
        note = (
            f"small sample: {eff_events} recessions drive this; effective "
            f"independent cases << {len(pos_arr) + len(neg_arr)} quarters -- "
            f"treat as descriptive, not conclusive"
        )
        horizon_results.append(HorizonResult(
            horizon=H,
            n_cases=len(pos_arr) + len(neg_arr),
            n_positive=len(pos_arr),
            mean_risk_before_recession=round(float(pos_arr.mean()), 1) if len(pos_arr) else float("nan"),
            mean_risk_otherwise=round(float(neg_arr.mean()), 1) if len(neg_arr) else float("nan"),
            auc=round(auc, 3) if not np.isnan(auc) else float("nan"),
            auc_note=note,
        ))

    # lead profile: risk at 0..8 quarters BEFORE each onset. Look values up
    # by PERIOD ORDINAL through an explicit map -- never by Series.get with an
    # int, which pandas treats as a deprecated positional lookup on a
    # PeriodIndex and would both warn and return the wrong row.
    ord_to_val = {p.ordinal: v for p, v in risk.items()}
    lead_rows = {}
    for o in onsets_in_range:
        lead_rows[str(o)] = {f"-{k}Q": round(float(ord_to_val.get(o.ordinal - k)), 1)
                             if ord_to_val.get(o.ordinal - k) is not None else np.nan
                             for k in range(0, 9)}
    risk_at_each_onset = pd.DataFrame.from_dict(lead_rows, orient="index")

    return RecessionBacktestResult(
        horizons=horizon_results,
        recession_onsets=[str(o) for o in onsets_in_range],
        n_recessions_in_sample=len(onsets_in_range),
        risk_at_each_onset=risk_at_each_onset,
    )
