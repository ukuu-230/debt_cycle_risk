"""
debt_cycle_scoring.labeling
============================

Implements the section 3B independent labeling pass: NBER recession
turning points (Source A), the BIS-style credit-to-GDP gap (Source B),
Bry-Boschan/Harding-Pagan mechanical turning-point detection (Source C),
and the consensus combiner (section 3B.4) that turns all three into
confidence-tiered stage labels -- the ground truth `stages.fit_archetypes`
(in the parent package) trains on, replacing the synthetic hand-built
labels used in the earlier demos.
"""
from .nber import extract_recession_turning_points
from .hp_filter import hp_filter_two_sided, one_sided_hp_trend
from .credit_gap import compute_credit_gap, BASEL_LOWER_THRESHOLD, BASEL_UPPER_THRESHOLD
from .turning_points import bry_boschan_turning_points
from .consensus import build_consensus_labels

__all__ = [
    "extract_recession_turning_points",
    "hp_filter_two_sided",
    "one_sided_hp_trend",
    "compute_credit_gap",
    "BASEL_LOWER_THRESHOLD",
    "BASEL_UPPER_THRESHOLD",
    "bry_boschan_turning_points",
    "build_consensus_labels",
]
