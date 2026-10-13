"""
Walk-forward backtest + calibration -- debt_cycle_scoring.scorer.validation

This is the module that makes every other claim about the scorer testable
rather than circular. It does two things, in dependency order:

  1. walk_forward_predict(): for each quarter T in an out-of-sample
     window, refit the scorer using ONLY information available as of T
     (metrics and training labels dated < T), then predict T. Roll
     forward. Produces genuinely out-of-sample stage predictions.

  2. calibration_report(): consumes those out-of-sample predictions and
     asks the one question that matters for an honest probabilistic
     model -- when it says "70% bubble", is it actually right ~70% of
     the time? Reported via multiclass Brier score and a reliability
     table, compared against a base-rate baseline.

Deliberately NOT here (documented rejections, not omissions):
  - Bootstrap confidence intervals: with only ~2-3 real debt cycles,
    even a block bootstrap (the only correct form given the heavy
    autocorrelation of quarterly macro data) yields intervals so wide
    they communicate nothing the training counts don't already. A naive
    IID bootstrap would be actively misleading (false precision from
    treating autocorrelated quarters as independent). Skipped on
    purpose.
  - GARCH volatility scaling: for high-frequency return series with
    volatility clustering, not ~100-260 quarterly macro observations.

Leakage discipline (the thing that makes a backtest honest):
  - FEATURES are backward-looking by construction (the engine's
    expanding percentile rank never uses a quarter's own future).
  - TRAINING LABELS for the fit at time T are restricted to quarters
    strictly before T, so a past quarter's label -- which itself was
    derived using future turning points -- cannot leak forward into a
    prediction. We accept that the labels are "finalized" ground truth
    for SCORING the prediction; we do not let them into the FIT ahead of
    time.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .engine import fit_and_score, prepare_scored_panel, TRAINABLE_STAGES


@dataclass
class WalkForwardResult:
    predictions: pd.DataFrame          # index=quarter, columns=stages, out-of-sample probabilities
    predicted_top: pd.Series           # index=quarter, argmax stage
    actual_label: pd.Series            # index=quarter, finalized ground-truth stage (trainable ones only)
    risk_score: pd.Series              # index=quarter, out-of-sample risk score
    n_train_at_each_step: pd.Series    # index=quarter, how many training rows the fit used
    first_scored: Optional[pd.Period]  # earliest quarter that could be predicted


def walk_forward_predict(
    scored_metrics: pd.DataFrame,
    metric_specs: Dict[str, object],
    stage_labels: pd.Series,
    training_mask: pd.Series,
    credit_gap: Optional[pd.Series] = None,
    min_train_per_start: int = 15,
    refit_every: int = 1,
) -> WalkForwardResult:
    """
    Walk-forward out-of-sample prediction.

    For each candidate quarter T (in chronological order, once enough
    training history exists), refit archetypes using only training-
    eligible quarters STRICTLY BEFORE T, then score T. The prediction for
    T therefore never saw T's own label or any later quarter.

    min_train_per_start: don't start predicting until at least this many
        training-eligible quarters exist before T (a fit on 3 total rows
        across all stages is meaningless).
    refit_every: refit cadence. 1 = refit every quarter (most rigorous,
        the default). Higher values refit less often for speed and reuse
        the last fit's archetypes for intervening quarters -- still
        leakage-free because the reused fit only ever saw earlier data.

    Returns out-of-sample predictions aligned to the quarters that could
    be scored. `actual_label` holds the finalized trainable label for each
    such quarter (NaN where the quarter isn't a trainable-stage example),
    for use by calibration_report.
    """
    # Precompute the full category-score panel ONCE. This is legitimate:
    # each metric's percentile rank at quarter t already uses only data
    # up to t (expanding window), so a value at t is identical whether
    # computed on the full series or a truncated one. We still restrict
    # the TRAINING LABELS per step below, which is where leakage would
    # actually occur.
    category_scores, _ = prepare_scored_panel(scored_metrics, metric_specs)
    all_quarters = category_scores.dropna(how="all").index.sort_values()

    train_labels_full = stage_labels.reindex(category_scores.index)
    train_mask_full = training_mask.reindex(category_scores.index, fill_value=False)

    pred_rows: Dict[pd.Period, Dict[str, float]] = {}
    risk_rows: Dict[pd.Period, float] = {}
    n_train_rows: Dict[pd.Period, int] = {}

    last_fit = None
    steps_since_fit = 0

    for i, T in enumerate(all_quarters):
        # training-eligible quarters strictly before T
        prior = category_scores.index[category_scores.index < T]
        prior_mask = train_mask_full.reindex(prior, fill_value=False)
        n_prior_train = int(prior_mask.sum())
        if n_prior_train < min_train_per_start:
            continue

        need_refit = (last_fit is None) or (steps_since_fit >= refit_every)
        if need_refit:
            # Build a training-only slice: metrics + labels + mask
            # restricted to quarters before T. Scoring is then requested
            # for T alone by passing a one-quarter panel.
            try:
                fit_result = fit_and_score(
                    scored_metrics=scored_metrics,
                    metric_specs=metric_specs,
                    stage_labels=_mask_after(stage_labels, T),
                    training_mask=_mask_after(training_mask, T, as_bool=True),
                    credit_gap=credit_gap,
                )
            except ValueError:
                # not enough usable training data yet under alignment
                continue
            last_fit = fit_result
            steps_since_fit = 0
        else:
            steps_since_fit += 1

        # pull T's prediction out of the (leakage-free) fit's scored output
        if last_fit is None or T not in last_fit.stage_probabilities.index:
            continue
        pred_rows[T] = last_fit.stage_probabilities.loc[T].to_dict()
        risk_rows[T] = float(last_fit.risk_score.loc[T])
        n_train_rows[T] = n_prior_train

    if not pred_rows:
        raise ValueError(
            "Walk-forward produced no predictions -- likely too few training-eligible "
            "quarters before any scorable quarter. Try lowering min_train_per_start."
        )

    predictions = pd.DataFrame.from_dict(pred_rows, orient="index").fillna(0.0)
    predicted_top = predictions.idxmax(axis=1)
    actual = train_labels_full.reindex(predictions.index)
    # only keep actuals that are trainable-stage ground truth AND were
    # training-eligible (i.e. genuinely confirmed), else NaN
    eligible = train_mask_full.reindex(predictions.index, fill_value=False)
    actual = actual.where(eligible & actual.isin(TRAINABLE_STAGES))

    return WalkForwardResult(
        predictions=predictions,
        predicted_top=predicted_top,
        actual_label=actual,
        risk_score=pd.Series(risk_rows),
        n_train_at_each_step=pd.Series(n_train_rows),
        first_scored=predictions.index.min(),
    )


def _mask_after(series: pd.Series, cutoff: pd.Period, as_bool: bool = False) -> pd.Series:
    """
    Return a copy of `series` with all entries at or after `cutoff` blanked
    (NaN for labels, False for a boolean mask). Everything the fit sees is
    therefore strictly earlier than `cutoff` -- except we must still let
    the ENGINE compute a category score for `cutoff` itself so it can be
    scored. The engine scores every scorable quarter; blanking the LABEL/
    MASK at/after cutoff is what prevents the fit from training on it.

    Note the one subtlety: the engine will still score quarters AFTER
    cutoff too (they're in the panel), but we only ever READ the cutoff
    quarter's prediction, and none of those later quarters entered the
    fit (their mask is False), so there is no leakage into the fit.
    """
    out = series.copy()
    out.loc[out.index >= cutoff] = False if as_bool else np.nan
    if as_bool:
        out = out.astype(bool)
    return out


@dataclass
class CalibrationResult:
    brier_score: float                 # multiclass Brier, lower is better
    baseline_brier: float              # base-rate baseline for comparison
    reliability_table: pd.DataFrame    # per confidence-bin: mean predicted vs observed frequency
    n_evaluated: int
    top1_accuracy: float
    per_stage_accuracy: Dict[str, float]


def calibration_report(wf: WalkForwardResult, n_bins: int = 5) -> CalibrationResult:
    """
    Calibration + accuracy on the walk-forward out-of-sample predictions.

    Only quarters with a finalized trainable ground-truth label are
    evaluated (you can't score a prediction against a NaN actual).

    Multiclass Brier score: mean over evaluated quarters of the sum of
    squared differences between the predicted probability vector and the
    one-hot actual. Compared against a baseline that always predicts the
    training base rates -- if the model can't beat that, its
    probabilities aren't adding information.

    Reliability table: bin every (stage, predicted-probability) pair by
    predicted probability, then compare the mean predicted probability in
    each bin to the observed frequency of that stage actually occurring.
    A well-calibrated model has these two columns roughly equal.
    """
    eval_idx = wf.actual_label.dropna().index
    if len(eval_idx) == 0:
        raise ValueError("No quarters with finalized ground-truth labels to calibrate against.")

    preds = wf.predictions.loc[eval_idx]
    actuals = wf.actual_label.loc[eval_idx]
    stages = list(preds.columns)

    # one-hot actuals aligned to prediction columns
    onehot = pd.DataFrame(0.0, index=eval_idx, columns=stages)
    for q in eval_idx:
        a = actuals.loc[q]
        if a in onehot.columns:
            onehot.loc[q, a] = 1.0

    # multiclass Brier
    brier = float(((preds[stages] - onehot[stages]) ** 2).sum(axis=1).mean())

    # base-rate baseline: predict the observed class frequencies every quarter
    base_rates = onehot[stages].mean(axis=0)
    baseline_pred = pd.DataFrame(
        np.tile(base_rates.values, (len(eval_idx), 1)), index=eval_idx, columns=stages
    )
    baseline_brier = float(((baseline_pred[stages] - onehot[stages]) ** 2).sum(axis=1).mean())

    # reliability table over all (quarter, stage) probability pairs
    flat_pred = preds[stages].values.flatten()
    flat_actual = onehot[stages].values.flatten()
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(flat_pred, bins) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = bin_idx == b
        if m.sum() == 0:
            continue
        rows.append({
            "bin": f"{bins[b]:.1f}-{bins[b+1]:.1f}",
            "mean_predicted": float(flat_pred[m].mean()),
            "observed_freq": float(flat_actual[m].mean()),
            "n": int(m.sum()),
        })
    reliability_table = pd.DataFrame(rows)

    # accuracy
    correct = (wf.predicted_top.loc[eval_idx] == actuals)
    top1 = float(correct.mean())
    per_stage = {}
    for stage in stages:
        stage_mask = actuals == stage
        if stage_mask.sum() > 0:
            per_stage[stage] = float(correct[stage_mask].mean())

    return CalibrationResult(
        brier_score=brier,
        baseline_brier=baseline_brier,
        reliability_table=reliability_table,
        n_evaluated=len(eval_idx),
        top1_accuracy=top1,
        per_stage_accuracy=per_stage,
    )
