"""
Tests for debt_cycle_scoring.scorer.validation. Run with:
    python3 -m debt_cycle_scoring.scorer.test_validation
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .validation import walk_forward_predict, calibration_report, _mask_after


def _qidx(start: str, n: int) -> pd.PeriodIndex:
    return pd.period_range(start, periods=n, freq="Q")


@dataclass
class FakeSpec:
    category: str
    higher_is_later_cycle: object


def _make_panel(n=120):
    idx = _qidx("1990Q1", n)
    rng = np.random.default_rng(0)
    # a clear cyclical structure so stages are learnable
    t = np.arange(n)
    cycle = np.sin(2 * np.pi * t / 40)  # ~10-year cycle
    scored = pd.DataFrame(
        {
            "debtA": 5 + cycle * 3 + rng.normal(0, 0.2, n),
            "debtB": 5 + cycle * 2.5 + rng.normal(0, 0.2, n),
            "rate": 3 + np.roll(cycle, 5) * 2 + rng.normal(0, 0.2, n),
        },
        index=idx,
    )
    specs = {
        "debtA": FakeSpec("debt_burden", True),
        "debtB": FakeSpec("debt_burden", True),
        "rate": FakeSpec("cb_stance", True),
    }
    # labels follow the cycle: peaks -> deleveraging, troughs -> reflation, rising -> bubble, low -> early
    labels = pd.Series("early", index=idx)
    labels[cycle > 0.5] = "bubble"
    labels[cycle < -0.5] = "reflation"
    labels[(cycle >= -0.5) & (cycle <= 0.5) & (np.gradient(cycle) < 0)] = "deleveraging_beautiful"
    # training mask: every 2nd quarter is "confirmed" (plenty of training data)
    mask = pd.Series(False, index=idx)
    mask.iloc[::2] = True
    return scored, specs, labels, mask, idx


def test_walk_forward_produces_out_of_sample_predictions():
    scored, specs, labels, mask, idx = _make_panel()
    wf = walk_forward_predict(scored, specs, labels, mask, min_train_per_start=15, refit_every=4)
    assert len(wf.predictions) > 0
    # probabilities sum to ~1 each quarter
    assert np.allclose(wf.predictions.sum(axis=1).values, 1.0, atol=1e-6)
    # predictions start only after enough training history
    assert wf.first_scored > idx[10]


def test_walk_forward_no_leakage_future_labels_do_not_change_past_prediction():
    # THE leakage-proof test. Predict the whole series, then CORRUPT all
    # labels from quarter 80 onward and predict again. Every prediction
    # for quarters < 80 must be byte-identical -- if any changed, future
    # labels leaked into a past prediction.
    scored, specs, labels, mask, idx = _make_panel()
    wf_a = walk_forward_predict(scored, specs, labels, mask, min_train_per_start=15, refit_every=1)

    corrupted = labels.copy()
    corrupted.iloc[80:] = "bubble"  # nonsense from q80 on
    wf_b = walk_forward_predict(scored, specs, corrupted, mask, min_train_per_start=15, refit_every=1)

    common_early = [q for q in wf_a.predictions.index if q < idx[80] and q in wf_b.predictions.index]
    assert len(common_early) > 5, "need enough early quarters to make the test meaningful"
    a = wf_a.predictions.loc[common_early]
    b = wf_b.predictions.loc[common_early]
    assert np.allclose(a.values, b.values, atol=1e-9), (
        "predictions before q80 changed when only post-q80 labels were corrupted -- FUTURE LABEL LEAKAGE"
    )


def test_mask_after_blanks_at_and_after_cutoff():
    idx = _qidx("2000Q1", 6)
    s = pd.Series(["early", "bubble", "top", "reflation", "early", "bubble"], index=idx)
    cutoff = idx[3]
    out = _mask_after(s, cutoff)
    assert out.iloc[0] == "early" and out.iloc[2] == "top"      # before cutoff kept
    assert pd.isna(out.iloc[3]) and pd.isna(out.iloc[5])         # at/after cutoff blanked

    bmask = pd.Series([True] * 6, index=idx)
    bout = _mask_after(bmask, cutoff, as_bool=True)
    assert bout.iloc[2] == True and bout.iloc[3] == False and bout.iloc[5] == False
    assert bout.dtype == bool


def test_calibration_report_runs_and_beats_or_matches_baseline_structure():
    scored, specs, labels, mask, idx = _make_panel()
    wf = walk_forward_predict(scored, specs, labels, mask, min_train_per_start=15, refit_every=2)
    cal = calibration_report(wf, n_bins=5)
    # brier scores are valid probabilities-of-error in [0, 2] for multiclass
    assert 0 <= cal.brier_score <= 2
    assert 0 <= cal.baseline_brier <= 2
    assert cal.n_evaluated > 0
    assert 0 <= cal.top1_accuracy <= 1
    # reliability table has the expected columns
    assert set(["bin", "mean_predicted", "observed_freq", "n"]).issubset(cal.reliability_table.columns)


def test_calibration_brier_perfect_predictions_is_zero():
    # sanity: if predictions are one-hot correct, Brier should be ~0
    idx = _qidx("2000Q1", 4)
    from .validation import WalkForwardResult
    stages = ["early", "bubble"]
    preds = pd.DataFrame([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]], index=idx, columns=stages)
    actual = pd.Series(["early", "bubble", "early", "bubble"], index=idx)
    wf = WalkForwardResult(
        predictions=preds, predicted_top=preds.idxmax(axis=1), actual_label=actual,
        risk_score=pd.Series(50.0, index=idx), n_train_at_each_step=pd.Series(10, index=idx),
        first_scored=idx[0],
    )
    cal = calibration_report(wf, n_bins=5)
    assert cal.brier_score < 1e-9, "perfect one-hot predictions should give ~0 Brier"
    assert cal.top1_accuracy == 1.0


def run_all():
    tests = [obj for name, obj in list(globals().items()) if name.startswith("test_")]
    failures = []
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failures.append(t.__name__)
            print(f"FAIL  {t.__name__}: {e}")
        except Exception as e:
            failures.append(t.__name__)
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print()
    if failures:
        print(f"{len(failures)} of {len(tests)} tests FAILED: {failures}")
        raise SystemExit(1)
    print(f"All {len(tests)} tests passed.")


if __name__ == "__main__":
    run_all()
