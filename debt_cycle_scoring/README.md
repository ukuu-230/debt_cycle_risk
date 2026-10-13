# debt_cycle_scoring

Reference implementation of the scoring math from the design doc,
section 3A. This is the calculation engine only — it does **not** pull
live FRED/BIS data yet (that's the ingestion layer, still to be built).
It takes a DataFrame of already-fetched metric time series and turns it
into stage probabilities, kept deliberately separate so the scoring math
is independently testable and independently backtestable before any
API/ingestion code exists.

## Files

| File | Design doc section | What it does |
|---|---|---|
| `normalize.py` | 3A.1 | Expanding-window percentile rank, shrinkage z-score for short-history metrics, winsorizing, direction encoding |
| `aggregate.py` | 3A.2 | PCA per category — turns correlated metric scores into one category score, weighted by the data's own covariance rather than hand-picked weights |
| `stages.py` | 3A.3 | Mahalanobis-distance analog matching — fits a centroid + covariance per labeled historical stage, then scores any new observation as a probability distribution over stages |
| `explain.py` | 3A.5 | Decomposes a stage's Mahalanobis distance into per-category contributions, so every probability is traceable to what's driving it |
| `demo.py` | — | Runs the full pipeline end-to-end on synthetic data shaped like a real bubble → top → deleveraging episode |
| `test_scoring.py` | — | Correctness tests on the actual math (no-look-ahead, direction consistency, distances/probabilities behaving as claimed) |

## Running it

```bash
python3 -m debt_cycle_scoring.test_scoring   # unit tests
python3 -m debt_cycle_scoring.demo           # end-to-end synthetic demo
```

## What the demo shows (and why it's a good sign)

The synthetic demo is hand-built to trace a specific path: early cycle →
bubble → top → beautiful deleveraging → reflation. Running it produces
exactly the pattern the design doc's section 7.3 says a working model
*should* produce: correct calls on the unambiguous early/bubble quarters,
and confusion concentrated on the **boundary** quarters between adjacent
stages (e.g., a late-bubble quarter reading as partly "top", a
just-turned deleveraging quarter still carrying some "top" probability).
That's the expected, healthy failure mode — real transitions between
adjacent stages actually are gradual. Confusion between *distant* stages
(e.g., a bubble quarter reading as deleveraging) would be the real red
flag, and the tests/demo don't show that here.

## What's deliberately NOT here yet

- **Data ingestion** (FRED/BIS/BOJ/ECB API clients) — section 4 of the
  design doc.
- **The independent labeling pass** (section 3B: NBER `USREC`, BIS
  credit-to-GDP gap, Bry-Boschan turning-point detection) — the demo
  uses hand-built synthetic labels as a stand-in.
- **The HMM regime-switching layer** (section 3A.3, Method 3) — Mahalanobis
  analog matching is the Phase 1 starting point per the design doc's own
  recommendation; the HMM is a later layer once enough labeled historical
  panel data exists.
- **The actual Phase 1 backtest** (section 7) — this package is the
  engine the backtest will run against, not the backtest itself.
