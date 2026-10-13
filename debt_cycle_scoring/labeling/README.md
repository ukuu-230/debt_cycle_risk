# debt_cycle_scoring.labeling

Implements the section 3B independent labeling pass, replacing the
hand-built synthetic labels used in `debt_cycle_scoring/demo.py`.

## Files

| File | Design doc section | What it does |
|---|---|---|
| `nber.py` | 3B.1, Source A | Extracts peak/trough turning points from the NBER `USREC` series (already ingestible via `ingestion/catalog.py`) |
| `hp_filter.py` | 3B.2 | One-sided (backward-looking-only, no-look-ahead) Hodrick-Prescott filter, matching BIS's own published methodology |
| `credit_gap.py` | 3B.2, Source B | Applies the filter to a credit-to-GDP ratio and flags quarters using Basel III's actual published thresholds (2pp / 10pp) |
| `turning_points.py` | 3B.3, Source C | Simplified Bry-Boschan/Harding-Pagan mechanical turning-point detector — the most independent source, since it's a fixed rule with no institutional judgment involved |
| `consensus.py` | 3B.4 | Combines all three into confidence tiers (`high_confidence` / `single_source` / `contested` / `silent`), then walks high-confidence events into full STAGES-vocabulary labels |
| `test_labeling.py` | — | 14 tests, including two regression tests for real bugs the integration demo caught (see below) |
| `demo.py` | — | Runs the real labeling pipeline (not a stand-in) on a synthetic-but-realistic bubble→top→deleveraging episode, then feeds the resulting labels into `stages.fit_archetypes` from the scoring engine |

## Data source used

The BIS credit-to-GDP gap needs a credit-to-GDP ratio as input. Rather
than building a separate BIS API client, this reuses the FRED client
already built: BIS publishes this exact series and FRED mirrors it —
confirmed live series ID **`QUSPAM770A`** ("Total Credit to Private
Non-Financial Sector, Adjusted for Breaks, for United States", as % of
GDP, back to the 1940s). Added to `ingestion/catalog.py` under
`category="label_source"`.

## Running it

```bash
python3 -m debt_cycle_scoring.labeling.test_labeling
python3 -m debt_cycle_scoring.labeling.demo
```

## Two real bugs this caught (not just "ran successfully")

1. **Bry-Boschan pruning could break alternation.** The first version
   enforced alternation, then pruned turning points too close together
   in a separate pass — but dropping an interior point can leave two
   same-type points adjacent on either side of the gap it left (e.g.
   pruning a peak between two troughs leaves trough-trough back to
   back). The integration demo's output visibly showed non-alternating
   sequences. Fixed by iterating merge-then-prune to a fixed point
   instead of a single pass; `test_bry_boschan_pruning_never_breaks_alternation`
   is a regression test for exactly this.
2. **`fit_archetypes` crashed with a single category.** `np.cov` on a
   single-column array returns a 0-d scalar instead of a 1×1 matrix,
   which crashed `np.diag()`. This didn't show up in `test_scoring.py`'s
   original tests (all multi-category), only when the integration demo
   here fit archetypes on a single "debt_burden" category. Fixed with
   `np.atleast_2d`; `test_fit_archetypes_single_category_does_not_crash`
   (added to `test_scoring.py`) is the regression test.

## An honest (non-)finding from the integration demo, not a bug

In the synthetic episode, the consensus combiner correctly identifies
the **top** with high confidence — NBER, the credit gap, and the
mechanical detector all converge on the same ~2-year window, which is
the single most important validation per the design doc's section 7.3
(the flagship case should score cleanly). But it never finds a
high-confidence **trough**: NBER's trough and the mechanical detector's
nearest trough land too far apart (more than the 4-quarter agreement
window) to agree. That's a real, structural limitation worth carrying
into the Phase 1 backtest, not something to tune away here by
adjusting parameters until the demo looks nicer — it's exactly the kind
of boundary case section 7.3 says to expect and investigate individually
rather than paper over.

## What's deliberately NOT here yet

- Wiring this against REAL FRED/BIS data end-to-end — this sandbox has
  no outbound network, so every test and demo here uses the real
  functions against synthetic or hand-built inputs, not a live pull.
  Point `ingestion.FredClient` at a real API key and real dates, run
  `build_quarterly_panel()`, then feed `panel.scored_metrics`,
  `panel.label_sources["NBER recession indicator"]`, and
  `panel.label_sources["US Credit-to-GDP ratio (BIS, via FRED)"]`
  straight into this subpackage's functions — the interfaces already
  line up.
- Extending Source A/B beyond the US (other countries' recession dating
  bodies and BIS credit-gap series) — in scope for Phase 3 per the
  design doc, not Phase 1.
- The Jordà-Schularick-Taylor Macrohistory Database as a pre-1961
  substitute for the BIS credit-gap series, mentioned as a real gap in
  earlier design-doc discussion but not implemented here.
- Tuning `turning_points.py`'s `window`/`min_phase_quarters` or
  `consensus.py`'s `agreement_window` against real data — the honest
  finding above suggests these matter a fair amount and shouldn't be
  finalized against synthetic data alone.
