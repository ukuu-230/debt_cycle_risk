# debt_cycle_scoring.ingestion

FRED data ingestion — design doc section 4. Feeds `QuarterlyPanel` output
directly into the scoring engine (`normalize.py` / `aggregate.py`) built
in the previous step.

## Files

| File | What it does |
|---|---|
| `fred_client.py` | `FredClient` — fetches a FRED series via the official JSON API (with a key) or the public `fredgraph.csv` download (no key), with retries, disk caching, and an **injectable transport** so it's testable without real network calls |
| `catalog.py` | `FRED_CATALOG` — maps every metric from design doc section 2 to a real FRED series ID, its category, native frequency, resample method, and direction (`higher_is_later_cycle`) |
| `panel.py` | `build_quarterly_panel()` — fetches the whole catalog and assembles one aligned quarterly `QuarterlyPanel` (scored metrics / reference series / label sources, kept separate) |
| `derived.py` | `merge_derived_metrics()` — computes the ratio metrics design doc section 2 actually calls for (Wilshire/GDP, real policy rate, net worth/DPI, Fed balance sheet/GDP, home price/DPI) and appends them to `scored_metrics`, since these aren't raw FRED series themselves |
| `test_fred_client.py` | 10 tests against a fake transport returning FRED-shaped JSON/CSV — parsing, missing-value handling, caching, retries, 404 handling, resampling, category routing |
| `test_derived.py` | 6 tests for the ratio layer — zero-denominator handling, index alignment, graceful skipping when an input is missing, and an exact manual-calculation check for the real rate |
| `demo.py` | Integration demo: synthetic FRED-shaped data → `build_quarterly_panel()` → `merge_derived_metrics()` → straight into `normalize.py`/`aggregate.py` from the scoring engine, proving raw and derived metrics combine correctly within a category |

## Running it

```bash
python3 -m debt_cycle_scoring.ingestion.test_fred_client
python3 -m debt_cycle_scoring.ingestion.demo
```

## Using it with real data

```python
from debt_cycle_scoring.ingestion import FredClient, build_quarterly_panel

client = FredClient(api_key="YOUR_FRED_API_KEY")   # get one free at
                                                     # https://fredaccount.stlouisfed.org/apikeys
panel = build_quarterly_panel(client, start_date="1962-01-01")

panel.scored_metrics     # DataFrame ready for normalize.py
panel.reference_series   # GDP, CPI, Core PCE -- build ratios (Wilshire/GDP, real rate) from these
panel.label_sources      # NBER USREC -- feeds the section 3B labeling pass
```

Without an `api_key`, `FredClient` automatically falls back to the no-key
`fredgraph.csv` download. That endpoint isn't a versioned API contract the
way the JSON one is (see the caveat in `fred_client.py`'s docstring) — it
was not verified against a live response while building this (no outbound
network access in the build sandbox), so **test it against a couple of
real series before relying on it**, or just register a free API key and
skip the fallback path entirely.

## A bug this test suite actually caught

`catalog.py` originally listed CPI and Core PCE under `category="cb_stance"`
(scored) with `higher_is_later_cycle=None` (meant as reference-only inputs
for computing a real interest rate elsewhere) — a self-contradictory entry
that would have silently broken `direction_encode()` downstream. `panel.py`
now raises a clear `ValueError` if any non-reference, non-label metric has
an undefined direction, and the two CPI/PCE entries were moved to
`category="reference"` where they belong. Left in here as documentation
that the test suite (specifically running the full pipeline end-to-end
in `demo.py`, not just the unit tests) is what surfaced this — the
individual unit tests for `fred_client.py` all passed even with the bug
present, since the bug was in `catalog.py`'s data, not in any function's logic.

## What's deliberately NOT here yet

- Non-US data sources (BIS, ECB SDW, BOJ, ONS) — section 4 of the design
  doc scopes these for Phase 3, US-only for Phase 1.
- The BIS credit-to-GDP gap and Bry-Boschan turning-point detection from
  section 3B — those are a separate ingestion+computation path, not part
  of this FRED client.
- Rate-limit-aware batching against FRED's real 120-requests/minute limit
  beyond the basic `inter_request_delay_seconds` parameter — fine for
  Phase 1's ~20-series catalog, worth revisiting if the catalog grows a lot.

## One honest, unresolved caveat in `derived.py`

- **`Fed balance sheet / GDP`** is marked `higher_is_later_cycle=True` as a
  rough default, but the real relationship is regime-dependent (QE mostly
  reflects being in/just past a crisis response, not a uniform "top"
  signal at every level) — worth watching how this metric actually behaves
  once the Phase 1 backtest (design doc section 7) runs against real data.

`Home price / income` previously used disposable personal income as a
stand-in for median household income (not in the original catalog) — now
fixed: `MEHOINUSA646N` (nominal median household income, Census Bureau,
annual back to 1984) was added and wired in, deliberately the NOMINAL
series rather than the more commonly-cited real/inflation-adjusted one,
since CSUSHPISA (Case-Shiller) is itself nominal — pairing it with a
real income series would have introduced a spurious inflation-adjustment
drift into the ratio.
