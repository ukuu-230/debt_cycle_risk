# Debt Cycle Risk Tool — Project Handoff / Catch-Up Brief

*Feed this file to Claude at the start of a new session to restore full context on this project. It captures the goal, architecture, every major decision and why, all the bugs found, and the current state as of 2026-10-01.*

> **Read §13 first.** A statistics audit on 2026-10-01 changed how the composite and backtest are computed. The live numbers in §5 and §10 are PRE-AUDIT and must be re-run.

---

## 1. What this project is

A Python tool that scores **where the US (and other major economies) sit in Ray Dalio's long-term debt cycle** and produces an **unbiased, data-driven risk assessment** from real, trackable metrics. Not a date-predictor — a risk gauge. Grounded in Dalio's *Principles for Navigating Big Debt Crises*.

- **User:** builds/runs this locally in PyCharm on Windows (PowerShell), path `E:\esext\Python programs`. Uses a free FRED API key (set via `$env:FRED_API_KEY="..."` each PowerShell session — it does NOT persist).
- **Data source:** FRED (Federal Reserve Economic Data) only, via a free API key. Decision was made NOT to add other central-bank APIs (ECB/BoE/BoJ) because they mostly re-serve the same BIS/OECD data FRED already mirrors.
- **Design doc** lives at `debt-cycle-framework-design-doc.md` in the project.

## 2. How the user works (important for collaboration)

- Wants **maximum accuracy and rigor**, and explicitly asked to use **quantitative-finance principles** and the **most defensible statistical methods**.
- **Values honesty over agreement.** Explicitly said: "if it's not the right call don't say it is. make the right call." Push back with real reasoning when something is wrong or would overfit. Don't rubber-stamp.
- Prefers being shown the honest tradeoff and the recommended option, then deciding.
- Recurring environment friction: files not landing in the right folder, stale `__pycache__`, PyCharm opened at the wrong root. When a "new code isn't running" symptom appears, check: (a) is the file actually updated on disk (`Select-String`), (b) clear `__pycache__`, (c) is PyCharm rooted at the parent of `debt_cycle_scoring/`.

## 3. Architecture (package layout)

```
debt_cycle_scoring/
  ingestion/     FredClient (skip-missing resilience), catalog.py (US series),
                 intl_catalog.py (per-country series), panel.py, derived.py
  labeling/      NBER + BIS credit-gap + Bry-Boschan turning points -> consensus
                 labels with confidence tiers (the historical "ground truth")
  scorer/        risk_flags.py         = THE PRODUCT (composite risk score)
                 short_horizon_flags.py = fast market-stress sub-score
                 recession_backtest.py  = validates risk-leads-recessions
                 credit_gap_monitor.py  = cross-country credit-gap comparison
                 engine.py + stages.py  = DEPRECATED stage classifier (see below)
                 validation.py          = walk-forward + calibration (for stages)
run_scorer.py       -> US composite risk read (+ --backtest)   [FLAGSHIP]
run_credit_gaps.py  -> cross-country credit-gap monitor        [multi-country]
run_country.py      -> per-country composite (DEPRECATED, redundant)
run_labeling.py     -> historical labels
run_live.py         -> raw panel fetch
```
**123 tests pass** across the project. Run any with `python -m debt_cycle_scoring.<subpkg>.<test_module>`.

## 4. THE key architectural story (read this — it explains everything)

The project went through a **major pivot** that a new session must understand:

### Phase A — stage classifier (Mahalanobis) — TRIED, then DEMOTED
Original design: normalize metrics -> PCA into categories -> Mahalanobis analog matching against historically-labeled stages (early/bubble/top/deleveraging/reflation) -> stage probabilities. Fully built and tested.
**Then the walk-forward backtest killed it as the primary output:** on real data the stage classifier does **NOT beat a base-rate baseline** (Brier ~0.43 vs ~0.42), accuracy carried almost entirely by the dominant "early" class, and it produced ZERO bubble/top calls in all of history (bubble had only 3 training examples; "top" was deliberately untrainable). With only ~2-3 real US debt crises, a learned classifier overfits. **Lesson the user endorsed: on a tiny dataset, sophisticated ≠ accurate; the restrained method wins.**
Stages still exist but are a **caveated footnote**, not the product.

### Phase B — composite risk score (THE PRODUCT) — `scorer/risk_flags.py`
Rebuilt the risk read as a **composite of independent, individually-validated risk flags**, following the Fed/OFR financial-stress-index method (normalize each flag 0-100, higher=risk, combine). NO dependence on the stage classifier.

**7 flags:** credit-to-GDP gap (BIS), debt service ratio, yield-curve inversion, credit spreads (HY OAS), M2 velocity, long-bond level, debasement/deflation (10y breakeven inflation, two-tailed).

**Weighting = fixed, regime-conditional, from EXTERNAL published research** (BIS credit-gap work, Fed stress indices, Adrian-Estrella-Shin on the curve). Regimes (calm/building/elevated) are set by the credit-gap level. **Never** weights fit to the user's own crises (that would overfit + bake in hindsight). Equal-weight composite always computed alongside; divergence is surfaced.

**Rejected, with reasons:**
- PCA to combine the flags → PC1 captures max-variance not max-risk, can be hijacked by one noisy flag.
- "Weight by whichever flag reads highest" → guarantees always-high risk; it's a maximizer, not a gauge.
- Track-record weighting → overfits to 2-3 crises.
- Bootstrap CIs → with ~8 non-independent recessions even a block bootstrap is meaningless; naive IID would be misleading. Skipped deliberately.
- GARCH → for high-frequency return series with vol clustering, not ~100-260 quarterly macro obs. Rejected.

**KEY VALIDATION (composite):** on real FRED data it independently flagged **2006Q1-2007Q1 as the highest-risk quarters in the entire record** — caught the GFC run-up using only backward-looking research-weighted flags never fit to the data.

## 5. Recession backtest RESULT (PRE-AUDIT -- superseded, see §13) — `scorer/recession_backtest.py`
Tests: does high composite risk PRECEDE NBER recession onsets? (AUC at 1Q/4Q/8Q lead; uses raw USREC onsets, not the tool's own labels, so no circularity; excludes already-in-recession quarters.)
**Result on live data:**
- AUC **0.65 (1Q) → 0.72 (4Q) → 0.78 (8Q)** — signal STRENGTHENS at longer horizons, exactly as a credit-cycle framework predicts.
- Lead profile shows the RIGHT discrimination: **2008 lit up (60-66 for two years prior), 1990 and 1970s-80s elevated; but 2020 COVID and 2001 dot-com correctly stayed quiet** (they weren't debt-cycle recessions — correct blindness, not failure).
- Honest caveat: 8 non-independent recessions → descriptive, not conclusive. But direction and pattern are coherent and correct.

## 6. Short-horizon sub-score — `scorer/short_horizon_flags.py`
A SEPARATE fast/acute market-stress gauge shown ALONGSIDE (never blended into) the slow structural risk — keeps the structural signal that flagged 2006-07 uncontaminated by fast market noise. Five raw components: VIX level, VIX spike, dollar surge (YoY), credit-spread widening, curve flattening. Robust median/MAD normalization + logistic squash (fat-tail-resistant, no look-ahead). **KEY VALIDATION:** the self-built sub-score lands within ~10 points of the Fed's NFCI ("in agreement") — independent corroboration. NFCI/STLFSI4 used ONLY as external cross-checks (they already contain VIX/spreads — blending would double-count).

## 7. Multi-country — pivoted to a CREDIT-GAP MONITOR — `scorer/credit_gap_monitor.py`
User asked whether multi-country adds value or is "extra slop." Honest quant answer: **the per-country composite is mostly slop** (non-US composites run on 2-3 flags, of which long-bond + spreads are globally correlated with the US, so redundant; and non-US scores can't be recession-backtested). **The ONE non-redundant signal is the BIS credit-to-GDP gap** — credit cycles are national, so the cross-country spread of gaps is real information the US-only model lacks.
So multi-country was refocused into `run_credit_gaps.py`: fetches BIS credit-to-GDP ratios for US/Euro/UK/Japan (all four IDs verified live: QUSPAM770A, QXMPAM770A, QGBPAM770A, QJPPAM770A), computes each gap identically, ranks vs Basel thresholds, with **confidence tiers** (high ≥120 usable quarters / moderate ≥60 / low) because euro has only ~69 quarters and its gap magnitude is soft.
**Live result:** Japan +6.8 (watch, the only economy above trend); US -11.5, UK -17.8, Euro -18.4 (all below-trend/deleveraged). Surfaced a genuinely non-obvious signal — Japan, not the US, has the most stretched credit gap.
`run_country.py` / `build_country_risk_series` still exist but are the DEPRECATED path.

## 8. Bugs found & fixed (all surfaced by LIVE data, not unit tests — a recurring theme)
- **Wilshire 5000 discontinued** on FRED → swapped to SP500 (short_history flag; rolling 10yr license window).
- **Gold PERMANENTLY gone:** FRED deleted ALL LBMA gold series 2022-01-31. Replaced the "debasement" flag with two-tailed 10y breakeven inflation (T10YIE) — cleaner anyway (isolates fiat purchasing-power erosion; won't vanish). All gold code removed.
- **fetch_many now SKIPS missing series with a warning** instead of crashing the whole run (durable fix after Wilshire + gold both broke runs).
- **Consensus priority bug:** vote-count check ran before the contested-pattern check → 2020 COVID misclassified as a debt-cycle top. Fixed (contested checked first).
- **merge_derived_metrics dropped the short_horizon frames** when rebuilding the panel → short-horizon sub-score silently vanished at runtime. Fixed + regression test.
- **min_flags floor:** a 1-flag "composite" was pinning at 100 in deep history (e.g. euro "max risk 1980-81" was just the lone long-bond flag during the Volcker spike, decades before the euro credit gap starts in 2008). Now <2 flags → NaN, count still reported.
- **QXMDSR (euro DSR) 404s** — FRED carries NO non-US debt-service ratio (BIS has it only on its own portal). Removed; euro DSR flag is honestly absent.
- Various pandas FutureWarnings and a to_period-on-PeriodIndex crash in the recession backtest wiring — fixed.

## 9. Cross-cutting principles honored throughout (keep honoring these)
- **No look-ahead anywhere:** all normalization uses expanding/backward-looking windows; the one-sided HP filter (λ=400,000, 40q burn-in) for the credit gap; walk-forward refits use only pre-T data. There's a hard test proving future labels can't change past predictions.
- **Honest small-sample framing:** no p-value theater; descriptive + rank stats with explicit caveats.
- **Transparency:** every risk number decomposes into its per-flag contributions.
- **Graceful degradation:** missing series/flags drop and the composite renormalizes, always reporting the count.
- **Verify FRED series IDs live before adding them** (learned hard from Wilshire/gold/QXMDSR — never pattern-match an ID and assume).

## 10. Live reads as of 2026-09-28 (PRE-AUDIT -- re-run, see §13)
- **US composite risk ≈ 40/100, "building"** — but credit gap itself ~0 (no credit buildup); the "building" regime + mid-range score come from other flags (M2 velocity low, curve, long-bond). Well-evidenced quarter ~2025Q4-2026Q1; more-recent quarters are provisional (quarterly BIS/DSR series lag the daily flags).
- **US short-horizon stress ≈ 43/100**, in agreement with Fed NFCI.
- **Cross-country credit gaps:** Japan +6.8 (watch); US -11.5, UK -17.8, Euro -18.4 (below-trend). US is deleveraged relative to its own trend; Japan most stretched.

## 11. What's open / candidate next steps
- **Visual dashboard** (offered, not yet built) — US composite + 4-country credit-gap comparison as charts instead of terminal tables. The most natural "make it usable day-to-day" step.
- UK/Japan could get full composites, but per §7 that's low-value (redundant flags); the credit-gap monitor already covers them well.
- Optional: ALFRED real-time-vintage data for a fully vintage-correct backtest (currently uses latest-vintage — a known, documented approximation).

## 12. How to run it (user's environment)
```powershell
$env:FRED_API_KEY="<key>"                              # each session
python run_scorer.py --start 1962-01-01 --backtest     # US flagship + recession AUC
python run_credit_gaps.py                              # cross-country credit gaps
```
If new code seems not to run: confirm the file is updated on disk, delete all `__pycache__`, confirm PyCharm project root is the folder CONTAINING `debt_cycle_scoring/`.


## 13. Statistics audit (2026-10-01) -- what changed and why

The user asked for a full review of signal coverage and statistical correctness. Fixes (all with regression tests; 146 tests pass):

**Composite (`scorer/risk_flags.py` `build_risk_series`)**
- **Publication lags.** Quarterly inputs were aligned by REFERENCE quarter, so the backtest used data before it was published (look-ahead that inflated lead time). Now re-timed: credit gap +2q, DSR +2q, M2 velocity +1q (`PUBLICATION_LAGS`). Side benefit: the current quarter is scored on the latest PUBLISHED values, so it normally carries all 7 flags (no more "provisional").
- **Stationarity.** Expanding percentiles of trending series measure the trend. M2 velocity now enters as its YoY % change (the level fell structurally after 1997 and read "alarm" continuously since 2008). Long-bond yield is ranked within a trailing 40q window (it rose 1962-81, so it sat near 100 through the 1970s). DSR stays a full-history percentile on purpose (BIS Drehmann-Juselius construction).
- **Curve = 10y-3m** (GS10 - TB3MS monthly, quarterly mean, back to 1953; per-quarter fallback to daily T10Y3M, never to 10y-2y). Matches the Estrella-Mishkin / NY Fed research the weight cites, and adds the curve to the 1969/1973 recessions (10y-2y only starts 1976).
- **Mid-rank percentiles** everywhere (ties get half credit). The old `<=` convention read ties as the top of the distribution.

**Backtest (`scorer/recession_backtest.py`)**
- **USREC collapsed with MAX** (any recession month), not `last` (catalog). `last` dropped quarters like 2020Q2; NBER troughs now match.
- **Right-censoring.** Quarters whose outcome window isn't observed are excluded instead of counted as "no recession followed". The last 4 quarters of USREC count as unobserved (`NBER_DATING_LAG`: NBER dates peaks 6-21 months late; FRED zero-fills). A window that already contains an onset is a known positive.
- No false onset when the sample starts mid-recession.
- **New robustness output** (`auc_at_horizon`, `flag_attribution`): AUC on all quarters / >=4 flags / 1985+; per-flag standalone AUC and drop-one delta. DESCRIPTIVE ONLY -- never re-weight from it.

**Other:** short-horizon robust z excludes the current point; HP filter raises on interior NaN; per-country builder kept in step; the flags frame keeps zero-flag quarters (was silently dropping rows).

**Coverage review.** Dalio checklist mapped against measured/tested metrics. Added 5 audit candidates: corporate equities/GDP (Z.1, Buffett, from 1945), building permits, policy-rate headroom, foreign share of federal debt, current-account deficit. Audit candidates are now publication-lagged; trending levels are ranked within a trailing 10y window. Not on FRED (noted, not tested): Gilchrist-Zakrajsek excess bond premium, margin debt, COFER reserve share.

**An independent reviewer agent** verified all fixes and found 3 further defects (censoring dropped known positives; no per-quarter curve fallback; country path unpatched) -- all fixed.

**Must re-run:** `run_scorer.py --backtest` (new AUCs + robustness tables) and `run_metric_audit.py`. The pre-audit AUC 0.78 may move, either way -- report whatever it is honestly. Earlier audit verdicts (no candidate improves the composite; fiscal = CONTEXT) were pre-lag and need confirming.

## 14. Dashboard + automation (2026-09-29 to 10-01)
- `export_dashboard.py` + `dashboard/template.html` -> `site/index.html`: every number computed, nothing hand-entered. Panels: composite + history (NBER shading), flag decomposition, backtest + lead profile, robustness, daily market pulse, fiscal & valuation context (NOT scored), cross-country gaps, method/limitations.
- `.github/workflows/update-dashboard.yml`: runs tests then rebuilds and deploys to GitHub Pages twice each business day (13:17 / 22:17 UTC). Key only in the `FRED_API_KEY` repo secret. Failed run = last good page stays up. Open tabs self-refresh.
- User chose daily over intraday: FRED posts once per day; intraday needs a paid feed + server, and adds noise for a debt-cycle gauge.
- The claude.ai artifacts published earlier are static snapshots (the grad-level one used ILLUSTRATIVE lead-profile numbers and a wrong debasement weight) -- superseded by the GitHub page.


## 15. Post-audit live run (2026-10-02) and follow-up fixes

**Result (user's live run, post-audit):** recession-lead AUC 0.835 / 0.844 / 0.831 (1Q/4Q/8Q). Robust: >=4 flags 0.85/0.86/0.83; 1985+ 0.75/0.81/0.82. The pre-audit 0.78 was NOT inflated -- the corrected composite scores higher.

**Attribution (8Q) -- the honest headline:** the yield curve alone scores 0.845 vs the composite's 0.831; removing it drops the composite to 0.668. So the composite's RECESSION-TIMING skill is essentially the 10y-3m curve. The credit gap LOWERS recession AUC (delta -0.053; standalone 0.66) -- expected: the gap predicts credit/banking crises, and only 2008 among the 8 recessions was one. M2 velocity standalone 0.35 (below 0.5; delta -0.021, noise). Weights deliberately NOT changed (would fit them to ~8 recessions). The dashboard now states this in words (robustness panel, computed).

**Bugs the live run exposed (fixed, with tests; 149 pass):**
- Credit-spread flag had 0 quarters: FRED appears to serve only ~3 years of ICE BofA HY OAS (BAMLH0A0HYM2) -- under the 20q percentile warm-up. Replaced as primary by Moody's Baa - 10y (BAA - GS10 monthly from 1919; daily BAA10Y fills the current quarter). HY OAS is now fallback only. Short-horizon spread-widening and the market pulse use the Baa spread too. UNVERIFIED that BAA/BAA10Y fetch live -- check the new "Input coverage" lines.
- Regime read "building" in 2026Q3 only because the lagged credit gap was missing that quarter (default). Fix: latest PUBLISHED value carries forward up to 2 quarters (`CARRY_FORWARD_QUARTERS`) -- real-time-available, no look-ahead.
- DSR showed only 66 quarters in attribution (TDSP should start 1980). Unexplained -- check "Input coverage" on next run.
- run_scorer now prints input coverage per flag (flags <80 quarters marked SHORT).

## 16. Live run with Baa spread active (2026-10-02, clean reinstall)

AUC 0.797 / 0.806 / 0.768 (1Q/4Q/8Q); >=4 flags 0.81/0.82/0.78; 1985+ 0.75/0.80/0.77. Down from 0.83 ONLY because the credit-spread flag now actually participates (it had 0 quarters before).
Attribution (8Q): curve standalone 0.845, delta +0.174 (carries the result); Baa spread standalone 0.403, delta -0.063; M2 velocity standalone 0.350, delta -0.040; long bond +0.027; credit gap +0.015.
Spread and velocity score BELOW 0.5: wide spreads / falling velocity are coincident-to-lagging (they peak in and just after recessions; spreads are TIGHT before them -- Lopez-Salido, Stein & Zakrajsek 2017). DECISION: no sign flips or re-weighting. A flip chosen after seeing this sample is a forking-paths fit; 0.77 is the honest validated number. If a "compression = risk" spread flag is ever wanted, it needs a pre-registered rationale and a separate test, and its AUC gain cannot count as validation.
Still to check: the top of the output (headline, regime should now be "calm", Input coverage list -- DSR showed only 66 quarters).
- Headline rule fixed: headline = latest quarter with the FULL recent flag set (max n_flags over last 8q), via `headline_quarter()`. Live run had promoted a 2-day-old 2026Q4 (5 flags, risk 24) over 2026Q3 (7 flags, risk 29.5, calm).
- "Highest-risk quarters" now ranked only over quarters with >=5 flags (1967-68 had topped the list on 3-4 flags).
- Input coverage: TDSP (DSR) flag only 2010Q3+ => FRED's TDSP now appears to start ~2005 (20q warm-up + 2q lag). DSR is absent for 7 of 8 recessions; its percentile compares only vs 2005+.
- After the Baa spread went live, 2006-07 no longer tops the record: tight spreads in 2006-07 pull the composite down. Same root cause as the spread's 0.40 standalone AUC. OPEN DECISION for the user (see §16): leave as is, move spread to the short-horizon gauge only, or "compression = risk".

## 17. Specification v2 (2026-10-05, user chose option 2)
- Credit-spread LEVEL removed from the structural composite (now 6 flags: credit gap, DSR, curve, M2 velocity, long bond, breakevens). It is still computed and shown as context (`RiskSeriesResult.context_flags`), and the short-horizon gauge now has a `spread_level` component next to `spread_widening`.
- v2 weights = v1 weights renormalized without the spread (`_renormalize_without`); nothing re-chosen. `SPEC_V1_FLAG_NAMES` / `SPEC_V1_WEIGHTS` kept; `composite_spec_v1()` rebuilds the v1 composite so run_scorer and the dashboard always print v1 vs v2 AUCs side by side.
- Framing (must stay honest): v1 (live 8Q AUC 0.768) is the clean validated number; any v2 gain was found after seeing the data. Expected v2 8Q AUC ~0.83 (= v1 drop-one result) -- confirm on the next live run.
- Dashboard callout logic fixed: only says one flag "carries" the result when it both scores >= composite alone and has the largest positive delta; says "no meaningful signal" when composite AUC < 0.6.
- CONFIRMED live (2026-10-05): v2 AUC 0.835 / 0.844 / 0.831; >=4 flags 0.85/0.86/0.83; 1985+ 0.75/0.81/0.82. v1 0.80/0.81/0.77 (shown alongside). Curve still carries (delta +0.171); credit gap delta -0.053 (dilutes recession timing; kept on purpose -- it measures debt-cycle position; no second post-hoc change).

## 18. Metric audit on spec v2 (live, 2026-10-13)
No ADD. Every candidate's incremental 8Q dAUC <= +0.017 (noise); the 6-flag composite stays as is.
Best standalone 8Q AUCs: core PCE 0.74 (REDUNDANT, rho 0.77 w/ long bond); building permits 0.67 (dAUC +0.015); current-account deficit 0.66 (only 90q); household debt Z.1 0.66 (REDUNDANT, rho 0.79 w/ credit gap -- confirms the gap captures household leverage); SLOOS 0.76 at 4Q / 0.62 at 8Q.
Fiscal again scores BELOW 0.5 (debt/GDP 0.27, deficit 0.36: they rise after recessions) -> stays CONTEXT, never scored.
Current readings (trending levels vs trailing 10y): federal debt/GDP 90th pct, deficit 87th, interest/receipts 80th, term premium 100th, corporate equities/GDP 100th, household net worth/income 98th.
Offered (not built): add corporate equities/GDP to the unscored fiscal & valuation panel -- it was missing only because SP500 history on FRED is short, not by choice.

## 19. Dalio context panel expanded (2026-10-13)
`scorer/fiscal_panel.py` now builds 18 UNSCORED rows in four Dalio groups (Sovereign debt & financing / Bubble gauges / Private leverage & credit / Monetary policy capacity MP1->MP3). New: r-g, foreign share of federal debt, current-account balance, real dollar yoy, corporate equities/GDP (Z.1), household and corporate debt/GDP 3y change, SLOOS, fed funds (low = risk), real policy rate, core PCE, Fed balance sheet/GDP 1y change. Building permits deliberately excluded (tests well but not a Dalio indicator).
- Each row has risk_dir (+1 / -1 / 0); bars are oriented so longer = riskier; two-sided rows are blue and labeled "unusual" at the tails.
- `_per_gdp()` divides dollar LEVELS by GDP (median magnitude >= 1000) and leaves ratios alone; such rows show a percentile only.
- Every row that shows a raw value has a plausibility range; outside it the value is hidden (guards against FRED unit surprises).
- BUG FOUND: NCBDBIQ027S is a Z.1 dollar LEVEL, not debt/GDP (catalog label was wrong). The audit's "Corporate debt/GDP, 3yr change" row measured nominal debt growth; now fixed via `_per_gdp`. Its old WEAK verdict should be re-checked on the next audit run.
- Workflow: actions/checkout@v5, actions/setup-python@v6, runs-on ubuntu-24.04 (pinned before GitHub's Oct 19 Ubuntu 26 switch).
- User's local env: project now at D:\debt cycle risk project with its own .venv (base Python C:\Users\esext\AppData\Local\Python\bin\python.exe; the WindowsApps python.exe is the Store stub). Repo: github.com/ukuu-230/debt_cycle_risk, deployed via GitHub Pages.

## 20. Plain-English layer (2026-10-13)
- `dashboard/explain.js`: ~60 explanation entries (plain English / how to read / why it matters / technical), keyed to `data-x` hooks on every tile, panel heading, flag, pulse series, context metric and country row. Edit wording here without touching logic.
- `dashboard/guide.js`: click-to-open ⓘ cards (bottom sheet on phones, Esc/outside-click closes, follows its button on scroll); a "Plain-English read" panel at the top rewritten from DATA on each update (band, 1-yr trend, scenario, why the score is where it is from flag contributions, extremes outside the score, what would change the picture); a "Scenario guide" panel with six Dalio scenarios: acute stress, deleveraging/policy response, late cycle (curve), credit bubble building, calm private / stretched public & valuations, healthy expansion. First match in that priority order = "where we are now".
- Portfolio text = documented historical tendencies per scenario + Dalio's All-Weather default, labeled "general historical information, not personal financial advice"; disclaimer uses the live 8Q AUC and states the tool was never tested on returns. Deliberately no buy/sell or allocation numbers.
- export_dashboard.py inlines both JS files (page stays single-file). Tested: each scenario forced via synthetic DATA selects correctly; 61 buttons all resolve; no JS errors; phone 390px no horizontal scroll.

## 21. Dalio's "The Numbers Behind the Bond Bear Market" folded in (2026-10-13)
Source: raydalio.substack.com/p/the-numbers-behind-the-bond-bear (same piece as the LinkedIn post, which blocks automated reading). Published Oct 13 2026.
- Explanations updated with his concepts, his numbers only as dated + attributed context (field `s` = source line in the card): debt/GDP is a weak gauge, interest/revenue is his preferred one (~21% of revenue; avg rate on existing debt ~3.5% vs 4-5.5% market; +1pp avg rate ~ +$310B/yr); "death spiral"; bills ~50% of issuance vs ~20% typical; foreigners ~32% / Fed ~14% of holdings, demand for long Treasuries "clearly weakening"; Fed as swing buyer; AI-driven corporate bond and equity issuance ("canary in the coal mine"); global bond-supply competition.
- New 7th scenario `bond_squeeze` ("Bond bear market: debt supply outrunning demand"), his mid-cycle stage: triggers when long-bond flag >= 75 AND (term premium or interest/receipts risk percentile >= 75) AND short-horizon stress < 65. Priority: after late_cycle, before credit_bubble. Portfolio text = historical bond-bear-market evidence (1965-81, 2022) + his attributed view ("probably has a long way to go"; "easier to be short than long US Treasuries", which he labels not investment advice).
- Context panel: interest/receipts now first in the sovereign group (his preferred gauge).
- Not built (needs data FRED doesn't carry): bills share of issuance, total debt service incl. principal rollovers (~200% of revenue per Dalio), Treasury holder breakdown by country.

## 22. Free US Treasury sources added (2026-10-13)
`ingestion/treasury_client.py` (no key; injectable transport, sha1 disk cache, retries, raises only TreasuryDataError; `fetch_treasury_extras` skips any failing source with a warning). Layouts verified live 2026-10-13:
- Fiscal Data API `v2/accounting/od/avg_interest_rates` (fields record_date, security_type_desc, security_desc, avg_interest_rate_amt; row Marketable / "Total Marketable"; Sep 2026 = 3.518%) -> TSY_AVG_RATE_MKT.
- Fiscal Data API `v1/debt/mspd/mspd_table_1` (security_type_desc, security_class_desc, total_mil_amt; "Bills" and "Total Marketable") -> TSY_BILLS_SHARE (Sep 2026 = 22.4%).
- TIC Table 5 `ticdata.treasury.gov/.../slt_table5.txt` (tab-separated, latest 13 months only; old mfh.txt froze in Mar 2023) -> TIC_JP_CN (Japan + China, Mainland), TIC_OFFICIAL ("Of Which: Foreign Official").
- FRED WSHOBL (Fed T-bill holdings, weekly, from 2002).
New context rows: refi_gap (DGS10 - avg rate on existing debt), bills_share (vs TBAC 15-20%), tic_jp_cn and tic_official (mode "chg12m": 12-month % change, no percentile because only 13 months are published), fed_bills (4-quarter $bn change, two-sided). `check_sources.py` prints live status of each Treasury source.
Still not available free/clean: bills share of NEW issuance (Dalio's ~50%; would need auction data), total debt service incl. principal rollovers (needs MSPD maturity detail), corporate bond issuance / hyperscaler concessions (SIFMA spreadsheets, not an API).
- 2026-10-13: context-panel explanation now covers the 12-month-change rows; bond_squeeze 'where we are' text cites refi_gap (if > 0) and the Japan+China 12-month change (if falling) when available.
