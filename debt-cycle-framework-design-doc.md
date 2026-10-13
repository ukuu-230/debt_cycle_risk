# Debt Cycle Tracker — Framework & Metrics Design Doc

**Purpose:** An app that applies Ray Dalio's *Principles for Navigating Big Debt Crises* framework to live economic data, showing — as objectively as possible — where the US and a handful of major economies currently sit in the long-term debt cycle.

**Scope for v1:** US, Eurozone (Germany as proxy + ECB-level aggregates), China, Japan, UK.
**Data approach:** Live pulls from free public APIs, primarily FRED, supplemented by OECD, ECB SDW, BOJ, and IMF where FRED doesn't cover a country.

---

## 1. The Framework, Distilled Into a State Machine

Dalio's model treats debt cycles as roughly mechanical: two nested cycles.

- **Short-term debt cycle** (~5–8 years, the "business cycle"): expansion → tightening → recession → easing, driven by central bank policy responding to inflation/growth.
- **Long-term debt cycle** (~50–75 years): debt grows faster than income across *many* short-term cycles until debt burdens become unsustainable, forcing a deleveraging.

The app's core job is to place the economy on **both** cycles simultaneously, since "early in a short-term cycle" reads very differently depending on where you are in the long-term cycle.

### 1.1 Long-term cycle stages (the ones that matter most for the "big picture" read)

| Stage | Description | Dominant signal |
|---|---|---|
| **Early** | Debt growth ≈ income growth; debt funds productive investment | Debt/GDP flat or slowly rising, credit growth tracks nominal GDP growth |
| **Bubble** | Debt growth > income growth, financed increasingly by expectation of future asset price gains rather than income | Rising debt/GDP, rising asset prices/GDP, narrowing risk premia, rising leverage in the marginal buyer |
| **Top** | Central bank tightens to fight inflation/overheating; short rates rise; this is the "MP2→tightening" inflection | Inverted or flattening yield curve, real rates rising, credit growth decelerating from a high level |
| **Depression / initial deleveraging** | Debt service becomes unpayable at current income; credit contracts; asset prices fall; often deflationary at first | Falling credit growth (or contraction), falling asset prices/GDP, rising defaults/spreads, GDP contraction |
| **"Beautiful" vs "ugly" deleveraging** | Deleveraging happens via 4 levers used in some mix: austerity, debt default/restructuring, wealth transfer (taxation), and money/debt monetization (printing). "Beautiful" = balanced mix, deflationary debt burden falls while nominal growth stays positive. "Ugly" = too much of one lever (depression if austerity/default dominate; hyperinflation if printing dominates) | Central bank balance sheet growth vs. GDP, real GDP growth vs. nominal, currency depreciation rate, inflation trend |
| **Reflation / normalization** | Debt burdens have been worked down; credit growth resumes matching income growth; cycle restarts | Debt/GDP stabilizing or falling, credit growth re-accelerating from trough, real rates low/negative early on |

### 1.2 Short-term cycle stages (nested inside the above)

Standard business-cycle phases: **Early expansion → Mid-cycle → Late-cycle/overheating → Recession/contraction**, read primarily through the output gap, unemployment gap, inflation trend, and the yield curve/policy rate cycle. This gives the "where in the weather, not just the climate" read.

### 1.3 Monetary policy stages (Dalio's MP1/MP2/MP3, useful for classifying central bank response)

- **MP1** — Interest-rate-based policy still has room (rates well above zero); conventional tightening/easing works.
- **MP2** — Rates near zero; central bank shifts to QE/asset purchases to ease.
- **MP3** — QE alone stops working (liquidity isn't reaching the real economy / hitting diminishing returns); coordination with fiscal policy (monetized deficit spending, "helicopter money"-adjacent tools) becomes more likely.

Classifying which of MP1/2/3 a central bank is currently capable of/using is itself a useful cycle marker, especially for Japan and increasingly relevant for the Eurozone.

---

## 2. Metrics, Grouped by What They Detect

For each metric below: what it's measuring, and a primary US data source (FRED series ID where one exists cleanly). Non-US equivalents are noted in §4.

### 2.1 Debt burden & growth (long-term cycle position)
| Metric | What it tells you | Source |
|---|---|---|
| Total (public + private) debt/GDP | Overall leverage in the system | FRED: `credit to non-financial sector`, BIS credit-to-GDP series |
| Household debt/GDP, corporate debt/GDP, government debt/GDP (split) | Which sector is carrying the leverage — matters for who deleverages | FRED: `HDTGPDUSQ163N`, `NCBDBIQ027S`, `GFDEGDQ188S` |
| Household debt service ratio | Can households actually service current debt out of income | FRED: `TDSP` |
| Credit growth rate (YoY, real) | Is credit expanding faster/slower than income — the core "bubble vs early cycle" test | FRED: bank credit series, `TOTBKCR` |
| Credit growth minus nominal GDP growth | Direct implementation of Dalio's "debt growing faster than income" test | Derived |

### 2.2 Asset prices & leverage-financed speculation (bubble detection)
| Metric | What it tells you | Source |
|---|---|---|
| Equity market cap/GDP ("Buffett Indicator") | Classic bubble gauge | FRED: Wilshire 5000/GDP |
| Household net worth/GDP or /disposable income | Asset-price-driven wealth effects | FRED: `TNWBSHNO` / `DPI` |
| Home price/income ratio | Real-estate-specific bubble signal | FRED: Case-Shiller / median household income |
| Margin debt / speculative leverage proxies | How much of asset buying is debt-financed | FINRA margin statistics (manual/periodic, no clean free API) |
| Credit spreads (high-yield OAS) | Risk premium compression = late bubble; blowout = deleveraging | FRED: `BAMLH0A0HYM2` |

### 2.3 Central bank stance & the short-term cycle
| Metric | What it tells you | Source |
|---|---|---|
| Real policy rate (fed funds − inflation) | Is policy actually tight or loose | FRED: `FEDFUNDS` minus `CPIAUCSL` YoY |
| Yield curve (10y−2y, 10y−3m) | Classic recession-timing / cycle-top signal | FRED: `T10Y2Y`, `T10Y3M` |
| Central bank balance sheet/GDP | QE intensity, MP2/MP3 classification | FRED: `WALCL` / GDP |
| Output gap | Overheating vs. slack | FRED: CBO potential GDP vs. actual GDP |
| Unemployment gap (U3 − natural rate) | Same, labor-market version | FRED: `UNRATE` − `NROU` |
| Core inflation trend (PCE, CPI, trimmed-mean) | Where in the inflation cycle | FRED: `PCEPILFE`, `CPILFESL` |

### 2.4 Deleveraging-in-progress signals (once a downturn starts)
| Metric | What it tells you | Source |
|---|---|---|
| Change in credit growth (accelerating contraction vs. stabilizing) | Is deleveraging still active or has it bottomed | Derived from 2.1 |
| Real GDP growth vs. nominal GDP growth | Widening gap = disinflation/deflation dominating the adjustment (more "ugly" if persistently negative real growth) | FRED |
| Currency value trend (real effective exchange rate) | Large depreciation = printing lever is dominant (relevant especially for non-reserve-currency economies) | FRED: `RBUSBIS` and country equivalents |
| Fiscal deficit/GDP | Austerity vs. stimulus lever in use | FRED: `FYFSGDA188S` |
| Central bank balance sheet growth rate | Money-printing lever intensity | FRED: `WALCL` |

### 2.5 External/global-specific (for non-US economies)
| Metric | What it tells you | Source |
|---|---|---|
| Current account balance/GDP | External vulnerability, reliance on foreign capital | IMF/OECD/ECB SDW |
| External debt (esp. FX-denominated) | Currency-crisis risk on top of the debt cycle | World Bank/IMF |
| FX reserves adequacy | Capacity to defend the currency during deleveraging | IMF/central bank data |

---

## 3. Turning Metrics Into a Stage Read — "Unbiased" Design Principles

The hard part isn't collecting metrics, it's not smuggling in a narrative. Design choices to keep it defensible:

1. **Percentile-rank everything against its own history**, not against an arbitrary "good/bad" threshold. A metric is "high" only relative to that country's own historical distribution (e.g., last 40–50 years, or since data starts if shorter).
2. **Score stages, don't declare one.** Output a probability-weighted distribution across the 6 long-term stages and 4 short-term stages (e.g., "62% Bubble / 28% Top / 10% Early"), not a single confident label. This mirrors real uncertainty and avoids false precision.
3. **Show the metric, the historical percentile, and the resulting sub-score side-by-side** for every input — full transparency into how the composite was built, so the user can disagree with a weighting rather than just trusting a black-box number.
4. **Separate description from prediction.** The app describes *where things currently stand* on each metric relative to history. It should explicitly avoid forecasting language ("recession coming in Q3") — Dalio's own framework is diagnostic/mechanical, not predictive of timing.
5. **User-adjustable weights.** Ship with a defensible default weighting (roughly: debt growth vs. income growth as the single highest-weighted input, per Dalio's own emphasis), but let the user reweight categories and see the stage-read update live. This surfaces how sensitive the "answer" is to judgment calls.
6. **Version and log methodology changes.** Every time a scoring rule changes, log it with a date, so historical stage-reads aren't silently revised — important for credibility over time.
7. **No cherry-picked lookback windows.** Use a fixed, stated lookback (e.g., full available history) for percentile calculations, not a window chosen because it produces a cleaner signal.

---

## 3A. Scoring Math — Deep Dive

Section 3 set the principles (percentile-rank, probabilistic output, transparency, user-adjustable weights). This section works through the actual statistical machinery to implement those principles as rigorously as the data allows, and is honest about where rigor trades off against data availability.

### 3A.1 Step 1 — Feature normalization (per metric)

Two options, used situationally:

- **Empirical percentile rank (default).** For a metric with a long history (roughly 30+ years of data, most FRED series qualify), rank the current observation against its own full historical distribution using the empirical CDF: `percentile = (# historical observations ≤ current value) / N`. This makes no distributional assumption (unlike a z-score, which assumes near-normality) and is robust to the fat tails and skew that debt/credit series actually have.
- **Shrinkage z-score (for short-history metrics).** For metrics with under ~15 years of history (e.g., some spread/leverage series only start in the 1980s–90s), a raw percentile is noisy — you're ranking against too few points. Instead compute a z-score against the metric's own short history, then shrink it toward a cross-country or cross-metric-family prior (e.g., shrink an emerging short-history leverage metric toward the pooled distribution of similar leverage metrics across the five tracked economies). This is a standard empirical-Bayes move: `shrunk_z = w·own_z + (1-w)·prior_z`, with `w` increasing as the metric's own sample size grows.

Two additional normalization details matter:
- **Winsorize at the 1st/99th percentile** before ranking, so a single crisis-quarter outlier (e.g., 2020 GDP print) doesn't distort the whole historical distribution used for ranking everything else.
- **Direction-encode every metric** so that, within each category, "higher score = later/more-stressed in the cycle" consistently. Some raw metrics need inverting (e.g., credit spreads compressing is late-bubble, so score = 1 − percentile for that one) so scores can be combined without sign errors.

### 3A.2 Step 2 — Category aggregation without double-counting correlated metrics

Within a category (e.g., "Debt Burden & Growth"), several metrics are highly correlated by construction (debt/GDP, credit growth, and debt-minus-income-growth all move together). Naively averaging them effectively over-weights whatever correlated cluster is largest in a category. Two rigorous fixes:

1. **PCA per category (recommended default).** Standardize the category's metrics, take the first principal component as the category score. PC1's explained-variance share tells you honestly how "unidimensional" that category's signal actually is — if PC1 explains 80%+ of variance, one clean signal is really driving the category and that's a strong, defensible score; if PC1 explains only 40%, the category is genuinely multi-dimensional and that should be flagged (possibly split into two sub-scores) rather than forced into one number. PCA loadings are also directly reportable in the UI ("this category's score is currently driven mostly by household debt service ratio and credit growth, and less by government debt/GDP") — so it stays explainable, not a black box.
2. **Ledoit-Wolf shrinkage covariance, if you want explicit hand-set weights instead of PCA.** If there's a product reason to keep human-chosen weights (e.g., "debt growth vs. income growth" should always be the single most important input, per Dalio's own emphasis, regardless of what PCA says), use a shrinkage estimator of the covariance matrix when computing the *combined* uncertainty of the weighted score, so correlated metrics don't silently inflate confidence. This is the "keep human judgment, but don't let correlation lie to you about precision" option.

Recommendation: PCA per category as the primary method (weights emerge from data, not from opinion), with the option documented in the methodology page to switch a given category to hand-set weights if there's a strong Dalio-framework reason to override what the data's covariance structure says.

### 3A.3 Step 3 — From category scores to a stage-probability distribution (the actual classification problem)

This is a macro-regime classification problem, and there's a real rigor ladder here:

| Method | How it works | Rigor | Data demands |
|---|---|---|---|
| **Rule-based thresholds** | e.g., "if Debt Burden score > 0.8 AND Asset Price score > 0.8 → Bubble" | Low — arbitrary cutoffs, brittle, no calibrated confidence | None |
| **Analog matching (Mahalanobis distance)** | Label historical quarters with a stage (Dalio's own book already labels dozens of historical debt crises — 1920s bubble/1930s deleveraging, 2003–06 bubble/2008–09 top/2009–12 deleveraging, etc.). Compute each stage-cluster's centroid and covariance in category-score space. For the current quarter, compute Mahalanobis distance (covariance-aware, unlike Euclidean) to each stage centroid, convert distances to a softmax probability distribution | Medium-high — statistically principled, and per-metric distance decomposition is directly explainable | Moderate — needs a labeled set of historical "archetype" periods, which Dalio's case studies largely provide |
| **Hidden Markov Model (regime-switching)** | Treat the stage as an unobserved latent state with a fitted transition matrix (constrained so stages can only move to adjacent stages — no jumping Early→Deleveraging) and each stage as a Gaussian emission distribution over the category-score vector. Fit via EM/Baum-Welch, semi-supervised using Dalio's historical labels as anchors. Use the forward algorithm to get a filtered probability distribution over the *current* stage given the full observed history | Highest — produces genuinely calibrated probabilities, encodes the fact that cycles move through stages in order, and gives time-consistency (last quarter's read informs this quarter's, rather than each quarter being scored in isolation) | High — needs a long, reasonably clean historical panel; only ~1–2 full long-cycle repetitions exist even using data back to the 1800s, so this has to be semi-supervised rather than fit blind |

**Recommendation:** build analog matching (Mahalanobis) first — it's the best rigor-to-effort ratio and ships an honest probability distribution quickly. Layer in the HMM once enough historical panel data is assembled and cleaned, since it's the statistically superior method for capturing that stages move in sequence rather than independently each period. Once both exist, run them as an **ensemble**: when they agree, confidence is high; when they diverge, surface that disagreement explicitly in the UI ("model uncertainty is elevated this quarter") rather than quietly averaging it away — the disagreement itself is a real, useful signal.

### 3A.4 Step 4 — Calibration and backtesting (this is what makes "unbiased" a testable claim, not a slogan)

- **Walk-forward validation.** Fit using only data available up to time T, generate a stage-probability read for T, compare against Dalio's own retrospective labels (or macro-historical consensus) for that period. Roll forward quarter by quarter. This is the only honest way to test the framework, since fitting on the full sample and then checking it against the same sample is circular.
- **Calibration/reliability check.** If the model says "70% probability of Bubble" across many historical quarters, it should actually be right about 70% of the time in those quarters — check this with a reliability diagram and Brier score. A model that's always overconfident (or underconfident) needs recalibrating even if its top-stage picks look right.
- **Confusion matrix across the 6 long-term stages.** Expect the most confusion between *adjacent* stages (Top vs. early Deleveraging, Reflation vs. Early) — that's actually the correct behavior, since those transitions are genuinely gradual in reality, not confirmation the model is broken.

### 3A.5 Step 5 — Explainability (every stage-read must decompose)

- With PCA category scores: show each category's score and its PCA loadings — "which metrics are driving this category's number right now."
- With Mahalanobis analog matching: distance-to-archetype is a sum of per-dimension (whitened) squared differences, so it decomposes cleanly into "this metric looks unusually late-cycle compared to the 2005-06 analog; this one looks unusually early-cycle."
- With the HMM: report each metric's emission log-likelihood under the current most-probable state, surfacing which observations are most/least "typical" for that stage.

### 3A.6 Step 6 — Structural breaks / non-stationarity (the honest caveat)

The economy hasn't been stationary — gold standard era, Bretton Woods, post-1971 fiat floating, and the 2009–2021 near-zero-rate era are genuinely different regimes, and the "same" debt/GDP number doesn't mean the same thing in each. Full fix (era-conditional distributions, blended with a smooth prior across eras) is a legitimate Phase-3+ enhancement. For v1, the practical, honest approach is: use an expanding historical window (not a short rolling one) for percentiles, and explicitly flag in the UI when a current reading falls **outside the entire historical training range** for that metric — an out-of-distribution warning — rather than silently force-fitting a percentile that implies false precision.

---


## 3B. Independent Labeling Pass — Ground Truth That Doesn't Come From Dalio's Own Book

Section 3A.3 flagged a real problem: analog matching and the HMM both need historical "this quarter was in stage X" labels to anchor on, and the obvious source — Dalio's own case studies — is itself one analyst's retrospective judgment. If the app's ground truth is Dalio's labels, it can never independently confirm or challenge the framework; it can only ever agree with him by construction. This section builds a labeling pass from three sources that have nothing to do with Dalio's book, each catching a different piece of the picture, then combines them into a consensus label with an explicit confidence tier.

### 3B.1 Source A — NBER short-term cycle turning points (via FRED, fully live)

The NBER Business Cycle Dating Committee dates US recession peaks and troughs retrospectively, using a broad, multi-indicator judgment call (not a single GDP rule) — this is the closest thing to an academically authoritative, non-Dalio record of short-term cycle turning points. Conveniently, FRED already publishes this as a ready-to-pull series: **`USREC`** (NBER-based recession indicator, monthly, back to 1854, updated automatically) — so this source is genuinely live with zero scraping/parsing effort, just a standard FRED API call.

- **What it labels:** short-term cycle peak (recession start) and trough (recession end) dates only. It says nothing directly about the *long-term* debt cycle.
- **Use in the pipeline:** every NBER peak becomes a candidate "short-term cycle Top" label; every trough becomes a candidate "short-term cycle Early" label.
- **Important limitation to build in explicitly:** not every NBER recession is *debt-cycle-driven*. 2020 (COVID, a pandemic supply/demand shock) and 1973/1990 (oil-price-shock-driven) are NBER recessions with weak or absent credit-cycle buildup beforehand. Treating every NBER recession as evidence of a Dalio-style debt-cycle top would bias the framework toward over-detecting debt crises. This is exactly why Source B is needed alongside it.

### 3B.2 Source B — BIS credit-to-GDP gap (long-term cycle proxy, quarterly, ~40 economies)

The BIS publishes a credit-to-GDP gap for roughly 40 economies (most of our v1 country list included) going back to the 1960s–1970s in most cases, calculated as the difference between the credit-to-GDP ratio and its long-run trend, where the trend is a **one-sided (backward-looking) Hodrick-Prescott filter with smoothing parameter λ = 400,000** on quarterly data — deliberately different math from an ordinary two-sided HP filter, precisely so the trend doesn't use future information the app wouldn't have had in real time either.

This series was adopted under Basel III as the reference guide for countercyclical bank capital buffers, with published, non-Dalio-derived thresholds:
- **Gap < 2 percentage points:** no elevated systemic risk signal.
- **Gap between 2–10 percentage points:** rising risk, buffer scales up linearly.
- **Gap > 10 percentage points:** maximum warning level — historically associated with most major banking crises in the BIS's own cross-country evaluation.
- The BIS's own guidance states this indicator is designed to breach the lower threshold **2–3 years before** a crisis materializes — i.e., it's explicitly a leading indicator, not coincident, which matters a lot for how it gets time-aligned against NBER dates below.

- **Use in the pipeline:** treat a gap crossing above 2pp as a "long-term Bubble phase building" flag, and a crossing above 10pp as a "long-term Top imminent" flag, then check whether an NBER-dated recession/credit-crisis event actually follows within roughly 2–4 years. If it does, that's independent corroboration of a debt-cycle-style top (not just an ordinary business-cycle recession). If a gap breach occurs with no subsequent crisis, or a crisis occurs with no prior gap breach, that's logged as a genuine disagreement rather than smoothed over.
- **Feasibility:** the BIS Data Portal (`data.bis.org`) exposes this via a standard SDMX API, free, no key — a legitimate live pull, though updated quarterly with a multi-month lag (this is a real central-bank statistical release, not a market series, so it won't update daily like the market-based FRED series).

### 3B.3 Source C — Data-driven turning-point detection (Bry-Boschan / Harding-Pagan "BBQ" algorithm)

Sources A and B both rely on external institutions' judgment (NBER's committee process; BIS's choice of HP filter and thresholds). Source C removes institutional judgment entirely: apply a purely mechanical, rules-based turning-point detector directly to the same FRED series already being ingested for the metrics in §2.

The **Bry-Boschan algorithm** (the same style of rule-based turning-point logic NBER itself has historically used informally, later formalized and adapted to quarterly data by Harding & Pagan as the "BBQ" algorithm) finds local peaks and troughs in a smoothed series subject to minimum-duration constraints — e.g., requiring a minimum phase length (so a single noisy quarter doesn't count as a full cycle turn) and a minimum full-cycle length. Applying it to:
- **Real credit growth** and **debt/GDP** → independent long-term-cycle peak/trough candidates, with no reference to any institution's crisis judgment at all.
- **Wilshire 5000/GDP** or another asset-price/GDP series → independent asset-bubble peak candidates.

- **Use in the pipeline:** these turning points are the purest cross-check, since they're generated from the same raw data the app already scores, using a fixed, published, replicable rule — not another analyst's narrative. Where a Bry-Boschan-detected peak in credit/GDP lines up with a BIS gap breach and precedes an NBER recession, that's a triple-confirmed "Top" — as strong a ground-truth label as this framework can produce without appealing to any single narrative history (Dalio's included).

### 3B.4 Combining the three into a consensus label with confidence tiers

For each historical quarter, tabulate what each of the three independent sources says (short-term Top/Trough from A; long-term Bubble/Top-warning from B; mechanically-detected peak/trough from C), then assign a confidence tier:

| Tier | Rule | Use |
|---|---|---|
| **High-confidence label** | ≥2 of 3 independent sources agree on both timing (within a few quarters) and direction | Used as an anchor point for the Mahalanobis stage centroids and as semi-supervised anchors for the HMM (§3A.3) |
| **Single-source flag** | Only one source flags a turning point | Logged and shown, but excluded from training ground truth — not strong enough evidence alone |
| **Contested period** | Sources actively disagree (e.g., NBER recession with no preceding BIS gap breach — a non-debt-cycle recession like 2020 or 1973) | Explicitly excluded from "debt-cycle" ground truth, but flagged in the app as "recession not attributed to a debt-cycle top" — this is itself valuable output, since it stops the app from crying "debt crisis" at every recession |
| **Silent/no data** | Insufficient history for one or more sources (e.g., pre-1960s outside the US, where BIS credit data doesn't reach) | Rely on whichever sources do have coverage; flag reduced confidence in the UI for that era/country |

This consensus table becomes the actual training/validation ground truth referenced in §3A.3 (analog matching centroids, HMM anchors) and §3A.4 (backtesting), replacing sole reliance on Dalio's book. It's also worth surfacing directly in the product: an "independent evidence" panel for any given historical period showing exactly which of the three sources agreed, which disagreed, and which were silent — this is the most concrete expression of the "show your work, don't just assert a stage" principle from §3.

---

## 4. Data Source Map by Country (v1 scope)

| Country/bloc | Primary source | Notes |
|---|---|---|
| US | FRED (near-complete coverage, free, no key limits on most series) | Best-covered by far; build the scoring engine here first |
| Eurozone | ECB Statistical Data Warehouse (SDW), Eurostat | FRED mirrors *some* ECB series; may need direct ECB SDW API for gaps |
| Germany (Eurozone proxy) | Bundesbank, Eurostat | Useful for asset-price/credit metrics FRED doesn't carry at Eurozone level |
| Japan | Bank of Japan (BOJ) Time-Series Data Search, FRED (partial mirror) | MP2/MP3 classification is most relevant here — good stress-test of the framework |
| China | People's Bank of China, National Bureau of Statistics, IMF/BIS for credit-to-GDP | Weakest free-API coverage; expect more manual/periodic updates here |
| UK | Bank of England, ONS, FRED (partial mirror) | Good FRED coverage relative to other non-US economies |

**Practical implication:** the live-API promise is realistic and strong for the US, workable for UK/Eurozone/Japan with some direct central-bank-API integration, and weakest for China — plan for that country to lean more on periodic manual/quarterly updates in v1 rather than true live pulls.

---

## 5. High-Level Architecture (for when we move past the design doc)

```
[Data Ingestion Layer]
  - FRED API client (US + partial others)
  - ECB SDW / BOJ / ONS / ECB-adjacent clients (per-country)
  - Scheduled pulls (daily for market-based series, monthly/quarterly for survey-based ones)
  - Raw data store (time series, per country, per metric)

[Calculation Engine]
  - Historical percentile-rank calculator per metric per country
  - Category sub-scores (Debt Burden, Asset Prices, CB Stance, Deleveraging Signals, External)
  - Composite long-term-cycle stage distribution
  - Composite short-term-cycle stage distribution
  - User-adjustable weight overrides

[Presentation Layer]
  - Country dashboard: current stage distribution (long + short term) with confidence
  - Metric drill-down: raw series, historical percentile, chart
  - Methodology page: weights, lookback window, source list, version log
  - Cross-country comparison view
```

## 6. Suggested Phasing

1. **Phase 1** — US-only backend: ingest the metrics in §2, build the percentile/scoring engine, validate the stage-read against known historical periods (e.g., does it correctly flag 2006–07 as "Bubble/Top" and 2009–2012 as "Deleveraging"?). This backtest is the real validation of the framework implementation.
2. **Phase 2** — Dashboard/UI on top of the validated US engine.
3. **Phase 3** — Extend to UK, Eurozone, Japan (better free-data coverage); add China last given weaker live-data availability.

---

## 7. Phase 1 Backtest Plan (US-only)

**Objective:** determine whether the scoring engine (§3A) — trained/anchored on the independent labels from §3B, not on Dalio's own case studies — actually reproduces sensible stage reads across US history, using pre-registered pass/fail criteria decided *before* looking at results, so this can't quietly become a fitting exercise.

### 7.1 Building the test panel

- **History window:** as far back as the input data supports at reasonable quality — FRED's core series (`FEDFUNDS`, `UNRATE`, `CPIAUCSL`, credit aggregates) mostly go back to the 1950s–60s; the BIS credit-to-GDP gap starts in 1961 for the US. So the panel realistically starts **~1962**, quarterly, giving roughly 255 quarters (~64 years) through today.
- **Labels:** pull the §3B consensus table for every quarter in that window — high-confidence, single-source, or contested. Only high-confidence quarters are used to *score* the backtest; single-source and contested quarters are still run through the model and logged, but treated as "no graded answer" rather than counted as a miss.

### 7.2 Walk-forward design (not a single train/test split)

Given ~64 years of quarterly data, a single train/test split would waste most of the history and only validate one period. Instead:

1. Pick a **burn-in point** — the first date with enough trailing history for percentile ranking to be meaningful (roughly 1985, giving ~23 years of trailing data for the earliest scored quarter).
2. From 1985Q1 forward, for each quarter T: fit/refit the percentile distributions, PCA category scores, and Mahalanobis stage centroids using **only data and labels dated before T** (no look-ahead), generate a stage-probability read for T, then compare against the label for T once labels through T are available.
3. Refit the model periodically (e.g., annually rather than every single quarter) to keep this computationally reasonable — refitting quarterly would be more rigorous but isn't necessary to get a meaningful read on whether the approach works at all.
4. **Known limitation, stated up front:** doing this properly would use *real-time-vintage* data (FRED's `ALFRED` vintage archive) so the backtest never accidentally uses a GDP or employment figure that was later revised — since NBER and BIS labels are themselves finalized well after the fact using revised data, a naive backtest using today's revised series has some look-ahead leakage baked in. Treat Phase 1 as using latest-vintage data for simplicity and speed, and treat a vintage-correct rebuild using ALFRED as a explicit Phase 1b follow-up before treating the backtest as fully rigorous.

### 7.3 Pre-registered pass/fail criteria

Decide these numbers now, before running anything, precisely so a disappointing result can't get quietly redefined as a pass after the fact:

| Test | Metric | Target | What failing it means |
|---|---|---|---|
| **Top-1 accuracy** | Does the model's highest-probability long-term stage match the high-confidence label? | ≥65–70% across all scored quarters | Below this, the composite scoring isn't tracking the independent labels well enough to trust |
| **Calibration** | Brier score / reliability diagram across probability bins | Reliability curve close to the 45° line; Brier score meaningfully better than a naive baseline (e.g., always predicting the unconditional base rate of each stage) | If the model claims "80% confidence" quarters that are only right 40% of the time, the probabilities are decorative, not real |
| **Confusion pattern** | Confusion matrix across the 6 long-term stages | Errors concentrated in **adjacent** stages (Top↔early Deleveraging, Reflation↔Early); very few errors between *distant* stages (e.g., Bubble misread as Deleveraging) | Distant-stage confusion means the model isn't capturing the underlying ordering at all — a much more serious failure than adjacent-stage fuzziness |
| **Negative-case discipline** | During §3B "contested" quarters (2020 COVID, 1973/1979 oil shocks) — does the model avoid assigning a high long-term "Bubble" or "Top" score despite the NBER recession flag? | Long-term stage read should stay closer to whatever it was pre-shock; only the *short-term* cycle read should move sharply | If the model spikes "debt-cycle Top" every time NBER flags any recession, it has effectively become a recession detector wearing a Dalio costume, not an actual debt-cycle read |
| **Lead-time behavior** | Does the model's own composite "Bubble" score rise ahead of an eventual high-confidence "Top" label, roughly tracking the BIS gap's documented ~2–3 year lead property? | Composite Bubble score should be elevated at least 4–8 quarters before a Top label, not coincident with or lagging it | A model that only "calls" the top after it's already obvious isn't adding value over just reading the BIS gap or the yield curve directly |

### 7.4 Specific historical fixtures to check by hand, not just in aggregate

Aggregate accuracy numbers can hide an engine that gets easy cases right and hard, interesting cases wrong. Spot-check these specific US episodes individually once the walk-forward run completes:

| Period | Why it's a useful test case |
|---|---|
| 1966–70 credit crunch / late-60s tightening | Early, less-documented episode — tests whether the engine works before the "obvious" modern cases |
| 1973–75 and 1979–82 oil-shock recessions | Expected **contested/negative case**: NBER recessions with comparatively weak preceding credit-gap buildup — a good test of the model *not* over-flagging every recession as a debt-cycle top |
| 1980 & 1981–82 Volcker double-dip | Short-term cycle Tops driven by deliberate monetary tightening rather than an organic asset bubble — tests whether the short-term/long-term stage split behaves sensibly when they diverge |
| Late-1980s S&L crisis / 1990–91 recession | A genuine credit-driven regional (real estate/banking) stress episode, useful mid-difficulty case |
| 2001 dot-com bust | Asset-price bubble (equities) with a comparatively muted credit-cycle signature — tests whether the engine can distinguish an asset bubble from a *debt*-cycle bubble, which Dalio's framework treats as related but distinct |
| 2003–07 housing boom → 2008 GFC | The cleanest, best-documented classic case — this one **should** score as an unambiguous high-confidence Bubble→Top→Deleveraging sequence; if the model doesn't get this one clearly right, something in the pipeline is broken |
| 2009–2012 | Should read as "Deleveraging," with the real-GDP-vs-credit-contraction split determining beautiful vs. ugly sub-classification |
| 2020 COVID recession | Second key **negative case** alongside the 1970s oil shocks — exogenous shock, should not read as a debt-cycle top despite the NBER recession flag |
| 2021–23 inflation surge / tightening cycle | Recent, still being revised in the data — useful test of how the engine handles a period that doesn't yet have a fully settled historical label |

### 7.5 Deliverable

A written backtest report containing: the walk-forward accuracy/calibration/confusion-matrix results against the pre-registered targets in §7.3, the individual fixture write-ups from §7.4, and an explicit go/no-go call — if the core 2003–08 case and the two negative cases (1970s, 2020) all behave correctly even if aggregate accuracy on obscure early periods is mediocre, that's a reasonable basis to proceed to Phase 2 (dashboard) while flagging the weaker periods; if the negative cases fail (model over-flags ordinary recessions as debt-cycle tops), that's a stop-and-rework signal on the scoring math in §3A before building any UI on top of it.

---

*Open design questions to resolve before building: (a) exact default category weights, (b) how to handle metrics with short history (e.g., some spread/leverage series only go back to the 1980s–90s) in the percentile calculation, (c) how to handle China's data-quality/availability gap without making the cross-country comparison misleading, (d) whether to invest in ALFRED real-time-vintage data for a fully look-ahead-free backtest before Phase 2, or accept latest-vintage data as a reasonable Phase 1 approximation.*
