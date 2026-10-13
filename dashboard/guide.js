// ============================================================================
// Plain-English layer: scenario engine, summary panel, scenario guide, and the
// click-to-explain popovers. Runs after the main dashboard script has rendered.
// Everything here is computed from DATA at page load -- no hand-entered numbers.
// ============================================================================
(function () {
  if (typeof DATA === "undefined" || !DATA) return;
  const $ = id => document.getElementById(id);
  const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const H = DATA.headline;

  // ------------------------------------------------------------ facts from DATA
  const F = Object.fromEntries((DATA.flags || []).map(f => [f.key, f]));
  const C = Object.fromEntries((DATA.fiscal || []).map(m => [m.key, m]));
  const rp = k => (C[k] && C[k].risk_percentile != null ? C[k].risk_percentile : null);
  const stress = DATA.short ? DATA.short.stress : null;
  const curve = F.yield_curve ? F.yield_curve.v : null;
  const hist = DATA.history || [];
  const recentRecession = hist.slice(-6).some(h => h.rec === 1);
  const stretched = (DATA.fiscal || []).filter(m => m.risk_percentile != null && m.risk_percentile >= 75);
  const fiscalStretched = stretched.filter(m => (m.group || "").startsWith("Sovereign"));
  const valuationStretched = stretched.filter(m => (m.group || "").startsWith("Bubble"));
  const a8 = (DATA.auc || []).find(a => a.h === 8);
  const auc8txt = a8 && a8.auc != null ? a8.auc.toFixed(2) : "n/a";
  const band = r => (r < 35 ? "low" : r < 50 ? "moderate" : r < 65 ? "elevated" : "high");
  const stressBand = s => (s == null ? null : s < 35 ? "unusually calm" : s < 65 ? "normal" : "unusually stressed");
  const yearAgo = hist.length >= 5 ? hist[hist.length - 5].r : null;
  const lastHist = hist.length ? hist[hist.length - 1].r : null;
  const trend = (yearAgo == null || lastHist == null) ? "" :
    (lastHist - yearAgo >= 5 ? "rising" : lastHist - yearAgo <= -5 ? "falling" : "roughly steady");

  const PLAIN_FLAG = {
    credit_gap: "private debt growth", debt_service_ratio: "household debt payments",
    yield_curve: "the yield curve", m2_velocity: "money velocity",
    long_bond_level: "long-term interest rates", debasement: "inflation expectations",
  };

  // ------------------------------------------------------------ scenarios
  // Priority order = order of the list. The first whose test passes is "where
  // we are now"; any others that also pass are shown as "also present".
  const UNIVERSAL = [
    "Dalio’s own answer to not knowing what comes next is balance: hold assets that do well when growth or inflation is rising and assets that do well when they’re falling (the “All Weather” idea), then rebalance rather than trying to time each turn.",
  ];
  const SCENARIOS = [
    { key: "acute_stress", title: "Acute market stress",
      test: () => stress != null && stress >= 65,
      now: () => `Markets are under acute stress right now (short-horizon gauge ${stress}/100, against a normal of about 50).`,
      matters: "Stress episodes have tended to be sharp but short when the debt backdrop was healthy (1987, 1998, 2020) and long and damaging when debt was stretched (2008). Compare this with the slow debt score to tell which kind this is.",
      portfolio: [
        "Historically, high-quality government bonds and cash have usually held their value or gained in stress spikes, while stocks and corporate credit fell together.",
        "The biggest permanent losses have come from forced selling — margin calls, or needing cash at the worst moment — rather than from the decline itself.",
        "Investors with a plan have typically focused on liquidity and on rebalancing back to targets rather than selling into the fall.",
      ],
      watch: ["Whether the slow debt score rises too (a debt-driven crisis) or stays calm (a shock that may pass).", "Credit spreads: still widening, or starting to narrow?"] },
    { key: "deleveraging", title: "Deleveraging and policy response",
      test: () => recentRecession || (rp("fed_funds") != null && rp("fed_funds") >= 90 && C.fed_bs_1y && C.fed_bs_1y.percentile >= 85),
      now: () => "The economy is in or just past a downturn, and policy is in rescue mode: rates are near the floor and/or the Fed is expanding its balance sheet.",
      matters: "Dalio separates “beautiful” deleveragings, where money printing offsets the deflationary pull of debt reduction, from “ugly” ones, where it doesn’t. Which one unfolds depends on how hard and how fast policy acts.",
      portfolio: [
        "Government bonds have usually rallied early, as rates were cut, then stalled once rates hit the floor.",
        "When money printing was strong enough, stocks and other risk assets tended to recover well before the economy did.",
        "If printing overshoots, inflation-protected bonds and real assets have tended to benefit; if it falls short, cash and high-quality bonds have held up best.",
      ],
      watch: ["Credit spreads narrowing (rescue working) or widening again (not working).", "Inflation expectations: drifting up toward or above 2% suggests the reflation is taking hold."] },
    { key: "late_cycle", title: "Late cycle: yield-curve warning",
      test: () => curve != null && curve >= 60,
      now: () => `The yield curve is flat or inverted (curve flag ${curve}/100): short-term rates are at or above long-term rates.`,
      matters: "An inverted 10-year/3-month curve has come before every US recession since the late 1960s, usually by 6 to 24 months, with very few false alarms. In this tool it is the strongest single recession signal.",
      portfolio: [
        "Stocks have often kept rising for a while after the curve inverted; the market peak tended to come later, so inversion alone has been a poor sell signal.",
        "In the recessions that followed, high-quality government bonds generally gained as rates were cut, while stocks, high-yield credit and economically sensitive sectors fell.",
        "Cash has paid relatively well in this phase because short-term rates are high.",
        "Investors commonly check whether their portfolio could ride out a 20–30% stock decline without forced selling, and avoid adding leverage.",
      ],
      watch: ["The curve re-steepening because short rates are being cut — historically this has often come just before or at the start of a recession.", "Rising unemployment and widening credit spreads."] },
    { key: "bond_squeeze", title: "Bond bear market: debt supply outrunning demand",
      test: () => F.long_bond_level && F.long_bond_level.v >= 75 &&
                  ((rp("term_premium") != null && rp("term_premium") >= 75) || (rp("interest") != null && rp("interest") >= 75)) &&
                  (stress == null || stress < 65),
      now: () => {
        const tp = rp("term_premium"), it = rp("interest");
        const parts = [];
        if (tp != null && tp >= 75) parts.push("investors are demanding a historically large premium to hold long-term bonds");
        if (it != null && it >= 75) parts.push("interest is taking a historically large share of federal revenue");
        // supporting detail from the Treasury data, when present
        const extra = [];
        if (C.refi_gap && C.refi_gap.latest != null && C.refi_gap.latest > 0)
          extra.push(`market rates are ${C.refi_gap.latest.toFixed(1)} points above the average rate on existing debt, so the interest bill will keep rising as old debt is refinanced`);
        if (C.tic_jp_cn && C.tic_jp_cn.latest != null && C.tic_jp_cn.latest < 0)
          extra.push(`Japan and China have cut their Treasury holdings ${Math.abs(C.tic_jp_cn.latest).toFixed(1)}% over the past year`);
        return `Long-term interest rates are at the top of their 10-year range (long-bond flag ${F.long_bond_level.v}/100), ${parts.join(" and ")}, while market stress is not yet elevated.` +
          (extra.length ? ` Behind that: ${extra.join("; ")}.` : "") +
          ` In Dalio’s sequence this is the middle stage: rates have risen, but stocks, credit and the economy haven’t cracked yet.`;
      },
      matters: "Dalio’s two conditions for a country “going broke” are more debt to sell than buyers want, and debt service squeezing out other spending. The first shows up as rising long-term rates. His sequence: rates rise first; credit spreads and stocks weaken later; then either a downturn shrinks borrowing or the central bank prints money to absorb the debt. Either way, holders of the debt lose in real terms.",
      portfolio: [
        "In past bond bear markets driven by inflation or heavy supply (the late 1960s to 1981, and 2022), long-dated bonds took the largest losses, while short-term bills and cash held their value and their yields rose as rates reset.",
        "Stocks have often held up at first, then fallen once higher rates fed through to borrowing costs and valuations. In 2022 stocks and bonds fell together — one of the worst years for a traditional stock-bond mix in decades.",
        "Inflation-protected bonds guard against inflation but not against rising real interest rates, so long-dated ones also lost money in 2022; shorter-dated ones held up better. Real assets and commodities did well through the 1970s.",
        "Dalio’s own reading (Oct 2026): the bond bear market “probably has a long way to go”, higher rates will eventually weigh on stocks and the economy, and he finds it “easier to be short than long US Treasuries”. That is his view, which he labels not investment advice.",
      ],
      watch: [
        "Credit spreads widening and corporate free cash flow falling — Dalio’s early signs that higher rates are biting.",
        "Large corporate borrowers having to offer bigger price concessions to sell bonds (his “canary in the coal mine”).",
        "Foreign holders, especially China and Japan, selling or hedging their Treasuries.",
        "The Fed’s Treasury-bill purchases broadening into larger buying — the money-printing exit, which devalues money and debt.",
      ] },
    { key: "credit_bubble", title: "Credit bubble building",
      test: () => H.regime !== "calm" && H.gap != null && H.gap >= 2,
      now: () => `Private debt is growing faster than its long-run trend (credit gap ${H.gap >= 0 ? "+" : ""}${H.gap.toFixed(1)} points of GDP): the build-up phase of Dalio’s cycle.`,
      matters: "Fast private credit growth has preceded most banking crises worldwide, usually by two to three years (BIS research). The danger tends to arrive late, after a period in which everything looks fine.",
      portfolio: [
        "During the build-up, stocks and other risk assets have often kept doing well — the bust comes later.",
        "Periods of fast credit growth and very tight credit spreads have been followed by weaker returns on corporate bonds and slower growth (López-Salido, Stein and Zakrajšek, 2017).",
        "In the bust, banks, heavily indebted companies and whatever asset the credit financed (housing in 2008) have been hit hardest.",
        "Investors commonly become wary of reaching for extra yield and favour borrowers that don’t depend on refinancing.",
      ],
      watch: ["The credit gap passing +10 points (the BIS maximum-warning level).", "Household debt payments rising and banks tightening lending standards."] },
    { key: "stretched_public", title: "Calm private sector, stretched government debt and valuations",
      test: () => H.risk < 50 && (fiscalStretched.length + valuationStretched.length) >= 3,
      now: () => {
        const names = [...fiscalStretched, ...valuationStretched].slice(0, 4).map(m => m.name);
        return `Private borrowing and near-term recession signals are not flashing, but several of Dalio’s longer-term warning signs are stretched: ${names.join(", ")}.`;
      },
      matters: "This is the setup Dalio describes in “How Countries Go Broke”: the risk sits with the government’s balance sheet and with asset prices rather than with households and banks. He sees two ways it resolves — rates rise and credit tightens, hurting bonds and then stocks; or the central bank prints money to absorb the debt, devaluing money and debt — and in both, holders of the debt lose in real terms. It builds slowly, and the timing is very uncertain.",
      portfolio: [
        "High valuations have historically been followed by lower average stock returns over the next ten years, but they have said little about the next one or two — markets can stay expensive for a long time.",
        "When heavily indebted governments run large deficits, the historical end game has usually been money printing and a weaker currency rather than outright default. That has eroded the real value of long-term fixed-rate bonds and cash.",
        "Assets that have historically held real value in that kind of ending include inflation-protected bonds (TIPS), real assets and commodities, gold, and holdings spread across countries and currencies. (Gold isn’t tracked here because FRED removed its gold price data.)",
        "The near-term recession signals in this tool are calm, so its evidence does not point to an abrupt turn; the risks here are slow-moving and their timing is unknown.",
      ],
      watch: ["Long-term rates and the term premium rising — signs buyers are balking at US debt (this would move us into the bond-bear-market scenario).", "The foreign share of federal debt falling.", "Breakeven inflation moving well above 2%.", "Interest costs taking an ever larger share of tax revenue."] },
    { key: "healthy", title: "Healthy expansion",
      test: () => true,
      now: () => "Debt growth, interest rates and market stress are all within normal ranges.",
      matters: "This is the stage of the cycle where growth is supported by income rather than borrowing. It can last many years.",
      portfolio: [
        "Stocks and corporate credit have historically done best in this phase, and the cost of being too defensive has been high.",
        "Cash and short-term bonds have tended to lag.",
        "Investors commonly stay invested and diversified, rebalancing back to their targets rather than chasing what has risen most.",
      ],
      watch: ["The yield curve flattening toward zero.", "The credit gap climbing above +2 points.", "Credit spreads widening from low levels."] },
  ];
  const matched = SCENARIOS.filter(s => { try { return s.test(); } catch (e) { return false; } });
  const primary = matched[0];
  const alsoPresent = matched.slice(1).filter(s => s.key !== "healthy");

  // ------------------------------------------------------------ why the score is where it is
  function whyRisk() {
    const fl = (DATA.flags || []).slice().sort((a, b) => b.contrib - a.contrib);
    if (!fl.length) return "";
    const top = fl.slice(0, 2).map(f => `${PLAIN_FLAG[f.key] || f.name} (${f.v}/100)`);
    const low = fl.slice().sort((a, b) => a.v - b.v).slice(0, 2).map(f => `${PLAIN_FLAG[f.key] || f.name} (${f.v})`);
    const reg = H.gap == null ? "the latest credit-gap reading isn’t available yet, so a neutral setting is used"
      : H.gap < 2 ? `private debt is not running ahead of the economy (credit gap ${H.gap.toFixed(1)} points)`
      : H.gap < 10 ? `private debt is growing faster than trend (credit gap +${H.gap.toFixed(1)} points)`
      : `private debt is far above trend (credit gap +${H.gap.toFixed(1)} points, the BIS danger zone)`;
    return `The biggest contributors to today’s score are ${top.join(" and ")}. The calmest ingredients are ${low.join(" and ")}. ` +
      `The regime is “${H.regime}” because ${reg}.`;
  }

  // ------------------------------------------------------------ plain-English panel
  function list(items) { return "<ul>" + items.map(x => "<li>" + esc(x) + "</li>").join("") + "</ul>"; }
  function buildPlain() {
    const box = $("plainBody");
    if (!box || !primary) return;
    const extremes = (DATA.fiscal || []).filter(m => m.risk_percentile != null && m.risk_percentile >= 90)
      .map(m => `${m.name} (${m.risk_dir < 0 ? "low" : m.percentile + "th percentile"})`);
    let html = "";
    html += `<p class="lead">Debt-cycle risk is <b>${band(H.risk)}</b> at <b>${H.risk}/100</b>` +
      (trend ? `, and has been <b>${trend}</b> over the past year` : "") + `. ` +
      (stress != null ? `Market nerves are <b>${stressBand(stress)}</b> (${stress}/100).` : "") + `</p>`;
    html += `<div class="pe-grid">`;
    html += `<div><h3>Where we are</h3><p><span class="scn-tag">${esc(primary.title)}</span></p><p>${esc(primary.now())}</p>` +
      (alsoPresent.length ? `<p class="muted">Also present: ${alsoPresent.map(s => esc(s.title)).join("; ")}.</p>` : "") + `</div>`;
    html += `<div><h3>Why the score is where it is</h3><p>${esc(whyRisk())}</p></div>`;
    html += `<div><h3>Why it matters</h3><p>${esc(primary.matters)}</p>` +
      (extremes.length ? `<p class="muted">Outside the score, at historical extremes: ${esc(extremes.slice(0, 5).join("; "))}.</p>` : "") + `</div>`;
    html += `<div><h3>What would change the picture</h3>${list(primary.watch)}</div>`;
    html += `</div>`;
    html += `<div class="pe-port"><h3>What history says for portfolios in this scenario</h3>${list(primary.portfolio.concat(UNIVERSAL))}` +
      `<p class="disclaim"><b>General historical information, not personal financial advice.</b> This tool has been tested on recession timing (8-quarter AUC ${auc8txt} across ${DATA.n_recessions || "a handful of"} recessions), never on investment returns, and past patterns may not repeat. What suits any one person depends on their goals, time horizon and other holdings; a fee-only fiduciary adviser can help apply it.</p></div>`;
    html += `<p class="muted"><a href="#scenarioGuide">See all seven scenarios and what each would look like →</a></p>`;
    box.innerHTML = html;
  }

  function buildGuide() {
    const box = $("scenarioBody");
    if (!box) return;
    box.innerHTML = SCENARIOS.map(s => {
      const isNow = primary && s.key === primary.key;
      const also = alsoPresent.some(a => a.key === s.key);
      const badge = isNow ? `<span class="scn-now">where we are now</span>` : also ? `<span class="scn-also">also present</span>` : "";
      return `<details${isNow ? " open" : ""} class="scn"><summary>${esc(s.title)} ${badge}</summary>` +
        `<div class="scn-b"><p><b>Why it matters:</b> ${esc(s.matters)}</p>` +
        `<p><b>What history says for portfolios:</b></p>${list(s.portfolio)}` +
        `<p><b>Signs it’s arriving or ending:</b></p>${list(s.watch)}</div></details>`;
    }).join("") +
      `<p class="disclaim">${esc(UNIVERSAL[0])} General historical information, not personal financial advice.</p>`;
  }

  // ------------------------------------------------------------ ⓘ popovers
  const pop = document.createElement("div");
  pop.className = "xpop"; pop.id = "xpop"; pop.setAttribute("role", "dialog"); pop.hidden = true;
  document.body.appendChild(pop);
  let openFor = null;

  function showExplain(key, btn) {
    const x = EXPLAIN[key];
    if (!x) return;
    pop.innerHTML =
      `<button class="xclose" aria-label="Close">×</button>` +
      `<div class="xt">${esc(x.t)}</div>` +
      `<div class="xs"><b>In plain English</b><p>${esc(x.e)}</p></div>` +
      (x.r ? `<div class="xs"><b>How to read it</b><p>${esc(x.r)}</p></div>` : "") +
      (x.w ? `<div class="xs"><b>Why it matters</b><p>${esc(x.w)}</p></div>` : "") +
      (x.q ? `<div class="xs xq"><b>Technical</b><p>${esc(x.q)}</p></div>` : "") +
      (x.s ? `<div class="xs xsrc"><b>Source</b><p>${esc(x.s)}</p></div>` : "");
    pop.hidden = false;
    const mobile = innerWidth < 640;
    pop.classList.toggle("sheet", mobile);
    if (!mobile) placeNear(btn); else { pop.style.left = ""; pop.style.top = ""; }
    pop.querySelector(".xclose").onclick = closeExplain;
    openFor = btn;
    btn.setAttribute("aria-expanded", "true");
    pop.querySelector(".xclose").focus({ preventScroll: true });
  }
  function placeNear(btn) {
    const r = btn.getBoundingClientRect();
    const w = pop.offsetWidth, h = pop.offsetHeight;
    const left = Math.min(Math.max(8, r.left - 20), innerWidth - w - 8);
    let top = r.bottom + 8;
    if (top + h > innerHeight - 8) top = Math.max(8, r.top - h - 8);
    pop.style.left = left + "px"; pop.style.top = top + "px";
  }
  function closeExplain() {
    if (pop.hidden) return;
    pop.hidden = true;
    if (openFor) { openFor.setAttribute("aria-expanded", "false"); openFor.focus({ preventScroll: true }); }
    openFor = null;
  }
  document.addEventListener("click", e => {
    const b = e.target.closest(".xbtn");
    if (b) { e.stopPropagation(); (openFor === b ? closeExplain() : showExplain(b.dataset.key, b)); return; }
    if (!pop.hidden && !pop.contains(e.target)) closeExplain();
  });
  document.addEventListener("keydown", e => { if (e.key === "Escape") closeExplain(); });
  addEventListener("resize", closeExplain);
  // On scroll, keep the card next to its button; close only once the button leaves the screen.
  addEventListener("scroll", () => {
    if (pop.hidden || !openFor || pop.classList.contains("sheet")) return;
    const r = openFor.getBoundingClientRect();
    if (r.bottom < 0 || r.top > innerHeight) { closeExplain(); return; }
    placeNear(openFor);
  }, { passive: true });

  function attachButtons() {
    document.querySelectorAll("[data-x]").forEach(el => {
      const key = el.getAttribute("data-x");
      if (!EXPLAIN[key] || el.querySelector(":scope > .xbtn")) return;
      const b = document.createElement("button");
      b.type = "button"; b.className = "xbtn"; b.dataset.key = key;
      b.setAttribute("aria-label", "Explain: " + EXPLAIN[key].t);
      b.setAttribute("aria-expanded", "false");
      b.textContent = "i";
      el.appendChild(b);
    });
  }

  buildPlain();
  buildGuide();
  attachButtons();
  // exposed for tests
  window.__guide = { primary: primary && primary.key, matched: matched.map(s => s.key) };
})();
