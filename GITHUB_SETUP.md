# Auto-updating dashboard: GitHub setup

Every business day, at about 9:17am and 6:17pm New York time, GitHub runs the full pipeline against live FRED data, runs the tests, and republishes the dashboard at `https://<your-username>.github.io/<repo-name>/`. Your PC does not need to be on. You can also trigger an update any time from the Actions tab.

## 1. Add the new files to your PyCharm project

Unzip `debt_cycle_project.zip` into the project root (the folder that **contains** `debt_cycle_scoring/`). You should end up with:

```
<project root>/
  .github/workflows/update-dashboard.yml    <- the twice-daily job
  dashboard/template.html                   <- the page layout (no data in it)
  debt_cycle_scoring/scorer/fiscal_panel.py        <- new
  debt_cycle_scoring/scorer/test_fiscal_panel.py   <- new
  debt_cycle_scoring/scorer/market_pulse.py        <- new (daily bond/equity panel)
  debt_cycle_scoring/scorer/test_market_pulse.py   <- new
  export_dashboard.py                       <- runs pipeline, writes site/index.html
  run_metric_audit.py
  requirements.txt
  .gitignore
  debt_cycle_scoring/ ...                   <- everything you already had
```

`.github` starts with a dot, so Windows Explorer may hide it. Check it's there with `dir -Force` in PowerShell.

## 2. Test it locally first

```powershell
$env:FRED_API_KEY="<your key>"
python -m debt_cycle_scoring.scorer.test_fiscal_panel
python -m debt_cycle_scoring.scorer.test_market_pulse
python export_dashboard.py
start site\index.html
```

The page should open with today's numbers. If this works locally, it will work on GitHub.

## 3. Make sure no API key is in any file

The repo will be **public**, since free GitHub Pages requires a public repo. That's fine: the key never goes in the code. Before uploading, search for anything that looks like a FRED key:

```powershell
Get-ChildItem -Recurse -Include *.py,*.md,*.txt,*.json,*.yml | Select-String -Pattern '[0-9a-f]{32}'
```

It should print nothing. If it finds a key, delete it from that file.

Both keys pasted in our earlier chats should be treated as exposed. If you haven't already, **generate a fresh key** at fredaccount.stlouisfed.org → API Keys, delete the old ones, and use only the new one below.

## 4. Put the project on GitHub

In PyCharm: **Git → GitHub → Share Project on GitHub**. Sign in, name the repo (e.g. `debt-cycle-monitor`), leave "Private" **unchecked**, and click Share. In the commit dialog, confirm `site/` and `__pycache__` are *not* listed (`.gitignore` excludes them), then commit.

## 5. Store the key as a secret

On github.com, open the repo and go to **Settings → Secrets and variables → Actions → New repository secret**.
- Name: `FRED_API_KEY`
- Secret: your new key

GitHub encrypts it. It is never shown in logs or on the page.

## 6. Turn on Pages

Go to **Settings → Pages → Build and deployment → Source: GitHub Actions**.

## 7. Run it the first time

Go to **Actions → Update debt-cycle dashboard → Run workflow**. It takes about 3-5 minutes. When it finishes with a green check, the deploy step shows your URL. Bookmark it.

## How it behaves

- **Update timing.** Bond and equity data refresh every business day. FRED posts each series once per day after the close, so the morning run picks up overnight posts (VIX, credit spreads, S&P 500, dollar) and the evening run picks up same-day posts (Treasury yields, curve, breakevens). Running more often would only download the same numbers.
- **An open tab refreshes itself.** The page checks every 15 minutes, and whenever you return to the tab, for a newer publish, and reloads automatically.
- **The structural score still changes quarterly, by design.** Daily market moves appear in the *Daily market pulse* panel and are never blended into the composite.
- **A failed run never breaks the page.** If FRED is down, a series is delisted, or a test fails, the deploy step doesn't run and the last good version stays up. You'll get an email about the failed run.
- **Delisted series degrade gracefully.** A missing series is skipped with a warning in the run log, as happened with Wilshire and gold, instead of crashing the run.
- **GitHub pauses scheduled jobs after 60 days with no commits** to a public repo. GitHub emails you first. Click "Enable workflow" in the Actions tab, or push any commit.
- **Updating the tool.** Edit code in PyCharm, then **Git → Commit and Push**. The next run uses the new code. To update right away, click Run workflow.
- **The page is public**, but the URL isn't listed anywhere. Anyone with the link can view it; no one can see the key.
