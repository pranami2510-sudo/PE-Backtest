# PE Discount-to-Median Backtest

Backtest a strategy that buys stocks when their **PE is at a discount** to their own **rolling median PE**, holds for a fixed number of quarters, then sells. Includes optional **fixed Q4** rebalancing (rebalance only at year-end, hold 3 quarters, cash in Q4).

## Strategy (short)

- **Screen:** Each quarter (or every H quarters), keep only names with  
  `PE ≤ (1 − discount) × median_PE`  
  where `median_PE` is the median of daily PE over the last **L** quarters.
- **Trade:** Buy passers at equal weight at quarter end; sell at the end of **H** quarters.
- **Benchmark:** NIFTY 50 (from Yahoo Finance if no CSV provided).

See **[strategy_log.md](strategy_log.md)** for a full description with diagrams.

## Setup

1. **Clone the repo** (or download).
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Data:** Place your input data in the project folder:
   - **Option A — Run backtest only:** Put company CSVs in **`Cleaned PE Data No Outliers`**.  
     Each CSV: `Date`, `Quarter`, `Close`, `PE` (one file per company).
   - **Option B — Rebuild no-outliers data:** Put company CSVs in **`Cleaned PE Data`**, then run:
     ```bash
     python build_cleaned_no_outliers.py
     ```
     This creates/updates **`Cleaned PE Data No Outliers`** (outlier companies dropped, PE/price cleaning applied).

Optional: **`exclude_companies.txt`** in the project root (one company name per line) to drop additional names when building or loading data.

## Run backtest

**Interactive (prompts for inputs):**
```bash
python pe_backtest.py
```

**Non-interactive (e.g. lookback=3, discount=20%, holding=2):**
```bash
python pe_backtest.py --lookback 3 --discount 0.2 --holding 2
```

**Option A (fixed Q4 rebalance):**
```bash
python pe_backtest.py --fixed-q4 --lookback 3 --discount 0.2
```

**Outputs** (in the project folder):

- `tradelog.csv` — all buys and sells  
- `equity_curve.csv` — daily normalized portfolio value  
- `equity_curve.png` — strategy vs NIFTY 50 (log scale)  
- Console: CAGR, Sharpe, Calmar, max drawdown, benchmark metrics  

## Run in Streamlit (web UI)

**Local:**
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```
Then open the URL shown (e.g. http://localhost:8501). Use the sidebar to set lookback, discount, and holding; click **Run backtest** to see the equity curve and metrics.

## Host on Streamlit Cloud

1. **Push the repo to GitHub** (e.g. [pranami2510-sudo/PE-Backtest](https://github.com/pranami2510-sudo/PE-Backtest)).
2. Go to **[share.streamlit.io](https://share.streamlit.io)** and **sign in with the same GitHub account that owns the repo** (e.g. `pranami2510-sudo`). If you use multiple GitHub accounts, sign out and sign in with the one that has the repo.
3. Click **"New app"** (or "Deploy an app").
4. Fill the form:
   - **Repository:** `pranami2510-sudo/PE-Backtest` (or paste the repo URL)
   - **Branch:** `main` (not `master`)
   - **Main file path:** `streamlit_app.py`
   - **App URL (optional):** e.g. `pricetoearningsbacktest` → app will be at `https://pricetoearningsbacktest.streamlit.app`
5. Click **Deploy**. Wait a few minutes for the first build. Open the app from the link Streamlit shows (or from **Your apps** in the dashboard).

**If you see "You do not have access to this app":**
- You're likely opening a URL for an app that wasn’t deployed from your account, or the app was never created. Always open the app from **share.streamlit.io → Your apps → [your app]** so you use the correct URL.
- Ensure you’re signed in with the **GitHub account that owns the repo**. In Streamlit Cloud settings, check **Source control** / connected account and reconnect GitHub if needed.
- If the repo is **private**, grant Streamlit access to it in GitHub (Settings → Applications → Authorized OAuth Apps), or make the repo **public**.

**Note:** The app reads data from **Cleaned PE Data No Outliers**. If that folder isn’t in the repo (e.g. too large), the hosted app will load but show "No data folder"; run backtests locally with your data or add a small sample of CSVs to the repo.

## Project structure

```
.
├── README.md
├── strategy_log.md          # Strategy description and diagrams
├── requirements.txt
├── pe_backtest.py           # Main backtest script
├── streamlit_app.py        # Streamlit web app
├── build_cleaned_no_outliers.py
├── Cleaned PE Data/         # Input: one CSV per company (if using build script)
├── Cleaned PE Data No Outliers/  # Input for backtest (or output of build script)
├── exclude_companies.txt    # Optional: companies to exclude
└── (outputs) tradelog.csv, equity_curve.csv, equity_curve.png
```

## Data format

Each company CSV should have at least:

- **Date** — trading date  
- **Quarter** — e.g. `2005-Q1` (or derived from Date)  
- **Close** — closing price  
- **PE** — price-to-earnings (invalid/outlier PE are handled in code)  

## Pushing to GitHub

1. **Create a new repository** on [GitHub](https://github.com/new). Do not add a README or .gitignore (this repo already has them).

2. **Initialize and push from your machine** (in the project folder):
   ```bash
   git init
   git add .
   git commit -m "Initial commit: PE discount backtest"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   git push -u origin main
   ```
   If the data folders are large, add them to `.gitignore` (uncomment the lines under "Optional" in `.gitignore`) before `git add .`, or use [Git LFS](https://git-lfs.github.com/) for data.

## License

Use and modify as you like. No warranty.
