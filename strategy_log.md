# PE Discount-to-Median Backtest — Strategy Log

## 1. Strategy in one sentence

At fixed rebalance dates, buy only stocks whose **current PE is at a discount** to their own **rolling median PE** (over the last L quarters); hold each cohort for **H quarters**, then sell; repeat. No overlapping cohorts.

---

## 2. Inputs

| Input       | Meaning | Example |
|------------|---------|---------|
| **Lookback (L)** | Number of quarters used to compute each stock’s median PE | 3, 6 |
| **Discount**     | Minimum discount to median PE (buy if PE ≤ (1 − discount) × median) | 20% → PE ≤ 80% of median |
| **Holding (H)**  | How many quarters to hold each cohort before selling | 1, 2, 3, 4, 5 |
| **Option A (fixed Q4)** | If used: rebalance only at end of Q4, hold 3 quarters (Q1–Q3), cash in Q4 | Flag `--fixed-q4` |

---

## 3. When we rebalance and when we sell

### 3.1 Standard (rolling) mode

- **First rebalance:** The first quarter for which we have **L quarters of history** (so we can compute median PE over L quarters).
- **Next rebalances:** Every **H quarters** after that (e.g. if H = 2, we rebalance at quarters 2, 4, 6, …).
- **Sell:** At the **last trading day of the quarter** that is **H quarters after** the rebalance quarter.

```
Quarter index:    0    1    2    3    4    5    6    7   ...
                  |    |    |    |    |    |    |    |
Rebalance (L=3):           ●         ●         ●         ...
                           ^         ^         ^
                        Buy end Q2  Buy end Q4  Buy end Q6
                        Sell end Q2+H          Sell end Q4+H
```

**Formula:** Number of rebalance quarters = floor((N − L + 1) / H), where N = total quarters in data.

### 3.2 Option A (fixed Q4)

- **Rebalance:** Only at **end of Q4** (e.g. 2005-Q4, 2006-Q4, …).
- **Hold:** 3 quarters (Q1, Q2, Q3 of the next year).
- **Sell:** End of Q3.
- **Cash:** Entire Q4 until the next rebalance at end of Q4.

```
Year 1          Year 2          Year 3
Q1  Q2  Q3  Q4 | Q1  Q2  Q3  Q4 | Q1  Q2  Q3  Q4
               |                |
    [=== hold 3Q ===]           [=== hold 3Q ===]
               ^                         ^
            Sell (end Q3)              Sell (end Q3)
[ cash Q4 ]         [ cash Q4 ]
      ^                   ^
   Buy (rebal)        Buy (rebal)
   end Q4             end Q4
```

---

## 4. At each rebalance: screening and execution

```
┌─────────────────────────────────────────────────────────────────┐
│  FOR EACH REBALANCE QUARTER Q                                    │
├─────────────────────────────────────────────────────────────────┤
│  1. MEDIAN PE (per stock)                                        │
│     • Take last L quarters ending at Q.                          │
│     • Collect all valid daily PE values in that window.          │
│     • Median PE = median of those values.                        │
│     • Require ≥ 30 valid PE days.                                │
│                                                                  │
│  2. SCREEN                                                       │
│     • For each stock: PE on last valid day of Q  (and Close).    │
│     • Keep only if:  PE ≤ (1 − discount) × median_PE             │
│       (and PE > 0, median_PE > 0).                              │
│                                                                  │
│  3. BUY                                                          │
│     • Equal weight across all names that passed.                 │
│     • Entry: each stock’s last day in Q with valid PE & Close.   │
│     • Price: Close on that day.                                  │
│                                                                  │
│  4. HOLD                                                         │
│     • Hold for H quarters (or 3 in Option A).                   │
│                                                                  │
│  5. SELL                                                         │
│     • Exit: last trading day of quarter Q + H (or end Q3 in A).  │
│     • Price: Close on that day (or last available if delisted).  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. Equity curve and chaining

- Each **cohort** (one rebalance’s buys) is tracked day-by-day: portfolio value starts at 1.0 for that cohort, then evolves with prices until exit.
- **Chaining:** The next cohort’s daily value series is **rescaled** so its first value equals the **previous cohort’s last value**. Result: one continuous equity curve from start to end of backtest.
- If **no names pass** the screen: stay in **cash** (flat value) until the next rebalance.

```
Cohort 1:  [==== value series 1 ====]  end value V1
Cohort 2:       [==== value series 2 ====]  end value V2
               ↑
               scaled so cohort 2 starts at V1

Final curve: 1.0 —— … —— V1 —— … —— V2 —— … (one continuous line)
```

---

## 6. Data and outlier handling

- **Source:** `Cleaned PE Data No Outliers` (one CSV per company: Date, Quarter, Close, PE).
- **PE:** PE ≤ 0 or PE > 200 → set to NaN (excluded from median and screen).
- **Price:** Daily returns winsorized to ±20%; Close is reconstructed from these returns so extreme single-day moves don’t distort the backtest.
- **Exclude list:** If `exclude_companies.txt` exists (one company name per line), those names are skipped when loading.

---

## 7. Outputs

| Output | Description |
|--------|-------------|
| **tradelog.csv** | Every buy and sell (date, company, action, price, weight). |
| **equity_curve.csv** | Daily normalized portfolio value. |
| **equity_curve.png** | Chart: strategy vs NIFTY 50 (same axis, log scale, both start at 1). |
| **Metrics** | CAGR, Sharpe (ann.), Calmar, Max DD; optional NIFTY 50 CAGR & Sharpe. |

---

## 8. Flow diagram (high level)

```
                    ┌──────────────────┐
                    │  Load company    │
                    │  CSVs (No       │
                    │  Outliers)      │
                    └────────┬────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │  List rebalance  │
                    │  quarters (every │
                    │  H quarters or   │
                    │  Q4 if Option A) │
                    └────────┬────────┘
                             │
         ┌───────────────────┴───────────────────┐
         ▼                                       ▼
  ┌──────────────┐                        ┌──────────────┐
  │ For each     │                        │ Screen:      │
  │ rebal quarter│                        │ PE ≤ (1-d)*  │
  │              │                        │ median_PE    │
  └──────┬───────┘                        └──────┬───────┘
         │                                      │
         │         ┌────────────────────────────┘
         ▼         ▼
  ┌──────────────┐
  │ Buy passers   │
  │ (equal weight│
  │  at last day │
  │  of quarter) │
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐     ┌──────────────┐
  │ Hold H       │────▶│ Sell at end  │
  │ quarters     │     │ of Q+H       │
  └──────────────┘     └──────┬───────┘
                              │
                              ▼
                      ┌──────────────┐
                      │ Chain value  │
                      │ → equity     │
                      │   curve     │
                      └──────────────┘
```

---

## 9. Example timeline (L=3, discount=20%, H=2)

```
Quarters:    2005-Q3    Q4  |  2006-Q1    Q2    Q3    Q4  |  2007-Q1  ...
             ─────────────────────────────────────────────────────────
Lookback:    [==== 3 quarters for median PE ====]
                                    ^
                        Rebalance at 2006-Q2
                        • Median PE = median(daily PE in 2005-Q4, 2006-Q1, 2006-Q2)
                        • Screen: PE (last day 2006-Q2) ≤ 80% of that median
                        • BUY at last valid PE day of 2006-Q2

Holding:                                [== H=2 quarters ==]
                                                          ^
                                              SELL at last day of 2006-Q4

Next rebalance:                                                          ^
                                                                  2007-Q2 (2006-Q2 + 2)
```

---

*This log describes the strategy implemented in `pe_backtest.py`. Data is built from `Cleaned PE Data` using `build_cleaned_no_outliers.py`.*
