"""
PE discount-to-median backtest.
Inputs: lookback quarters, discount to median PE, holding period (quarters).
Outputs: tradelog CSV, equity curve plot, CAGR, Sharpe, Calmar, num trades, NIFTY 50 benchmark (Yahoo Finance).
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

_SCRIPT_DIR = Path(__file__).resolve().parent
CLEANED_DIR = _SCRIPT_DIR / "Cleaned PE Data No Outliers"
MIN_VALID_PE_DAYS = 30
RF_ANNUAL = 0.067  # 6.7%
NIFTY50_YAHOO_TICKER = "^NSEI"  # NIFTY 50 on Yahoo Finance
PE_CAP = 200  # Outlier handling: ignore PE > 200 (and PE <= 0) for screening/median
MAX_DAILY_RETURN = 0.20  # Price outlier: winsorize daily returns to ±20%, reconstruct Close


def fetch_nifty50_yahoo(start_date, end_date, equity_index):
    """
    Fetch NIFTY 50 (^NSEI) from Yahoo Finance and return daily return series aligned to equity_index.
    Returns None on failure.
    """
    try:
        import yfinance as yf
        start = pd.Timestamp(start_date).strftime("%Y-%m-%d")
        end = pd.Timestamp(end_date).strftime("%Y-%m-%d")
        data = yf.download(NIFTY50_YAHOO_TICKER, start=start, end=end, progress=False, auto_adjust=True)
        if data.empty or len(data) < 2:
            return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        close = data["Close"].copy()
        close.index = pd.to_datetime(close.index).tz_localize(None)
        close = close[~close.index.duplicated(keep="first")]
        close = close.reindex(equity_index).ffill().bfill()
        return close.pct_change().dropna()
    except Exception as e:
        print(f"Yahoo Finance benchmark fetch failed: {e}")
        return None


def _load_exclude_companies():
    """If exclude_companies.txt exists (one name per line), return set of names to skip."""
    excl_path = CLEANED_DIR.parent / "exclude_companies.txt"
    if not excl_path.exists():
        return set()
    names = set()
    for line in excl_path.read_text(encoding="utf-8").splitlines():
        name = line.strip()
        if name and not name.startswith("#"):
            names.add(name)
    return names


def load_all_data(progress=True):
    """Load all company CSVs into a dict: company_name -> DataFrame with Date, Quarter, Close, PE."""
    exclude = _load_exclude_companies()
    paths = sorted(CLEANED_DIR.glob("*.csv"))
    data = {}
    for i, p in enumerate(paths):
        if p.stem in exclude:
            continue
        try:
            df = pd.read_csv(p)
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.dropna(subset=["Date"])
            df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
            df["PE"] = pd.to_numeric(df["PE"], errors="coerce")
            # Outlier handling: treat PE <= 0 or PE > PE_CAP as invalid (set NaN)
            df.loc[(df["PE"] <= 0) | (df["PE"] > PE_CAP), "PE"] = np.nan
            # Price outlier: winsorize daily returns to ±20% and reconstruct Close
            ret = df["Close"].pct_change(fill_method=None)
            ret_clipped = ret.clip(lower=-MAX_DAILY_RETURN, upper=MAX_DAILY_RETURN)
            scale = (1 + ret_clipped).fillna(1).cumprod()
            df["Close"] = df["Close"].iloc[0] * scale
            if "Quarter" not in df.columns:
                df["Quarter"] = df["Date"].dt.year.astype(str) + "-Q" + df["Date"].dt.quarter.astype(str)
            data[p.stem] = df[["Date", "Quarter", "Close", "PE"]].copy()
        except Exception as e:
            if progress:
                print(f"  Skip {p.name}: {e}")
        if progress and (i + 1) % 100 == 0:
            print(f"  Loading companies... {i + 1}/{len(paths)}")
    if progress and paths and len(data) % 100 != 0:
        print(f"  Loading companies... {len(data)}/{len(paths)}")
    return data


def get_quarter_order(quarter_str):
    """Map '2005-Q1' -> (2005, 1) for sorting."""
    try:
        y, q = quarter_str.strip().split("-Q")
        return (int(y), int(q))
    except Exception:
        return (0, 0)


def get_all_quarters_sorted(data):
    """Unique quarters across all companies, sorted."""
    quarters = set()
    for df in data.values():
        quarters.update(df["Quarter"].dropna().unique().tolist())
    return sorted(quarters, key=get_quarter_order)


def last_day_valid_pe_in_quarter(df, year, q):
    """Last date in (year, q) with valid PE > 0 and valid Close. Returns (date, pe, close) or (None, None, None)."""
    mask = (df["Date"].dt.year == year) & (df["Date"].dt.quarter == q)
    sub = df.loc[mask].dropna(subset=["PE", "Close"])
    sub = sub[sub["PE"] > 0]
    if sub.empty:
        return None, None, None
    last_row = sub.iloc[-1]
    return last_row["Date"], last_row["PE"], last_row["Close"]


def last_trading_day_quarter(df, year, q):
    """Last date in (year, q) with valid Close. Returns date or None."""
    mask = (df["Date"].dt.year == year) & (df["Date"].dt.quarter == q)
    sub = df.loc[mask].dropna(subset=["Close"])
    if sub.empty:
        return None
    return sub["Date"].max()


def median_pe_over_lookback(df, end_year, end_q, L):
    """Median of valid daily PE over L quarters ending at (end_year, end_q). Returns (median, n_valid_days)."""
    # L quarters: (end_year, end_q) and (L-1) quarters before
    # Quarter before (y,q) is q-1 if q>1 else (y-1, 4)
    window_dates = []
    y, q = end_year, end_q
    for _ in range(L):
        window_dates.append((y, q))
        if q == 1:
            y, q = y - 1, 4
        else:
            q = q - 1
    # Collect all valid PE in those quarters
    pe_list = []
    for (yy, qq) in window_dates:
        mask = (df["Date"].dt.year == yy) & (df["Date"].dt.quarter == qq)
        pes = df.loc[mask, "PE"].dropna()
        pes = pes[pes > 0]
        pe_list.extend(pes.tolist())
    if len(pe_list) < MIN_VALID_PE_DAYS:
        return None, len(pe_list)
    return np.median(pe_list), len(pe_list)


def screen_cohort(data, quarter_str, L, discount, all_quarters_sorted):
    """
    At quarter_str (e.g. '2005-Q1'), screen companies: PE at last valid day in quarter <= (1-discount)*median_PE.
    median_PE over L quarters ending at quarter_str. Require >= 30 valid PE days, PE > 0, median_PE > 0.
    Returns list of (company, entry_date, entry_price, weight) with equal weight.
    """
    y, q = int(quarter_str.split("-Q")[0]), int(quarter_str.split("-Q")[1])
    idx = all_quarters_sorted.index(quarter_str) if quarter_str in all_quarters_sorted else -1
    if idx < L - 1:
        return []  # Not enough history
    passed = []
    for name, df in data.items():
        entry_date, pe_screen, close_screen = last_day_valid_pe_in_quarter(df, y, q)
        if entry_date is None or pe_screen is None or close_screen is None or pe_screen <= 0:
            continue
        median_pe, n_days = median_pe_over_lookback(df, y, q, L)
        if median_pe is None or median_pe <= 0 or n_days < MIN_VALID_PE_DAYS:
            continue
        if pe_screen > (1 - discount) * median_pe:
            continue
        passed.append((name, entry_date, close_screen, pe_screen, median_pe))
    if not passed:
        return []
    n = len(passed)
    return [(name, entry_date, entry_price, 1.0 / n) for name, entry_date, entry_price, _, _ in passed]


def exit_dates_prices(data, cohort, exit_year, exit_q):
    """Cohort is list of (name, entry_date, entry_price, weight). Get exit date and price per name."""
    exits = []
    for name, entry_date, entry_price, weight in cohort:
        df = data[name]
        last_dt = last_trading_day_quarter(df, exit_year, exit_q)
        if last_dt is None:
            # Delisted: last available close before or in exit quarter
            end_of_exit_q = pd.Timestamp(exit_year, min(3 * exit_q, 12), 28)
            sub = df[(df["Date"] <= end_of_exit_q)].dropna(subset=["Close"])
            if sub.empty:
                exit_price = entry_price  # no data, assume no change
                exit_dt = entry_date
            else:
                last_row = sub.iloc[-1]
                exit_dt = last_row["Date"]
                exit_price = last_row["Close"]
        else:
            row = df[(df["Date"] == last_dt) & df["Close"].notna()]
            exit_price = row["Close"].iloc[0] if len(row) > 0 else entry_price
            exit_dt = last_dt
        exits.append((name, exit_dt, exit_price, weight))
    return exits


def quarter_plus_h(quarter_str, H):
    """Add H quarters to '2005-Q1' -> '2006-Q1' etc."""
    y, q = int(quarter_str.split("-Q")[0]), int(quarter_str.split("-Q")[1])
    q += H
    while q > 4:
        q -= 4
        y += 1
    while q < 1:
        q += 4
        y -= 1
    return f"{y}-Q{q}"


def build_daily_values(data, cohort_entries, cohort_exits, all_trading_dates):
    """
    Cohort: staggered entries (name, entry_date, entry_price, weight).
    Exits: (name, exit_date, exit_price, weight).
    Return series: date -> portfolio value (normalized), using only Close for each name.
    """
    entry_dates = [e[1] for e in cohort_entries]
    exit_dates = [e[1] for e in cohort_exits]
    start_d = min(entry_dates)
    end_d = max(exit_dates)
    names = [e[0] for e in cohort_entries]
    weights = {e[0]: e[3] for e in cohort_entries}
    entry_prices = {e[0]: e[2] for e in cohort_entries}
    entry_dates_d = {e[0]: e[1] for e in cohort_entries}
    exit_dates_d = {e[0]: e[1] for e in cohort_exits}

    # Pre-build date->close per name for the date range (faster than repeated df lookups)
    close_by_name_date = {}
    for name in names:
        df = data[name]
        sub = df[(df["Date"] >= start_d) & (df["Date"] <= end_d) & df["Close"].notna()]
        close_by_name_date[name] = dict(zip(sub["Date"].tolist(), sub["Close"].tolist()))

    dates_sorted = sorted(all_trading_dates)
    dates_in_range = [d for d in dates_sorted if start_d <= d <= end_d]
    if not dates_in_range:
        return pd.Series(dtype=float)

    # Equal weight: not-yet-entered capital stays in cash (contributes weight*1). So portfolio value starts at 1.0.
    value_series = []
    for d in dates_in_range:
        v = 0.0
        for name in names:
            ed = entry_dates_d[name]
            xd = exit_dates_d[name]
            if d < ed:
                v += weights[name] * 1.0  # not yet entered, cash
            elif d > xd:
                pass  # already exited
            else:
                close_today = close_by_name_date[name].get(d)
                if close_today is None or (isinstance(close_today, float) and np.isnan(close_today)):
                    close_today = entry_prices[name]
                v += weights[name] * (float(close_today) / entry_prices[name])
        value_series.append((d, v))
    return pd.Series({d: v for d, v in value_series})


def _last_trading_day_of_quarter(all_trading_dates, year, q):
    """Last trading date in (year, q) from all_trading_dates."""
    dates_in_q = [d for d in all_trading_dates if d.year == year and (d.month - 1) // 3 + 1 == q]
    return max(dates_in_q) if dates_in_q else None


def run_backtest(data, L, discount, H, benchmark_path=None, progress_callback=None, fixed_q4=False):
    all_quarters = get_all_quarters_sorted(data)
    if fixed_q4:
        # Option A: rebalance only at end of Q4; hold 3 quarters (Q1, Q2, Q3); cash in Q4
        rebal_quarters = [q for i, q in enumerate(all_quarters) if q.endswith("-Q4") and i >= L - 1]
        H_eff = 3
    else:
        # Original: first possible at index L-1, then every H quarters
        first_rebal_idx = L - 1
        if first_rebal_idx >= len(all_quarters):
            return None, None, None, None, None
        rebal_quarters = []
        i = first_rebal_idx
        while i < len(all_quarters):
            rebal_quarters.append(all_quarters[i])
            i += H
        H_eff = H

    if not rebal_quarters:
        return None, None, None, None, None

    total_quarters = len(rebal_quarters)

    # Collect all trading dates from data for building daily series
    all_dates = set()
    for df in data.values():
        all_dates.update(df["Date"].dropna().tolist())
    all_trading_dates = sorted(all_dates)

    tradelog_rows = []
    daily_values_list = []
    prev_value = 1.0

    for step, rebal_q in enumerate(rebal_quarters, start=1):
        cohort = screen_cohort(data, rebal_q, L, discount, all_quarters)
        if progress_callback:
            progress_callback(step, total_quarters, rebal_q, len(cohort))
        y, q = int(rebal_q.split("-Q")[0]), int(rebal_q.split("-Q")[1])
        exit_q_str = quarter_plus_h(rebal_q, H_eff)
        exit_y, exit_q = int(exit_q_str.split("-Q")[0]), int(exit_q_str.split("-Q")[1])

        if not cohort:
            # Stay in cash: flat return over the period
            if fixed_q4 and step < total_quarters:
                # Cash from this rebal quarter through end of next rebal quarter (next Q4)
                next_rebal = rebal_quarters[step]  # next Q4
                ny, nq = int(next_rebal.split("-Q")[0]), int(next_rebal.split("-Q")[1])
                last_next = _last_trading_day_of_quarter(all_trading_dates, ny, nq)
                start_d = pd.Timestamp(y, min(3 * q, 12), 1)
                end_d = last_next if last_next else pd.Timestamp(exit_y, min(3 * exit_q, 12), 28)
                dates_cash = [d for d in all_trading_dates if start_d <= d <= end_d]
            else:
                start_d = pd.Timestamp(y, min(3 * q, 12), 1)
                end_d = pd.Timestamp(exit_y, min(3 * exit_q, 12), 28)
                dates_cash = [d for d in all_trading_dates if start_d <= d <= end_d]
            if dates_cash:
                for d in dates_cash:
                    daily_values_list.append((d, prev_value))
            continue

        exits = exit_dates_prices(data, cohort, exit_y, exit_q)
        # Tradelog: buys (staggered) and sells
        for name, entry_date, entry_price, weight in cohort:
            tradelog_rows.append({"date": entry_date, "company": name, "action": "buy", "price": entry_price, "weight": weight})
        for name, exit_date, exit_price, weight in exits:
            tradelog_rows.append({"date": exit_date, "company": name, "action": "sell", "price": exit_price, "weight": weight})

        daily_val = build_daily_values(data, cohort, exits, all_trading_dates)
        if daily_val.empty:
            continue
        # Chain: scale this cohort's value so it starts at prev_value
        daily_val = daily_val.sort_index()
        first_val = daily_val.iloc[0]
        if first_val and first_val > 0:
            scale = prev_value / first_val
            daily_val = daily_val * scale
        for d, v in daily_val.items():
            daily_values_list.append((d, v))
        if len(daily_values_list) > 0:
            prev_value = daily_values_list[-1][1]

        # Option A: cash gap from day after exit quarter through day before next rebalance (Q4)
        if fixed_q4 and step < total_quarters:
            exit_last = _last_trading_day_of_quarter(all_trading_dates, exit_y, exit_q)
            next_rebal = rebal_quarters[step]
            ny, nq = int(next_rebal.split("-Q")[0]), int(next_rebal.split("-Q")[1])
            last_next = _last_trading_day_of_quarter(all_trading_dates, ny, nq)
            if exit_last is not None and last_next is not None:
                cash_dates = [d for d in all_trading_dates if exit_last < d < last_next]
                for d in cash_dates:
                    daily_values_list.append((d, prev_value))
                prev_value = daily_values_list[-1][1] if daily_values_list else prev_value

    if not daily_values_list:
        return None, None, None, None, None

    tradelog = pd.DataFrame(tradelog_rows)
    daily_df = pd.DataFrame(daily_values_list, columns=["Date", "Value"]).drop_duplicates(subset=["Date"], keep="last").sort_values("Date")
    daily_df = daily_df.set_index("Date")
    # Forward-fill missing days so we have a continuous series for returns
    full_idx = pd.date_range(daily_df.index.min(), daily_df.index.max(), freq="B")
    equity = daily_df.reindex(full_idx).ffill().bfill()
    equity = equity.squeeze()

    # Benchmark from CSV if provided
    benchmark_series = None
    if benchmark_path and Path(benchmark_path).exists():
        try:
            bm = pd.read_csv(benchmark_path)
            bm["Date"] = pd.to_datetime(bm["Date"], errors="coerce")
            bm = bm.dropna(subset=["Date"])
            close_col = "Close" if "Close" in bm.columns else bm.columns[1]
            bm = bm.set_index("Date")[close_col]
            bm = bm[bm.index >= equity.index.min()]
            bm = bm[bm.index <= equity.index.max()]
            if len(bm) > 1:
                bm = bm.reindex(equity.index).ffill().bfill()
                benchmark_series = bm.pct_change().dropna()
        except Exception as e:
            print(f"Benchmark CSV load failed: {e}")

    return tradelog, equity, benchmark_series, daily_values_list, prev_value


def metrics(equity, benchmark_series=None):
    """CAGR, Sharpe (daily ann.), Calmar, num trades from tradelog."""
    if equity is None or len(equity) < 2:
        return {}
    ret = equity.pct_change().dropna()
    if ret.empty:
        return {}
    years = (equity.index.max() - equity.index.min()).days / 365.25
    if years <= 0:
        return {}
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1
    rf_daily = (1 + RF_ANNUAL) ** (1 / 252) - 1
    sharpe = (ret.mean() - rf_daily) / ret.std() * np.sqrt(252) if ret.std() > 0 else np.nan
    cum = (1 + ret).cumprod()
    roll_max = cum.cummax()
    dd = (cum - roll_max) / roll_max
    max_dd = dd.min()
    calmar = cagr / (-max_dd) if max_dd and max_dd != 0 else np.nan
    out = {"CAGR": cagr, "Sharpe": sharpe, "Calmar": calmar, "MaxDD": max_dd}
    if benchmark_series is not None and len(benchmark_series) > 0:
        bm_cum = (1 + benchmark_series).cumprod()
        bm_start, bm_end = bm_cum.iloc[0], bm_cum.iloc[-1]
        bm_years = (benchmark_series.index.max() - benchmark_series.index.min()).days / 365.25
        out["Benchmark_CAGR"] = (bm_end / bm_start) ** (1 / bm_years) - 1 if bm_years > 0 else np.nan
        out["Benchmark_Sharpe"] = (benchmark_series.mean() - rf_daily) / benchmark_series.std() * np.sqrt(252) if benchmark_series.std() > 0 else np.nan
    return out


def _validate_lookback(L):
    """Return int in [1, 20] or None."""
    try:
        L = int(L)
        if 1 <= L <= 20:
            return L
    except (TypeError, ValueError):
        pass
    return None


def _validate_discount(d):
    """Return float in [0.1, 0.9] and multiple of 0.05, or None."""
    try:
        d = float(d)
        if d < 0.1 or d > 0.9:
            return None
        r = round(d / 0.05) * 0.05
        if abs(r - d) > 1e-9:
            return None
        return round(r, 2)
    except (TypeError, ValueError):
        return None


def _validate_holding(H):
    """Return int in [1, 5] or None."""
    try:
        H = int(H)
        if 1 <= H <= 5:
            return H
    except (TypeError, ValueError):
        pass
    return None


def main():
    parser = argparse.ArgumentParser(description="PE discount-to-median backtest")
    parser.add_argument("--benchmark", type=str, default="", help="Path to NIFTY 50 CSV (Date, Close)")
    parser.add_argument("--out-dir", type=str, default=None, help="Directory for tradelog and equity curve (default: PE Backtest folder)")
    parser.add_argument("--fixed-q4", action="store_true", help="Option A: rebalance only at end of Q4, hold 3Q (Q1-Q3), cash in Q4")
    parser.add_argument("--lookback", type=str, default=None, help="Lookback quarters (1-20); if set with --discount and --holding, run non-interactive")
    parser.add_argument("--discount", type=str, default=None, help="Discount to median PE (e.g. 0.2)")
    parser.add_argument("--holding", type=str, default=None, help="Holding period in quarters (1-5); ignored if --fixed-q4")
    args = parser.parse_args()

    fixed_q4 = getattr(args, "fixed_q4", False)

    print("\n=== INPUTS ===")
    L = _validate_lookback(args.lookback) if args.lookback else None
    discount = _validate_discount(args.discount) if args.discount else None
    H = _validate_holding(args.holding) if args.holding else None
    if fixed_q4:
        H = 3  # Option A uses 3-quarter hold; display only
    if L is None or discount is None or (H is None and not fixed_q4):
        print("Enter backtest parameters:")
    while L is None:
        raw = input("  Lookback period in quarters (1-20): ").strip() if L is None and args.lookback is None else (args.lookback or "")
        if not raw and args.lookback is None:
            raw = input("  Lookback period in quarters (1-20): ").strip()
        L = _validate_lookback(raw)
        if L is None:
            print("  Invalid. Enter an integer between 1 and 20.")
    while discount is None:
        raw = input("  Discount to median PE (0.1 to 0.9, step 0.05): ").strip() if discount is None and args.discount is None else (args.discount or "")
        if not raw and args.discount is None:
            raw = input("  Discount to median PE (0.1 to 0.9, step 0.05): ").strip()
        discount = _validate_discount(raw)
        if discount is None:
            print("  Invalid. Use 0.1, 0.15, 0.2, 0.25, ... 0.9.")
    while H is None:
        raw = input("  Holding period in quarters (1-5): ").strip() if H is None and args.holding is None else (args.holding or "")
        if not raw and args.holding is None:
            raw = input("  Holding period in quarters (1-5): ").strip()
        H = _validate_holding(raw)
        if H is None:
            print("  Invalid. Enter an integer between 1 and 5.")

    if fixed_q4:
        print("\n  Option A (fixed Q4): rebalance end of Q4 only, hold 3Q (Q1-Q3), cash in Q4")
    print("\n  Lookback: {} quarters  |  Discount: {:.0%}  |  Holding: {} quarters".format(L, discount, H))
    if not (args.lookback and args.discount and args.holding):
        print("  Backtest may take a few minutes. Press Enter to start...")
        input()
    else:
        print("  Running backtest (non-interactive)...")

    out_dir = Path(args.out_dir) if args.out_dir else Path(CLEANED_DIR).parent
    benchmark_path = args.benchmark.strip() or str(out_dir / "NIFTY50.csv")

    print("\n=== LOADING DATA ===")
    data = load_all_data(progress=True)
    print("  Loaded {} companies.".format(len(data)))

    print("\n=== RUNNING BACKTEST ===")
    def on_quarter(step, total, rebal_q, n_passed):
        print("  Quarter {}/{}: {} — {} companies passed".format(step, total, rebal_q, n_passed))
    tradelog, equity, benchmark_series, _, _ = run_backtest(
        data, L, discount, H, benchmark_path, progress_callback=on_quarter, fixed_q4=fixed_q4
    )
    if tradelog is None or equity is None:
        print("\n  No results. No companies passed the screen for this configuration.")
        print("  Try a lower discount (e.g. 0.1) or shorter lookback.")
        return

    print("\n=== BENCHMARK ===")
    if benchmark_series is None or len(benchmark_series) == 0:
        print("  Fetching NIFTY 50 from Yahoo Finance (^NSEI)...")
        benchmark_series = fetch_nifty50_yahoo(
            equity.index.min(), equity.index.max(), equity.index
        )
        if benchmark_series is not None and len(benchmark_series) > 0:
            print("  Benchmark loaded.")
        else:
            print("  Benchmark fetch failed (check network or add NIFTY50.csv).")
    else:
        print("  Using benchmark from CSV.")

    # One round-trip (one buy + one sell) = one trade; count sells
    num_trades = int((tradelog["action"] == "sell").sum()) if "action" in tradelog.columns else len(tradelog) // 2
    m = metrics(equity, benchmark_series)

    print("\n=== OUTPUTS ===")
    tradelog_path = out_dir / "tradelog.csv"
    tradelog.to_csv(tradelog_path, index=False)
    print("  Tradelog:       {}".format(tradelog_path))
    equity.to_csv(out_dir / "equity_curve.csv")
    print("  Equity curve:   {}".format(out_dir / "equity_curve.csv"))
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(equity.index, equity.values, label="Strategy", color="steelblue", linewidth=1.2)
        if benchmark_series is not None and len(benchmark_series) > 0:
            bm_cum = (1 + benchmark_series).cumprod()
            bm_cum = bm_cum / bm_cum.iloc[0]  # NIFTY normalized to start at 1
            ax.plot(bm_cum.index, bm_cum.values, label="NIFTY 50", color="gray", linewidth=1.2, alpha=0.9)
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative return (normalized, start = 1)")
        ax.set_yscale("log")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3, which="both")
        title = f"PE discount backtest (lookback={L}Q, discount={discount:.0%}, hold={H}Q)"
        if fixed_q4:
            title += " [Option A: fixed Q4]"
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(out_dir / "equity_curve.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  Plot:           {}".format(out_dir / "equity_curve.png"))
    except Exception as e:
        print("  Plot failed: {}".format(e))

    print("\n=== RESULTS ===")
    print("  {:<22} {:>12}".format("Metric", "Value"))
    print("  " + "-" * 36)
    print("  {:<22} {:>12}".format("Number of trades", num_trades))
    print("  {:<22} {:>11.2%}".format("CAGR", m.get("CAGR", np.nan)))
    print("  {:<22} {:>12.2f}".format("Sharpe ratio (ann.)", m.get("Sharpe", np.nan)))
    print("  {:<22} {:>12.2f}".format("Calmar ratio", m.get("Calmar", np.nan)))
    print("  {:<22} {:>11.2%}".format("Max drawdown", m.get("MaxDD", np.nan)))
    if "Benchmark_CAGR" in m:
        print("  " + "-" * 36)
        print("  {:<22} {:>11.2%}".format("Benchmark (NIFTY 50) CAGR", m["Benchmark_CAGR"]))
        print("  {:<22} {:>12.2f}".format("Benchmark Sharpe", m.get("Benchmark_Sharpe", np.nan)))
    print("")


if __name__ == "__main__":
    main()
