"""
PE Discount-to-Median Backtest — Streamlit app.
Run: streamlit run streamlit_app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Ensure we can import pe_backtest (repo root = script dir)
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pe_backtest import load_all_data, run_backtest, metrics

st.set_page_config(page_title="PE Discount Backtest", page_icon="📈", layout="wide")

st.title("PE Discount-to-Median Backtest")
st.markdown(
    "Buy stocks when **PE ≤ (1 − discount) × median PE** (over last L quarters); hold H quarters; sell. "
    "Optional **Option A**: rebalance only at end of Q4, hold 3Q, cash in Q4."
)

# Sidebar inputs
st.sidebar.header("Strategy parameters")
lookback = st.sidebar.slider("Lookback (quarters)", min_value=1, max_value=20, value=3)
discount = st.sidebar.slider("Discount to median PE", min_value=0.10, max_value=0.90, value=0.20, step=0.05)
fixed_q4 = st.sidebar.checkbox("Option A (fixed Q4)", value=False)
if fixed_q4:
    holding = 3
    st.sidebar.caption("Holding fixed at 3 quarters for Option A.")
else:
    holding = st.sidebar.slider("Holding (quarters)", min_value=1, max_value=5, value=2)

run = st.sidebar.button("Run backtest")

if run:
    with st.spinner("Loading data..."):
        data = load_all_data(progress=False)
    if not data:
        st.error("No data found. Ensure 'Cleaned PE Data No Outliers' exists with company CSVs.")
        st.stop()
    st.sidebar.success(f"Loaded {len(data)} companies.")

    with st.spinner("Running backtest..."):
        tradelog, equity, benchmark_series, _, _ = run_backtest(
            data, lookback, discount, holding,
            benchmark_path="", progress_callback=None, fixed_q4=fixed_q4
        )

    if equity is None or len(equity) < 2:
        st.warning("No results: no companies passed the screen for this configuration. Try a lower discount or shorter lookback.")
        st.stop()

    m = metrics(equity, benchmark_series)
    num_trades = len(tradelog) if tradelog is not None else 0

    # Equity curve (log scale, strategy + NIFTY)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(equity.index, equity.values, label="Strategy", color="steelblue", linewidth=1.2)
    if benchmark_series is not None and len(benchmark_series) > 0:
        bm_cum = (1 + benchmark_series).cumprod()
        bm_cum = bm_cum / bm_cum.iloc[0]
        ax.plot(bm_cum.index, bm_cum.values, label="NIFTY 50", color="gray", linewidth=1.2, alpha=0.9)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative return (normalized, start = 1)")
    ax.set_yscale("log")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3, which="both")
    title = f"PE discount backtest (lookback={lookback}Q, discount={discount:.0%}, hold={holding}Q)"
    if fixed_q4:
        title += " [Option A: fixed Q4]"
    ax.set_title(title)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Metrics
    st.subheader("Results")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Number of trades", num_trades)
    col2.metric("CAGR", f"{m.get('CAGR', 0):.2%}")
    col3.metric("Sharpe (ann.)", f"{m.get('Sharpe', 0):.2f}")
    col4.metric("Calmar", f"{m.get('Calmar', 0):.2f}")
    col5.metric("Max drawdown", f"{m.get('MaxDD', 0):.2%}")
    if "Benchmark_CAGR" in m:
        st.caption(f"Benchmark (NIFTY 50): CAGR {m['Benchmark_CAGR']:.2%}  |  Sharpe {m.get('Benchmark_Sharpe', 0):.2f}")

    # Tradelog (last 100 rows)
    if tradelog is not None and len(tradelog) > 0:
        with st.expander("Tradelog (last 100 rows)"):
            st.dataframe(tradelog.tail(100), use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown("[View strategy log](https://github.com/pranami2510-sudo/PE-Backtest/blob/main/strategy_log.md)")
