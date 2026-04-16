"""
PE Discount-to-Median Backtest — Streamlit frontend (interactive).
Run: streamlit run streamlit_app.py
"""
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Must be first Streamlit command
st.set_page_config(
    page_title="PE Discount Backtest",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

try:
    from pe_backtest import load_all_data, run_backtest, metrics, fetch_nifty50_yahoo
except Exception as e:
    st.error(f"**Failed to load backtest module:** {e}")
    st.stop()

CLEANED_DIR = _SCRIPT_DIR / "Cleaned PE Data No Outliers"

def _data_available():
    """True if the data folder exists and has at least one CSV."""
    try:
        return CLEANED_DIR.is_dir() and len(list(CLEANED_DIR.glob("*.csv"))) > 0
    except Exception:
        return False

# Shared FCF CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    [data-testid="stMetric"] {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    [data-testid="stMetricValue"] {
        color: #000000 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #000000 !important;
    }
    [data-testid="stMetricDelta"] {
        color: #000000 !important;
    }
</style>
""", unsafe_allow_html=True)

# Session state
if "backtest_results" not in st.session_state:
    st.session_state.backtest_results = None
for key, default in [("lookback", 3), ("discount", 0.2), ("holding", 2)]:
    if key not in st.session_state:
        st.session_state[key] = default

# ---------------------- Sidebar ----------------------
st.sidebar.header("⚙️ Strategy Parameters")

with st.sidebar.expander("What do these mean?", expanded=False):
    st.markdown("""
    - **Lookback:** Quarters of history for median PE.
    - **Discount:** Buy only if PE ≤ (1 − discount) × median PE.
    - **Holding:** Quarters to hold each cohort.
    """)

lookback = st.sidebar.slider("Lookback (quarters)", 1, 20, value=st.session_state.get("lookback", 3), key="sb_lookback")
discount = st.sidebar.slider("Discount to median PE", 0.10, 0.90, value=float(st.session_state.get("discount", 0.2)), step=0.05, key="sb_discount")
holding = st.sidebar.slider("Holding (quarters)", 1, 5, value=st.session_state.get("holding", 2), key="sb_holding")
st.session_state["lookback"] = lookback
st.session_state["discount"] = discount
st.session_state["holding"] = holding

max_pct = (1 - discount) * 100
st.sidebar.markdown(f"**Screen:** PE ≤ **{max_pct:.0f}%** of median PE")
st.sidebar.markdown("---")

# Data status
try:
    _n_csvs = len(list(CLEANED_DIR.glob("*.csv"))) if CLEANED_DIR.is_dir() else 0
except Exception:
    _n_csvs = 0
if _n_csvs > 0:
    st.sidebar.success(f"✓ Data ready ({_n_csvs} companies)")
else:
    st.sidebar.warning("⚠ No data folder — add **Cleaned PE Data No Outliers** with CSVs to run backtests.")

run = st.sidebar.button("▶️ Run backtest", type="primary", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown("[📄 Strategy log (GitHub)](https://github.com/pranami2510-sudo/PE-Backtest/blob/main/strategy_log.md)")

# ---------------------- Main Header ----------------------
st.markdown('<h1 class="main-header">📈 PE Discount-to-Median Backtest</h1>', unsafe_allow_html=True)

if not _data_available():
    st.warning(
        "**No data folder found.** The app needs the **Cleaned PE Data No Outliers** folder with company CSVs. "
        "On Streamlit Community Cloud this folder is often not in the repo (size limits). "
        "To run backtests: clone the repo, add that folder locally, and run `streamlit run streamlit_app.py`, or add a small sample of CSVs to the repo for the web app."
    )

# ---------------------- Run Backtest ----------------------
if run:
    with st.spinner("Loading data..."):
        data = load_all_data(progress=False)
    if not data:
        st.error(
            "No data found. Ensure **Cleaned PE Data No Outliers** exists with company CSVs. "
            "On the web, the repo may not include this folder—run the app locally with your data."
        )
        st.stop()
    n_companies = len(data)
    progress_bar = st.progress(0, text="Running backtest...")
    progress_placeholder = st.empty()

    def progress_cb(step, total, rebal_q, n_passed):
        progress_bar.progress(step / max(total, 1), text=f"Quarter {step}/{total}: {rebal_q} — {n_passed} companies passed")

    with progress_placeholder:
        with st.spinner("Running backtest (typically 1–2 min)..."):
            tradelog, equity, benchmark_series, _, _ = run_backtest(
                data, lookback, discount, holding,
                benchmark_path="", progress_callback=progress_cb, fixed_q4=False
            )
    progress_bar.empty()
    progress_placeholder.empty()

    # Fetch NIFTY 50 from Yahoo if no benchmark CSV was used
    if (benchmark_series is None or len(benchmark_series) == 0) and equity is not None and len(equity) >= 2:
        with st.spinner("Fetching NIFTY 50 benchmark..."):
            benchmark_series = fetch_nifty50_yahoo(
                equity.index.min(), equity.index.max(), equity.index
            )

    if equity is None or len(equity) < 2:
        st.warning("No results: no companies passed the screen. Try a **lower discount** or **shorter lookback**.")
        st.stop()

    m = metrics(equity, benchmark_series)
    st.session_state.backtest_results = {
        "tradelog": tradelog,
        "equity": equity,
        "metrics": m,
        "params": {"lookback": lookback, "discount": discount, "holding": holding},
        "n_companies": n_companies,
        "benchmark_series": benchmark_series,
    }
    st.sidebar.success(f"Loaded {n_companies} companies. Backtest done.")
    st.rerun()

# ---------------------- Display Results ----------------------
if st.session_state.backtest_results is None:
    st.info("👈 Set parameters in the sidebar and click **Run backtest** to see results.")
    st.caption("Suggested starting point: Lookback 3, Discount 20%, Holding 2.")
else:
    res = st.session_state.backtest_results
    equity = res["equity"]
    m = res["metrics"]
    tl = res["tradelog"]
    num_trades = int((tl["action"] == "sell").sum()) if tl is not None and len(tl) > 0 and "action" in tl.columns else 0
    lookback = res["params"]["lookback"]
    discount = res["params"]["discount"]
    holding = res["params"]["holding"]
    benchmark_series = res.get("benchmark_series")

    # ---- Strategy Parameters ----
    st.subheader("⚙️ Strategy Parameters")
    param_col1, param_col2, param_col3 = st.columns(3)
    with param_col1:
        st.metric("Lookback", f"{lookback} Q", help="Quarters of history for median PE")
    with param_col2:
        st.metric("Discount", f"{discount:.0%}", help=f"Buy when PE ≤ {(1-discount)*100:.0f}% of median")
    with param_col3:
        st.metric("Holding", f"{holding} Q", help="Quarters to hold each cohort")

    # ---- Performance Metrics (8 metrics, 2 rows of 4) ----
    st.subheader("📊 Performance Metrics")

    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        st.metric("CAGR", f"{m.get('CAGR', 0):.2%}", help="Compound Annual Growth Rate")
    with mc2:
        st.metric("Sharpe Ratio", f"{m.get('Sharpe', 0):.2f}", help="Annualized risk-adjusted return")
    with mc3:
        st.metric("Calmar Ratio", f"{m.get('Calmar', 0):.2f}", help="CAGR / Max Drawdown")
    with mc4:
        st.metric("Max Drawdown", f"{m.get('MaxDD', 0):.2%}", help="Largest peak-to-trough decline")

    mc5, mc6, mc7, mc8 = st.columns(4)
    with mc5:
        st.metric("Total Trades", f"{num_trades:,}", help="Number of round-trip trades")
    with mc6:
        # Win ratio: count trades where sell price > buy price
        win_ratio = "N/A"
        if tl is not None and len(tl) > 0 and "action" in tl.columns and "price" in tl.columns:
            buys = tl[tl["action"] == "buy"].reset_index(drop=True)
            sells = tl[tl["action"] == "sell"].reset_index(drop=True)
            if len(buys) > 0 and len(sells) > 0:
                # Match by company
                wins = 0
                total = 0
                for company in buys["company"].unique():
                    cb = buys[buys["company"] == company]["price"].values
                    cs = sells[sells["company"] == company]["price"].values
                    pairs = min(len(cb), len(cs))
                    if pairs > 0:
                        wins += int((cs[:pairs] > cb[:pairs]).sum())
                        total += pairs
                if total > 0:
                    win_ratio = f"{wins/total:.2%}"
        st.metric("Win Ratio", win_ratio, help="Percentage of profitable trades")
    with mc7:
        st.metric("Initial Capital", "₹1.00", help="Normalized starting value")
    with mc8:
        final_val = equity.iloc[-1] if len(equity) > 0 else 0
        delta_val = f"{final_val - 1:.2f}" if final_val != 0 else None
        st.metric("Final Value", f"₹{final_val:.2f}", delta=delta_val, help="Final normalized portfolio value")

    # ---- Download Buttons ----
    st.subheader("📂 Download Strategy Files")
    p = res["params"]
    run_label = f"L{p['lookback']}_d{int(p['discount']*100)}_H{p['holding']}"

    dl_col1, dl_col2, dl_col3 = st.columns(3)
    with dl_col1:
        equity_df = equity.to_frame(name="Value")
        equity_df.index.name = "Date"
        st.download_button("📥 Download Equity Curve (CSV)", data=equity_df.to_csv(),
                          file_name=f"equity_curve_{run_label}.csv", mime="text/csv",
                          key="dl_equity", use_container_width=True)
    with dl_col2:
        if tl is not None and len(tl) > 0:
            st.download_button("📥 Download Tradelog (CSV)", data=tl.to_csv(index=False),
                              file_name=f"tradelog_{run_label}.csv", mime="text/csv",
                              key="dl_tradelog", use_container_width=True)
        else:
            st.info("No tradelog for this run.")
    with dl_col3:
        summary_data = {
            'Metric': ['CAGR', 'Sharpe', 'Calmar', 'Max Drawdown', 'Trades'],
            'Value': [f"{m.get('CAGR',0):.2%}", f"{m.get('Sharpe',0):.2f}",
                     f"{m.get('Calmar',0):.2f}", f"{m.get('MaxDD',0):.2%}", str(num_trades)]
        }
        summary_csv = pd.DataFrame(summary_data).to_csv(index=False)
        st.download_button("📥 Download Summary (CSV)", data=summary_csv,
                          file_name=f"summary_{run_label}.csv", mime="text/csv",
                          key="dl_summary", use_container_width=True)

    # ---- Equity Curve (Plotly) ----
    st.subheader("📉 Equity Curve")

    # Optional date range zoom
    chart_start = equity.index.min().date()
    chart_end = equity.index.max().date()
    with st.expander("🔍 Zoom chart to date range", expanded=False):
        dr_col1, dr_col2 = st.columns(2)
        with dr_col1:
            chart_start = st.date_input("Start date", value=chart_start, key="chart_start")
        with dr_col2:
            chart_end = st.date_input("End date", value=chart_end, key="chart_end")
    equity_sub = equity[(equity.index.date >= chart_start) & (equity.index.date <= chart_end)]
    if len(equity_sub) < 2:
        equity_sub = equity

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=equity_sub.index, y=equity_sub.values, name="Strategy",
        line=dict(color="#1f77b4", width=2.5), mode="lines"
    ))

    bm_plotted = False
    if benchmark_series is not None and len(benchmark_series) > 0:
        bm_cum = (1 + benchmark_series).cumprod()
        bm_cum = bm_cum / bm_cum.iloc[0]
        bm_sub = bm_cum[(bm_cum.index.date >= chart_start) & (bm_cum.index.date <= chart_end)]
        if len(bm_sub) < 2:
            bm_sub = bm_cum.reindex(equity_sub.index).ffill().bfill().dropna()
        if len(bm_sub) > 0:
            fig.add_trace(go.Scatter(
                x=bm_sub.index, y=bm_sub.values, name="NIFTY 50",
                line=dict(color="#ff7f0e", width=2, dash="dash"), mode="lines"
            ))
            bm_plotted = True

    fig.add_hline(y=1.0, line_dash="dot", line_color="#666666",
                 annotation_text="Initial Capital", annotation_position="right")

    fig.update_layout(
        yaxis_type="log",
        xaxis_title="Date",
        yaxis_title="Cumulative return (normalized, start = 1)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)
    if not bm_plotted:
        st.caption("**NIFTY 50** line not shown: benchmark could not be loaded (e.g. Yahoo Finance unavailable).")

    # ---- Benchmark Comparison ----
    if bm_plotted and "Benchmark_CAGR" in m:
        st.subheader("📊 Strategy vs Benchmark Comparison")
        comp1, comp2, comp3 = st.columns(3)
        with comp1:
            st.metric("Strategy CAGR", f"{m.get('CAGR', 0):.2%}")
        with comp2:
            st.metric("Benchmark CAGR", f"{m.get('Benchmark_CAGR', 0):.2%}")
        with comp3:
            excess = m.get('CAGR', 0) - m.get('Benchmark_CAGR', 0)
            st.metric("Excess Return", f"{excess:.2%}", delta=f"{excess:.2%}")

    # ---- Tradelog (expandable) ----
    if tl is not None and len(tl) > 0:
        with st.expander("📜 Tradelog", expanded=False):
            tl_display = tl.copy()
            tl_display["date"] = pd.to_datetime(tl_display["date"])

            filter_col1, filter_col2, filter_col3 = st.columns(3)
            with filter_col1:
                action_filter = st.selectbox("Action", ["All", "buy", "sell"], key="tl_action")
            with filter_col2:
                companies = sorted(tl_display["company"].dropna().unique().tolist())
                company_filter = st.multiselect("Company (leave empty = all)", options=companies, default=[], key="tl_company")
            with filter_col3:
                min_date = tl_display["date"].min().date()
                max_date = tl_display["date"].max().date()
                date_range = st.date_input("Date range", value=(min_date, max_date), key="tl_date")

            df_show = tl_display.copy()
            if action_filter != "All":
                df_show = df_show[df_show["action"] == action_filter]
            if company_filter:
                df_show = df_show[df_show["company"].isin(company_filter)]
            if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
                df_show = df_show[(df_show["date"].dt.date >= date_range[0]) & (df_show["date"].dt.date <= date_range[1])]

            st.caption(f"Showing **{len(df_show)}** of **{len(tl_display)}** trades.")
            st.dataframe(df_show, use_container_width=True, height=400)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; padding: 20px;'>"
    "📈 PE Discount Backtest | Filter Coffee Finance"
    "</div>",
    unsafe_allow_html=True
)
