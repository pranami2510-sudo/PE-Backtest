"""
PE Discount-to-Median Backtest — Streamlit frontend (interactive).
Run: streamlit run streamlit_app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pe_backtest import load_all_data, run_backtest, metrics

# Optional: Plotly for interactive chart
try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

st.set_page_config(
    page_title="PE Discount Backtest",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .hero { padding: 1.5rem 0; border-bottom: 1px solid #eee; margin-bottom: 1.5rem; }
    .param-card { background: #f0f2f6; padding: 0.75rem 1rem; border-radius: 6px; margin: 0.25rem 0; font-size: 0.9rem; }
    .preset-btn { margin: 0.2rem 0; }
</style>
""", unsafe_allow_html=True)

# Session state
if "backtest_results" not in st.session_state:
    st.session_state.backtest_results = None
for key, default in [("lookback", 3), ("discount", 0.2), ("holding", 2), ("fixed_q4", False)]:
    if key not in st.session_state:
        st.session_state[key] = default

# Presets: update session state when clicked
st.sidebar.header("⚙️ Strategy parameters")
st.sidebar.caption("Quick presets (click to apply):")
preset_col1, preset_col2 = st.sidebar.columns(2)
with preset_col1:
    if st.button("Conservative", key="preset_cons", help="L=4, 20%, H=3"):
        st.session_state.update(lookback=4, discount=0.2, holding=3, fixed_q4=False)
    if st.button("Moderate", key="preset_mod", help="L=3, 25%, H=2"):
        st.session_state.update(lookback=3, discount=0.25, holding=2, fixed_q4=False)
with preset_col2:
    if st.button("Aggressive", key="preset_agg", help="L=2, 50%, H=1"):
        st.session_state.update(lookback=2, discount=0.5, holding=1, fixed_q4=False)
    if st.button("Option A", key="preset_q4", help="L=3, 20%, fixed Q4"):
        st.session_state.update(lookback=3, discount=0.2, holding=3, fixed_q4=True)
st.sidebar.markdown("---")

with st.sidebar.expander("What do these mean?", expanded=False):
    st.markdown("""
    - **Lookback:** Quarters of history for median PE.
    - **Discount:** Buy only if PE ≤ (1 − discount) × median PE.
    - **Holding:** Quarters to hold each cohort.
    - **Option A:** Rebalance only at end of Q4, hold 3Q, cash in Q4.
    """)

lookback = st.sidebar.slider("Lookback (quarters)", 1, 20, value=st.session_state.get("lookback", 3), key="sb_lookback")
discount = st.sidebar.slider("Discount to median PE", 0.10, 0.90, value=float(st.session_state.get("discount", 0.2)), step=0.05, key="sb_discount")
fixed_q4 = st.sidebar.checkbox("Option A (fixed Q4)", value=st.session_state.get("fixed_q4", False), key="sb_fixed_q4")
if fixed_q4:
    holding = 3
    st.sidebar.caption("Holding fixed at 3 quarters.")
else:
    holding = st.sidebar.slider("Holding (quarters)", 1, 5, value=st.session_state.get("holding", 2), key="sb_holding")
# Persist for next run
st.session_state["lookback"] = lookback
st.session_state["discount"] = discount
st.session_state["holding"] = holding
st.session_state["fixed_q4"] = fixed_q4

# Reactive summary
max_pct = (1 - discount) * 100
st.sidebar.markdown(f"**Screen:** PE ≤ **{max_pct:.0f}%** of median PE")
st.sidebar.markdown("---")

run = st.sidebar.button("▶️ Run backtest", type="primary", use_container_width=True)
st.sidebar.caption("Data: Cleaned PE Data No Outliers")

# Header
st.markdown('<div class="hero">', unsafe_allow_html=True)
st.title("📈 PE Discount-to-Median Backtest")
st.markdown(
    "Backtest a strategy that buys stocks when **PE ≤ (1 − discount) × median PE**, holds for **H** quarters, then sells. Compare to NIFTY 50."
)
st.markdown('</div>', unsafe_allow_html=True)

tab_overview, tab_backtest, tab_tradelog, tab_downloads = st.tabs(["📋 Overview", "📊 Backtest & results", "📜 Tradelog", "⬇️ Downloads"])

# Overview
with tab_overview:
    st.subheader("Strategy in brief")
    st.markdown("""
    1. **Screen** — Each rebalance quarter: median PE over last **L** quarters; keep only **PE ≤ (1 − discount) × median PE**.
    2. **Buy** — Equal weight at each stock’s last valid day in the quarter.
    3. **Hold** — **H** quarters (or 3 in Option A).
    4. **Sell** — Last trading day of exit quarter.
    5. **Repeat** — No overlapping cohorts.
    """)
    st.subheader("Current parameters")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Lookback**")
        st.caption(f"{lookback} quarters")
    with c2:
        st.markdown("**Discount**")
        st.caption(f"{discount:.0%} → buy when PE ≤ {max_pct:.0f}% of median")
    with c3:
        st.markdown("**Holding**")
        st.caption(f"{holding} quarter(s)")
    if fixed_q4:
        st.info("Option A: rebalance only at end of Q4, hold 3Q, cash in Q4.")

# Run backtest
if run:
    with st.spinner("Loading data..."):
        data = load_all_data(progress=False)
    if not data:
        st.error("No data found. Ensure **Cleaned PE Data No Outliers** exists with company CSVs.")
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
                benchmark_path="", progress_callback=progress_cb, fixed_q4=fixed_q4
            )
    progress_bar.empty()
    progress_placeholder.empty()

    if equity is None or len(equity) < 2:
        st.warning("No results: no companies passed the screen. Try a **lower discount** or **shorter lookback**.")
        st.stop()

    m = metrics(equity, benchmark_series)
    st.session_state.backtest_results = {
        "tradelog": tradelog,
        "equity": equity,
        "metrics": m,
        "params": {"lookback": lookback, "discount": discount, "holding": holding, "fixed_q4": fixed_q4},
        "n_companies": n_companies,
        "benchmark_series": benchmark_series,
    }
    st.sidebar.success(f"Loaded {n_companies} companies. Backtest done.")
    st.rerun()

# Backtest tab
with tab_backtest:
    if st.session_state.backtest_results is None:
        st.info("👈 Set parameters and click **Run backtest** to see the equity curve and metrics.")
    else:
        res = st.session_state.backtest_results
        equity = res["equity"]
        m = res["metrics"]
        num_trades = len(res["tradelog"]) if res["tradelog"] is not None else 0
        lookback = res["params"]["lookback"]
        discount = res["params"]["discount"]
        holding = res["params"]["holding"]
        fixed_q4 = res["params"]["fixed_q4"]
        benchmark_series = res.get("benchmark_series")

        # Optional date range for chart
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

        st.subheader("Equity curve (strategy vs NIFTY 50)")
        if HAS_PLOTLY:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=equity_sub.index, y=equity_sub.values, name="Strategy",
                line=dict(color="#1f77b4", width=2), mode="lines"
            ))
            if benchmark_series is not None and len(benchmark_series) > 0:
                bm_cum = (1 + benchmark_series).cumprod()
                bm_cum = bm_cum / bm_cum.iloc[0]
                bm_sub = bm_cum[(bm_cum.index.date >= chart_start) & (bm_cum.index.date <= chart_end)]
                if len(bm_sub) > 0:
                    fig.add_trace(go.Scatter(
                        x=bm_sub.index, y=bm_sub.values, name="NIFTY 50",
                        line=dict(color="#7f7f7f", width=1.5, dash="dot"), mode="lines"
                    ))
            fig.update_layout(
                yaxis_type="log",
                xaxis_title="Date",
                yaxis_title="Cumulative return (normalized, start = 1)",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(t=50),
                template="plotly_white",
                height=450,
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Interactive: zoom, pan, hover for values. Use the toolbar to reset view.")
        else:
            fig, ax = plt.subplots(figsize=(11, 5))
            ax.plot(equity_sub.index, equity_sub.values, label="Strategy", color="#1f77b4", linewidth=1.5)
            if benchmark_series is not None and len(benchmark_series) > 0:
                bm_cum = (1 + benchmark_series).cumprod()
                bm_cum = bm_cum / bm_cum.iloc[0]
                bm_sub = bm_cum[(bm_cum.index >= equity_sub.index.min()) & (bm_cum.index <= equity_sub.index.max())]
                ax.plot(bm_sub.index, bm_sub.values, label="NIFTY 50", color="#7f7f7f", linewidth=1.2, alpha=0.9)
            ax.set_xlabel("Date")
            ax.set_ylabel("Cumulative return (normalized, start = 1)")
            ax.set_yscale("log")
            ax.legend(loc="upper left")
            ax.grid(True, alpha=0.3, which="both")
            ax.set_title(f"Lookback={lookback}Q, Discount={discount:.0%}, Hold={holding}Q" + (" [Option A]" if fixed_q4 else ""))
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.subheader("Performance metrics")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Trades", num_trades)
        col2.metric("CAGR", f"{m.get('CAGR', 0):.2%}")
        col3.metric("Sharpe (ann.)", f"{m.get('Sharpe', 0):.2f}")
        col4.metric("Calmar", f"{m.get('Calmar', 0):.2f}")
        col5.metric("Max drawdown", f"{m.get('MaxDD', 0):.2%}")
        if "Benchmark_CAGR" in m:
            st.caption(f"**Benchmark (NIFTY 50):** CAGR {m['Benchmark_CAGR']:.2%}  ·  Sharpe {m.get('Benchmark_Sharpe', 0):.2f}")

# Tradelog tab
with tab_tradelog:
    if st.session_state.backtest_results is None:
        st.info("Run a backtest first to see the tradelog.")
    else:
        tradelog = st.session_state.backtest_results["tradelog"]
        if tradelog is not None and len(tradelog) > 0:
            tradelog["date"] = pd.to_datetime(tradelog["date"])
            st.caption(f"Total rows: {len(tradelog)}. Filter below.")
            filter_col1, filter_col2, filter_col3 = st.columns(3)
            with filter_col1:
                action_filter = st.selectbox("Action", ["All", "buy", "sell"], key="tl_action")
            with filter_col2:
                companies = sorted(tradelog["company"].dropna().unique().tolist())
                company_filter = st.multiselect("Company (leave empty = all)", options=companies, default=[], key="tl_company")
            with filter_col3:
                min_date = tradelog["date"].min().date()
                max_date = tradelog["date"].max().date()
                date_range = st.date_input("Date range", value=(min_date, max_date), key="tl_date")
            df_show = tradelog.copy()
            if action_filter != "All":
                df_show = df_show[df_show["action"] == action_filter]
            if company_filter:
                df_show = df_show[df_show["company"].isin(company_filter)]
            if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
                df_show = df_show[(df_show["date"].dt.date >= date_range[0]) & (df_show["date"].dt.date <= date_range[1])]
            st.dataframe(df_show, use_container_width=True, height=400)
        else:
            st.write("No trades in this run.")

# Downloads tab
with tab_downloads:
    if st.session_state.backtest_results is None:
        st.info("Run a backtest first to download results.")
    else:
        res = st.session_state.backtest_results
        equity = res["equity"]
        tradelog = res["tradelog"]
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Equity curve")
            equity_df = equity.to_frame(name="Value")
            equity_df.index.name = "Date"
            st.download_button("Download equity_curve.csv", data=equity_df.to_csv(), file_name="equity_curve.csv", mime="text/csv", key="dl_equity")
        with col_b:
            st.subheader("Tradelog")
            if tradelog is not None and len(tradelog) > 0:
                st.download_button("Download tradelog.csv", data=tradelog.to_csv(index=False), file_name="tradelog.csv", mime="text/csv", key="dl_tradelog")
            else:
                st.caption("No tradelog for this run.")

st.sidebar.markdown("---")
st.sidebar.markdown("[📄 Strategy log (GitHub)](https://github.com/pranami2510-sudo/PE-Backtest/blob/main/strategy_log.md)")
