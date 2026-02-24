"""
app.py — QuantPulse Streamlit Dashboard
========================================
Regime-Based Trading App powered by HMM + Technical Analysis + Claude AI

Layout
──────
  Sidebar   : Ticker input, run button, settings
  Top row   : Current Signal + Detected Regime chips
  Chart     : Plotly candlestick with regime-coloured background + volume
  Metrics   : Total Return | Alpha | Win Rate | Max Drawdown
  Decision  : Claude LLM recommendation card
  Trade Log : Expandable table of all historical trades
  Equity    : Portfolio value curve
"""

from __future__ import annotations

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dotenv import load_dotenv

from src.data_loader import fetch_ohlcv, validate_ticker_with_llm
from src.backtester import Backtester, INITIAL_CAPITAL
from src.strategy import get_latest_signal, evaluate_confirmations
from src.llm_analyzer import get_trade_recommendation
from src.regime_engine import REGIME_BULL, REGIME_BEAR, REGIME_NEUTRAL

load_dotenv()

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="QuantPulse",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .metric-card {
        background: #1e1e2e;
        border-radius: 12px;
        padding: 16px 20px;
        margin: 4px 0;
        border-left: 4px solid #7c3aed;
    }
    .signal-long  { color: #22c55e; font-weight: 700; font-size: 1.6rem; }
    .signal-cash  { color: #f59e0b; font-weight: 700; font-size: 1.6rem; }
    .regime-bull  { color: #22c55e; font-weight: 700; }
    .regime-bear  { color: #ef4444; font-weight: 700; }
    .regime-neutral { color: #94a3b8; font-weight: 700; }
    .decision-buy  { background: #14532d; border-radius: 8px; padding: 12px; }
    .decision-sell { background: #450a0a; border-radius: 8px; padding: 12px; }
    .decision-wait { background: #1c1917; border-radius: 8px; padding: 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Cached functions
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_data(ticker: str) -> pd.DataFrame:
    return fetch_ohlcv(ticker)


@st.cache_data(show_spinner=False)
def run_backtest(ticker: str) -> dict:
    data = load_data(ticker)
    bt = Backtester()
    return bt.run(data)


@st.cache_data(show_spinner=False)
def llm_analysis(ticker, regime, signal, indicators_json, metrics_json, price):
    import json
    indicators = json.loads(indicators_json)
    metrics    = json.loads(metrics_json)
    return get_trade_recommendation(ticker, regime, signal, indicators, metrics, price)


# ---------------------------------------------------------------------------
# Chart builder
# ---------------------------------------------------------------------------

def _regime_color(regime: str, alpha: float = 0.10) -> str:
    if regime == REGIME_BULL:
        return f"rgba(34,197,94,{alpha})"
    if regime == REGIME_BEAR:
        return f"rgba(239,68,68,{alpha})"
    return f"rgba(148,163,184,{alpha})"


def build_chart(df: pd.DataFrame) -> go.Figure:
    """
    Build an interactive Plotly candlestick chart with:
      • regime-coloured background spans
      • volume bars (colour-coded by regime)
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.02,
        subplot_titles=("", "Volume"),
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price",
            increasing_line_color="#22c55e",
            decreasing_line_color="#ef4444",
            increasing_fillcolor="#22c55e",
            decreasing_fillcolor="#ef4444",
        ),
        row=1, col=1,
    )

    # EMA 50 / 200
    for col, color, dash in [
        ("EMA_50",  "#60a5fa", "solid"),
        ("EMA_200", "#f59e0b", "dot"),
    ]:
        if col in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=df[col],
                    mode="lines",
                    line=dict(color=color, width=1, dash=dash),
                    name=col.replace("_", " "),
                    opacity=0.8,
                ),
                row=1, col=1,
            )

    # Volume bars
    vol_colors = [
        "#22c55e" if r == REGIME_BULL else "#ef4444" if r == REGIME_BEAR else "#64748b"
        for r in df.get("Regime", [REGIME_NEUTRAL] * len(df))
    ]
    fig.add_trace(
        go.Bar(
            x=df.index, y=df["Volume"],
            marker_color=vol_colors,
            name="Volume",
            opacity=0.6,
            showlegend=False,
        ),
        row=2, col=1,
    )

    # Regime background spans
    if "Regime" in df.columns:
        df_regime = df[["Regime"]].copy()
        df_regime["block"] = (df_regime["Regime"] != df_regime["Regime"].shift()).cumsum()

        for _, grp in df_regime.groupby("block"):
            regime = grp["Regime"].iloc[0]
            x0, x1 = grp.index[0], grp.index[-1]
            color = _regime_color(regime)

            for row_idx in [1, 2]:
                fig.add_vrect(
                    x0=x0, x1=x1,
                    fillcolor=color,
                    layer="below",
                    line_width=0,
                    row=row_idx, col=1,
                )

    fig.update_layout(
        paper_bgcolor="#0f0f1a",
        plot_bgcolor="#0f0f1a",
        font=dict(color="#e2e8f0"),
        xaxis_rangeslider_visible=False,
        legend=dict(
            bgcolor="rgba(0,0,0,0.4)",
            bordercolor="#334155",
            borderwidth=1,
            font_size=11,
        ),
        margin=dict(l=10, r=10, t=30, b=10),
        height=550,
    )
    fig.update_xaxes(gridcolor="#1e293b", showgrid=True)
    fig.update_yaxes(gridcolor="#1e293b", showgrid=True)
    return fig


def build_equity_chart(equity: pd.Series, initial: float) -> go.Figure:
    pct = (equity / initial - 1) * 100.0

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=equity.index, y=pct,
            mode="lines",
            line=dict(color="#7c3aed", width=2),
            fill="tozeroy",
            fillcolor="rgba(124,58,237,0.15)",
            name="Strategy PnL %",
        )
    )
    fig.add_hline(y=0, line_color="#475569", line_dash="dot")
    fig.update_layout(
        paper_bgcolor="#0f0f1a",
        plot_bgcolor="#0f0f1a",
        font=dict(color="#e2e8f0"),
        margin=dict(l=10, r=10, t=10, b=10),
        height=250,
        xaxis=dict(gridcolor="#1e293b"),
        yaxis=dict(gridcolor="#1e293b", ticksuffix=" %"),
    )
    return fig


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("⚡ QuantPulse")
    st.caption("Regime-Based Algorithmic Trading")
    st.divider()

    ticker_input = st.text_input(
        "Ticker Symbol",
        value="AAPL",
        placeholder="e.g. AAPL, ^NSEI, RELIANCE.NS, BTC-USD",
        help="Any Yahoo Finance symbol. For NSE India append .NS (e.g. TCS.NS)",
    ).strip().upper()

    run_btn = st.button("▶  Run Analysis", use_container_width=True, type="primary")

    st.divider()
    st.markdown("**Strategy Settings**")
    st.markdown(f"• Starting Capital : ₹{INITIAL_CAPITAL:,.0f}")
    st.markdown("• Leverage         : 2.5×")
    st.markdown("• Cooldown         : 48 h after exit")
    st.markdown("• HMM Components   : 7")
    st.markdown("• Min Confirmations: 7 / 8")
    st.divider()
    st.caption("Powered by **Claude claude-opus-4-6** · [Anthropic](https://anthropic.com)")

# ---------------------------------------------------------------------------
# Hero section
# ---------------------------------------------------------------------------

st.markdown(
    "<h1 style='margin-bottom:0;'>📈 QuantPulse</h1>"
    "<p style='color:#64748b;margin-top:0;'>Regime-Based Trading · HMM + 8-Confirmation Voting · Claude AI Decisions</p>",
    unsafe_allow_html=True,
)

if not run_btn:
    st.info("Enter a ticker symbol in the sidebar and click **▶ Run Analysis** to start.", icon="💡")
    st.stop()

# ---------------------------------------------------------------------------
# Data loading & backtest
# ---------------------------------------------------------------------------

progress = st.progress(0, text="Fetching market data…")
try:
    data = load_data(ticker_input)
    progress.progress(25, text="Running HMM regime detection…")
except Exception as e:
    st.error(f"**Data error:** {e}")
    st.stop()

try:
    results = run_backtest(ticker_input)
    progress.progress(75, text="Computing LLM decision…")
except Exception as e:
    st.error(f"**Backtest error:** {e}")
    st.stop()

df_full: pd.DataFrame = results["df_with_indicators"]

# ---------------------------------------------------------------------------
# Current signal & regime
# ---------------------------------------------------------------------------

current_signal, confs = get_latest_signal(df_full)
current_regime: str = df_full["Regime"].iloc[-1]
current_price: float = float(df_full["Close"].iloc[-1])

# ---------------------------------------------------------------------------
# LLM recommendation
# ---------------------------------------------------------------------------

import json as _json

_ind_json = _json.dumps({
    k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else bool(v))
    for k, v in confs.items()
    if k != "readings"          # 'readings' is a nested dict — exclude from LLM JSON
})
_met_json = _json.dumps({
    k: v for k, v in results.items()
    if isinstance(v, (int, float, str))
})

try:
    recommendation = llm_analysis(
        ticker_input,
        current_regime,
        current_signal,
        _ind_json,
        _met_json,
        current_price,
    )
except Exception:
    recommendation = {
        "decision": "Hold/Wait",
        "holding_period": "N/A",
        "confidence": 0,
        "reason": "LLM analysis unavailable. Check ANTHROPIC_API_KEY.",
        "key_risks": [],
    }

progress.progress(100, text="Done")
progress.empty()

# ---------------------------------------------------------------------------
# ── TOP ROW:  Signal   |   Regime   |   Price
# ---------------------------------------------------------------------------

col_sig, col_reg, col_price = st.columns(3)

signal_class = "signal-long" if current_signal == "LONG" else "signal-cash"
signal_icon  = "🟢 LONG" if current_signal == "LONG" else "🟡 CASH"

regime_class = {
    REGIME_BULL:    "regime-bull",
    REGIME_BEAR:    "regime-bear",
    REGIME_NEUTRAL: "regime-neutral",
}.get(current_regime, "regime-neutral")

with col_sig:
    st.markdown(
        f"<div class='metric-card'>"
        f"<small>Current Signal</small><br>"
        f"<span class='{signal_class}'>{signal_icon}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

with col_reg:
    st.markdown(
        f"<div class='metric-card'>"
        f"<small>Detected Regime</small><br>"
        f"<span class='{regime_class}' style='font-size:1.3rem;'>{current_regime}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

with col_price:
    st.markdown(
        f"<div class='metric-card'>"
        f"<small>Latest Close</small><br>"
        f"<span style='font-size:1.4rem; font-weight:700;'>{current_price:,.4f}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# ── CHART
# ---------------------------------------------------------------------------

# Show last 90 days on the chart for readability
chart_df = df_full.iloc[-90 * 7:]   # ~90 days of hourly bars (≈630 bars)

st.subheader("Price Chart with Regime Overlay")
st.plotly_chart(build_chart(chart_df), use_container_width=True)

# ---------------------------------------------------------------------------
# ── METRICS ROW
# ---------------------------------------------------------------------------

st.subheader("Backtest Performance")

m1, m2, m3, m4 = st.columns(4)

def _delta_fmt(v: float) -> str:
    return f"{v:+.2f}%"

m1.metric(
    "Total Return",
    f"{results['total_return']:.2f}%",
    delta=_delta_fmt(results["total_return"]),
)
m2.metric(
    "Alpha vs Buy & Hold",
    f"{results['alpha']:.2f}%",
    delta=_delta_fmt(results["alpha"]),
)
m3.metric(
    "Win Rate",
    f"{results['win_rate']:.1f}%",
    delta=f"{results['num_trades']} trades",
)
m4.metric(
    "Max Drawdown",
    f"{results['max_drawdown']:.2f}%",
    delta=f"B&H: {results['bh_return']:.2f}%",
    delta_color="off",
)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# ── DECISION SECTION
# ---------------------------------------------------------------------------

st.subheader("🤖 AI-Powered Trade Decision")

decision   = recommendation.get("decision", "Hold/Wait")
holding    = recommendation.get("holding_period", "N/A")
confidence = recommendation.get("confidence", 0)
reason     = recommendation.get("reason", "")
risks      = recommendation.get("key_risks", [])

if "Buy" in decision:
    card_class = "decision-buy"
    icon = "🟢"
elif "Sell" in decision:
    card_class = "decision-sell"
    icon = "🔴"
else:
    card_class = "decision-wait"
    icon = "🟡"

dec_col, conf_col = st.columns([3, 1])

with dec_col:
    st.markdown(
        f"<div class='{card_class}'>"
        f"<h3 style='margin:0;'>{icon} {decision}</h3>"
        f"<p style='margin:6px 0 0;'>Suggested holding period: <b>{holding}</b></p>"
        f"<p style='margin:8px 0 4px; color:#cbd5e1;'>{reason}</p>"
        "</div>",
        unsafe_allow_html=True,
    )

with conf_col:
    st.metric("Confidence", f"{confidence}%")
    if risks:
        st.markdown("**Key Risks**")
        for r in risks:
            st.markdown(f"• {r}")

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# ── 8-CONFIRMATION BREAKDOWN
# ---------------------------------------------------------------------------

with st.expander("📊 8-Confirmation Signal Breakdown", expanded=True):
    readings = confs.get("readings", {})
    skip     = {"total_confirmations", "readings"}

    breakdown_rows = []
    for condition, passed in confs.items():
        if condition in skip:
            continue
        breakdown_rows.append({
            "Condition": condition,
            "Status":    "✅  Pass" if passed else "❌  Fail",
            "Actual Value": readings.get(condition, "—"),
        })

    breakdown_df = pd.DataFrame(breakdown_rows)
    st.dataframe(
        breakdown_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Status": st.column_config.TextColumn(width="small"),
            "Actual Value": st.column_config.TextColumn(width="large"),
        },
    )
    total = confs["total_confirmations"]
    if total >= 7:
        color = "green"
    elif total >= 5:
        color = "orange"
    else:
        color = "red"
    st.markdown(
        f"**Confirmations: <span style='color:{color}'>{total} / 8</span>**"
        f"   — need ≥ 7 + Bull regime for LONG entry",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# ── EQUITY CURVE
# ---------------------------------------------------------------------------

st.subheader("Portfolio Equity Curve")
st.plotly_chart(
    build_equity_chart(results["equity_curve"], INITIAL_CAPITAL),
    use_container_width=True,
)

# ---------------------------------------------------------------------------
# ── TRADE LOG
# ---------------------------------------------------------------------------

with st.expander(f"📋 Trade Log ({results['num_trades']} trades)", expanded=False):
    tl: pd.DataFrame = results["trade_log"]
    if tl.empty:
        st.info("No completed trades in the backtest period.")
    else:
        tl_display = tl.copy()
        tl_display["PnL %"] = tl_display["PnL %"].apply(
            lambda x: f"{x:+.2f}%" if pd.notna(x) else "—"
        )
        st.dataframe(tl_display, use_container_width=True, hide_index=True)

        # Download button
        csv = tl.to_csv(index=False).encode()
        st.download_button(
            label="⬇ Download Trade Log (CSV)",
            data=csv,
            file_name=f"quantpulse_{ticker_input}_trades.csv",
            mime="text/csv",
        )
