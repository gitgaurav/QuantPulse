"""
app.py — QuantPulse Streamlit Dashboard
========================================
Regime-Based Trading App powered by HMM + Technical Analysis + Claude AI

Layout
──────
  Sidebar   : Ticker input, run button, settings
  Top row   : Current Signal + Detected Regime chips (with one-liner notes)
  Chart     : Plotly candlestick with regime spans + Supertrend + RSI subplot
  Metrics   : Total Return | Alpha | Win Rate | Max Drawdown
  Growth    : Strategy vs Buy-and-Hold growth comparison
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
from src.strategy import (
    get_latest_signal,
    evaluate_confirmations,
    SIGNAL_STRONG_BUY,
    SIGNAL_BUY,
    SIGNAL_CASH,
)
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
    .signal-strong-buy { color: #4ade80; font-weight: 700; font-size: 1.6rem; letter-spacing: 0.5px; }
    .signal-buy   { color: #22c55e; font-weight: 700; font-size: 1.6rem; }
    .signal-cash  { color: #f59e0b; font-weight: 700; font-size: 1.6rem; }
    .regime-bull  { color: #22c55e; font-weight: 700; }
    .regime-bear  { color: #ef4444; font-weight: 700; }
    .regime-neutral { color: #94a3b8; font-weight: 700; }
    .decision-buy  { background: #14532d; border-radius: 8px; padding: 12px; }
    .decision-sell { background: #450a0a; border-radius: 8px; padding: 12px; }
    .decision-wait { background: #1c1917; border-radius: 8px; padding: 12px; }
    .note-text { color: #94a3b8; font-size: 0.78rem; margin-top: 4px; }
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
# Chart builders
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
      • Regime-coloured background spans
      • EMA 50 / 200 overlays
      • Supertrend line (green = bullish, red = bearish)
      • RSI (14) subplot with 30 / 55 / 70 reference lines
      • Volume bars colour-coded by regime
    """
    has_rsi = "RSI" in df.columns
    has_st  = "Supertrend" in df.columns and "Supertrend_Dir" in df.columns

    rsi_row  = 2 if has_rsi else None
    vol_row  = 3 if has_rsi else 2
    n_rows   = 3 if has_rsi else 2
    heights  = [0.55, 0.20, 0.25] if has_rsi else [0.75, 0.25]
    titles   = ("", "RSI (14)", "Volume") if has_rsi else ("", "Volume")

    fig = make_subplots(
        rows=n_rows, cols=1,
        shared_xaxes=True,
        row_heights=heights,
        vertical_spacing=0.03,
        subplot_titles=titles,
    )

    # ── Candlestick ──────────────────────────────────────────────────────────
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
            name="Price",
            increasing_line_color="#22c55e", decreasing_line_color="#ef4444",
            increasing_fillcolor="#22c55e", decreasing_fillcolor="#ef4444",
        ),
        row=1, col=1,
    )

    # ── EMA 50 / 200 ─────────────────────────────────────────────────────────
    for col, color, dash in [
        ("EMA_50",  "#60a5fa", "solid"),
        ("EMA_200", "#f59e0b", "dot"),
    ]:
        if col in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=df[col], mode="lines",
                    line=dict(color=color, width=1, dash=dash),
                    name=col.replace("_", " "), opacity=0.8,
                ),
                row=1, col=1,
            )

    # ── Supertrend overlay ───────────────────────────────────────────────────
    if has_st:
        st_bull = df["Supertrend"].where(df["Supertrend_Dir"] == 1,  np.nan)
        st_bear = df["Supertrend"].where(df["Supertrend_Dir"] == -1, np.nan)
        fig.add_trace(
            go.Scatter(
                x=df.index, y=st_bull, mode="lines",
                line=dict(color="#22c55e", width=1.8),
                name="Supertrend ↑", opacity=0.95,
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index, y=st_bear, mode="lines",
                line=dict(color="#ef4444", width=1.8),
                name="Supertrend ↓", opacity=0.95,
            ),
            row=1, col=1,
        )

    # ── RSI subplot ──────────────────────────────────────────────────────────
    if has_rsi and rsi_row:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["RSI"], mode="lines",
                line=dict(color="#a78bfa", width=1.5),
                name="RSI (14)",
            ),
            row=rsi_row, col=1,
        )
        # Shade the entry zone 55–70
        fig.add_hrect(
            y0=55, y1=70,
            fillcolor="rgba(34,197,94,0.08)",
            layer="below", line_width=0,
            row=rsi_row, col=1,
        )
        # Reference lines: 70 (overbought), 55 (entry lower), 30 (oversold)
        for y_val, color, dash in [
            (70, "#ef4444", "dot"),
            (55, "#22c55e", "dash"),
            (30, "#60a5fa", "dot"),
        ]:
            fig.add_shape(
                type="line",
                x0=df.index[0], x1=df.index[-1],
                y0=y_val, y1=y_val,
                line=dict(color=color, width=1, dash=dash),
                row=rsi_row, col=1,
            )
        fig.update_yaxes(range=[0, 100], row=rsi_row, col=1)

    # ── Volume bars ──────────────────────────────────────────────────────────
    vol_colors = [
        "#22c55e" if r == REGIME_BULL else "#ef4444" if r == REGIME_BEAR else "#64748b"
        for r in df.get("Regime", [REGIME_NEUTRAL] * len(df))
    ]
    fig.add_trace(
        go.Bar(
            x=df.index, y=df["Volume"],
            marker_color=vol_colors,
            name="Volume", opacity=0.6, showlegend=False,
        ),
        row=vol_row, col=1,
    )

    # ── Regime background spans ──────────────────────────────────────────────
    if "Regime" in df.columns:
        df_regime = df[["Regime"]].copy()
        df_regime["block"] = (df_regime["Regime"] != df_regime["Regime"].shift()).cumsum()

        for _, grp in df_regime.groupby("block"):
            regime = grp["Regime"].iloc[0]
            x0, x1 = grp.index[0], grp.index[-1]
            color = _regime_color(regime)
            for row_idx in range(1, n_rows + 1):
                fig.add_vrect(
                    x0=x0, x1=x1, fillcolor=color,
                    layer="below", line_width=0,
                    row=row_idx, col=1,
                )

    fig.update_layout(
        paper_bgcolor="#0f0f1a",
        plot_bgcolor="#0f0f1a",
        font=dict(color="#e2e8f0"),
        xaxis_rangeslider_visible=False,
        legend=dict(
            bgcolor="rgba(0,0,0,0.4)", bordercolor="#334155",
            borderwidth=1, font_size=11,
        ),
        margin=dict(l=10, r=10, t=30, b=10),
        height=680,
    )
    fig.update_xaxes(gridcolor="#1e293b", showgrid=True)
    fig.update_yaxes(gridcolor="#1e293b", showgrid=True)
    return fig


def build_equity_chart(equity: pd.Series, initial: float) -> go.Figure:
    pct = (equity / initial - 1) * 100.0
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=equity.index, y=pct, mode="lines",
            line=dict(color="#7c3aed", width=2),
            fill="tozeroy", fillcolor="rgba(124,58,237,0.15)",
            name="Strategy PnL %",
        )
    )
    fig.add_hline(y=0, line_color="#475569", line_dash="dot")
    fig.update_layout(
        paper_bgcolor="#0f0f1a", plot_bgcolor="#0f0f1a",
        font=dict(color="#e2e8f0"),
        margin=dict(l=10, r=10, t=10, b=10),
        height=250,
        xaxis=dict(gridcolor="#1e293b"),
        yaxis=dict(gridcolor="#1e293b", ticksuffix=" %"),
    )
    return fig


def build_growth_chart(equity: pd.Series, df_full: pd.DataFrame, initial: float) -> go.Figure:
    """
    Compare strategy growth vs buy-and-hold over the same backtest window.

    Both curves start at 0% (i.e. ₹5L baseline) so the gap between them
    shows the strategy's alpha at every point in time.
    """
    # Align close prices to the equity curve timestamps
    aligned_close = df_full["Close"].reindex(equity.index, method="nearest")
    first_close   = float(aligned_close.iloc[0])

    bh_equity    = (aligned_close / first_close) * initial
    strategy_pct = (equity    / initial - 1) * 100.0
    bh_pct       = (bh_equity / initial - 1) * 100.0

    final_strat = float(strategy_pct.iloc[-1])
    final_bh    = float(bh_pct.iloc[-1])
    alpha       = final_strat - final_bh

    fig = go.Figure()

    # Buy & Hold (bottom layer)
    fig.add_trace(
        go.Scatter(
            x=bh_equity.index, y=bh_pct, mode="lines",
            line=dict(color="#f59e0b", width=2, dash="dot"),
            name=f"Buy & Hold  ({final_bh:+.1f}%)",
        )
    )

    # Strategy (top layer)
    fig.add_trace(
        go.Scatter(
            x=equity.index, y=strategy_pct, mode="lines",
            line=dict(color="#7c3aed", width=2),
            fill="tonexty",
            fillcolor="rgba(124,58,237,0.12)" if alpha >= 0 else "rgba(239,68,68,0.08)",
            name=f"Strategy  ({final_strat:+.1f}%)",
        )
    )

    # Zero baseline
    fig.add_hline(y=0, line_color="#475569", line_dash="dot")

    # Annotations for final values
    fig.add_annotation(
        x=equity.index[-1], y=final_strat,
        text=f"Strategy {final_strat:+.1f}%",
        showarrow=True, arrowhead=2, ax=-60, ay=-30,
        font=dict(color="#a78bfa", size=12),
        bgcolor="rgba(15,15,26,0.8)",
    )
    fig.add_annotation(
        x=bh_equity.index[-1], y=final_bh,
        text=f"B&H {final_bh:+.1f}%",
        showarrow=True, arrowhead=2, ax=-60, ay=30,
        font=dict(color="#f59e0b", size=12),
        bgcolor="rgba(15,15,26,0.8)",
    )

    alpha_color = "#22c55e" if alpha >= 0 else "#ef4444"
    alpha_label = f"Alpha: {alpha:+.1f}% vs Buy & Hold"
    fig.add_annotation(
        x=0.01, y=0.97,
        xref="paper", yref="paper",
        text=alpha_label,
        showarrow=False,
        font=dict(color=alpha_color, size=13),
        bgcolor="rgba(15,15,26,0.8)",
        bordercolor=alpha_color,
        borderwidth=1,
    )

    fig.update_layout(
        paper_bgcolor="#0f0f1a", plot_bgcolor="#0f0f1a",
        font=dict(color="#e2e8f0"),
        margin=dict(l=10, r=10, t=10, b=10),
        height=320,
        xaxis=dict(gridcolor="#1e293b"),
        yaxis=dict(gridcolor="#1e293b", ticksuffix=" %"),
        legend=dict(
            bgcolor="rgba(0,0,0,0.4)", bordercolor="#334155",
            borderwidth=1, font_size=12,
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
        ),
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
    st.divider()
    st.markdown("**Entry Rules**")
    st.markdown("🟢 **STRONG BUY** (all 4 pass):")
    st.markdown("• HMM Regime = Bull Run")
    st.markdown("• RSI in (55–70)")
    st.markdown("• Supertrend Bullish")
    st.markdown("• Price > EMA 200")
    st.markdown("• ADX > 25")
    st.markdown("🟡 **BUY** (core 3 pass):")
    st.markdown("• HMM Regime = Bull Run")
    st.markdown("• RSI in (55–70)")
    st.markdown("• Supertrend Bullish")
    st.markdown("• ADX > 25")
    st.markdown("*(EMA 200 may fail)*")
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
current_regime: str   = df_full["Regime"].iloc[-1]
current_price: float  = float(df_full["Close"].iloc[-1])

# ---------------------------------------------------------------------------
# LLM recommendation
# ---------------------------------------------------------------------------

import json as _json

# Build indicator payload — include actual numeric values from the last valid row
# so the LLM prompt receives real RSI/ADX/etc. instead of N/A.
_last_row = df_full.iloc[-1]
def _safe_float(val, default=0.0):
    try:
        f = float(val)
        return f if np.isfinite(f) else default
    except Exception:
        return default

_ind_json = _json.dumps({
    "mandatory_count":   confs.get("mandatory_count", 0),
    "mandatory_all_met": confs.get("mandatory_all_met", False),
    "core_3_met":        confs.get("core_3_met", False),
    "optional_count":    confs.get("optional_count", 0),
    "RSI":               round(_safe_float(_last_row.get("RSI")), 2),
    "ADX":               round(_safe_float(_last_row.get("ADX")), 2),
    "Momentum":          round(_safe_float(_last_row.get("Momentum")), 2),
    "Volatility":        round(_safe_float(_last_row.get("Volatility")), 2),
    "Above_EMA50":       bool(_safe_float(_last_row.get("Close")) > _safe_float(_last_row.get("EMA_50"), 1)),
    "Above_EMA200":      bool(_safe_float(_last_row.get("Close")) > _safe_float(_last_row.get("EMA_200"), 1)),
    "MACD_bullish":      bool(_safe_float(_last_row.get("MACD")) > _safe_float(_last_row.get("MACD_Signal"))),
    "Supertrend_bullish": bool(_safe_float(_last_row.get("Supertrend_Dir")) > 0),
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
# ── TOP ROW:  Signal  |  Regime  |  Price
# ---------------------------------------------------------------------------

# One-liner explanation for the Current Signal card
m_count = confs.get("mandatory_count", 0)
if current_signal == SIGNAL_STRONG_BUY:
    signal_note = "All 4/4 mandatory pass (RSI + Supertrend + EMA 200 + ADX) — highest conviction"
elif current_signal == SIGNAL_BUY:
    signal_note = "Core 3/3 pass (RSI + Supertrend + ADX) — EMA 200 not confirmed"
else:
    if current_regime == REGIME_BEAR:
        signal_note = "Bear/Crash regime detected — no long entries permitted"
    elif current_regime == REGIME_NEUTRAL:
        signal_note = "Neutral regime — waiting for Bull Run confirmation"
    else:
        failing = [c for c, v in confs.get("mandatory", {}).items() if not v]
        if failing:
            short_fail = [c.split("(")[0].strip() for c in failing[:2]]
            signal_note = f"Core 3 not met — blocked by: {', '.join(short_fail)}"
        else:
            signal_note = "Insufficient indicator history — still warming up"

# One-liner explanation for the Detected Regime card
_regime_desc = {
    REGIME_BULL:    "HMM detects sustained upward momentum — favorable for long positions",
    REGIME_BEAR:    "HMM detects negative return state — protect capital, avoid longs",
    REGIME_NEUTRAL: "HMM shows mixed market signals — no directional conviction yet",
}
regime_note = _regime_desc.get(current_regime, "")

col_sig, col_reg, col_price = st.columns(3)

if current_signal == SIGNAL_STRONG_BUY:
    signal_class = "signal-strong-buy"
    signal_icon  = "🟢 STRONG BUY"
elif current_signal == SIGNAL_BUY:
    signal_class = "signal-buy"
    signal_icon  = "🟢 BUY"
else:
    signal_class = "signal-cash"
    signal_icon  = "🟡 CASH"

regime_class = {
    REGIME_BULL:    "regime-bull",
    REGIME_BEAR:    "regime-bear",
    REGIME_NEUTRAL: "regime-neutral",
}.get(current_regime, "regime-neutral")

with col_sig:
    st.markdown(
        f"<div class='metric-card'>"
        f"<small>Current Signal</small><br>"
        f"<span class='{signal_class}'>{signal_icon}</span><br>"
        f"<span class='note-text'>{signal_note}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

with col_reg:
    st.markdown(
        f"<div class='metric-card'>"
        f"<small>Detected Regime</small><br>"
        f"<span class='{regime_class}' style='font-size:1.3rem;'>{current_regime}</span><br>"
        f"<span class='note-text'>{regime_note}</span>"
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
# ── CHART  (price + Supertrend + RSI + Volume)
# ---------------------------------------------------------------------------

# Show last 90 days on the chart for readability (~630 hourly bars)
chart_df = df_full.iloc[-90 * 7:]

st.subheader("Price Chart with Regime Overlay · Supertrend · RSI")
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
# ── STRATEGY vs BUY & HOLD GROWTH CHART
# ---------------------------------------------------------------------------

st.subheader("Strategy Growth vs Buy & Hold")
st.caption(
    "Purple = QuantPulse strategy return  ·  Gold dashed = passive buy-and-hold return  "
    "over the same backtest window (shaded area = alpha generated by the strategy)"
)
st.plotly_chart(
    build_growth_chart(results["equity_curve"], df_full, INITIAL_CAPITAL),
    use_container_width=True,
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

if decision == "Strong Buy":
    card_class = "decision-buy"
    icon = "🟢🟢"
elif decision == "Buy":
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

with st.expander("📊 Indicator Breakdown — Mandatory & Optional", expanded=True):
    readings  = confs.get("readings",  {})
    reasons   = confs.get("reasons",   {})
    mandatory = confs.get("mandatory", {})
    optional  = confs.get("optional",  {})

    col_config = {
        "Condition":    st.column_config.TextColumn("Condition",    width="medium"),
        "Status":       st.column_config.TextColumn("Status",       width="small"),
        "Actual Value": st.column_config.TextColumn("Actual Value", width="medium"),
        "Reasoning":    st.column_config.TextColumn("Reasoning",    width="large"),
    }

    def _build_rows(conditions: dict) -> pd.DataFrame:
        return pd.DataFrame([
            {
                "Condition":    cond,
                "Status":       "✅  Pass" if passed else "❌  Fail",
                "Actual Value": readings.get(cond, "—"),
                "Reasoning":    reasons.get(cond, "—"),
            }
            for cond, passed in conditions.items()
        ])

    # ── Mandatory section ─────────────────────────────────────────────────
    m_count = confs.get("mandatory_count", 0)
    m_color = "green" if m_count == 4 else "orange" if m_count == 3 else "red"
    st.markdown(
        f"#### 🔴 Mandatory Indicators "
        f"<span style='color:{m_color}'>({m_count} / 4 passed)</span>"
        f" — **4/4 = STRONG BUY · 3/4 with RSI+ST+ADX = BUY**",
        unsafe_allow_html=True,
    )
    st.dataframe(_build_rows(mandatory), use_container_width=True,
                 hide_index=True, column_config=col_config)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Optional section ──────────────────────────────────────────────────
    o_count = confs.get("optional_count", 0)
    st.markdown(
        f"#### 📊 Optional Indicators "
        f"({o_count} / 5 passed)"
        f" — *for analysis only, do not affect trade decision*",
    )
    st.dataframe(_build_rows(optional), use_container_width=True,
                 hide_index=True, column_config=col_config)

    # ── Summary ───────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    bull = current_regime == REGIME_BULL
    if current_signal == SIGNAL_STRONG_BUY:
        st.success("4/4 mandatory pass + Bull regime → **STRONG BUY** — all confirmations met")
    elif current_signal == SIGNAL_BUY:
        st.info("RSI + Supertrend + ADX pass + Bull regime → **BUY** — EMA 200 not confirmed (elevated risk)")
    elif not bull:
        st.warning(f"Regime is **{current_regime}** — Bull Run required for entry")
    else:
        failing = [c for c, v in mandatory.items() if not v]
        st.error(f"Core 3 not met ({m_count}/4 mandatory pass) — blocked by: {', '.join(failing)}")

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

        csv = tl.to_csv(index=False).encode()
        st.download_button(
            label="⬇ Download Trade Log (CSV)",
            data=csv,
            file_name=f"quantpulse_{ticker_input}_trades.csv",
            mime="text/csv",
        )
