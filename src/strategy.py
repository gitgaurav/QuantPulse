"""
strategy.py — Three-layer Decision Architecture
================================================

Layer 1 — Indicator Calculation
    Computes all 9 indicators: RSI, Supertrend, EMA-50, EMA-200, MACD,
    Momentum, ATR Volatility, ADX, Volume MA.

Layer 2 — Mandatory Decision Logic  (4 indicators, ALL must pass)
    RSI in (55–70)      — momentum sweet-spot
    Supertrend Bullish  — price above Supertrend line (trend confirmed)
    Price > EMA 200     — long-term bull structure intact
    MACD > Signal       — short-term momentum positive

    Entry signal = LONG  iff  HMM regime == "Bull Run"
                             AND all 4 mandatory conditions are True.
    Any single mandatory failure → CASH / no trade.

Layer 3 — Optional Analytical Indicators  (5 indicators, display only)
    Momentum > 1%       — short-term acceleration
    ATR Volatility < 6% — calm market conditions
    ADX > 25            — trend strength
    Price > EMA 50      — medium-term trend
    Volume > MA(20)     — volume participation

    These are computed, shown in the breakdown table, but do NOT affect
    the final trade decision.

Supertrend
──────────
    Period = 10 bars, Multiplier = 3.0  (widely accepted defaults).
    Direction 1 = Bullish (price ≥ lower band).
    Direction -1 = Bearish (price ≤ upper band).
"""

import numpy as np
import pandas as pd
import ta

from src.regime_engine import REGIME_BULL

SIGNAL_STRONG_BUY = "STRONG BUY"
SIGNAL_BUY        = "BUY"
SIGNAL_CASH       = "CASH"

# ── Condition label constants ────────────────────────────────────────────────
# Mandatory (Layer 2)
C_RSI    = "RSI in (55–70)"
C_ST     = "Supertrend Bullish"
C_EMA200 = "Price > EMA 200"
C_MACD   = "MACD > Signal"

# Optional (Layer 3)
C_MOM    = "Momentum > 1%"
C_VOL    = "ATR Volatility < 6%"
C_ADX    = "ADX > 25"
C_EMA50  = "Price > EMA 50"
C_VOLUME = "Volume > MA(20)"

MANDATORY_CONDITIONS = [C_RSI, C_ST, C_EMA200, C_MACD, C_ADX]
OPTIONAL_CONDITIONS  = [C_MOM, C_VOL, C_EMA50, C_VOLUME]

# All DataFrame columns that must be finite before any evaluation
REQUIRED_INDICATOR_COLS = [
    "RSI", "Supertrend", "Supertrend_Dir",
    "EMA_50", "EMA_200", "MACD", "MACD_Signal",
    "Momentum", "Volatility", "ADX", "Volume_MA",
]


# ---------------------------------------------------------------------------
# Supertrend (Layer 1 helper — not in `ta` library, implemented manually)
# ---------------------------------------------------------------------------

def _compute_supertrend(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 10,
    multiplier: float = 3.0,
) -> tuple[pd.Series, pd.Series]:
    """
    Compute Supertrend indicator.

    Returns
    -------
    supertrend_line : pd.Series  — the band price currently acting as support/resistance
    direction       : pd.Series  —  1 = Bullish (price above lower band)
                                   -1 = Bearish (price below upper band)
    """
    atr = ta.volatility.AverageTrueRange(
        high=high, low=low, close=close, window=period
    ).average_true_range()

    hl2          = (high + low) / 2.0
    upper_basic  = hl2 + multiplier * atr
    lower_basic  = hl2 - multiplier * atr

    n   = len(close)
    idx = close.index

    final_upper = np.full(n, np.nan)
    final_lower = np.full(n, np.nan)
    supertrend  = np.full(n, np.nan)
    direction   = np.full(n, np.nan)

    # Find first bar where ATR is valid
    first_valid_pos = atr.first_valid_index()
    if first_valid_pos is None:
        return (
            pd.Series(supertrend, index=idx, name="Supertrend"),
            pd.Series(direction,  index=idx, name="Supertrend_Dir"),
        )

    start = idx.get_loc(first_valid_pos)

    # Initialise at the first valid bar
    final_upper[start] = float(upper_basic.iloc[start])
    final_lower[start] = float(lower_basic.iloc[start])
    if float(close.iloc[start]) > final_upper[start]:
        supertrend[start] = final_lower[start]
        direction[start]  = 1.0
    else:
        supertrend[start] = final_upper[start]
        direction[start]  = -1.0

    for i in range(start + 1, n):
        ub = float(upper_basic.iloc[i])
        lb = float(lower_basic.iloc[i])
        cl = float(close.iloc[i])
        prev_cl = float(close.iloc[i - 1])

        if np.isnan(ub) or np.isnan(lb):
            final_upper[i] = final_upper[i - 1]
            final_lower[i] = final_lower[i - 1]
            supertrend[i]  = supertrend[i - 1]
            direction[i]   = direction[i - 1]
            continue

        # Final Upper Band — only tighten; reset if prev close was above it
        if ub < final_upper[i - 1] or prev_cl > final_upper[i - 1]:
            final_upper[i] = ub
        else:
            final_upper[i] = final_upper[i - 1]

        # Final Lower Band — only tighten; reset if prev close was below it
        if lb > final_lower[i - 1] or prev_cl < final_lower[i - 1]:
            final_lower[i] = lb
        else:
            final_lower[i] = final_lower[i - 1]

        # Direction flip logic
        prev_st = supertrend[i - 1]
        if np.isnan(prev_st):
            direction[i]  = -1.0
            supertrend[i] = final_upper[i]
        elif prev_st == final_upper[i - 1]:      # was bearish
            if cl <= final_upper[i]:
                supertrend[i] = final_upper[i]
                direction[i]  = -1.0
            else:
                supertrend[i] = final_lower[i]
                direction[i]  = 1.0
        else:                                    # was bullish
            if cl >= final_lower[i]:
                supertrend[i] = final_lower[i]
                direction[i]  = 1.0
            else:
                supertrend[i] = final_upper[i]
                direction[i]  = -1.0

    return (
        pd.Series(supertrend, index=idx, name="Supertrend"),
        pd.Series(direction,  index=idx, name="Supertrend_Dir"),
    )


# ---------------------------------------------------------------------------
# Layer 1 — Indicator Calculation
# ---------------------------------------------------------------------------

def compute_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all 9 indicators and attach them to a copy of `data`.

    Does NOT coerce NaN — callers must use has_valid_indicators() before
    evaluating signals.
    """
    df     = data.copy()
    close  = df["Close"]
    high   = df["High"]
    low    = df["Low"]
    volume = df["Volume"]

    # RSI (14-period Wilder smoothing)
    df["RSI"] = ta.momentum.RSIIndicator(close=close, window=14).rsi()

    # Supertrend (10-period ATR, ×3 multiplier)
    df["Supertrend"], df["Supertrend_Dir"] = _compute_supertrend(high, low, close)

    # EMA 50 / 200
    df["EMA_50"]  = ta.trend.EMAIndicator(close=close, window=50).ema_indicator()
    df["EMA_200"] = ta.trend.EMAIndicator(close=close, window=200).ema_indicator()

    # MACD (12 / 26 / 9)
    _macd = ta.trend.MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
    df["MACD"]        = _macd.macd()
    df["MACD_Signal"] = _macd.macd_signal()

    # ADX (14-period)
    df["ADX"] = ta.trend.ADXIndicator(
        high=high, low=low, close=close, window=14
    ).adx()

    # Momentum — percentage change over 10 bars
    df["Momentum"] = close.pct_change(periods=10) * 100.0

    # ATR Volatility — 14-period ATR as % of close
    _atr = ta.volatility.AverageTrueRange(
        high=high, low=low, close=close, window=14
    ).average_true_range()
    df["Volatility"] = (_atr / close.replace(0, np.nan)) * 100.0

    # Volume 20-bar MA
    df["Volume_MA"] = volume.rolling(window=20, min_periods=20).mean()

    return df


# ---------------------------------------------------------------------------
# NaN guard
# ---------------------------------------------------------------------------

def has_valid_indicators(row: pd.Series) -> bool:
    """
    Return True only when ALL required indicator columns are finite.
    Must be called before evaluate_confirmations() or generate_signal().
    """
    for col in REQUIRED_INDICATOR_COLS:
        val = row.get(col)
        if val is None:
            return False
        try:
            if not np.isfinite(float(val)):
                return False
        except (TypeError, ValueError):
            return False
    return True


# ---------------------------------------------------------------------------
# Safe division helper
# ---------------------------------------------------------------------------

def _safe_div(numerator: float, denominator: float, fallback: float = 0.0) -> float:
    """Return numerator / denominator, or fallback when denominator is zero."""
    return numerator / denominator if denominator > 0.0 else fallback


# ---------------------------------------------------------------------------
# Layer 2 & 3 — Signal Evaluation
# ---------------------------------------------------------------------------

def evaluate_confirmations(row: pd.Series) -> dict:
    """
    Evaluate all 9 indicators for a single bar and separate results into
    mandatory (Layer 2) and optional (Layer 3) layers.

    Pre-condition : has_valid_indicators(row) is True.

    Returns
    -------
    dict keys:
        mandatory          : {condition: bool}  — 4 mandatory conditions
        optional           : {condition: bool}  — 5 optional conditions
        mandatory_all_met  : bool               — True when all 4 pass
        mandatory_count    : int  (0–4)
        optional_count     : int  (0–5)
        readings           : {condition: str}   — formatted numeric values
        reasons            : {condition: str}   — contextual explanations
    """
    close    = float(row["Close"])
    rsi      = float(row["RSI"])
    st_line  = float(row["Supertrend"])
    st_dir   = float(row["Supertrend_Dir"])
    ema50    = float(row["EMA_50"])
    ema200   = float(row["EMA_200"])
    macd     = float(row["MACD"])
    macd_sig = float(row["MACD_Signal"])
    mom      = float(row["Momentum"])
    vol_pct  = float(row["Volatility"])
    adx      = float(row["ADX"])
    volume   = float(row["Volume"])
    vol_ma   = float(row["Volume_MA"])

    # ── Layer 2: Mandatory conditions ────────────────────────────────────
    rsi_pass  = 55.0 < rsi < 70.0
    st_pass   = st_dir > 0.0                # Supertrend bullish (direction = 1)
    e200_pass = close > ema200
    macd_pass = macd > macd_sig
    adx_pass  = adx > 25.0

    mandatory = {
        C_RSI:    rsi_pass,
        C_ST:     st_pass,
        C_EMA200: e200_pass,
        C_MACD:   macd_pass,
        C_ADX:    adx_pass,
    }
    mandatory_count   = int(sum(mandatory.values()))
    mandatory_all_met = mandatory_count >= 4   # eligible for BUY or STRONG BUY

    # ── Layer 3: Optional conditions ─────────────────────────────────────
    mom_pass   = mom > 1.0
    vol_pass   = vol_pct < 6.0
    ema50_pass = close > ema50
    vol_above  = volume > vol_ma

    optional = {
        C_MOM:    mom_pass,
        C_VOL:    vol_pass,
        C_EMA50:  ema50_pass,
        C_VOLUME: vol_above,
    }
    optional_count = int(sum(optional.values()))

    # ── Readings (actual numeric values) ─────────────────────────────────
    st_label  = "↑ Bullish" if st_pass else "↓ Bearish"
    e200_diff = (_safe_div(close, ema200, 1.0) - 1) * 100
    e50_diff  = (_safe_div(close, ema50,  1.0) - 1) * 100
    vol_ratio = _safe_div(volume, vol_ma, 0.0)

    readings = {
        C_RSI:    f"{rsi:.2f}  (target 55–70)",
        C_ST:     f"ST {st_line:.4f}  |  Price {close:.4f}  |  {st_label}",
        C_EMA200: f"Price {close:.4f}  |  EMA-200 {ema200:.4f}  ({e200_diff:+.2f}%)",
        C_MACD:   f"MACD {macd:.4f}  |  Signal {macd_sig:.4f}  |  Δ {macd-macd_sig:+.4f}",
        C_ADX:    f"{adx:.2f}  (target > 25)",
        C_MOM:    f"{mom:+.2f}%  (target > +1%)",
        C_VOL:    f"{vol_pct:.2f}%  (target < 6%)",
        C_EMA50:  f"Price {close:.4f}  |  EMA-50 {ema50:.4f}  ({e50_diff:+.2f}%)",
        C_VOLUME: f"Vol {volume:,.0f}  |  MA-20 {vol_ma:,.0f}  (×{vol_ratio:.2f})",
    }

    # ── Reasons (contextual, pass-aware) ─────────────────────────────────
    if rsi_pass:
        rsi_reason = (
            f"RSI {rsi:.1f} is in the momentum sweet-spot (55–70): "
            "strong bullish momentum without being overbought."
        )
    elif rsi < 55:
        rsi_reason = (
            f"RSI {rsi:.1f} is below 55 — insufficient bullish momentum. "
            "Wait for buyers to push RSI into the 55–70 zone."
        )
    else:
        rsi_reason = (
            f"RSI {rsi:.1f} exceeds 70 — market is overbought. "
            "Risk of a short-term pullback before continuation."
        )

    st_reason = (
        f"Supertrend line at {st_line:.4f}. "
        + (f"Price ({close:.4f}) is above the Supertrend — uptrend confirmed. "
           "This is a strong structural buy signal."
           if st_pass else
           f"Price ({close:.4f}) has crossed below the Supertrend — downtrend in force. "
           "No long entries until price reclaims the Supertrend line.")
    )

    e200_reason = (
        f"Price is {e200_diff:+.2f}% relative to EMA-200. "
        + ("Above EMA-200 confirms the long-term bull structure is intact."
           if e200_pass else
           "Price is below EMA-200 — the long-term trend is bearish. Elevated downside risk.")
    )

    macd_delta  = macd - macd_sig
    macd_reason = (
        f"MACD Δ is {macd_delta:+.4f}. "
        + ("MACD above Signal confirms positive short-term momentum and a bullish crossover."
           if macd_pass else
           "MACD is below Signal — short-term momentum is negative. Avoid new long entries.")
    )

    adx_reason = (
        f"ADX reads {adx:.1f}. "
        + ("A reading above 25 confirms a strong, directional trend — mandatory for entry."
           if adx_pass else
           "ADX below 25 indicates a weak or ranging market. Entry blocked until trend strengthens.")
    )

    mom_reason = (
        f"10-bar price change is {mom:+.2f}%. "
        + ("Short-term price action is accelerating upward — confirms trend direction."
           if mom_pass else
           "Insufficient price gain over 10 bars. The move lacks short-term conviction.")
    )

    vol_reason = (
        f"ATR is {vol_pct:.2f}% of price. "
        + ("Market conditions are calm — low noise reduces whipsaw risk on entry."
           if vol_pass else
           "Elevated ATR signals choppy conditions. Wait for volatility to subside.")
    )

    e50_reason = (
        f"Price is {e50_diff:+.2f}% relative to EMA-50. "
        + ("Trading above EMA-50 confirms the medium-term uptrend is intact."
           if ema50_pass else
           "Price has crossed below EMA-50 — medium-term trend is no longer supportive.")
    )

    volume_reason = (
        f"Volume is {vol_ratio:.2f}× its 20-bar average. "
        + ("Above-average volume confirms strong market participation in the move."
           if vol_above else
           "Below-average volume suggests weak conviction. Low-volume moves are less reliable.")
    )

    reasons = {
        C_RSI:    rsi_reason,
        C_ST:     st_reason,
        C_EMA200: e200_reason,
        C_MACD:   macd_reason,
        C_ADX:    adx_reason,
        C_MOM:    mom_reason,
        C_VOL:    vol_reason,
        C_EMA50:  e50_reason,
        C_VOLUME: volume_reason,
    }

    return {
        "mandatory":         mandatory,
        "optional":          optional,
        "mandatory_all_met": mandatory_all_met,
        "mandatory_count":   mandatory_count,
        "optional_count":    optional_count,
        "readings":          readings,
        "reasons":           reasons,
    }


# ---------------------------------------------------------------------------
# Layer 2 — Signal Generation
# ---------------------------------------------------------------------------

def generate_signal(regime: str, mandatory_count: int) -> str:
    """
    Return signal based on regime and how many mandatory indicators pass:
      • STRONG BUY — Bull Run + all 5 mandatory pass
      • BUY        — Bull Run + exactly 4 mandatory pass
      • CASH       — anything else (regime not Bull, or < 4 mandatory)
    """
    if regime == REGIME_BULL:
        if mandatory_count >= 5:
            return SIGNAL_STRONG_BUY
        if mandatory_count >= 4:
            return SIGNAL_BUY
    return SIGNAL_CASH


def get_latest_signal(df_with_indicators: pd.DataFrame) -> tuple[str, dict]:
    """
    Return (signal, confirmations_dict) for the most recent fully-valid bar.
    Scans backwards to skip any trailing NaN rows.
    """
    for i in range(len(df_with_indicators) - 1, -1, -1):
        row = df_with_indicators.iloc[i]
        if has_valid_indicators(row):
            confs = evaluate_confirmations(row)
            sig   = generate_signal(row["Regime"], confs["mandatory_count"])
            return sig, confs

    # Fallback — dataset too short for any valid bar
    return SIGNAL_CASH, {
        "mandatory":         dict.fromkeys(MANDATORY_CONDITIONS, False),
        "optional":          dict.fromkeys(OPTIONAL_CONDITIONS,  False),
        "mandatory_all_met": False,
        "mandatory_count":   0,
        "optional_count":    0,
        "readings":          {},
        "reasons":           {},
    }
