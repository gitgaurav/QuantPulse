"""
strategy.py — 8-Confirmation Voting Strategy.

Entry rule:
  HMM regime == "Bull Run"
  AND ALL 8 indicators are valid (non-NaN)
  AND at least 7 out of 8 technical conditions are satisfied.

The 8 conditions
────────────────
  1. RSI in (55, 70)           — momentum sweet-spot, not overbought
  2. Momentum  > +1 %          — 10-bar price change trending up
  3. Volatility (ATR%) < 6 %   — intrabar ATR as % of close is subdued
  4. ADX > 25                  — trend is strong
  5. Price > EMA 50            — medium-term trend intact
  6. Price > EMA 200           — long-term trend intact
  7. MACD > Signal Line        — short-term momentum positive
  8. Volume > 20-bar MA        — volume confirming the move

Volatility note
───────────────
  We use 14-period ATR normalised by Close price (ATR%) rather than an
  annualised rolling std.  For hourly bars:
    • Normal market : 0.1 – 2 %   → condition passes
    • Earnings / crash : 3 – 15 % → condition blocks entry
  The 6 % threshold therefore filters genuinely chaotic periods while
  letting steady bull runs through.
"""

import numpy as np
import pandas as pd
import ta

from src.regime_engine import REGIME_BULL

MIN_CONFIRMATIONS = 7
SIGNAL_LONG = "LONG"
SIGNAL_CASH = "CASH"

# Condition labels — used as dict keys in evaluate_confirmations
C_RSI      = "RSI in (55–70)"
C_MOM      = "Momentum > 1%"
C_VOL      = "ATR Volatility < 6%"
C_ADX      = "ADX > 25"
C_EMA50    = "Price > EMA 50"
C_EMA200   = "Price > EMA 200"
C_MACD     = "MACD > Signal"
C_VOLUME   = "Volume > MA(20)"

# All columns that must be present and non-NaN before we can evaluate signals
REQUIRED_INDICATOR_COLS = [
    "RSI", "Momentum", "Volatility", "ADX",
    "EMA_50", "EMA_200", "MACD", "MACD_Signal", "Volume_MA",
]


# ---------------------------------------------------------------------------
# Indicator computation
# ---------------------------------------------------------------------------

def compute_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Append all required technical indicators to a copy of `data`.

    Input  : OHLCV DataFrame (hourly, any timezone)
    Output : same DataFrame with extra indicator columns — no NaN coercion,
             callers must check has_valid_indicators() before evaluating.
    """
    df = data.copy()
    close  = df["Close"]
    high   = df["High"]
    low    = df["Low"]
    volume = df["Volume"]

    # ── RSI (14-period Wilder smoothing) ──────────────────────────────────
    df["RSI"] = ta.momentum.RSIIndicator(close=close, window=14).rsi()

    # ── EMAs ─────────────────────────────────────────────────────────────
    df["EMA_50"]  = ta.trend.EMAIndicator(close=close, window=50).ema_indicator()
    df["EMA_200"] = ta.trend.EMAIndicator(close=close, window=200).ema_indicator()

    # ── MACD (12 / 26 / 9) ───────────────────────────────────────────────
    _macd = ta.trend.MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
    df["MACD"]        = _macd.macd()
    df["MACD_Signal"] = _macd.macd_signal()

    # ── ADX (14-period) ───────────────────────────────────────────────────
    df["ADX"] = ta.trend.ADXIndicator(
        high=high, low=low, close=close, window=14
    ).adx()

    # ── Momentum — percentage change over 10 bars ─────────────────────────
    df["Momentum"] = close.pct_change(periods=10) * 100.0

    # ── Volatility — 14-period Average True Range as % of Close ──────────
    # ATR% gives the average intrabar range relative to price.
    # Typical values: 0.2–2 % (calm) → 3–15 % (crisis/earnings).
    # The threshold of 6 % cleanly separates normal from chaotic regimes.
    _atr = ta.volatility.AverageTrueRange(
        high=high, low=low, close=close, window=14
    ).average_true_range()
    df["Volatility"] = (_atr / close.replace(0, np.nan)) * 100.0

    # ── Volume 20-bar moving average ─────────────────────────────────────
    df["Volume_MA"] = volume.rolling(window=20, min_periods=20).mean()

    return df


# ---------------------------------------------------------------------------
# NaN guard
# ---------------------------------------------------------------------------

def has_valid_indicators(row: pd.Series) -> bool:
    """
    Return True only when every required indicator column exists and holds
    a finite (non-NaN, non-Inf) numeric value.

    This must be checked before calling evaluate_confirmations() or
    generate_signal() to guarantee all 8 conditions are evaluated on real data.
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
# Signal evaluation
# ---------------------------------------------------------------------------

def evaluate_confirmations(row: pd.Series) -> dict:
    """
    Evaluate all 8 conditions for a single bar.

    Pre-condition : has_valid_indicators(row) is True.

    Returns
    -------
    dict with:
        • one bool per named condition
        • 'total_confirmations'  (int, 0–8)
        • 'values'               (dict of actual indicator readings for display)
    """
    close    = float(row["Close"])
    rsi      = float(row["RSI"])
    mom      = float(row["Momentum"])
    vol_pct  = float(row["Volatility"])
    adx      = float(row["ADX"])
    ema50    = float(row["EMA_50"])
    ema200   = float(row["EMA_200"])
    macd     = float(row["MACD"])
    macd_sig = float(row["MACD_Signal"])
    volume   = float(row["Volume"])
    vol_ma   = float(row["Volume_MA"])

    conditions = {
        C_RSI:    55.0 < rsi < 70.0,
        C_MOM:    mom > 1.0,
        C_VOL:    vol_pct < 6.0,
        C_ADX:    adx > 25.0,
        C_EMA50:  close > ema50,
        C_EMA200: close > ema200,
        C_MACD:   macd > macd_sig,
        C_VOLUME: volume > vol_ma,
    }

    total = int(sum(conditions.values()))

    # Actual readings for the dashboard breakdown table
    readings = {
        C_RSI:    f"{rsi:.2f}  (target 55–70)",
        C_MOM:    f"{mom:+.2f}%  (target > 1%)",
        C_VOL:    f"{vol_pct:.2f}%  (target < 6%)",
        C_ADX:    f"{adx:.2f}  (target > 25)",
        C_EMA50:  f"{close:.4f} vs {ema50:.4f}  ({(close/ema50-1)*100:+.2f}%)",
        C_EMA200: f"{close:.4f} vs {ema200:.4f}  ({(close/ema200-1)*100:+.2f}%)",
        C_MACD:   f"MACD {macd:.4f}  Sig {macd_sig:.4f}  Δ{macd-macd_sig:+.4f}",
        C_VOLUME: f"{volume:,.0f} vs MA {vol_ma:,.0f}  (×{volume/vol_ma:.2f})",
    }

    return {
        **conditions,
        "total_confirmations": total,
        "readings": readings,
    }


def generate_signal(regime: str, total_confirmations: int) -> str:
    """Return LONG only when regime is Bull AND all indicators checked and
    confirmations reach the minimum threshold; otherwise CASH."""
    if regime == REGIME_BULL and total_confirmations >= MIN_CONFIRMATIONS:
        return SIGNAL_LONG
    return SIGNAL_CASH


def get_latest_signal(df_with_indicators: pd.DataFrame) -> tuple[str, dict]:
    """
    Return (signal, confirmations_dict) for the most recent valid bar.

    Scans backwards from the last row to find the first bar where all
    indicators are valid — guarantees the caller always gets a fully
    evaluated result rather than a partially-NaN snapshot.
    """
    for i in range(len(df_with_indicators) - 1, -1, -1):
        row = df_with_indicators.iloc[i]
        if has_valid_indicators(row):
            confs = evaluate_confirmations(row)
            sig = generate_signal(row["Regime"], confs["total_confirmations"])
            return sig, confs

    # Fallback if no valid bar found (extremely short dataset)
    empty_confs = {
        C_RSI: False, C_MOM: False, C_VOL: False, C_ADX: False,
        C_EMA50: False, C_EMA200: False, C_MACD: False, C_VOLUME: False,
        "total_confirmations": 0,
        "readings": {},
    }
    return SIGNAL_CASH, empty_confs
