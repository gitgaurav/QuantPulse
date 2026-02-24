"""
strategy.py — 8-Confirmation Voting Strategy.

Entry rule:
  HMM regime == "Bull Run"
  AND at least 7 out of 8 technical conditions are satisfied.

The 8 conditions
────────────────
  1. RSI in (55, 70)           — momentum sweet-spot, not overbought
  2. Momentum  > +1 %          — price trending up over last 10 bars
  3. Volatility < 6 %          — annualised hourly vol is subdued
  4. ADX > 25                  — trend is strong
  5. Price > EMA 50            — medium-term trend intact
  6. Price > EMA 200           — long-term trend intact
  7. MACD > Signal Line        — short-term momentum positive
  8. Volume > 20-bar MA        — volume confirming the move
"""

import numpy as np
import pandas as pd
import ta

from src.regime_engine import REGIME_BULL

# Approximate number of trading hours per year
# (252 trading days × 6.5 hours for US / ~6.25 hours for NSE ≈ 1638)
_ANN_FACTOR = np.sqrt(252 * 6.5)

MIN_CONFIRMATIONS = 7
SIGNAL_LONG = "LONG"
SIGNAL_CASH = "CASH"


# ---------------------------------------------------------------------------
# Indicator computation
# ---------------------------------------------------------------------------

def compute_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Append all required technical indicators to a copy of `data`.

    Input  : OHLCV DataFrame (hourly)
    Output : same DataFrame with extra indicator columns
    """
    df = data.copy()
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # RSI (14-period)
    df["RSI"] = ta.momentum.RSIIndicator(close=close, window=14).rsi()

    # EMA 50 / 200
    df["EMA_50"] = ta.trend.EMAIndicator(close=close, window=50).ema_indicator()
    df["EMA_200"] = ta.trend.EMAIndicator(close=close, window=200).ema_indicator()

    # MACD
    _macd = ta.trend.MACD(close=close)
    df["MACD"] = _macd.macd()
    df["MACD_Signal"] = _macd.macd_signal()

    # ADX (14-period)
    df["ADX"] = ta.trend.ADXIndicator(high=high, low=low, close=close, window=14).adx()

    # Momentum — percentage change over 10 bars
    df["Momentum"] = close.pct_change(periods=10) * 100.0

    # Annualised hourly volatility (20-bar rolling std of returns)
    df["Volatility"] = close.pct_change().rolling(window=20).std() * _ANN_FACTOR * 100.0

    # Volume moving average (20-bar)
    df["Volume_MA"] = volume.rolling(window=20).mean()

    return df


# ---------------------------------------------------------------------------
# Signal evaluation
# ---------------------------------------------------------------------------

def evaluate_confirmations(row: pd.Series) -> dict:
    """
    Evaluate all 8 conditions for a single bar.

    Returns
    -------
    dict with one bool per condition plus 'total_confirmations' (int).
    """
    c = {
        "RSI_in_range":      bool(55 < row["RSI"] < 70),
        "Momentum_positive": bool(row["Momentum"] > 1.0),
        "Volatility_low":    bool(row["Volatility"] < 6.0),
        "ADX_strong":        bool(row["ADX"] > 25),
        "Above_EMA50":       bool(row["Close"] > row["EMA_50"]),
        "Above_EMA200":      bool(row["Close"] > row["EMA_200"]),
        "MACD_bullish":      bool(row["MACD"] > row["MACD_Signal"]),
        "Volume_above_avg":  bool(row["Volume"] > row["Volume_MA"]),
    }
    c["total_confirmations"] = int(sum(v for v in c.values() if isinstance(v, bool)))
    return c


def generate_signal(regime: str, total_confirmations: int) -> str:
    """Return LONG if all gate conditions are met, else CASH."""
    if regime == REGIME_BULL and total_confirmations >= MIN_CONFIRMATIONS:
        return SIGNAL_LONG
    return SIGNAL_CASH


def get_latest_signal(df_with_indicators: pd.DataFrame) -> tuple[str, dict]:
    """
    Given a DataFrame that already has regime + indicator columns,
    return (signal, confirmations_dict) for the most recent bar.
    """
    row = df_with_indicators.iloc[-1]
    confs = evaluate_confirmations(row)
    sig = generate_signal(row["Regime"], confs["total_confirmations"])
    return sig, confs
