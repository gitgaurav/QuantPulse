"""
backtester.py — Walk-forward simulation of the QuantPulse strategy.

Parameters
  Starting capital : ₹5,00,000 (5 Lakhs INR)
  Leverage         : 2.5×  (applied to PnL)
  Cooldown         : 48 hours after any exit before re-entry is allowed
  Exit rule        : immediate close when regime flips to Bear/Crash

NaN safety
  A bar is only eligible for entry/exit evaluation when ALL 8 indicator
  columns hold finite values (has_valid_indicators() must return True).
  Bars that fail this check are recorded in the equity curve with the
  current capital value but skipped for trade logic.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from dataclasses import dataclass
from datetime import timedelta
from typing import Optional

from src.regime_engine import RegimeEngine, REGIME_BEAR
from src.strategy import (
    compute_indicators,
    evaluate_confirmations,
    generate_signal,
    has_valid_indicators,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
INITIAL_CAPITAL: float = 500_000.0   # ₹5 Lakhs INR
LEVERAGE: float = 2.5
COOLDOWN_HOURS: int = 48

# Skip the first N bars so every indicator has a full lookback window.
# EMA-200 needs 200 bars; MACD signal needs ~35; we use 250 for a safe margin.
WARMUP_BARS: int = 250


# ---------------------------------------------------------------------------
# Trade record
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    entry_time:  pd.Timestamp
    entry_price: float
    direction:   str = "LONG"

    exit_time:   Optional[pd.Timestamp] = None
    exit_price:  Optional[float]        = None
    exit_reason: Optional[str]          = None
    pnl_pct:     Optional[float]        = None   # leveraged return as fraction

    def close(self, exit_time: pd.Timestamp, exit_price: float, reason: str) -> float:
        """Close the trade and return the leveraged PnL fraction."""
        self.exit_time   = exit_time
        self.exit_price  = exit_price
        self.exit_reason = reason
        if self.entry_price > 0.0:
            raw_ret = (exit_price - self.entry_price) / self.entry_price
            self.pnl_pct = max(raw_ret * LEVERAGE, -1.0)
        else:
            self.pnl_pct = 0.0
        return self.pnl_pct

    def to_dict(self) -> dict:
        return {
            "Entry Time":  self.entry_time,
            "Exit Time":   self.exit_time,
            "Entry Price": round(self.entry_price, 4),
            "Exit Price":  round(self.exit_price, 4) if self.exit_price is not None else None,
            "Direction":   self.direction,
            "PnL %":       round(self.pnl_pct * 100, 2) if self.pnl_pct is not None else None,
            "Exit Reason": self.exit_reason,
        }


# ---------------------------------------------------------------------------
# Backtester
# ---------------------------------------------------------------------------

class Backtester:
    """Run a full backtest and return metrics + trade log + equity curve."""

    def __init__(self):
        self.regime_engine = RegimeEngine()
        self.trades: list[Trade] = []
        self.equity_curve: pd.Series = pd.Series(dtype=float)

    # ------------------------------------------------------------------

    def run(self, data: pd.DataFrame) -> dict:
        """
        Execute the backtest on `data` (OHLCV, hourly).

        Returns
        -------
        dict with keys:
            total_return, bh_return, alpha, win_rate, max_drawdown,
            final_capital, num_trades, trade_log (DataFrame),
            equity_curve (Series), df_with_indicators (DataFrame)
        """
        # 1. Fit HMM on the full dataset and get per-bar regime labels
        self.regime_engine.fit(data)
        regime_series = self.regime_engine.get_regime_series(data)

        # 2. Compute all 8 technical indicators
        df = compute_indicators(data)
        df["Regime"] = regime_series

        # 3. Walk-forward simulation loop
        capital          = INITIAL_CAPITAL
        capital_at_entry = INITIAL_CAPITAL
        position: Optional[Trade]          = None
        last_exit_time: Optional[pd.Timestamp] = None
        self.trades = []

        equity_values: list[float]        = []
        equity_times:  list[pd.Timestamp] = []

        for i in range(WARMUP_BARS, len(df)):
            row    = df.iloc[i]
            ts     = df.index[i]
            price  = float(row["Close"])
            regime = str(row["Regime"])

            # ── Mark-to-market equity (unrealised PnL while in position) ──
            if position is not None and position.entry_price > 0.0:
                raw_ret = (price - position.entry_price) / position.entry_price
                current_equity = max(capital_at_entry * (1.0 + raw_ret * LEVERAGE), 0.0)
            else:
                current_equity = capital

            equity_values.append(current_equity)
            equity_times.append(ts)

            # ── Skip bars where any indicator is invalid ──────────────────
            # We still record equity above so the curve has no gaps,
            # but we do NOT trade on incomplete data.
            if not has_valid_indicators(row):
                continue

            # ── Exit: regime flipped to Bear/Crash ────────────────────────
            if position is not None and regime == REGIME_BEAR:
                pnl = position.close(ts, price, "Regime → Bear/Crash")
                capital = max(capital_at_entry * (1.0 + pnl), 0.0)
                self.trades.append(position)
                last_exit_time = ts
                position = None
                equity_values[-1] = capital   # overwrite MTM with realised value
                continue

            # ── Cooldown: too soon after last exit ────────────────────────
            in_cooldown = (
                last_exit_time is not None
                and (ts - last_exit_time) < timedelta(hours=COOLDOWN_HOURS)
            )
            if in_cooldown or position is not None:
                continue

            # ── Entry: all 4 mandatory indicators must pass ───────────────
            confs  = evaluate_confirmations(row)
            signal = generate_signal(regime, confs["mandatory_all_met"])

            if signal == "LONG":
                position         = Trade(entry_time=ts, entry_price=price)
                capital_at_entry = capital

        # ── Close any still-open position at the last bar ─────────────────
        if position is not None:
            last_row = df.iloc[-1]
            pnl = position.close(
                df.index[-1], float(last_row["Close"]), "End of Data"
            )
            capital = max(capital_at_entry * (1.0 + pnl), 0.0)
            self.trades.append(position)

        self.equity_curve = pd.Series(equity_values, index=equity_times, name="Equity")

        return self._compute_metrics(data, df, capital)

    # ------------------------------------------------------------------

    def _compute_metrics(
        self,
        raw_data: pd.DataFrame,
        df: pd.DataFrame,
        final_capital: float,
    ) -> dict:
        total_return = (
            (final_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100.0
            if INITIAL_CAPITAL > 0.0 else 0.0
        )

        # Buy-and-hold benchmark (same period as the backtest)
        first_close = float(raw_data["Close"].iloc[0])
        last_close  = float(raw_data["Close"].iloc[-1])
        bh_ret = (
            (last_close - first_close) / first_close * 100.0
            if first_close > 0.0 else 0.0
        )
        alpha = total_return - bh_ret

        # Win-rate
        closed = [t for t in self.trades if t.pnl_pct is not None]
        if closed:
            wins     = sum(1 for t in closed if t.pnl_pct > 0)
            win_rate = wins / len(closed) * 100.0
        else:
            win_rate = 0.0

        # Maximum drawdown from equity curve
        eq      = self.equity_curve.values.astype(float)
        peak    = np.maximum.accumulate(eq)
        denominator = np.where(peak == 0, 1.0, peak)
        dd      = (eq - peak) / denominator * 100.0
        max_drawdown = float(dd.min())

        trade_log = (
            pd.DataFrame([t.to_dict() for t in closed])
            if closed
            else pd.DataFrame()
        )

        return {
            "total_return":       round(total_return, 2),
            "bh_return":          round(bh_ret, 2),
            "alpha":              round(alpha, 2),
            "win_rate":           round(win_rate, 2),
            "max_drawdown":       round(max_drawdown, 2),
            "final_capital":      round(final_capital, 2),
            "num_trades":         len(closed),
            "trade_log":          trade_log,
            "equity_curve":       self.equity_curve,
            "df_with_indicators": df,       # used by app for chart/signal display
        }
