"""
llm_analyzer.py — Claude-powered trade decision analysis.

Sends current market state (regime, indicators, backtest metrics) to
Claude claude-opus-4-6 and receives a structured JSON recommendation:
  decision       : "Strong Buy" | "Strong Sell" | "Hold/Wait"
  holding_period : e.g. "2-5 days"
  confidence     : 0-100
  reason         : 2-3 sentence narrative
  key_risks      : list of 2-3 risk factors
"""

from __future__ import annotations

import os
import json
import logging

from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger(__name__)


def get_trade_recommendation(
    ticker: str,
    regime: str,
    signal: str,
    indicators: dict,
    metrics: dict,
    recent_price: float,
) -> dict:
    """
    Call Claude claude-opus-4-6 to generate a detailed trade recommendation.

    Parameters
    ----------
    ticker        : instrument symbol
    regime        : current HMM regime label
    signal        : "LONG" or "CASH"
    indicators    : dict from evaluate_confirmations()
    metrics       : backtest results dict
    recent_price  : latest closing price

    Returns
    -------
    dict with keys: decision, holding_period, confidence, reason, key_risks
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return _fallback(regime, signal)

    prompt = _build_prompt(ticker, regime, signal, indicators, metrics, recent_price)

    try:
        from anthropic import Anthropic
        client = Anthropic(api_key=api_key)

        msg = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )

        text = msg.content[0].text.strip()
        start, end = text.find("{"), text.rfind("}") + 1
        result = json.loads(text[start:end])
        return result

    except Exception as exc:
        log.warning("LLM analysis failed: %s", exc)
        return _fallback(regime, signal)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_prompt(ticker, regime, signal, indicators, metrics, price) -> str:
    def _fmt(v, decimals=2):
        try:
            return f"{float(v):.{decimals}f}"
        except Exception:
            return str(v)

    above_ema50  = "Above" if indicators.get("Above_EMA50")  else "Below"
    above_ema200 = "Above" if indicators.get("Above_EMA200") else "Below"
    macd_bias    = "Bullish" if indicators.get("MACD_bullish") else "Bearish"
    confirmations = indicators.get("total_confirmations", 0)

    return f"""You are a quantitative trading analyst. Based on the data below, provide a concise trading recommendation.

INSTRUMENT  : {ticker}
PRICE       : {_fmt(price, 4)}
HMM REGIME  : {regime}
STRATEGY    : {signal}  ({confirmations}/8 confirmations)

INDICATORS
  RSI          : {_fmt(indicators.get("RSI", "N/A"))}
  ADX          : {_fmt(indicators.get("ADX", "N/A"))}
  Momentum     : {_fmt(indicators.get("Momentum", "N/A"))} %
  Volatility   : {_fmt(indicators.get("Volatility", "N/A"))} % (annualised)
  vs EMA-50    : {above_ema50}
  vs EMA-200   : {above_ema200}
  MACD bias    : {macd_bias}

BACKTEST (historical)
  Total Return : {_fmt(metrics.get("total_return", 0))} %
  Alpha        : {_fmt(metrics.get("alpha", 0))} %
  Win Rate     : {_fmt(metrics.get("win_rate", 0))} %
  Max Drawdown : {_fmt(metrics.get("max_drawdown", 0))} %
  Trades       : {metrics.get("num_trades", 0)}

Reply with ONLY a JSON object with these exact keys:
  "decision"       : "Strong Buy" | "Strong Sell" | "Hold/Wait"
  "holding_period" : suggested holding window (e.g. "2-5 days", "1-2 weeks")
  "confidence"     : integer 0-100
  "reason"         : 2-3 sentence narrative explaining the decision
  "key_risks"      : array of 2-3 main risk factors as strings
"""


def _fallback(regime: str, signal: str) -> dict:
    if signal == "LONG":
        decision = "Strong Buy"
        reason = (
            f"The market is in a {regime} regime with all major technical "
            "confirmations aligned. Risk-reward favours a long entry."
        )
    else:
        decision = "Hold/Wait"
        reason = (
            f"Current regime is {regime}. Insufficient confirmations to enter. "
            "Staying in cash preserves capital for a better setup."
        )

    return {
        "decision":       decision,
        "holding_period": "2-5 days",
        "confidence":     55,
        "reason":         reason,
        "key_risks":      ["Regime reversal", "Macro shock", "Liquidity gap"],
    }
