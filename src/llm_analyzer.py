"""
llm_analyzer.py — Claude-powered trade decision analysis.

Sends current market state (regime, signal, indicators, backtest metrics) to
Claude claude-opus-4-6 and receives a structured JSON recommendation.

Decision is anchored to the system signal:
  STRONG BUY  → LLM must return "Strong Buy"   (all 4 mandatory pass)
  BUY         → LLM must return "Buy"           (core 3: RSI + ST + ADX pass)
  CASH        → LLM must return "Hold/Wait"     (conditions not met)
"""

from __future__ import annotations

import os
import json
import logging

from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger(__name__)

# Map system signals to required LLM decision values
_SIGNAL_TO_DECISION = {
    "STRONG BUY": "Strong Buy",
    "BUY":        "Buy",
    "CASH":       "Hold/Wait",
}


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

    The decision field in the response is anchored to the system signal —
    the LLM provides reasoning and risk narrative, not the direction.

    Parameters
    ----------
    ticker        : instrument symbol
    regime        : current HMM regime label
    signal        : "STRONG BUY" | "BUY" | "CASH"
    indicators    : dict with RSI, ADX, mandatory_count, etc.
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

        # Enforce decision alignment with system signal regardless of LLM output
        required_decision = _SIGNAL_TO_DECISION.get(signal, "Hold/Wait")
        result["decision"] = required_decision

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

    # Indicator readings
    rsi        = _fmt(indicators.get("RSI", "N/A"))
    adx        = _fmt(indicators.get("ADX", "N/A"))
    momentum   = _fmt(indicators.get("Momentum", "N/A"))
    volatility = _fmt(indicators.get("Volatility", "N/A"))
    above_ema50  = "Above" if indicators.get("Above_EMA50")  else "Below"
    above_ema200 = "Above" if indicators.get("Above_EMA200") else "Below"
    macd_bias    = "Bullish" if indicators.get("MACD_bullish")      else "Bearish"
    st_bias      = "Bullish" if indicators.get("Supertrend_bullish") else "Bearish"
    m_count      = indicators.get("mandatory_count", 0)
    core_3       = indicators.get("core_3_met", False)

    # Required decision and instructions based on system signal
    required_decision = _SIGNAL_TO_DECISION.get(signal, "Hold/Wait")

    if signal == "STRONG BUY":
        signal_context = (
            f"The QuantPulse system has generated a STRONG BUY signal. "
            f"All 4 mandatory indicators pass: RSI is in the 55–70 momentum zone, "
            f"Supertrend is bullish, price is above EMA-200, and ADX confirms a strong trend. "
            f"Your decision MUST be \"{required_decision}\". "
            f"Explain why these conditions justify a high-conviction long entry."
        )
    elif signal == "BUY":
        signal_context = (
            f"The QuantPulse system has generated a BUY signal. "
            f"Core 3 indicators pass (RSI, Supertrend, ADX) but EMA-200 is not confirmed, "
            f"indicating a valid but moderately cautious entry. "
            f"Your decision MUST be \"{required_decision}\". "
            f"Acknowledge the EMA-200 caveat in your reasoning."
        )
    else:
        signal_context = (
            f"The QuantPulse system signal is CASH — entry conditions are not met "
            f"({m_count}/4 mandatory indicators pass, core 3 met: {core_3}). "
            f"Your decision MUST be \"{required_decision}\". "
            f"Explain why staying in cash is prudent."
        )

    return f"""You are a quantitative trading analyst. Provide a concise trade recommendation based on the data below.

INSTRUMENT  : {ticker}
PRICE       : {_fmt(price, 4)}
HMM REGIME  : {regime}
SYSTEM SIGNAL : {signal}  ({m_count}/4 mandatory indicators pass)

TECHNICAL INDICATORS
  RSI              : {rsi}  (entry zone: 55–70)
  ADX              : {adx}  (trend strength; >25 = strong)
  Supertrend       : {st_bias}
  vs EMA-200       : {above_ema200}
  vs EMA-50        : {above_ema50}
  MACD bias        : {macd_bias}
  Momentum (10-bar): {momentum} %
  ATR Volatility   : {volatility} %

BACKTEST PERFORMANCE (historical)
  Total Return : {_fmt(metrics.get("total_return", 0))} %
  Alpha vs B&H : {_fmt(metrics.get("alpha", 0))} %
  Win Rate     : {_fmt(metrics.get("win_rate", 0))} %
  Max Drawdown : {_fmt(metrics.get("max_drawdown", 0))} %
  Trades       : {metrics.get("num_trades", 0)}

INSTRUCTION
{signal_context}

Reply with ONLY a valid JSON object (no markdown, no extra text):
{{
  "decision"       : "{required_decision}",
  "holding_period" : "<suggested window, e.g. 2-5 days or 1-2 weeks>",
  "confidence"     : <integer 0-100>,
  "reason"         : "<2-3 sentence narrative explaining the decision>",
  "key_risks"      : ["<risk 1>", "<risk 2>", "<risk 3>"]
}}
"""


def _fallback(regime: str, signal: str) -> dict:
    """Rule-based fallback when the LLM is unavailable."""
    if signal == "STRONG BUY":
        decision = "Strong Buy"
        confidence = 82
        reason = (
            f"All 4 mandatory indicators confirmed in a {regime} regime: "
            "RSI in momentum zone, Supertrend bullish, price above EMA-200, and ADX signals a strong trend. "
            "Risk-reward strongly favours a long entry."
        )
    elif signal == "BUY":
        decision = "Buy"
        confidence = 65
        reason = (
            f"Core indicators (RSI, Supertrend, ADX) are aligned in a {regime} regime. "
            "EMA-200 confirmation is absent, adding some long-term trend risk. "
            "Entry is valid but position sizing should reflect the elevated uncertainty."
        )
    else:
        decision = "Hold/Wait"
        confidence = 40
        reason = (
            f"Current regime is {regime} and mandatory entry conditions are not fully met. "
            "Staying in cash preserves capital for a higher-probability setup."
        )

    return {
        "decision":       decision,
        "holding_period": "2-5 days" if signal != "CASH" else "N/A",
        "confidence":     confidence,
        "reason":         reason,
        "key_risks":      ["Regime reversal", "Macro shock", "Liquidity gap"],
    }
