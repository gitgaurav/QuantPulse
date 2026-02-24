"""
data_loader.py — Fetch OHLCV data for any stock or index.

Primary source : yfinance (hourly, last 730 days)
Secondary role : Claude LLM for ticker validation and symbol suggestions
"""

import os
import json
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_ohlcv(ticker: str, period_days: int = 730, interval: str = "1h") -> pd.DataFrame:
    """
    Fetch OHLCV data from Yahoo Finance.

    Parameters
    ----------
    ticker      : e.g. 'AAPL', '^NSEI', 'RELIANCE.NS', 'BTC-USD'
    period_days : look-back window in calendar days (max 730 for 1-hour data)
    interval    : yfinance interval string ('1h', '1d', '15m', …)

    Returns
    -------
    pd.DataFrame with columns [Open, High, Low, Close, Volume]
    and a timezone-aware DatetimeIndex.
    """
    end_dt = datetime.utcnow()
    start_dt = end_dt - timedelta(days=min(period_days, 729))  # yfinance hard cap

    raw = yf.download(
        ticker,
        start=start_dt.strftime("%Y-%m-%d"),
        end=end_dt.strftime("%Y-%m-%d"),
        interval=interval,
        auto_adjust=True,
        progress=False,
        threads=False,
    )

    if raw.empty:
        raise ValueError(
            f"No data returned for '{ticker}'. "
            "Check the symbol and try a different interval."
        )

    # yfinance may return a MultiIndex when downloading a single ticker
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    # Keep standard OHLCV columns only
    raw = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    raw.index = pd.to_datetime(raw.index)
    raw = raw.dropna()

    if len(raw) < 300:
        raise ValueError(
            f"Insufficient data for '{ticker}' "
            f"({len(raw)} rows). Need at least 300 hourly bars."
        )

    return raw


def validate_ticker_with_llm(ticker: str) -> dict:
    """
    Ask Claude whether a ticker symbol is valid and return basic metadata.

    Returns a dict with keys:
        valid    : bool
        name     : str  (company / index name)
        type     : 'stock' | 'index' | 'etf' | 'crypto' | 'unknown'
        exchange : str
        currency : str
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return {"valid": True, "name": ticker, "type": "unknown",
                "exchange": "unknown", "currency": "USD"}

    try:
        from anthropic import Anthropic
        client = Anthropic(api_key=api_key)

        msg = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=256,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Is '{ticker}' a valid stock, index, ETF, or crypto ticker symbol "
                        "tradeable on any major exchange? Reply with a single JSON object "
                        "containing: valid (bool), name (string), type "
                        "(stock/index/etf/crypto/unknown), exchange (string), "
                        "currency (string). No explanation — only the JSON."
                    ),
                }
            ],
        )

        text = msg.content[0].text.strip()
        start, end = text.find("{"), text.rfind("}") + 1
        return json.loads(text[start:end])

    except Exception:
        return {"valid": True, "name": ticker, "type": "unknown",
                "exchange": "unknown", "currency": "USD"}
