# QuantPulse — Regime-Based Algorithmic Trading Dashboard

A professional Python trading dashboard that combines **Hidden Markov Models (HMM)** for market-regime detection, an **8-confirmation technical voting system**, and **Claude AI (claude-opus-4-6)** for intelligent trade-decision analysis.

---

## Features

| Feature | Details |
|---|---|
| Regime detection | GaussianHMM with 7 hidden states trained on Returns, Range & Volume Volatility |
| Signal generation | Bull regime + ≥7 / 8 technical confirmations required |
| Risk management | 2.5× leverage simulation · 48-hour cooldown · Bear-regime forced exit |
| Backtest engine | ₹5,00,000 starting capital · full trade log · equity curve · alpha vs B&H |
| AI decision | Claude claude-opus-4-6 provides decision (Strong Buy / Sell / Wait), holding period & rationale |
| Dashboard | Streamlit + Plotly candlestick chart with colour-coded regime backgrounds |

---

## Project Structure

```
QuantPulse/
├── app.py                      # Streamlit dashboard (entry point)
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variable template
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # yfinance data fetcher + Claude ticker validation
│   ├── regime_engine.py        # HMM regime detection (Bull / Bear / Neutral)
│   ├── strategy.py             # 8-confirmation voting strategy
│   ├── backtester.py           # Walk-forward backtest simulation
│   └── llm_analyzer.py         # Claude AI trade recommendation
│
└── assets/
    └── requirements/
        └── product_requirements.txt
```

---

## Architecture

```
yfinance (OHLCV hourly)
        │
        ▼
  data_loader.py  ──────────────────►  Claude LLM (ticker validation)
        │
        ▼
 regime_engine.py  (GaussianHMM × 7)
  Features: log-returns | (H-L)/C | vol-vol
  Labels:   Bull Run | Bear/Crash | Neutral
        │
        ▼
   strategy.py  (8 confirmations)
  RSI · Momentum · Volatility · ADX
  EMA50 · EMA200 · MACD · Volume
        │
        ▼
  backtester.py  (₹5L capital · 2.5× leverage · 48h cooldown)
        │
        ▼
  llm_analyzer.py  ───► Claude claude-opus-4-6  ───► decision + reason
        │
        ▼
     app.py  (Streamlit dashboard)
```

---

## Prerequisites

- Python 3.10+
- An [Anthropic API key](https://console.anthropic.com) (for the AI decision section)

---

## Installation

### 1. Clone / navigate to the project

```bash
cd /path/to/QuantPulse
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the API key

Copy the example env file and fill in your key:

```bash
cp .env.example .env
```

Open `.env` and replace the placeholder:

```
ANTHROPIC_API_KEY=sk-ant-api03-...your-real-key-here...
```

> **Note:** The dashboard runs without an API key, but the AI Decision section will show a fallback message instead of a Claude-generated recommendation.

---

## Running the Dashboard

```bash
streamlit run app.py
```

The dashboard opens at `http://localhost:8501` in your browser.

---

## Using the Dashboard

1. **Enter a ticker** in the left sidebar.
   - US stocks: `AAPL`, `TSLA`, `MSFT`
   - US indices: `^GSPC` (S&P 500), `^NDX` (Nasdaq 100)
   - Indian NSE: `RELIANCE.NS`, `TCS.NS`, `^NSEI`
   - Crypto: `BTC-USD`, `ETH-USD`

2. Click **▶ Run Analysis**. The app will:
   - Fetch the last 730 days of hourly OHLCV data
   - Train the HMM regime model
   - Compute technical indicators
   - Run the backtest simulation
   - Call Claude AI for a trade recommendation

3. **Review the results:**

   | Section | Description |
   |---|---|
   | Signal chip | Current trading signal: LONG (in position) or CASH |
   | Regime chip | Detected market regime: Bull Run / Bear/Crash / Neutral |
   | Candlestick chart | Green background = Bull, Red = Bear, Grey = Neutral |
   | Performance metrics | Total Return · Alpha · Win Rate · Max Drawdown |
   | AI Decision | Strong Buy / Strong Sell / Hold, holding period, reason, risks |
   | Confirmation breakdown | Which of the 8 conditions are currently satisfied |
   | Equity curve | Portfolio value over time (leveraged) |
   | Trade log | Every simulated trade with entry/exit/PnL |

---

## Strategy Details

### Regime Detection (HMM)
- **Model:** `hmmlearn.GaussianHMM` — 7 components, full covariance
- **Features** (3, standardised):
  1. Log return of hourly close
  2. Normalised intrabar range `(High − Low) / Close`
  3. 5-bar rolling std of log-volume changes
- **Labels:** Bull Run = state with highest mean return · Bear/Crash = lowest mean return

### Entry Rules
Enter **LONG** only when **all** of the following are true:
1. HMM regime = **Bull Run**
2. At least **7 out of 8** confirmations:

| # | Condition | Rationale |
|---|---|---|
| 1 | 55 < RSI < 70 | Momentum sweet-spot, not overbought |
| 2 | Momentum > +1 % | 10-bar price change trending up |
| 3 | Annualised Vol < 6 % | Market is calm |
| 4 | ADX > 25 | Strong established trend |
| 5 | Price > EMA-50 | Medium-term trend intact |
| 6 | Price > EMA-200 | Long-term trend intact |
| 7 | MACD > Signal | Short-term momentum positive |
| 8 | Volume > 20-bar MA | Volume confirms the move |

### Exit Rules
- **Regime flip** to Bear/Crash → immediate close
- **End of data** → close at last available price

### Risk Parameters
| Parameter | Value |
|---|---|
| Starting capital | ₹5,00,000 |
| Leverage | 2.5× (applied to PnL) |
| Cooldown after exit | 48 hours |
| Maximum loss per trade | −100% of capital (no margin call) |

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | Yes (for AI section) | Anthropic API key for Claude |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `No data returned for ticker` | Check symbol format; NSE tickers need `.NS` suffix |
| `Insufficient data` | Try a more liquid symbol with longer history |
| `LLM analysis unavailable` | Set `ANTHROPIC_API_KEY` in `.env` |
| `HMM convergence warning` | Normal for noisy data; results are still valid |
| `AttributeError: module 'ta'` | Run `pip install --upgrade ta` |

---

## Disclaimer

QuantPulse is a research and educational tool. It does **not** constitute financial advice. Past simulated performance is not indicative of future results. Always conduct your own due diligence before trading with real capital.


#### How to execute:
<!-- /* python3 -m venv .venv 
source .venv/bin/activate

#pip install -r requirements.txt
.venv/bin/python3 -m pip install --upgrade pip

streamlit run app.p

http://localhost:8501

*/ -->