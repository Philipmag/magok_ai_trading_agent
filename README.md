# 🤖 AI Trading Agent

> Event-driven AI trading system that reads logistics signals to make smarter market decisions.

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Status](https://img.shields.io/badge/Status-Paper%20Trading%20Only-yellow?style=flat-square)](https://github.com/Philipmag/magok_ai_trading_agent)

---

## Overview

Most trading bots react to price data alone. This system goes a step further — it ingests real-world shipping and logistics signals (route deviations, port congestion, supply chain anomalies) and combines them with market data to detect trading opportunities before they show up in price charts.

Built as a learning project to explore the intersection of event-driven systems, machine learning, and financial engineering. The system runs entirely in paper-trading mode, meaning no real money is ever at risk.

---

## Demo

> **Status: Paper trading only — no live brokerage connection.**

Run it locally to see the trading loop in action:

```bash
git clone https://github.com/Philipmag/magok_ai_trading_agent
cd magok_ai_trading_agent
pip install -r requirements.txt
python main.py
```

Sample output:
```
[INFO] Event detected: SHIPPING_DELAY on AAPL route (severity: HIGH)
[INFO] ML model selected strategy: momentum (confidence: 0.82)
[INFO] Risk check passed — executing paper trade: BUY AAPL @ $182.40, qty: 5
[INFO] Trade logged. Open positions: 1 / 3
```

---

## Features

- **Logistics-aware signal detection** — Flags shipping anomalies (delays, route changes, congestion) as early market signals before they appear in price data.
- **ML strategy selection** — A RandomForest classifier picks the best trading strategy (momentum, mean reversion, or latency arbitrage) based on the current feature vector.
- **Built-in risk management** — Hard limits: max 1% capital per trade, max 3 concurrent positions, 3% daily loss cap. The system refuses to trade outside these bounds.
- **Structured JSON logging** — Every decision, prediction, and trade is logged with full context for post-hoc analysis.
- **Paper trading only** — A simulated broker executes all trades safely, making this safe to run without any brokerage account.

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.x |
| ML Model | scikit-learn (RandomForestClassifier) |
| Data Processing | Pandas, NumPy |
| Logging | python-json-logger |
| Execution | Simulated paper broker (no live API) |

---

## Getting Started

```bash
git clone https://github.com/Philipmag/magok_ai_trading_agent
cd magok_ai_trading_agent
pip install -r requirements.txt
cp .env.example .env   # configure API keys if using live data sources
python main.py
```

All data sources default to mock mode (`use_mock_data: True` in `config/settings.py`). No API keys are required to run.

---

## How It Works

The system runs as a continuous loop with five stages:

1. **Data Ingestion** → Pulls mock shipping, market, and news data every 10–30 seconds via `data/ingest.py`.
2. **Event Detection** → `events/detector.py` scans for anomalies (delays, price breakouts, news spikes) using a rule engine.
3. **Feature Engineering** → `features/builder.py` combines event data with technical indicators (RSI, moving averages, volume spikes) into a structured feature vector.
4. **ML Prediction** → `ml/predict.py` feeds the feature vector into a trained RandomForest model, which returns a strategy recommendation with a confidence score.
5. **Risk-Gated Execution** → `risk/manager.py` validates the trade against hard limits before `execution/broker.py` places a paper order.

---

## What I Learned

- **Event-driven architecture** requires careful state management — learned to use a clean pub/sub pattern to decouple data ingestion from decision logic.
- **ML in trading is mostly feature engineering** — model accuracy improved significantly once shipping delay severity was weighted correctly relative to price signals.
- **Risk management is not optional** — adding hard position limits and daily loss caps transformed the system from a toy into something that behaves like a real trading desk.

---

## Roadmap

- [ ] Connect to a real paper trading API (Alpaca or Polygon.io) to replace mock data with live feeds.
- [ ] Add a backtesting module to evaluate strategy performance on historical data.
- [ ] Build a lightweight web dashboard to visualize open positions and decision logs in real time.

---

> **Disclaimer:** This is a paper trading system for educational purposes only. Do not use with real money.
