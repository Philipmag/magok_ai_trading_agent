# AI Trading Agent

A modular, event-driven AI trading system that uses public shipping/logistics signals and market data to generate and execute trades in a safe, rule-based, and compliant way.

## Features

- **Data Ingestion**: Real-time data from shipping/logistics, market prices, and news sources
- **Event Detection**: Detects anomalies like route deviations, delays, congestion spikes, price breakouts
- **Feature Engineering**: Combines event + market data into structured feature vectors with technical indicators
- **ML Decision Layer**: RandomForest classifier selects the best trading strategy (momentum, latency, mean reversion)
- **Risk Management**: Max 1% capital per trade, max 3 concurrent trades, daily loss cap (3%)
- **Paper Trading**: Simulated broker for safe execution
- **Structured Logging**: JSON-formatted logs for all decisions and trades

## Project Structure

```
ai_trading_agent/
├── config/              # Configuration settings
│   ├── settings.py      # Trading parameters
│   └── assets.py        # Asset definitions
├── data/                # Data ingestion
│   ├── ingest.py        # Unified data interface
│   └── sources/         # Mock data sources
│       ├── shipping_api.py
│       ├── market_api.py
│       └── news_api.py
├── events/              # Event detection
│   ├── detector.py       # Anomaly detection
│   └── rules.py         # Rule engine
├── features/            # Feature engineering
│   ├── builder.py       # Feature vectors
│   └── indicators.py    # Technical indicators
├── ml/                  # Machine learning
│   ├── model.py         # Strategy classifier
│   ├── predict.py       # Prediction engine
│   └── train.py         # Model training
├── strategies/          # Trading strategies
│   ├── momentum.py      # Momentum breakout
│   ├── latency.py       # Latency arbitrage
│   └── mean_reversion.py # Mean reversion
├── risk/               # Risk management
│   └── manager.py      # Position & risk controls
├── execution/          # Trade execution
│   ├── broker.py       # Paper trading broker
│   └── orders.py       # Order management
├── backtest/           # Backtesting
│   └── engine.py       # Strategy evaluation
├── logs/               # Log files
├── main.py             # Main trading loop
└── requirements.txt    # Dependencies
```

## Installation

```bash
pip install numpy pandas scikit-learn
```

## Usage

Run the demo trading agent:

```bash
cd ai_trading_agent
python main.py
```

The system will run for 60 seconds by default, continuously:
1. Fetching data every 10 seconds
2. Detecting shipping/logistics events
3. Building feature vectors
4. Making ML predictions
5. Applying risk checks
6. Executing paper trades
7. Logging all decisions

## Configuration

Edit [`config/settings.py`](config/settings.py) to customize:

- Initial capital
- Risk limits (max trades, daily loss cap)
- Loop intervals
- Model confidence threshold

## Trading Strategies

### Momentum Strategy
- Entry: Price breakout + volume confirmation + favorable RSI
- Exit: Stop loss, take profit, or momentum reversal

### Latency Arbitrage Strategy
- Entry: Price divergence between correlated assets after shipping event
- Exit: Correlation reestablished or convergence

### Mean Reversion Strategy
- Entry: RSI at extreme (oversold/overbought) + Bollinger Band contact
- Exit: RSI returned to neutral or price at mean

## Risk Controls

- Max 1% capital per trade
- Max 3 concurrent trades
- Daily loss cap (3%)
- Required stop-loss and take-profit

## Disclaimer

This is a paper trading system for educational purposes. Do not use with real money. Past performance does not guarantee future results.
