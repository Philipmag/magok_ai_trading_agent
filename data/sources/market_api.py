"""
Market Data API Source.

Mock implementation for market price and volume data.
"""

import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import numpy as np
from collections import deque


@dataclass
class PriceQuote:
    """Represents a single price quote."""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    bid: float
    ask: float


@dataclass
class MarketSnapshot:
    """Container for market data snapshot."""
    timestamp: datetime
    quotes: Dict[str, PriceQuote] = field(default_factory=dict)
    market_sentiment: float = 0.5  # 0-1 scale
    vix_level: float = 20.0


class MarketAPISource:
    """
    Mock market data API.
    
    In production, this would connect to real market data providers like:
    - Alpaca
    - Polygon.io
    - Alpha Vantage
    - Yahoo Finance
    """
    
    def __init__(self, use_mock: bool = True, symbols: List[str] = None):
        self.use_mock = use_mock
        self.symbols = symbols or ["FDX", "UPS", "CHRW", "DAL", "IYT"]
        
        # Price simulation state
        self._prices: Dict[str, float] = {
            "FDX": 250.0, "UPS": 180.0, "CHRW": 140.0,
            "DAL": 55.0, "IYT": 220.0
        }
        self._price_histories: Dict[str, deque] = {
            s: deque(maxlen=200) for s in self.symbols
        }
        self._volume_histories: Dict[str, deque] = {
            s: deque(maxlen=200) for s in self.symbols
        }
        
        # Initialize histories
        for symbol in self.symbols:
            base_price = self._prices.get(symbol, 100.0)
            for i in range(100):
                price = base_price * (1 + random.uniform(-0.02, 0.02))
                self._price_histories[symbol].append(price)
                self._volume_histories[symbol].append(int(random.uniform(1000000, 5000000)))
    
    def fetch_quote(self, symbol: str) -> Optional[PriceQuote]:
        """Fetch current quote for a symbol."""
        if self.use_mock:
            return self._generate_mock_quote(symbol)
        else:
            return self._fetch_real_quote(symbol)
    
    def fetch_market_data(self) -> MarketSnapshot:
        """Fetch current market data for all symbols."""
        quotes = {}
        for symbol in self.symbols:
            quote = self.fetch_quote(symbol)
            if quote:
                quotes[symbol] = quote
        
        # Calculate market sentiment
        sentiment = self._calculate_sentiment()
        
        return MarketSnapshot(
            timestamp=datetime.now(),
            quotes=quotes,
            market_sentiment=sentiment,
            vix_level=round(random.uniform(15, 30), 2)
        )
    
    def _generate_mock_quote(self, symbol: str) -> PriceQuote:
        """Generate a realistic mock price quote."""
        now = datetime.now()
        
        # Get or initialize base price
        base_price = self._prices.get(symbol, 100.0)
        
        # Simulate price movement
        trend = np.sin(time.time() / 300) * 0.001  # Gentle trend
        noise = np.random.normal(0, 0.001)
        volatility = 0.002
        
        price_change = base_price * (trend + noise)
        new_price = base_price + price_change
        new_price = max(base_price * 0.9, min(base_price * 1.1, new_price))
        self._prices[symbol] = new_price
        
        # Generate OHLC
        open_price = base_price * (1 + random.uniform(-0.001, 0.001))
        high_price = max(open_price, new_price) * (1 + random.uniform(0, volatility))
        low_price = min(open_price, new_price) * (1 - random.uniform(0, volatility))
        close_price = new_price
        
        # Generate volume
        base_volume = 2000000
        volume = int(base_volume * random.uniform(0.5, 2.0))
        
        # Bid/ask spread
        spread = close_price * 0.0001
        bid = close_price - spread
        ask = close_price + spread
        
        # Store in history
        self._price_histories[symbol].append(close_price)
        self._volume_histories[symbol].append(volume)
        
        return PriceQuote(
            symbol=symbol,
            timestamp=now,
            open=round(open_price, 2),
            high=round(high_price, 2),
            low=round(low_price, 2),
            close=round(close_price, 2),
            volume=volume,
            bid=round(bid, 2),
            ask=round(ask, 2)
        )
    
    def _fetch_real_quote(self, symbol: str) -> Optional[PriceQuote]:
        """
        Placeholder for real API integration.
        
        In production, implement actual API calls here.
        """
        raise NotImplementedError("Real API not implemented. Set use_mock=True.")
    
    def _calculate_sentiment(self) -> float:
        """Calculate market sentiment based on price trends."""
        sentiments = []
        for symbol in self.symbols:
            history = list(self._price_histories[symbol])
            if len(history) >= 10:
                # Simple momentum indicator
                recent = np.mean(history[-5:])
                older = np.mean(history[-10:-5])
                sentiment = 0.5 + (recent - older) / older * 10
                sentiment = max(0.0, min(1.0, sentiment))
                sentiments.append(sentiment)
        
        return round(np.mean(sentiments) if sentiments else 0.5, 3)
    
    def get_price_history(self, symbol: str, periods: int = 50) -> List[float]:
        """Get historical prices for a symbol."""
        history = list(self._price_histories.get(symbol, []))
        return history[-periods:] if len(history) >= periods else history
    
    def get_volume_history(self, symbol: str, periods: int = 50) -> List[int]:
        """Get historical volumes for a symbol."""
        history = list(self._volume_histories.get(symbol, []))
        return history[-periods:] if len(history) >= periods else history
    
    def get_returns(self, symbol: str, periods: int = 20) -> List[float]:
        """Calculate returns for a symbol."""
        prices = self.get_price_history(symbol, periods + 1)
        if len(prices) < 2:
            return []
        returns = []
        for i in range(1, len(prices)):
            ret = (prices[i] - prices[i-1]) / prices[i-1]
            returns.append(ret)
        return returns


# Singleton instance
_market_source = None

def get_market_source(use_mock: bool = True, symbols: List[str] = None) -> MarketAPISource:
    """Get or create the market source singleton."""
    global _market_source
    if _market_source is None:
        _market_source = MarketAPISource(use_mock=use_mock, symbols=symbols)
    return _market_source
