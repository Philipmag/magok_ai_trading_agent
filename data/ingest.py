"""
Data Ingestion Module.

Unified interface for all data sources.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime
from threading import Lock

from .sources.shipping_api import ShippingData, get_shipping_source, ShippingAPISource
from .sources.market_api import MarketSnapshot, PriceQuote, get_market_source, MarketAPISource
from .sources.news_api import NewsSnapshot, NewsArticle, get_news_source, NewsAPISource


@dataclass
class CombinedDataSnapshot:
    """Unified data snapshot combining all sources."""
    timestamp: datetime
    
    # Market data
    market_data: MarketSnapshot
    
    # Shipping data
    shipping_data: ShippingData
    
    # News data
    news_data: NewsSnapshot
    
    # Symbol-specific quotes
    quotes: Dict[str, PriceQuote] = field(default_factory=dict)
    
    # Active shipping events
    active_events: List = field(default_factory=list)
    
    # Market metadata
    market_sentiment: float = 0.5
    shipping_sentiment: float = 0.0


class DataIngestion:
    """
    Unified data ingestion interface.
    
    Combines data from all sources and provides a single access point.
    Thread-safe for concurrent access.
    """
    
    def __init__(self, symbols: List[str] = None, use_mock: bool = True):
        self.symbols = symbols or ["FDX", "UPS", "CHRW", "DAL", "IYT"]
        self.use_mock = use_mock
        
        # Initialize data sources
        self._shipping_source = get_shipping_source(use_mock=use_mock)
        self._market_source = get_market_source(use_mock=use_mock, symbols=self.symbols)
        self._news_source = get_news_source(use_mock=use_mock)
        
        # State
        self._last_snapshot: Optional[CombinedDataSnapshot] = None
        self._lock = Lock()
        
        # Historical buffers
        self._event_history: List = []
        self._max_history = 1000
    
    def fetch_all(self) -> CombinedDataSnapshot:
        """Fetch data from all sources."""
        with self._lock:
            # Fetch from all sources
            shipping_data = self._shipping_source.fetch_shipping_data()
            market_data = self._market_source.fetch_market_data()
            news_data = self._news_source.fetch_news(symbols=self.symbols)
            
            # Combine into unified snapshot
            snapshot = CombinedDataSnapshot(
                timestamp=datetime.now(),
                market_data=market_data,
                shipping_data=shipping_data,
                news_data=news_data,
                quotes=market_data.quotes,
                active_events=shipping_data.events + news_data.articles,
                market_sentiment=market_data.market_sentiment,
                shipping_sentiment=news_data.shipping_sentiment
            )
            
            self._last_snapshot = snapshot
            
            # Update event history
            for event in shipping_data.events:
                self._add_event_to_history(event)
            
            return snapshot
    
    def fetch_market_only(self) -> MarketSnapshot:
        """Fetch market data only."""
        return self._market_source.fetch_market_data()
    
    def fetch_shipping_only(self) -> ShippingData:
        """Fetch shipping data only."""
        return self._shipping_source.fetch_shipping_data()
    
    def fetch_news_only(self) -> NewsSnapshot:
        """Fetch news data only."""
        return self._news_source.fetch_news(symbols=self.symbols)
    
    def get_last_snapshot(self) -> Optional[CombinedDataSnapshot]:
        """Get the last fetched snapshot."""
        return self._last_snapshot
    
    def get_quote(self, symbol: str) -> Optional[PriceQuote]:
        """Get current quote for a symbol."""
        if self._last_snapshot:
            return self._last_snapshot.quotes.get(symbol)
        return self._market_source.fetch_quote(symbol)
    
    def get_historical_prices(self, symbol: str, periods: int = 50) -> List[float]:
        """Get historical prices for a symbol."""
        return self._market_source.get_price_history(symbol, periods)
    
    def get_historical_volumes(self, symbol: str, periods: int = 50) -> List[int]:
        """Get historical volumes for a symbol."""
        return self._market_source.get_volume_history(symbol, periods)
    
    def get_returns(self, symbol: str, periods: int = 20) -> List[float]:
        """Get returns for a symbol."""
        return self._market_source.get_returns(symbol, periods)
    
    def get_event_history(self, limit: int = 100) -> List:
        """Get historical events."""
        return self._event_history[-limit:]
    
    def _add_event_to_history(self, event):
        """Add event to history with size limit."""
        self._event_history.append(event)
        if len(self._event_history) > self._max_history:
            self._event_history = self._event_history[-self._max_history:]
    
    def get_shipping_impact(self, symbol: str) -> float:
        """Get shipping impact factor for a symbol."""
        return self._shipping_source.get_correlation_impact(symbol)
    
    def get_news_sentiment(self, symbol: str) -> float:
        """Get news sentiment for a symbol."""
        return self._news_source.get_sentiment_for_symbol(symbol)


# Singleton instance
_ingestion_instance = None

def get_data_ingestion(symbols: List[str] = None, use_mock: bool = True) -> DataIngestion:
    """Get or create the data ingestion singleton."""
    global _ingestion_instance
    if _ingestion_instance is None:
        _ingestion_instance = DataIngestion(symbols=symbols, use_mock=use_mock)
    return _ingestion_instance
