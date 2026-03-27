"""
Feature Builder Module.

Builds feature vectors from raw data and events.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime
import numpy as np

from .indicators import TechnicalIndicators, calculate_all_indicators
from events.detector import DetectedEvent, EventType
from data.ingest import CombinedDataSnapshot


@dataclass
class FeatureVector:
    """Represents a complete feature vector for a symbol."""
    symbol: str
    timestamp: datetime
    
    # Price features
    price: float
    price_change_short: float  # 5-period %
    price_change_medium: float  # 20-period %
    price_change_long: float  # 50-period %
    
    # Volume features
    volume: int
    volume_ratio: float
    volume_ma: float
    
    # Technical indicators
    rsi: float
    macd: float
    macd_histogram: float
    bollinger_position: float  # 0-1 position within bands
    
    # Event features
    event_severity: float  # Max severity of recent events
    time_since_event: float  # Seconds since last event
    event_count: int  # Number of recent events
    
    # Sentiment features
    news_sentiment: float  # -1 to 1
    market_sentiment: float  # 0-1
    shipping_sentiment: float  # 0-1
    
    # Shipping impact
    shipping_impact: float
    port_congestion: float
    avg_delay: float
    
    # Derived
    momentum_score: float
    mean_reversion_score: float
    trend_strength: float
    
    # Raw data for reference
    raw_indicators: TechnicalIndicators = None
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for ML models."""
        return np.array([
            self.price_change_short,
            self.price_change_medium,
            self.volume_ratio,
            self.rsi,
            self.macd_histogram,
            self.event_severity,
            self.time_since_event,
            self.news_sentiment,
            self.shipping_impact,
            self.momentum_score,
            self.mean_reversion_score,
            self.trend_strength,
        ])


class FeatureBuilder:
    """
    Builds feature vectors from combined data.
    
    Combines:
    - Market data (prices, volumes)
    - Technical indicators
    - Event data (severity, recency)
    - Sentiment data
    - Shipping/logistics data
    """
    
    def __init__(self, symbols: List[str] = None):
        self.symbols = symbols or ["FDX", "UPS", "CHRW", "DAL", "IYT"]
        self._price_histories: Dict[str, List[float]] = {s: [] for s in self.symbols}
        self._volume_histories: Dict[str, List[int]] = {s: [] for s in self.symbols}
        self._recent_events: Dict[str, List[DetectedEvent]] = {s: [] for s in self.symbols}
        self._max_event_history = 50
    
    def build_features(self, data: CombinedDataSnapshot) -> Dict[str, FeatureVector]:
        """Build feature vectors for all symbols."""
        features = {}
        
        for symbol in self.symbols:
            quote = data.quotes.get(symbol)
            if not quote:
                continue
            
            # Update price/volume histories
            self._update_histories(symbol, quote.close, quote.volume)
            
            # Get histories
            prices = self._price_histories[symbol]
            volumes = self._volume_histories[symbol]
            
            # Calculate technical indicators
            indicators = calculate_all_indicators(prices, volumes)
            
            # Get event features
            event_features = self._get_event_features(symbol, data)
            
            # Get sentiment features
            sentiment = self._get_sentiment_features(symbol, data)
            
            # Get shipping features
            shipping_features = self._get_shipping_features(symbol, data)
            
            # Calculate derived scores
            momentum_score = self._calculate_momentum_score(indicators, event_features)
            mean_reversion_score = self._calculate_mean_reversion_score(indicators)
            trend_strength = self._calculate_trend_strength(indicators, prices)
            
            # Bollinger position
            bollinger_pos = 0.5
            if indicators.bollinger_upper != indicators.bollinger_lower:
                bollinger_pos = (quote.close - indicators.bollinger_lower) / (
                    indicators.bollinger_upper - indicators.bollinger_lower
                )
            
            features[symbol] = FeatureVector(
                symbol=symbol,
                timestamp=datetime.now(),
                
                # Price features
                price=quote.close,
                price_change_short=self._calculate_price_change(prices, 5),
                price_change_medium=self._calculate_price_change(prices, 20),
                price_change_long=self._calculate_price_change(prices, 50),
                
                # Volume features
                volume=quote.volume,
                volume_ratio=indicators.volume_ratio,
                volume_ma=np.mean(volumes[-20:]) if len(volumes) >= 20 else np.mean(volumes),
                
                # Technical features
                rsi=indicators.rsi,
                macd=indicators.macd,
                macd_histogram=indicators.macd_histogram,
                bollinger_position=round(bollinger_pos, 3),
                
                # Event features
                event_severity=event_features["max_severity"],
                time_since_event=event_features["time_since_event"],
                event_count=event_features["event_count"],
                
                # Sentiment features
                news_sentiment=sentiment["news"],
                market_sentiment=data.market_sentiment,
                shipping_sentiment=data.shipping_sentiment,
                
                # Shipping features
                shipping_impact=shipping_features["impact"],
                port_congestion=data.shipping_data.port_congestion_level,
                avg_delay=data.shipping_data.average_delay_hours,
                
                # Derived scores
                momentum_score=round(momentum_score, 3),
                mean_reversion_score=round(mean_reversion_score, 3),
                trend_strength=round(trend_strength, 3),
                
                # Raw indicators
                raw_indicators=indicators
            )
        
        return features
    
    def _update_histories(self, symbol: str, price: float, volume: int):
        """Update price and volume histories."""
        self._price_histories[symbol].append(price)
        self._volume_histories[symbol].append(volume)
        
        # Keep limited history
        max_history = 200
        if len(self._price_histories[symbol]) > max_history:
            self._price_histories[symbol] = self._price_histories[symbol][-max_history:]
        if len(self._volume_histories[symbol]) > max_history:
            self._volume_histories[symbol] = self._volume_histories[symbol][-max_history:]
    
    def _calculate_price_change(self, prices: List[float], periods: int) -> float:
        """Calculate price change percentage."""
        if len(prices) < periods:
            return 0.0
        old_price = prices[-periods]
        if old_price == 0:
            return 0.0
        return round((prices[-1] - old_price) / old_price * 100, 2)
    
    def _get_event_features(self, symbol: str, data: CombinedDataSnapshot) -> Dict:
        """Extract event features for a symbol."""
        # Get events related to this symbol
        symbol_events = [
            e for e in data.active_events
            if hasattr(e, 'symbol') and e.symbol == symbol
        ]
        
        # Also include high-severity shipping events
        high_severity_events = [
            e for e in data.shipping_data.events
            if e.severity > 0.5
        ]
        
        all_events = symbol_events + high_severity_events
        
        # Update recent events
        self._recent_events[symbol].extend(all_events)
        if len(self._recent_events[symbol]) > self._max_event_history:
            self._recent_events[symbol] = self._recent_events[symbol][-self._max_event_history:]
        
        # Calculate features
        max_severity = max([e.severity for e in all_events], default=0.0)
        event_count = len(all_events)
        
        # Time since most recent event
        time_since_event = 0.0
        if all_events:
            most_recent = max(all_events, key=lambda e: e.timestamp)
            time_since_event = (datetime.now() - most_recent.timestamp).total_seconds()
        
        return {
            "max_severity": max_severity,
            "event_count": event_count,
            "time_since_event": time_since_event
        }
    
    def _get_sentiment_features(self, symbol: str, data: CombinedDataSnapshot) -> Dict:
        """Extract sentiment features for a symbol."""
        # News sentiment for this symbol
        news_sentiment = 0.0
        for article in data.news_data.articles:
            if symbol in article.related_symbols:
                news_sentiment = article.sentiment_score
                break
        
        return {
            "news": news_sentiment,
            "market": data.market_sentiment,
            "shipping": data.shipping_sentiment
        }
    
    def _get_shipping_features(self, symbol: str, data: CombinedDataSnapshot) -> Dict:
        """Extract shipping impact for a symbol."""
        # Correlation impact based on symbol
        impact_weights = {
            "FDX": 0.9, "UPS": 0.85, "CHRW": 0.95, "DAL": 0.5, "IYT": 0.8
        }
        base_impact = impact_weights.get(symbol, 0.5)
        
        # Adjust by current conditions
        congestion_factor = 1.0 + data.shipping_data.port_congestion_level * 0.5
        
        return {
            "impact": round(base_impact * congestion_factor, 3)
        }
    
    def _calculate_momentum_score(self, indicators: TechnicalIndicators, 
                                   event_features: Dict) -> float:
        """Calculate momentum score (0-1)."""
        score = 0.0
        
        # RSI contribution
        if indicators.rsi > 50:
            score += (indicators.rsi - 50) / 50 * 0.3
        
        # MACD histogram
        if indicators.macd_histogram > 0:
            score += min(1.0, indicators.macd_histogram * 10) * 0.3
        
        # Event severity contribution
        score += event_features["max_severity"] * 0.4
        
        return min(1.0, score)
    
    def _calculate_mean_reversion_score(self, indicators: TechnicalIndicators) -> float:
        """Calculate mean reversion score (0-1)."""
        score = 0.0
        
        # Bollinger position (extreme = high score) - use internal calculation
        upper = indicators.bollinger_upper
        lower = indicators.bollinger_lower
        if upper != lower:
            # This would need current price which we don't have here
            # Use a default based on band width instead
            band_width = upper - lower
            avg_price = (upper + lower) / 2
            bollinger_extreme = min(1.0, band_width / avg_price * 5) if avg_price > 0 else 0.0
        else:
            bollinger_extreme = 0.0
        
        # RSI extreme
        if indicators.rsi < 30:
            rsi_extreme = (30 - indicators.rsi) / 30
        elif indicators.rsi > 70:
            rsi_extreme = (indicators.rsi - 70) / 30
        else:
            rsi_extreme = 0.0
        
        score = (bollinger_extreme + rsi_extreme) / 2
        
        return min(1.0, score)
    
    def _calculate_trend_strength(self, indicators: TechnicalIndicators, 
                                  prices: List[float]) -> float:
        """Calculate trend strength (0-1)."""
        if len(prices) < 20:
            return 0.0
        
        # ADX-like calculation (simplified)
        up_moves = 0
        down_moves = 0
        
        for i in range(1, min(20, len(prices))):
            diff = prices[-i] - prices[-i-1]
            if diff > 0:
                up_moves += diff
            else:
                down_moves += abs(diff)
        
        total_moves = up_moves + down_moves
        if total_moves == 0:
            return 0.0
        
        # Directional movement
        directional = abs(up_moves - down_moves) / total_moves
        
        # Combine with RSI trend
        rsi_trend = abs(indicators.rsi - 50) / 50
        
        return round((directional + rsi_trend) / 2, 3)


# Singleton
_feature_builder = None

def get_feature_builder(symbols: List[str] = None) -> FeatureBuilder:
    """Get or create the feature builder singleton."""
    global _feature_builder
    if _feature_builder is None:
        _feature_builder = FeatureBuilder(symbols=symbols)
    return _feature_builder
