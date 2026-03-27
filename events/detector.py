"""
Event Detection Module.

Detects anomalies and significant events from data sources.
"""

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
import numpy as np


class EventType(Enum):
    """Types of detectable events."""
    ROUTE_DEVIATION = "route_deviation"
    PORT_DELAY = "port_delay"
    CONGESTION_SPIKE = "congestion_spike"
    PRICE_BREAKOUT = "price_breakout"
    VOLUME_SPIKE = "volume_spike"
    NEGATIVE_NEWS = "negative_news"
    POSITIVE_NEWS = "positive_news"
    SENTIMENT_SHIFT = "sentiment_shift"
    CORRELATION_BREAK = "correlation_break"


@dataclass
class DetectedEvent:
    """Represents a detected event."""
    event_id: str
    event_type: EventType
    timestamp: datetime
    severity: float  # 0-1
    symbol: Optional[str] = None
    description: str = ""
    raw_data: Dict = field(default_factory=dict)
    
    # Event metadata
    confidence: float = 1.0
    source: str = "detector"
    
    def __post_init__(self):
        if not self.event_id:
            self.event_id = f"{self.event_type.value}-{self.timestamp.strftime('%Y%m%d%H%M%S')}"


@dataclass
class EventSignal:
    """Trading signal derived from events."""
    symbol: str
    direction: str  # "long", "short", "neutral"
    strength: float  # 0-1
    event_types: List[EventType] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    reason: str = ""


class EventDetector:
    """
    Detects events and anomalies from combined data.
    
    Analyzes:
    - Shipping/logistics disruptions
    - Price and volume anomalies
    - News sentiment changes
    - Correlation breaks
    """
    
    def __init__(self, symbols: List[str] = None):
        self.symbols = symbols or ["FDX", "UPS", "CHRW", "DAL", "IYT"]
        self._event_counter = 0
        self._last_prices: Dict[str, float] = {}
        self._last_volumes: Dict[str, int] = {}
        self._baseline_congestion: float = 0.5
        
    def detect_all(self, data) -> List[DetectedEvent]:
        """Detect all events from combined data."""
        events = []
        
        # Shipping events
        shipping_events = self._detect_shipping_events(data)
        events.extend(shipping_events)
        
        # Market events
        market_events = self._detect_market_events(data)
        events.extend(market_events)
        
        # News events
        news_events = self._detect_news_events(data)
        events.extend(news_events)
        
        # Correlation events
        correlation_events = self._detect_correlation_events(data)
        events.extend(correlation_events)
        
        return events
    
    def _detect_shipping_events(self, data) -> List[DetectedEvent]:
        """Detect shipping/logistics events."""
        events = []
        shipping_data = data.shipping_data
        
        # Port congestion spike
        if shipping_data.port_congestion_level > 0.7:
            self._event_counter += 1
            events.append(DetectedEvent(
                event_id=f"SE-{datetime.now().strftime('%Y%m%d%H%M%S')}-{self._event_counter}",
                event_type=EventType.CONGESTION_SPIKE,
                timestamp=datetime.now(),
                severity=min(1.0, (shipping_data.port_congestion_level - 0.7) / 0.3),
                description=f"High port congestion: {shipping_data.port_congestion_level:.1%}",
                raw_data={"congestion_level": shipping_data.port_congestion_level},
                source="shipping"
            ))
        
        # Delay threshold
        if shipping_data.average_delay_hours > 24:
            self._event_counter += 1
            events.append(DetectedEvent(
                event_id=f"SE-{datetime.now().strftime('%Y%m%d%H%M%S')}-{self._event_counter}",
                event_type=EventType.PORT_DELAY,
                timestamp=datetime.now(),
                severity=min(1.0, (shipping_data.average_delay_hours - 24) / 48),
                description=f"Average delay: {shipping_data.average_delay_hours:.1f} hours",
                raw_data={"delay_hours": shipping_data.average_delay_hours},
                source="shipping"
            ))
        
        # Individual shipping events from data
        for shipping_event in shipping_data.events:
            self._event_counter += 1
            events.append(DetectedEvent(
                event_id=f"SE-{datetime.now().strftime('%Y%m%d%H%M%S')}-{self._event_counter}",
                event_type=EventType.ROUTE_DEVIATION,
                timestamp=shipping_event.timestamp,
                severity=shipping_event.severity,
                description=shipping_event.description,
                raw_data={"event": shipping_event.__dict__},
                source="shipping"
            ))
        
        return events
    
    def _detect_market_events(self, data) -> List[DetectedEvent]:
        """Detect price and volume events."""
        events = []
        
        for symbol in self.symbols:
            quote = data.quotes.get(symbol)
            if not quote:
                continue
            
            # Price breakout detection
            if symbol in self._last_prices:
                price_change = (quote.close - self._last_prices[symbol]) / self._last_prices[symbol]
                
                # Significant price move (>2%)
                if abs(price_change) > 0.02:
                    self._event_counter += 1
                    events.append(DetectedEvent(
                        event_id=f"ME-{datetime.now().strftime('%Y%m%d%H%M%S')}-{self._event_counter}",
                        event_type=EventType.PRICE_BREAKOUT,
                        timestamp=datetime.now(),
                        severity=min(1.0, abs(price_change) * 10),
                        symbol=symbol,
                        description=f"{symbol} {'up' if price_change > 0 else 'down'} {abs(price_change):.2%}",
                        raw_data={
                            "price_change": price_change,
                            "old_price": self._last_prices[symbol],
                            "new_price": quote.close
                        },
                        source="market"
                    ))
            
            # Volume spike detection
            if symbol in self._last_volumes:
                volume_ratio = quote.volume / max(1, self._last_volumes[symbol])
                
                if volume_ratio > 2.0:  # 2x average volume
                    self._event_counter += 1
                    events.append(DetectedEvent(
                        event_id=f"ME-{datetime.now().strftime('%Y%m%d%H%M%S')}-{self._event_counter}",
                        event_type=EventType.VOLUME_SPIKE,
                        timestamp=datetime.now(),
                        severity=min(1.0, (volume_ratio - 2.0) / 3.0),
                        symbol=symbol,
                        description=f"{symbol} volume {volume_ratio:.1f}x average",
                        raw_data={
                            "volume_ratio": volume_ratio,
                            "current_volume": quote.volume,
                            "average_volume": self._last_volumes[symbol]
                        },
                        source="market"
                    ))
            
            # Update state
            self._last_prices[symbol] = quote.close
            self._last_volumes[symbol] = quote.volume
        
        return events
    
    def _detect_news_events(self, data) -> List[DetectedEvent]:
        """Detect news-based events."""
        events = []
        
        for article in data.news_data.articles:
            if article.relevance_score < 0.3:
                continue
            
            self._event_counter += 1
            event_type = EventType.NEGATIVE_NEWS if article.sentiment_score < -0.2 else EventType.POSITIVE_NEWS
            
            if article.sentiment_score > -0.2 and article.sentiment_score < 0.2:
                continue  # Skip neutral news
            
            events.append(DetectedEvent(
                event_id=f"NE-{datetime.now().strftime('%Y%m%d%H%M%S')}-{self._event_counter}",
                event_type=event_type,
                timestamp=article.timestamp,
                severity=abs(article.sentiment_score) * article.relevance_score,
                symbol=article.related_symbols[0] if article.related_symbols else None,
                description=article.headline,
                raw_data={"article_id": article.article_id, "sentiment": article.sentiment_score},
                confidence=article.relevance_score,
                source="news"
            ))
        
        return events
    
    def _detect_correlation_events(self, data) -> List[DetectedEvent]:
        """Detect correlation breaks between assets."""
        events = []
        
        # Check correlation between shipping index and logistics stocks
        shipping_index = data.shipping_data.global_shipping_index
        
        # Get FDX as representative
        fdx_quote = data.quotes.get("FDX")
        if fdx_quote and fdx_quote.close > 0:
            # Normalized price (assume base 250)
            normalized_price = fdx_quote.close / 250.0
            normalized_shipping = shipping_index / 1500.0
            
            # Correlation ratio
            ratio = normalized_price / max(0.1, normalized_shipping)
            
            # Significant divergence (>20%)
            if ratio > 1.2 or ratio < 0.8:
                self._event_counter += 1
                events.append(DetectedEvent(
                    event_id=f"CE-{datetime.now().strftime('%Y%m%d%H%M%S')}-{self._event_counter}",
                    event_type=EventType.CORRELATION_BREAK,
                    timestamp=datetime.now(),
                    severity=abs(ratio - 1.0) * 2.5,
                    symbol="FDX",
                    description=f"Correlation break: shipping index {ratio:.2f}x stock price",
                    raw_data={"ratio": ratio, "shipping_index": shipping_index},
                    source="correlation"
                ))
        
        return events
    
    def generate_signals(self, events: List[DetectedEvent]) -> List[EventSignal]:
        """Convert detected events into trading signals."""
        signals = []
        
        # Group events by symbol
        symbol_events: Dict[str, List[DetectedEvent]] = {}
        for event in events:
            if event.symbol:
                if event.symbol not in symbol_events:
                    symbol_events[event.symbol] = []
                symbol_events[event.symbol].append(event)
        
        # Generate signals
        for symbol, evts in symbol_events.items():
            if not evts:
                continue
            
            # Calculate composite signal
            long_signals = sum(1 for e in evts if e.event_type in [
                EventType.POSITIVE_NEWS,
                EventType.PRICE_BREAKOUT
            ] and e.raw_data.get("price_change", 0) > 0)
            
            short_signals = sum(1 for e in evts if e.event_type in [
                EventType.NEGATIVE_NEWS,
                EventType.PRICE_BREAKOUT
            ] and e.raw_data.get("price_change", 0) < 0)
            
            short_signals += sum(1 for e in evts if e.event_type in [
                EventType.CONGESTION_SPIKE,
                EventType.PORT_DELAY,
                EventType.ROUTE_DEVIATION
            ])
            
            avg_severity = sum(e.severity for e in evts) / len(evts)
            
            if long_signals > short_signals:
                direction = "long"
                strength = min(1.0, (long_signals - short_signals) / len(evts) + avg_severity * 0.5)
            elif short_signals > long_signals:
                direction = "short"
                strength = min(1.0, (short_signals - long_signals) / len(evts) + avg_severity * 0.5)
            else:
                direction = "neutral"
                strength = 0.3
            
            signals.append(EventSignal(
                symbol=symbol,
                direction=direction,
                strength=strength,
                event_types=[e.event_type for e in evts],
                reason=f"Based on {len(evts)} events"
            ))
        
        return signals


# Singleton
_detector_instance = None

def get_event_detector(symbols: List[str] = None) -> EventDetector:
    """Get or create the event detector singleton."""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = EventDetector(symbols=symbols)
    return _detector_instance
