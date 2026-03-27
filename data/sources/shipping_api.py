"""
Shipping and Logistics Data Source.

Mock implementation for shipping/logistics data.
"""

import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import numpy as np


@dataclass
class ShippingEvent:
    """Represents a shipping/logistics event."""
    event_id: str
    event_type: str  # route_deviation, delay, congestion, port_congestion
    location: str
    severity: float  # 0-1
    timestamp: datetime
    affected_routes: List[str] = field(default_factory=list)
    estimated_impact: str = ""  # high, medium, low
    description: str = ""


@dataclass
class ShippingData:
    """Container for shipping data snapshot."""
    timestamp: datetime
    global_shipping_index: float
    port_congestion_level: float  # 0-1
    average_delay_hours: float
    active_vessels: int
    events: List[ShippingEvent] = field(default_factory=list)


class ShippingAPISource:
    """
    Mock shipping data API.
    
    In production, this would connect to real shipping APIs like:
    - MarineTraffic
    - FleetMon
    - Bloomberg Shipping
    - Clarksons
    """
    
    def __init__(self, use_mock: bool = True):
        self.use_mock = use_mock
        self._last_update = None
        self._base_index = 1500.0
        self._event_counter = 0
        
    def fetch_shipping_data(self) -> ShippingData:
        """Fetch current shipping data."""
        if self.use_mock:
            return self._generate_mock_data()
        else:
            return self._fetch_real_data()
    
    def _generate_mock_data(self) -> ShippingData:
        """Generate realistic mock shipping data."""
        now = datetime.now()
        
        # Simulate market movement in shipping index
        noise = np.random.normal(0, 10)
        trend = np.sin(time.time() / 3600) * 50  # Daily cycle
        self._base_index += noise + trend * 0.01
        self._base_index = max(1000, min(2000, self._base_index))
        
        # Generate random events
        events = []
        if random.random() < 0.15:  # 15% chance of event
            events.append(self._generate_random_event(now))
        
        # Port congestion simulation
        congestion = 0.3 + 0.4 * random.random()
        
        return ShippingData(
            timestamp=now,
            global_shipping_index=round(self._base_index, 2),
            port_congestion_level=round(congestion, 3),
            average_delay_hours=round(random.uniform(2, 48), 1),
            active_vessels=int(random.uniform(50000, 60000)),
            events=events
        )
    
    def _generate_random_event(self, timestamp: datetime) -> ShippingEvent:
        """Generate a random shipping event."""
        self._event_counter += 1
        
        event_types = [
            ("route_deviation", "Route Deviation Detected", 0.7),
            ("delay", "Port Congestion Delay", 0.6),
            ("congestion", "Shipping Lane Congestion", 0.5),
            ("port_congestion", "Major Port Congestion", 0.8),
            ("weather", "Weather-Related Disruption", 0.4),
        ]
        
        locations = [
            "Port of Shanghai", "Port of Los Angeles", "Port of Rotterdam",
            "Suez Canal", "Panama Canal", "Singapore Strait",
            "Hamburg Port", "Busan Port", "Dubai Port"
        ]
        
        routes = [
            ["Asia-US West Coast", "Asia-US East Coast"],
            ["Europe-Asia", "Trans-Pacific"],
            ["Trans-Atlantic", "Intra-Asia"],
            ["Middle East-Europe", "US Gulf-Asia"]
        ]
        
        event_type, desc, base_severity = random.choice(event_types)
        severity = base_severity + random.uniform(-0.2, 0.2)
        severity = max(0.1, min(1.0, severity))
        
        return ShippingEvent(
            event_id=f"EVT-{timestamp.strftime('%Y%m%d')}-{self._event_counter:04d}",
            event_type=event_type,
            location=random.choice(locations),
            severity=round(severity, 3),
            timestamp=timestamp,
            affected_routes=random.choice(routes),
            estimated_impact="high" if severity > 0.7 else "medium" if severity > 0.4 else "low",
            description=f"{desc} reported near {random.choice(locations)}"
        )
    
    def _fetch_real_data(self) -> ShippingData:
        """
        Placeholder for real API integration.
        
        In production, implement actual API calls here.
        """
        raise NotImplementedError("Real API not implemented. Set use_mock=True.")
    
    def get_historical_events(self, hours: int = 24) -> List[ShippingEvent]:
        """Get historical events for the specified time period."""
        # In a real implementation, this would query historical data
        events = []
        now = datetime.now()
        for i in range(random.randint(2, 8)):
            event_time = now - timedelta(hours=random.uniform(0, hours))
            events.append(self._generate_random_event(event_time))
        return sorted(events, key=lambda e: e.timestamp, reverse=True)
    
    def get_correlation_impact(self, symbol: str) -> float:
        """
        Get the correlation impact factor for a symbol based on current shipping conditions.
        Returns a factor between 0.5 and 1.5.
        """
        data = self.fetch_shipping_data()
        
        # Base impact
        base_factor = 1.0
        
        # Adjust for port congestion
        if data.port_congestion_level > 0.7:
            base_factor *= 1.2
        elif data.port_congestion_level < 0.3:
            base_factor *= 0.9
        
        # Adjust for recent high-severity events
        high_severity_events = [e for e in data.events if e.severity > 0.7]
        if high_severity_events:
            base_factor *= 1.15
        
        # Add noise
        base_factor += random.uniform(-0.1, 0.1)
        
        return round(max(0.5, min(1.5, base_factor)), 3)


# Singleton instance
_shipping_source = None

def get_shipping_source(use_mock: bool = True) -> ShippingAPISource:
    """Get or create the shipping source singleton."""
    global _shipping_source
    if _shipping_source is None:
        _shipping_source = ShippingAPISource(use_mock=use_mock)
    return _shipping_source
