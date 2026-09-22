"""
Real Shipping and Logistics Data Sources.

Production implementations for:
- MarineTraffic API (vessel tracking, port congestion)
- Logistics-to-ticker mapping
"""

import os
import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import aiohttp
import json
import hashlib
from collections import defaultdict

from .shipping_api import ShippingEvent, ShippingData, ShippingAPISource


logger = logging.getLogger(__name__)


@dataclass
class MarineTrafficConfig:
    """MarineTraffic API configuration."""
    api_key: str
    base_url: str = "https://services.marinetraffic.com/api"
    timeout_seconds: int = 10


@dataclass
class VesselInfo:
    """Vessel tracking information."""
    mmsi: str
    imo: str
    vessel_name: str
    vessel_type: str
    latitude: float
    longitude: float
    speed: float
    course: float
    heading: float
    destination: str
    eta: Optional[datetime]
    timestamp: datetime


@dataclass
class PortCongestion:
    """Port congestion metrics."""
    port_name: str
    port_code: str
    congestion_level: float  # 0-1
    vessels_at_berth: int
    vessels_at_anchorage: int
    average_wait_time_hours: float
    timestamp: datetime


class MarineTrafficSource(ShippingAPISource):
    """
    Real shipping data via MarineTraffic API.
    
    Provides:
    - Vessel positions and movements
    - Port congestion levels
    - ETA deviations
    - Route disruptions
    """
    
    def __init__(self, config: MarineTrafficConfig, symbols: List[str] = None):
        super().__init__(use_mock=False)
        self.config = config
        self.symbols = symbols or []
        self._session: Optional[aiohttp.ClientSession] = None
        
        # State
        self._vessels: Dict[str, VesselInfo] = {}
        self._port_congestion: Dict[str, PortCongestion] = {}
        self._recent_events: List[ShippingEvent] = []
        self._event_counter = 0
        
        # Load logistics-to-ticker mapping
        self._ticker_mapping = self._load_ticker_mapping()
        
        # Cached data timestamps
        self._last_vessel_update: Optional[datetime] = None
        self._last_congestion_update: Optional[datetime] = None
    
    def _load_ticker_mapping(self) -> Dict[str, List[str]]:
        """Load logistics-to-ticker mapping from JSON file."""
        mapping_file = "logistics_to_ticker.json"
        
        default_mapping = {
            "Port of Los Angeles": ["FDX", "UPS", "DAL"],
            "Port of Long Beach": ["FDX", "UPS", "DAL"],
            "Port of Shanghai": ["FDX", "UPS", "CHRW"],
            "Port of Rotterdam": ["FDX", "UPS", "CHRW"],
            "Port of Singapore": ["FDX", "UPS", "DAL"],
            "Suez Canal": ["FDX", "UPS", "DAL", "CHRW"],
            "Panama Canal": ["FDX", "UPS", "DAL"],
            "Port of Hamburg": ["FDX", "UPS"],
            "Port of Busan": ["FDX", "UPS", "CHRW"],
            "Port of Dubai": ["FDX", "UPS", "DAL"]
        }
        
        try:
            if os.path.exists(mapping_file):
                with open(mapping_file, 'r') as f:
                    mapping = json.load(f)
                    logger.info(f"Loaded {len(mapping)} entries from {mapping_file}")
                    return mapping
        except Exception as e:
            logger.warning(f"Failed to load {mapping_file}: {e}, using defaults")
        
        return default_mapping
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session
    
    async def fetch_vessel_positions(
        self,
        area: str = None,
        port_code: str = None,
        limit: int = 100
    ) -> List[VesselInfo]:
        """Fetch vessel positions from MarineTraffic API."""
        session = await self._get_session()
        
        params = {
            "key": self.config.api_key,
            "type": "json"
        }
        
        if area:
            params["area"] = area
        if port_code:
            params["port"] = port_code
        
        url = f"{self.config.base_url}/vesselPositions"
        
        try:
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    logger.warning(f"MarineTraffic API returned status {response.status}")
                    return []
                
                data = await response.json()
                vessels = []
                
                for v in data.get("data", [])[:limit]:
                    try:
                        eta = None
                        if v.get("eta"):
                            eta = datetime.strptime(v["eta"], "%Y-%m-%d %H:%M:%S")
                        
                        vessel = VesselInfo(
                            mmsi=str(v.get("MMSI", "")),
                            imo=str(v.get("IMO", "")),
                            vessel_name=v.get("SHIPNAME", ""),
                            vessel_type=v.get("TYPE_SUMMARY", ""),
                            latitude=float(v.get("LAT", 0)),
                            longitude=float(v.get("LON", 0)),
                            speed=float(v.get("SPEED", 0)),
                            course=float(v.get("COURSE", 0)),
                            heading=float(v.get("HEADING", 0)),
                            destination=v.get("DESTINATION", ""),
                            eta=eta,
                            timestamp=datetime.now()
                        )
                        vessels.append(vessel)
                    except Exception as e:
                        logger.warning(f"Error parsing vessel data: {e}")
                
                self._vessels = {v.mmsi: v for v in vessels}
                self._last_vessel_update = datetime.now()
                return vessels
                
        except Exception as e:
            logger.error(f"Error fetching vessel positions: {e}")
            return []
    
    async def fetch_port_congestion(self, port_codes: List[str] = None) -> List[PortCongestion]:
        """Fetch port congestion data from MarineTraffic API."""
        session = await self._get_session()
        
        if not port_codes:
            # Default major ports
            port_codes = [
                "USLAX", "USLGB", "CNSHA", "NLRTM", "SGSIN",
                "EGSUZ", "PAUN5", "DEHAM", "KRPUS", "AEDXB"
            ]
        
        congestions = []
        
        for port_code in port_codes:
            try:
                params = {
                    "key": self.config.api_key,
                    "type": "json",
                    "port": port_code
                }
                
                url = f"{self.config.base_url}/portStats"
                
                async with session.get(url, params=params) as response:
                    if response.status != 200:
                        continue
                    
                    data = await response.json()
                    
                    # Parse congestion metrics
                    at_berth = data.get("at_berth", 0)
                    at_anchorage = data.get("at_anchorage", 0)
                    avg_wait = data.get("avg_waiting_time", 0)
                    
                    # Calculate congestion level (0-1)
                    total_capacity = 50  # Assumed max
                    congestion = min(1.0, (at_berth + at_anchorage * 0.5) / total_capacity)
                    
                    congestion_data = PortCongestion(
                        port_name=data.get("port_name", port_code),
                        port_code=port_code,
                        congestion_level=round(congestion, 3),
                        vessels_at_berth=at_berth,
                        vessels_at_anchorage=at_anchorage,
                        average_wait_time_hours=avg_wait,
                        timestamp=datetime.now()
                    )
                    congestions.append(congestion_data)
                    self._port_congestion[port_code] = congestion_data
                    
            except Exception as e:
                logger.warning(f"Error fetching congestion for {port_code}: {e}")
        
        self._last_congestion_update = datetime.now()
        return congestions
    
    async def detect_eta_deviations(self, threshold_hours: float = 6.0) -> List[ShippingEvent]:
        """Detect vessels with significant ETA deviations."""
        events = []
        
        if not self._vessels:
            await self.fetch_vessel_positions(limit=200)
        
        for mmsi, vessel in self._vessels.items():
            if not vessel.eta:
                continue
            
            # Calculate expected vs actual ETA
            # This is simplified - in production would use historical route times
            distance_to_dest = self._estimate_distance(vessel)
            expected_hours = distance_to_dest / max(vessel.speed, 1)
            actual_hours = (vessel.eta - datetime.now()).total_seconds() / 3600
            
            deviation = abs(actual_hours - expected_hours)
            
            if deviation > threshold_hours:
                self._event_counter += 1
                severity = min(1.0, deviation / 24)  # Scale by days
                
                event = ShippingEvent(
                    event_id=f"ETA-{datetime.now().strftime('%Y%m%d')}-{self._event_counter:04d}",
                    event_type="delay",
                    location=vessel.destination or "Unknown",
                    severity=round(severity, 3),
                    timestamp=datetime.now(),
                    affected_routes=[vessel.destination],
                    estimated_impact="high" if severity > 0.7 else "medium" if severity > 0.4 else "low",
                    description=f"Vessel {vessel.vessel_name} (MMSI: {mmsi}) ETA deviation: {deviation:.1f}h"
                )
                events.append(event)
        
        self._recent_events.extend(events)
        return events
    
    def _estimate_distance(self, vessel: VesselInfo) -> float:
        """Estimate distance to destination in nautical miles (simplified)."""
        # In production, would use proper geodesic calculations
        # and port coordinate database
        return 1000.0  # Placeholder
    
    async def fetch_shipping_data(self) -> ShippingData:
        """Fetch comprehensive shipping data."""
        now = datetime.now()
        
        # Fetch all data concurrently
        await asyncio.gather(
            self.fetch_vessel_positions(limit=200),
            self.fetch_port_congestion(),
            self.detect_eta_deviations()
        )
        
        # Calculate global shipping index (simplified)
        base_index = 1500.0
        if self._port_congestion:
            avg_congestion = sum(p.congestion_level for p in self._port_congestion.values()) / len(self._port_congestion)
            base_index *= (1 + avg_congestion * 0.2)
        
        # Count active vessels
        active_vessels = len(self._vessels)
        
        # Aggregate events
        all_events = list(self._recent_events[-10:])  # Last 10 events
        
        # Calculate average delay
        avg_delay = 0.0
        if self._port_congestion:
            avg_delay = sum(p.average_wait_time_hours for p in self._port_congestion.values()) / len(self._port_congestion)
        
        return ShippingData(
            timestamp=now,
            global_shipping_index=round(base_index, 2),
            port_congestion_level=round(sum(p.congestion_level for p in self._port_congestion.values()) / max(len(self._port_congestion), 1), 3),
            average_delay_hours=round(avg_delay, 1),
            active_vessels=active_vessels,
            events=all_events
        )
    
    def get_correlation_impact(self, symbol: str) -> float:
        """Get shipping impact factor for a specific symbol."""
        base_factor = 1.0
        
        # Find ports/routes affecting this symbol
        affected_ports = [
            port for port, tickers in self._ticker_mapping.items()
            if symbol in tickers
        ]
        
        if not affected_ports:
            return base_factor
        
        # Adjust based on congestion at relevant ports
        for port in affected_ports:
            port_code = self._get_port_code(port)
            if port_code in self._port_congestion:
                congestion = self._port_congestion[port_code].congestion_level
                if congestion > 0.7:
                    base_factor *= 1.2
                elif congestion < 0.3:
                    base_factor *= 0.9
        
        # Adjust for recent events
        symbol_events = [
            e for e in self._recent_events
            if any(symbol in (self._ticker_mapping.get(e.location, []) or []))
        ]
        
        if symbol_events:
            avg_severity = sum(e.severity for e in symbol_events) / len(symbol_events)
            base_factor *= (1 + avg_severity * 0.15)
        
        return round(max(0.5, min(1.5, base_factor)), 3)
    
    def _get_port_code(self, port_name: str) -> str:
        """Map port name to code."""
        mapping = {
            "Port of Los Angeles": "USLAX",
            "Port of Long Beach": "USLGB",
            "Port of Shanghai": "CNSHA",
            "Port of Rotterdam": "NLRTM",
            "Port of Singapore": "SGSIN",
            "Suez Canal": "EGSUZ",
            "Panama Canal": "PAUN5",
            "Port of Hamburg": "DEHAM",
            "Port of Busan": "KRPUS",
            "Port of Dubai": "AEDXB"
        }
        return mapping.get(port_name, "")
    
    async def close(self):
        """Close HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()


def get_marine_traffic_source(symbols: List[str] = None) -> MarineTrafficSource:
    """Create MarineTraffic source from environment variables."""
    api_key = os.getenv("MARINETRAFFIC_API_KEY")
    
    if not api_key:
        raise ValueError("MARINETRAFFIC_API_KEY must be set in .env")
    
    config = MarineTrafficConfig(api_key=api_key)
    return MarineTrafficSource(config=config, symbols=symbols)
