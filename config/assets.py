"""
Asset definitions for the AI Trading Agent.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum


class AssetType(Enum):
    """Asset type enumeration."""
    STOCK = "stock"
    ETF = "etf"
    COMMODITY = "commodity"
    FOREX = "forex"
    CRYPTO = "crypto"


class AssetClass(Enum):
    """Asset class for correlation analysis."""
    TRANSPORTATION = "transportation"
    LOGISTICS = "logistics"
    ENERGY = "energy"
    TECHNOLOGY = "technology"
    CONSUMER = "consumer"
    INDUSTRIAL = "industrial"


@dataclass
class Asset:
    """Represents a tradeable asset."""
    symbol: str
    name: str
    asset_type: AssetType
    asset_class: AssetClass
    
    # Related assets for correlation analysis
    correlated_assets: List[str] = None
    
    # Shipping/logistics relevance
    shipping_relevance: float = 0.0  # 0-1 scale
    
    # Position limits
    max_position_size: float = 1.0  # Max % of portfolio
    min_position_size: float = 0.0
    
    def __post_init__(self):
        if self.correlated_assets is None:
            self.correlated_assets = []


@dataclass
class TradingPair:
    """Represents a trading pair (for latency strategy)."""
    primary: Asset
    secondary: Asset
    correlation: float  # Historical correlation coefficient
    lag_threshold: float  # Max lag in seconds for signal


# Define default trading assets
DEFAULT_ASSETS: Dict[str, Asset] = {
    # Transportation & Logistics stocks
    "FDX": Asset(
        symbol="FDX",
        name="FedEx Corporation",
        asset_type=AssetType.STOCK,
        asset_class=AssetClass.LOGISTICS,
        correlated_assets=["UPS", "DAL", "CHRW"],
        shipping_relevance=0.9
    ),
    "UPS": Asset(
        symbol="UPS",
        name="United Parcel Service",
        asset_type=AssetType.STOCK,
        asset_class=AssetClass.LOGISTICS,
        correlated_assets=["FDX", "DAL", "CHRW"],
        shipping_relevance=0.9
    ),
    "DAL": Asset(
        symbol="DAL",
        name="Delta Air Lines",
        asset_type=AssetType.STOCK,
        asset_class=AssetClass.TRANSPORTATION,
        correlated_assets=["FDX", "UPS", "UAL"],
        shipping_relevance=0.6
    ),
    "CHRW": Asset(
        symbol="CHRW",
        name="C.H. Robinson Worldwide",
        asset_type=AssetType.STOCK,
        asset_class=AssetClass.LOGISTICS,
        correlated_assets=["FDX", "UPS", "JBHT"],
        shipping_relevance=0.95
    ),
    "JBHT": Asset(
        symbol="JBHT",
        name="J.B. Hunt Transport Services",
        asset_type=AssetType.STOCK,
        asset_class=AssetClass.LOGISTICS,
        correlated_assets=["FDX", "UPS", "CHRW"],
        shipping_relevance=0.9
    ),
    
    # Industrial & Energy
    "CAT": Asset(
        symbol="CAT",
        name="Caterpillar Inc.",
        asset_type=AssetType.STOCK,
        asset_class=AssetClass.INDUSTRIAL,
        correlated_assets=["DE", "BA"],
        shipping_relevance=0.5
    ),
    
    # ETFs
    "IYT": Asset(
        symbol="IYT",
        name="iShares Transportation Average ETF",
        asset_type=AssetType.ETF,
        asset_class=AssetClass.TRANSPORTATION,
        correlated_assets=["FDX", "UPS", "DAL"],
        shipping_relevance=0.8
    ),
    "XLI": Asset(
        symbol="XLI",
        name="Industrial Select Sector SPDR",
        asset_type=AssetType.ETF,
        asset_class=AssetClass.INDUSTRIAL,
        correlated_assets=["CAT", "BA"],
        shipping_relevance=0.4
    ),
    
    # Shipping indices
    "GNSS": Asset(
        symbol="GNSS",
        name="Global Shipping Index (simulated)",
        asset_type=AssetType.ETF,
        asset_class=AssetClass.LOGISTICS,
        correlated_assets=["FDX", "UPS", "CHRW"],
        shipping_relevance=1.0
    ),
}

# Trading pairs for latency arbitrage
DEFAULT_TRADING_PAIRS: List[TradingPair] = [
    TradingPair(
        primary=DEFAULT_ASSETS["FDX"],
        secondary=DEFAULT_ASSETS["UPS"],
        correlation=0.85,
        lag_threshold=30.0
    ),
    TradingPair(
        primary=DEFAULT_ASSETS["FDX"],
        secondary=DEFAULT_ASSETS["IYT"],
        correlation=0.80,
        lag_threshold=20.0
    ),
    TradingPair(
        primary=DEFAULT_ASSETS["CHRW"],
        secondary=DEFAULT_ASSETS["JBHT"],
        correlation=0.75,
        lag_threshold=45.0
    ),
]


def get_asset(symbol: str) -> Optional[Asset]:
    """Get asset by symbol."""
    return DEFAULT_ASSETS.get(symbol.upper())


def get_assets_by_class(asset_class: AssetClass) -> List[Asset]:
    """Get all assets of a specific class."""
    return [a for a in DEFAULT_ASSETS.values() if a.asset_class == asset_class]


def get_assets_by_type(asset_type: AssetType) -> List[Asset]:
    """Get all assets of a specific type."""
    return [a for a in DEFAULT_ASSETS.values() if a.asset_type == asset_type]


def get_high_relevance_assets(threshold: float = 0.8) -> List[Asset]:
    """Get assets with high shipping relevance."""
    return [a for a in DEFAULT_ASSETS.values() if a.shipping_relevance >= threshold]


def get_correlated_assets(symbol: str) -> List[Asset]:
    """Get all correlated assets for a given symbol."""
    asset = get_asset(symbol)
    if not asset:
        return []
    
    correlated = []
    for corr_symbol in asset.correlated_assets:
        corr_asset = get_asset(corr_symbol)
        if corr_asset:
            correlated.append(corr_asset)
    return correlated


# Active trading configuration
ACTIVE_SYMBOLS: List[str] = ["FDX", "UPS", "CHRW", "DAL", "IYT"]
