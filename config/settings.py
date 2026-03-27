"""
Configuration settings for the AI Trading Agent.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List


# Base paths
BASE_DIR = Path(__file__).parent.parent
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)


@dataclass
class TradingSettings:
    """Main trading configuration."""
    
    # Capital settings
    initial_capital: float = 100000.0
    max_capital_per_trade: float = 0.01  # 1% of capital
    
    # Risk limits
    max_concurrent_trades: int = 3
    daily_loss_cap: float = 0.03  # 3% daily loss limit
    stop_loss_pct: float = 0.02  # 2% stop loss
    take_profit_pct: float = 0.04  # 4% take profit
    
    # Execution settings
    paper_trading: bool = True
    order_execution_delay: float = 0.5  # seconds
    
    # Data refresh intervals
    market_data_interval: int = 10  # seconds
    shipping_data_interval: int = 30  # seconds
    
    # ML settings
    model_confidence_threshold: float = 0.6
    use_mock_data: bool = True  # Use mock data for testing
    
    # Logging
    log_level: str = "INFO"
    log_to_file: bool = True
    log_to_console: bool = True


@dataclass
class APISettings:
    """API configuration for data sources."""
    
    # Market data (mock by default)
    market_api_url: str = "https://api.example.com/market"
    market_api_key: str = ""
    
    # Shipping data (mock by default)
    shipping_api_url: str = "https://api.example.com/shipping"
    shipping_api_key: str = ""
    
    # News data (mock by default)
    news_api_url: str = "https://api.example.com/news"
    news_api_key: str = ""
    
    # Broker API (paper trading mock)
    broker_api_url: str = "https://paper-trading.example.com/api"
    broker_api_key: str = "mock_paper_key"


@dataclass
class FeatureSettings:
    """Feature engineering configuration."""
    
    # Price change windows (in periods)
    short_window: int = 5
    medium_window: int = 20
    long_window: int = 50
    
    # RSI settings
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    
    # Volume settings
    volume_ma_period: int = 20
    volume_spike_threshold: float = 2.0
    
    # Event weights
    event_severity_weight: float = 1.0
    time_decay_factor: float = 0.95


@dataclass
class SystemSettings:
    """System-wide settings."""
    
    # Trading loop
    loop_interval: int = 10  # seconds
    enable_backtest: bool = False
    
    # State persistence
    state_file: str = "trading_state.json"
    
    # Feature flags
    enable_momentum_strategy: bool = True
    enable_latency_strategy: bool = True
    enable_mean_reversion_strategy: bool = True
    
    # Max events to keep in memory
    max_event_history: int = 1000


# Global settings instance
settings = TradingSettings()
api_settings = APISettings()
feature_settings = FeatureSettings()
system_settings = SystemSettings()


def get_log_path(filename: str) -> Path:
    """Get full path for log file."""
    return LOGS_DIR / filename


def update_setting(category: str, key: str, value: any):
    """Update a setting dynamically."""
    settings_map = {
        "trading": settings,
        "api": api_settings,
        "feature": feature_settings,
        "system": system_settings
    }
    
    if category in settings_map and hasattr(settings_map[category], key):
        setattr(settings_map[category], key, value)
