"""
Technical Indicators Module.

Computes technical indicators for trading signals.
"""

from typing import List, Tuple, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class TechnicalIndicators:
    """Container for technical indicators."""
    # Price-based
    sma_short: float = 0.0
    sma_medium: float = 0.0
    sma_long: float = 0.0
    ema_short: float = 0.0
    ema_long: float = 0.0
    
    # Momentum
    rsi: float = 50.0  # 0-100
    macd: float = 0.0
    macd_signal: float = 0.0
    macd_histogram: float = 0.0
    
    # Volatility
    bollinger_upper: float = 0.0
    bollinger_middle: float = 0.0
    bollinger_lower: float = 0.0
    atr: float = 0.0
    std_dev: float = 0.0
    
    # Volume
    volume_ratio: float = 1.0
    obv: float = 0.0
    
    # Trend
    trend_strength: float = 0.0
    trend_direction: str = "neutral"
    
    # Custom
    price_momentum: float = 0.0
    price_acceleration: float = 0.0


def calculate_sma(prices: List[float], period: int) -> float:
    """Calculate Simple Moving Average."""
    if len(prices) < period:
        return np.mean(prices) if prices else 0.0
    return np.mean(prices[-period:])


def calculate_ema(prices: List[float], period: int) -> float:
    """Calculate Exponential Moving Average."""
    if len(prices) < period:
        return np.mean(prices) if prices else 0.0
    
    multiplier = 2 / (period + 1)
    ema = prices[0]
    
    for price in prices[1:]:
        ema = (price - ema) * multiplier + ema
    
    return ema


def calculate_rsi(prices: List[float], period: int = 14) -> float:
    """Calculate Relative Strength Index."""
    if len(prices) < period + 1:
        return 50.0
    
    # Calculate price changes
    deltas = np.diff(prices)
    
    # Separate gains and losses
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    # Calculate average gains and losses
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    
    if avg_loss == 0:
        return 100.0
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return round(rsi, 2)


def calculate_macd(prices: List[float], 
                    fast_period: int = 12, 
                    slow_period: int = 26, 
                    signal_period: int = 9) -> Tuple[float, float, float]:
    """Calculate MACD (Moving Average Convergence Divergence)."""
    if len(prices) < slow_period:
        return (0.0, 0.0, 0.0)
    
    # Calculate EMAs
    ema_fast = calculate_ema(prices, fast_period)
    ema_slow = calculate_ema(prices, slow_period)
    
    # MACD line
    macd = ema_fast - ema_slow
    
    # Signal line (EMA of MACD)
    # Simplified: use recent MACD values
    macd_hist = [macd * 0.8]  # Placeholder
    signal = np.mean(macd_hist) if len(macd_hist) >= signal_period else macd
    
    # Histogram
    histogram = macd - signal
    
    return (round(macd, 4), round(signal, 4), round(histogram, 4))


def calculate_bollinger_bands(prices: List[float], 
                               period: int = 20, 
                               std_dev_mult: float = 2.0) -> Tuple[float, float, float]:
    """Calculate Bollinger Bands."""
    if len(prices) < period:
        middle = np.mean(prices) if prices else 0.0
        return (middle, middle, middle)
    
    recent_prices = prices[-period:]
    middle = np.mean(recent_prices)
    std = np.std(recent_prices)
    
    upper = middle + (std * std_dev_mult)
    lower = middle - (std * std_dev_mult)
    
    return (round(upper, 2), round(middle, 2), round(lower, 2))


def calculate_atr(highs: List[float], 
                   lows: List[float], 
                   closes: List[float], 
                   period: int = 14) -> float:
    """Calculate Average True Range."""
    if len(highs) < period + 1:
        return 0.0
    
    tr_list = []
    for i in range(1, len(closes)):
        high_low = highs[i] - lows[i]
        high_close = abs(highs[i] - closes[i-1])
        low_close = abs(lows[i] - closes[i-1])
        tr = max(high_low, high_close, low_close)
        tr_list.append(tr)
    
    if len(tr_list) < period:
        return np.mean(tr_list) if tr_list else 0.0
    
    return round(np.mean(tr_list[-period:]), 4)


def calculate_volume_ratio(volumes: List[int], period: int = 20) -> float:
    """Calculate volume ratio vs moving average."""
    if len(volumes) < period:
        return 1.0
    
    avg_volume = np.mean(volumes[-period:])
    current_volume = volumes[-1]
    
    if avg_volume == 0:
        return 1.0
    
    return round(current_volume / avg_volume, 2)


def calculate_obv(prices: List[float], volumes: List[int]) -> float:
    """Calculate On-Balance Volume."""
    if len(prices) < 2 or len(volumes) < 2:
        return float(sum(volumes)) if volumes else 0.0
    
    obv = 0.0
    for i in range(1, len(prices)):
        if prices[i] > prices[i-1]:
            obv += volumes[i]
        elif prices[i] < prices[i-1]:
            obv -= volumes[i]
    
    return round(obv, 0)


def calculate_momentum(prices: List[float], period: int = 10) -> float:
    """Calculate price momentum."""
    if len(prices) < period:
        return 0.0
    
    momentum = (prices[-1] - prices[-period]) / prices[-period] if prices[-period] != 0 else 0.0
    return round(momentum * 100, 2)  # As percentage


def calculate_acceleration(prices: List[float], period: int = 5) -> float:
    """Calculate price acceleration (change in momentum)."""
    if len(prices) < period * 2:
        return 0.0
    
    mom1 = calculate_momentum(prices, period)
    older_prices = prices[:-period]
    mom2 = calculate_momentum(older_prices, period)
    
    acceleration = mom1 - mom2
    return round(acceleration, 2)


def calculate_all_indicators(prices: List[float], 
                              volumes: List[int],
                              highs: List[float] = None,
                              lows: List[float] = None) -> TechnicalIndicators:
    """Calculate all technical indicators."""
    # Use prices as highs/lows if not provided
    if highs is None:
        highs = prices
    if lows is None:
        lows = prices
    
    indicators = TechnicalIndicators()
    
    # Price-based
    indicators.sma_short = calculate_sma(prices, 5)
    indicators.sma_medium = calculate_sma(prices, 20)
    indicators.sma_long = calculate_sma(prices, 50)
    indicators.ema_short = calculate_ema(prices, 5)
    indicators.ema_long = calculate_ema(prices, 20)
    
    # Momentum
    indicators.rsi = calculate_rsi(prices, 14)
    macd, signal, hist = calculate_macd(prices)
    indicators.macd = macd
    indicators.macd_signal = signal
    indicators.macd_histogram = hist
    
    # Volatility
    upper, middle, lower = calculate_bollinger_bands(prices)
    indicators.bollinger_upper = upper
    indicators.bollinger_middle = middle
    indicators.bollinger_lower = lower
    indicators.atr = calculate_atr(highs, lows, prices)
    indicators.std_dev = np.std(prices[-20:]) if len(prices) >= 20 else np.std(prices)
    
    # Volume
    indicators.volume_ratio = calculate_volume_ratio(volumes)
    indicators.obv = calculate_obv(prices, volumes)
    
    # Trend
    indicators.trend_direction = "uptrend" if indicators.sma_short > indicators.sma_long else "downtrend" if indicators.sma_short < indicators.sma_long else "neutral"
    indicators.trend_strength = abs(indicators.sma_short - indicators.sma_long) / indicators.sma_long if indicators.sma_long != 0 else 0
    
    # Custom
    indicators.price_momentum = calculate_momentum(prices)
    indicators.price_acceleration = calculate_acceleration(prices)
    
    return indicators


def is_overbought(rsi: float, threshold: float = 70.0) -> bool:
    """Check if RSI indicates overbought condition."""
    return rsi >= threshold


def is_oversold(rsi: float, threshold: float = 30.0) -> bool:
    """Check if RSI indicates oversold condition."""
    return rsi <= threshold


def is_bollinger_breakout(price: float, upper_band: float, lower_band: float) -> str:
    """Check for Bollinger Band breakout."""
    if price >= upper_band:
        return "upper"
    elif price <= lower_band:
        return "lower"
    return "within"
