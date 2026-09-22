"""
Real Market Data Sources.

Production implementations for:
- Alpaca Markets (primary WebSocket)
- Polygon.io (backup REST)
"""

import os
import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import websockets
import json
import aiohttp
from collections import deque
import numpy as np

from .market_api import PriceQuote, MarketSnapshot, MarketAPISource


logger = logging.getLogger(__name__)


@dataclass
class AlpacaConfig:
    """Alpaca API configuration."""
    api_key: str
    api_secret: str
    ws_url: str = "wss://stream.data.alpaca.markets/v2/iex"
    rest_url: str = "https://data.alpaca.markets/v2"
    use_paper: bool = True


@dataclass
class PolygonConfig:
    """Polygon.io API configuration."""
    api_key: str
    rest_url: str = "https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/minute/{from_date}/{to_date}"
    quote_url: str = "https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/tickers/{ticker}"


class AlpacaWebSocketSource(MarketAPISource):
    """
    Real-time market data via Alpaca WebSocket.
    
    Primary data source with automatic failover to Polygon.io.
    """
    
    def __init__(self, config: AlpacaConfig, symbols: List[str], fallback_to_polygon: bool = True):
        super().__init__(use_mock=False, symbols=symbols)
        self.config = config
        self.symbols = symbols
        self.fallback_enabled = fallback_to_polygon
        
        # WebSocket state
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._connected = False
        self._last_message_time: Optional[datetime] = None
        
        # Real-time quote storage
        self._live_quotes: Dict[str, PriceQuote] = {}
        self._quote_buffers: Dict[str, deque] = {s: deque(maxlen=500) for s in symbols}
        
        # Fallback instance
        self._polygon_source: Optional[PolygonRESTSource] = None
        if fallback_to_polygon:
            polygon_key = os.getenv("POLYGON_API_KEY")
            if polygon_key:
                self._polygon_source = PolygonRESTSource(
                    PolygonConfig(api_key=polygon_key),
                    symbols=symbols
                )
        
        # Connection retry state
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 10
    
    async def connect(self):
        """Establish WebSocket connection to Alpaca."""
        try:
            headers = {
                "APCA-API-KEY-ID": self.config.api_key,
                "APCA-API-SECRET-KEY": self.config.api_secret
            }
            
            self._ws = await websockets.connect(
                self.config.ws_url,
                extra_headers=headers,
                ping_interval=30,
                ping_timeout=10
            )
            
            # Subscribe to trades and quotes for all symbols
            subscribe_msg = {
                "action": "subscribe",
                "trades": self.symbols,
                "quotes": self.symbols,
                "bars": self.symbols
            }
            
            await self._ws.send(json.dumps(subscribe_msg))
            logger.info(f"Subscribed to {len(self.symbols)} symbols on Alpaca WebSocket")
            
            self._connected = True
            self._reconnect_attempts = 0
            logger.info("Alpaca WebSocket connected successfully")
            
        except Exception as e:
            logger.error(f"Failed to connect to Alpaca WebSocket: {e}")
            self._connected = False
            raise
    
    async def _receive_messages(self):
        """Continuously receive and process WebSocket messages."""
        if not self._ws or not self._connected:
            return
        
        try:
            async for message in self._ws:
                try:
                    data = json.loads(message)
                    await self._process_message(data)
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON message: {e}")
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
        except websockets.ConnectionClosed as e:
            logger.warning(f"WebSocket connection closed: {e}")
            self._connected = False
            await self._reconnect()
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            self._connected = False
    
    async def _process_message(self, data: dict):
        """Process incoming WebSocket message."""
        if not isinstance(data, list):
            return
        
        for item in data:
            if not isinstance(item, dict):
                continue
            
            msg_type = item.get("T")
            symbol = item.get("S")
            
            if not symbol or symbol not in self.symbols:
                continue
            
            if msg_type == "t":  # Trade
                await self._process_trade(item, symbol)
            elif msg_type == "q":  # Quote
                await self._process_quote(item, symbol)
            elif msg_type == "b":  # Bar
                await self._process_bar(item, symbol)
    
    async def _process_trade(self, trade_data: dict, symbol: str):
        """Process trade message."""
        timestamp = datetime.fromtimestamp(trade_data.get("t", 0) / 1000)
        price = trade_data.get("p", 0)
        volume = trade_data.get("s", 0)
        
        # Update live quote
        current_quote = self._live_quotes.get(symbol)
        if current_quote:
            new_quote = PriceQuote(
                symbol=symbol,
                timestamp=timestamp,
                open=current_quote.open,
                high=max(current_quote.high, price),
                low=min(current_quote.low, price),
                close=price,
                volume=current_quote.volume + int(volume),
                bid=current_quote.bid,
                ask=current_quote.ask
            )
        else:
            new_quote = PriceQuote(
                symbol=symbol,
                timestamp=timestamp,
                open=price,
                high=price,
                low=price,
                close=price,
                volume=int(volume),
                bid=price * 0.9999,
                ask=price * 1.0001
            )
        
        self._live_quotes[symbol] = new_quote
        self._quote_buffers[symbol].append(new_quote)
        self._last_message_time = datetime.now()
    
    async def _process_quote(self, quote_data: dict, symbol: str):
        """Process quote message."""
        timestamp = datetime.fromtimestamp(quote_data.get("t", 0) / 1000)
        bid = quote_data.get("bp", 0)
        ask = quote_data.get("ap", 0)
        bid_size = quote_data.get("bs", 0)
        ask_size = quote_data.get("as", 0)
        
        # Get or create base quote
        current_quote = self._live_quotes.get(symbol)
        if current_quote:
            new_quote = PriceQuote(
                symbol=symbol,
                timestamp=timestamp,
                open=current_quote.open,
                high=current_quote.high,
                low=current_quote.low,
                close=current_quote.close,
                volume=current_quote.volume,
                bid=bid,
                ask=ask
            )
        else:
            mid_price = (bid + ask) / 2
            new_quote = PriceQuote(
                symbol=symbol,
                timestamp=timestamp,
                open=mid_price,
                high=mid_price,
                low=mid_price,
                close=mid_price,
                volume=0,
                bid=bid,
                ask=ask
            )
        
        self._live_quotes[symbol] = new_quote
        self._quote_buffers[symbol].append(new_quote)
        self._last_message_time = datetime.now()
    
    async def _process_bar(self, bar_data: dict, symbol: str):
        """Process bar (OHLCV) message."""
        timestamp = datetime.fromtimestamp(bar_data.get("t", 0) / 1000)
        open_price = bar_data.get("o", 0)
        high = bar_data.get("h", 0)
        low = bar_data.get("l", 0)
        close = bar_data.get("c", 0)
        volume = bar_data.get("v", 0)
        
        new_quote = PriceQuote(
            symbol=symbol,
            timestamp=timestamp,
            open=open_price,
            high=high,
            low=low,
            close=close,
            volume=int(volume),
            bid=close * 0.9999,
            ask=close * 1.0001
        )
        
        self._live_quotes[symbol] = new_quote
        self._quote_buffers[symbol].append(new_quote)
        self._last_message_time = datetime.now()
    
    async def _reconnect(self):
        """Attempt to reconnect to WebSocket."""
        if self._reconnect_attempts >= self._max_reconnect_attempts:
            logger.error("Max reconnection attempts reached")
            return
        
        self._reconnect_attempts += 1
        delay = min(2 ** self._reconnect_attempts, 30)
        logger.info(f"Reconnecting in {delay}s (attempt {self._reconnect_attempts})")
        
        await asyncio.sleep(delay)
        
        try:
            await self.connect()
        except Exception as e:
            logger.error(f"Reconnection failed: {e}")
    
    async def fetch_market_data(self) -> MarketSnapshot:
        """Fetch current market data from live WebSocket or fallback."""
        if self._connected and self._live_quotes:
            return self._build_snapshot_from_live()
        
        # Fallback to Polygon.io REST
        if self.fallback_enabled and self._polygon_source:
            logger.warning("Using Polygon.io fallback for market data")
            return await self._polygon_source.fetch_market_data()
        
        # Last resort: use cached data
        if self._live_quotes:
            return self._build_snapshot_from_live()
        
        raise RuntimeError("No market data available")
    
    def _build_snapshot_from_live(self) -> MarketSnapshot:
        """Build MarketSnapshot from live quotes."""
        now = datetime.now()
        
        # Calculate market sentiment from recent price movements
        sentiments = []
        for symbol in self.symbols:
            buffer = self._quote_buffers[symbol]
            if len(buffer) >= 10:
                prices = [q.close for q in list(buffer)[-10:]]
                recent = np.mean(prices[-5:])
                older = np.mean(prices[:5])
                if older > 0:
                    sentiment = 0.5 + (recent - older) / older * 10
                    sentiment = max(0.0, min(1.0, sentiment))
                    sentiments.append(sentiment)
        
        market_sentiment = np.mean(sentiments) if sentiments else 0.5
        
        return MarketSnapshot(
            timestamp=now,
            quotes=dict(self._live_quotes),
            market_sentiment=round(market_sentiment, 3),
            vix_level=20.0  # Could fetch from VIX data
        )
    
    def get_price_history(self, symbol: str, periods: int = 50) -> List[float]:
        """Get historical prices from buffer."""
        buffer = self._quote_buffers.get(symbol, [])
        prices = [q.close for q in list(buffer)[-periods:]]
        return prices if prices else super().get_price_history(symbol, periods)
    
    def get_volume_history(self, symbol: str, periods: int = 50) -> List[int]:
        """Get historical volumes from buffer."""
        buffer = self._quote_buffers.get(symbol, [])
        volumes = [q.volume for q in list(buffer)[-periods:]]
        return volumes if volumes else super().get_volume_history(symbol, periods)
    
    async def close(self):
        """Close WebSocket connection."""
        if self._ws:
            await self._ws.close()
            self._connected = False
            logger.info("Alpaca WebSocket connection closed")


class PolygonRESTSource(MarketAPISource):
    """
    Backup market data source via Polygon.io REST API.
    
    Used when Alpaca WebSocket is unavailable.
    """
    
    def __init__(self, config: PolygonConfig, symbols: List[str]):
        super().__init__(use_mock=False, symbols=symbols)
        self.config = config
        self.symbols = symbols
        self._session: Optional[aiohttp.ClientSession] = None
        self._cached_quotes: Dict[str, PriceQuote] = {}
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=10)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session
    
    async def fetch_market_data(self) -> MarketSnapshot:
        """Fetch market data from Polygon.io REST API."""
        quotes = {}
        
        session = await self._get_session()
        
        for symbol in self.symbols:
            try:
                quote = await self._fetch_quote(session, symbol)
                if quote:
                    quotes[symbol] = quote
            except Exception as e:
                logger.error(f"Failed to fetch quote for {symbol}: {e}")
        
        if not quotes:
            raise RuntimeError("Failed to fetch any quotes from Polygon.io")
        
        return MarketSnapshot(
            timestamp=datetime.now(),
            quotes=quotes,
            market_sentiment=0.5,
            vix_level=20.0
        )
    
    async def _fetch_quote(self, session: aiohttp.ClientSession, symbol: str) -> Optional[PriceQuote]:
        """Fetch single quote from Polygon.io."""
        url = self.config.quote_url.format(ticker=symbol)
        params = {"apiKey": self.config.api_key}
        
        try:
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    logger.warning(f"Polygon API returned status {response.status} for {symbol}")
                    return None
                
                data = await response.json()
                
                if "ticker" not in data:
                    return None
                
                ticker = data["ticker"]
                last_trade = ticker.get("lastTrade", {})
                
                if not last_trade:
                    return None
                
                price = last_trade.get("p", 0)
                timestamp = datetime.fromtimestamp(last_trade.get("t", 0) / 1000)
                
                quote = PriceQuote(
                    symbol=symbol,
                    timestamp=timestamp,
                    open=price,
                    high=price,
                    low=price,
                    close=price,
                    volume=last_trade.get("s", 0),
                    bid=ticker.get("todaysChange", 0),
                    ask=price * 1.0001
                )
                
                self._cached_quotes[symbol] = quote
                return quote
                
        except Exception as e:
            logger.error(f"Error fetching quote for {symbol}: {e}")
            return None
    
    async def fetch_historical_bars(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        multiplier: int = 1,
        timespan: str = "minute"
    ) -> List[Dict[str, Any]]:
        """Fetch historical bars from Polygon.io."""
        session = await self._get_session()
        
        url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
        params = {
            "apiKey": self.config.api_key,
            "adjusted": "true",
            "sort": "asc"
        }
        
        try:
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    return []
                
                data = await response.json()
                results = data.get("results", [])
                return results
                
        except Exception as e:
            logger.error(f"Error fetching historical bars for {symbol}: {e}")
            return []
    
    async def close(self):
        """Close HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()


def get_alpaca_source(symbols: List[str], use_paper: bool = True) -> AlpacaWebSocketSource:
    """Create Alpaca WebSocket source from environment variables."""
    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_API_SECRET")
    
    if not api_key or not api_secret:
        raise ValueError("ALPACA_API_KEY and ALPACA_API_SECRET must be set in .env")
    
    config = AlpacaConfig(
        api_key=api_key,
        api_secret=api_secret,
        use_paper=use_paper
    )
    
    return AlpacaWebSocketSource(config=config, symbols=symbols)


def get_polygon_source(symbols: List[str]) -> PolygonRESTSource:
    """Create Polygon.io REST source from environment variables."""
    api_key = os.getenv("POLYGON_API_KEY")
    
    if not api_key:
        raise ValueError("POLYGON_API_KEY must be set in .env")
    
    config = PolygonConfig(api_key=api_key)
    return PolygonRESTSource(config=config, symbols=symbols)
