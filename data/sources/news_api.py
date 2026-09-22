"""
Real News and Sentiment Data Sources.

Production implementations for:
- NewsAPI.org (financial news)
- SEC EDGAR RSS (8-K filings)
- FinBERT sentiment analysis
"""

import os
import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import aiohttp
import feedparser
import re
from collections import deque


logger = logging.getLogger(__name__)


@dataclass
class NewsArticle:
    """Represents a news article."""
    article_id: str
    headline: str
    source: str
    timestamp: datetime
    sentiment_score: float  # -1 to 1
    relevance_score: float  # 0 to 1 for shipping/logistics
    url: str = ""
    summary: str = ""
    related_symbols: List[str] = field(default_factory=list)


@dataclass
class NewsSnapshot:
    """Container for news data."""
    timestamp: datetime
    articles: List[NewsArticle] = field(default_factory=list)
    overall_sentiment: float = 0.0  # -1 to 1
    shipping_sentiment: float = 0.0  # -1 to 1


class NewsAPISourceBase:
    """Base class for news API sources."""
    
    def fetch_news(self, symbols: List[str] = None) -> NewsSnapshot:
        raise NotImplementedError
    
    def get_sentiment_for_symbol(self, symbol: str, lookback_hours: int = 24) -> float:
        raise NotImplementedError


@dataclass
class NewsAPIConfig:
    """NewsAPI.org configuration."""
    api_key: str
    base_url: str = "https://newsapi.org/v2"
    timeout_seconds: int = 10


@dataclass
class EDGARConfig:
    """SEC EDGAR RSS configuration."""
    base_url: str = "https://www.sec.gov/cgi-bin/browse-edgar"
    timeout_seconds: int = 10


class NewsAPISourceReal(NewsAPISourceBase):
    """
    Real news data via NewsAPI.org.
    
    Provides:
    - Financial news headlines
    - Company-specific news
    - Sector news
    """
    
    def __init__(self, config: NewsAPIConfig, symbols: List[str] = None):
        super().__init__()
        self.config = config
        self.symbols = symbols or []
        self._session: Optional[aiohttp.ClientSession] = None
        
        # Article cache
        self._articles: deque = deque(maxlen=500)
        self._last_fetch: Optional[datetime] = None
        
        # Symbol-specific queries
        self._symbol_names = {
            "FDX": "FedEx",
            "UPS": "UPS United Parcel Service",
            "CHRW": "C.H. Robinson",
            "DAL": "Delta Air Lines",
            "IYT": "transportation ETF"
        }
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session
    
    async def fetch_news(self, symbols: List[str] = None) -> NewsSnapshot:
        """Fetch latest news from NewsAPI.org."""
        symbols = symbols or self.symbols
        now = datetime.now()
        
        articles = []
        
        # Fetch news for each symbol
        tasks = [self._fetch_symbol_news(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, list):
                articles.extend(result)
            elif isinstance(result, Exception):
                logger.warning(f"News fetch error: {result}")
        
        # Also fetch general shipping/supply chain news
        try:
            general_articles = await self._fetch_general_shipping_news()
            articles.extend(general_articles)
        except Exception as e:
            logger.warning(f"Failed to fetch general shipping news: {e}")
        
        # Calculate sentiments
        if articles:
            overall_sentiment = sum(a.sentiment_score for a in articles) / len(articles)
            shipping_articles = [a for a in articles if a.relevance_score > 0.5]
            shipping_sentiment = (
                sum(a.sentiment_score for a in shipping_articles) / len(shipping_articles)
                if shipping_articles else 0.0
            )
        else:
            overall_sentiment = 0.0
            shipping_sentiment = 0.0
        
        snapshot = NewsSnapshot(
            timestamp=now,
            articles=articles,
            overall_sentiment=round(overall_sentiment, 3),
            shipping_sentiment=round(shipping_sentiment, 3)
        )
        
        self._articles.extend(articles)
        self._last_fetch = now
        
        return snapshot
    
    async def _fetch_symbol_news(self, symbol: str) -> List[NewsArticle]:
        """Fetch news for a specific symbol."""
        session = await self._get_session()
        company_name = self._symbol_names.get(symbol, symbol)
        
        url = f"{self.config.base_url}/everything"
        params = {
            "q": f'"{company_name}" OR "{symbol}"',
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 10,
            "apiKey": self.config.api_key
        }
        
        try:
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    logger.warning(f"NewsAPI returned status {response.status} for {symbol}")
                    return []
                
                data = await response.json()
                
                if data.get("status") != "ok":
                    return []
                
                articles = []
                for item in data.get("articles", [])[:5]:
                    article = self._parse_article(item, symbol)
                    if article:
                        articles.append(article)
                
                return articles
                
        except Exception as e:
            logger.error(f"Error fetching news for {symbol}: {e}")
            return []
    
    async def _fetch_general_shipping_news(self) -> List[NewsArticle]:
        """Fetch general shipping/supply chain news."""
        session = await self._get_session()
        
        url = f"{self.config.base_url}/everything"
        params = {
            "q": "shipping OR \"supply chain\" OR logistics OR \"port congestion\" OR freight",
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 10,
            "apiKey": self.config.api_key
        }
        
        try:
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    return []
                
                data = await response.json()
                
                if data.get("status") != "ok":
                    return []
                
                articles = []
                for item in data.get("articles", [])[:5]:
                    article = self._parse_article(item, None)
                    if article:
                        article.relevance_score = max(article.relevance_score, 0.7)
                        articles.append(article)
                
                return articles
                
        except Exception as e:
            logger.error(f"Error fetching general shipping news: {e}")
            return []
    
    def _parse_article(self, item: dict, symbol: str = None) -> Optional[NewsArticle]:
        """Parse NewsAPI article into NewsArticle."""
        try:
            headline = item.get("title", "")
            description = item.get("description", "") or ""
            published_at = item.get("publishedAt", "")
            
            if published_at:
                timestamp = datetime.strptime(published_at, "%Y-%m-%dT%H:%M:%SZ")
            else:
                timestamp = datetime.now()
            
            # Simple sentiment based on keywords (FinBERT would be better)
            sentiment = self._simple_sentiment_analysis(headline + " " + description)
            
            # Relevance for shipping/logistics
            keywords = ["shipping", "logistics", "port", "cargo", "supply chain", 
                       "delivery", "freight", "transport", "warehouse"]
            relevance = 0.3 + 0.4 * sum(1 for kw in keywords if kw.lower() in headline.lower() or kw.lower() in description.lower())
            relevance = min(1.0, relevance)
            
            # Related symbols
            related_symbols = [symbol] if symbol else []
            for sym, name in self._symbol_names.items():
                if name.lower() in headline.lower() or name.lower() in description.lower():
                    related_symbols.append(sym)
            
            return NewsArticle(
                article_id=item.get("url", "") or f"NEWS-{timestamp.strftime('%Y%m%d')}-{len(self._articles)}",
                headline=headline,
                source=item.get("source", {}).get("name", "Unknown"),
                timestamp=timestamp,
                sentiment_score=round(sentiment, 3),
                relevance_score=round(relevance, 3),
                url=item.get("url", ""),
                summary=description[:200] if description else "",
                related_symbols=list(set(related_symbols))
            )
            
        except Exception as e:
            logger.warning(f"Error parsing article: {e}")
            return None
    
    def _simple_sentiment_analysis(self, text: str) -> float:
        """Simple keyword-based sentiment (placeholder for FinBERT)."""
        positive_words = ["beat", "growth", "gain", "rise", "surge", "strong", "positive", 
                         "profit", "upgrade", "bullish", "optimistic", "record"]
        negative_words = ["miss", "loss", "decline", "fall", "drop", "weak", "negative",
                         "deficit", "downgrade", "bearish", "pessimistic", "crash"]
        
        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        total = pos_count + neg_count
        if total == 0:
            return 0.0
        
        return (pos_count - neg_count) / total
    
    async def close(self):
        """Close HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()


class EDGARSource:
    """
    SEC EDGAR RSS feed parser for 8-K filings.
    
    Monitors material supply chain disclosures.
    """
    
    def __init__(self, config: EDGARConfig = None):
        self.config = config or EDGARConfig()
        self._session: Optional[aiohttp.ClientSession] = None
        
        # CIK mappings for tracked companies
        self._cik_mapping = {
            "FDX": "0000035259",
            "UPS": "0000109072",
            "CHRW": "0000868693",
            "DAL": "0000027904"
        }
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                headers={"User-Agent": "Magok AI Trading Bot (your@email.com)"}
            )
        return self._session
    
    async def fetch_8k_filings(self, symbols: List[str] = None) -> List[Dict[str, Any]]:
        """Fetch recent 8-K filings for specified symbols."""
        symbols = symbols or list(self._cik_mapping.keys())
        filings = []
        
        tasks = [self._fetch_symbol_8k(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, list):
                filings.extend(result)
            elif isinstance(result, Exception):
                logger.warning(f"EDGAR fetch error: {result}")
        
        return filings
    
    async def _fetch_symbol_8k(self, symbol: str) -> List[Dict[str, Any]]:
        """Fetch 8-K filings for a single symbol."""
        cik = self._cik_mapping.get(symbol)
        if not cik:
            return []
        
        session = await self._get_session()
        
        # EDGAR RSS feed URL for 8-K filings
        url = f"{self.config.base_url}?action=getcompany&CIK={cik}&type=8-K&dateb=&owner=exclude&count=5&output=atom"
        
        try:
            async with session.get(url) as response:
                if response.status != 200:
                    return []
                
                xml_content = await response.text()
                feed = feedparser.parse(xml_content)
                
                filings = []
                for entry in feed.entries[:3]:
                    filing = {
                        "symbol": symbol,
                        "cik": cik,
                        "title": entry.title,
                        "link": entry.link,
                        "published": entry.published,
                        "summary": entry.summary if hasattr(entry, 'summary') else "",
                        "filing_type": "8-K"
                    }
                    filings.append(filing)
                
                return filings
                
        except Exception as e:
            logger.error(f"Error fetching 8-K for {symbol}: {e}")
            return []
    
    async def parse_supply_chain_disclosures(self, filings: List[Dict[str, Any]]) -> List[NewsArticle]:
        """Parse 8-K filings for supply chain related disclosures."""
        articles = []
        
        supply_chain_keywords = [
            "supply chain", "logistics", "shipping", "freight", "transportation",
            "port", "warehouse", "distribution", "inventory", "delay", "disruption"
        ]
        
        for filing in filings:
            text = (filing.get("title", "") + " " + filing.get("summary", "")).lower()
            
            # Check if filing contains supply chain keywords
            relevance = sum(1 for kw in supply_chain_keywords if kw in text) / len(supply_chain_keywords)
            
            if relevance > 0.1:  # At least some relevance
                try:
                    timestamp = datetime.strptime(
                        filing.get("published", "")[:19],
                        "%Y-%m-%dT%H:%M:%S"
                    )
                except:
                    timestamp = datetime.now()
                
                article = NewsArticle(
                    article_id=f"EDGAR-{filing.get('cik', '')}-{timestamp.strftime('%Y%m%d')}",
                    headline=filing.get("title", "8-K Filing"),
                    source="SEC EDGAR",
                    timestamp=timestamp,
                    sentiment_score=-0.1 if "delay" in text or "disruption" in text else 0.0,
                    relevance_score=min(1.0, 0.5 + relevance),
                    url=filing.get("link", ""),
                    summary=filing.get("summary", "")[:200],
                    related_symbols=[filing.get("symbol")]
                )
                articles.append(article)
        
        return articles
    
    async def close(self):
        """Close HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()


def get_news_api_source(symbols: List[str] = None) -> NewsAPISourceReal:
    """Create NewsAPI source from environment variables."""
    api_key = os.getenv("NEWSAPI_KEY")
    
    if not api_key:
        raise ValueError("NEWSAPI_KEY must be set in .env")
    
    config = NewsAPIConfig(api_key=api_key)
    return NewsAPISourceReal(config=config, symbols=symbols)


def get_edgar_source() -> EDGARSource:
    """Create EDGAR source."""
    return EDGARSource()
