"""
News Data API Source.

Mock implementation for news and sentiment data.
"""

import random
from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime
import re


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


class NewsAPISource:
    """
    Mock news data API.
    
    In production, this would connect to news APIs like:
    - NewsAPI
    - Alpha Vantage News
    - Finnhub
    - Bloomberg
    """
    
    def __init__(self, use_mock: bool = True):
        self.use_mock = use_mock
        self._article_counter = 0
        self._headline_templates = [
            ("{company} Reports Strong Quarterly Earnings, Beats Estimates", 0.3, ["FDX", "UPS", "DAL", "CHRW"]),
            ("Shipping Industry Faces New Regulatory Challenges", -0.2, ["FDX", "UPS", "IYT"]),
            ("Global Supply Chain Disruptions Impact Retailers", -0.3, ["FDX", "UPS", "CHRW"]),
            ("{location} Port Congestion Reaches Record Levels", -0.2, ["FDX", "UPS", "DAL"]),
            ("{company} Announces New Route Expansion", 0.2, ["FDX", "UPS", "DAL"]),
            ("Oil Prices Surge Amid Middle East Tensions", -0.1, ["DAL", "FDX"]),
            ("E-Commerce Growth Boosts Logistics Sector", 0.4, ["FDX", "UPS", "CHRW"]),
            ("Air Cargo Demand Reaches Pre-Pandemic Levels", 0.3, ["FDX", "UPS", "DAL"]),
            ("Supply Chain Technology Investment Increases", 0.2, ["FDX", "UPS", "CHRW"]),
            ("FedEx and UPS Announce Fuel Surcharge Changes", -0.1, ["FDX", "UPS"]),
            ("Labor Shortages Affect Major Shipping Hubs", -0.3, ["FDX", "UPS", "CHRW"]),
            ("{location} Weather Disrupts Air Travel", -0.2, ["DAL", "FDX"]),
            ("Logistics Stocks Rally on Strong Holiday Forecast", 0.3, ["FDX", "UPS", "IYT"]),
            ("Port Authority Announces Infrastructure Upgrades", 0.1, ["FDX", "UPS"]),
            ("Trade War Concerns Weigh on Shipping Sector", -0.2, ["FDX", "UPS", "DAL"]),
        ]
        self._locations = [
            "Shanghai", "Los Angeles", "Rotterdam", "Singapore", "Hong Kong",
            "Hamburg", "Busan", "Dubai", "New York", "Tokyo"
        ]
        self._sources = [
            "Reuters", "Bloomberg", "CNBC", "Wall Street Journal",
            "Financial Times", "Associated Press", "MarketWatch"
        ]
    
    def fetch_news(self, symbols: List[str] = None) -> NewsSnapshot:
        """Fetch latest news articles."""
        if self.use_mock:
            return self._generate_mock_news(symbols)
        else:
            return self._fetch_real_news(symbols)
    
    def _generate_mock_news(self, symbols: List[str] = None) -> NewsSnapshot:
        """Generate realistic mock news data."""
        now = datetime.now()
        articles = []
        
        # Generate 0-3 articles per fetch cycle
        num_articles = random.randint(0, 3)
        
        for _ in range(num_articles):
            article = self._generate_article(now)
            if symbols:
                # Filter by relevance
                if any(s in article.related_symbols for s in symbols):
                    articles.append(article)
            else:
                articles.append(article)
        
        # Calculate overall sentiment
        if articles:
            overall = sum(a.sentiment_score for a in articles) / len(articles)
            shipping_articles = [a for a in articles if a.relevance_score > 0.5]
            shipping_sent = sum(a.sentiment_score for a in shipping_articles) / len(shipping_articles) if shipping_articles else 0
        else:
            overall = 0.0
            shipping_sent = 0.0
        
        return NewsSnapshot(
            timestamp=now,
            articles=articles,
            overall_sentiment=round(overall, 3),
            shipping_sentiment=round(shipping_sent, 3)
        )
    
    def _generate_article(self, timestamp: datetime) -> NewsArticle:
        """Generate a single mock article."""
        self._article_counter += 1
        
        template, base_sentiment, related = random.choice(self._headline_templates)
        
        # Fill in template variables
        companies = ["FedEx", "UPS", "Delta", "C.H. Robinson"]
        location = random.choice(self._locations)
        
        headline = template.format(
            company=random.choice(companies),
            location=location
        )
        
        # Add some variation
        if random.random() < 0.3:
            headline = headline + " in " + location
        
        # Calculate sentiment with noise
        sentiment = base_sentiment + random.uniform(-0.1, 0.1)
        sentiment = max(-1.0, min(1.0, sentiment))
        
        # Relevance for shipping/logistics
        keywords = ["shipping", "logistics", "port", "cargo", "supply chain", "delivery"]
        relevance = 0.3 + 0.4 * sum(1 for kw in keywords if kw.lower() in headline.lower())
        relevance = min(1.0, relevance)
        
        # Add related symbols
        related_symbols = [s for s in related if s in ["FDX", "UPS", "DAL", "CHRW", "IYT"]]
        if not related_symbols:
            related_symbols = random.sample(["FDX", "UPS", "DAL", "CHRW", "IYT"], k=2)
        
        return NewsArticle(
            article_id=f"NEWS-{timestamp.strftime('%Y%m%d')}-{self._article_counter:05d}",
            headline=headline,
            source=random.choice(self._sources),
            timestamp=timestamp,
            sentiment_score=round(sentiment, 3),
            relevance_score=round(relevance, 3),
            url=f"https://example.com/news/{self._article_counter}",
            summary=f"Brief summary of: {headline}",
            related_symbols=related_symbols
        )
    
    def _fetch_real_news(self, symbols: List[str] = None) -> NewsSnapshot:
        """
        Placeholder for real API integration.
        
        In production, implement actual API calls here.
        """
        raise NotImplementedError("Real API not implemented. Set use_mock=True.")
    
    def get_sentiment_for_symbol(self, symbol: str, lookback_hours: int = 24) -> float:
        """
        Get aggregated sentiment for a symbol over a time period.
        Returns average sentiment score between -1 and 1.
        """
        snapshot = self.fetch_news(symbols=[symbol])
        
        if not snapshot.articles:
            return 0.0
        
        # Weighted by recency
        now = datetime.now()
        total_weight = 0.0
        weighted_sentiment = 0.0
        
        for article in snapshot.articles:
            age_hours = (now - article.timestamp).total_seconds() / 3600
            if age_hours <= lookback_hours:
                weight = 1.0 / (1.0 + age_hours)  # More recent = higher weight
                weighted_sentiment += article.sentiment_score * weight * article.relevance_score
                total_weight += weight
        
        if total_weight > 0:
            return round(weighted_sentiment / total_weight, 3)
        return 0.0


# Singleton instance
_news_source = None

def get_news_source(use_mock: bool = True) -> NewsAPISource:
    """Get or create the news source singleton."""
    global _news_source
    if _news_source is None:
        _news_source = NewsAPISource(use_mock=use_mock)
    return _news_source
