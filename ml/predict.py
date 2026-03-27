"""
Prediction Module.

High-level interface for model predictions.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass

from .model import TradingModel, ModelPrediction, get_trading_model, STRATEGIES
from features.builder import FeatureVector


@dataclass
class TradingDecision:
    """Trading decision based on model prediction."""
    symbol: str
    recommended_strategy: str
    confidence: float
    should_trade: bool
    direction: str  # "long", "short", "neutral"
    reasons: List[str]
    alternative_strategies: Dict[str, float]  # strategy -> confidence


class PredictionEngine:
    """
    Engine for generating trading decisions from features.
    
    Wraps the ML model with additional logic for:
    - Confidence thresholds
    - Strategy filtering
    - Direction determination
    """
    
    def __init__(self, confidence_threshold: float = 0.6):
        self.model = get_trading_model()
        self.confidence_threshold = confidence_threshold
    
    def make_decision(self, features: FeatureVector) -> TradingDecision:
        """Make a trading decision for a symbol's features."""
        # Get model prediction
        prediction = self.model.predict_from_features(features)
        
        # Determine if we should trade
        should_trade = (
            prediction.strategy != "no_trade" and
            prediction.confidence >= self.confidence_threshold
        )
        
        # Determine direction
        direction = self._determine_direction(features, prediction)
        
        # Generate reasons
        reasons = self._generate_reasons(features, prediction)
        
        # Get alternative strategies
        alternatives = {
            k: v for k, v in prediction.probabilities.items()
            if k != prediction.strategy and v > 0.2
        }
        
        return TradingDecision(
            symbol=features.symbol,
            recommended_strategy=prediction.strategy,
            confidence=prediction.confidence,
            should_trade=should_trade,
            direction=direction,
            reasons=reasons,
            alternative_strategies=alternatives
        )
    
    def make_decisions(self, features_dict: Dict[str, FeatureVector]) -> Dict[str, TradingDecision]:
        """Make trading decisions for all symbols."""
        decisions = {}
        
        for symbol, features in features_dict.items():
            decisions[symbol] = self.make_decision(features)
        
        return decisions
    
    def _determine_direction(self, features: FeatureVector, 
                           prediction: ModelPrediction) -> str:
        """Determine trade direction based on features."""
        if prediction.strategy == "no_trade":
            return "neutral"
        
        if prediction.strategy == "momentum":
            # Follow momentum direction
            if features.price_change_short > 0 and features.rsi > 50:
                return "long"
            elif features.price_change_short < 0 and features.rsi < 50:
                return "short"
            return "long" if features.momentum_score > 0.5 else "neutral"
        
        elif prediction.strategy == "mean_reversion":
            # Mean reversion: trade against extremes
            if features.rsi < 35:
                return "long"  # Oversold - expect bounce
            elif features.rsi > 65:
                return "short"  # Overbought - expect pullback
            return "neutral"
        
        elif prediction.strategy == "latency":
            # Latency: follow shipping impact
            if features.shipping_impact > 1.0:
                return "long" if features.price_change_medium < 0 else "neutral"
            return "neutral"
        
        # Default
        return "neutral"
    
    def _generate_reasons(self, features: FeatureVector,
                         prediction: ModelPrediction) -> List[str]:
        """Generate human-readable reasons for the decision."""
        reasons = []
        
        # Strategy-specific reasons
        if prediction.strategy == "momentum":
            if features.volume_ratio > 1.5:
                reasons.append(f"High volume ({features.volume_ratio:.1f}x average)")
            if features.rsi > 55:
                reasons.append(f"Strong RSI ({features.rsi:.1f})")
            if features.price_change_short > 1:
                reasons.append(f"Price breaking out ({features.price_change_short:.1f}%)")
        
        elif prediction.strategy == "mean_reversion":
            if features.rsi < 35:
                reasons.append(f"Oversold condition (RSI: {features.rsi:.1f})")
            elif features.rsi > 65:
                reasons.append(f"Overbought condition (RSI: {features.rsi:.1f})")
            if abs(features.bollinger_position - 0.5) > 0.3:
                reasons.append("Price at Bollinger Band extreme")
        
        elif prediction.strategy == "latency":
            reasons.append(f"High shipping impact ({features.shipping_impact:.2f})")
            reasons.append(f"Port congestion: {features.port_congestion:.1%}")
        
        # General factors
        if features.event_severity > 0.5:
            reasons.append(f"Significant event activity (severity: {features.event_severity:.2f})")
        
        if features.news_sentiment > 0.2:
            reasons.append("Positive news sentiment")
        elif features.news_sentiment < -0.2:
            reasons.append("Negative news sentiment")
        
        if not reasons:
            reasons.append(f"Model confident in {prediction.strategy} strategy")
        
        return reasons


# Singleton
_prediction_engine = None

def get_prediction_engine(confidence_threshold: float = 0.6) -> PredictionEngine:
    """Get or create the prediction engine singleton."""
    global _prediction_engine
    if _prediction_engine is None:
        _prediction_engine = PredictionEngine(confidence_threshold)
    return _prediction_engine
