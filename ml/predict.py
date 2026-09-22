"""
Prediction Module - Phase 2.

High-level interface for model predictions with ensemble support and SHAP explainability.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field

from .model import TradingModel, ModelPrediction, get_trading_model, STRATEGIES
from .ensemble import EnsembleTradingModel, EnsemblePrediction, get_ensemble_model
from .explainer import SHAPExplainer, FeatureAttribution, ExplanationResult, get_shap_explainer
from features.builder import FeatureVector


@dataclass
class TradingDecision:
    """Trading decision based on model prediction with full attribution."""
    symbol: str
    recommended_strategy: str
    confidence: float
    should_trade: bool
    direction: str  # "long", "short", "neutral"
    reasons: List[str]
    alternative_strategies: Dict[str, float]  # strategy -> confidence
    feature_attributions: Dict[str, float] = field(default_factory=dict)  # Top 5 features with SHAP values
    xgb_confidence: float = 0.0
    lstm_confidence: float = 0.0
    finbert_confidence: float = 0.0
    explanation_summary: str = ""


class PredictionEngine:
    """
    Engine for generating trading decisions from features.
    
    Supports both single RandomForest and 3-model ensemble.
    Wraps ML model with additional logic for:
    - Confidence thresholds
    - Strategy filtering
    - Direction determination
    - SHAP feature attribution
    """
    
    def __init__(self, 
                 confidence_threshold: float = 0.6,
                 use_ensemble: bool = True):
        self.confidence_threshold = confidence_threshold
        self.use_ensemble = use_ensemble
        
        if use_ensemble:
            self.ensemble_model = get_ensemble_model()
            self.single_model = None
        else:
            self.single_model = get_trading_model()
            self.ensemble_model = None
        
        self.explainer = get_shap_explainer()
    
    def make_decision(self, 
                     features: FeatureVector,
                     prices_history: Optional[List[float]] = None,
                     volumes_history: Optional[List[float]] = None,
                     news_text: str = "") -> TradingDecision:
        """Make a trading decision for a symbol's features."""
        
        if self.use_ensemble and self.ensemble_model is not None:
            # Use ensemble model
            if prices_history is None or volumes_history is None:
                # Fallback to single model if no sequential data
                return self._make_single_decision(features)
            
            # Convert histories to numpy arrays
            import numpy as np
            prices = np.array(prices_history[-60:]) if len(prices_history) >= 60 else None
            volumes = np.array(volumes_history[-60:]) if len(volumes_history) >= 60 else None
            
            if prices is None or volumes is None or len(prices) < 60:
                return self._make_single_decision(features)
            
            # Get ensemble prediction
            tabular_features = features.to_array()
            prediction = self.ensemble_model.predict(
                tabular_features=tabular_features,
                prices=prices,
                volumes=volumes,
                news_text=news_text if news_text else "No news"
            )
            
            # Determine if we should trade
            should_trade = (
                prediction.strategy != "no_trade" and
                prediction.confidence >= self.confidence_threshold
            )
            
            # Determine direction
            direction = self._determine_direction(features, type('obj', (object,), {
                'strategy': prediction.strategy,
                'confidence': prediction.confidence,
                'probabilities': prediction.probabilities
            })())
            
            # Generate reasons
            reasons = self._generate_reasons(features, type('obj', (object,), {
                'strategy': prediction.strategy,
                'confidence': prediction.confidence,
                'probabilities': prediction.probabilities
            })())
            
            # Get SHAP attribution
            feature_attribs = self._get_feature_attribution(
                features, prediction.strategy, prediction.confidence
            )
            
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
                alternative_strategies=alternatives,
                feature_attributions=feature_attribs,
                xgb_confidence=prediction.xgb_confidence,
                lstm_confidence=prediction.lstm_confidence,
                finbert_confidence=prediction.finbert_confidence,
                explanation_summary=feature_attribs.get('summary', '') if isinstance(feature_attribs, dict) else ''
            )
        else:
            return self._make_single_decision(features)
    
    def _make_single_decision(self, features: FeatureVector) -> TradingDecision:
        """Make decision using single RandomForest model."""
        # Get model prediction
        prediction = self.single_model.predict_from_features(features)
        
        # Determine if we should trade
        should_trade = (
            prediction.strategy != "no_trade" and
            prediction.confidence >= self.confidence_threshold
        )
        
        # Determine direction
        direction = self._determine_direction(features, prediction)
        
        # Generate reasons
        reasons = self._generate_reasons(features, prediction)
        
        # Get SHAP attribution
        feature_attribs = self._get_feature_attribution(
            features, prediction.strategy, prediction.confidence
        )
        
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
            alternative_strategies=alternatives,
            feature_attributions=feature_attribs.get('attributions', {}) if isinstance(feature_attribs, dict) else {},
            explanation_summary=feature_attribs.get('summary', '') if isinstance(feature_attribs, dict) else ''
        )
    
    def _get_feature_attribution(self, 
                                features: FeatureVector,
                                strategy: str,
                                confidence: float) -> Dict:
        """Get SHAP-based feature attribution."""
        try:
            X = features.to_array().reshape(1, -1)
            result = self.explainer.explain_prediction(
                features=X,
                predicted_strategy=strategy,
                confidence=confidence,
                symbol=features.symbol
            )
            
            attributions = {
                attr.feature_name: attr.shap_value 
                for attr in result.feature_attributions[:5]
            }
            
            return {
                'attributions': attributions,
                'summary': result.summary
            }
        except Exception as e:
            # Return empty attribution on error
            return {'attributions': {}, 'summary': f'Attribution failed: {str(e)}'}
    
    def make_decisions(self, features_dict: Dict[str, FeatureVector],
                      prices_history: Optional[Dict[str, List[float]]] = None,
                      volumes_history: Optional[Dict[str, List[float]]] = None,
                      news_texts: Optional[Dict[str, str]] = None) -> Dict[str, TradingDecision]:
        """Make trading decisions for all symbols."""
        decisions = {}
        
        prices_history = prices_history or {}
        volumes_history = volumes_history or {}
        news_texts = news_texts or {}
        
        for symbol, features in features_dict.items():
            decisions[symbol] = self.make_decision(
                features=features,
                prices_history=prices_history.get(symbol),
                volumes_history=volumes_history.get(symbol),
                news_text=news_texts.get(symbol, "")
            )
        
        return decisions
    
    def _determine_direction(self, features: FeatureVector, 
                           prediction) -> str:
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
                         prediction) -> List[str]:
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


def get_prediction_engine(confidence_threshold: float = 0.6, 
                         use_ensemble: bool = True) -> PredictionEngine:
    """Get or create the prediction engine singleton."""
    global _prediction_engine
    if _prediction_engine is None:
        _prediction_engine = PredictionEngine(confidence_threshold, use_ensemble)
    return _prediction_engine
