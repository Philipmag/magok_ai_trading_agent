"""
ML Model Module.

Machine learning model for strategy selection.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import pickle
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


# Strategy labels
STRATEGIES = ["no_trade", "momentum", "latency", "mean_reversion"]


@dataclass
class ModelPrediction:
    """Model prediction result."""
    strategy: str
    confidence: float  # 0-1
    probabilities: Dict[str, float]
    timestamp: datetime


class TradingModel:
    """
    ML model for selecting trading strategies.
    
    Uses RandomForest classifier to predict best strategy
    based on feature vectors.
    """
    
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        self.scaler = StandardScaler()
        self._is_trained = False
        self._feature_names = [
            "price_change_short", "price_change_medium", "volume_ratio",
            "rsi", "macd_histogram", "event_severity", "time_since_event",
            "news_sentiment", "shipping_impact", "momentum_score",
            "mean_reversion_score", "trend_strength"
        ]
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Train the model on labeled data."""
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        self._is_trained = True
        
        # Calculate training metrics
        train_score = self.model.score(X_scaled, y)
        
        return {
            "train_accuracy": train_score,
            "n_samples": len(y),
            "n_features": X.shape[1],
            "classes": list(self.model.classes_)
        }
    
    def predict(self, X: np.ndarray) -> ModelPrediction:
        """Make prediction on feature vector(s)."""
        if not self._is_trained:
            # Return default prediction
            return ModelPrediction(
                strategy="no_trade",
                confidence=0.5,
                probabilities={s: 0.25 for s in STRATEGIES},
                timestamp=datetime.now()
            )
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Get prediction
        predicted_class = self.model.predict(X_scaled)[0]
        
        # Get probabilities
        probs = self.model.predict_proba(X_scaled)[0]
        prob_dict = {
            cls: float(prob) for cls, prob in zip(self.model.classes_, probs)
        }
        
        # Ensure all strategies have probabilities
        for strategy in STRATEGIES:
            if strategy not in prob_dict:
                prob_dict[strategy] = 0.0
        
        confidence = float(max(prob_dict.values()))
        predicted_strategy = str(predicted_class)
        
        return ModelPrediction(
            strategy=predicted_strategy,
            confidence=confidence,
            probabilities=prob_dict,
            timestamp=datetime.now()
        )
    
    def predict_from_features(self, features) -> ModelPrediction:
        """Make prediction from feature vector object."""
        X = features.to_array().reshape(1, -1)
        return self.predict(X)
    
    def save(self, path: str):
        """Save model to disk."""
        model_path = Path(path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'is_trained': self._is_trained,
                'feature_names': self._feature_names
            }, f)
    
    def load(self, path: str):
        """Load model from disk."""
        model_path = Path(path)
        if not model_path.exists():
            return False
        
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.scaler = data['scaler']
            self._is_trained = data['is_trained']
            self._feature_names = data['feature_names']
        
        return True
    
    @property
    def is_trained(self) -> bool:
        return self._is_trained
    
    @property
    def feature_names(self) -> List[str]:
        return self._feature_names


def generate_synthetic_training_data(n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic training data for initial model.
    
    In production, this would be replaced with real historical data.
    """
    X = []
    y = []
    
    for _ in range(n_samples):
        # Generate random features
        price_change_short = np.random.normal(0, 2)
        price_change_medium = np.random.normal(0, 5)
        volume_ratio = np.random.exponential(1)
        rsi = np.random.normal(50, 15)
        macd_histogram = np.random.normal(0, 0.5)
        event_severity = np.random.uniform(0, 1)
        time_since_event = np.random.exponential(600)
        news_sentiment = np.random.normal(0, 0.3)
        shipping_impact = np.random.uniform(0.5, 1.5)
        momentum_score = np.random.uniform(0, 1)
        mean_reversion_score = np.random.uniform(0, 1)
        trend_strength = np.random.uniform(0, 1)
        
        features = [
            price_change_short, price_change_medium, volume_ratio,
            rsi, macd_histogram, event_severity, time_since_event,
            news_sentiment, shipping_impact, momentum_score,
            mean_reversion_score, trend_strength
        ]
        
        # Determine label based on heuristics
        if event_severity < 0.3 and abs(price_change_short) < 1:
            label = "no_trade"
        elif rsi > 60 and price_change_short > 1 and volume_ratio > 1.2:
            label = "momentum"
        elif shipping_impact > 1.1 and abs(price_change_medium) > 3:
            label = "latency"
        elif (rsi < 35 or rsi > 65) and volume_ratio > 1.5:
            label = "mean_reversion"
        else:
            # Use momentum as default
            if momentum_score > mean_reversion_score:
                label = "momentum"
            else:
                label = "mean_reversion"
        
        X.append(features)
        y.append(label)
    
    return np.array(X), np.array(y)


# Singleton
_trading_model = None

def get_trading_model() -> TradingModel:
    """Get or create the trading model singleton."""
    global _trading_model
    if _trading_model is None:
        _trading_model = TradingModel()
        # Train with synthetic data initially
        X, y = generate_synthetic_training_data(500)
        _trading_model.train(X, y)
    return _trading_model
