"""
Training Module.

Model training utilities.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from pathlib import Path
import json

from .model import TradingModel, generate_synthetic_training_data, STRATEGIES


class ModelTrainer:
    """
    Handles model training and evaluation.
    
    In production, this would use:
    - Historical labeled data
    - Cross-validation
    - Model selection
    - Online learning updates
    """
    
    def __init__(self, model: TradingModel = None):
        self.model = model or TradingModel()
        self._training_history: List[Dict] = []
    
    def train_initial(self, n_samples: int = 1000) -> Dict:
        """Train model with synthetic data."""
        X, y = generate_synthetic_training_data(n_samples)
        metrics = self.model.train(X, y)
        
        self._training_history.append({
            "timestamp": datetime.now().isoformat(),
            "type": "initial_synthetic",
            "n_samples": n_samples,
            "metrics": metrics
        })
        
        return metrics
    
    def retrain_with_feedback(self, X: np.ndarray, y: np.ndarray, 
                             incremental: bool = False) -> Dict:
        """
        Retrain model with new labeled samples.
        
        In production, this would implement:
        - Online learning (incremental updates)
        - Concept drift detection
        - Model versioning
        """
        if incremental:
            # Incremental update (partial_fit would be used in production)
            metrics = self._incremental_update(X, y)
        else:
            # Full retrain
            metrics = self.model.train(X, y)
        
        self._training_history.append({
            "timestamp": datetime.now().isoformat(),
            "type": "retrain",
            "incremental": incremental,
            "n_samples": len(y),
            "metrics": metrics
        })
        
        return metrics
    
    def _incremental_update(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Perform incremental update (placeholder)."""
        # In production, use partial_fit or online learning
        # For now, combine with existing data and retrain
        X_old, y_old = generate_synthetic_training_data(200)
        X_combined = np.vstack([X_old, X])
        y_combined = np.concatenate([y_old, y])
        
        return self.model.train(X_combined, y_combined)
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, 
                      n_folds: int = 5) -> Dict:
        """
        Perform cross-validation.
        
        Note: This is a simplified version. In production,
        use sklearn's cross_validate.
        """
        from sklearn.model_selection import cross_val_score
        
        scores = cross_val_score(
            self.model.model, X, y,
            cv=n_folds,
            scoring='accuracy'
        )
        
        return {
            "mean_accuracy": float(np.mean(scores)),
            "std_accuracy": float(np.std(scores)),
            "fold_scores": [float(s) for s in scores]
        }
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Evaluate model on test set."""
        from sklearn.metrics import classification_report, confusion_matrix
        
        predictions = self.model.model.predict(X_test)
        
        # Get probabilities for confidence
        probs = self.model.model.predict_proba(X_test)
        avg_confidence = float(np.mean(np.max(probs, axis=1)))
        
        return {
            "accuracy": float((predictions == y_test).mean()),
            "avg_confidence": avg_confidence,
            "n_test_samples": len(y_test),
            "class_distribution": {
                str(cls): int((y_test == cls).sum())
                for cls in np.unique(y_test)
            }
        }
    
    def save_training_history(self, path: str):
        """Save training history to file."""
        history_path = Path(path)
        history_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(history_path, 'w') as f:
            json.dump(self._training_history, f, indent=2)
    
    def get_training_history(self) -> List[Dict]:
        """Get training history."""
        return self._training_history


class LabelGenerator:
    """
    Generates labels from historical data.
    
    In production, this would use:
    - Backtest results as labels
    - Expert annotations
    - Outcome-based labeling
    """
    
    def __init__(self, profit_threshold: float = 0.01, loss_threshold: float = -0.005):
        self.profit_threshold = profit_threshold
        self.loss_threshold = loss_threshold
    
    def generate_labels_from_outcomes(self, 
                                      features: List,
                                      outcomes: List[float]) -> np.ndarray:
        """
        Generate labels based on trade outcomes.
        
        Positive outcome -> momentum
        Negative outcome -> mean_reversion (reversal)
        Neutral -> no_trade
        """
        labels = []
        
        for outcome in outcomes:
            if outcome > self.profit_threshold:
                labels.append("momentum")
            elif outcome < self.loss_threshold:
                labels.append("mean_reversion")
            else:
                labels.append("no_trade")
        
        return np.array(labels)
    
    def generate_labels_from_heuristics(self, features_list: List) -> np.ndarray:
        """
        Generate labels using heuristic rules.
        
        This is a simplified version for initial training.
        """
        labels = []
        
        for features in features_list:
            # Use momentum score as main factor
            if features.momentum_score > 0.7 and features.volume_ratio > 1.3:
                labels.append("momentum")
            elif features.mean_reversion_score > 0.7:
                labels.append("mean_reversion")
            elif features.shipping_impact > 1.2:
                labels.append("latency")
            else:
                labels.append("no_trade")
        
        return np.array(labels)


# Singleton trainer
_trainer = None

def get_trainer(model: TradingModel = None) -> ModelTrainer:
    """Get or create the model trainer singleton."""
    global _trainer
    if _trainer is None:
        _trainer = ModelTrainer(model)
    return _trainer
