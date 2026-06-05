"""
ML Trainer Module.

Handles model retraining with weekly scheduling using logged trade data.
"""

import os
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report


class ModelTrainer:
    """
    Trainer for ML models with scheduled retraining.
    
    Features:
    - Weekly retraining scheduler
    - Uses last 90 days of logged trades
    - Versioned model storage
    - Performance tracking
    """
    
    def __init__(self, model_dir: str = "ml/models", lookback_days: int = 90):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.lookback_days = lookback_days
        
        # Feature names (must match feature builder output)
        self.feature_names = [
            "price_change_short", "price_change_medium", "volume_ratio",
            "rsi", "macd_histogram", "event_severity", "time_since_event",
            "news_sentiment", "shipping_impact", "momentum_score",
            "mean_reversion_score", "trend_strength"
        ]
        
        # Strategy labels
        self.strategies = ["no_trade", "momentum", "latency", "mean_reversion"]
    
    def prepare_training_data(self, trade_history: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data from trade history.
        
        Args:
            trade_history: List of trade records with features and outcomes
        
        Returns:
            X: Feature matrix
            y: Strategy labels
        """
        X = []
        y = []
        
        for trade in trade_history:
            # Extract features
            features = self._extract_features_from_trade(trade)
            if features is None:
                continue
            
            # Determine optimal strategy label based on trade outcome
            # This is a simplified approach - in production, use more sophisticated labeling
            label = self._determine_optimal_strategy(trade)
            
            X.append(features)
            y.append(label)
        
        return np.array(X), np.array(y)
    
    def _extract_features_from_trade(self, trade: Dict) -> Optional[List[float]]:
        """Extract feature vector from trade record."""
        try:
            features = [
                trade.get('price_change_short', 0.0),
                trade.get('price_change_medium', 0.0),
                trade.get('volume_ratio', 1.0),
                trade.get('rsi', 50.0),
                trade.get('macd_histogram', 0.0),
                trade.get('event_severity', 0.0),
                trade.get('time_since_event', 600),
                trade.get('news_sentiment', 0.0),
                trade.get('shipping_impact', 1.0),
                trade.get('momentum_score', 0.5),
                trade.get('mean_reversion_score', 0.5),
                trade.get('trend_strength', 0.5)
            ]
            return features
        except Exception:
            return None
    
    def _determine_optimal_strategy(self, trade: Dict) -> str:
        """
        Determine which strategy would have been optimal for this trade.
        
        Simplified heuristic based on market conditions and trade outcome.
        """
        pnl = trade.get('pnl', 0.0)
        rsi = trade.get('rsi', 50.0)
        momentum_score = trade.get('momentum_score', 0.5)
        event_severity = trade.get('event_severity', 0.0)
        
        # Heuristic labeling
        if event_severity < 0.3 and abs(pnl) < 100:
            return "no_trade"
        elif momentum_score > 0.7 and pnl > 0:
            return "momentum"
        elif event_severity > 0.7 and 'shipping' in str(trade.get('strategy', '')).lower():
            return "latency"
        elif (rsi < 35 or rsi > 65) and pnl > 0:
            return "mean_reversion"
        elif momentum_score > 0.5:
            return "momentum"
        else:
            return "mean_reversion"
    
    def train_model(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Train a new model on the provided data.
        
        Args:
            X: Feature matrix
            y: Strategy labels
        
        Returns:
            Training metrics
        """
        if len(X) < 50:
            raise ValueError(f"Insufficient training data: {len(X)} samples (need at least 50)")
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
        )
        
        model.fit(X_scaled, y)
        
        # Calculate metrics
        predictions = model.predict(X_scaled)
        accuracy = accuracy_score(y, predictions)
        
        return {
            'model': model,
            'scaler': scaler,
            'accuracy': accuracy,
            'n_samples': len(y),
            'classes': list(model.classes_),
            'timestamp': datetime.now()
        }
    
    def save_model(self, trained_model: Dict, version: str = None) -> str:
        """
        Save trained model to disk with versioning.
        
        Args:
            trained_model: Dict containing model, scaler, and metadata
            version: Optional version string (defaults to timestamp)
        
        Returns:
            Path to saved model
        """
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        model_path = self.model_dir / f"{version}.pkl"
        
        with open(model_path, 'wb') as f:
            pickle.dump(trained_model, f)
        
        # Also save as latest
        latest_path = self.model_dir / "latest.pkl"
        with open(latest_path, 'wb') as f:
            pickle.dump(trained_model, f)
        
        return str(model_path)
    
    def load_model(self, version: str = "latest") -> Optional[Dict]:
        """
        Load a model from disk.
        
        Args:
            version: Version to load (default: latest)
        
        Returns:
            Loaded model dict or None if not found
        """
        if version == "latest":
            model_path = self.model_dir / "latest.pkl"
        else:
            model_path = self.model_dir / f"{version}.pkl"
        
        if not model_path.exists():
            return None
        
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    
    def should_retrain(self, last_train_time: datetime) -> bool:
        """Check if retraining is due (weekly schedule)."""
        if last_train_time is None:
            return True
        
        next_train_time = last_train_time + timedelta(days=7)
        return datetime.now() >= next_train_time
    
    def get_training_data_from_db(self, db) -> List[Dict]:
        """
        Get training data from database.
        
        Args:
            db: PositionDatabase instance
        
        Returns:
            List of trade records
        """
        cutoff_date = datetime.now() - timedelta(days=self.lookback_days)
        trade_history = db.get_trade_history(limit=1000)
        
        # Filter by date
        filtered_trades = []
        for trade in trade_history:
            exit_time = trade.get('exit_time')
            if exit_time:
                if isinstance(exit_time, str):
                    exit_time = datetime.fromisoformat(exit_time)
                if exit_time >= cutoff_date:
                    filtered_trades.append(trade)
        
        return filtered_trades
    
    def run_training_cycle(self, db) -> Optional[str]:
        """
        Run a complete training cycle.
        
        Args:
            db: PositionDatabase instance
        
        Returns:
            Path to saved model or None if training failed
        """
        try:
            # Get training data
            trade_history = self.get_training_data_from_db(db)
            
            if len(trade_history) < 50:
                print(f"Insufficient trade history for training: {len(trade_history)} trades")
                return None
            
            # Prepare data
            X, y = self.prepare_training_data(trade_history)
            
            if len(X) < 50:
                print(f"Insufficient features after preprocessing: {len(X)} samples")
                return None
            
            # Train model
            trained_model = self.train_model(X, y)
            
            # Save model
            model_path = self.save_model(trained_model)
            
            print(f"Model trained successfully:")
            print(f"  - Accuracy: {trained_model['accuracy']:.2%}")
            print(f"  - Samples: {trained_model['n_samples']}")
            print(f"  - Classes: {trained_model['classes']}")
            print(f"  - Saved to: {model_path}")
            
            return model_path
            
        except Exception as e:
            print(f"Training cycle failed: {e}")
            return None


# Singleton instance
_trainer_instance = None


def get_model_trainer(model_dir: str = "ml/models", lookback_days: int = 90) -> ModelTrainer:
    """Get or create the model trainer singleton."""
    global _trainer_instance
    if _trainer_instance is None:
        _trainer_instance = ModelTrainer(model_dir, lookback_days)
    return _trainer_instance
