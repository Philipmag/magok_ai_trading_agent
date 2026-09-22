"""
ML Ensemble Model Module (Phase 2).

3-model ensemble for strategy selection:
- XGBoost (tabular features)
- LSTM (sequential price + volume)
- FinBERT (news/sentiment text)
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import pickle
from pathlib import Path
import xgboost as xgb
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import StandardScaler
import joblib


# Strategy labels
STRATEGIES = ["no_trade", "momentum", "latency", "mean_reversion"]


@dataclass
class EnsemblePrediction:
    """Ensemble prediction result with individual model confidences."""
    strategy: str
    confidence: float  # Weighted average 0-1
    probabilities: Dict[str, float]
    xgb_confidence: float
    lstm_confidence: float
    finbert_confidence: float
    timestamp: datetime


class XGBoostModel:
    """
    XGBoost classifier for tabular features.
    
    Features: RSI, volume, shipping severity, price momentum, etc.
    """
    
    def __init__(self):
        self.model: Optional[xgb.XGBClassifier] = None
        self.scaler = StandardScaler()
        self._is_trained = False
        self._feature_names = [
            "price_change_short", "price_change_medium", "volume_ratio",
            "rsi", "macd_histogram", "event_severity", "time_since_event",
            "news_sentiment", "shipping_impact", "momentum_score",
            "mean_reversion_score", "trend_strength", "bollinger_position",
            "atr_ratio", "volume_trend"
        ]
    
    def train(self, X: np.ndarray, y: np.ndarray, 
              eval_set: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Dict:
        """Train XGBoost model."""
        X_scaled = self.scaler.fit_transform(X)
        
        self.model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='mlogloss',
            early_stopping_rounds=20,
            use_label_encoder=False
        )
        
        if eval_set is not None:
            X_eval, y_eval = eval_set
            X_eval_scaled = self.scaler.transform(X_eval)
            self.model.fit(
                X_scaled, y,
                eval_set=[(X_eval_scaled, y_eval)],
                verbose=False
            )
        else:
            self.model.fit(X_scaled, y, verbose=False)
        
        self._is_trained = True
        
        train_score = self.model.score(X_scaled, y)
        
        return {
            "train_accuracy": train_score,
            "n_samples": len(y),
            "n_features": X.shape[1],
            "best_iteration": getattr(self.model, 'best_iteration', None)
        }
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability predictions."""
        if not self._is_trained or self.model is None:
            return np.ones((len(X), len(STRATEGIES))) / len(STRATEGIES)
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def save(self, path: str):
        """Save model to disk."""
        model_path = Path(path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'is_trained': self._is_trained,
            'feature_names': self._feature_names
        }, model_path)
    
    def load(self, path: str) -> bool:
        """Load model from disk."""
        model_path = Path(path)
        if not model_path.exists():
            return False
        
        data = joblib.load(model_path)
        self.model = data['model']
        self.scaler = data['scaler']
        self._is_trained = data['is_trained']
        self._feature_names = data['feature_names']
        return True
    
    @property
    def feature_names(self) -> List[str]:
        return self._feature_names


class SequenceDataset(Dataset):
    """PyTorch Dataset for sequential data."""
    
    def __init__(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y) if y is not None else None
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx], None


class LSTMModel(nn.Module):
    """
    LSTM network for sequential price + volume features.
    
    Input: 60-bar lookback window of [price, volume, returns]
    """
    
    def __init__(self, input_size: int = 3, hidden_size: int = 64, 
                 num_layers: int = 2, num_classes: int = 4, dropout: float = 0.3):
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        # Take last time step
        last_hidden = lstm_out[:, -1, :]
        out = self.fc(last_hidden)
        return out


class LSTMEncoder:
    """
    Wrapper for LSTM model with training and prediction logic.
    """
    
    def __init__(self, sequence_length: int = 60):
        self.sequence_length = sequence_length
        self.model: Optional[LSTMModel] = None
        self.scaler = StandardScaler()
        self._is_trained = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def _create_sequences(self, prices: np.ndarray, volumes: np.ndarray,
                         labels: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Create sliding window sequences."""
        # Normalize prices and volumes
        price_vol = np.column_stack([prices, volumes])
        price_vol_scaled = self.scaler.fit_transform(price_vol)
        
        # Calculate returns
        returns = np.diff(prices) / prices[:-1]
        returns = np.pad(returns, (1, 0), mode='constant', constant_values=0)
        returns_scaled = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Stack features
        features = np.column_stack([
            price_vol_scaled[:, 0],  # normalized price
            price_vol_scaled[:, 1],  # normalized volume
            returns_scaled           # normalized returns
        ])
        
        # Create sequences
        X, y = [], []
        for i in range(len(features) - self.sequence_length):
            X.append(features[i:i + self.sequence_length])
            if labels is not None:
                y.append(labels[i + self.sequence_length])
        
        X = np.array(X)
        y = np.array(y) if len(y) > 0 else None
        
        return X, y
    
    def train(self, prices: np.ndarray, volumes: np.ndarray, 
              labels: np.ndarray, epochs: int = 50, batch_size: int = 32,
              validation_split: float = 0.2) -> Dict:
        """Train LSTM model on price/volume sequences."""
        X, y = self._create_sequences(prices, volumes, labels)
        
        if len(X) == 0:
            raise ValueError("Not enough data to create sequences")
        
        # Train/validation split
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Create datasets and dataloaders
        train_dataset = SequenceDataset(X_train, y_train)
        val_dataset = SequenceDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        self.model = LSTMModel(
            input_size=3,
            hidden_size=64,
            num_layers=2,
            num_classes=len(STRATEGIES),
            dropout=0.3
        ).to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()
            
            val_loss /= len(val_loader)
            val_acc = correct / total
            
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= 10:
                break
        
        self._is_trained = True
        
        return {
            "final_train_loss": train_loss,
            "final_val_loss": val_loss,
            "final_val_accuracy": val_acc,
            "epochs_trained": epoch + 1
        }
    
    def predict_proba(self, prices: np.ndarray, volumes: np.ndarray) -> np.ndarray:
        """Get probability predictions from sequences."""
        if not self._is_trained or self.model is None:
            return np.ones((1, len(STRATEGIES))) / len(STRATEGIES)
        
        X, _ = self._create_sequences(prices, volumes)
        
        if len(X) == 0:
            return np.ones((1, len(STRATEGIES))) / len(STRATEGIES)
        
        # Use most recent sequence
        X_recent = X[-1:].copy()
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_recent).to(self.device)
            outputs = self.model(X_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
        
        return probs
    
    def save(self, path: str):
        """Save model to disk."""
        model_path = Path(path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.model is not None:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'scaler': self.scaler,
                'is_trained': self._is_trained,
                'sequence_length': self.sequence_length
            }, model_path)
    
    def load(self, path: str) -> bool:
        """Load model from disk."""
        model_path = Path(path)
        if not model_path.exists():
            return False
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.model = LSTMModel(
            input_size=3,
            hidden_size=64,
            num_layers=2,
            num_classes=len(STRATEGIES),
            dropout=0.3
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.scaler = checkpoint['scaler']
        self._is_trained = checkpoint['is_trained']
        self.sequence_length = checkpoint['sequence_length']
        
        return True


class FinBERTModel:
    """
    FinBERT model for news/sentiment text analysis.
    
    Uses pre-trained FinBERT to extract sentiment embeddings,
    then classifies trading strategy.
    """
    
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('ProsusAI/finbert')
        self.bert = BertModel.from_pretrained('ProsusAI/finbert')
        self.classifier = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, len(STRATEGIES))
        )
        self._is_trained = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.bert.to(self.device)
        self.classifier.to(self.device)
    
    def _encode_texts(self, texts: List[str], max_length: int = 128) -> torch.Tensor:
        """Encode texts using FinBERT."""
        encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        
        encodings = {k: v.to(self.device) for k, v in encodings.items()}
        
        with torch.no_grad():
            outputs = self.bert(**encodings)
            # Use CLS token embedding
            embeddings = outputs.last_hidden_state[:, 0, :]
        
        return embeddings
    
    def train(self, texts: List[str], labels: np.ndarray,
              epochs: int = 20, batch_size: int = 16,
              validation_split: float = 0.2) -> Dict:
        """Train FinBERT classifier."""
        # Encode all texts
        embeddings = self._encode_texts(texts).cpu().numpy()
        
        # Train/validation split
        split_idx = int(len(embeddings) * (1 - validation_split))
        X_train, X_val = embeddings[:split_idx], embeddings[split_idx:]
        y_train, y_val = labels[:split_idx], labels[split_idx:]
        
        # Create datasets
        train_dataset = SequenceDataset(X_train, y_train)
        val_dataset = SequenceDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.classifier.parameters(), lr=0.0001)
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training
            self.classifier.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.classifier(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation
            self.classifier.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    
                    outputs = self.classifier(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()
            
            val_loss /= len(val_loader)
            val_acc = correct / total
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
        
        self._is_trained = True
        
        return {
            "final_train_loss": train_loss,
            "final_val_loss": val_loss,
            "final_val_accuracy": val_acc,
            "epochs_trained": epoch + 1
        }
    
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Get probability predictions from text."""
        if not self._is_trained:
            return np.ones((len(texts), len(STRATEGIES))) / len(STRATEGIES)
        
        embeddings = self._encode_texts(texts)
        
        self.classifier.eval()
        with torch.no_grad():
            outputs = self.classifier(embeddings)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
        
        return probs
    
    def save(self, path: str):
        """Save model to disk."""
        model_path = Path(path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'classifier_state_dict': self.classifier.state_dict(),
            'is_trained': self._is_trained
        }, model_path)
    
    def load(self, path: str) -> bool:
        """Load model from disk."""
        model_path = Path(path)
        if not model_path.exists():
            return False
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.classifier.load_state_dict(checkpoint['classifier_state_dict'])
        self._is_trained = checkpoint['is_trained']
        
        return True


class EnsembleTradingModel:
    """
    Ensemble of XGBoost, LSTM, and FinBERT models.
    
    Weighted voting: XGB (40%), LSTM (35%), FinBERT (25%)
    """
    
    def __init__(self):
        self.xgb_model = XGBoostModel()
        self.lstm_model = LSTMEncoder(sequence_length=60)
        self.finbert_model = FinBERTModel()
        
        # Ensemble weights
        self.weights = {
            'xgb': 0.40,
            'lstm': 0.35,
            'finbert': 0.25
        }
        
        self._is_trained = False
    
    def train_all(self, 
                  tabular_features: np.ndarray,
                  tabular_labels: np.ndarray,
                  prices: np.ndarray,
                  volumes: np.ndarray,
                  sequential_labels: np.ndarray,
                  news_texts: List[str],
                  text_labels: np.ndarray,
                  **kwargs) -> Dict:
        """Train all three models."""
        results = {}
        
        # Train XGBoost
        print("Training XGBoost model...")
        results['xgboost'] = self.xgb_model.train(tabular_features, tabular_labels)
        
        # Train LSTM
        print("Training LSTM model...")
        results['lstm'] = self.lstm_model.train(prices, volumes, sequential_labels, **kwargs)
        
        # Train FinBERT
        print("Training FinBERT model...")
        results['finbert'] = self.finbert_model.train(news_texts, text_labels, **kwargs)
        
        self._is_trained = True
        
        return results
    
    def predict(self, 
                tabular_features: np.ndarray,
                prices: np.ndarray,
                volumes: np.ndarray,
                news_text: str) -> EnsemblePrediction:
        """Make ensemble prediction."""
        # Get predictions from each model
        xgb_probs = self.xgb_model.predict_proba(tabular_features.reshape(1, -1))[0]
        lstm_probs = self.lstm_model.predict_proba(prices, volumes)[0]
        finbert_probs = self.finbert_model.predict_proba([news_text])[0]
        
        # Weighted average
        combined_probs = (
            self.weights['xgb'] * xgb_probs +
            self.weights['lstm'] * lstm_probs +
            self.weights['finbert'] * finbert_probs
        )
        
        # Get predicted strategy
        pred_idx = np.argmax(combined_probs)
        pred_strategy = STRATEGIES[pred_idx]
        confidence = float(combined_probs[pred_idx])
        
        prob_dict = {strategy: float(prob) for strategy, prob in zip(STRATEGIES, combined_probs)}
        
        return EnsemblePrediction(
            strategy=pred_strategy,
            confidence=confidence,
            probabilities=prob_dict,
            xgb_confidence=float(np.max(xgb_probs)),
            lstm_confidence=float(np.max(lstm_probs)),
            finbert_confidence=float(np.max(finbert_probs)),
            timestamp=datetime.now()
        )
    
    def save(self, base_path: str):
        """Save all models to disk."""
        base = Path(base_path)
        base.parent.mkdir(parents=True, exist_ok=True)
        
        self.xgb_model.save(str(base.with_suffix('.xgb.pkl')))
        self.lstm_model.save(str(base.with_suffix('.lstm.pt')))
        self.finbert_model.save(str(base.with_suffix('.finbert.pt')))
        
        # Save ensemble metadata
        metadata_path = base.with_suffix('.ensemble.json')
        import json
        with open(metadata_path, 'w') as f:
            json.dump({
                'weights': self.weights,
                'is_trained': self._is_trained,
                'timestamp': datetime.now().isoformat()
            }, f)
    
    def load(self, base_path: str) -> bool:
        """Load all models from disk."""
        base = Path(base_path)
        
        if not base.with_suffix('.xgb.pkl').exists():
            return False
        
        self.xgb_model.load(str(base.with_suffix('.xgb.pkl')))
        self.lstm_model.load(str(base.with_suffix('.lstm.pt')))
        self.finbert_model.load(str(base.with_suffix('.finbert.pt')))
        
        # Load ensemble metadata
        metadata_path = base.with_suffix('.ensemble.json')
        if metadata_path.exists():
            import json
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.weights = metadata.get('weights', self.weights)
                self._is_trained = metadata.get('is_trained', False)
        else:
            self._is_trained = (
                self.xgb_model._is_trained and
                self.lstm_model._is_trained and
                self.finbert_model._is_trained
            )
        
        return True
    
    @property
    def is_trained(self) -> bool:
        return self._is_trained


# Singleton
_ensemble_model: Optional[EnsembleTradingModel] = None


def get_ensemble_model() -> EnsembleTradingModel:
    """Get or create the ensemble model singleton."""
    global _ensemble_model
    if _ensemble_model is None:
        _ensemble_model = EnsembleTradingModel()
    return _ensemble_model
