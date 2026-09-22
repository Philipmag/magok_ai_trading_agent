"""
ML Package - Phase 2 Ensemble Models.

Provides ensemble ML models (XGBoost + LSTM + FinBERT) with SHAP explainability.
"""

from .ensemble import (
    EnsembleTradingModel,
    EnsemblePrediction,
    XGBoostModel,
    LSTMEncoder,
    FinBERTModel,
    get_ensemble_model,
    STRATEGIES
)

from .explainer import (
    SHAPExplainer,
    FeatureAttribution,
    ExplanationResult,
    get_shap_explainer
)

from .tune import (
    HyperparameterTuner,
    run_full_tuning_pipeline,
    tune_model
)

# Keep backward compatibility with original model
from .model import (
    TradingModel,
    ModelPrediction,
    get_trading_model
)

from .predict import (
    PredictionEngine,
    TradingDecision,
    get_prediction_engine
)

__all__ = [
    # Ensemble
    "EnsembleTradingModel",
    "EnsemblePrediction",
    "XGBoostModel",
    "LSTMEncoder",
    "FinBERTModel",
    "get_ensemble_model",
    
    # Explainability
    "SHAPExplainer",
    "FeatureAttribution",
    "ExplanationResult",
    "get_shap_explainer",
    
    # Tuning
    "HyperparameterTuner",
    "run_full_tuning_pipeline",
    "tune_model",
    
    # Backward compatibility
    "TradingModel",
    "ModelPrediction",
    "get_trading_model",
    "PredictionEngine",
    "TradingDecision",
    "get_prediction_engine",
    "STRATEGIES"
]
