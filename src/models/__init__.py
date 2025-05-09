# Cosmic Market Oracle - Models Module

"""
The models module implements the multi-modal deep learning architectures
for market prediction based on astrological and financial data.

This module contains various model implementations including:
- Time series models (LSTM, TCN, WaveNet)
- Ensemble methods (XGBoost, LightGBM, CatBoost)
- Reinforcement learning components
- Transformer-based models with astrological attention mechanisms
"""

__all__ = [
    'time_series',
    'ensemble',
    'reinforcement',
    'transformers',
    'model_factory',
    'evaluation'
]