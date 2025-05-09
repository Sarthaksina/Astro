# Cosmic Market Oracle - Feature Engineering Module

"""
The feature_engineering module handles the conversion of raw market and astrological data
into machine learning-compatible features. This includes specialized transformations
for cyclical data, planetary relationships, and technical market indicators.

This module is critical for translating complex astrological phenomena into
numerical representations that can be used by machine learning models.
"""

__all__ = [
    'astrological_features',
    'market_features',
    'feature_selection',
    'feature_importance',
]