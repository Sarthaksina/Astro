"""
Centralized Enums for Feature Engineering Module.

This module contains all enums used across the feature engineering components,
centralizing them to avoid redundancy and ensure consistency.
"""

from enum import Enum, auto

class TimeFrame(Enum):
    """Time frames for astrological analysis."""
    INTRADAY = auto()
    DAILY = auto()
    WEEKLY = auto()
    MONTHLY = auto()
    QUARTERLY = auto()
    YEARLY = auto()

class FactorCategory(Enum):
    """Categories of astrological factors."""
    PLANETARY_POSITION = auto()
    PLANETARY_ASPECT = auto()
    ZODIACAL_DISTRIBUTION = auto()
    VEDIC_FACTOR = auto()
    TRANSIT = auto()
    DASHA = auto()

class FeatureType(Enum):
    """Types of features for machine learning."""
    ASTROLOGICAL = auto()
    MARKET = auto()
    COMPOSITE = auto()
    DERIVED = auto()
    INTERACTION = auto()

class ValidationLevel(Enum):
    """Levels of expert validation for features."""
    UNVALIDATED = auto()
    REVIEWED = auto()
    VALIDATED = auto()
    REJECTED = auto()
