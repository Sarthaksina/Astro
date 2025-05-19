"""
Feature Definitions Module for the Cosmic Market Oracle.

This module defines the core data structures for feature engineering,
centralizing feature definitions that were previously scattered across multiple files.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set, Optional, Any, Callable, Union
from enum import Enum
from datetime import datetime

from .enums import FactorCategory, TimeFrame, FeatureType


@dataclass
class FeatureDefinition:
    """Definition of a feature for machine learning models."""
    
    name: str
    description: str
    feature_type: FeatureType
    category: Optional[FactorCategory] = None
    time_frames: List[TimeFrame] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    transformation: Optional[str] = None
    source_features: List[str] = field(default_factory=list)
    creation_date: datetime = field(default_factory=datetime.now)
    last_modified: datetime = field(default_factory=datetime.now)
    version: str = "1.0.0"
    tags: List[str] = field(default_factory=list)
    complexity: int = 1
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the feature definition to a dictionary."""
        result = {
            "name": self.name,
            "description": self.description,
            "feature_type": self.feature_type.name if isinstance(self.feature_type, Enum) else self.feature_type,
            "parameters": self.parameters,
            "transformation": self.transformation,
            "source_features": self.source_features,
            "creation_date": self.creation_date.isoformat() if self.creation_date else None,
            "last_modified": self.last_modified.isoformat() if self.last_modified else None,
            "version": self.version,
            "tags": self.tags,
            "complexity": self.complexity,
            "performance_metrics": self.performance_metrics
        }
        
        if self.category:
            result["category"] = self.category.name if isinstance(self.category, Enum) else self.category
            
        if self.time_frames:
            result["time_frames"] = [tf.name if isinstance(tf, Enum) else tf for tf in self.time_frames]
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeatureDefinition':
        """Create a feature definition from a dictionary."""
        # Convert string enums back to enum values
        if "feature_type" in data and isinstance(data["feature_type"], str):
            data["feature_type"] = FeatureType[data["feature_type"]]
            
        if "category" in data and isinstance(data["category"], str):
            data["category"] = FactorCategory[data["category"]]
            
        if "time_frames" in data:
            data["time_frames"] = [TimeFrame[tf] if isinstance(tf, str) else tf for tf in data["time_frames"]]
            
        # Convert ISO date strings back to datetime objects
        if "creation_date" in data and isinstance(data["creation_date"], str):
            data["creation_date"] = datetime.fromisoformat(data["creation_date"])
            
        if "last_modified" in data and isinstance(data["last_modified"], str):
            data["last_modified"] = datetime.fromisoformat(data["last_modified"])
            
        return cls(**data)
