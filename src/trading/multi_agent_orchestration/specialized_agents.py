#!/usr/bin/env python
# Cosmic Market Oracle - Multi-Agent Orchestration Network Specialized Agents

"""
Specialized agents for the Multi-Agent Orchestration Network.

This module implements various specialized agents that perform specific roles
within the multi-agent system, including market regime detection, signal
generation, risk management, and macro environment analysis.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import json

from src.trading.multi_agent_orchestration.base import (
    AgentInterface, AgentMessage, MessageType
)
from src.astro_engine.planetary_positions import PlanetaryCalculator
from src.astro_engine.astrological_aspects import AspectCalculator
from src.feature_engineering.astrological_features import AstrologicalFeatureGenerator
from src.feature_engineering.pattern_detection import PatternDetector

# Configure logging
logger = logging.getLogger(__name__)


class MarketRegimeAgent(AgentInterface):
    """
    Agent for detecting and tracking market regimes.
    
    This agent analyzes market data to identify the current market regime
    (e.g., bull market, bear market, sideways, volatile) and notifies
    other agents when regime changes occur.
    """
    
    def __init__(self, agent_id: str, agent_type: str = "market_regime",
                regime_window: int = 60, min_regime_duration: int = 20,
                confidence_threshold: float = 0.7):
        """
        Initialize the market regime agent.
        
        Args:
            agent_id: Unique identifier for the agent
            agent_type: Type of agent (default: 'market_regime')
            regime_window: Window size for regime detection (days)
            min_regime_duration: Minimum duration for a regime (days)
            confidence_threshold: Confidence threshold for regime change detection
        """
        super().__init__(agent_id, agent_type)
        self.regime_window = regime_window
        self.min_regime_duration = min_regime_duration
        self.confidence_threshold = confidence_threshold
        
        # State variables
        self.current_regime = "unknown"
        self.regime_start_date = None
        self.regime_probabilities = {}
        self.regime_history = []
        
        # Register specific message handlers
        self.message_handlers[MessageType.QUERY] = self._handle_query
        
    def _handle_query(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle query messages."""
        query_type = message.content.get("query_type")
        
        if query_type == "current_regime":
            return self.create_message(
                message_type=MessageType.RESPONSE,
                content={
                    "query_type": query_type,
                    "current_regime": self.current_regime,
                    "regime_start_date": self.regime_start_date.isoformat() if self.regime_start_date else None,
                    "regime_duration": (datetime.now() - self.regime_start_date).days if self.regime_start_date else 0,
                    "regime_probabilities": self.regime_probabilities
                },
                recipient_id=message.sender_id,
                correlation_id=message.message_id
            )
        elif query_type == "regime_history":
            return self.create_message(
                message_type=MessageType.RESPONSE,
                content={
                    "query_type": query_type,
                    "regime_history": self.regime_history[-20:]  # Return last 20 regimes
                },
                recipient_id=message.sender_id,
                correlation_id=message.message_id
            )
            
        return None
        
    def process(self, data: Dict[str, pd.DataFrame]) -> List[AgentMessage]:
        """
        Process market data to detect regimes.
        
        Args:
            data: Dictionary of data frames
            
        Returns:
            List of messages generated by the agent
        """
        messages = []
        
        # Check if market data is available
        if "market_data" not in data:
            logger.warning(f"Agent {self.agent_id}: No market data available")
            return messages
            
        market_data = data["market_data"]
        
        # Ensure we have enough data
        if len(market_data) < self.regime_window:
            logger.warning(f"Agent {self.agent_id}: Insufficient data for regime detection")
            return messages
            
        # Extract features for regime detection
        features = self._extract_regime_features(market_data)
        
        # Detect regime
        regime, probabilities = self._detect_regime(features)
        
        # Check if regime has changed
        if regime != self.current_regime and probabilities[regime] >= self.confidence_threshold:
            # Record previous regime if it existed
            if self.current_regime != "unknown" and self.regime_start_date:
                self.regime_history.append({
                    "regime": self.current_regime,
                    "start_date": self.regime_start_date.isoformat(),
                    "end_date": datetime.now().isoformat(),
                    "duration_days": (datetime.now() - self.regime_start_date).days
                })
                
            # Update current regime
            self.current_regime = regime
            self.regime_start_date = datetime.now()
            
            # Notify other agents of regime change
            messages.append(self.create_message(
                message_type=MessageType.REGIME_CHANGE,
                content={
                    "previous_regime": self.current_regime,
                    "new_regime": regime,
                    "confidence": probabilities[regime],
                    "regime_features": features.to_dict() if isinstance(features, pd.Series) else features
                },
                priority=8  # High priority for regime changes
            ))
            
            logger.info(f"Agent {self.agent_id}: Detected regime change to {regime} "
                      f"with confidence {probabilities[regime]:.2f}")
                      
        # Update probabilities
        self.regime_probabilities = probabilities
        
        return messages
        
    def _extract_regime_features(self, market_data: pd.DataFrame) -> pd.Series:
        """
        Extract features for regime detection.
        
        Args:
            market_data: Market data frame
            
        Returns:
            Series of features
        """
        # Use recent data within the regime window
        recent_data = market_data.iloc[-self.regime_window:]
        
        # Calculate features
        returns = recent_data["close"].pct_change().dropna()
        
        features = pd.Series({
            "mean_return": returns.mean(),
            "volatility": returns.std(),
            "skewness": returns.skew(),
            "kurtosis": returns.kurt(),
            "positive_days_ratio": (returns > 0).mean(),
            "max_drawdown": self._calculate_max_drawdown(recent_data["close"]),
            "trend_strength": self._calculate_trend_strength(recent_data["close"]),
            "volume_trend": self._calculate_volume_trend(recent_data)
        })
        
        return features
        
    def _detect_regime(self, features: pd.Series) -> Tuple[str, Dict[str, float]]:
        """
        Detect the current market regime based on features.
        
        Args:
            features: Series of market features
            
        Returns:
            Tuple of (regime name, probabilities dictionary)
        """
        # Simple rule-based regime detection
        # In a real implementation, this would use a trained classifier
        
        # Initialize probabilities
        probabilities = {
            "bull": 0.0,
            "bear": 0.0,
            "sideways": 0.0,
            "volatile": 0.0
        }
        
        # Bull market indicators
        if features["mean_return"] > 0.001 and features["positive_days_ratio"] > 0.6:
            probabilities["bull"] += 0.5
        if features["trend_strength"] > 0.7:
            probabilities["bull"] += 0.3
        if features["max_drawdown"] > -0.05:
            probabilities["bull"] += 0.2
            
        # Bear market indicators
        if features["mean_return"] < -0.001 and features["positive_days_ratio"] < 0.4:
            probabilities["bear"] += 0.5
        if features["trend_strength"] < -0.7:
            probabilities["bear"] += 0.3
        if features["max_drawdown"] < -0.15:
            probabilities["bear"] += 0.2
            
        # Sideways market indicators
        if abs(features["mean_return"]) < 0.0005 and features["volatility"] < 0.01:
            probabilities["sideways"] += 0.6
        if abs(features["trend_strength"]) < 0.3:
            probabilities["sideways"] += 0.4
            
        # Volatile market indicators
        if features["volatility"] > 0.015:
            probabilities["volatile"] += 0.5
        if abs(features["skewness"]) > 1.0:
            probabilities["volatile"] += 0.3
        if features["kurtosis"] > 3.0:
            probabilities["volatile"] += 0.2
            
        # Normalize probabilities
        total = sum(probabilities.values())
        if total > 0:
            probabilities = {k: v / total for k, v in probabilities.items()}
            
        # Select regime with highest probability
        regime = max(probabilities, key=probabilities.get)
        
        return regime, probabilities
        
    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """
        Calculate maximum drawdown from price series.
        
        Args:
            prices: Series of prices
            
        Returns:
            Maximum drawdown as a negative percentage
        """
        peak = prices.iloc[0]
        max_drawdown = 0.0
        
        for price in prices:
            if price > peak:
                peak = price
            else:
                drawdown = (price - peak) / peak
                max_drawdown = min(max_drawdown, drawdown)
                
        return max_drawdown
        
    def _calculate_trend_strength(self, prices: pd.Series) -> float:
        """
        Calculate trend strength from price series.
        
        Args:
            prices: Series of prices
            
        Returns:
            Trend strength (-1 to 1)
        """
        # Use linear regression slope as trend indicator
        x = np.arange(len(prices))
        y = prices.values
        
        # Calculate slope
        n = len(x)
        xy_sum = (x * y).sum()
        x_sum = x.sum()
        y_sum = y.sum()
        x2_sum = (x * x).sum()
        
        slope = (n * xy_sum - x_sum * y_sum) / (n * x2_sum - x_sum * x_sum)
        
        # Normalize to -1 to 1 range
        normalized_slope = np.tanh(slope * 100)
        
        return normalized_slope
        
    def _calculate_volume_trend(self, market_data: pd.DataFrame) -> float:
        """
        Calculate volume trend from market data.
        
        Args:
            market_data: Market data frame
            
        Returns:
            Volume trend indicator (-1 to 1)
        """
        if "volume" not in market_data.columns:
            return 0.0
            
        # Get volume data
        volume = market_data["volume"].values
        
        # Calculate trend using simple moving average ratio
        short_ma = np.mean(volume[-10:])
        long_ma = np.mean(volume[-30:])
        
        if long_ma == 0:
            return 0.0
            
        ratio = short_ma / long_ma - 1
        
        # Normalize to -1 to 1 range
        normalized_ratio = np.tanh(ratio * 5)
        
        return normalized_ratio
