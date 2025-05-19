# Cosmic Market Oracle - Trading Strategy Framework

"""
This module provides a framework for developing trading strategies based on
Vedic astrological signals and market data.

It includes:
- Base Strategy class for implementing different trading approaches
- Signal generation based on astrological configurations
- Position sizing based on astrological strength
- Risk management rules incorporating astrological factors
- Performance evaluation metrics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
from abc import ABC, abstractmethod

from src.astro_engine.planetary_positions import (
    PlanetaryCalculator, analyze_market_trend, analyze_financial_yogas
)
from src.utils.logger import setup_logger

# Configure logging
logger = setup_logger("strategy_framework")


class BaseStrategy(ABC):
    """Base class for all trading strategies."""
    
    def __init__(self, name: str, description: str = ""):
        """
        Initialize the strategy.
        
        Args:
            name: Strategy name
            description: Strategy description
        """
        self.name = name
        self.description = description
        self.positions = {}  # Current positions {symbol: quantity}
        self.trades = []     # Trade history
        self.calculator = PlanetaryCalculator()
        
        # Performance metrics
        self.metrics = {
            "total_return": 0.0,
            "annualized_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0
        }
    
    @abstractmethod
    def generate_signals(self, market_data: pd.DataFrame, planetary_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on market and planetary data.
        
        Args:
            market_data: DataFrame containing market data
            planetary_data: DataFrame containing planetary data
            
        Returns:
            DataFrame with trading signals
        """
        pass
    
    def calculate_position_size(self, signal_strength: float, account_size: float, 
                               risk_per_trade: float = 0.02) -> float:
        """
        Calculate position size based on signal strength and risk parameters.
        
        Args:
            signal_strength: Strength of the trading signal (0-1)
            account_size: Current account size
            risk_per_trade: Maximum risk per trade as a fraction of account
            
        Returns:
            Position size in currency units
        """
        # Base risk is a percentage of account size
        base_risk = account_size * risk_per_trade
        
        # Adjust risk based on signal strength
        adjusted_risk = base_risk * signal_strength
        
        return adjusted_risk
    
    def apply_risk_management(self, entry_price: float, signal: Dict, 
                             market_data: pd.DataFrame, planetary_data: pd.DataFrame) -> Dict:
        """
        Apply risk management rules to determine stop loss and take profit levels.
        
        Args:
            entry_price: Entry price for the trade
            signal: Signal dictionary with direction and strength
            market_data: Market data for reference
            planetary_data: Planetary data for reference
            
        Returns:
            Dictionary with stop loss and take profit levels
        """
        # Default risk management (2% stop loss, 6% take profit)
        stop_loss = entry_price * 0.98 if signal["direction"] == "buy" else entry_price * 1.02
        take_profit = entry_price * 1.06 if signal["direction"] == "buy" else entry_price * 0.94
        
        # Adjust based on market volatility
        if "volatility" in market_data.columns:
            volatility = market_data["volatility"].iloc[-1]
            stop_loss = entry_price * (0.98 - volatility * 0.5) if signal["direction"] == "buy" else entry_price * (1.02 + volatility * 0.5)
            take_profit = entry_price * (1.06 + volatility) if signal["direction"] == "buy" else entry_price * (0.94 - volatility)
        
        # Adjust based on astrological reversal probability
        if "reversal_probability" in planetary_data.columns:
            reversal_prob = planetary_data["reversal_probability"].iloc[-1] / 100.0  # Normalize to 0-1
            stop_loss = entry_price * (0.98 - reversal_prob * 0.01) if signal["direction"] == "buy" else entry_price * (1.02 + reversal_prob * 0.01)
        
        return {
            "stop_loss": stop_loss,
            "take_profit": take_profit
        }
    
    def execute_trade(self, symbol: str, direction: str, quantity: float, 
                     price: float, timestamp: datetime, signal_strength: float) -> Dict:
        """
        Execute a trade and record it in the trade history.
        
        Args:
            symbol: Trading symbol
            direction: Trade direction ("buy" or "sell")
            quantity: Trade quantity
            price: Execution price
            timestamp: Trade timestamp
            signal_strength: Strength of the signal that generated this trade
            
        Returns:
            Trade record dictionary
        """
        # Create trade record
        trade = {
            "symbol": symbol,
            "direction": direction,
            "quantity": quantity,
            "price": price,
            "timestamp": timestamp,
            "signal_strength": signal_strength
        }
        
        # Update positions
        if direction == "buy":
            self.positions[symbol] = self.positions.get(symbol, 0) + quantity
        else:  # sell
            self.positions[symbol] = self.positions.get(symbol, 0) - quantity
        
        # Record trade
        self.trades.append(trade)
        
        logger.info(f"Executed {direction} trade: {quantity} {symbol} @ {price}")
        
        return trade
    
    def calculate_performance(self, market_data: pd.DataFrame) -> Dict:
        """
        Calculate performance metrics for the strategy.
        
        Args:
            market_data: Market data for the backtest period
            
        Returns:
            Dictionary of performance metrics
        """
        if not self.trades:
            logger.warning("No trades to calculate performance metrics")
            return self.metrics
        
        # Convert trades to DataFrame
        trades_df = pd.DataFrame(self.trades)
        trades_df.set_index("timestamp", inplace=True)
        
        # Calculate returns
        trades_df["return"] = 0.0
        
        # Calculate P&L for each trade
        buy_trades = trades_df[trades_df["direction"] == "buy"]
        sell_trades = trades_df[trades_df["direction"] == "sell"]
        
        # Simple P&L calculation (this would be more complex in a real implementation)
        total_buy_value = (buy_trades["price"] * buy_trades["quantity"]).sum()
        total_sell_value = (sell_trades["price"] * sell_trades["quantity"]).sum()
        
        total_return = (total_sell_value - total_buy_value) / total_buy_value if total_buy_value > 0 else 0
        
        # Calculate other metrics
        self.metrics["total_return"] = total_return
        
        # Annualized return
        days = (trades_df.index.max() - trades_df.index.min()).days
        if days > 0:
            self.metrics["annualized_return"] = (1 + total_return) ** (365 / days) - 1
        
        # Win rate (simplified)
        winning_trades = len([t for t in self.trades if 
                             (t["direction"] == "buy" and t["price"] < market_data["Close"].iloc[-1]) or
                             (t["direction"] == "sell" and t["price"] > market_data["Close"].iloc[-1])])
        self.metrics["win_rate"] = winning_trades / len(self.trades) if self.trades else 0
        
        logger.info(f"Strategy performance: {self.metrics}")
        
        return self.metrics
    
    def reset(self):
        """Reset the strategy state."""
        self.positions = {}
        self.trades = []
        self.metrics = {
            "total_return": 0.0,
            "annualized_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0
        }


class VedicAstrologyStrategy(BaseStrategy):
    """
    [PARTIALLY DEPRECATED] This implementation is partially deprecated.
    
    For new code, please use the specialized signal generators from signal_generator.py
    combined with the BaseStrategy class for more modular and maintainable code.
    
    Trading strategy based on Vedic astrological signals.
    """
    
    def __init__(self, name: str = "Vedic Astrology Strategy", 
                description: str = "Trading strategy based on Vedic astrological signals"):
        """
        Initialize the Vedic Astrology strategy.
        
        Args:
            name: Strategy name
            description: Strategy description
        """
        super().__init__(name, description)
        
        # Strategy parameters
        self.min_signal_strength = 0.6  # Minimum signal strength to generate a trade
        self.use_yogas = True           # Whether to use financial yogas
        self.use_nakshatras = True      # Whether to use nakshatra analysis
        self.use_dashas = True          # Whether to use dasha periods
    
    def generate_signals(self, market_data: pd.DataFrame, planetary_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on Vedic astrological factors.
        
        Args:
            market_data: DataFrame containing market data
            planetary_data: DataFrame containing planetary data
            
        Returns:
            DataFrame with trading signals
        """
        if market_data.empty or planetary_data.empty:
            logger.warning("Empty data provided for signal generation")
            return pd.DataFrame()
        
        # Initialize signals DataFrame
        signals = pd.DataFrame(index=market_data.index)
        signals["signal"] = 0  # 0: no signal, 1: buy, -1: sell
        signals["strength"] = 0.0
        signals["reason"] = ""
        
        # Process each day
        for date in signals.index:
            # Skip if no planetary data for this date
            if date not in planetary_data.index:
                continue
            
            # Get planetary data for this date
            planets_for_date = planetary_data.loc[date]
            
            # Initialize signal components
            signal_direction = 0
            signal_strength = 0.0
            signal_reasons = []
            
            # 1. Check market trend from analyze_market_trend
            if "market_trend_primary_trend" in planets_for_date:
                trend = planets_for_date["market_trend_primary_trend"]
                trend_strength = planets_for_date["market_trend_strength"] / 100.0  # Normalize to 0-1
                
                if trend == "bullish" and trend_strength > self.min_signal_strength:
                    signal_direction = 1
                    signal_strength = trend_strength
                    signal_reasons.append(f"Bullish market trend ({trend_strength:.2f})")
                elif trend == "bearish" and trend_strength > self.min_signal_strength:
                    signal_direction = -1
                    signal_strength = trend_strength
                    signal_reasons.append(f"Bearish market trend ({trend_strength:.2f})")
            
            # 2. Check for financial yogas
            if self.use_yogas:
                bullish_yogas = planets_for_date.get("bullish_yoga_count", 0)
                bearish_yogas = planets_for_date.get("bearish_yoga_count", 0)
                
                if bullish_yogas > bearish_yogas and bullish_yogas > 0:
                    yoga_strength = min(bullish_yogas * 0.2, 0.8)  # Cap at 0.8
                    if yoga_strength > signal_strength:
                        signal_direction = 1
                        signal_strength = yoga_strength
                        signal_reasons = [f"Bullish yogas ({bullish_yogas})"]
                    elif signal_direction == 1:
                        signal_strength = (signal_strength + yoga_strength) / 2
                        signal_reasons.append(f"Bullish yogas ({bullish_yogas})")
                
                elif bearish_yogas > bullish_yogas and bearish_yogas > 0:
                    yoga_strength = min(bearish_yogas * 0.2, 0.8)  # Cap at 0.8
                    if yoga_strength > signal_strength:
                        signal_direction = -1
                        signal_strength = yoga_strength
                        signal_reasons = [f"Bearish yogas ({bearish_yogas})"]
                    elif signal_direction == -1:
                        signal_strength = (signal_strength + yoga_strength) / 2
                        signal_reasons.append(f"Bearish yogas ({bearish_yogas})")
            
            # 3. Check Moon nakshatra
            if self.use_nakshatras and "moon_nakshatra_financial" in planets_for_date:
                nakshatra_financial = planets_for_date["moon_nakshatra_financial"]
                
                if nakshatra_financial == "bullish":
                    nakshatra_strength = 0.7
                    if signal_direction == 0 or nakshatra_strength > signal_strength:
                        signal_direction = 1
                        signal_strength = nakshatra_strength
                        signal_reasons = [f"Bullish Moon nakshatra"]
                    elif signal_direction == 1:
                        signal_strength = (signal_strength + nakshatra_strength) / 2
                        signal_reasons.append(f"Bullish Moon nakshatra")
                
                elif nakshatra_financial == "bearish":
                    nakshatra_strength = 0.7
                    if signal_direction == 0 or nakshatra_strength > signal_strength:
                        signal_direction = -1
                        signal_strength = nakshatra_strength
                        signal_reasons = [f"Bearish Moon nakshatra"]
                    elif signal_direction == -1:
                        signal_strength = (signal_strength + nakshatra_strength) / 2
                        signal_reasons.append(f"Bearish Moon nakshatra")
            
            # 4. Check current dasha lord
            if self.use_dashas and "current_dasha_lord" in planets_for_date:
                dasha_lord = planets_for_date["current_dasha_lord"]
                
                # Financial nature of dasha lords
                dasha_financial_map = {
                    "Sun": 0.6,      # Moderately bullish
                    "Moon": 0.5,     # Neutral
                    "Mars": 0.3,     # Bearish
                    "Rahu": 0.25,    # Highly volatile/bearish
                    "Jupiter": 0.9,  # Strongly bullish
                    "Saturn": 0.2,   # Strongly bearish
                    "Mercury": 0.7,  # Moderately bullish
                    "Ketu": 0.3,     # Bearish and volatile
                    "Venus": 0.8     # Bullish
                }
                
                dasha_strength = dasha_financial_map.get(dasha_lord, 0.5)
                
                if dasha_strength > 0.6:  # Bullish
                    if signal_direction == 0 or dasha_strength > signal_strength:
                        signal_direction = 1
                        signal_strength = dasha_strength
                        signal_reasons = [f"Bullish dasha lord ({dasha_lord})"]
                    elif signal_direction == 1:
                        signal_strength = (signal_strength + dasha_strength) / 2
                        signal_reasons.append(f"Bullish dasha lord ({dasha_lord})")
                
                elif dasha_strength < 0.4:  # Bearish
                    dasha_strength = 1 - dasha_strength  # Invert for bearish strength
                    if signal_direction == 0 or dasha_strength > signal_strength:
                        signal_direction = -1
                        signal_strength = dasha_strength
                        signal_reasons = [f"Bearish dasha lord ({dasha_lord})"]
                    elif signal_direction == -1:
                        signal_strength = (signal_strength + dasha_strength) / 2
                        signal_reasons.append(f"Bearish dasha lord ({dasha_lord})")
            
            # 5. Check reversal probability
            if "market_trend_reversal_probability" in planets_for_date:
                reversal_prob = planets_for_date["market_trend_reversal_probability"] / 100.0
                
                if reversal_prob > 0.7:
                    # High reversal probability might flip the signal
                    if signal_direction == 1:
                        signal_direction = -1
                        signal_strength = reversal_prob
                        signal_reasons = [f"High reversal probability ({reversal_prob:.2f})"]
                    elif signal_direction == -1:
                        signal_direction = 1
                        signal_strength = reversal_prob
                        signal_reasons = [f"High reversal probability ({reversal_prob:.2f})"]
            
            # Record signal if strong enough
            if abs(signal_direction) > 0 and signal_strength >= self.min_signal_strength:
                signals.loc[date, "signal"] = signal_direction
                signals.loc[date, "strength"] = signal_strength
                signals.loc[date, "reason"] = ", ".join(signal_reasons)
        
        return signals
    
    def backtest(self, market_data: pd.DataFrame, planetary_data: pd.DataFrame, 
                initial_capital: float = 100000.0) -> pd.DataFrame:
        """
        [DEPRECATED] This method is deprecated. Please use the BacktestEngine class from 
        src.trading.backtest instead for more comprehensive backtesting capabilities.
        
        Backtest the strategy on historical data.
        
        Args:
            market_data: DataFrame containing market data
            planetary_data: DataFrame containing planetary data
            initial_capital: Initial capital for the backtest
            
        Returns:
            DataFrame with backtest results
        """
        # Reset strategy state
        self.reset()
        
        # Generate signals
        signals = self.generate_signals(market_data, planetary_data)
        
        # Initialize backtest results
        results = pd.DataFrame(index=market_data.index)
        results["close"] = market_data["Close"]
        results["signal"] = signals["signal"]
        results["strength"] = signals["strength"]
        results["position"] = 0.0
        results["cash"] = initial_capital
        results["equity"] = initial_capital
        results["returns"] = 0.0
        
        # Track current position
        position = 0.0
        
        # Process each day
        for i, date in enumerate(results.index):
            if i == 0:
                # First day, just record initial state
                results.loc[date, "position"] = position
                results.loc[date, "cash"] = initial_capital
                results.loc[date, "equity"] = initial_capital
                continue
            
            # Get previous day's state
            prev_date = results.index[i-1]
            prev_position = results.loc[prev_date, "position"]
            prev_cash = results.loc[prev_date, "cash"]
            
            # Get current price and signal
            price = results.loc[date, "close"]
            signal = results.loc[date, "signal"]
            strength = results.loc[date, "strength"]
            
            # Determine trade action
            if signal == 1 and prev_position <= 0:  # Buy signal
                # Calculate position size based on signal strength
                position_value = self.calculate_position_size(strength, prev_cash)
                position = position_value / price
                
                # Update cash
                cash = prev_cash - position_value
                
                # Record trade
                self.execute_trade(
                    symbol=market_data.columns[0] if isinstance(market_data.columns[0], str) else "MARKET",
                    direction="buy",
                    quantity=position,
                    price=price,
                    timestamp=date,
                    signal_strength=strength
                )
                
            elif signal == -1 and prev_position >= 0:  # Sell signal
                # Calculate position size based on signal strength
                if prev_position > 0:
                    # Sell existing position
                    cash = prev_cash + (prev_position * price)
                    
                    # Record trade
                    self.execute_trade(
                        symbol=market_data.columns[0] if isinstance(market_data.columns[0], str) else "MARKET",
                        direction="sell",
                        quantity=prev_position,
                        price=price,
                        timestamp=date,
                        signal_strength=strength
                    )
                    
                    position = 0
                    
                # Short selling (if allowed)
                position_value = self.calculate_position_size(strength, cash)
                short_position = position_value / price
                position = -short_position
                cash = cash + position_value
                
                # Record trade
                self.execute_trade(
                    symbol=market_data.columns[0] if isinstance(market_data.columns[0], str) else "MARKET",
                    direction="sell",
                    quantity=short_position,
                    price=price,
                    timestamp=date,
                    signal_strength=strength
                )
                
            else:
                # No new signal, maintain position
                position = prev_position
                cash = prev_cash
            
            # Calculate equity
            equity = cash + (position * price)
            
            # Calculate returns
            prev_equity = results.loc[prev_date, "equity"]
            returns = (equity - prev_equity) / prev_equity if prev_equity > 0 else 0
            
            # Record results
            results.loc[date, "position"] = position
            results.loc[date, "cash"] = cash
            results.loc[date, "equity"] = equity
            results.loc[date, "returns"] = returns
        
        # Calculate cumulative returns
        results["cumulative_returns"] = (1 + results["returns"]).cumprod() - 1
        
        # Calculate drawdowns
        results["peak"] = results["equity"].cummax()
        results["drawdown"] = (results["equity"] - results["peak"]) / results["peak"]
        
        # Calculate performance metrics
        self.calculate_performance(market_data)
        
        return results
