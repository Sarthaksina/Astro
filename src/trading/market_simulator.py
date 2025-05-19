"""
Market Simulator for the Cosmic Market Oracle.

This module provides market simulation capabilities for backtesting
trading strategies, including position management, trade execution,
and equity curve calculation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime
import logging

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class MarketSimulator:
    """
    Simulator for market trading based on strategy signals.
    
    This class simulates market trading based on strategy signals,
    handling position management, trade execution, and equity curve
    calculation for backtesting purposes.
    """
    
    def __init__(self, initial_capital: float = 100000.0, commission: float = 0.001):
        """
        Initialize the market simulator.
        
        Args:
            initial_capital: Initial capital for simulation
            commission: Commission rate per trade (as a fraction)
        """
        self.initial_capital = initial_capital
        self.commission = commission
        logger.info(f"Initialized market simulator with capital: {initial_capital}, commission: {commission}")
    
    def simulate(self, signals_df: pd.DataFrame) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
        """
        Simulate trading based on strategy signals.
        
        Args:
            signals_df: DataFrame with trading signals
            
        Returns:
            Tuple of (trades list, equity curve DataFrame)
        """
        logger.info("Starting market simulation")
        
        # Ensure required columns exist
        required_columns = ['date', 'close', 'signal']
        if not all(col in signals_df.columns for col in required_columns):
            raise ValueError(f"Signals DataFrame must contain columns: {required_columns}")
        
        # Initialize simulation variables
        capital = self.initial_capital
        position = 0
        trades = []
        equity_history = []
        
        # Prepare results DataFrame
        equity_curve = pd.DataFrame()
        equity_curve['date'] = signals_df['date']
        equity_curve['close'] = signals_df['close']
        equity_curve['signal'] = signals_df['signal']
        
        # Add columns for tracking
        equity_curve['position'] = 0
        equity_curve['equity'] = self.initial_capital
        equity_curve['cash'] = self.initial_capital
        equity_curve['holdings'] = 0
        equity_curve['trade_count'] = 0
        equity_curve['win_count'] = 0
        equity_curve['loss_count'] = 0
        equity_curve['profit'] = 0
        equity_curve['loss'] = 0
        
        # Track current trade
        current_trade = None
        trade_count = 0
        win_count = 0
        loss_count = 0
        total_profit = 0
        total_loss = 0
        
        # Simulate trading
        for i in range(len(signals_df)):
            date = signals_df['date'].iloc[i]
            price = signals_df['close'].iloc[i]
            signal = signals_df['signal'].iloc[i]
            
            # Process signal
            new_position = self._process_signal(signal)
            
            # Handle position changes
            if new_position != position:
                # Close existing position if any
                if position != 0:
                    # Calculate trade result
                    exit_value = position * price
                    exit_commission = abs(exit_value) * self.commission
                    trade_result = exit_value - current_trade['entry_value'] - current_trade['entry_commission'] - exit_commission
                    
                    # Update capital
                    capital += exit_value - exit_commission
                    
                    # Complete the trade record
                    current_trade.update({
                        'exit_date': date,
                        'exit_price': price,
                        'exit_value': exit_value,
                        'exit_commission': exit_commission,
                        'pnl': trade_result,
                        'return': trade_result / abs(current_trade['entry_value'])
                    })
                    
                    # Add to trades list
                    trades.append(current_trade)
                    
                    # Update trade statistics
                    trade_count += 1
                    if trade_result > 0:
                        win_count += 1
                        total_profit += trade_result
                    else:
                        loss_count += 1
                        total_loss += trade_result
                    
                    # Reset position
                    position = 0
                    current_trade = None
                
                # Open new position if signal indicates
                if new_position != 0:
                    # Calculate entry details
                    entry_value = new_position * price
                    entry_commission = abs(entry_value) * self.commission
                    
                    # Update capital
                    capital -= entry_commission
                    
                    # Create new trade record
                    current_trade = {
                        'entry_date': date,
                        'entry_price': price,
                        'position': new_position,
                        'entry_value': entry_value,
                        'entry_commission': entry_commission
                    }
                    
                    # Update position
                    position = new_position
            
            # Calculate equity
            holdings = position * price
            equity = capital + holdings
            
            # Update equity history
            equity_history.append({
                'date': date,
                'close': price,
                'position': position,
                'cash': capital,
                'holdings': holdings,
                'equity': equity,
                'trade_count': trade_count,
                'win_count': win_count,
                'loss_count': loss_count,
                'profit': total_profit,
                'loss': total_loss
            })
        
        # Convert equity history to DataFrame
        equity_curve = pd.DataFrame(equity_history)
        
        logger.info(f"Simulation completed with {len(trades)} trades")
        
        return trades, equity_curve
    
    def _process_signal(self, signal: float) -> int:
        """
        Process trading signal to determine position.
        
        Args:
            signal: Trading signal value
            
        Returns:
            Position size (positive for long, negative for short, 0 for no position)
        """
        # Simple signal processing logic
        # This can be extended for more sophisticated position sizing
        if signal > 0:
            return 1  # Long position
        elif signal < 0:
            return -1  # Short position
        else:
            return 0  # No position
    
    def calculate_position_size(self, capital: float, price: float, risk_per_trade: float = 0.02) -> int:
        """
        Calculate position size based on risk management rules.
        
        Args:
            capital: Available capital
            price: Current price
            risk_per_trade: Risk per trade as fraction of capital
            
        Returns:
            Position size in number of units
        """
        # Calculate position size based on fixed percentage risk
        risk_amount = capital * risk_per_trade
        position_value = risk_amount * 10  # Assume 10x leverage of risk amount
        
        # Calculate units based on price
        units = int(position_value / price)
        
        return units
