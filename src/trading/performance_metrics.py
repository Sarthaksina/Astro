"""
Performance Metrics for the Cosmic Market Oracle.

This module provides comprehensive performance analysis for trading strategies,
including risk-adjusted returns, drawdowns, and statistical measures.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
import logging
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.utils.logger import get_logger

logger = get_logger(__name__)


class PerformanceAnalyzer:
    """
    Analyzer for trading strategy performance metrics.
    
    This class calculates a comprehensive set of performance metrics for
    evaluating trading strategies, including:
    1. Return metrics (total return, annualized return, etc.)
    2. Risk metrics (volatility, drawdown, VaR, etc.)
    3. Risk-adjusted metrics (Sharpe ratio, Sortino ratio, etc.)
    4. Trade statistics (win rate, profit factor, etc.)
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize the performance analyzer.
        
        Args:
            risk_free_rate: Annual risk-free rate for risk-adjusted metrics (default: 2%)
        """
        self.risk_free_rate = risk_free_rate
        logger.info(f"Initialized performance analyzer with risk-free rate: {risk_free_rate}")
    
    def calculate_metrics(self, equity_curve: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate comprehensive performance metrics from equity curve.
        
        Args:
            equity_curve: DataFrame with equity curve data
            
        Returns:
            Dictionary with calculated performance metrics
        """
        logger.info("Calculating performance metrics")
        
        # Ensure equity curve has required columns
        required_columns = ['date', 'equity']
        if not all(col in equity_curve.columns for col in required_columns):
            raise ValueError(f"Equity curve must contain columns: {required_columns}")
        
        # Set date as index if it's not already
        if not isinstance(equity_curve.index, pd.DatetimeIndex):
            equity_curve = equity_curve.set_index('date')
        
        # Calculate daily returns
        daily_returns = equity_curve['equity'].pct_change().dropna()
        
        # Calculate metrics
        metrics = {}
        
        # Return metrics
        metrics.update(self._calculate_return_metrics(equity_curve, daily_returns))
        
        # Risk metrics
        metrics.update(self._calculate_risk_metrics(equity_curve, daily_returns))
        
        # Risk-adjusted metrics
        metrics.update(self._calculate_risk_adjusted_metrics(daily_returns, metrics))
        
        # Trade statistics (if available)
        if 'trade_count' in equity_curve.columns:
            metrics.update(self._calculate_trade_statistics(equity_curve))
        
        return metrics
    
    def _calculate_return_metrics(self, equity_curve: pd.DataFrame, daily_returns: pd.Series) -> Dict[str, float]:
        """
        Calculate return-based performance metrics.
        
        Args:
            equity_curve: DataFrame with equity curve data
            daily_returns: Series with daily returns
            
        Returns:
            Dictionary with return metrics
        """
        # Get start and end values
        start_equity = equity_curve['equity'].iloc[0]
        end_equity = equity_curve['equity'].iloc[-1]
        
        # Calculate time period in years
        start_date = equity_curve.index[0]
        end_date = equity_curve.index[-1]
        years = (end_date - start_date).days / 365.25
        
        # Calculate return metrics
        total_return = (end_equity / start_equity) - 1
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # Calculate compound annual growth rate (CAGR)
        cagr = (end_equity / start_equity) ** (1 / years) - 1 if years > 0 else 0
        
        # Calculate monthly returns
        monthly_returns = equity_curve['equity'].resample('M').last().pct_change().dropna()
        
        # Calculate yearly returns
        yearly_returns = equity_curve['equity'].resample('Y').last().pct_change().dropna()
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'cagr': cagr,
            'monthly_return_mean': monthly_returns.mean() if len(monthly_returns) > 0 else 0,
            'monthly_return_std': monthly_returns.std() if len(monthly_returns) > 0 else 0,
            'positive_months': (monthly_returns > 0).sum() / len(monthly_returns) if len(monthly_returns) > 0 else 0,
            'yearly_return_mean': yearly_returns.mean() if len(yearly_returns) > 0 else 0,
            'yearly_return_std': yearly_returns.std() if len(yearly_returns) > 0 else 0,
            'positive_years': (yearly_returns > 0).sum() / len(yearly_returns) if len(yearly_returns) > 0 else 0
        }
    
    def _calculate_risk_metrics(self, equity_curve: pd.DataFrame, daily_returns: pd.Series) -> Dict[str, float]:
        """
        Calculate risk-based performance metrics.
        
        Args:
            equity_curve: DataFrame with equity curve data
            daily_returns: Series with daily returns
            
        Returns:
            Dictionary with risk metrics
        """
        # Calculate volatility (annualized)
        volatility = daily_returns.std() * np.sqrt(252) if len(daily_returns) > 0 else 0
        
        # Calculate drawdowns
        drawdown_series = 1 - equity_curve['equity'] / equity_curve['equity'].cummax()
        max_drawdown = drawdown_series.max() if len(drawdown_series) > 0 else 0
        
        # Calculate drawdown duration
        is_drawdown = drawdown_series > 0
        drawdown_start = is_drawdown.ne(is_drawdown.shift()).cumsum()
        drawdown_duration = is_drawdown.groupby(drawdown_start).cumsum()
        max_drawdown_duration = drawdown_duration.max() if len(drawdown_duration) > 0 else 0
        
        # Calculate Value at Risk (VaR)
        var_95 = np.percentile(daily_returns, 5) if len(daily_returns) > 0 else 0
        var_99 = np.percentile(daily_returns, 1) if len(daily_returns) > 0 else 0
        
        # Calculate Conditional Value at Risk (CVaR)
        cvar_95 = daily_returns[daily_returns <= var_95].mean() if len(daily_returns[daily_returns <= var_95]) > 0 else 0
        cvar_99 = daily_returns[daily_returns <= var_99].mean() if len(daily_returns[daily_returns <= var_99]) > 0 else 0
        
        # Calculate downside deviation
        downside_returns = daily_returns[daily_returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        
        return {
            'volatility': volatility,
            'max_drawdown': max_drawdown,
            'max_drawdown_duration': max_drawdown_duration,
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'cvar_99': cvar_99,
            'downside_deviation': downside_deviation
        }
    
    def _calculate_risk_adjusted_metrics(self, daily_returns: pd.Series, metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate risk-adjusted performance metrics.
        
        Args:
            daily_returns: Series with daily returns
            metrics: Dictionary with previously calculated metrics
            
        Returns:
            Dictionary with risk-adjusted metrics
        """
        # Extract required metrics
        annualized_return = metrics['annualized_return']
        volatility = metrics['volatility']
        downside_deviation = metrics['downside_deviation']
        
        # Calculate daily risk-free rate
        daily_risk_free = (1 + self.risk_free_rate) ** (1 / 252) - 1
        
        # Calculate excess returns
        excess_returns = daily_returns - daily_risk_free
        
        # Calculate Sharpe ratio
        sharpe_ratio = (annualized_return - self.risk_free_rate) / volatility if volatility > 0 else 0
        
        # Calculate Sortino ratio
        sortino_ratio = (annualized_return - self.risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
        
        # Calculate Calmar ratio
        calmar_ratio = annualized_return / metrics['max_drawdown'] if metrics['max_drawdown'] > 0 else 0
        
        # Calculate Omega ratio
        threshold = daily_risk_free
        omega_ratio = sum(daily_returns[daily_returns > threshold] - threshold) / abs(sum(daily_returns[daily_returns < threshold] - threshold)) if sum(daily_returns[daily_returns < threshold] - threshold) != 0 else np.inf
        
        # Calculate Gain-to-Pain ratio
        gain_to_pain = sum(daily_returns[daily_returns > 0]) / abs(sum(daily_returns[daily_returns < 0])) if sum(daily_returns[daily_returns < 0]) != 0 else np.inf
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'omega_ratio': omega_ratio,
            'gain_to_pain_ratio': gain_to_pain,
            'information_ratio': excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0
        }
    
    def _calculate_trade_statistics(self, equity_curve: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate trade-based statistics.
        
        Args:
            equity_curve: DataFrame with equity curve and trade data
            
        Returns:
            Dictionary with trade statistics
        """
        # Check if required columns are available
        required_columns = ['trade_count', 'win_count', 'loss_count', 'profit', 'loss']
        if not all(col in equity_curve.columns for col in required_columns):
            logger.warning("Trade statistics columns not available in equity curve")
            return {}
        
        # Get final values
        total_trades = equity_curve['trade_count'].iloc[-1]
        win_count = equity_curve['win_count'].iloc[-1]
        loss_count = equity_curve['loss_count'].iloc[-1]
        total_profit = equity_curve['profit'].iloc[-1]
        total_loss = abs(equity_curve['loss'].iloc[-1])
        
        # Calculate trade statistics
        win_rate = win_count / total_trades if total_trades > 0 else 0
        profit_factor = total_profit / total_loss if total_loss > 0 else np.inf
        
        # Calculate average trade metrics
        avg_profit = total_profit / win_count if win_count > 0 else 0
        avg_loss = total_loss / loss_count if loss_count > 0 else 0
        
        # Calculate expectancy
        expectancy = (win_rate * avg_profit) - ((1 - win_rate) * avg_loss)
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'expectancy': expectancy,
            'avg_trade': (total_profit - total_loss) / total_trades if total_trades > 0 else 0
        }
    
    def generate_performance_report(self, 
                                   equity_curve: pd.DataFrame, 
                                   trades: List[Dict[str, Any]],
                                   output_dir: Optional[Union[str, Path]] = None,
                                   strategy_name: str = "Strategy") -> Dict[str, Any]:
        """
        Generate comprehensive performance report with visualizations.
        
        Args:
            equity_curve: DataFrame with equity curve data
            trades: List of trade dictionaries
            output_dir: Directory to save report visualizations
            strategy_name: Name of the strategy for report
            
        Returns:
            Dictionary with report data and paths to visualizations
        """
        logger.info(f"Generating performance report for {strategy_name}")
        
        # Calculate performance metrics
        metrics = self.calculate_metrics(equity_curve)
        
        # Create output directory if specified
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate visualizations
        visualization_paths = {}
        if output_dir:
            # Equity curve chart
            equity_path = output_dir / f"{strategy_name.lower().replace(' ', '_')}_equity_curve.png"
            self._plot_equity_curve(equity_curve, strategy_name, equity_path)
            visualization_paths['equity_curve'] = str(equity_path)
            
            # Drawdown chart
            drawdown_path = output_dir / f"{strategy_name.lower().replace(' ', '_')}_drawdown.png"
            self._plot_drawdowns(equity_curve, strategy_name, drawdown_path)
            visualization_paths['drawdown'] = str(drawdown_path)
            
            # Monthly returns heatmap
            monthly_path = output_dir / f"{strategy_name.lower().replace(' ', '_')}_monthly_returns.png"
            self._plot_monthly_returns(equity_curve, strategy_name, monthly_path)
            visualization_paths['monthly_returns'] = str(monthly_path)
            
            # Trade analysis
            if trades:
                trades_path = output_dir / f"{strategy_name.lower().replace(' ', '_')}_trade_analysis.png"
                self._plot_trade_analysis(trades, strategy_name, trades_path)
                visualization_paths['trade_analysis'] = str(trades_path)
        
        # Compile report
        report = {
            'strategy_name': strategy_name,
            'metrics': metrics,
            'visualizations': visualization_paths
        }
        
        # Save report as JSON if output directory specified
        if output_dir:
            report_path = output_dir / f"{strategy_name.lower().replace(' ', '_')}_report.json"
            with open(report_path, 'w') as f:
                import json
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Performance report saved to {report_path}")
        
        return report
    
    def _plot_equity_curve(self, equity_curve: pd.DataFrame, strategy_name: str, output_path: Optional[Path] = None):
        """
        Plot equity curve with benchmark comparison.
        
        Args:
            equity_curve: DataFrame with equity curve data
            strategy_name: Name of the strategy
            output_path: Path to save the plot
        """
        plt.figure(figsize=(12, 6))
        
        # Plot equity curve
        plt.plot(equity_curve.index, equity_curve['equity'], label=strategy_name, linewidth=2)
        
        # Plot benchmark if available
        if 'benchmark' in equity_curve.columns:
            plt.plot(equity_curve.index, equity_curve['benchmark'], label='Benchmark', linewidth=1, alpha=0.7)
        
        # Add initial capital reference line
        plt.axhline(y=equity_curve['equity'].iloc[0], color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
        
        # Format plot
        plt.title(f'Equity Curve - {strategy_name}', fontsize=14)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Equity ($)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save or show plot
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def _plot_drawdowns(self, equity_curve: pd.DataFrame, strategy_name: str, output_path: Optional[Path] = None):
        """
        Plot drawdown chart.
        
        Args:
            equity_curve: DataFrame with equity curve data
            strategy_name: Name of the strategy
            output_path: Path to save the plot
        """
        # Calculate drawdowns
        drawdown_series = 1 - equity_curve['equity'] / equity_curve['equity'].cummax()
        
        plt.figure(figsize=(12, 6))
        
        # Plot drawdowns
        plt.fill_between(drawdown_series.index, 0, drawdown_series * 100, color='red', alpha=0.3)
        plt.plot(drawdown_series.index, drawdown_series * 100, color='red', linewidth=1)
        
        # Format plot
        plt.title(f'Drawdowns - {strategy_name}', fontsize=14)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Drawdown (%)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.gca().invert_yaxis()  # Invert y-axis for better visualization
        
        # Save or show plot
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def _plot_monthly_returns(self, equity_curve: pd.DataFrame, strategy_name: str, output_path: Optional[Path] = None):
        """
        Plot monthly returns heatmap.
        
        Args:
            equity_curve: DataFrame with equity curve data
            strategy_name: Name of the strategy
            output_path: Path to save the plot
        """
        # Calculate daily returns
        daily_returns = equity_curve['equity'].pct_change().dropna()
        
        # Resample to monthly returns
        monthly_returns = daily_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        
        # Create a pivot table for the heatmap
        monthly_returns.index = monthly_returns.index.to_period('M')
        monthly_pivot = monthly_returns.groupby([monthly_returns.index.year, monthly_returns.index.month]).first().unstack()
        
        # Replace month numbers with names
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        monthly_pivot.columns = month_names[:len(monthly_pivot.columns)]
        
        plt.figure(figsize=(12, 8))
        
        # Create heatmap
        sns.heatmap(monthly_pivot * 100, 
                   annot=True, 
                   fmt=".2f", 
                   cmap='RdYlGn', 
                   center=0, 
                   linewidths=1, 
                   cbar_kws={'label': 'Return (%)'})
        
        # Format plot
        plt.title(f'Monthly Returns (%) - {strategy_name}', fontsize=14)
        plt.ylabel('Year', fontsize=12)
        
        # Save or show plot
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def _plot_trade_analysis(self, trades: List[Dict[str, Any]], strategy_name: str, output_path: Optional[Path] = None):
        """
        Plot trade analysis charts.
        
        Args:
            trades: List of trade dictionaries
            strategy_name: Name of the strategy
            output_path: Path to save the plot
        """
        # Convert trades to DataFrame
        trades_df = pd.DataFrame(trades)
        
        # Calculate trade durations
        if 'entry_date' in trades_df.columns and 'exit_date' in trades_df.columns:
            trades_df['duration'] = (pd.to_datetime(trades_df['exit_date']) - pd.to_datetime(trades_df['entry_date'])).dt.days
        
        plt.figure(figsize=(15, 10))
        
        # Create subplot grid
        gs = plt.GridSpec(2, 2, figure=plt.gcf())
        
        # Plot 1: Trade PnL
        ax1 = plt.subplot(gs[0, 0])
        trades_df['pnl'].cumsum().plot(ax=ax1)
        ax1.set_title('Cumulative PnL', fontsize=12)
        ax1.set_xlabel('Trade Number')
        ax1.set_ylabel('Cumulative PnL ($)')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Trade PnL Distribution
        ax2 = plt.subplot(gs[0, 1])
        sns.histplot(trades_df['pnl'], kde=True, ax=ax2)
        ax2.axvline(x=0, color='red', linestyle='--')
        ax2.set_title('PnL Distribution', fontsize=12)
        ax2.set_xlabel('PnL ($)')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Win/Loss by Duration
        if 'duration' in trades_df.columns:
            ax3 = plt.subplot(gs[1, 0])
            trades_df['win'] = trades_df['pnl'] > 0
            sns.scatterplot(x='duration', y='pnl', hue='win', data=trades_df, ax=ax3)
            ax3.set_title('PnL by Trade Duration', fontsize=12)
            ax3.set_xlabel('Duration (days)')
            ax3.set_ylabel('PnL ($)')
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Monthly Trade Count
        ax4 = plt.subplot(gs[1, 1])
        if 'entry_date' in trades_df.columns:
            trades_df['entry_month'] = pd.to_datetime(trades_df['entry_date']).dt.to_period('M')
            monthly_count = trades_df.groupby('entry_month').size()
            monthly_count.plot(kind='bar', ax=ax4)
            ax4.set_title('Monthly Trade Count', fontsize=12)
            ax4.set_xlabel('Month')
            ax4.set_ylabel('Number of Trades')
            ax4.tick_params(axis='x', rotation=45)
            ax4.grid(True, alpha=0.3)
        
        # Adjust layout
        plt.suptitle(f'Trade Analysis - {strategy_name}', fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        # Save or show plot
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
