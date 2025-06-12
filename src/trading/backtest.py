# Cosmic Market Oracle - Backtesting Framework

"""
This module provides a framework for backtesting trading strategies
based on Vedic astrological signals and market data.

It includes:
- Backtesting engine for evaluating strategy performance
- Performance metrics calculation and visualization
- Trade analysis and reporting
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
import os

from src.trading.strategy_framework import BaseStrategy # VedicAstrologyStrategy removed
from src.trading.signal_generator import (
    CombinedSignalGenerator, SignalFilter,
    VedicNakshatraSignalGenerator, VedicYogaSignalGenerator, VedicDashaSignalGenerator
)
from src.utils.logger import get_logger # Changed to get_logger
from src.utils.visualization import create_performance_chart

# Helper strategy class for BacktestRunner
class _ModularStrategy(BaseStrategy):
    """Helper strategy to wrap a signal generator for BacktestRunner."""
    def __init__(self, signal_generator_instance, name="Modular Strategy", description="Uses a given signal generator"):
        super().__init__(name, description)
        self.signal_generator = signal_generator_instance

    def generate_signals(self, market_data: pd.DataFrame, planetary_data: pd.DataFrame) -> pd.DataFrame:
        return self.signal_generator.generate_signals(market_data, planetary_data)

# Configure logging
logger = get_logger("backtest") # Changed to get_logger


class BacktestEngine:
    """Engine for backtesting trading strategies."""
    
    def __init__(self, initial_capital: float = 100000.0, commission: float = 0.001):
        """
        Initialize the backtest engine.
        
        Args:
            initial_capital: Initial capital for the backtest
            commission: Commission rate per trade (as a fraction)
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.results = {}
    
    def run_backtest(self, strategy: BaseStrategy, market_data: pd.DataFrame, 
                    planetary_data: pd.DataFrame, symbol: str = "MARKET") -> Dict:
        """
        Run a backtest for a trading strategy.
        
        Args:
            strategy: Trading strategy to backtest
            market_data: Market data for the backtest period
            planetary_data: Planetary data for the backtest period
            symbol: Trading symbol
            
        Returns:
            Dictionary with backtest results
        """
        logger.info(f"Starting backtest for {strategy.name} on {symbol}")
        
        # Reset strategy state
        strategy.reset()
        
        # Generate signals
        signals = strategy.generate_signals(market_data, planetary_data)
        
        # Apply signal filtering
        signal_filter = SignalFilter()
        filtered_signals = signal_filter.filter_signals(signals, market_data)
        
        # Initialize backtest results
        results = pd.DataFrame(index=market_data.index)
        results["close"] = market_data["Close"]
        results["signal"] = filtered_signals["signal"]
        results["strength"] = filtered_signals["strength"]
        results["position"] = 0.0
        results["cash"] = self.initial_capital
        results["equity"] = self.initial_capital
        results["returns"] = 0.0
        results["trade"] = False
        
        # Track current position
        position = 0.0
        cash = self.initial_capital
        
        # Process each day
        for i, date in enumerate(results.index):
            if i == 0:
                # First day, just record initial state
                results.loc[date, "position"] = position
                results.loc[date, "cash"] = cash
                results.loc[date, "equity"] = cash
                continue
            
            # Get previous day's state
            prev_date = results.index[i-1]
            prev_position = results.loc[prev_date, "position"]
            prev_cash = results.loc[prev_date, "cash"]
            
            # Get current price and signal
            price = results.loc[date, "close"]
            signal = results.loc[date, "signal"]
            strength = results.loc[date, "strength"]
            
            # Initialize with previous values
            position = prev_position
            cash = prev_cash
            trade_executed = False
            
            # Determine trade action
            if signal == 1 and prev_position <= 0:  # Buy signal
                # Calculate position size based on signal strength
                position_value = strategy.calculate_position_size(strength, prev_cash)
                new_position = position_value / price
                
                # Apply commission
                commission_cost = position_value * self.commission
                
                # Update position and cash
                position = new_position
                cash = prev_cash - position_value - commission_cost
                
                # Record trade
                strategy.execute_trade(
                    symbol=symbol,
                    direction="buy",
                    quantity=new_position,
                    price=price,
                    timestamp=date,
                    signal_strength=strength
                )
                
                trade_executed = True
                
            elif signal == -1 and prev_position >= 0:  # Sell signal
                if prev_position > 0:
                    # Sell existing position
                    position_value = prev_position * price
                    commission_cost = position_value * self.commission
                    
                    # Update cash
                    cash = prev_cash + position_value - commission_cost
                    
                    # Record trade
                    strategy.execute_trade(
                        symbol=symbol,
                        direction="sell",
                        quantity=prev_position,
                        price=price,
                        timestamp=date,
                        signal_strength=strength
                    )
                    
                    position = 0
                    trade_executed = True
                
                # Short selling (if allowed)
                position_value = strategy.calculate_position_size(strength, cash)
                short_position = position_value / price
                commission_cost = position_value * self.commission
                
                position = -short_position
                cash = cash + position_value - commission_cost
                
                # Record trade
                strategy.execute_trade(
                    symbol=symbol,
                    direction="sell",
                    quantity=short_position,
                    price=price,
                    timestamp=date,
                    signal_strength=strength
                )
                
                trade_executed = True
            
            # Calculate equity
            equity = cash + (position * price)
            
            # Calculate returns
            prev_equity = results.loc[prev_date, "equity"]
            daily_return = (equity - prev_equity) / prev_equity if prev_equity > 0 else 0
            
            # Record results
            results.loc[date, "position"] = position
            results.loc[date, "cash"] = cash
            results.loc[date, "equity"] = equity
            results.loc[date, "returns"] = daily_return
            results.loc[date, "trade"] = trade_executed
        
        # Calculate cumulative returns
        results["cumulative_returns"] = (1 + results["returns"]).cumprod() - 1
        
        # Calculate drawdowns
        results["peak"] = results["equity"].cummax()
        results["drawdown"] = (results["equity"] - results["peak"]) / results["peak"]
        
        # Calculate performance metrics
        metrics = self._calculate_performance_metrics(results, strategy.trades)
        
        # Store results
        self.results[strategy.name] = {
            "results": results,
            "metrics": metrics,
            "trades": strategy.trades
        }
        
        logger.info(f"Backtest completed for {strategy.name} on {symbol}")
        logger.info(f"Performance metrics: {metrics}")
        
        return {
            "results": results,
            "metrics": metrics,
            "trades": strategy.trades
        }
    
    def _calculate_performance_metrics(self, results: pd.DataFrame, trades: List[Dict]) -> Dict:
        """
        Calculate performance metrics for a backtest.
        
        Args:
            results: Backtest results DataFrame
            trades: List of trade records
            
        Returns:
            Dictionary of performance metrics
        """
        metrics = {}
        
        # Total return
        metrics["total_return"] = results["cumulative_returns"].iloc[-1]
        
        # Annualized return
        days = (results.index[-1] - results.index[0]).days
        if days > 0:
            metrics["annualized_return"] = (1 + metrics["total_return"]) ** (365 / days) - 1
        else:
            metrics["annualized_return"] = 0
        
        # Volatility
        metrics["volatility"] = results["returns"].std() * np.sqrt(252)  # Annualized
        
        # Sharpe ratio
        risk_free_rate = 0.02  # Assumed risk-free rate
        if metrics["volatility"] > 0:
            metrics["sharpe_ratio"] = (metrics["annualized_return"] - risk_free_rate) / metrics["volatility"]
        else:
            metrics["sharpe_ratio"] = 0
        
        # Maximum drawdown
        metrics["max_drawdown"] = results["drawdown"].min()
        
        # Win rate
        if trades:
            # Calculate P&L for each trade
            buy_trades = [t for t in trades if t["direction"] == "buy"]
            sell_trades = [t for t in trades if t["direction"] == "sell"]
            
            # Pair buy and sell trades to calculate P&L
            # This is a simplified approach; a real implementation would be more complex
            winning_trades = 0
            losing_trades = 0
            
            for i in range(min(len(buy_trades), len(sell_trades))):
                buy_price = buy_trades[i]["price"]
                sell_price = sell_trades[i]["price"]
                
                if sell_price > buy_price:
                    winning_trades += 1
                else:
                    losing_trades += 1
            
            total_trades = winning_trades + losing_trades
            metrics["win_rate"] = winning_trades / total_trades if total_trades > 0 else 0
            metrics["total_trades"] = len(trades)
        else:
            metrics["win_rate"] = 0
            metrics["total_trades"] = 0
        
        # Average trade duration
        if len(trades) >= 2:
            durations = []
            for i in range(1, len(trades)):
                if trades[i]["direction"] != trades[i-1]["direction"]:
                    duration = (trades[i]["timestamp"] - trades[i-1]["timestamp"]).days
                    durations.append(duration)
            
            metrics["avg_trade_duration"] = np.mean(durations) if durations else 0
        else:
            metrics["avg_trade_duration"] = 0
        
        return metrics
    
    def compare_strategies(self, strategies: List[BaseStrategy], market_data: pd.DataFrame, 
                         planetary_data: pd.DataFrame, symbol: str = "MARKET") -> Dict:
        """
        Compare multiple trading strategies.
        
        Args:
            strategies: List of trading strategies to compare
            market_data: Market data for the backtest period
            planetary_data: Planetary data for the backtest period
            symbol: Trading symbol
            
        Returns:
            Dictionary with comparison results
        """
        # Run backtest for each strategy
        for strategy in strategies:
            self.run_backtest(strategy, market_data, planetary_data, symbol)
        
        # Combine equity curves
        equity_curves = pd.DataFrame(index=market_data.index)
        equity_curves["Market"] = market_data["Close"] / market_data["Close"].iloc[0] * self.initial_capital
        
        for strategy_name, result in self.results.items():
            equity_curves[strategy_name] = result["results"]["equity"]
        
        # Combine metrics
        metrics_comparison = pd.DataFrame()
        for strategy_name, result in self.results.items():
            metrics = result["metrics"]
            metrics_df = pd.DataFrame(metrics, index=[strategy_name])
            metrics_comparison = pd.concat([metrics_comparison, metrics_df])
        
        # Add market metrics
        market_returns = market_data["Close"].pct_change().dropna()
        market_cumulative_return = (1 + market_returns).cumprod().iloc[-1] - 1
        market_annualized_return = (1 + market_cumulative_return) ** (365 / days) - 1 if days > 0 else 0
        market_volatility = market_returns.std() * np.sqrt(252)
        market_sharpe = (market_annualized_return - 0.02) / market_volatility if market_volatility > 0 else 0
        
        market_metrics = {
            "total_return": market_cumulative_return,
            "annualized_return": market_annualized_return,
            "volatility": market_volatility,
            "sharpe_ratio": market_sharpe,
            "max_drawdown": (market_data["Close"] / market_data["Close"].cummax() - 1).min(),
            "win_rate": np.nan,
            "total_trades": 0,
            "avg_trade_duration": np.nan
        }
        
        metrics_comparison = pd.concat([metrics_comparison, pd.DataFrame(market_metrics, index=["Market"])])
        
        return {
            "equity_curves": equity_curves,
            "metrics_comparison": metrics_comparison
        }
    
    def visualize_results(self, output_dir: str = "reports"):
        """
        Visualize backtest results using the continuous learning framework.
        
        Args:
            output_dir: Directory to save the visualizations
        """
        from src.trading.continuous_learning import PerformanceMonitor, ABTestingFramework
        
        # Create a temporary PerformanceMonitor to use its visualization
        monitor = PerformanceMonitor(
            strategy=None,
            backtest_engine=self,
            retrain_callback=lambda: None,
            market_data=None,
            planetary_data=None,
            window_size=30
        )
        
        # Add results to performance history
        for strategy_name, result in self.results.items():
            monitor.performance_history.append({
                'date': result["results"].index[-1],
                'returns': result["results"]["returns"],
                'equity': result["results"]["equity"],
                'strategy_name': strategy_name
            })
        
        # Use PerformanceMonitor's visualization
        monitor.plot_performance_history(metric="returns", save_path=os.path.join(output_dir, "returns.png"))
        monitor.plot_performance_history(metric="equity", save_path=os.path.join(output_dir, "equity.png"))
        
        # Create performance report
        report_path = os.path.join(output_dir, "performance_report.txt")
        with open(report_path, "w") as f:
            f.write("=== Performance Report ===\n")
            for strategy_name, result in self.results.items():
                f.write(f"Strategy: {strategy_name}\n")
                f.write(f"Start Date: {result['results'].index[0]}\n")
                f.write(f"End Date: {result['results'].index[-1]}\n")
                f.write(f"Total Return: {result['results']['returns'].sum():.2%}\n")
                f.write(f"Annualized Return: {result['results']['returns'].mean() * 252:.2%}\n")
                f.write(f"Sharpe Ratio: {result['results']['returns'].mean() / result['results']['returns'].std() * np.sqrt(252):.2f}\n")
                f.write(f"Max Drawdown: {result['results']['equity'].min() / result['results']['equity'].iloc[0] - 1:.2%}\n")
                f.write(f"Number of Trades: {result['results']['trade'].sum()}\n")
                f.write(f"Win Rate: {result['results'][result['results']['trade'] == True]['returns'].gt(0).mean():.2%}\n")
                f.write("\n")
    
    def generate_report(self, output_dir: str = "reports") -> str:
        """
        Generate a backtest report.
        
        Args:
            output_dir: Directory to save the report
            
        Returns:
            Path to the generated report
        """
        if not self.results:
            logger.warning("No backtest results to generate report")
            return ""
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate report filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"backtest_report_{timestamp}.html"
        report_path = os.path.join(output_dir, report_filename)
        
        # Generate HTML report
        with open(report_path, "w") as f:
            f.write("<html>\n")
            f.write("<head>\n")
            f.write("<title>Cosmic Market Oracle - Backtest Report</title>\n")
            f.write("<style>\n")
            f.write("body { font-family: Arial, sans-serif; margin: 20px; }\n")
            f.write("h1 { color: #2c3e50; }\n")
            f.write("h2 { color: #3498db; }\n")
            f.write("table { border-collapse: collapse; width: 100%; }\n")
            f.write("th, td { border: 1px solid #ddd; padding: 8px; text-align: right; }\n")
            f.write("th { background-color: #f2f2f2; }\n")
            f.write("tr:nth-child(even) { background-color: #f9f9f9; }\n")
            f.write("</style>\n")
            f.write("</head>\n")
            f.write("<body>\n")
            
            # Report header
            f.write("<h1>Cosmic Market Oracle - Backtest Report</h1>\n")
            f.write(f"<p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>\n")
            
            # Performance metrics
            f.write("<h2>Performance Metrics</h2>\n")
            f.write("<table>\n")
            
            # Table header
            f.write("<tr>\n")
            f.write("<th>Strategy</th>\n")
            f.write("<th>Total Return</th>\n")
            f.write("<th>Annualized Return</th>\n")
            f.write("<th>Volatility</th>\n")
            f.write("<th>Sharpe Ratio</th>\n")
            f.write("<th>Max Drawdown</th>\n")
            f.write("<th>Win Rate</th>\n")
            f.write("<th>Total Trades</th>\n")
            f.write("<th>Avg Trade Duration</th>\n")
            f.write("</tr>\n")
            
            # Table rows
            for strategy_name, result in self.results.items():
                metrics = result["metrics"]
                f.write("<tr>\n")
                f.write(f"<td>{strategy_name}</td>\n")
                f.write(f"<td>{metrics['total_return']:.2%}</td>\n")
                f.write(f"<td>{metrics['annualized_return']:.2%}</td>\n")
                f.write(f"<td>{metrics['volatility']:.2%}</td>\n")
                f.write(f"<td>{metrics['sharpe_ratio']:.2f}</td>\n")
                f.write(f"<td>{metrics['max_drawdown']:.2%}</td>\n")
                f.write(f"<td>{metrics['win_rate']:.2%}</td>\n")
                f.write(f"<td>{metrics['total_trades']}</td>\n")
                f.write(f"<td>{metrics['avg_trade_duration']:.1f} days</td>\n")
                f.write("</tr>\n")
            
            f.write("</table>\n")
            
            # Trade details for each strategy
            for strategy_name, result in self.results.items():
                trades = result["trades"]
                
                f.write(f"<h2>Trades - {strategy_name}</h2>\n")
                
                if trades:
                    f.write("<table>\n")
                    f.write("<tr>\n")
                    f.write("<th>Date</th>\n")
                    f.write("<th>Direction</th>\n")
                    f.write("<th>Symbol</th>\n")
                    f.write("<th>Quantity</th>\n")
                    f.write("<th>Price</th>\n")
                    f.write("<th>Signal Strength</th>\n")
                    f.write("</tr>\n")
                    
                    for trade in trades:
                        f.write("<tr>\n")
                        f.write(f"<td>{trade['timestamp'].strftime('%Y-%m-%d')}</td>\n")
                        f.write(f"<td>{trade['direction'].upper()}</td>\n")
                        f.write(f"<td>{trade['symbol']}</td>\n")
                        f.write(f"<td>{trade['quantity']:.2f}</td>\n")
                        f.write(f"<td>{trade['price']:.2f}</td>\n")
                        f.write(f"<td>{trade['signal_strength']:.2f}</td>\n")
                        f.write("</tr>\n")
                    
                    f.write("</table>\n")
                else:
                    f.write("<p>No trades executed</p>\n")
            
            f.write("</body>\n")
            f.write("</html>\n")
        
        logger.info(f"Backtest report generated: {report_path}")
        
        return report_path
    
    def plot_results(self, save_path: str = None) -> None:
        """
        Plot backtest results.
        
        Args:
            save_path: Path to save the plot
        """
        if not self.results:
            logger.warning("No backtest results to plot")
            return
        
        # Set up the plot
        plt.figure(figsize=(12, 8))
        
        # Plot equity curves
        for strategy_name, result in self.results.items():
            equity = result["results"]["equity"]
            plt.plot(equity.index, equity, label=strategy_name)
        
        # Add labels and title
        plt.xlabel("Date")
        plt.ylabel("Equity")
        plt.title("Backtest Results - Equity Curves")
        plt.legend()
        plt.grid(True)
        
        # Save or show the plot
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_drawdowns(self, save_path: str = None) -> None:
        """
        Plot drawdowns for each strategy.
        
        Args:
            save_path: Path to save the plot
        """
        if not self.results:
            logger.warning("No backtest results to plot")
            return
        
        # Set up the plot
        plt.figure(figsize=(12, 8))
        
        # Plot drawdowns
        for strategy_name, result in self.results.items():
            drawdowns = result["results"]["drawdown"]
            plt.plot(drawdowns.index, drawdowns, label=strategy_name)
        
        # Add labels and title
        plt.xlabel("Date")
        plt.ylabel("Drawdown")
        plt.title("Backtest Results - Drawdowns")
        plt.legend()
        plt.grid(True)
        
        # Save or show the plot
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Drawdown plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()


class BacktestRunner:
    """Helper class to run backtests with different configurations."""
    
    def __init__(self, market_data: pd.DataFrame, planetary_data: pd.DataFrame):
        """
        Initialize the backtest runner.
        
        Args:
            market_data: Market data for the backtest period
            planetary_data: Planetary data for the backtest period
        """
        self.market_data = market_data
        self.planetary_data = planetary_data
        self.engine = BacktestEngine()
    
    def run_vedic_strategy_backtest(self, symbol: str = "MARKET", 
                                   use_yogas: bool = True, 
                                   use_nakshatras: bool = True, 
                                   use_dashas: bool = True,
                                   min_signal_strength: float = 0.6) -> Dict:
        """
        Run a backtest for the Vedic Astrology strategy.
        
        Args:
            symbol: Trading symbol
            use_yogas: Whether to use financial yogas
            use_nakshatras: Whether to use nakshatra analysis
            use_dashas: Whether to use dasha periods
            min_signal_strength: Minimum signal strength to generate a trade
            
        Returns:
            Dictionary with backtest results
        """
        # Instantiate and Configure CombinedSignalGenerator
        combined_gen = CombinedSignalGenerator()
        combined_gen.generators = []  # Clear default generators
        combined_gen.weights = {}

        active_generator_details = [] # To store (generator_instance, weight_proportion_if_all_active)
        
        # Define desired weights if all were active
        default_weights = {'yoga': 0.4, 'nakshatra': 0.3, 'dasha': 0.3}

        if use_yogas:
            active_generator_details.append( (VedicYogaSignalGenerator(), default_weights['yoga']) )
        if use_nakshatras:
            active_generator_details.append( (VedicNakshatraSignalGenerator(), default_weights['nakshatra']) )
        if use_dashas:
            active_generator_details.append( (VedicDashaSignalGenerator(), default_weights['dasha']) )

        # Add selected generators and normalize their weights
        total_weight_for_active = sum(details[1] for details in active_generator_details)

        if active_generator_details:
            for gen_instance, relative_weight in active_generator_details:
                combined_gen.generators.append(gen_instance)
                # Normalize weight based on selected generators
                combined_gen.weights[gen_instance.name] = relative_weight / total_weight_for_active if total_weight_for_active > 0 else 0
        else:
            logger.warning(f"No signal generators selected for Vedic Strategy in BacktestRunner. Strategy will produce no signals.")
            # combined_gen will remain empty and produce no signals

        combined_gen.min_signal_strength = min_signal_strength

        # Instantiate the Wrapper Strategy
        strategy_name = f"Modular Vedic (Y:{use_yogas}, N:{use_nakshatras}, D:{use_dashas}, S:{min_signal_strength})"
        strategy = _ModularStrategy(signal_generator_instance=combined_gen, name=strategy_name)
        
        # Run backtest
        results = self.engine.run_backtest(strategy, self.market_data, self.planetary_data, symbol)
        
        return results
    
    def run_parameter_sweep(self, symbol: str = "MARKET") -> Dict:
        """
        Run a parameter sweep to find the best strategy configuration.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary with sweep results
        """
        # Define parameter combinations
        param_combinations = [
            {"use_yogas": True, "use_nakshatras": True, "use_dashas": True, "min_signal_strength": 0.6},
            {"use_yogas": True, "use_nakshatras": True, "use_dashas": False, "min_signal_strength": 0.6},
            {"use_yogas": True, "use_nakshatras": False, "use_dashas": True, "min_signal_strength": 0.6},
            {"use_yogas": False, "use_nakshatras": True, "use_dashas": True, "min_signal_strength": 0.6},
            {"use_yogas": True, "use_nakshatras": True, "use_dashas": True, "min_signal_strength": 0.5},
            {"use_yogas": True, "use_nakshatras": True, "use_dashas": True, "min_signal_strength": 0.7}
        ]
        
        # Run backtest for each combination
        for params in param_combinations:
            self.run_vedic_strategy_backtest(
                symbol=symbol,
                use_yogas=params["use_yogas"],
                use_nakshatras=params["use_nakshatras"],
                use_dashas=params["use_dashas"],
                min_signal_strength=params["min_signal_strength"]
            )
        
        # Compare strategies
        comparison = self.engine.compare_strategies([], self.market_data, self.planetary_data, symbol)
        
        # Generate report
        report_path = self.engine.generate_report()
        
        return {
            "comparison": comparison,
            "report_path": report_path
        }
    
    def visualize_results(self, output_dir: str = "reports") -> None:
        """
        Visualize backtest results.
        
        Args:
            output_dir: Directory to save the visualizations
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Plot equity curves
        equity_path = os.path.join(output_dir, f"equity_curves_{timestamp}.png")
        self.engine.plot_results(equity_path)
        
        # Plot drawdowns
        drawdown_path = os.path.join(output_dir, f"drawdowns_{timestamp}.png")
        self.engine.plot_drawdowns(drawdown_path)
        
        logger.info(f"Visualizations saved to {output_dir}")
