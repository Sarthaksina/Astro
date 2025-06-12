#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Vedic Trading Strategy Example

This script demonstrates how to use the Cosmic Market Oracle's trading strategy framework
to create, backtest, and visualize trading strategies based on Vedic astrological signals.

It shows:
1. How to load market and planetary data
2. How to configure and run a Vedic astrology-based trading strategy
3. How to backtest the strategy and analyze performance
4. How to visualize the results

Usage:
    python vedic_trading_strategy_example.py

Requirements:
    - pandas
    - matplotlib
    - numpy
    - src.trading modules
    - src.data_processing modules
    - src.astro_engine modules
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging
from src.trading.unified_mcts import MCTS, MCTSPredictor
from src.trading.modular_hierarchical_rl import ModularHierarchicalRLAgent, MCTSStrategicPlanner, PPOTacticalExecutor

# Ensure Python 3.10 compatibility
if sys.version_info.major != 3 or sys.version_info.minor != 10:
    print("Warning: This script is optimized for Python 3.10")

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import project modules
from src.trading.strategy_framework import BaseStrategy # Changed: VedicAstrologyStrategy removed, BaseStrategy kept/added
from src.trading.signal_generator import (
    VedicNakshatraSignalGenerator, 
    VedicYogaSignalGenerator,
    VedicDashaSignalGenerator,
    CombinedSignalGenerator
)
from src.trading.backtest import BacktestEngine, BacktestRunner
from src.data_processing.market_data import MarketDataFetcher
from src.astro_engine.planetary_positions import PlanetaryCalculator # Ensure this is present
# VedicAnalyzer and FinancialYogaAnalyzer are confirmed to be no longer needed directly.
from src.feature_engineering.astrological_features import AstrologicalFeatureGenerator # Ensure this is present
from src.utils.logger import get_logger

# Configure logging
logger = get_logger("vedic_trading_example")


def load_market_data(symbol: str = "^DJI", start_date: str = "2018-01-01", 
                    end_date: str = "2022-12-31") -> pd.DataFrame:
    """
    Load market data for the specified symbol and date range.
    
    Args:
        symbol: Market symbol to fetch data for
        start_date: Start date for data (YYYY-MM-DD)
        end_date: End date for data (YYYY-MM-DD)
        
    Returns:
        DataFrame with market data
    """
    logger.info(f"Loading market data for {symbol} from {start_date} to {end_date}")
    
    try:
        # Try to use MarketDataFetcher
        fetcher = MarketDataFetcher()
        market_data = fetcher.fetch_historical_data(symbol, start_date, end_date)
        
    except Exception as e:
        logger.warning(f"Error using MarketDataFetcher: {e}")
        logger.info("Falling back to sample data")
        
        # Generate sample data if fetcher fails
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        days = (end - start).days + 1
        
        # Create date range
        date_range = [start + timedelta(days=i) for i in range(days)]
        
        # Generate random price data with a trend
        np.random.seed(42)  # For reproducibility
        
        # Start price
        price = 25000
        
        # Generate prices with random walk and some seasonality
        prices = []
        for i in range(days):
            # Random daily change (-1% to +1%)
            daily_return = np.random.normal(0.0003, 0.01)
            
            # Add some seasonality (annual cycle)
            seasonal_factor = 0.1 * np.sin(2 * np.pi * i / 365)
            
            # Update price
            price = price * (1 + daily_return + seasonal_factor)
            prices.append(price)
        
        # Create DataFrame
        market_data = pd.DataFrame({
            "Date": date_range,
            "Open": prices,
            "High": [p * (1 + np.random.uniform(0, 0.01)) for p in prices],
            "Low": [p * (1 - np.random.uniform(0, 0.01)) for p in prices],
            "Close": prices,
            "Volume": [int(np.random.uniform(1000000, 5000000)) for _ in range(days)]
        })
        
        # Set index
        market_data.set_index("Date", inplace=True)
    
    logger.info(f"Loaded {len(market_data)} days of market data")
    
    return market_data


def generate_planetary_data(start_date: str = "2018-01-01", 
                          end_date: str = "2022-12-31") -> pd.DataFrame:
    """
    Generate planetary data for the specified date range.
    
    Args:
        start_date: Start date for data (YYYY-MM-DD)
        end_date: End date for data (YYYY-MM-DD)
        
    Returns:
        DataFrame with planetary data
    """
    logger.info(f"Generating astrological features from {start_date} to {end_date}")
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    
    # Create a list of dates for the feature generator
    # Ensure dates are datetime objects, not just strings, for AstrologicalFeatureGenerator
    date_list = [start_dt + timedelta(days=x) for x in range((end_dt - start_dt).days + 1)]
    
    # Initialize AstrologicalFeatureGenerator
    astro_feature_gen = AstrologicalFeatureGenerator()
    
    # Generate features
    # generate_features_for_dates returns a DataFrame with dates as index
    astro_features_df = astro_feature_gen.generate_features_for_dates(date_list)
    
    logger.info(f"Generated {len(astro_features_df.columns)} astrological features for {len(astro_features_df)} days")
    return astro_features_df


def run_single_strategy_example(market_data: pd.DataFrame, planetary_data: pd.DataFrame):
    """
    Run a single Vedic astrology trading strategy example.
    
    Args:
        market_data: Market data DataFrame
        planetary_data: Planetary data DataFrame
    """
    logger.info("Running single strategy example")

    # Define a new simple strategy class locally
    class MyExampleStrategy(BaseStrategy):
        def __init__(self, signal_generator_instance, name="Modular Vedic Strategy", description="Uses CombinedSignalGenerator"):
            super().__init__(name, description)
            self.signal_generator = signal_generator_instance

        def generate_signals(self, market_data: pd.DataFrame, planetary_data: pd.DataFrame) -> pd.DataFrame:
            # planetary_data here is the DataFrame of astrological features
            return self.signal_generator.generate_signals(market_data, planetary_data)

    # Instantiate CombinedSignalGenerator
    combined_gen = CombinedSignalGenerator()
    # Optionally, add specific generators to combined_gen if defaults are not desired
    # e.g., combined_gen.add_generator(VedicNakshatraSignalGenerator(), weight=0.5)
    # For this example, default CombinedSignalGenerator (which includes Nakshatra, Yoga, Dasha) is fine.

    # Instantiate the new strategy
    strategy = MyExampleStrategy(signal_generator_instance=combined_gen)
    
    # Create backtest engine
    engine = BacktestEngine(initial_capital=100000.0, commission=0.001)
    
    # Run backtest
    results = engine.run_backtest(strategy, market_data, planetary_data, "MARKET")
    
    # Generate report
    report_path = engine.generate_report("reports")
    
    # Plot results
    engine.plot_results("reports/equity_curve.png")
    engine.plot_drawdowns("reports/drawdowns.png")
    
    logger.info(f"Backtest report generated: {report_path}")
    logger.info(f"Performance metrics: {results['metrics']}")
    logger.info(f"Total trades: {len(results['trades'])}")


def run_strategy_comparison_example(market_data: pd.DataFrame, planetary_data: pd.DataFrame):
    """
    Run a comparison of different Vedic astrology trading strategies.
    
    Args:
        market_data: Market data DataFrame
        planetary_data: Planetary data DataFrame
    """
    logger.info("Running strategy comparison example")
    
    # Create backtest runner
    runner = BacktestRunner(market_data, planetary_data)
    
    # Run parameter sweep
    sweep_results = runner.run_parameter_sweep("MARKET")
    
    # Visualize results
    runner.visualize_results("reports")
    
    logger.info(f"Parameter sweep completed")
    logger.info(f"Report generated: {sweep_results['report_path']}")


def run_signal_generator_example(market_data: pd.DataFrame, planetary_data: pd.DataFrame):
    """
    Run an example using different signal generators.
    
    Args:
        market_data: Market data DataFrame
        planetary_data: Planetary data DataFrame
    """
    logger.info("Running signal generator example")
    
    # Create individual signal generators
    nakshatra_generator = VedicNakshatraSignalGenerator()
    yoga_generator = VedicYogaSignalGenerator()
    dasha_generator = VedicDashaSignalGenerator()
    
    # Create combined signal generator
    combined_generator = CombinedSignalGenerator()
    combined_generator.add_generator(nakshatra_generator, 0.3)
    combined_generator.add_generator(yoga_generator, 0.4)
    combined_generator.add_generator(dasha_generator, 0.3)
    
    # Generate signals
    nakshatra_signals = nakshatra_generator.generate_signals(market_data, planetary_data)
    yoga_signals = yoga_generator.generate_signals(market_data, planetary_data)
    dasha_signals = dasha_generator.generate_signals(market_data, planetary_data)
    combined_signals = combined_generator.generate_signals(market_data, planetary_data)
    
    # Count signals
    nakshatra_count = (nakshatra_signals["signal"] != 0).sum()
    yoga_count = (yoga_signals["signal"] != 0).sum()
    dasha_count = (dasha_signals["signal"] != 0).sum()
    combined_count = (combined_signals["signal"] != 0).sum()
    
    logger.info(f"Nakshatra signals: {nakshatra_count}")
    logger.info(f"Yoga signals: {yoga_count}")
    logger.info(f"Dasha signals: {dasha_count}")
    logger.info(f"Combined signals: {combined_count}")
    
    # Plot signals
    plt.figure(figsize=(12, 8))
    
    # Plot market data
    plt.subplot(2, 1, 1)
    plt.plot(market_data.index, market_data["Close"], label="Market")
    plt.title("Market Price")
    plt.legend()
    plt.grid(True)
    
    # Plot signals
    plt.subplot(2, 1, 2)
    
    # Buy signals
    buy_dates = combined_signals[combined_signals["signal"] > 0].index
    buy_strengths = combined_signals.loc[buy_dates, "strength"].values
    plt.scatter(buy_dates, [1] * len(buy_dates), s=buy_strengths * 100, 
               marker="^", color="green", label="Buy")
    
    # Sell signals
    sell_dates = combined_signals[combined_signals["signal"] < 0].index
    sell_strengths = combined_signals.loc[sell_dates, "strength"].values
    plt.scatter(sell_dates, [-1] * len(sell_dates), s=sell_strengths * 100, 
               marker="v", color="red", label="Sell")
    
    plt.title("Trading Signals")
    plt.yticks([-1, 0, 1], ["Sell", "Hold", "Buy"])
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("reports/trading_signals.png")
    plt.close()
    
    logger.info("Signal visualization saved to reports/trading_signals.png")


def main():
    """Main function to run the example."""
    # Create reports directory if it doesn't exist
    os.makedirs("reports", exist_ok=True)
    
    # Set date range
    start_date = "2018-01-01"
    end_date = "2022-12-31"
    
    # Load market data
    market_data = load_market_data("^DJI", start_date, end_date)
    
    # Generate planetary data
    planetary_data = generate_planetary_data(start_date, end_date)
    
    # Run examples
    run_single_strategy_example(market_data, planetary_data)
    # TODO: Refactor BacktestRunner or this example to use modular strategies
    # run_strategy_comparison_example(market_data, planetary_data)
    run_signal_generator_example(market_data, planetary_data)
    
    logger.info("Example completed successfully")


if __name__ == "__main__":
    main()
