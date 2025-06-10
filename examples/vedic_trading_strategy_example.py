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
from src.trading.strategy_framework import VedicAstrologyStrategy
from src.trading.signal_generator import (
    VedicNakshatraSignalGenerator, 
    VedicYogaSignalGenerator,
    VedicDashaSignalGenerator,
    CombinedSignalGenerator
)
from src.trading.backtest import BacktestEngine, BacktestRunner
from src.data_processing.market_data import MarketDataFetcher
from src.astro_engine.planetary_positions import PlanetaryCalculator
from src.astro_engine.vedic_analysis import VedicAnalyzer # Added
from src.astro_engine.financial_yogas import FinancialYogaAnalyzer # Added
from src.feature_engineering.astrological_features import AstrologicalFeatureGenerator
from src.utils.logger import setup_logger

# Configure logging
logger = setup_logger("vedic_trading_example")


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
    logger.info(f"Generating planetary data from {start_date} to {end_date}")
    
    # Convert string dates to datetime
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    days = (end - start).days + 1
    
    # Create date range
    date_range = [start + timedelta(days=i) for i in range(days)]
    
    # Initialize planetary calculator
    calculator = PlanetaryCalculator()
    vedic_analyzer = VedicAnalyzer() # Added
    yoga_analyzer = FinancialYogaAnalyzer(calculator) # Added
    
    # Initialize feature generator
    feature_generator = AstrologicalFeatureGenerator() # This itself uses VedicAnalyzer and FinancialYogaAnalyzer now
    
    # Generate data for each date
    planetary_data = []
    
    for date in date_range:
        # Get basic planetary positions
        positions = calculator.calculate_planet_positions(date)
        
        # Get nakshatra details for Moon
        moon_longitude = positions["Moon"]["longitude"]
        nakshatra_details = calculator.get_nakshatra_details(moon_longitude)
        
        # Get current dasha lord
        dasha_info = calculator.calculate_vimshottari_dasha(date)
        
        # Analyze market trend, financial yogas using VedicAnalyzer
        vedic_analysis_results = vedic_analyzer.analyze_date(date)
        market_trend_info = vedic_analysis_results.get("integrated_forecast", {})
        
        # Extract yoga information (key_yogas is a list of strings, not dicts with 'market_impact')
        # The original code iterated `yogas` (a list of dicts) for 'market_impact'.
        # VedicAnalyzer's `key_yogas` is a list of strings.
        # For a more direct replacement of yoga counts, we might need to use yoga_analyzer directly here
        # or ensure VedicAnalyzer provides the structured yoga list.
        # For now, let's get key_yogas and adapt if possible, or default counts.
        
        key_yogas_list = vedic_analysis_results.get("key_yogas", [])
        # This is a simplified yoga count based on presence of yoga names; original code was more detailed.
        # This part may need further refinement based on actual structure of `key_yogas` if they were dicts.
        # Assuming key_yogas are strings, we can't directly get 'market_impact'.
        # For a placeholder:
        bullish_yogas_count = sum(1 for yoga_name in key_yogas_list if "bullish" in yoga_name.lower())
        bearish_yogas_count = sum(1 for yoga_name in key_yogas_list if "bearish" in yoga_name.lower())
        volatile_yogas_count = sum(1 for yoga_name in key_yogas_list if "volatile" in yoga_name.lower())
        neutral_yogas_count = len(key_yogas_list) - bullish_yogas_count - bearish_yogas_count - volatile_yogas_count


        # Generate special features
        special_features = feature_generator.generate_special_features(date)
        
        # Combine all data
        data = {
            "Date": date,
            # Basic planetary positions
            "sun_longitude": positions["Sun"]["longitude"],
            "moon_longitude": moon_longitude,
            "mercury_longitude": positions["Mercury"]["longitude"],
            "venus_longitude": positions["Venus"]["longitude"],
            "mars_longitude": positions["Mars"]["longitude"],
            "jupiter_longitude": positions["Jupiter"]["longitude"],
            "saturn_longitude": positions["Saturn"]["longitude"],
            
            # Nakshatra details
            "moon_nakshatra": nakshatra_details["nakshatra_name"],
            "moon_nakshatra_pada": nakshatra_details["pada"],
            "moon_nakshatra_financial": "bullish" if nakshatra_details["nakshatra_number"] % 3 == 1 else 
                                       "bearish" if nakshatra_details["nakshatra_number"] % 3 == 2 else 
                                       "neutral",
            
            # Dasha information
            "current_dasha_lord": dasha_info["mahadasha"]["planet"],
            "current_antardasha_lord": dasha_info["antardasha"]["planet"],
            
            # Market trend
            "market_trend_primary_trend": market_trend_info.get("trend", "Neutral"),
            "market_trend_strength": market_trend_info.get("trend_score", 0.0) * 100, # Example scaling
            "market_trend_reversal_probability": 0.0, # Placeholder
            
            # Financial yogas (using simplified counts from above)
            "bullish_yoga_count": bullish_yogas_count,
            "bearish_yoga_count": bearish_yogas_count,
            "neutral_yoga_count": neutral_yogas_count, # Note: original had neutral, new has volatile
            # Adding volatile_yoga_count as it was in original grep results for astrological_features
            "volatile_yoga_count": volatile_yogas_count
        }
        
        # Add special features
        data.update(special_features)
        
        planetary_data.append(data)
    
    # Convert to DataFrame
    df = pd.DataFrame(planetary_data)
    df.set_index("Date", inplace=True)
    
    logger.info(f"Generated planetary data for {len(df)} days")
    
    return df


def run_single_strategy_example(market_data: pd.DataFrame, planetary_data: pd.DataFrame):
    """
    Run a single Vedic astrology trading strategy example.
    
    Args:
        market_data: Market data DataFrame
        planetary_data: Planetary data DataFrame
    """
    logger.info("Running single strategy example")
    
    # Create strategy
    strategy = VedicAstrologyStrategy(
        name="Vedic Astrology Strategy",
        description="Trading strategy based on Vedic astrological signals"
    )
    
    # Configure strategy
    strategy.use_yogas = True
    strategy.use_nakshatras = True
    strategy.use_dashas = True
    strategy.min_signal_strength = 0.6
    
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
    run_strategy_comparison_example(market_data, planetary_data)
    run_signal_generator_example(market_data, planetary_data)
    
    logger.info("Example completed successfully")


if __name__ == "__main__":
    main()
