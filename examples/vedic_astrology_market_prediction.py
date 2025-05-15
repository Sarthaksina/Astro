#!/usr/bin/env python
# Cosmic Market Oracle - Vedic Astrology Market Prediction Example

"""
This example demonstrates how to use the enhanced Vedic astrological features
for financial market prediction in the Cosmic Market Oracle project.

It shows how to:
1. Calculate planetary positions and nakshatras
2. Analyze market trends using Vedic principles
3. Detect financial yogas (planetary combinations)
4. Calculate Vimshottari dasha periods for timing
5. Generate features for machine learning models
6. Make predictions using the enhanced Vedic features
"""

import os
import sys
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.astro_engine.planetary_positions import (
    PlanetaryCalculator, analyze_market_trend, analyze_financial_yogas,
    SUN, MOON, MERCURY, VENUS, MARS, JUPITER, SATURN, URANUS, NEPTUNE, PLUTO, RAHU, KETU
)
from src.feature_engineering.astrological_features import AstrologicalFeatureGenerator
from src.data_acquisition.market_data import fetch_historical_data
from src.pipeline.prediction_pipeline import PredictionPipeline


def demonstrate_basic_vedic_calculations():
    """Demonstrate basic Vedic astrological calculations."""
    print("\n=== Basic Vedic Astrological Calculations ===\n")
    
    # Initialize calculator
    calculator = PlanetaryCalculator()
    
    # Current date
    today = datetime.now().strftime("%Y-%m-%d")
    print(f"Calculations for: {today}\n")
    
    # Get planetary positions
    positions = calculator.get_all_planets(today)
    
    # Display positions of key financial planets
    financial_planets = [SUN, MOON, MERCURY, VENUS, MARS, JUPITER, SATURN]
    planet_names = {
        SUN: "Sun", MOON: "Moon", MERCURY: "Mercury", VENUS: "Venus",
        MARS: "Mars", JUPITER: "Jupiter", SATURN: "Saturn"
    }
    
    print("Planetary Positions:")
    for planet in financial_planets:
        sign = int(positions[planet]["longitude"] / 30)
        zodiac_signs = ["Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo", 
                        "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"]
        sign_name = zodiac_signs[sign]
        retrograde = " (R)" if positions[planet]["is_retrograde"] else ""
        print(f"  {planet_names[planet]}: {sign_name} {positions[planet]['longitude'] % 30:.2f}°{retrograde}")
    
    # Get nakshatra details for Moon
    moon_longitude = positions[MOON]["longitude"]
    moon_nakshatra = calculator.get_nakshatra_details(moon_longitude)
    
    print("\nMoon Nakshatra Details:")
    print(f"  Nakshatra: {moon_nakshatra['nakshatra_name']} ({moon_nakshatra['nakshatra']})")
    print(f"  Pada: {moon_nakshatra['pada']}")
    print(f"  Ruler: {moon_nakshatra['ruler']}")
    print(f"  Financial Nature: {moon_nakshatra['financial_nature']}")
    
    # Calculate Vimshottari dasha periods
    dasha_periods = calculator.calculate_vimshottari_dasha(moon_longitude, today)
    
    print("\nCurrent and Upcoming Dasha Periods:")
    for i, (lord, end_date) in enumerate(list(dasha_periods.items())[:3]):
        print(f"  {lord} Dasha until {end_date}")
        if i == 0:
            print(f"    Financial Nature: {'Bullish' if lord in ['Jupiter', 'Venus', 'Mercury'] else 'Bearish' if lord in ['Saturn', 'Mars', 'Ketu'] else 'Volatile' if lord in ['Rahu'] else 'Neutral'}")
    
    # Clean up
    calculator.close()


def analyze_market_with_vedic_astrology(start_date, end_date, symbol="^DJI"):
    """
    Analyze market movements using Vedic astrological principles.
    
    Args:
        start_date: Start date for analysis (YYYY-MM-DD)
        end_date: End date for analysis (YYYY-MM-DD)
        symbol: Market symbol to analyze
    """
    print(f"\n=== Market Analysis with Vedic Astrology ({symbol}) ===\n")
    
    # Initialize calculator
    calculator = PlanetaryCalculator()
    
    # Fetch historical market data
    print(f"Fetching market data for {symbol} from {start_date} to {end_date}...")
    market_data = fetch_historical_data(start_date, end_date, symbol)
    
    # Create date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days
    
    # Initialize results storage
    results = []
    
    # Analyze each date
    print("\nAnalyzing market trends with Vedic astrology...")
    for date in date_range:
        date_str = date.strftime("%Y-%m-%d")
        
        # Skip if market data not available for this date
        if date_str not in market_data.index:
            continue
        
        # Get planetary positions
        positions = calculator.get_all_planets(date_str)
        
        # Analyze market trend
        trend = analyze_market_trend(positions, date_str, calculator)
        
        # Detect financial yogas
        yogas = analyze_financial_yogas(positions, calculator)
        
        # Store results
        results.append({
            'date': date_str,
            'close': market_data.loc[date_str, 'Close'] if date_str in market_data.index else None,
            'primary_trend': trend['primary_trend'],
            'trend_strength': trend['strength'],
            'reversal_probability': trend['reversal_probability'],
            'yoga_count': len(yogas),
            'bullish_yogas': sum(1 for yoga in yogas if yoga['market_impact'] == 'bullish'),
            'bearish_yogas': sum(1 for yoga in yogas if yoga['market_impact'] == 'bearish'),
            'volatile_yogas': sum(1 for yoga in yogas if yoga['market_impact'] == 'volatile'),
        })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    results_df.set_index('date', inplace=True)
    
    # Calculate daily returns
    results_df['daily_return'] = results_df['close'].pct_change() * 100
    
    # Display summary
    print("\nSummary of Vedic Analysis:")
    print(f"  Total days analyzed: {len(results_df)}")
    print(f"  Bullish days: {sum(results_df['primary_trend'] == 'bullish')}")
    print(f"  Bearish days: {sum(results_df['primary_trend'] == 'bearish')}")
    print(f"  Neutral days: {sum(results_df['primary_trend'] == 'neutral')}")
    print(f"  Volatile days: {sum(results_df['primary_trend'] == 'volatile')}")
    
    # Analyze accuracy
    bullish_returns = results_df[results_df['primary_trend'] == 'bullish']['daily_return'].mean()
    bearish_returns = results_df[results_df['primary_trend'] == 'bearish']['daily_return'].mean()
    
    print("\nPrediction Accuracy:")
    print(f"  Average return on bullish days: {bullish_returns:.2f}%")
    print(f"  Average return on bearish days: {bearish_returns:.2f}%")
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Market prices with trend overlay
    plt.subplot(2, 1, 1)
    plt.title(f"{symbol} Price with Vedic Astrological Trend Analysis")
    plt.plot(results_df.index, results_df['close'], label='Close Price')
    
    # Color points based on trend
    bullish_days = results_df[results_df['primary_trend'] == 'bullish']
    bearish_days = results_df[results_df['primary_trend'] == 'bearish']
    volatile_days = results_df[results_df['primary_trend'] == 'volatile']
    
    plt.scatter(bullish_days.index, bullish_days['close'], color='green', label='Bullish', alpha=0.7)
    plt.scatter(bearish_days.index, bearish_days['close'], color='red', label='Bearish', alpha=0.7)
    plt.scatter(volatile_days.index, volatile_days['close'], color='orange', label='Volatile', alpha=0.7)
    
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Trend strength and reversal probability
    plt.subplot(2, 1, 2)
    plt.title("Vedic Astrological Indicators")
    plt.plot(results_df.index, results_df['trend_strength'], label='Trend Strength', color='blue')
    plt.plot(results_df.index, results_df['reversal_probability'], label='Reversal Probability', color='purple')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('vedic_market_analysis.png')
    print("\nAnalysis chart saved as 'vedic_market_analysis.png'")
    
    # Clean up
    calculator.close()
    
    return results_df


def generate_vedic_features_for_ml():
    """Generate Vedic astrological features for machine learning."""
    print("\n=== Generating Vedic Features for Machine Learning ===\n")
    
    # Initialize feature generator
    feature_gen = AstrologicalFeatureGenerator()
    
    # Generate features for a date range
    start_date = "2023-01-01"
    end_date = "2023-01-10"
    dates = pd.date_range(start=start_date, end=end_date)
    
    # Generate features
    print(f"Generating features from {start_date} to {end_date}...")
    features_df = feature_gen.generate_features_for_dates(dates)
    
    # Display feature count
    print(f"\nGenerated {features_df.shape[1]} features for {features_df.shape[0]} days")
    
    # Display sample of Vedic-specific features
    vedic_features = [col for col in features_df.columns if any(x in col for x in 
                     ['nakshatra', 'yoga', 'dasha', 'market_trend', 'bullish', 'bearish'])]
    
    print("\nSample of Vedic Astrological Features:")
    for feature in vedic_features[:10]:  # Show first 10 Vedic features
        print(f"  {feature}")
    
    # Show correlation between features
    if len(vedic_features) >= 5:
        print("\nCorrelation between select Vedic features:")
        correlation = features_df[vedic_features[:5]].corr()
        print(correlation)
    
    return features_df


def run_prediction_with_vedic_features():
    """Run a prediction using the enhanced Vedic features."""
    print("\n=== Running Prediction with Enhanced Vedic Features ===\n")
    
    # Initialize prediction pipeline
    pipeline = PredictionPipeline()
    
    # Define prediction parameters
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")  # 1 year ago
    end_date = datetime.now().strftime("%Y-%m-%d")  # Today
    symbol = "^DJI"  # Dow Jones Industrial Average
    model_name = "default_model"  # Use default model
    horizon = 5  # 5-day prediction horizon
    
    print(f"Running prediction for {symbol} from {start_date} to {end_date}...")
    print(f"Using model: {model_name}")
    print(f"Prediction horizon: {horizon} days")
    
    try:
        # Run prediction pipeline
        predictions = pipeline.run_pipeline(
            start_date=start_date,
            end_date=end_date,
            symbol=symbol,
            model_name=model_name,
            horizon=horizon
        )
        
        # Display predictions
        print("\nPredictions:")
        print(predictions.head())
        
        # Save predictions to CSV
        predictions.to_csv("vedic_enhanced_predictions.csv")
        print("\nPredictions saved to 'vedic_enhanced_predictions.csv'")
        
    except Exception as e:
        print(f"\nError running prediction: {str(e)}")
        print("Note: This example requires a trained model. If no model is available, run the training scripts first.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Demonstrate Vedic Astrology for Market Prediction")
    parser.add_argument("--demo", choices=["basic", "market", "features", "predict", "all"], 
                        default="all", help="Which demonstration to run")
    parser.add_argument("--start_date", default="2023-01-01", help="Start date for analysis (YYYY-MM-DD)")
    parser.add_argument("--end_date", default="2023-12-31", help="End date for analysis (YYYY-MM-DD)")
    parser.add_argument("--symbol", default="^DJI", help="Market symbol to analyze")
    
    args = parser.parse_args()
    
    # Run selected demonstrations
    if args.demo in ["basic", "all"]:
        demonstrate_basic_vedic_calculations()
    
    if args.demo in ["market", "all"]:
        analyze_market_with_vedic_astrology(args.start_date, args.end_date, args.symbol)
    
    if args.demo in ["features", "all"]:
        generate_vedic_features_for_ml()
    
    if args.demo in ["predict", "all"]:
        run_prediction_with_vedic_features()
