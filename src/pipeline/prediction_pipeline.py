#!/usr/bin/env python
# Cosmic Market Oracle - Prediction Pipeline

"""
Prediction Pipeline for the Cosmic Market Oracle.

This module orchestrates the entire prediction process, from data acquisition to model inference,
creating a seamless pipeline for generating market predictions based on astrological data.
"""

import os
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import mlflow

from src.data_acquisition.market_data import fetch_historical_data, process_market_data
from src.astro_engine.planetary_positions import PlanetaryCalculator, analyze_market_trend, analyze_financial_yogas
from src.data_processing.integrator import integrate_market_and_planetary_data
from src.feature_engineering.feature_generator import FeatureGenerator
from src.feature_engineering.astrological_features import AstrologicalFeatureGenerator
from src.feature_engineering.market_features import MarketFeatureGenerator
from src.models.model_factory import ModelFactory
from src.utils.config import Config
from src.utils.logger import setup_logger

# Configure logging
logger = setup_logger("prediction_pipeline")


class PredictionPipeline:
    """
    Orchestrates the entire prediction process for the Cosmic Market Oracle.
    
    This class manages the flow of data from acquisition to prediction, integrating
    market data with astrological data, generating features, and using trained models
    to make market predictions.
    """
    
    def __init__(self, config_path: str = "config/prediction_config.yaml"):
        """
        Initialize the prediction pipeline.
        
        Args:
            config_path: Path to the configuration file.
        """
        self.config = Config(config_path)
        self.planetary_calculator = PlanetaryCalculator()
        self.astro_feature_generator = AstrologicalFeatureGenerator()
        self.market_feature_generator = MarketFeatureGenerator()
        self.feature_generator = FeatureGenerator()
        
        # Load the model factory with the specified model configuration
        self.model_factory = ModelFactory(self.config.get("model"))
        
        # Set up MLflow tracking
        mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(mlflow_uri)
        
        # Initialize model cache
        self._model_cache = {}
    
    def get_model(self, model_name: str):
        """
        Get a model from the cache or load it if not already loaded.
        
        Args:
            model_name: Name of the model to load.
            
        Returns:
            The loaded model.
        """
        if model_name not in self._model_cache:
            model_path = os.path.join(self.config.get("model_dir"), f"{model_name}.pt")
            model = self.model_factory.load_model(model_path)
            self._model_cache[model_name] = model
        
        return self._model_cache[model_name]
    
    def acquire_data(self, start_date: str, end_date: str, symbol: str = "^DJI") -> pd.DataFrame:
        """
        Acquire market data for the specified period.
        
        Args:
            start_date: Start date for data acquisition (YYYY-MM-DD).
            end_date: End date for data acquisition (YYYY-MM-DD).
            symbol: Market symbol to fetch data for.
            
        Returns:
            DataFrame containing market data.
        """
        logger.info(f"Acquiring market data for {symbol} from {start_date} to {end_date}")
        
        # Fetch historical market data
        market_data = fetch_historical_data(
            start_date=start_date,
            end_date=end_date,
            symbol=symbol,
            source=self.config.get("data_source", "yahoo")
        )
        
        # Process the market data
        processed_data = process_market_data(market_data)
        
        logger.info(f"Acquired {len(processed_data)} days of market data")
        return processed_data
    
    def get_planetary_data(self, dates: List[datetime]) -> pd.DataFrame:
        """
        Get planetary positions and advanced Vedic astrological calculations for the specified dates.
        
        Args:
            dates: List of dates to get planetary positions for.
            
        Returns:
            DataFrame containing planetary positions and Vedic astrological insights.
        """
        logger.info(f"Calculating planetary positions for {len(dates)} dates")
        
        # Create a DataFrame to store planetary data
        planetary_data = pd.DataFrame(index=dates)
        
        # Get planetary positions and advanced Vedic calculations for each date
        for date in dates:
            # Get basic planetary positions
            positions = self.planetary_calculator.get_all_planets(date)
            
            # Flatten the positions dictionary for storage in DataFrame
            for planet, position in positions.items():
                for key, value in position.items():
                    column_name = f"{planet}_{key}"
                    planetary_data.loc[date, column_name] = value
            
            # Add enhanced Vedic astrological calculations
            
            # 1. Market trend analysis based on Vedic principles
            market_trend = analyze_market_trend(positions, date, self.planetary_calculator)
            for key, value in market_trend.items():
                if isinstance(value, list):
                    # For list values like key_factors, store as string
                    planetary_data.loc[date, f"market_trend_{key}"] = ",".join(value)
                else:
                    planetary_data.loc[date, f"market_trend_{key}"] = value
            
            # 2. Financial yogas (planetary combinations)
            yogas = analyze_financial_yogas(positions, self.planetary_calculator)
            # Store count of different yoga types
            bullish_yogas = sum(1 for yoga in yogas if yoga["market_impact"] == "bullish")
            bearish_yogas = sum(1 for yoga in yogas if yoga["market_impact"] == "bearish")
            volatile_yogas = sum(1 for yoga in yogas if yoga["market_impact"] == "volatile")
            
            planetary_data.loc[date, "bullish_yoga_count"] = bullish_yogas
            planetary_data.loc[date, "bearish_yoga_count"] = bearish_yogas
            planetary_data.loc[date, "volatile_yoga_count"] = volatile_yogas
            
            # Store yoga names and strengths
            yoga_names = [yoga["name"] for yoga in yogas]
            yoga_strengths = [yoga["strength"] for yoga in yogas]
            
            planetary_data.loc[date, "yoga_names"] = ",".join(yoga_names) if yoga_names else ""
            planetary_data.loc[date, "yoga_strengths"] = ",".join(map(str, yoga_strengths)) if yoga_strengths else ""
            
            # 3. Nakshatra details for key planets (Moon and Sun)
            if 1 in positions:  # Moon
                moon_nakshatra = self.planetary_calculator.get_nakshatra_details(positions[1]["longitude"])
                for key, value in moon_nakshatra.items():
                    planetary_data.loc[date, f"moon_nakshatra_{key}"] = value
            
            if 0 in positions:  # Sun
                sun_nakshatra = self.planetary_calculator.get_nakshatra_details(positions[0]["longitude"])
                for key, value in sun_nakshatra.items():
                    planetary_data.loc[date, f"sun_nakshatra_{key}"] = value
            
            # 4. Vimshottari Dasha periods (using Moon position)
            if 1 in positions:  # Moon
                moon_longitude = positions[1]["longitude"]
                dasha_periods = self.planetary_calculator.calculate_vimshottari_dasha(moon_longitude, date)
                if dasha_periods:
                    # Get current dasha lord (first key in the ordered dict)
                    current_dasha_lord = list(dasha_periods.keys())[0]
                    planetary_data.loc[date, "current_dasha_lord"] = current_dasha_lord
                    
                    # Store dasha end date
                    dasha_end = list(dasha_periods.values())[0]
                    planetary_data.loc[date, "dasha_end_date"] = dasha_end
            
            # 5. Divisional charts (Varga) for financial planets
            financial_planets = [4, 5, 2, 6]  # Jupiter, Venus, Mercury, Saturn
            varga_divisions = [9, 10]  # D9 (Navamsa) and D10 (Dasamsa) are most relevant for finance
            
            for planet_id in financial_planets:
                if planet_id in positions:
                    longitude = positions[planet_id]["longitude"]
                    for division in varga_divisions:
                        varga_longitude = self.planetary_calculator.calculate_divisional_chart(longitude, division)
                        planetary_data.loc[date, f"planet_{planet_id}_d{division}"] = varga_longitude
        
        logger.info(f"Calculated planetary positions with {planetary_data.shape[1]} features")
        return planetary_data
    
    def integrate_data(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """
        Integrate market data with planetary data.
        
        Args:
            market_data: DataFrame containing market data.
            
        Returns:
            DataFrame containing integrated market and planetary data.
        """
        logger.info("Integrating market and planetary data")
        
        # Get planetary data for the dates in market_data
        planetary_data = self.get_planetary_data(market_data.index.tolist())
        
        # Integrate market and planetary data
        integrated_data = integrate_market_and_planetary_data(market_data, planetary_data)
        
        logger.info(f"Integrated data has {integrated_data.shape[1]} features")
        return integrated_data
    
    def generate_features(self, integrated_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate features for prediction.
        
        Args:
            integrated_data: DataFrame containing integrated market and planetary data.
            
        Returns:
            DataFrame containing features for prediction.
        """
        logger.info("Generating features for prediction")
        
        # Generate astrological features
        astro_features = self.astro_feature_generator.generate_features(integrated_data)
        
        # Generate market features
        market_features = self.market_feature_generator.generate_features(integrated_data)
        
        # Combine features
        combined_data = pd.concat([integrated_data, astro_features, market_features], axis=1)
        
        # Generate interaction features
        features = self.feature_generator.generate_features(combined_data)
        
        logger.info(f"Generated {features.shape[1]} features for prediction")
        return features
    
    def make_prediction(self, features: pd.DataFrame, model_name: str = "default_model", 
                       horizon: int = 1) -> pd.DataFrame:
        """
        Make predictions using the specified model.
        
        Args:
            features: DataFrame containing features for prediction.
            model_name: Name of the model to use for prediction.
            horizon: Prediction horizon in days.
            
        Returns:
            DataFrame containing predictions.
        """
        logger.info(f"Making predictions with model {model_name} for horizon {horizon}")
        
        # Get the model
        model = self.get_model(model_name)
        
        # Prepare features for prediction
        X = self.prepare_features_for_prediction(features)
        
        # Make predictions
        predictions = model.predict(X)
        
        # Create a DataFrame with predictions
        prediction_dates = features.index[-len(predictions):]
        target_dates = [date + timedelta(days=horizon) for date in prediction_dates]
        
        predictions_df = pd.DataFrame({
            'prediction_date': prediction_dates,
            'target_date': target_dates,
            'predicted_return': predictions.flatten(),
            'confidence': model.get_confidence(X) if hasattr(model, 'get_confidence') else None
        })
        
        logger.info(f"Made {len(predictions)} predictions")
        return predictions_df
    
    def prepare_features_for_prediction(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for prediction by handling missing values, scaling, etc.
        
        Args:
            features: DataFrame containing features for prediction.
            
        Returns:
            DataFrame containing prepared features.
        """
        # Handle missing values
        features = features.fillna(0)
        
        # Select only the features used by the model
        model_features = self.config.get("model_features", [])
        if model_features:
            features = features[model_features]
        
        # Apply any necessary scaling or transformations
        # (This would depend on how the model was trained)
        
        return features
    
    def run_pipeline(self, start_date: str, end_date: str, symbol: str = "^DJI",
                    model_name: str = "default_model", horizon: int = 1) -> pd.DataFrame:
        """
        Run the complete prediction pipeline.
        
        Args:
            start_date: Start date for data acquisition (YYYY-MM-DD).
            end_date: End date for data acquisition (YYYY-MM-DD).
            symbol: Market symbol to fetch data for.
            model_name: Name of the model to use for prediction.
            horizon: Prediction horizon in days.
            
        Returns:
            DataFrame containing predictions.
        """
        logger.info(f"Running prediction pipeline for {symbol} from {start_date} to {end_date}")
        
        # Acquire market data
        market_data = self.acquire_data(start_date, end_date, symbol)
        
        # Integrate market and planetary data
        integrated_data = self.integrate_data(market_data)
        
        # Generate features
        features = self.generate_features(integrated_data)
        
        # Make predictions
        predictions = self.make_prediction(features, model_name, horizon)
        
        logger.info("Prediction pipeline completed successfully")
        return predictions


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run the Cosmic Market Oracle prediction pipeline")
    parser.add_argument("--start_date", required=True, help="Start date for data acquisition (YYYY-MM-DD)")
    parser.add_argument("--end_date", required=True, help="End date for data acquisition (YYYY-MM-DD)")
    parser.add_argument("--symbol", default="^DJI", help="Market symbol to fetch data for")
    parser.add_argument("--model", default="default_model", help="Name of the model to use for prediction")
    parser.add_argument("--horizon", type=int, default=1, help="Prediction horizon in days")
    parser.add_argument("--output", default="predictions.csv", help="Output file for predictions")
    
    args = parser.parse_args()
    
    # Run the pipeline
    pipeline = PredictionPipeline()
    predictions = pipeline.run_pipeline(
        start_date=args.start_date,
        end_date=args.end_date,
        symbol=args.symbol,
        model_name=args.model,
        horizon=args.horizon
    )
    
    # Save predictions to file
    predictions.to_csv(args.output, index=False)
    print(f"Predictions saved to {args.output}")
