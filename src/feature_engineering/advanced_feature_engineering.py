"""
Advanced Astrological Feature Engineering Module for the Cosmic Market Oracle.

This module integrates all the advanced astrological feature engineering components,
providing a unified interface for generating, discovering, and validating astrological
features for financial market prediction.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Set, Optional, Any, Callable, Union
from datetime import datetime, timedelta
# import logging # Removed
import os
from pathlib import Path
from src.utils.logger import get_logger # Added
import json

from .astrological_features import AstrologicalFeatureGenerator
from .feature_definitions import FeatureDefinition
from .feature_generator import FeatureGenerator
from .genetic_feature_discovery import GeneticFeatureDiscovery, GeneticIndividual
from .expert_validation import ExpertValidationFramework, ExpertFeedback, ExpertValidationUI
from .pattern_detection import PatternDetectionManager
from .composite_indicators import CompositeIndicatorManager
from .feature_importance import FeatureImportanceAnalyzer
# Define the CyclicalFeatureManager class which is still needed
class CyclicalFeatureManager:
    """Manager for cyclical features."""
    
    def __init__(self, calculator):
        self.calculator = calculator
        self.planetary_cycle_extractor = None  # Will be initialized later

# Configure logging
# logging.basicConfig( # Removed
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
logger = get_logger(__name__) # Changed


class AdvancedFeatureEngineeringManager:
    """Manager for advanced astrological feature engineering."""
    
    def __init__(self, calculator, financial_data_provider=None,
                output_dir: str = "data/feature_engineering"):
        """
        Initialize the advanced feature engineering manager.
        
        Args:
            calculator: Astronomical calculator instance
            financial_data_provider: Provider of financial data (optional)
            output_dir: Directory for output files
        """
        self.calculator = calculator
        self.financial_data_provider = financial_data_provider
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize components
        self._initialize_components()
        
        # Initialize feature catalog
        self.feature_catalog = {}
        
        # Initialize feature metadata
        self.feature_metadata = {}
    
    def _initialize_components(self):
        """Initialize all feature engineering components."""
        # Initialize pattern detection manager
        self.pattern_detector = PatternDetectionManager(self.calculator)
        
        # Initialize cyclical feature manager
        self.cycle_extractor = CyclicalFeatureManager(self.calculator)
        
        # Initialize composite indicator manager
        self.indicator_manager = CompositeIndicatorManager(
            self.calculator, self.pattern_detector, self.cycle_extractor.planetary_cycle_extractor
        )
        
        # Initialize feature generator
        self.feature_generator = FeatureGenerator(
            self.calculator, self.pattern_detector, 
            self.cycle_extractor.planetary_cycle_extractor,
            self.indicator_manager
        )
        
        # Initialize genetic feature discovery
        self.genetic_discovery = GeneticFeatureDiscovery(
            self.feature_generator, self.financial_data_provider
        )
        
        # Initialize expert validation framework
        self.validation_framework = ExpertValidationFramework(
            self.feature_generator, os.path.join(self.output_dir, "expert_validation")
        )
        
        # Initialize validation UI
        self.validation_ui = ExpertValidationUI(self.validation_framework)
        
        # Initialize feature importance analyzer
        self.importance_analyzer = FeatureImportanceAnalyzer()
        
        logger.info("Initialized all feature engineering components")
    
    def generate_base_features(self, date: datetime) -> Dict[str, float]:
        """
        Generate base astrological features for a given date.
        
        Args:
            date: Date to analyze
            
        Returns:
            Dictionary of base features
        """
        return self.feature_generator.generate_base_features(date)
    
    def detect_patterns(self, date: datetime) -> Dict[str, List[Any]]:
        """
        Detect astrological patterns for a given date.
        
        Args:
            date: Date to analyze
            
        Returns:
            Dictionary of detected patterns
        """
        return self.pattern_detector.detect_all_patterns(date)
    
    def extract_cyclical_features(self, date: datetime) -> Dict[str, Dict[str, float]]:
        """
        Extract cyclical features for a given date.
        
        Args:
            date: Date to analyze
            
        Returns:
            Dictionary of cyclical features
        """
        return self.cycle_extractor.extract_all_features(date)
    
    def calculate_composite_indicators(self, date: datetime) -> Dict[str, float]:
        """
        Calculate composite indicators for a given date.
        
        Args:
            date: Date to analyze
            
        Returns:
            Dictionary of composite indicators
        """
        return self.indicator_manager.calculate_all_indicators(date)
    
    def discover_features(self, start_date: datetime, end_date: datetime, 
                         target_symbol: str = "SPY", population_size: int = 50, 
                         generations: int = 20) -> List[FeatureDefinition]:
        """
        Discover optimal features using genetic programming.
        
        Args:
            start_date: Start date for evaluation
            end_date: End date for evaluation
            target_symbol: Financial symbol to target
            population_size: Size of the genetic population
            generations: Number of generations to evolve
            
        Returns:
            List of discovered feature definitions
        """
        # Configure genetic discovery
        self.genetic_discovery.population_size = population_size
        self.genetic_discovery.generations = generations
        
        # Evolve to discover features
        self.genetic_discovery.evolve(start_date, end_date, target_symbol)
        
        # Get best features
        best_features = self.genetic_discovery.get_best_features()
        
        # Add to feature catalog
        for feature_def in best_features:
            self.feature_catalog[feature_def.name] = feature_def
            
            # Add metadata
            self.feature_metadata[feature_def.name] = {
                "discovery_method": "genetic_programming",
                "discovery_date": datetime.now().isoformat(),
                "target_symbol": target_symbol,
                "evaluation_period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat()
                }
            }
        
        return best_features
    
    def generate_features_for_date_range(self, start_date: datetime, end_date: datetime, 
                                       feature_defs: List[FeatureDefinition] = None) -> pd.DataFrame:
        """
        Generate features for a date range.
        
        Args:
            start_date: Start date
            end_date: End date
            feature_defs: List of feature definitions to generate (optional)
            
        Returns:
            DataFrame with generated features
        """
        # Use all features in catalog if not specified
        if feature_defs is None:
            feature_defs = list(self.feature_catalog.values())
        
        # Generate dates
        current_date = start_date
        dates = []
        
        while current_date <= end_date:
            dates.append(current_date)
            current_date += timedelta(days=1)
        
        # Initialize results
        results = {
            "date": dates
        }
        
        # Get base features for all dates
        base_features = {}
        for date in dates:
            base_features[date] = self.feature_generator.generate_base_features(date)
        
        # Generate features for each date
        for feature_def in feature_defs:
            feature_name = feature_def.name
            results[feature_name] = []
            
            for date in dates:
                value = self.feature_generator.calculate_feature_value(
                    feature_def, date, base_features[date]
                )
                results[feature_name].append(value)
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        df.set_index("date", inplace=True)
        
        return df
    
    def evaluate_feature_importance(self, features_df: pd.DataFrame, 
                                  target_df: pd.DataFrame, 
                                  target_column: str = "close",
                                  returns_periods: List[int] = [1, 5, 10, 20],
                                  method: str = "random_forest") -> Dict[str, pd.DataFrame]:
        """
        Evaluate feature importance for predicting financial returns.
        
        Args:
            features_df: DataFrame with features
            target_df: DataFrame with target data
            target_column: Column to use as target
            returns_periods: List of periods for calculating returns
            method: Method for evaluating importance
            
        Returns:
            Dictionary of importance DataFrames by return period
        """
        # Align dates
        aligned_df = pd.merge(
            features_df, target_df[[target_column]], 
            left_index=True, right_index=True,
            how="inner"
        )
        
        # Calculate returns for different periods
        for period in returns_periods:
            aligned_df[f"return_{period}d"] = aligned_df[target_column].pct_change(period).shift(-period)
        
        # Drop rows with NaN returns
        aligned_df.dropna(inplace=True)
        
        # Initialize results
        importance_results = {}
        
        # Evaluate importance for each return period
        for period in returns_periods:
            # Prepare data
            X = aligned_df.drop([target_column] + [f"return_{p}d" for p in returns_periods], axis=1)
            y = aligned_df[f"return_{period}d"]
            
            # Train a model
            if method == "random_forest":
                from sklearn.ensemble import RandomForestRegressor
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X, y)
                
                # Get importance
                importance_df = self.importance_analyzer.get_model_feature_importance(model, X)
            
            elif method == "permutation":
                from sklearn.ensemble import RandomForestRegressor
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X, y)
                
                # Get permutation importance
                importance_df = self.importance_analyzer.get_permutation_importance(model, X, y)
            
            elif method == "shap":
                from sklearn.ensemble import RandomForestRegressor
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X, y)
                
                # Get SHAP importance
                importance_df = self.importance_analyzer.get_shap_importance(model, X)
            
            # Store results
            importance_results[f"return_{period}d"] = importance_df
        
        return importance_results
    
    def generate_feature_report(self, feature_name: str, 
                              start_date: datetime, end_date: datetime,
                              target_symbol: str = "SPY") -> Dict[str, Any]:
        """
        Generate a comprehensive report for a feature.
        
        Args:
            feature_name: Feature name
            start_date: Start date for evaluation
            end_date: End date for evaluation
            target_symbol: Financial symbol to target
            
        Returns:
            Feature report
        """
        # Check if feature exists
        if feature_name not in self.feature_catalog:
            return {
                "feature_name": feature_name,
                "status": "not_found",
                "message": f"Feature {feature_name} not found in catalog"
            }
        
        feature_def = self.feature_catalog[feature_name]
        
        # Get feature metadata
        metadata = self.feature_metadata.get(feature_name, {})
        
        # Generate feature values for date range
        feature_df = self.generate_features_for_date_range(
            start_date, end_date, [feature_def]
        )
        
        # Get financial data if available
        financial_data = None
        market_correlation = None
        
        if self.financial_data_provider:
            financial_data = self.financial_data_provider.get_historical_data(
                target_symbol, start_date, end_date
            )
            
            # Calculate correlation with returns
            if financial_data is not None and len(financial_data) > 0:
                # Align dates
                aligned_df = pd.merge(
                    feature_df, financial_data[["close"]], 
                    left_index=True, right_index=True,
                    how="inner"
                )
                
                # Calculate returns
                aligned_df["return_1d"] = aligned_df["close"].pct_change()
                
                # Calculate correlation
                correlation = aligned_df[[feature_name, "return_1d"]].corr().iloc[0, 1]
                
                market_correlation = correlation
        
        # Get validation report if available
        validation_report = self.validation_framework.generate_validation_report(feature_name)
        
        # Create report
        report = {
            "feature_name": feature_name,
            "description": feature_def.description,
            "feature_type": feature_def.feature_type,
            "parameters": feature_def.parameters,
            "metadata": metadata,
            "statistics": {
                "mean": feature_df[feature_name].mean(),
                "std": feature_df[feature_name].std(),
                "min": feature_df[feature_name].min(),
                "max": feature_df[feature_name].max(),
                "market_correlation": market_correlation
            },
            "validation": validation_report
        }
        
        return report
    
    def export_feature_catalog(self, output_file: str = None):
        """
        Export the feature catalog to a JSON file.
        
        Args:
            output_file: Output file path (optional)
        """
        if output_file is None:
            output_file = os.path.join(self.output_dir, "feature_catalog.json")
        
        # Create export data
        export_data = {
            "feature_count": len(self.feature_catalog),
            "export_date": datetime.now().isoformat(),
            "features": {}
        }
        
        # Add feature definitions
        for feature_name, feature_def in self.feature_catalog.items():
            export_data["features"][feature_name] = {
                "name": feature_def.name,
                "description": feature_def.description,
                "feature_type": feature_def.feature_type,
                "parameters": feature_def.parameters,
                "generation": feature_def.generation,
                "parent_features": feature_def.parent_features,
                "metadata": self.feature_metadata.get(feature_name, {})
            }
        
        # Save to file
        try:
            with open(output_file, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Exported feature catalog to {output_file}")
        except Exception as e:
            logger.error(f"Error exporting feature catalog: {e}")
    
    def import_feature_catalog(self, input_file: str):
        """
        Import a feature catalog from a JSON file.
        
        Args:
            input_file: Input file path
        """
        try:
            with open(input_file, 'r') as f:
                import_data = json.load(f)
            
            # Import feature definitions
            for feature_name, feature_data in import_data["features"].items():
                # Create feature definition
                feature_def = FeatureDefinition(
                    name=feature_data["name"],
                    description=feature_data["description"],
                    feature_type=feature_data["feature_type"],
                    parameters=feature_data["parameters"],
                    generation=feature_data.get("generation", 0),
                    parent_features=feature_data.get("parent_features")
                )
                
                # Add to catalog
                self.feature_catalog[feature_name] = feature_def
                
                # Add metadata
                if "metadata" in feature_data:
                    self.feature_metadata[feature_name] = feature_data["metadata"]
            
            logger.info(f"Imported {len(import_data['features'])} features from {input_file}")
        except Exception as e:
            logger.error(f"Error importing feature catalog: {e}")
    
    def generate_feature_engineering_dashboard(self, output_file: str = None):
        """
        Generate a dashboard for feature engineering.
        
        Args:
            output_file: Output file path (optional)
        """
        if output_file is None:
            output_file = os.path.join(self.output_dir, "feature_engineering_dashboard.html")
        
        # Get validated features
        validated_features = self.validation_framework.get_all_validated_features()
        
        # Create HTML dashboard
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Advanced Astrological Feature Engineering Dashboard</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .section { margin-bottom: 30px; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                tr:nth-child(even) { background-color: #f9f9f9; }
                .high { color: green; }
                .medium { color: orange; }
                .low { color: red; }
            </style>
        </head>
        <body>
            <h1>Advanced Astrological Feature Engineering Dashboard</h1>
            
            <div class="section">
                <h2>Feature Catalog Summary</h2>
                <p>Total features in catalog: """ + str(len(self.feature_catalog)) + """</p>
                <p>Validated features: """ + str(len(validated_features)) + """</p>
            </div>
            
            <div class="section">
                <h2>Feature Categories</h2>
                <table>
                    <tr>
                        <th>Category</th>
                        <th>Count</th>
                    </tr>
        """
        
        # Count features by type
        type_counts = {}
        for feature_def in self.feature_catalog.values():
            feature_type = feature_def.feature_type
            type_counts[feature_type] = type_counts.get(feature_type, 0) + 1
        
        # Add rows for each type
        for feature_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
            html += f"""
                <tr>
                    <td>{feature_type}</td>
                    <td>{count}</td>
                </tr>
            """
        
        html += """
                </table>
            </div>
            
            <div class="section">
                <h2>Top Validated Features</h2>
                <table>
                    <tr>
                        <th>Feature Name</th>
                        <th>Relevance Score</th>
                        <th>Reliability Score</th>
                        <th>Market Impact</th>
                        <th>Time Frames</th>
                    </tr>
        """
        
        # Sort validated features by average score
        sorted_validated = sorted(
            validated_features.items(),
            key=lambda x: (x[1].get("relevance_score", 0) + x[1].get("reliability_score", 0)) / 2,
            reverse=True
        )
        
        # Add rows for top 10 validated features
        for name, data in sorted_validated[:10]:
            # Determine validation level
            avg_score = (data.get("relevance_score", 0) + data.get("reliability_score", 0)) / 2
            validation_class = "high" if avg_score >= 7.5 else ("medium" if avg_score >= 5.0 else "low")
            
            html += f"""
                <tr class="{validation_class}">
                    <td>{name}</td>
                    <td>{data.get("relevance_score", 0):.1f}</td>
                    <td>{data.get("reliability_score", 0):.1f}</td>
                    <td>{data.get("market_impact", "unknown")}</td>
                    <td>{", ".join(data.get("time_frames", []))}</td>
                </tr>
            """
        
        html += """
                </table>
            </div>
        </body>
        </html>
        """
        
        # Save to file
        try:
            with open(output_file, 'w') as f:
                f.write(html)
            
            logger.info(f"Generated feature engineering dashboard saved to {output_file}")
        except Exception as e:
            logger.error(f"Error saving feature engineering dashboard: {e}")


# Example usage
if __name__ == "__main__":
    from src.astro_engine.astronomical_calculator import AstronomicalCalculator
    from src.data_acquisition.financial_data import FinancialDataProvider
    
    # Initialize calculator
    calculator = AstronomicalCalculator()
    
    # Initialize financial data provider
    financial_provider = FinancialDataProvider()
    
    # Initialize advanced feature engineering manager
    manager = AdvancedFeatureEngineeringManager(calculator, financial_provider)
    
    # Generate base features for current date
    current_date = datetime.now()
    base_features = manager.generate_base_features(current_date)
    
    print(f"Generated {len(base_features)} base features for {current_date.strftime('%Y-%m-%d')}")
    
    # Detect patterns
    patterns = manager.detect_patterns(current_date)
    
    print(f"Detected patterns:")
    for detector_name, detector_patterns in patterns.items():
        print(f"  {detector_name}: {len(detector_patterns)} patterns")
    
    # Extract cyclical features
    cyclical_features = manager.extract_cyclical_features(current_date)
    
    print(f"Extracted cyclical features:")
    for extractor_name, features in cyclical_features.items():
        print(f"  {extractor_name}: {len(features)} features")
    
    # Calculate composite indicators
    indicators = manager.calculate_composite_indicators(current_date)
    
    print(f"Calculated {len(indicators)} composite indicators")
    
    # Define date range for feature discovery
    end_date = current_date
    start_date = end_date - timedelta(days=365)
    
    # Discover features (commented out to avoid long execution time)
    # best_features = manager.discover_features(start_date, end_date, "SPY", 30, 5)
    # print(f"Discovered {len(best_features)} features")
    
    # Export feature catalog
    manager.export_feature_catalog()
    
    # Generate dashboard
    manager.generate_feature_engineering_dashboard()
