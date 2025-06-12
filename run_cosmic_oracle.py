#!/usr/bin/env python
# Cosmic Market Oracle - Main Runner Script

"""
Main entry point for running the Cosmic Market Oracle.

This script provides a command-line interface for running different components
of the Cosmic Market Oracle project, including data acquisition, model training,
hyperparameter optimization, and prediction.
"""

import os
import sys
import argparse
import logging
from src.trading.unified_mcts import MCTS, MCTSPredictor
from src.trading.modular_hierarchical_rl import ModularHierarchicalRLAgent, MCTSStrategicPlanner, PPOTacticalExecutor
from datetime import datetime
from pathlib import Path

# Ensure Python 3.10 compatibility
if sys.version_info.major != 3 or sys.version_info.minor != 10:
    print(f"WARNING: This project is optimized for Python 3.10.")
    print(f"You are using Python {sys.version_info.major}.{sys.version_info.minor}")
    user_input = input("Continue anyway? (y/n): ").strip().lower()
    if user_input != 'y':
        print("Exiting. Please use Python 3.10 for optimal compatibility.")
        sys.exit(1)

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

# Import project modules
from src.utils.logger import get_logger # Changed
from src.utils.config import Config
from src.pipeline.prediction_pipeline import PredictionPipeline

# Configure logging
logger = get_logger("cosmic_oracle") # Changed

def setup_environment():
    """Set up the environment for running the Cosmic Market Oracle."""
    # Set MLflow tracking URI
    os.environ["MLFLOW_TRACKING_URI"] = f"file://{project_root / 'mlruns'}"
    
    # Create necessary directories
    directories = [
        "mlruns",
        "models",
        "results/figures",
        "logs"
    ]
    
    for directory in directories:
        dir_path = project_root / directory
        if not dir_path.exists():
            logger.info(f"Creating directory: {dir_path}")
            dir_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("Environment setup complete")

def run_data_acquisition(args):
    """Run data acquisition process."""
    from src.data_acquisition.market_data import fetch_historical_data, process_market_data
    from src.astro_engine.planetary_positions import PlanetaryCalculator
    
    logger.info(f"Acquiring market data for {args.symbol} from {args.start_date} to {args.end_date}")
    
    # Fetch market data
    market_data = fetch_historical_data(
        start_date=args.start_date,
        end_date=args.end_date,
        symbol=args.symbol
    )
    
    # Process market data
    processed_data = process_market_data(market_data)
    
    # Save market data
    output_path = project_root / "data" / f"{args.symbol}_{args.start_date}_{args.end_date}.csv"
    processed_data.to_csv(output_path)
    
    logger.info(f"Market data saved to {output_path}")
    
    # Fetch planetary data
    planetary_calculator = PlanetaryCalculator()
    dates = processed_data.index.tolist()
    planetary_data = planetary_calculator.get_planetary_positions(dates)
    
    # Save planetary data
    planetary_output_path = project_root / "data" / f"planetary_{args.start_date}_{args.end_date}.csv"
    planetary_data.to_csv(planetary_output_path)
    
    logger.info(f"Planetary data saved to {planetary_output_path}")
    
    return processed_data, planetary_data

def run_optimization(args):
    """Run hyperparameter optimization."""
    logger.info("Running hyperparameter optimization")
    
    # Import here to avoid circular imports
    from scripts.train.optimize_and_train import main as optimize_main
    
    # Set up arguments for optimization
    sys.argv = [
        "optimize_and_train.py",
        "--config", args.config
    ]
    
    # Run optimization
    optimize_main()
    
    logger.info("Hyperparameter optimization complete")
    logger.info("You can view the results in the MLflow UI at: http://localhost:5000")

def run_prediction(args):
    """Run prediction pipeline."""
    logger.info("Running prediction pipeline")
    
    # Create prediction pipeline
    pipeline = PredictionPipeline(config_path=args.config)
    
    # Run pipeline
    predictions = pipeline.run_pipeline(
        start_date=args.start_date,
        end_date=args.end_date,
        symbol=args.symbol,
        model_name=args.model,
        horizon=args.horizon
    )
    
    # Save predictions
    output_path = project_root / "results" / f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    predictions.to_csv(output_path, index=False)
    
    logger.info(f"Predictions saved to {output_path}")
    
    return predictions

def run_mlflow_ui():
    """Run MLflow UI."""
    import subprocess
    
    logger.info("Starting MLflow UI")
    
    # Command to run MLflow UI
    cmd = [sys.executable, "-m", "mlflow", "ui", "--host", "localhost", "--port", "5000"]
    
    # Run MLflow UI in a subprocess
    subprocess.Popen(cmd)
    
    logger.info("MLflow UI started at: http://localhost:5000")

def main():
    """Main entry point for the Cosmic Market Oracle."""
    parser = argparse.ArgumentParser(description="Cosmic Market Oracle")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Data acquisition command
    data_parser = subparsers.add_parser("data", help="Run data acquisition")
    data_parser.add_argument("--start_date", required=True, help="Start date (YYYY-MM-DD)")
    data_parser.add_argument("--end_date", required=True, help="End date (YYYY-MM-DD)")
    data_parser.add_argument("--symbol", default="^DJI", help="Market symbol")
    
    # Optimization command
    optimize_parser = subparsers.add_parser("optimize", help="Run hyperparameter optimization")
    optimize_parser.add_argument("--config", default="config/model_config.yaml", help="Path to config file")
    
    # Prediction command
    predict_parser = subparsers.add_parser("predict", help="Run prediction pipeline")
    predict_parser.add_argument("--start_date", required=True, help="Start date (YYYY-MM-DD)")
    predict_parser.add_argument("--end_date", required=True, help="End date (YYYY-MM-DD)")
    predict_parser.add_argument("--symbol", default="^DJI", help="Market symbol")
    predict_parser.add_argument("--model", default="best_model", help="Model name")
    predict_parser.add_argument("--horizon", type=int, default=1, help="Prediction horizon in days")
    predict_parser.add_argument("--config", default="config/prediction_config.yaml", help="Path to config file")
    
    # MLflow UI command
    mlflow_parser = subparsers.add_parser("mlflow", help="Run MLflow UI")
    
    args = parser.parse_args()
    
    # Set up environment
    setup_environment()
    
    # Run command
    if args.command == "data":
        run_data_acquisition(args)
    elif args.command == "optimize":
        run_optimization(args)
    elif args.command == "predict":
        run_prediction(args)
    elif args.command == "mlflow":
        run_mlflow_ui()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
