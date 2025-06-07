#!/usr/bin/env python
# Run Optimization and Training for Cosmic Market Oracle

"""
Script to run hyperparameter optimization and model training for the Cosmic Market Oracle project.
This script:
1. Sets up the environment for Python 3.10
2. Runs the optimization and training process
3. Logs results to MLflow for visualization
"""

import os
import sys
import subprocess
import argparse


from pathlib import Path

def setup_environment():
    """Set up the environment for running optimization."""
    # Get the project root directory
    project_root = Path(__file__).resolve().parents[1]
    
    # Add project root to path
    sys.path.append(str(project_root))
    
    # Set MLflow tracking URI
    os.environ["MLFLOW_TRACKING_URI"] = f"file://{project_root / 'mlruns'}"
    
    # Create necessary directories
    (project_root / "results" / "figures").mkdir(parents=True, exist_ok=True)
    (project_root / "models").mkdir(parents=True, exist_ok=True)
    
    return project_root

def run_optimization(config_path):
    """Run the hyperparameter optimization and training process."""
    project_root = setup_environment()
    
    # Path to the optimization script
    optimize_script = project_root / "scripts" / "train" / "optimize_and_train.py"
    
    # Command to run optimization
    cmd = [
        sys.executable,
        str(optimize_script),
        "--config", config_path
    ]
    
    print("\n" + "="*80)
    print("Running hyperparameter optimization and model training...")
    print(f"Configuration file: {config_path}")
    print("="*80 + "\n")
    
    try:
        subprocess.run(cmd, check=True)
        print("\nOptimization and training completed successfully!")
        print("\nYou can view the results in the MLflow UI at: http://localhost:5000")
        print("To start the MLflow UI, run: python scripts/setup_and_run_mlflow.py")
    except subprocess.CalledProcessError as e:
        print(f"\nError during optimization and training: {e}")
        sys.exit(1)

def main():
    """Main function to run optimization and training."""
    parser = argparse.ArgumentParser(description="Run optimization and training for Cosmic Market Oracle")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/model_config.yaml", 
        help="Path to configuration file"
    )
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("Cosmic Market Oracle - Optimization and Training")
    print("="*80 + "\n")
    
    # Check Python version
    python_version = sys.version_info
    print(f"Current Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version.major != 3 or python_version.minor != 10:
        print("WARNING: This project is optimized for Python 3.10.")
        print(f"You are using Python {python_version.major}.{python_version.minor}")
        
        user_input = input("Continue anyway? (y/n): ").strip().lower()
        if user_input != 'y':
            print("Exiting. Please use Python 3.10 for optimal compatibility.")
            sys.exit(1)
    
    # Run optimization
    run_optimization(args.config)

if __name__ == "__main__":
    main()
