#!/usr/bin/env python
# Setup and Run MLflow UI for Cosmic Market Oracle

"""
Script to set up and run the MLflow UI for the Cosmic Market Oracle project.
This script:
1. Checks for Python 3.10 compatibility
2. Installs required dependencies if needed
3. Sets up the MLflow tracking server
4. Provides instructions for accessing the UI
"""

import os
import sys
import subprocess
import platform
from src.trading.unified_mcts import MCTS, MCTSPredictor
from src.trading.modular_hierarchical_rl import ModularHierarchicalRLAgent, MCTSStrategicPlanner, PPOTacticalExecutor
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    python_version = sys.version_info
    print(f"Current Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version.major != 3 or python_version.minor != 10:
        print("WARNING: This project is optimized for Python 3.10.")
        print(f"You are using Python {python_version.major}.{python_version.minor}")
        
        if python_version.major > 3 or (python_version.major == 3 and python_version.minor > 10):
            print("Your Python version is newer than recommended. Some packages may not be compatible.")
        else:
            print("Your Python version is older than recommended. Some features may not work correctly.")
        
        return False
    return True

def install_dependencies():
    """Install required dependencies for MLflow."""
    print("Installing required dependencies...")
    
    # Get the project root directory
    project_root = Path(__file__).resolve().parents[1]
    requirements_file = project_root / "requirements-hyperopt.txt"
    
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(requirements_file)], check=True)
        print("Dependencies installed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        return False

def setup_mlflow_directory():
    """Set up MLflow directory structure."""
    project_root = Path(__file__).resolve().parents[1]
    mlflow_dir = project_root / "mlruns"
    
    if not mlflow_dir.exists():
        print(f"Creating MLflow directory at {mlflow_dir}")
        mlflow_dir.mkdir(parents=True, exist_ok=True)
    
    return str(mlflow_dir)

def run_mlflow_ui():
    """Run the MLflow UI server."""
    mlflow_dir = setup_mlflow_directory()
    
    print("\n" + "="*80)
    print("Starting MLflow UI server...")
    print(f"MLflow tracking directory: {mlflow_dir}")
    print("="*80 + "\n")
    
    # Set environment variables for MLflow
    os.environ["MLFLOW_TRACKING_URI"] = f"file://{mlflow_dir}"
    
    # Command to run MLflow UI
    cmd = [sys.executable, "-m", "mlflow", "ui", "--host", "localhost", "--port", "5000"]
    
    print("Running command:", " ".join(cmd))
    print("\nMLflow UI will be available at: http://localhost:5000")
    print("\nPress Ctrl+C to stop the server when done.\n")
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nMLflow UI server stopped.")

def main():
    """Main function to set up and run MLflow UI."""
    print("\n" + "="*80)
    print("Cosmic Market Oracle - MLflow UI Setup")
    print("="*80 + "\n")
    
    # Check Python version
    python_compatible = check_python_version()
    if not python_compatible:
        user_input = input("Continue anyway? (y/n): ").strip().lower()
        if user_input != 'y':
            print("Exiting. Please use Python 3.10 for optimal compatibility.")
            sys.exit(1)
    
    # Install dependencies
    deps_installed = install_dependencies()
    if not deps_installed:
        user_input = input("Continue anyway? (y/n): ").strip().lower()
        if user_input != 'y':
            print("Exiting. Please fix dependency installation issues.")
            sys.exit(1)
    
    # Run MLflow UI
    run_mlflow_ui()

if __name__ == "__main__":
    main()
