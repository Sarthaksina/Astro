#!/usr/bin/env python
# Setup Python 3.10 Environment for Cosmic Market Oracle

"""
Script to set up a Python 3.10 environment for the Cosmic Market Oracle project.
This script:
1. Verifies Python 3.10 is being used
2. Installs all required dependencies
3. Sets up the project structure
4. Provides instructions for running the project
"""

import os
import sys
import subprocess
import platform
from src.trading.unified_mcts import MCTS, MCTSPredictor
from src.trading.modular_hierarchical_rl import ModularHierarchicalRLAgent, MCTSStrategicPlanner, PPOTacticalExecutor
from pathlib import Path

def check_python_version():
    """Check if Python version is 3.10."""
    python_version = sys.version_info
    print(f"Current Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version.major != 3 or python_version.minor != 10:
        print("ERROR: This script must be run with Python 3.10.")
        print(f"You are using Python {python_version.major}.{python_version.minor}")
        print("\nPlease run this script with Python 3.10:")
        print("py -3.10 scripts/setup_python310.py")
        sys.exit(1)
    
    print("[OK] Using Python 3.10")
    return True

def setup_project_structure():
    """Set up the project directory structure."""
    project_root = Path(__file__).resolve().parents[1]
    
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
            print(f"Creating directory: {dir_path}")
            dir_path.mkdir(parents=True, exist_ok=True)
    
    print("[OK] Project structure set up")
    return project_root

def install_dependencies():
    """Install required dependencies for the project."""
    project_root = Path(__file__).resolve().parents[1]
    requirements_file = project_root / "requirements-hyperopt.txt"
    
    print(f"Installing dependencies from {requirements_file}...")
    
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(requirements_file)], check=True)
        print("[OK] Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        return False

def create_run_scripts():
    """Create batch scripts to run the project with Python 3.10."""
    project_root = Path(__file__).resolve().parents[1]
    
    # Create batch script to run MLflow UI
    mlflow_bat = project_root / "run_mlflow_ui.bat"
    with open(mlflow_bat, "w") as f:
        f.write("@echo off\n")
        f.write("echo Starting MLflow UI with Python 3.10...\n")
        f.write("py -3.10 scripts/setup_and_run_mlflow.py\n")
    
    # Create batch script to run optimization
    optimize_bat = project_root / "run_optimization.bat"
    with open(optimize_bat, "w") as f:
        f.write("@echo off\n")
        f.write("echo Running optimization with Python 3.10...\n")
        f.write("py -3.10 scripts/run_optimization.py %*\n")
    
    print("[OK] Created run scripts:")
    print(f"  - {mlflow_bat}")
    print(f"  - {optimize_bat}")

def main():
    """Main function to set up Python 3.10 environment."""
    print("\n" + "="*80)
    print("Cosmic Market Oracle - Python 3.10 Setup")
    print("="*80 + "\n")
    
    # Check Python version
    check_python_version()
    
    # Set up project structure
    project_root = setup_project_structure()
    
    # Install dependencies
    install_dependencies()
    
    # Create run scripts
    create_run_scripts()
    
    print("\n" + "="*80)
    print("Setup complete! You can now run the following commands:")
    print("="*80)
    print("\n1. Start MLflow UI:")
    print("   run_mlflow_ui.bat")
    print("\n2. Run hyperparameter optimization:")
    print("   run_optimization.bat")
    print("\n3. Access MLflow UI in your browser:")
    print("   http://localhost:5000")
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
