#!/usr/bin/env python
# Cosmic Market Oracle - Environment Setup Script

import os
import subprocess
import sys
import platform
# Imports for MCTS and ModularHierarchicalRLAgent seem unused in this script, consider removing if not needed.
# from src.trading.unified_mcts import MCTS, MCTSPredictor
# from src.trading.modular_hierarchical_rl import ModularHierarchicalRLAgent, MCTSStrategicPlanner, PPOTacticalExecutor
from pathlib import Path
from src.utils.environment import check_python_version
from src.utils.logger import get_logger # Added import

logger = get_logger(__name__) # Added module-level logger

# Removed local check_python_version function

def create_virtual_environment():
    """Create a Python virtual environment"""
    venv_path = Path("venv")
    
    if venv_path.exists():
        logger.info("Virtual environment already exists")
        return True
    
    logger.info("Creating virtual environment...")
    try:
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        logger.info("Virtual environment created successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error creating virtual environment: {e}")
        return False


def install_dependencies():
    """Install project dependencies from requirements.txt"""
    logger.info("Installing dependencies...")
    
    # Determine the path to pip based on the operating system
    if platform.system() == "Windows":
        pip_path = Path("venv", "Scripts", "pip")
    else:
        pip_path = Path("venv", "bin", "pip")
    
    try:
        # Upgrade pip first
        subprocess.run([str(pip_path), "install", "--upgrade", "pip"], check=True)
        
        # Install requirements
        subprocess.run([str(pip_path), "install", "-r", "requirements.txt"], check=True)
        
        # Install the package in development mode
        subprocess.run([str(pip_path), "install", "-e", "."], check=True)
        
        logger.info("Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error installing dependencies: {e}")
        return False


def setup_environment_variables():
    """Create a .env file with default environment variables"""
    env_file = Path(".env")
    
    if env_file.exists():
        logger.info(".env file already exists")
        return True
    
    logger.info("Creating .env file with default settings...")
    
    env_content = """\
# Database Configuration
DB_USER=postgres
DB_PASSWORD=postgres
DB_HOST=localhost
DB_PORT=5432
DB_NAME=cosmic_market_oracle

# Swiss Ephemeris Configuration
EPHE_PATH=./data/ephemeris

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# MLflow Configuration
MLFLOW_TRACKING_URI=http://localhost:5000
"""
    
    try:
        with open(env_file, "w") as f:
            f.write(env_content)
        logger.info(".env file created successfully")
        return True
    except Exception as e:
        logger.error(f"Error creating .env file: {e}")
        return False


def create_data_directories():
    """Create necessary data directories"""
    directories = [
        Path("data", "market_data"),
        Path("data", "ephemeris"),
        Path("data", "processed"),
        Path("data", "models"),
    ]
    
    logger.info("Creating data directories...")
    
    try:
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        logger.info("Data directories created successfully")
        return True
    except Exception as e:
        logger.error(f"Error creating data directories: {e}")
        return False


def main():
    """Main setup function"""
    logger.info("=== Cosmic Market Oracle - Environment Setup ===")
    
    # Change to the project root directory
    project_root = Path(__file__).parent.parent.parent
    os.chdir(project_root)
    logger.info(f"Working directory: {os.getcwd()}")
    
    # Run setup steps
    steps = [
        ("Checking Python version", lambda: check_python_version(3, 10)), # Updated to use imported function
        ("Creating virtual environment", create_virtual_environment),
        ("Installing dependencies", install_dependencies),
        ("Setting up environment variables", setup_environment_variables),
        ("Creating data directories", create_data_directories),
    ]
    
    success = True
    for step_name, step_func in steps:
        logger.info(f"\n{step_name}...")
        if not step_func():
            success = False
            logger.error(f"Error in step: {step_name}")
            break
    
    if success:
        logger.info("\n=== Setup completed successfully! ===")
        # Print messages for user interaction remain as print()
        print("\nTo activate the virtual environment:")
        if platform.system() == "Windows":
            print("    .\\venv\\Scripts\\activate")
        else:
            print("    source venv/bin/activate")
        print("\nTo start the API server:")
        print("    python -m src.api.app")
    else:
        logger.error("\n=== Setup failed! Please check the errors above. ===")
        print("\n=== Setup failed! Please check the errors above. ===") # Keep print for user


if __name__ == "__main__":
    main()