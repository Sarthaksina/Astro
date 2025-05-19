#!/usr/bin/env python
# Cosmic Market Oracle - Environment Setup Script

import os
import subprocess
import sys
import platform
from src.trading.unified_mcts import MCTS, MCTSPredictor
from src.trading.modular_hierarchical_rl import ModularHierarchicalRLAgent, MCTSStrategicPlanner, PPOTacticalExecutor
from pathlib import Path


def check_python_version():
    """Check if Python version meets requirements (3.10+)"""
    required_major = 3
    required_minor = 10
    
    current_major = sys.version_info.major
    current_minor = sys.version_info.minor
    
    if current_major < required_major or (current_major == required_major and current_minor < required_minor):
        print(f"Error: Python {required_major}.{required_minor}+ is required. You have {current_major}.{current_minor}")
        return False
    
    print(f"Python version check passed: {current_major}.{current_minor}")
    return True


def create_virtual_environment():
    """Create a Python virtual environment"""
    venv_path = Path("venv")
    
    if venv_path.exists():
        print("Virtual environment already exists")
        return True
    
    print("Creating virtual environment...")
    try:
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        print("Virtual environment created successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error creating virtual environment: {e}")
        return False


def install_dependencies():
    """Install project dependencies from requirements.txt"""
    print("Installing dependencies...")
    
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
        
        print("Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        return False


def setup_environment_variables():
    """Create a .env file with default environment variables"""
    env_file = Path(".env")
    
    if env_file.exists():
        print(".env file already exists")
        return True
    
    print("Creating .env file with default settings...")
    
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
        print(".env file created successfully")
        return True
    except Exception as e:
        print(f"Error creating .env file: {e}")
        return False


def create_data_directories():
    """Create necessary data directories"""
    directories = [
        Path("data", "market_data"),
        Path("data", "ephemeris"),
        Path("data", "processed"),
        Path("data", "models"),
    ]
    
    print("Creating data directories...")
    
    try:
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        print("Data directories created successfully")
        return True
    except Exception as e:
        print(f"Error creating data directories: {e}")
        return False


def main():
    """Main setup function"""
    print("=== Cosmic Market Oracle - Environment Setup ===")
    
    # Change to the project root directory
    project_root = Path(__file__).parent.parent.parent
    os.chdir(project_root)
    print(f"Working directory: {os.getcwd()}")
    
    # Run setup steps
    steps = [
        ("Checking Python version", check_python_version),
        ("Creating virtual environment", create_virtual_environment),
        ("Installing dependencies", install_dependencies),
        ("Setting up environment variables", setup_environment_variables),
        ("Creating data directories", create_data_directories),
    ]
    
    success = True
    for step_name, step_func in steps:
        print(f"\n{step_name}...")
        if not step_func():
            success = False
            print(f"Error in step: {step_name}")
            break
    
    if success:
        print("\n=== Setup completed successfully! ===")
        print("\nTo activate the virtual environment:")
        if platform.system() == "Windows":
            print("    .\\venv\\Scripts\\activate")
        else:
            print("    source venv/bin/activate")
        print("\nTo start the API server:")
        print("    python -m src.api.app")
    else:
        print("\n=== Setup failed! Please check the errors above. ===")


if __name__ == "__main__":
    main()