#!/usr/bin/env python
# Cosmic Market Oracle - Production Deployment Script

"""
Production deployment script for the Cosmic Market Oracle.

This script automates the deployment process for production environments,
including database migrations, Docker container management, and health checks.
"""

import os
import sys
import argparse
import subprocess
import time
import requests
from src.trading.unified_mcts import MCTS, MCTSPredictor
from src.trading.modular_hierarchical_rl import ModularHierarchicalRLAgent, MCTSStrategicPlanner, PPOTacticalExecutor
from pathlib import Path
from typing import Dict, List, Optional, Any
# import logging # Removed
from src.utils.logger import get_logger # Added

# Configure logging
# logging.basicConfig(...) # Removed
logger = get_logger("deployment") # Changed


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Deploy Cosmic Market Oracle to production")
    parser.add_argument("--env", choices=["staging", "production"], default="production",
                        help="Environment to deploy to")
    parser.add_argument("--skip-build", action="store_true", help="Skip building Docker images")
    parser.add_argument("--skip-migrations", action="store_true", help="Skip database migrations")
    parser.add_argument("--force-recreate", action="store_true", help="Force recreation of containers")
    return parser.parse_args()


def check_prerequisites() -> bool:
    """Check if all prerequisites are installed."""
    logger.info("Checking prerequisites...")
    
    # Check Docker
    try:
        subprocess.run(["docker", "--version"], check=True, capture_output=True)
        logger.info("Docker is installed")
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("Docker is not installed or not in PATH")
        return False
    
    # Check Docker Compose
    try:
        subprocess.run(["docker-compose", "--version"], check=True, capture_output=True)
        logger.info("Docker Compose is installed")
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("Docker Compose is not installed or not in PATH")
        return False
    
    # Check if .env file exists
    if not Path(".env").exists():
        logger.error(".env file not found")
        return False
    
    return True


def build_docker_images() -> bool:
    """Build Docker images."""
    logger.info("Building Docker images...")
    
    try:
        subprocess.run(["docker-compose", "build"], check=True)
        logger.info("Docker images built successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error building Docker images: {e}")
        return False


def run_database_migrations() -> bool:
    """Run database migrations."""
    logger.info("Running database migrations...")
    
    try:
        # Start only the database container if it's not running
        subprocess.run(["docker-compose", "up", "-d", "db"], check=True)
        
        # Wait for database to be ready
        logger.info("Waiting for database to be ready...")
        time.sleep(10)
        
        # Run migrations
        subprocess.run([
            "docker-compose", "run", "--rm", "api",
            "alembic", "upgrade", "head"
        ], check=True)
        
        logger.info("Database migrations completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running database migrations: {e}")
        return False


def start_services(force_recreate: bool = False) -> bool:
    """Start all services."""
    logger.info("Starting services...")
    
    cmd = ["docker-compose", "up", "-d"]
    if force_recreate:
        cmd.append("--force-recreate")
    
    try:
        subprocess.run(cmd, check=True)
        logger.info("Services started successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error starting services: {e}")
        return False


def check_service_health() -> bool:
    """Check if all services are healthy."""
    logger.info("Checking service health...")
    
    # Wait for services to start
    time.sleep(15)
    
    # Check API health
    try:
        response = requests.get("http://localhost:8000/health", timeout=10)
        if response.status_code == 200:
            logger.info("API service is healthy")
        else:
            logger.error(f"API service health check failed with status {response.status_code}")
            return False
    except requests.RequestException as e:
        logger.error(f"Error connecting to API service: {e}")
        return False
    
    # Check MLflow
    try:
        response = requests.get("http://localhost:5000", timeout=10)
        if response.status_code == 200:
            logger.info("MLflow service is healthy")
        else:
            logger.error(f"MLflow service health check failed with status {response.status_code}")
            return False
    except requests.RequestException as e:
        logger.error(f"Error connecting to MLflow service: {e}")
        return False
    
    # Check MinIO
    try:
        response = requests.get("http://localhost:9001", timeout=10)
        if response.status_code == 200:
            logger.info("MinIO service is healthy")
        else:
            logger.error(f"MinIO service health check failed with status {response.status_code}")
            return False
    except requests.RequestException as e:
        logger.error(f"Error connecting to MinIO service: {e}")
        return False
    
    logger.info("All services are healthy")
    return True


def main() -> int:
    """Main deployment function."""
    args = parse_arguments()
    
    logger.info(f"Starting deployment to {args.env} environment")
    
    # Change to project root directory
    project_root = Path(__file__).parent.parent.parent
    os.chdir(project_root)
    logger.info(f"Working directory: {os.getcwd()}")
    
    # Check prerequisites
    if not check_prerequisites():
        logger.error("Prerequisites check failed")
        return 1
    
    # Build Docker images if not skipped
    if not args.skip_build:
        if not build_docker_images():
            logger.error("Docker image build failed")
            return 1
    
    # Run database migrations if not skipped
    if not args.skip_migrations:
        if not run_database_migrations():
            logger.error("Database migrations failed")
            return 1
    
    # Start services
    if not start_services(args.force_recreate):
        logger.error("Starting services failed")
        return 1
    
    # Check service health
    if not check_service_health():
        logger.error("Service health check failed")
        return 1
    
    logger.info(f"Deployment to {args.env} environment completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
