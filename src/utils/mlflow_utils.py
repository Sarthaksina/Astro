import os
import mlflow
import mlflow.pytorch
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

def setup_mlflow(
    tracking_uri: str,
    experiment_name: str,
    artifact_location: Optional[str] = None
) -> str:
    """
    Set up MLflow tracking and experiment.
    
    Args:
        tracking_uri: URI for MLflow tracking server
        experiment_name: Name of the experiment
        artifact_location: Optional location for storing artifacts
        
    Returns:
        Experiment ID
    """
    # Set up MLflow tracking
    mlflow.set_tracking_uri(tracking_uri)
    
    # Create or get experiment
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(
                experiment_name,
                artifact_location=artifact_location
            )
            logger.info(f"Created new experiment: {experiment_name}")
        else:
            experiment_id = experiment.experiment_id
            logger.info(f"Using existing experiment: {experiment_name}")
    except Exception as e:
        logger.error(f"Failed to set up MLflow experiment: {e}")
        raise
    
    return experiment_id

def log_model_params(model_config: Dict[str, Any]):
    """
    Log model parameters to MLflow.
    
    Args:
        model_config: Model configuration dictionary
    """
    try:
        # Log model type and specific configuration
        mlflow.log_param("model_type", model_config.get('model_type'))
        
        # Log specific model configuration
        config_key = f"{model_config['model_type']}_config"
        if config_key in model_config:
            for key, value in model_config[config_key].items():
                mlflow.log_param(f"{config_key}.{key}", value)
        
        # Log optimizer configuration
        if 'optimizer_config' in model_config:
            for key, value in model_config['optimizer_config'].items():
                mlflow.log_param(f"optimizer.{key}", value)
        
        # Log scheduler configuration
        if 'scheduler_config' in model_config:
            for key, value in model_config['scheduler_config'].items():
                mlflow.log_param(f"scheduler.{key}", value)
    except Exception as e:
        logger.error(f"Failed to log model parameters: {e}")
        raise

def log_metrics(metrics: Dict[str, float], step: Optional[int] = None):
    """
    Log metrics to MLflow.
    
    Args:
        metrics: Dictionary of metric names and values
        step: Optional step number for the metrics
    """
    try:
        for metric_name, value in metrics.items():
            mlflow.log_metric(metric_name, value, step=step)
    except Exception as e:
        logger.error(f"Failed to log metrics: {e}")
        raise

def save_model(model, model_path: str, config: Dict[str, Any]):
    """
    Save model with MLflow.
    
    Args:
        model: PyTorch model instance
        model_path: Path to save the model
        config: Model configuration dictionary
    """
    try:
        # Save model with MLflow
        mlflow.pytorch.log_model(
            model,
            model_path,
            conda_env={
                'name': 'cosmic_market_oracle',
                'channels': ['defaults', 'pytorch'],
                'dependencies': [
                    'python>=3.8',
                    'pytorch>=1.8.0',
                    'numpy>=1.19.2'
                ]
            }
        )
        
        # Log model configuration as artifact
        config_path = Path(model_path) / 'model_config.json'
        mlflow.log_dict(config, config_path.as_posix())
        
        logger.info(f"Model saved successfully at {model_path}")
    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        raise

def load_model(model_uri: str) -> Any:
    """
    Load model from MLflow.
    
    Args:
        model_uri: URI of the model in MLflow
        
    Returns:
        Loaded model instance
    """
    try:
        model = mlflow.pytorch.load_model(model_uri)
        logger.info(f"Model loaded successfully from {model_uri}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise