import os
import json
import mlflow
import mlflow.pytorch
from typing import Dict, Any, Optional, List, Union, Tuple
from pathlib import Path
from datetime import datetime
import pandas as pd
from mlflow.entities import Run
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.models.signature import ModelSignature
from mlflow.types import Schema, ColSpec
from src.utils.logger import get_logger

logger = get_logger(__name__)

def setup_mlflow(
    tracking_uri: str,
    experiment_name: str,
    artifact_location: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None
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
        
        # Set experiment tags if provided
        if tags:
            client = MlflowClient()
            for key, value in tags.items():
                client.set_experiment_tag(experiment_id, key, value)
            logger.info(f"Set {len(tags)} tags for experiment: {experiment_name}")
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


def register_model(
    model_name: str,
    model_uri: str,
    description: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
    version_description: Optional[str] = None
) -> str:
    """
    Register a model in the MLflow Model Registry.
    
    Args:
        model_name: Name to register the model under
        model_uri: URI of the model in MLflow
        description: Optional description for the model
        tags: Optional tags for the model
        version_description: Optional description for this specific model version
        
    Returns:
        Model version
    """
    try:
        client = MlflowClient()
        
        # Check if model already exists
        try:
            client.get_registered_model(model_name)
            logger.info(f"Using existing registered model: {model_name}")
        except MlflowException:
            # Create new registered model
            client.create_registered_model(
                name=model_name,
                description=description or "",
                tags=tags or {}
            )
            logger.info(f"Created new registered model: {model_name}")
        
        # Register new model version
        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=model_name
        )
        
        # Add version description if provided
        if version_description:
            client.update_model_version(
                name=model_name,
                version=model_version.version,
                description=version_description
            )
        
        # Add tags to model version if provided
        if tags:
            for key, value in tags.items():
                client.set_model_version_tag(
                    name=model_name,
                    version=model_version.version,
                    key=key,
                    value=value
                )
        
        logger.info(f"Registered model {model_name} version {model_version.version}")
        return model_version.version
    except Exception as e:
        logger.error(f"Failed to register model: {e}")
        raise


def transition_model_stage(
    model_name: str,
    version: str,
    stage: str,
    archive_existing_versions: bool = False
) -> None:
    """
    Transition a model version to a different stage in the MLflow Model Registry.
    
    Args:
        model_name: Name of the registered model
        version: Version of the model to transition
        stage: Target stage (Staging, Production, Archived)
        archive_existing_versions: Whether to archive existing versions in the target stage
    """
    try:
        client = MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage,
            archive_existing_versions=archive_existing_versions
        )
        logger.info(f"Transitioned model {model_name} version {version} to {stage}")
    except Exception as e:
        logger.error(f"Failed to transition model stage: {e}")
        raise


def compare_runs(
    experiment_id: str,
    metric_key: str,
    max_results: int = 10,
    ascending: bool = False
) -> pd.DataFrame:
    """
    Compare runs in an experiment based on a specific metric.
    
    Args:
        experiment_id: ID of the experiment
        metric_key: Metric to compare runs on
        max_results: Maximum number of runs to return
        ascending: Whether to sort in ascending order (True) or descending order (False)
        
    Returns:
        DataFrame with run information sorted by the specified metric
    """
    try:
        # Get all runs for the experiment
        runs = mlflow.search_runs(
            experiment_ids=[experiment_id],
            filter_string=f"metrics.{metric_key} IS NOT NULL",
            max_results=max_results,
            order_by=[f"metrics.{metric_key} {'ASC' if ascending else 'DESC'}"],
        )
        
        if runs.empty:
            logger.warning(f"No runs found with metric {metric_key} in experiment {experiment_id}")
            return pd.DataFrame()
        
        logger.info(f"Found {len(runs)} runs with metric {metric_key} in experiment {experiment_id}")
        return runs
    except Exception as e:
        logger.error(f"Failed to compare runs: {e}")
        raise


def log_artifacts_with_validation(
    artifacts: Dict[str, Union[str, Path, bytes, Dict, List]],
    artifact_dir: Optional[str] = None
) -> None:
    """
    Log artifacts with validation to MLflow.
    
    Args:
        artifacts: Dictionary of artifacts to log (name -> content)
        artifact_dir: Optional directory to log artifacts to
    """
    try:
        for name, content in artifacts.items():
            if isinstance(content, (str, Path)) and os.path.isfile(content):
                # Log file artifact
                mlflow.log_artifact(content, artifact_dir)
                logger.info(f"Logged file artifact: {name}")
            elif isinstance(content, (dict, list)):
                # Log JSON artifact
                mlflow.log_dict(content, f"{artifact_dir}/{name}.json" if artifact_dir else f"{name}.json")
                logger.info(f"Logged JSON artifact: {name}")
            elif isinstance(content, bytes):
                # Log binary artifact
                temp_path = Path(f"/tmp/{name}")
                with open(temp_path, "wb") as f:
                    f.write(content)
                mlflow.log_artifact(temp_path, artifact_dir)
                os.remove(temp_path)
                logger.info(f"Logged binary artifact: {name}")
            else:
                # Log text artifact
                mlflow.log_text(str(content), f"{artifact_dir}/{name}.txt" if artifact_dir else f"{name}.txt")
                logger.info(f"Logged text artifact: {name}")
    except Exception as e:
        logger.error(f"Failed to log artifacts: {e}")
        raise


def create_model_signature(
    inputs: List[Tuple[str, str]], 
    outputs: List[Tuple[str, str]]
) -> ModelSignature:
    """
    Create a model signature for MLflow model logging.
    
    Args:
        inputs: List of (name, type) tuples for input features
        outputs: List of (name, type) tuples for output features
        
    Returns:
        MLflow ModelSignature object
    """
    try:
        input_schema = Schema([ColSpec(type=dtype, name=name) for name, dtype in inputs])
        output_schema = Schema([ColSpec(type=dtype, name=name) for name, dtype in outputs])
        return ModelSignature(inputs=input_schema, outputs=output_schema)
    except Exception as e:
        logger.error(f"Failed to create model signature: {e}")
        raise