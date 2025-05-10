import os
import time
import psutil
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime
from pathlib import Path
from prometheus_client import Counter, Gauge, Histogram, Summary, start_http_server
from src.utils.logging_config import get_logger, MarketLogger

logger = get_logger(__name__)

# Define Prometheus metrics
prediction_counter = Counter('model_predictions_total', 'Total number of predictions made', ['model_name', 'version'])
prediction_latency = Histogram('model_prediction_latency_seconds', 'Prediction latency in seconds', ['model_name', 'version'])
model_accuracy = Gauge('model_accuracy', 'Model accuracy metric', ['model_name', 'version', 'metric_name'])
data_drift_gauge = Gauge('data_drift_score', 'Data drift detection score', ['feature_name'])
system_memory_usage = Gauge('system_memory_usage_bytes', 'System memory usage in bytes')
system_cpu_usage = Gauge('system_cpu_usage_percent', 'System CPU usage percentage')


def start_monitoring_server(port: int = 8000) -> None:
    """
    Start Prometheus metrics server for monitoring.
    
    Args:
        port: Port to expose metrics on
    """
    try:
        start_http_server(port)
        logger.info(f"Started monitoring server on port {port}")
    except Exception as e:
        logger.error(f"Failed to start monitoring server: {e}")
        raise


def track_prediction(model_name: str, version: str, latency: float) -> None:
    """
    Track a model prediction event with its latency.
    
    Args:
        model_name: Name of the model
        version: Version of the model
        latency: Prediction latency in seconds
    """
    try:
        prediction_counter.labels(model_name=model_name, version=version).inc()
        prediction_latency.labels(model_name=model_name, version=version).observe(latency)
        logger.debug(f"Tracked prediction for {model_name} v{version} with latency {latency:.4f}s")
    except Exception as e:
        logger.error(f"Failed to track prediction: {e}")


def update_model_metrics(model_name: str, version: str, metrics: Dict[str, float]) -> None:
    """
    Update model performance metrics.
    
    Args:
        model_name: Name of the model
        version: Version of the model
        metrics: Dictionary of metric names and values
    """
    try:
        for metric_name, value in metrics.items():
            model_accuracy.labels(
                model_name=model_name, 
                version=version, 
                metric_name=metric_name
            ).set(value)
        logger.info(f"Updated metrics for {model_name} v{version}: {metrics}")
    except Exception as e:
        logger.error(f"Failed to update model metrics: {e}")


def track_data_drift(feature_drifts: Dict[str, float]) -> None:
    """
    Track data drift for model features.
    
    Args:
        feature_drifts: Dictionary of feature names and their drift scores
    """
    try:
        for feature_name, drift_score in feature_drifts.items():
            data_drift_gauge.labels(feature_name=feature_name).set(drift_score)
        
        # Log high drift features
        high_drift_features = {k: v for k, v in feature_drifts.items() if v > 0.1}
        if high_drift_features:
            logger.warning(f"High drift detected in features: {high_drift_features}")
    except Exception as e:
        logger.error(f"Failed to track data drift: {e}")


def monitor_system_resources(interval: int = 60) -> None:
    """
    Monitor and record system resource usage.
    
    Args:
        interval: Monitoring interval in seconds
    """
    try:
        while True:
            # Memory usage
            memory = psutil.virtual_memory()
            system_memory_usage.set(memory.used)
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            system_cpu_usage.set(cpu_percent)
            
            # Log if resources are running low
            if memory.percent > 90:
                logger.warning(f"High memory usage: {memory.percent}%")
            if cpu_percent > 90:
                logger.warning(f"High CPU usage: {cpu_percent}%")
                
            time.sleep(interval)
    except Exception as e:
        logger.error(f"Failed to monitor system resources: {e}")
        raise


def log_prediction_event(
    logger: MarketLogger,
    model_name: str,
    version: str,
    input_data: Dict[str, Any],
    prediction: Any,
    confidence: float,
    execution_time: float,
    astrological_factors: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log a prediction event with detailed information.
    
    Args:
        logger: Logger instance
        model_name: Name of the model
        version: Version of the model
        input_data: Input data for the prediction (will be summarized)
        prediction: Prediction result
        confidence: Confidence score for the prediction
        execution_time: Execution time in seconds
        astrological_factors: Optional astrological factors that influenced the prediction
    """
    try:
        # Create a summary of input data (avoid logging full data which could be large)
        input_summary = {
            "feature_count": len(input_data) if isinstance(input_data, dict) else "non-dict input",
            "timestamp": datetime.now().isoformat()
        }
        
        # Prepare log message
        log_data = {
            "model": model_name,
            "version": version,
            "input_summary": input_summary,
            "prediction": prediction,
            "confidence": confidence,
            "execution_time_ms": execution_time * 1000
        }
        
        # Add astrological factors if available
        if astrological_factors:
            log_data["astrological_factors"] = astrological_factors
        
        # Log as market signal
        logger.market_signal(f"Prediction: {log_data}")
        
        # Track in monitoring system
        track_prediction(model_name, version, execution_time)
    except Exception as e:
        logger.error(f"Failed to log prediction event: {e}")


def calculate_drift_metrics(
    reference_data: np.ndarray,
    current_data: np.ndarray,
    feature_names: List[str]
) -> Dict[str, float]:
    """
    Calculate drift metrics between reference and current data distributions.
    
    Args:
        reference_data: Reference data distribution
        current_data: Current data distribution
        feature_names: Names of the features
        
    Returns:
        Dictionary of feature names and their drift scores
    """
    try:
        drift_scores = {}
        
        for i, feature_name in enumerate(feature_names):
            # Extract feature data
            ref_feature = reference_data[:, i]
            curr_feature = current_data[:, i]
            
            # Calculate basic statistical drift
            ref_mean, ref_std = np.mean(ref_feature), np.std(ref_feature)
            curr_mean, curr_std = np.mean(curr_feature), np.std(curr_feature)
            
            # Calculate normalized difference in distributions
            mean_diff = abs(ref_mean - curr_mean) / (ref_std if ref_std > 0 else 1)
            std_ratio = abs(ref_std - curr_std) / (ref_std if ref_std > 0 else 1)
            
            # Combined drift score (higher means more drift)
            drift_score = (mean_diff + std_ratio) / 2
            drift_scores[feature_name] = drift_score
            
            # Log significant drift
            if drift_score > 0.25:
                logger.warning(f"Significant drift detected in feature '{feature_name}': {drift_score:.4f}")
        
        return drift_scores
    except Exception as e:
        logger.error(f"Failed to calculate drift metrics: {e}")
        raise