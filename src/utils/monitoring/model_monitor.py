import os
import json
import time
import logging
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import numpy as np
from datetime import datetime

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelMonitor:
    """
    Model monitoring system for the Cosmic Market Oracle project.
    
    This class provides functionality for:
    1. Tracking model performance metrics over time
    2. Detecting data drift in input features
    3. Monitoring prediction distribution shifts
    4. Generating alerts when metrics exceed thresholds
    5. Exporting Prometheus metrics for integration with monitoring dashboards
    """
    
    def __init__(self, model_name: str, version: str, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize the model monitoring system.
        
        Args:
            model_name: Name of the model to monitor
            version: Version of the model
            config_path: Path to monitoring configuration file
        """
        self.model_name = model_name
        self.version = version
        self.start_time = datetime.now()
        
from src.utils.config import Config # Added import

        # Load configuration
        if config_path:
            self.config = Config(config_path)
        else:
            default_monitoring_config_path = Path("config/monitoring.json")
            if default_monitoring_config_path.exists():
                self.config = Config(default_monitoring_config_path)
            else:
                logger.warning(f"Monitoring config file {default_monitoring_config_path} not found. Using internal defaults for ModelMonitor.")
                self.config = Config(None) # Pass None to avoid file load attempt
                self.config.config = self._get_default_config() # Manually set the config dict

        # Set up monitoring directory
        self.monitoring_dir = Path(self.config.get("monitoring_dir", self._get_default_config()["monitoring_dir"]))
        self.monitoring_dir.mkdir(parents=True, exist_ok=True)
        
        # Model-specific monitoring directory
        self.model_dir = self.monitoring_dir / model_name / version
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metrics storage
        self.metrics_history: Dict[str, List[Dict[str, Any]]] = {}
        self.feature_stats: Dict[str, Dict[str, Any]] = {}
        self.prediction_stats: Dict[str, Dict[str, Any]] = {}
        self.alerts: List[Dict[str, Any]] = []
        
        # Load historical data if available
        self._load_historical_data()
        
        logger.info(f"Initialized model monitoring for {model_name} version {version}")
    
    # Removed _load_config method as Config class handles loading.
    # _get_default_config is kept for now if direct Config init with defaults is needed as fallback.

    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default monitoring configuration.
        
        Returns:
            Default configuration dictionary
        """
        return {
            "monitoring_dir": "monitoring",
            "metrics": {
                "performance": ["accuracy", "precision", "recall", "f1"],
                "thresholds": {
                    "accuracy": 0.75,
                    "precision": 0.70,
                    "recall": 0.70,
                    "f1": 0.70
                }
            },
            "data_drift": {
                "detection_method": "ks_test",
                "significance_level": 0.05,
                "min_samples_required": 100
            },
            "alerts": {
                "enabled": True,
            "channels": ["log"], # Ensure this is a list
                "throttle_period_seconds": 3600  # Don't send duplicate alerts for 1 hour
            },
            "prometheus": {
                "enabled": True,
                "export_path": "monitoring/prometheus",
                "update_interval_seconds": 60
            }
        }
    
    def _load_historical_data(self) -> None:
        """
        Load historical monitoring data if available.
        """
        try:
            metrics_file = self.model_dir / "metrics_history.json"
            if metrics_file.exists():
                with open(metrics_file, "r") as f:
                    self.metrics_history = json.load(f)
                logger.info(f"Loaded historical metrics for {self.model_name} {self.version}")
            
            feature_stats_file = self.model_dir / "feature_stats.json"
            if feature_stats_file.exists():
                with open(feature_stats_file, "r") as f:
                    self.feature_stats = json.load(f)
                logger.info(f"Loaded feature statistics for {self.model_name} {self.version}")
            
            prediction_stats_file = self.model_dir / "prediction_stats.json"
            if prediction_stats_file.exists():
                with open(prediction_stats_file, "r") as f:
                    self.prediction_stats = json.load(f)
                logger.info(f"Loaded prediction statistics for {self.model_name} {self.version}")
            
            alerts_file = self.model_dir / "alerts.json"
            if alerts_file.exists():
                with open(alerts_file, "r") as f:
                    self.alerts = json.load(f)
                logger.info(f"Loaded alert history for {self.model_name} {self.version}")
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
    
    def save_data(self) -> None:
        """
        Save current monitoring data to disk.
        """
        try:
            metrics_file = self.model_dir / "metrics_history.json"
            with open(metrics_file, "w") as f:
                json.dump(self.metrics_history, f, indent=2)
            
            feature_stats_file = self.model_dir / "feature_stats.json"
            with open(feature_stats_file, "w") as f:
                json.dump(self.feature_stats, f, indent=2)
            
            prediction_stats_file = self.model_dir / "prediction_stats.json"
            with open(prediction_stats_file, "w") as f:
                json.dump(self.prediction_stats, f, indent=2)
            
            alerts_file = self.model_dir / "alerts.json"
            with open(alerts_file, "w") as f:
                json.dump(self.alerts, f, indent=2)
            
            logger.info(f"Saved monitoring data for {self.model_name} {self.version}")
        except Exception as e:
            logger.error(f"Error saving monitoring data: {e}")
    
    def record_metrics(self, metrics: Dict[str, float], timestamp: Optional[str] = None) -> None:
        """
        Record performance metrics for the model.
        
        Args:
            metrics: Dictionary of metric names and values
            timestamp: Optional timestamp string (ISO format), defaults to current time
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        # Add timestamp to metrics
        metrics_entry = {"timestamp": timestamp, **metrics}
        
        # Initialize metrics history for each metric if not exists
        for metric_name in metrics.keys():
            if metric_name not in self.metrics_history:
                self.metrics_history[metric_name] = []
            
            # Add new metrics
            self.metrics_history[metric_name].append(metrics_entry)
        
        # Check for alerts
        self._check_metric_thresholds(metrics)
        
        # Update Prometheus metrics if enabled
        if self.config.get("prometheus.enabled", False):
            self._update_prometheus_metrics(metrics)
        
        logger.info(f"Recorded metrics for {self.model_name} {self.version}: {metrics}")
    
    def record_feature_statistics(self, feature_name: str, statistics: Dict[str, Any]) -> None:
        """
        Record statistics for a feature to monitor data drift.
        
        Args:
            feature_name: Name of the feature
            statistics: Dictionary of statistics (mean, std, min, max, etc.)
        """
        # Add timestamp
        statistics["timestamp"] = datetime.now().isoformat()
        
        # Store feature statistics
        if feature_name not in self.feature_stats:
            self.feature_stats[feature_name] = []
        
        self.feature_stats[feature_name].append(statistics)
        
        # Check for data drift if we have historical data
        if len(self.feature_stats[feature_name]) > 1:
            self._check_data_drift(feature_name)
        
        logger.debug(f"Recorded statistics for feature {feature_name}")
    
    def record_prediction_statistics(self, statistics: Dict[str, Any]) -> None:
        """
        Record statistics about model predictions to detect concept drift.
        
        Args:
            statistics: Dictionary of prediction statistics
        """
        # Add timestamp
        statistics["timestamp"] = datetime.now().isoformat()
        
        # Store prediction statistics
        timestamp = statistics["timestamp"]
        self.prediction_stats[timestamp] = statistics
        
        # Check for prediction drift if we have enough data
        if len(self.prediction_stats) > 1:
            self._check_prediction_drift(statistics)
        
        logger.debug(f"Recorded prediction statistics")
    
    def _check_metric_thresholds(self, metrics: Dict[str, float]) -> None:
        """
        Check if any metrics have crossed defined thresholds.
        
        Args:
            metrics: Dictionary of metric names and values
        """
        thresholds = self.config.get("metrics", {}).get("thresholds", {})
        
        for metric_name, value in metrics.items():
            if metric_name in thresholds: # Accessing potentially nested dict
                threshold_val = thresholds[metric_name] # This could be a direct value or another dict
                # Assuming thresholds is a flat dict like: {"accuracy": 0.75}
                if isinstance(threshold_val, (int, float)) and value < threshold_val:
                    alert_message = f"Metric {metric_name} below threshold: {value:.4f} < {threshold_val:.4f}"
                    self._create_alert("metric_threshold", alert_message, {
                        "metric": metric_name,
                        "value": value,
                        "threshold": threshold_val
                    })
                elif isinstance(threshold_val, dict): # Example: {"accuracy": {"min": 0.7, "max": 0.9}}
                    if "min" in threshold_val and value < threshold_val["min"]:
                         alert_message = f"Metric {metric_name} below min threshold: {value:.4f} < {threshold_val['min']:.4f}"
                         self._create_alert("metric_threshold_min", alert_message, {
                             "metric": metric_name,
                             "value": value,
                             "threshold_min": threshold_val['min']
                         })
                    if "max" in threshold_val and value > threshold_val["max"]:
                        alert_message = f"Metric {metric_name} above max threshold: {value:.4f} > {threshold_val['max']:.4f}"
                        self._create_alert("metric_threshold_max", alert_message, {
                            "metric": metric_name,
                            "value": value,
                            "threshold_max": threshold_val['max']
                        })
                        "value": value,
                        "threshold": threshold
                    })
    
    def _check_data_drift(self, feature_name: str) -> None:
        """
        Check for data drift in a feature.
        
        Args:
            feature_name: Name of the feature to check
        """
        # This is a simplified implementation
        # In a real system, this would use statistical tests like KS test
        try:
            current_stats = self.feature_stats[feature_name][-1]
            baseline_stats = self.feature_stats[feature_name][0]
            
            # Simple check: compare mean and std
            if "mean" in current_stats and "mean" in baseline_stats:
                mean_diff = abs(current_stats["mean"] - baseline_stats["mean"])
                mean_pct_change = mean_diff / abs(baseline_stats["mean"]) if baseline_stats["mean"] != 0 else float('inf')
                
                # Alert if mean has changed by more than 20%
                if mean_pct_change > 0.2:
                    alert_message = f"Data drift detected in feature {feature_name}: mean changed by {mean_pct_change:.2%}"
                    self._create_alert("data_drift", alert_message, {
                        "feature": feature_name,
                        "metric": "mean",
                        "baseline": baseline_stats["mean"],
                        "current": current_stats["mean"],
                        "change_pct": mean_pct_change
                    })
        except Exception as e:
            logger.error(f"Error checking data drift for {feature_name}: {e}")
    
    def _check_prediction_drift(self, current_stats: Dict[str, Any]) -> None:
        """
        Check for drift in prediction distributions.
        
        Args:
            current_stats: Current prediction statistics
        """
        # This is a simplified implementation
        # In a real system, this would use distribution comparison tests
        try:
            # Get baseline stats (first recorded stats)
            baseline_timestamp = list(self.prediction_stats.keys())[0]
            baseline_stats = self.prediction_stats[baseline_timestamp]
            
            # Check for drift in prediction distribution
            if "class_distribution" in current_stats and "class_distribution" in baseline_stats:
                baseline_dist = baseline_stats["class_distribution"]
                current_dist = current_stats["class_distribution"]
                
                # Calculate distribution difference
                total_diff = 0
                for class_name in set(baseline_dist.keys()) | set(current_dist.keys()):
                    baseline_val = baseline_dist.get(class_name, 0)
                    current_val = current_dist.get(class_name, 0)
                    total_diff += abs(baseline_val - current_val)
                
                # Alert if distribution has changed significantly
                if total_diff > 0.3:  # 30% change in distribution
                    alert_message = f"Prediction drift detected: distribution changed by {total_diff:.2%}"
                    self._create_alert("prediction_drift", alert_message, {
                        "baseline_distribution": baseline_dist,
                        "current_distribution": current_dist,
                        "total_difference": total_diff
                    })
        except Exception as e:
            logger.error(f"Error checking prediction drift: {e}")
    
    def _create_alert(self, alert_type: str, message: str, details: Dict[str, Any]) -> None:
        """
        Create and record an alert.
        
        Args:
            alert_type: Type of alert (e.g., "metric_threshold", "data_drift")
            message: Alert message
            details: Additional alert details
        """
            if not self.config.get("alerts.enabled", True):
            return
        
        # Create alert
        alert = {
            "type": alert_type,
            "message": message,
            "details": details,
            "timestamp": datetime.now().isoformat(),
            "model": self.model_name,
            "version": self.version
        }
        
        # Check for duplicate alerts within throttle period
        throttle_period = self.config.get("alerts.throttle_period_seconds", 3600)
        duplicate = False
        
        if self.alerts:
            last_alert = self.alerts[-1]
            if last_alert["type"] == alert_type and last_alert["message"] == message: # Simple field check
                last_time = datetime.fromisoformat(last_alert["timestamp"])
                current_time = datetime.fromisoformat(alert["timestamp"])
                seconds_diff = (current_time - last_time).total_seconds()
                
                if seconds_diff < throttle_period:
                    duplicate = True
        
        if not duplicate:
            self.alerts.append(alert)
            
            # Send alert through configured channels
            channels = self.config.get("alerts.channels", ["log"]) # Ensure list
            
            if "log" in channels:
                logger.warning(f"ALERT: {message}")
            
            # Additional channels would be implemented here (email, Slack, etc.)
        
        # Save alerts to disk
        self.save_data()
    
    def _update_prometheus_metrics(self, metrics: Dict[str, float]) -> None:
        """
        Update Prometheus metrics file for scraping.
        
        Args:
            metrics: Dictionary of metric names and values
        """
        try:
            if not self.config.get("prometheus.enabled", False):
                return
            
            export_path = Path(self.config.get("prometheus.export_path", "monitoring/prometheus"))
            export_path.mkdir(parents=True, exist_ok=True)
            
            metrics_file = export_path / f"{self.model_name}_{self.version}_metrics.prom"
            
            # Format metrics in Prometheus format
            lines = []
            timestamp_ms = int(time.time() * 1000)
            
            for metric_name, value in metrics.items():
                metric_id = f"cosmic_oracle_model_{metric_name}"
                help_line = f"# HELP {metric_id} Model {self.model_name} {metric_name} metric"
                type_line = f"# TYPE {metric_id} gauge"
                value_line = f"{metric_id}{{model=\"{self.model_name}\", version=\"{self.version}\"}} {value} {timestamp_ms}"
                
                lines.extend([help_line, type_line, value_line, ""])
            
            # Write to file
            with open(metrics_file, "w") as f:
                f.write("\n".join(lines))
            
            logger.debug(f"Updated Prometheus metrics at {metrics_file}")
        except Exception as e:
            logger.error(f"Error updating Prometheus metrics: {e}")
    
    def get_uptime(self) -> float:
        """
        Get model monitoring uptime in seconds.
        
        Returns:
            Uptime in seconds
        """
        return (datetime.now() - self.start_time).total_seconds()
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the most recent metrics.
        
        Returns:
            Dictionary with metric summaries
        """
        summary = {
            "model": self.model_name,
            "version": self.version,
            "uptime_seconds": self.get_uptime(),
            "metrics": {},
            "alerts": len(self.alerts),
            "last_update": datetime.now().isoformat()
        }
        
        # Add latest metrics
        for metric_name, history in self.metrics_history.items():
            if history:
                latest = history[-1]
                summary["metrics"][metric_name] = {
                    "current": latest.get(metric_name),
                    "timestamp": latest.get("timestamp")
                }
                
                # Add trend if we have enough data
                if len(history) >= 2:
                    previous = history[-2].get(metric_name)
                    current = latest.get(metric_name)
                    if previous is not None and current is not None:
                        change = current - previous
                        summary["metrics"][metric_name]["change"] = change
                        summary["metrics"][metric_name]["trend"] = "up" if change > 0 else "down" if change < 0 else "stable"
        
        return summary


def create_default_monitoring_config():
    """
    Create a default monitoring configuration file.
    """
    config_dir = Path("config")
    config_dir.mkdir(parents=True, exist_ok=True)
    
    config_file = config_dir / "monitoring.json"
    
    if config_file.exists():
        logger.info(f"Monitoring config file {config_file} already exists, skipping creation")
        return
    
    default_config = {
        "monitoring_dir": "monitoring",
        "metrics": {
            "performance": ["accuracy", "precision", "recall", "f1"],
            "thresholds": {
                "accuracy": 0.75,
                "precision": 0.70,
                "recall": 0.70,
                "f1": 0.70
            }
        },
        "data_drift": {
            "detection_method": "ks_test",
            "significance_level": 0.05,
            "min_samples_required": 100,
            "features_to_monitor": ["all"],
            "reference_dataset": "data/processed/reference.csv"
        },
        "alerts": {
            "enabled": True,
            "channels": ["log", "email", "slack"],
            "throttle_period_seconds": 3600,
            "email": {
                "recipients": ["team@cosmicmarketoracle.com"],
                "subject_prefix": "[Model Alert]"
            },
            "slack": {
                "webhook_url": "",
                "channel": "#model-alerts"
            }
        },
        "prometheus": {
            "enabled": True,
            "export_path": "monitoring/prometheus",
            "update_interval_seconds": 60,
            "metrics_to_export": ["all"]
        },
        "logging": {
            "level": "INFO",
            "retention_days": 30
        }
    }
    
    with open(config_file, "w") as f:
        json.dump(default_config, f, indent=2)
    
    logger.info(f"Created default monitoring configuration at {config_file}")


if __name__ == "__main__":
    create_default_monitoring_config()