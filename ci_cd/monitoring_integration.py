import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

from src.utils.logging_config import get_logger
from src.utils.monitoring.model_monitor import ModelMonitor

logger = get_logger(__name__)


class MonitoringIntegration:
    """
    Integration between CI/CD pipeline and model monitoring system.
    
    This class provides functionality for:
    1. Registering models with the monitoring system during deployment
    2. Collecting and reporting monitoring metrics during CI/CD runs
    3. Triggering alerts based on monitoring thresholds
    4. Generating monitoring reports for CI/CD pipelines
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize the monitoring integration.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config_path = Path(config_path) if config_path else Path("config/monitoring.json")
        self.config = self._load_config()
        
        # Set up monitoring directory
        self.monitoring_dir = Path(self.config.get("monitoring_dir", "monitoring"))
        self.monitoring_dir.mkdir(parents=True, exist_ok=True)
        
        # Active monitors
        self.active_monitors: Dict[str, ModelMonitor] = {}
        
        logger.info("Initialized monitoring integration")
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load monitoring configuration.
        
        Returns:
            Configuration dictionary
        """
        try:
            if not self.config_path.exists():
                logger.warning(f"Monitoring config file {self.config_path} not found, using defaults")
                return self._get_default_config()
            
            with open(self.config_path, "r") as f:
                config = json.load(f)
            
            logger.info(f"Loaded monitoring configuration from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load monitoring config: {e}")
            return self._get_default_config()
    
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
            "alerts": {
                "enabled": True,
                "channels": ["log"],
                "throttle_period_seconds": 3600
            }
        }
    
    def register_model(self, model_name: str, version: str, metrics: Dict[str, float]) -> ModelMonitor:
        """
        Register a model with the monitoring system during deployment.
        
        Args:
            model_name: Name of the model
            version: Version of the model
            metrics: Initial performance metrics
            
        Returns:
            ModelMonitor instance
        """
        try:
            # Create or get monitor for this model
            monitor_key = f"{model_name}_{version}"
            if monitor_key in self.active_monitors:
                monitor = self.active_monitors[monitor_key]
                logger.info(f"Using existing monitor for {model_name} version {version}")
            else:
                monitor = ModelMonitor(model_name, version, self.config_path)
                self.active_monitors[monitor_key] = monitor
                logger.info(f"Registered new monitor for {model_name} version {version}")
            
            # Record initial metrics
            monitor.record_metrics(metrics)
            
            return monitor
        except Exception as e:
            logger.error(f"Error registering model with monitoring system: {e}")
            raise
    
    def record_deployment(self, model_name: str, version: str, environment: str, 
                         metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Record a model deployment with the monitoring system.
        
        Args:
            model_name: Name of the model
            version: Version of the model
            environment: Deployment environment
            metrics: Performance metrics
            
        Returns:
            Deployment record
        """
        try:
            # Register model with monitoring system
            monitor = self.register_model(model_name, version, metrics)
            
            # Create deployment record
            deployment_record = {
                "model_name": model_name,
                "version": version,
                "environment": environment,
                "timestamp": datetime.now().isoformat(),
                "metrics": metrics,
                "monitoring_enabled": True
            }
            
            # Save deployment record
            deployments_dir = self.monitoring_dir / "deployments"
            deployments_dir.mkdir(parents=True, exist_ok=True)
            
            record_file = deployments_dir / f"{model_name}_{version}_{environment}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(record_file, "w") as f:
                json.dump(deployment_record, f, indent=2)
            
            logger.info(f"Recorded deployment of {model_name} version {version} to {environment}")
            return deployment_record
        except Exception as e:
            logger.error(f"Error recording deployment: {e}")
            raise
    
    def generate_monitoring_report(self, model_name: str, version: str) -> Dict[str, Any]:
        """
        Generate a monitoring report for a model.
        
        Args:
            model_name: Name of the model
            version: Version of the model
            
        Returns:
            Monitoring report
        """
        try:
            monitor_key = f"{model_name}_{version}"
            if monitor_key not in self.active_monitors:
                logger.warning(f"No active monitor found for {model_name} version {version}")
                # Try to create a new monitor
                monitor = ModelMonitor(model_name, version, self.config_path)
            else:
                monitor = self.active_monitors[monitor_key]
            
            # Generate report
            report = {
                "model_name": model_name,
                "version": version,
                "timestamp": datetime.now().isoformat(),
                "metrics_summary": monitor.get_metrics_summary(),
                "alerts": len(monitor.alerts),
                "uptime_seconds": monitor.get_uptime()
            }
            
            # Save report
            reports_dir = self.monitoring_dir / "reports"
            reports_dir.mkdir(parents=True, exist_ok=True)
            
            report_file = reports_dir / f"{model_name}_{version}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, "w") as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Generated monitoring report for {model_name} version {version}")
            return report
        except Exception as e:
            logger.error(f"Error generating monitoring report: {e}")
            raise
    
    def check_model_health(self, model_name: str, version: str) -> Dict[str, Any]:
        """
        Check the health of a deployed model.
        
        Args:
            model_name: Name of the model
            version: Version of the model
            
        Returns:
            Health check results
        """
        try:
            monitor_key = f"{model_name}_{version}"
            if monitor_key not in self.active_monitors:
                logger.warning(f"No active monitor found for {model_name} version {version}")
                return {
                    "model_name": model_name,
                    "version": version,
                    "status": "unknown",
                    "message": "No active monitor found for this model"
                }
            
            monitor = self.active_monitors[monitor_key]
            metrics_summary = monitor.get_metrics_summary()
            
            # Check if any metrics are below thresholds
            thresholds = self.config.get("metrics", {}).get("thresholds", {})
            below_threshold = []
            
            for metric_name, threshold in thresholds.items():
                if metric_name in metrics_summary.get("metrics", {}):
                    current_value = metrics_summary["metrics"][metric_name].get("current")
                    if current_value is not None and current_value < threshold:
                        below_threshold.append({
                            "metric": metric_name,
                            "current": current_value,
                            "threshold": threshold
                        })
            
            # Determine overall status
            if below_threshold:
                status = "unhealthy"
                message = f"{len(below_threshold)} metrics below threshold"
            else:
                status = "healthy"
                message = "All metrics above thresholds"
            
            # Create health check result
            health_check = {
                "model_name": model_name,
                "version": version,
                "timestamp": datetime.now().isoformat(),
                "status": status,
                "message": message,
                "metrics_below_threshold": below_threshold,
                "alerts_count": len(monitor.alerts)
            }
            
            logger.info(f"Health check for {model_name} version {version}: {status}")
            return health_check
        except Exception as e:
            logger.error(f"Error checking model health: {e}")
            return {
                "model_name": model_name,
                "version": version,
                "status": "error",
                "message": f"Error checking health: {str(e)}"
            }


def integrate_monitoring_with_pipeline(model_name: str, version: str, environment: str, metrics: Dict[str, float]) -> Dict[str, Any]:
    """
    Utility function to integrate monitoring with CI/CD pipeline.
    
    Args:
        model_name: Name of the model
        version: Version of the model
        environment: Deployment environment
        metrics: Performance metrics
        
    Returns:
        Integration results
    """
    try:
        integration = MonitoringIntegration()
        deployment_record = integration.record_deployment(model_name, version, environment, metrics)
        health_check = integration.check_model_health(model_name, version)
        
        return {
            "deployment": deployment_record,
            "health_check": health_check,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error integrating monitoring with pipeline: {e}")
        return {
            "status": "error",
            "message": str(e)
        }


if __name__ == "__main__":
    # Example usage
    from src.utils.monitoring.model_monitor import create_default_monitoring_config
    create_default_monitoring_config()
    
    # Simulate model deployment with monitoring integration
    result = integrate_monitoring_with_pipeline(
        model_name="cosmic_oracle_v1",
        version="0.1.0",
        environment="staging",
        metrics={
            "accuracy": 0.85,
            "precision": 0.82,
            "recall": 0.79,
            "f1": 0.80
        }
    )
    
    print(f"Integration result: {result['status']}")
    if result["status"] == "success":
        print(f"Model health: {result['health_check']['status']}")