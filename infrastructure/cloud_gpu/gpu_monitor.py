"""GPU Monitoring System for Cosmic Market Oracle.

This module provides functionality for monitoring GPU utilization, memory usage,
and cost tracking for cloud GPU instances used in the Cosmic Market Oracle project.
"""

import os
import json
import time
# import logging # Removed
import yaml
from src.utils.logger import get_logger # Added
from src.trading.unified_mcts import MCTS, MCTSPredictor
from src.trading.modular_hierarchical_rl import ModularHierarchicalRLAgent, MCTSStrategicPlanner, PPOTacticalExecutor
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta
import threading
import pandas as pd
import matplotlib.pyplot as plt
from prometheus_client import start_http_server, Gauge, Counter

# Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s') # Removed
logger = get_logger(__name__) # Changed

# Define Prometheus metrics
GPU_UTILIZATION = Gauge('cosmic_oracle_gpu_utilization', 'GPU Utilization Percentage', ['instance_id', 'gpu_index'])
GPU_MEMORY_USAGE = Gauge('cosmic_oracle_gpu_memory_usage', 'GPU Memory Usage in MB', ['instance_id', 'gpu_index'])
GPU_POWER_USAGE = Gauge('cosmic_oracle_gpu_power_usage', 'GPU Power Usage in Watts', ['instance_id', 'gpu_index'])
GPU_TEMPERATURE = Gauge('cosmic_oracle_gpu_temperature', 'GPU Temperature in Celsius', ['instance_id', 'gpu_index'])
INSTANCE_COST = Counter('cosmic_oracle_instance_cost', 'Instance Cost in USD', ['instance_id', 'instance_type'])

class GPUMonitor:
    """GPU monitoring system for cloud instances.
    
    This class provides functionality for:
    1. Monitoring GPU utilization, memory usage, and temperature
    2. Tracking instance costs over time
    3. Generating utilization reports and visualizations
    4. Exporting metrics to Prometheus for dashboard integration
    5. Alerting on low utilization or high costs
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None, 
                 monitoring_dir: Optional[Union[str, Path]] = None):
        """Initialize the GPU monitoring system.
        
        Args:
            config_path: Path to GPU instance configuration file
            monitoring_dir: Directory to store monitoring data
        """
        self.config_path = Path(config_path) if config_path else Path("config/gpu_instances.yaml")
        self.monitoring_dir = Path(monitoring_dir) if monitoring_dir else Path("monitoring/gpu")
        
        # Create monitoring directory if it doesn't exist
        self.monitoring_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize monitoring data
        self.monitoring_data: Dict[str, Dict[str, Any]] = {}
        self.cost_data: Dict[str, List[Dict[str, Any]]] = {}
        
        # Load existing monitoring data
        self._load_monitoring_data()
        
        # Monitoring thread
        self.monitoring_thread = None
        self.stop_monitoring = threading.Event()
        
        logger.info(f"Initialized GPU monitoring system")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load GPU instance configuration.
        
        Returns:
            Configuration dictionary
        """
        try:
            if not self.config_path.exists():
                logger.warning(f"Config file {self.config_path} not found")
                return {}
            
            with open(self.config_path, "r") as f:
                config = yaml.safe_load(f)
            
            logger.info(f"Loaded configuration from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {}
    
    def _load_monitoring_data(self) -> None:
        """Load existing monitoring data from disk."""
        try:
            # Load instance monitoring data
            monitoring_file = self.monitoring_dir / "monitoring_data.json"
            if monitoring_file.exists():
                with open(monitoring_file, "r") as f:
                    self.monitoring_data = json.load(f)
                logger.info(f"Loaded monitoring data for {len(self.monitoring_data)} instances")
            
            # Load cost data
            cost_file = self.monitoring_dir / "cost_data.json"
            if cost_file.exists():
                with open(cost_file, "r") as f:
                    self.cost_data = json.load(f)
                logger.info(f"Loaded cost data for {len(self.cost_data)} instances")
        except Exception as e:
            logger.error(f"Error loading monitoring data: {e}")
    
    def _save_monitoring_data(self) -> None:
        """Save monitoring data to disk."""
        try:
            # Save instance monitoring data
            monitoring_file = self.monitoring_dir / "monitoring_data.json"
            with open(monitoring_file, "w") as f:
                json.dump(self.monitoring_data, f, indent=2)
            
            # Save cost data
            cost_file = self.monitoring_dir / "cost_data.json"
            with open(cost_file, "w") as f:
                json.dump(self.cost_data, f, indent=2)
            
            logger.debug(f"Saved monitoring data")
        except Exception as e:
            logger.error(f"Error saving monitoring data: {e}")
    
    def start_monitoring(self, interval: int = 300) -> None:
        """Start the GPU monitoring thread.
        
        Args:
            interval: Monitoring interval in seconds (default: 5 minutes)
        """
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            logger.warning("Monitoring thread is already running")
            return
        
        self.stop_monitoring.clear()
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            daemon=True
        )
        self.monitoring_thread.start()
        
        logger.info(f"Started GPU monitoring with {interval}s interval")
    
    def stop_monitoring_thread(self) -> None:
        """Stop the GPU monitoring thread."""
        if not self.monitoring_thread or not self.monitoring_thread.is_alive():
            logger.warning("Monitoring thread is not running")
            return
        
        self.stop_monitoring.set()
        self.monitoring_thread.join(timeout=10)
        
        if self.monitoring_thread.is_alive():
            logger.warning("Monitoring thread did not stop gracefully")
        else:
            logger.info("Stopped GPU monitoring thread")
    
    def _monitoring_loop(self, interval: int) -> None:
        """Main monitoring loop.
        
        Args:
            interval: Monitoring interval in seconds
        """
        while not self.stop_monitoring.is_set():
            try:
                # Get list of active instances from the instances directory
                instances_dir = Path(self.config.get("instances_dir", "./instances"))
                if instances_dir.exists():
                    for file_path in instances_dir.glob("*.json"):
                        try:
                            with open(file_path, "r") as f:
                                instance_info = json.load(f)
                                instance_id = instance_info.get("id")
                                
                                if instance_id and instance_info.get("status") == "running":
                                    self._monitor_instance(instance_id, instance_info)
                        except Exception as e:
                            logger.error(f"Error monitoring instance {file_path.stem}: {e}")
                
                # Save monitoring data
                self._save_monitoring_data()
                
                # Check for underutilized instances
                self._check_utilization_alerts()
                
                # Update Prometheus metrics
                self._update_prometheus_metrics()
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
            
            # Sleep until next interval
            for _ in range(interval):
                if self.stop_monitoring.is_set():
                    break
                time.sleep(1)
    
    def _monitor_instance(self, instance_id: str, instance_info: Dict[str, Any]) -> None:
        """Monitor a specific GPU instance.
        
        Args:
            instance_id: Instance ID
            instance_info: Instance information dictionary
        """
        # This is a placeholder for actual GPU monitoring
        # In a real implementation, this would connect to the instance and get GPU metrics
        # using tools like nvidia-smi or cloud provider APIs
        
        # For demonstration purposes, we'll generate some sample metrics
        timestamp = datetime.now().isoformat()
        instance_type = instance_info.get("type", "unknown")
        
        # Get number of GPUs for this instance type
        instance_type_config = self.config.get("instance_types", {}).get(instance_type, {})
        gpu_count = instance_type_config.get("gpu_count", 1)
        
        # Generate metrics for each GPU
        gpu_metrics = []
        for i in range(gpu_count):
            # In a real implementation, these would be actual metrics from nvidia-smi
            # For now, we'll generate random values for demonstration
            import random
            utilization = random.uniform(50, 95)  # 50-95% utilization
            memory_usage = random.uniform(1000, 16000)  # 1-16 GB
            power_usage = random.uniform(100, 300)  # 100-300 watts
            temperature = random.uniform(50, 85)  # 50-85°C
            
            gpu_metrics.append({
                "gpu_index": i,
                "utilization": utilization,
                "memory_usage": memory_usage,
                "power_usage": power_usage,
                "temperature": temperature
            })
        
        # Calculate cost
        hourly_rate = instance_info.get("instance_data", {}).get("dph_total", 0.0)
        cost_increment = hourly_rate * (interval / 3600)  # Cost for this monitoring interval
        
        # Store monitoring data
        if instance_id not in self.monitoring_data:
            self.monitoring_data[instance_id] = {
                "instance_type": instance_type,
                "metrics": {}
            }
        
        self.monitoring_data[instance_id]["metrics"][timestamp] = {
            "gpu_metrics": gpu_metrics,
            "hourly_rate": hourly_rate
        }
        
        # Store cost data
        if instance_id not in self.cost_data:
            self.cost_data[instance_id] = []
        
        self.cost_data[instance_id].append({
            "timestamp": timestamp,
            "cost_increment": cost_increment,
            "hourly_rate": hourly_rate
        })
        
        logger.debug(f"Monitored instance {instance_id} ({instance_type})")
    
    def _check_utilization_alerts(self) -> None:
        """Check for underutilized instances and generate alerts."""
        # Get monitoring configuration
        monitoring_config = self.config.get("monitoring", {})
        utilization_threshold = monitoring_config.get("utilization_threshold", 0.1)
        
        for instance_id, data in self.monitoring_data.items():
            # Skip instances with no metrics
            if not data.get("metrics"):
                continue
            
            # Get latest metrics
            latest_timestamp = max(data["metrics"].keys())
            latest_metrics = data["metrics"][latest_timestamp]
            
            # Calculate average GPU utilization
            gpu_metrics = latest_metrics.get("gpu_metrics", [])
            if not gpu_metrics:
                continue
            
            avg_utilization = sum(gpu["utilization"] for gpu in gpu_metrics) / len(gpu_metrics) / 100.0
            
            # Check if utilization is below threshold
            if avg_utilization < utilization_threshold:
                logger.warning(f"Alert: Instance {instance_id} has low utilization ({avg_utilization:.2%})")
                
                # In a real implementation, this would send alerts via email, Slack, etc.
    
    def _update_prometheus_metrics(self) -> None:
        """Update Prometheus metrics for all monitored instances."""
        for instance_id, data in self.monitoring_data.items():
            # Skip instances with no metrics
            if not data.get("metrics"):
                continue
            
            # Get latest metrics
            latest_timestamp = max(data["metrics"].keys())
            latest_metrics = data["metrics"][latest_timestamp]
            
            # Update GPU metrics
            for gpu in latest_metrics.get("gpu_metrics", []):
                gpu_index = gpu.get("gpu_index", 0)
                
                GPU_UTILIZATION.labels(instance_id=instance_id, gpu_index=gpu_index).set(gpu.get("utilization", 0))
                GPU_MEMORY_USAGE.labels(instance_id=instance_id, gpu_index=gpu_index).set(gpu.get("memory_usage", 0))
                GPU_POWER_USAGE.labels(instance_id=instance_id, gpu_index=gpu_index).set(gpu.get("power_usage", 0))
                GPU_TEMPERATURE.labels(instance_id=instance_id, gpu_index=gpu_index).set(gpu.get("temperature", 0))
            
            # Update cost metrics
            if instance_id in self.cost_data and self.cost_data[instance_id]:
                latest_cost = self.cost_data[instance_id][-1]
                INSTANCE_COST.labels(
                    instance_id=instance_id, 
                    instance_type=data.get("instance_type", "unknown")
                ).inc(latest_cost.get("cost_increment", 0))
    
    def start_prometheus_server(self, port: int = 8000) -> None:
        """Start Prometheus metrics server.
        
        Args:
            port: Port to expose metrics on
        """
        try:
            start_http_server(port)
            logger.info(f"Started Prometheus metrics server on port {port}")
        except Exception as e:
            logger.error(f"Failed to start Prometheus metrics server: {e}")
    
    def get_total_cost(self, instance_id: Optional[str] = None) -> float:
        """Get total cost for an instance or all instances.
        
        Args:
            instance_id: Instance ID (optional, if None, returns total for all instances)
            
        Returns:
            Total cost in USD
        """
        total_cost = 0.0
        
        if instance_id:
            # Get cost for specific instance
            if instance_id in self.cost_data:
                for cost_entry in self.cost_data[instance_id]:
                    total_cost += cost_entry.get("cost_increment", 0.0)
        else:
            # Get cost for all instances
            for instance_costs in self.cost_data.values():
                for cost_entry in instance_costs:
                    total_cost += cost_entry.get("cost_increment", 0.0)
        
        return total_cost
    
    def generate_cost_report(self, output_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """Generate a cost report for all instances.
        
        Args:
            output_path: Path to write the report to (optional)
            
        Returns:
            Cost report dictionary
        """
        report = {
            "generated_at": datetime.now().isoformat(),
            "total_cost": self.get_total_cost(),
            "instances": {}
        }
        
        # Generate report for each instance
        for instance_id, instance_costs in self.cost_data.items():
            if not instance_costs:
                continue
            
            instance_type = self.monitoring_data.get(instance_id, {}).get("instance_type", "unknown")
            instance_total = sum(entry.get("cost_increment", 0.0) for entry in instance_costs)
            
            # Get first and last timestamp
            first_timestamp = instance_costs[0].get("timestamp")
            last_timestamp = instance_costs[-1].get("timestamp")
            
            # Calculate duration
            if first_timestamp and last_timestamp:
                start_time = datetime.fromisoformat(first_timestamp)
                end_time = datetime.fromisoformat(last_timestamp)
                duration_hours = (end_time - start_time).total_seconds() / 3600
            else:
                duration_hours = 0
            
            report["instances"][instance_id] = {
                "instance_type": instance_type,
                "total_cost": instance_total,
                "duration_hours": duration_hours,
                "hourly_rate": instance_costs[-1].get("hourly_rate", 0.0) if instance_costs else 0.0,
                "first_timestamp": first_timestamp,
                "last_timestamp": last_timestamp
            }
        
        # Write report to file if output path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, "w") as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Generated cost report at {output_path}")
        
        return report
    
    def generate_utilization_report(self, instance_id: str, 
                                  start_time: Optional[datetime] = None,
                                  end_time: Optional[datetime] = None,
                                  output_path: Optional[Union[str, Path]] = None) -> Optional[Dict[str, Any]]:
        """Generate a utilization report for an instance.
        
        Args:
            instance_id: Instance ID
            start_time: Start time for the report (optional)
            end_time: End time for the report (optional)
            output_path: Path to write the report to (optional)
            
        Returns:
            Utilization report dictionary
        """
        if instance_id not in self.monitoring_data:
            logger.error(f"No monitoring data for instance {instance_id}")
            return None
        
        instance_data = self.monitoring_data[instance_id]
        metrics = instance_data.get("metrics", {})
        
        if not metrics:
            logger.error(f"No metrics for instance {instance_id}")
            return None
        
        # Filter metrics by time range
        filtered_metrics = {}
        for timestamp, metric_data in metrics.items():
            metric_time = datetime.fromisoformat(timestamp)
            
            if start_time and metric_time < start_time:
                continue
            
            if end_time and metric_time > end_time:
                continue
            
            filtered_metrics[timestamp] = metric_data
        
        if not filtered_metrics:
            logger.error(f"No metrics in specified time range for instance {instance_id}")
            return None
        
        # Calculate average utilization for each GPU
        gpu_count = max(len(metric_data.get("gpu_metrics", [])) for metric_data in filtered_metrics.values())
        gpu_avg_utilization = [0.0] * gpu_count
        gpu_avg_memory = [0.0] * gpu_count
        gpu_avg_power = [0.0] * gpu_count
        gpu_avg_temperature = [0.0] * gpu_count
        
        for metric_data in filtered_metrics.values():
            for gpu in metric_data.get("gpu_metrics", []):
                gpu_index = gpu.get("gpu_index", 0)
                if gpu_index < gpu_count:
                    gpu_avg_utilization[gpu_index] += gpu.get("utilization", 0.0)
                    gpu_avg_memory[gpu_index] += gpu.get("memory_usage", 0.0)
                    gpu_avg_power[gpu_index] += gpu.get("power_usage", 0.0)
                    gpu_avg_temperature[gpu_index] += gpu.get("temperature", 0.0)
        
        # Calculate averages
        metric_count = len(filtered_metrics)
        for i in range(gpu_count):
            gpu_avg_utilization[i] /= metric_count
            gpu_avg_memory[i] /= metric_count
            gpu_avg_power[i] /= metric_count
            gpu_avg_temperature[i] /= metric_count
        
        # Create report
        report = {
            "instance_id": instance_id,
            "instance_type": instance_data.get("instance_type", "unknown"),
            "start_time": start_time.isoformat() if start_time else min(filtered_metrics.keys()),
            "end_time": end_time.isoformat() if end_time else max(filtered_metrics.keys()),
            "metric_count": metric_count,
            "gpu_count": gpu_count,
            "gpu_metrics": []
        }
        
        for i in range(gpu_count):
            report["gpu_metrics"].append({
                "gpu_index": i,
                "avg_utilization": gpu_avg_utilization[i],
                "avg_memory_usage": gpu_avg_memory[i],
                "avg_power_usage": gpu_avg_power[i],
                "avg_temperature": gpu_avg_temperature[i]
            })
        
        # Write report to file if output path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, "w") as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Generated utilization report at {output_path}")
        
        return report
    
    def plot_utilization(self, instance_id: str, output_path: Optional[Union[str, Path]] = None) -> None:
        """Plot GPU utilization over time for an instance.
        
        Args:
            instance_id: Instance ID
            output_path: Path to save the plot (optional)
        """
        if instance_id not in self.monitoring_data:
            logger.error(f"No monitoring data for instance {instance_id}")
            return
        
        instance_data = self.monitoring_data[instance_id]
        metrics = instance_data.get("metrics", {})
        
        if not metrics:
            logger.error(f"No metrics for instance {instance_id}")
            return
        
        # Extract timestamps and utilization data
        timestamps = []
        utilization_data = []
        
        for timestamp, metric_data in sorted(metrics.items()):
            timestamps.append(datetime.fromisoformat(timestamp))
            
            # Calculate average utilization across all GPUs
            gpu_metrics = metric_data.get("gpu_metrics", [])
            if gpu_metrics:
                avg_util = sum(gpu.get("utilization", 0.0) for gpu in gpu_metrics) / len(gpu_metrics)
                utilization_data.append(avg_util)
            else:
                utilization_data.append(0.0)
        
        # Create plot
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, utilization_data)
        plt.title(f"GPU Utilization for Instance {instance_id}")
        plt.xlabel("Time")
        plt.ylabel("Utilization (%)")
        plt.grid(True)
        plt.tight_layout()
        
        # Save or show plot
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path)
            logger.info(f"Saved utilization plot to {output_path}")
        else:
            plt.show()
        
        plt.close()


def get_gpu_monitor(config_path: Optional[Union[str, Path]] = None,
                   monitoring_dir: Optional[Union[str, Path]] = None) -> GPUMonitor:
    """Get a GPU monitor instance.
    
    Args:
        config_path: Path to GPU instance configuration file
        monitoring_dir: Directory to store monitoring data
        
    Returns:
        GPUMonitor instance
    """
    return GPUMonitor(config_path, monitoring_dir)


def start_monitoring_server(port: int = 8000) -> None:
    """Start the GPU monitoring server.
    
    Args:
        port: Port to expose metrics on
    """
    monitor = get_gpu_monitor()
    monitor.start_prometheus_server(port)
    monitor.start_monitoring()
    
    logger.info(f"Started GPU monitoring server on port {port}")
    
    try:
        # Keep the server running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopping GPU monitoring server")
        monitor.stop_monitoring_thread()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="GPU Monitoring System")
    parser.add_argument("--port", type=int, default=8000, help="Port to expose metrics on")
    parser.add_argument("--config", type=str, help="Path to GPU instance configuration file")
    parser.add_argument("--monitoring-dir", type=str, help="Directory to store monitoring data")
    
    args = parser.parse_args()
    
    if args.config:
        config_path = Path(args.config)
    else:
        config_path = None
    
    if args.monitoring_dir:
        monitoring_dir = Path(args.monitoring_dir)
    else:
        monitoring_dir = None
    
    monitor = get_gpu_monitor(config_path, monitoring_dir)
    monitor.start_prometheus_server(args.port)
    monitor.start_monitoring()
    
    logger.info(f"Started GPU monitoring server on port {args.port}")
    
    try:
        # Keep the server running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopping GPU monitoring server")
        monitor.stop_monitoring_thread()