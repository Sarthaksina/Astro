#!/usr/bin/env python3
"""
ThunderCompute GPU Monitoring Tool

This script provides utilities for monitoring GPU utilization and costs
on ThunderCompute instances, helping to optimize resource usage and
minimize expenses.
"""

import argparse
import json
import os
import sys
import time
from src.trading.unified_mcts import MCTS, MCTSPredictor
from src.trading.modular_hierarchical_rl import ModularHierarchicalRLAgent, MCTSStrategicPlanner, PPOTacticalExecutor
from datetime import datetime, timedelta
from pathlib import Path
import csv
import matplotlib.pyplot as plt

# Configuration
CONFIG_DIR = Path.home() / ".thundercompute"
CONFIG_FILE = CONFIG_DIR / "config.json"
MONITORING_DIR = Path("./monitoring_data")


class GPUMonitor:
    """Monitors GPU utilization and costs on ThunderCompute instances."""
    
    def __init__(self, instance_id, monitoring_dir=None):
        """Initialize the GPU monitor.
        
        Args:
            instance_id: ID of the ThunderCompute instance to monitor
            monitoring_dir: Directory to store monitoring data
        """
        self.instance_id = instance_id
        self.monitoring_dir = Path(monitoring_dir or MONITORING_DIR)
        self.monitoring_dir.mkdir(parents=True, exist_ok=True)
        
        # Create instance-specific directory
        self.instance_dir = self.monitoring_dir / instance_id
        self.instance_dir.mkdir(exist_ok=True)
        
        # Files for storing monitoring data
        self.utilization_file = self.instance_dir / "gpu_utilization.csv"
        self.cost_file = self.instance_dir / "cost_tracking.csv"
        
        # Initialize files if they don't exist
        self._initialize_files()
    
    def _initialize_files(self):
        """Initialize monitoring data files if they don't exist."""
        if not self.utilization_file.exists():
            with open(self.utilization_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "gpu_utilization", "memory_utilization", "temperature", "power_usage"])
        
        if not self.cost_file.exists():
            with open(self.cost_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "hourly_cost", "accumulated_cost", "estimated_daily_cost"])
    
    def collect_gpu_metrics(self):
        """Collect current GPU metrics from the instance.
        
        Returns:
            Dictionary containing GPU metrics
        """
        # In a real implementation, this would query the ThunderCompute API or use nvidia-smi
        # For demonstration, we'll generate simulated metrics
        import random
        
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "gpu_utilization": random.randint(0, 100),  # percentage
            "memory_utilization": random.randint(0, 100),  # percentage
            "temperature": random.randint(30, 85),  # Celsius
            "power_usage": random.randint(100, 350)  # Watts
        }
        
        return metrics
    
    def calculate_costs(self, gpu_type="RTX 4090", hours_running=None):
        """Calculate current and projected costs.
        
        Args:
            gpu_type: Type of GPU in the instance
            hours_running: How long the instance has been running (if None, calculated from logs)
            
        Returns:
            Dictionary containing cost metrics
        """
        # Hourly rates for different GPU types (example rates)
        hourly_rates = {
            "RTX 3080": 0.60,  # $0.60 per hour
            "RTX 4090": 1.20,  # $1.20 per hour
            "A100": 2.50,     # $2.50 per hour
        }
        
        if gpu_type not in hourly_rates:
            print(f"Warning: Unknown GPU type '{gpu_type}'. Using default rate.")
            hourly_rate = 1.00  # Default rate
        else:
            hourly_rate = hourly_rates[gpu_type]
        
        # Calculate hours running if not provided
        if hours_running is None:
            if self.utilization_file.exists():
                with open(self.utilization_file, 'r') as f:
                    reader = csv.reader(f)
                    next(reader)  # Skip header
                    try:
                        first_row = next(reader)
                        first_timestamp = datetime.fromisoformat(first_row[0])
                        hours_running = (datetime.now() - first_timestamp).total_seconds() / 3600
                    except StopIteration:
                        hours_running = 0  # No data yet
            else:
                hours_running = 0
        
        accumulated_cost = hourly_rate * hours_running
        estimated_daily_cost = hourly_rate * 24
        
        cost_data = {
            "timestamp": datetime.now().isoformat(),
            "hourly_cost": hourly_rate,
            "accumulated_cost": accumulated_cost,
            "estimated_daily_cost": estimated_daily_cost
        }
        
        return cost_data
    
    def record_metrics(self, gpu_type="RTX 4090"):
        """Record current GPU metrics and cost data."""
        # Collect GPU metrics
        gpu_metrics = self.collect_gpu_metrics()
        
        # Record GPU utilization
        with open(self.utilization_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                gpu_metrics["timestamp"],
                gpu_metrics["gpu_utilization"],
                gpu_metrics["memory_utilization"],
                gpu_metrics["temperature"],
                gpu_metrics["power_usage"]
            ])
        
        # Calculate and record costs
        cost_data = self.calculate_costs(gpu_type)
        with open(self.cost_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                cost_data["timestamp"],
                cost_data["hourly_cost"],
                cost_data["accumulated_cost"],
                cost_data["estimated_daily_cost"]
            ])
        
        return {**gpu_metrics, **cost_data}
    
    def generate_report(self, days=1):
        """Generate a monitoring report for the specified time period.
        
        Args:
            days: Number of days to include in the report
            
        Returns:
            Path to the generated report file
        """
        # Calculate start time for the report
        start_time = datetime.now() - timedelta(days=days)
        
        # Load utilization data
        utilization_data = []
        if self.utilization_file.exists():
            with open(self.utilization_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    timestamp = datetime.fromisoformat(row["timestamp"])
                    if timestamp >= start_time:
                        utilization_data.append({
                            "timestamp": timestamp,
                            "gpu_utilization": float(row["gpu_utilization"]),
                            "memory_utilization": float(row["memory_utilization"]),
                            "temperature": float(row["temperature"]),
                            "power_usage": float(row["power_usage"])
                        })
        
        # Load cost data
        cost_data = []
        if self.cost_file.exists():
            with open(self.cost_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    timestamp = datetime.fromisoformat(row["timestamp"])
                    if timestamp >= start_time:
                        cost_data.append({
                            "timestamp": timestamp,
                            "hourly_cost": float(row["hourly_cost"]),
                            "accumulated_cost": float(row["accumulated_cost"]),
                            "estimated_daily_cost": float(row["estimated_daily_cost"])
                        })
        
        # Generate report file
        report_file = self.instance_dir / f"report_{datetime.now().strftime('%Y%m%d')}.html"
        
        # In a real implementation, this would generate a proper HTML report with charts
        # For demonstration, we'll just create a simple text file
        with open(report_file, 'w') as f:
            f.write(f"<html>\n<head>\n<title>GPU Monitoring Report - {self.instance_id}</title>\n</head>\n<body>\n")
            f.write(f"<h1>GPU Monitoring Report - {self.instance_id}</h1>\n")
            f.write(f"<p>Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>\n")
            
            # Summary statistics
            if utilization_data:
                avg_gpu_util = sum(d["gpu_utilization"] for d in utilization_data) / len(utilization_data)
                avg_mem_util = sum(d["memory_utilization"] for d in utilization_data) / len(utilization_data)
                max_gpu_util = max(d["gpu_utilization"] for d in utilization_data)
                max_mem_util = max(d["memory_utilization"] for d in utilization_data)
                
                f.write("<h2>Utilization Summary</h2>\n")
                f.write("<table border='1'>\n")
                f.write("<tr><th>Metric</th><th>Average</th><th>Maximum</th></tr>\n")
                f.write(f"<tr><td>GPU Utilization</td><td>{avg_gpu_util:.2f}%</td><td>{max_gpu_util:.2f}%</td></tr>\n")
                f.write(f"<tr><td>Memory Utilization</td><td>{avg_mem_util:.2f}%</td><td>{max_mem_util:.2f}%</td></tr>\n")
                f.write("</table>\n")
            
            # Cost summary
            if cost_data:
                latest_cost = cost_data[-1]
                f.write("<h2>Cost Summary</h2>\n")
                f.write("<table border='1'>\n")
                f.write("<tr><th>Metric</th><th>Value</th></tr>\n")
                f.write(f"<tr><td>Hourly Cost</td><td>${latest_cost['hourly_cost']:.2f}</td></tr>\n")
                f.write(f"<tr><td>Accumulated Cost</td><td>${latest_cost['accumulated_cost']:.2f}</td></tr>\n")
                f.write(f"<tr><td>Estimated Daily Cost</td><td>${latest_cost['estimated_daily_cost']:.2f}</td></tr>\n")
                f.write("</table>\n")
            
            f.write("</body>\n</html>")
        
        print(f"Report generated: {report_file}")
        return report_file
    
    def start_monitoring(self, interval_seconds=60, gpu_type="RTX 4090"):
        """Start continuous monitoring of the instance.
        
        Args:
            interval_seconds: How often to collect metrics (in seconds)
            gpu_type: Type of GPU in the instance for cost calculation
        """
        print(f"Starting monitoring for instance {self.instance_id} (Press Ctrl+C to stop)")
        try:
            while True:
                metrics = self.record_metrics(gpu_type)
                print(f"[{datetime.now().strftime('%H:%M:%S')}] GPU: {metrics['gpu_utilization']}%, "  
                      f"Mem: {metrics['memory_utilization']}%, "  
                      f"Cost: ${metrics['accumulated_cost']:.2f}")
                time.sleep(interval_seconds)
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")
            # Generate a final report
            self.generate_report()


def main():
    parser = argparse.ArgumentParser(description="ThunderCompute GPU Monitoring Tool")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Monitor command
    monitor_parser = subparsers.add_parser("monitor", help="Start monitoring an instance")
    monitor_parser.add_argument("instance_id", help="ID of the instance to monitor")
    monitor_parser.add_argument("--interval", type=int, default=60, help="Monitoring interval in seconds")
    monitor_parser.add_argument("--gpu-type", default="RTX 4090", help="GPU type for cost calculation")
    
    # Report command
    report_parser = subparsers.add_parser("report", help="Generate a monitoring report")
    report_parser.add_argument("instance_id", help="ID of the instance to generate a report for")
    report_parser.add_argument("--days", type=int, default=1, help="Number of days to include in the report")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List monitored instances")
    
    args = parser.parse_args()
    
    if args.command == "monitor":
        monitor = GPUMonitor(args.instance_id)
        monitor.start_monitoring(args.interval, args.gpu_type)
    
    elif args.command == "report":
        monitor = GPUMonitor(args.instance_id)
        monitor.generate_report(args.days)
    
    elif args.command == "list":
        if not MONITORING_DIR.exists():
            print("No monitoring data found.")
            return
        
        instances = [d.name for d in MONITORING_DIR.iterdir() if d.is_dir()]
        if not instances:
            print("No monitored instances found.")
            return
        
        print("Monitored instances:")
        for instance in instances:
            print(f"- {instance}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()