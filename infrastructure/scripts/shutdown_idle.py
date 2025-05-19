#!/usr/bin/env python3
"""
Automatic Shutdown for Idle ThunderCompute Instances

This script monitors GPU utilization and automatically shuts down instances
that have been idle for a specified period, helping to minimize costs.
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
import subprocess
import signal

# Configuration
DEFAULT_IDLE_THRESHOLD = 30  # minutes
DEFAULT_CHECK_INTERVAL = 5   # minutes
DEFAULT_UTILIZATION_THRESHOLD = 10  # percentage


class IdleShutdown:
    """Monitors GPU utilization and shuts down idle instances."""
    
    def __init__(self, idle_threshold_minutes=DEFAULT_IDLE_THRESHOLD, 
                 utilization_threshold=DEFAULT_UTILIZATION_THRESHOLD,
                 check_interval_minutes=DEFAULT_CHECK_INTERVAL):
        """Initialize the idle shutdown monitor.
        
        Args:
            idle_threshold_minutes: Minutes of idle time before shutdown
            utilization_threshold: GPU utilization percentage below which is considered idle
            check_interval_minutes: How often to check utilization
        """
        self.idle_threshold_minutes = idle_threshold_minutes
        self.utilization_threshold = utilization_threshold
        self.check_interval_minutes = check_interval_minutes
        self.idle_start_time = None
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)
        
        # Create log file
        self.log_file = Path("/var/log/idle_shutdown.log")
        if not self.log_file.parent.exists():
            self.log_file = Path("./idle_shutdown.log")
    
    def _log(self, message):
        """Log a message to the log file and stdout."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        with open(self.log_file, 'a') as f:
            f.write(log_message + "\n")
    
    def _handle_signal(self, signum, frame):
        """Handle termination signals."""
        self._log(f"Received signal {signum}. Shutting down idle monitor.")
        sys.exit(0)
    
    def get_gpu_utilization(self):
        """Get current GPU utilization.
        
        Returns:
            Float representing GPU utilization percentage
        """
        try:
            # Try to use nvidia-smi to get actual GPU utilization
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                check=True
            )
            # Parse the output - average if multiple GPUs
            utilizations = [float(u.strip()) for u in result.stdout.strip().split("\n")]
            return sum(utilizations) / len(utilizations) if utilizations else 0
        except (subprocess.SubprocessError, FileNotFoundError):
            # If nvidia-smi fails or isn't available, use a simulated approach
            self._log("Warning: Could not get GPU utilization from nvidia-smi. Using simulated approach.")
            
            # Check if there are any GPU-intensive processes running
            try:
                result = subprocess.run(
                    ["ps", "-eo", "%cpu,command"],
                    capture_output=True,
                    text=True
                )
                # Look for common GPU processes
                gpu_processes = ["python", "pytorch", "tensorflow", "jupyter"]
                high_cpu_processes = []
                
                for line in result.stdout.strip().split("\n"):
                    parts = line.strip().split(None, 1)
                    if len(parts) >= 2:
                        cpu_pct = float(parts[0])
                        command = parts[1]
                        
                        if cpu_pct > 10 and any(proc in command.lower() for proc in gpu_processes):
                            high_cpu_processes.append((cpu_pct, command))
                
                # If we have high CPU usage on likely GPU processes, assume GPU is being used
                if high_cpu_processes:
                    return 50.0  # Assume moderate utilization
                return 0.0  # Assume idle
            except Exception as e:
                self._log(f"Error checking processes: {e}")
                return 0.0  # Assume idle in case of error
    
    def should_shutdown(self):
        """Check if the instance should be shut down based on idle time.
        
        Returns:
            Boolean indicating whether to shut down
        """
        utilization = self.get_gpu_utilization()
        self._log(f"Current GPU utilization: {utilization:.2f}%")
        
        # Check if GPU is idle
        if utilization <= self.utilization_threshold:
            # If this is the first idle check, record the start time
            if self.idle_start_time is None:
                self.idle_start_time = datetime.now()
                self._log(f"Instance became idle at {self.idle_start_time}")
            
            # Check if we've been idle long enough to shut down
            idle_duration = datetime.now() - self.idle_start_time
            idle_minutes = idle_duration.total_seconds() / 60
            
            self._log(f"Instance has been idle for {idle_minutes:.2f} minutes")
            
            if idle_minutes >= self.idle_threshold_minutes:
                self._log(f"Idle threshold of {self.idle_threshold_minutes} minutes reached")
                return True
        else:
            # If GPU is active, reset idle timer
            if self.idle_start_time is not None:
                self._log("Instance is no longer idle. Resetting idle timer.")
                self.idle_start_time = None
        
        return False
    
    def shutdown_instance(self):
        """Shut down the instance."""
        self._log("Initiating instance shutdown...")
        
        # Save any important logs or data before shutdown
        self._log("Saving logs and data before shutdown...")
        
        # In a real implementation, this would use the ThunderCompute API or system shutdown
        # For demonstration, we'll just log the shutdown
        self._log("SHUTDOWN: Instance would be shut down here in a real implementation")
        
        # On a real system, you would use one of these:
        # subprocess.run(["sudo", "shutdown", "-h", "now"])
        # or use the cloud provider's API to terminate the instance
    
    def run(self):
        """Run the idle shutdown monitor continuously."""
        self._log(f"Starting idle shutdown monitor (threshold: {self.idle_threshold_minutes} minutes, "  
                 f"utilization threshold: {self.utilization_threshold}%)")
        
        try:
            while True:
                if self.should_shutdown():
                    self.shutdown_instance()
                    break
                
                # Wait for the next check interval
                time.sleep(self.check_interval_minutes * 60)
        except Exception as e:
            self._log(f"Error in idle shutdown monitor: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(description="ThunderCompute Idle Instance Shutdown")
    parser.add_argument("--threshold", type=int, default=DEFAULT_IDLE_THRESHOLD,
                        help=f"Idle time threshold in minutes (default: {DEFAULT_IDLE_THRESHOLD})")
    parser.add_argument("--utilization", type=float, default=DEFAULT_UTILIZATION_THRESHOLD,
                        help=f"GPU utilization threshold percentage (default: {DEFAULT_UTILIZATION_THRESHOLD})")
    parser.add_argument("--interval", type=int, default=DEFAULT_CHECK_INTERVAL,
                        help=f"Check interval in minutes (default: {DEFAULT_CHECK_INTERVAL})")
    
    args = parser.parse_args()
    
    shutdown_monitor = IdleShutdown(
        idle_threshold_minutes=args.threshold,
        utilization_threshold=args.utilization,
        check_interval_minutes=args.interval
    )
    
    shutdown_monitor.run()


if __name__ == "__main__":
    main()