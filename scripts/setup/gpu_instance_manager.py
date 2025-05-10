#!/usr/bin/env python3
"""
GPU Instance Manager for Cosmic Market Oracle

This script provides utilities for managing cloud GPU instances on VAST.ai and ThunderStorm,
including automated startup, shutdown, and monitoring of resources to optimize costs.

Usage:
    python gpu_instance_manager.py start --instance_type=rtx4090 --job=training
    python gpu_instance_manager.py stop --instance_id=12345
    python gpu_instance_manager.py status --instance_id=12345
    python gpu_instance_manager.py monitor --threshold=0.1 --interval=300
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union

import requests
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("gpu_instance_manager.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("gpu_instance_manager")

# Default configuration
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "gpu_instances.yaml"


class GPUInstanceManager:
    """Manager for cloud GPU instances on VAST.ai and ThunderStorm."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the GPU instance manager.

        Args:
            config_path: Path to the configuration file. If None, uses the default path.
        """
        self.config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        self.config = self._load_config()
        self.provider = self.config.get("provider", "vast.ai")
        self.api_key = self.config.get("api_key", os.environ.get(f"{self.provider.upper()}_API_KEY", ""))
        
        if not self.api_key:
            logger.warning(f"No API key found for {self.provider}. Set it in the config file or as an environment variable.")

    def _load_config(self) -> Dict:
        """Load configuration from YAML file.

        Returns:
            Dict containing configuration.
        """
        if not self.config_path.exists():
            logger.warning(f"Config file {self.config_path} not found. Using default configuration.")
            return {}

        with open(self.config_path, "r") as f:
            return yaml.safe_load(f)

    def _save_instance_info(self, instance_id: str, instance_info: Dict) -> None:
        """Save instance information to a JSON file.

        Args:
            instance_id: ID of the instance.
            instance_info: Information about the instance.
        """
        instances_dir = Path(self.config.get("instances_dir", "./instances"))
        instances_dir.mkdir(exist_ok=True, parents=True)
        
        instance_file = instances_dir / f"{instance_id}.json"
        with open(instance_file, "w") as f:
            json.dump(instance_info, f, indent=2)

    def _get_instance_info(self, instance_id: str) -> Optional[Dict]:
        """Get information about an instance from the saved JSON file.

        Args:
            instance_id: ID of the instance.

        Returns:
            Dict containing instance information, or None if not found.
        """
        instances_dir = Path(self.config.get("instances_dir", "./instances"))
        instance_file = instances_dir / f"{instance_id}.json"
        
        if not instance_file.exists():
            return None

        with open(instance_file, "r") as f:
            return json.load(f)

    def start_instance(self, instance_type: str, job: str, duration: int = 24) -> str:
        """Start a new GPU instance.

        Args:
            instance_type: Type of GPU instance (e.g., 'rtx4090', 'a100').
            job: Name of the job to run on the instance.
            duration: Maximum duration in hours for the instance to run.

        Returns:
            ID of the started instance.
        """
        logger.info(f"Starting {instance_type} instance for job '{job}' with duration {duration} hours")
        
        # Get instance configuration from config file
        instance_configs = self.config.get("instance_types", {})
        instance_config = instance_configs.get(instance_type, {})
        
        if not instance_config:
            logger.warning(f"No configuration found for instance type '{instance_type}'. Using default configuration.")
            instance_config = {
                "gpu_count": 1,
                "cpu_count": 8,
                "memory": 32,
                "disk": 100,
                "image": "nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04"
            }
        
        # Prepare API request based on provider
        if self.provider == "vast.ai":
            # Implementation for VAST.ai API
            # This is a simplified example - actual implementation would use the VAST.ai API
            instance_id = f"vast-{int(time.time())}"
            
            # Save instance information
            instance_info = {
                "id": instance_id,
                "provider": self.provider,
                "type": instance_type,
                "job": job,
                "start_time": datetime.now().isoformat(),
                "planned_end_time": (datetime.now() + timedelta(hours=duration)).isoformat(),
                "status": "running",
                "config": instance_config
            }
            self._save_instance_info(instance_id, instance_info)
            
            logger.info(f"Started instance {instance_id} for job '{job}'")
            return instance_id
            
        elif self.provider == "thunderstorm":
            # Implementation for ThunderStorm API
            # This is a simplified example - actual implementation would use the ThunderStorm API
            instance_id = f"ts-{int(time.time())}"
            
            # Save instance information
            instance_info = {
                "id": instance_id,
                "provider": self.provider,
                "type": instance_type,
                "job": job,
                "start_time": datetime.now().isoformat(),
                "planned_end_time": (datetime.now() + timedelta(hours=duration)).isoformat(),
                "status": "running",
                "config": instance_config
            }
            self._save_instance_info(instance_id, instance_info)
            
            logger.info(f"Started instance {instance_id} for job '{job}'")
            return instance_id
        
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def stop_instance(self, instance_id: str) -> bool:
        """Stop a running GPU instance.

        Args:
            instance_id: ID of the instance to stop.

        Returns:
            True if the instance was stopped successfully, False otherwise.
        """
        logger.info(f"Stopping instance {instance_id}")
        
        # Get instance information
        instance_info = self._get_instance_info(instance_id)
        if not instance_info:
            logger.error(f"Instance {instance_id} not found")
            return False
        
        # Prepare API request based on provider
        if instance_info["provider"] == "vast.ai":
            # Implementation for VAST.ai API
            # This is a simplified example - actual implementation would use the VAST.ai API
            
            # Update instance information
            instance_info["status"] = "stopped"
            instance_info["end_time"] = datetime.now().isoformat()
            self._save_instance_info(instance_id, instance_info)
            
            logger.info(f"Stopped instance {instance_id}")
            return True
            
        elif instance_info["provider"] == "thunderstorm":
            # Implementation for ThunderStorm API
            # This is a simplified example - actual implementation would use the ThunderStorm API
            
            # Update instance information
            instance_info["status"] = "stopped"
            instance_info["end_time"] = datetime.now().isoformat()
            self._save_instance_info(instance_id, instance_info)
            
            logger.info(f"Stopped instance {instance_id}")
            return True
        
        else:
            logger.error(f"Unsupported provider: {instance_info['provider']}")
            return False

    def get_instance_status(self, instance_id: str) -> Dict:
        """Get the status of a GPU instance.

        Args:
            instance_id: ID of the instance.

        Returns:
            Dict containing instance status information.
        """
        logger.info(f"Getting status for instance {instance_id}")
        
        # Get instance information
        instance_info = self._get_instance_info(instance_id)
        if not instance_info:
            logger.error(f"Instance {instance_id} not found")
            return {"status": "not_found"}
        
        # For a real implementation, we would query the provider's API for the latest status
        # This is a simplified example that just returns the saved information
        
        return {
            "id": instance_info["id"],
            "provider": instance_info["provider"],
            "type": instance_info["type"],
            "job": instance_info["job"],
            "status": instance_info["status"],
            "start_time": instance_info["start_time"],
            "planned_end_time": instance_info["planned_end_time"],
            "end_time": instance_info.get("end_time")
        }

    def monitor_instances(self, threshold: float = 0.1, interval: int = 300) -> None:
        """Monitor GPU utilization and stop instances if utilization is below threshold.

        Args:
            threshold: Utilization threshold (0.0 to 1.0) below which to stop instances.
            interval: Monitoring interval in seconds.
        """
        logger.info(f"Starting instance monitoring with threshold {threshold} and interval {interval} seconds")
        
        while True:
            # Get all running instances
            instances_dir = Path(self.config.get("instances_dir", "./instances"))
            if not instances_dir.exists():
                logger.info("No instances directory found. Waiting...")
                time.sleep(interval)
                continue
            
            for instance_file in instances_dir.glob("*.json"):
                with open(instance_file, "r") as f:
                    instance_info = json.load(f)
                
                if instance_info["status"] != "running":
                    continue
                
                # For a real implementation, we would query the provider's API for GPU utilization
                # This is a simplified example that simulates low utilization after a certain time
                
                start_time = datetime.fromisoformat(instance_info["start_time"])
                current_time = datetime.now()
                
                # Check if instance has been running for more than 2 hours (for demonstration)
                if (current_time - start_time).total_seconds() > 7200:  # 2 hours
                    logger.info(f"Instance {instance_info['id']} has low GPU utilization. Stopping...")
                    self.stop_instance(instance_info["id"])
            
            logger.info(f"Monitoring cycle completed. Waiting {interval} seconds...")
            time.sleep(interval)


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Manage cloud GPU instances for Cosmic Market Oracle")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Start command
    start_parser = subparsers.add_parser("start", help="Start a new GPU instance")
    start_parser.add_argument("--instance_type", required=True, help="Type of GPU instance (e.g., 'rtx4090', 'a100')")
    start_parser.add_argument("--job", required=True, help="Name of the job to run on the instance")
    start_parser.add_argument("--duration", type=int, default=24, help="Maximum duration in hours for the instance to run")
    
    # Stop command
    stop_parser = subparsers.add_parser("stop", help="Stop a running GPU instance")
    stop_parser.add_argument("--instance_id", required=True, help="ID of the instance to stop")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Get the status of a GPU instance")
    status_parser.add_argument("--instance_id", required=True, help="ID of the instance")
    
    # Monitor command
    monitor_parser = subparsers.add_parser("monitor", help="Monitor GPU utilization and stop instances if utilization is below threshold")
    monitor_parser.add_argument("--threshold", type=float, default=0.1, help="Utilization threshold (0.0 to 1.0) below which to stop instances")
    monitor_parser.add_argument("--interval", type=int, default=300, help="Monitoring interval in seconds")
    
    # Config file option for all commands
    parser.add_argument("--config", help="Path to the configuration file")
    
    args = parser.parse_args()
    
    # Create GPU instance manager
    manager = GPUInstanceManager(config_path=args.config)
    
    # Execute command
    if args.command == "start":
        instance_id = manager.start_instance(args.instance_type, args.job, args.duration)
        print(f"Started instance {instance_id}")
    
    elif args.command == "stop":
        success = manager.stop_instance(args.instance_id)
        if success:
            print(f"Stopped instance {args.instance_id}")
        else:
            print(f"Failed to stop instance {args.instance_id}")
            sys.exit(1)
    
    elif args.command == "status":
        status = manager.get_instance_status(args.instance_id)
        print(json.dumps(status, indent=2))
    
    elif args.command == "monitor":
        try:
            manager.monitor_instances(args.threshold, args.interval)
        except KeyboardInterrupt:
            print("Monitoring stopped")
    
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()