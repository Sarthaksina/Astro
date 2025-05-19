\"""Cloud GPU Instance Manager for Cosmic Market Oracle.

This module provides a unified interface for managing cloud GPU instances
across different providers (VAST.ai, ThunderCompute) with automated deployment,
monitoring, and cost optimization.
"""

import os
import json
import time
import logging
import yaml
import argparse
from src.trading.unified_mcts import MCTS, MCTSPredictor
from src.trading.modular_hierarchical_rl import ModularHierarchicalRLAgent, MCTSStrategicPlanner, PPOTacticalExecutor
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta
import subprocess
import threading

# Import provider managers
from infrastructure.cloud_gpu.vast_ai_manager import get_vast_ai_manager
from infrastructure.cloud_gpu.thunder_compute_manager import get_thunder_compute_manager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CloudGPUManager:
    """Unified manager for cloud GPU instances across providers.
    
    This class provides functionality for:
    1. Creating and managing GPU instances on different cloud providers
    2. Automating deployment of code and data to instances
    3. Scheduling workloads during lower-cost periods
    4. Implementing cost optimization strategies
    5. Providing a unified interface for different providers
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialize the cloud GPU manager.
        
        Args:
            config_path: Path to GPU instance configuration file
        """
        self.config_path = Path(config_path) if config_path else Path("config/gpu_instances.yaml")
        self.config = self._load_config()
        
        # Default provider
        self.default_provider = self.config.get("provider", "vast.ai")
        
        # Provider-specific managers
        self.providers = {}
        
        # Initialize provider managers
        self._init_providers()
        
        logger.info(f"Initialized Cloud GPU Manager with default provider: {self.default_provider}")
    
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
    
    def _init_providers(self) -> None:
        """Initialize provider-specific managers."""
        # Initialize VAST.ai manager
        if self.default_provider == "vast.ai":
            try:
                self.providers["vast.ai"] = get_vast_ai_manager(self.config_path)
                logger.info("Initialized VAST.ai manager")
            except Exception as e:
                logger.error(f"Failed to initialize VAST.ai manager: {e}")
        
        # Initialize ThunderCompute manager
        if self.default_provider == "thundercompute":
            try:
                self.providers["thundercompute"] = get_thunder_compute_manager(self.config_path)
                logger.info("Initialized ThunderCompute manager")
            except Exception as e:
                logger.error(f"Failed to initialize ThunderCompute manager: {e}")

    
    def create_instance(self, instance_type: str, provider: Optional[str] = None, 
                       docker_image: Optional[str] = None, onstart_script: Optional[str] = None) -> Optional[str]:
        """Create a new GPU instance.
        
        Args:
            instance_type: Type of instance to create (e.g., 'rtx4090', 'a100')
            provider: Cloud provider to use (defaults to default_provider)
            docker_image: Docker image to use
            onstart_script: Script to run on instance startup
            
        Returns:
            Instance ID if successful, None otherwise
        """
        provider = provider or self.default_provider
        
        if provider not in self.providers:
            logger.error(f"Provider {provider} not initialized")
            return None
        
        return self.providers[provider].create_instance(instance_type, docker_image, onstart_script)
    
    def stop_instance(self, instance_id: str, provider: Optional[str] = None) -> bool:
        """Stop a GPU instance.
        
        Args:
            instance_id: Instance ID
            provider: Cloud provider (defaults to default_provider)
            
        Returns:
            True if successful, False otherwise
        """
        provider = provider or self.default_provider
        
        if provider not in self.providers:
            logger.error(f"Provider {provider} not initialized")
            return False
        
        return self.providers[provider].stop_instance(instance_id)
    
    def destroy_instance(self, instance_id: str, provider: Optional[str] = None) -> bool:
        """Destroy a GPU instance.
        
        Args:
            instance_id: Instance ID
            provider: Cloud provider (defaults to default_provider)
            
        Returns:
            True if successful, False otherwise
        """
        provider = provider or self.default_provider
        
        if provider not in self.providers:
            logger.error(f"Provider {provider} not initialized")
            return False
        
        return self.providers[provider].destroy_instance(instance_id)
    
    def deploy_code(self, instance_id: str, git_repo: Optional[str] = None, 
                   local_path: Optional[Union[str, Path]] = None, 
                   provider: Optional[str] = None) -> bool:
        """Deploy code to a GPU instance.
        
        Args:
            instance_id: Instance ID
            git_repo: Git repository URL
            local_path: Local path to code
            provider: Cloud provider (defaults to default_provider)
            
        Returns:
            True if successful, False otherwise
        """
        provider = provider or self.default_provider
        
        if provider not in self.providers:
            logger.error(f"Provider {provider} not initialized")
            return False
        
        return self.providers[provider].deploy_code(instance_id, git_repo, local_path)
    
    def sync_data(self, instance_id: str, local_path: Union[str, Path], remote_path: str, 
                 direction: str = "upload", provider: Optional[str] = None) -> bool:
        """Sync data between local and remote instance.
        
        Args:
            instance_id: Instance ID
            local_path: Local path
            remote_path: Remote path
            direction: 'upload' or 'download'
            provider: Cloud provider (defaults to default_provider)
            
        Returns:
            True if successful, False otherwise
        """
        provider = provider or self.default_provider
        
        if provider not in self.providers:
            logger.error(f"Provider {provider} not initialized")
            return False
        
        return self.providers[provider].sync_data(instance_id, local_path, remote_path, direction)
    
    def get_instance_status(self, instance_id: str, provider: Optional[str] = None) -> Dict[str, Any]:
        """Get status of a GPU instance.
        
        Args:
            instance_id: Instance ID
            provider: Cloud provider (defaults to default_provider)
            
        Returns:
            Instance status information
        """
        provider = provider or self.default_provider
        
        if provider not in self.providers:
            logger.error(f"Provider {provider} not initialized")
            return {"status": "error", "error": f"Provider {provider} not initialized"}
        
        return self.providers[provider].get_instance_status(instance_id)
    
    def get_cost_estimate(self, instance_id: str, provider: Optional[str] = None) -> float:
        """Get estimated cost for an instance.
        
        Args:
            instance_id: Instance ID
            provider: Cloud provider (defaults to default_provider)
            
        Returns:
            Estimated cost in USD
        """
        provider = provider or self.default_provider
        
        if provider not in self.providers:
            logger.error(f"Provider {provider} not initialized")
            return 0.0
        
        return self.providers[provider].get_cost_estimate(instance_id)
    
    def get_total_cost(self, provider: Optional[str] = None) -> float:
        """Get total estimated cost for all instances.
        
        Args:
            provider: Cloud provider (defaults to default_provider)
            
        Returns:
            Total estimated cost in USD
        """
        provider = provider or self.default_provider
        
        if provider not in self.providers:
            logger.error(f"Provider {provider} not initialized")
            return 0.0
        
        return self.providers[provider].get_total_cost()
    
    def schedule_workload(self, instance_type: str, docker_image: str, 
                         command: str, start_time: Optional[datetime] = None,
                         provider: Optional[str] = None) -> Optional[str]:
        """Schedule a workload to run at a specific time or during low-cost period.
        
        Args:
            instance_type: Type of instance to create
            docker_image: Docker image to use
            command: Command to run
            start_time: Specific time to start (if None, will use preferred hours)
            provider: Cloud provider (defaults to default_provider)
            
        Returns:
            Instance ID if successful, None otherwise
        """
        provider = provider or self.default_provider
        
        # If no specific start time, find the next preferred time slot
        if not start_time:
            start_time = self._find_next_preferred_time()
        
        # Calculate time until start
        now = datetime.now()
        wait_seconds = max(0, (start_time - now).total_seconds())
        
        if wait_seconds > 0:
            logger.info(f"Scheduling workload to start at {start_time} (in {wait_seconds/3600:.2f} hours)")
            
            # Start a timer thread to create the instance at the scheduled time
            threading.Timer(wait_seconds, self._start_scheduled_workload, 
                           args=[instance_type, docker_image, command, provider]).start()
            
            return f"scheduled_{int(time.time())}"
        else:
            # Start immediately
            return self._start_scheduled_workload(instance_type, docker_image, command, provider)
    
    def _start_scheduled_workload(self, instance_type: str, docker_image: str, 
                                command: str, provider: str) -> Optional[str]:
        """Start a scheduled workload.
        
        Args:
            instance_type: Type of instance to create
            docker_image: Docker image to use
            command: Command to run
            provider: Cloud provider
            
        Returns:
            Instance ID if successful, None otherwise
        """
        # Create startup script that runs the command
        onstart_script = f"""#!/bin/bash
{command}

# Shutdown after completion if using spot instance
if [ -f /vast/shutdown_on_complete ]; then
    echo "Shutting down instance after completion"
    /opt/vast/current/vast shutdown
fi
"""
        
        # Create the instance
        instance_id = self.create_instance(instance_type, provider, docker_image, onstart_script)
        
        if instance_id:
            logger.info(f"Started scheduled workload on instance {instance_id}")
        else:
            logger.error("Failed to start scheduled workload")
        
        return instance_id
    
    def _find_next_preferred_time(self) -> datetime:
        """Find the next preferred time slot based on scheduling configuration.
        
        Returns:
            Datetime of the next preferred time slot
        """
        now = datetime.now()
        scheduling_config = self.config.get("scheduling", {})
        preferred_hours = scheduling_config.get("preferred_hours", [])
        
        if not preferred_hours:
            # No preferred hours defined, start immediately
            return now
        
        # Get current day of week (0=Monday, 6=Sunday)
        current_day = now.weekday()
        current_day_name = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][current_day]
        
        # Find the next preferred time slot
        for slot in preferred_hours:
            days = slot.get("days", [])
            start_time_str = slot.get("start", "00:00")
            end_time_str = slot.get("end", "23:59")
            
            # Skip if current day not in preferred days
            if days and current_day_name not in days:
                continue
            
            # Parse start and end times
            start_hour, start_minute = map(int, start_time_str.split(":"))
            end_hour, end_minute = map(int, end_time_str.split(":"))
            
            # Create datetime objects for today's start and end times
            start_time = now.replace(hour=start_hour, minute=start_minute, second=0, microsecond=0)
            end_time = now.replace(hour=end_hour, minute=end_minute, second=0, microsecond=0)
            
            # Check if current time is within the preferred slot
            if start_time <= now <= end_time:
                # We're already in a preferred slot, start immediately
                return now
            elif now < start_time:
                # Preferred slot is later today
                return start_time
        
        # No preferred slot found today, check tomorrow
        tomorrow = now + timedelta(days=1)
        tomorrow = tomorrow.replace(hour=0, minute=0, second=0, microsecond=0)
        
        for slot in preferred_hours:
            days = slot.get("days", [])
            tomorrow_day_name = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][tomorrow.weekday()]
            
            # Skip if tomorrow not in preferred days
            if days and tomorrow_day_name not in days:
                continue
            
            # Parse start time
            start_hour, start_minute = map(int, slot.get("start", "00:00").split(":"))
            
            # Return tomorrow's start time
            return tomorrow.replace(hour=start_hour, minute=start_minute)
        
        # No preferred slot found tomorrow either, just start immediately
        return now
    
    def monitor_instances(self, provider: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """Monitor all instances and update their status.
        
        Args:
            provider: Cloud provider (defaults to default_provider)
            
        Returns:
            Dictionary of instance information
        """
        provider = provider or self.default_provider
        
        if provider not in self.providers:
            logger.error(f"Provider {provider} not initialized")
            return {}
        
        return self.providers[provider].monitor_instances()
    
    def optimize_resources(self, provider: Optional[str] = None) -> None:
        """Optimize resource usage by stopping underutilized instances.
        
        Args:
            provider: Cloud provider (defaults to default_provider)
        """
        provider = provider or self.default_provider
        
        if provider not in self.providers:
            logger.error(f"Provider {provider} not initialized")
            return
        
        # Monitor instances to check utilization and stop if needed
        self.monitor_instances(provider)
        
        logger.info(f"Optimized resources for provider {provider}")


def get_cloud_gpu_manager(config_path: Optional[Union[str, Path]] = None) -> CloudGPUManager:
    """Get a cloud GPU manager instance.
    
    Args:
        config_path: Path to GPU instance configuration file
        
    Returns:
        CloudGPUManager instance
    """
    return CloudGPUManager(config_path)


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Cloud GPU Instance Manager")
    parser.add_argument("--config", type=str, help="Path to GPU instance configuration file")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Create instance command
    create_parser = subparsers.add_parser("create", help="Create a new GPU instance")
    create_parser.add_argument("--type", type=str, required=True, help="Instance type (e.g., rtx4090, a100)")
    create_parser.add_argument("--provider", type=str, help="Cloud provider (vast.ai, thunderstorm)")
    create_parser.add_argument("--image", type=str, help="Docker image to use")
    create_parser.add_argument("--script", type=str, help="Path to startup script")
    
    # Stop instance command
    stop_parser = subparsers.add_parser("stop", help="Stop a GPU instance")
    stop_parser.add_argument("--id", type=str, required=True, help="Instance ID")
    stop_parser.add_argument("--provider", type=str, help="Cloud provider")
    
    # Destroy instance command
    destroy_parser = subparsers.add_parser("destroy", help="Destroy a GPU instance")
    destroy_parser.add_argument("--id", type=str, required=True, help="Instance ID")
    destroy_parser.add_argument("--provider", type=str, help="Cloud provider")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Get instance status")
    status_parser.add_argument("--id", type=str, help="Instance ID (if omitted, shows all instances)")
    status_parser.add_argument("--provider", type=str, help="Cloud provider")
    
    # Deploy code command
    deploy_parser = subparsers.add_parser("deploy", help="Deploy code to an instance")
    deploy_parser.add_argument("--id", type=str, required=True, help="Instance ID")
    deploy_parser.add_argument("--git", type=str, help="Git repository URL")
    deploy_parser.add_argument("--path", type=str, help="Local path to code")
    deploy_parser.add_argument("--provider", type=str, help="Cloud provider")
    
    # Sync data command
    sync_parser = subparsers.add_parser("sync", help="Sync data with an instance")
    sync_parser.add_argument("--id", type=str, required=True, help="Instance ID")
    sync_parser.add_argument("--local", type=str, required=True, help="Local path")
    sync_parser.add_argument("--remote", type=str, required=True, help="Remote path")
    sync_parser.add_argument("--direction", type=str, default="upload", choices=["upload", "download"], 
                           help="Sync direction (upload or download)")
    sync_parser.add_argument("--provider", type=str, help="Cloud provider")
    
    # Schedule workload command
    schedule_parser = subparsers.add_parser("schedule", help="Schedule a workload")
    schedule_parser.add_argument("--type", type=str, required=True, help="Instance type (e.g., rtx4090, a100)")
    schedule_parser.add_argument("--image", type=str, required=True, help="Docker image to use")
    schedule_parser.add_argument("--command", type=str, required=True, help="Command to run")
    schedule_parser.add_argument("--time", type=str, help="Start time (format: YYYY-MM-DD HH:MM)")
    schedule_parser.add_argument("--provider", type=str, help="Cloud provider")
    
    # Cost command
    cost_parser = subparsers.add_parser("cost", help="Get cost information")
    cost_parser.add_argument("--id", type=str, help="Instance ID (if omitted, shows total cost)")
    cost_parser.add_argument("--provider", type=str, help="Cloud provider")
    
    # Optimize command
    optimize_parser = subparsers.add_parser("optimize", help="Optimize resource usage")
    optimize_parser.add_argument("--provider", type=str, help="Cloud provider")
    
    args = parser.parse_args()
    
    # Create manager
    manager = get_cloud_gpu_manager(args.config)
    
    # Execute command
    if args.command == "create":
        # Read startup script if provided
        onstart_script = None
        if args.script:
            with open(args.script, "r") as f:
                onstart_script = f.read()
        
        instance_id = manager.create_instance(args.type, args.provider, args.image, onstart_script)
        if instance_id:
            print(f"Created instance {instance_id}")
        else:
            print("Failed to create instance")
    
    elif args.command == "stop":
        success = manager.stop_instance(args.id, args.provider)
        if success:
            print(f"Stopped instance {args.id}")
        else:
            print(f"Failed to stop instance {args.id}")
    
    elif args.command == "destroy":
        success = manager.destroy_instance(args.id, args.provider)
        if success:
            print(f"Destroyed instance {args.id}")
        else:
            print(f"Failed to destroy instance {args.id}")
    
    elif args.command == "status":
        if args.id:
            status = manager.get_instance_status(args.id, args.provider)
            print(f"Status for instance {args.id}:")
            print(json.dumps(status, indent=2))
        else:
            instances = manager.monitor_instances(args.provider)
            print(f"Status for all instances:")
            print(json.dumps(instances, indent=2))
    
    elif args.command == "deploy":
        success = manager.deploy_code(args.id, args.git, args.path, args.provider)
        if success:
            print(f"Deployed code to instance {args.id}")
        else:
            print(f"Failed to deploy code to instance {args.id}")
    
    elif args.command == "sync":
        success = manager.sync_data(args.id, args.local, args.remote, args.direction, args.provider)
        if success:
            print(f"{args.direction.capitalize()}ed data to/from instance {args.id}")
        else:
            print(f"Failed to {args.direction} data to/from instance {args.id}")
    
    elif args.command == "schedule":
        start_time = None
        if args.time:
            start_time = datetime.strptime(args.time, "%Y-%m-%d %H:%M")
        
        instance_id = manager.schedule_workload(args.type, args.image, args.command, start_time, args.provider)
        if instance_id:
            print(f"Scheduled workload on instance {instance_id}")
        else:
            print("Failed to schedule workload")
    
    elif args.command == "cost":
        if args.id:
            cost = manager.get_cost_estimate(args.id, args.provider)
            print(f"Estimated cost for instance {args.id}: ${cost:.2f}")
        else:
            cost = manager.get_total_cost(args.provider)
            print(f"Total estimated cost: ${cost:.2f}")
    
    elif args.command == "optimize":
        manager.optimize_resources(args.provider)
        print("Optimized resource usage")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()