#!/usr/bin/env python3
"""
ThunderCompute Manager for Cosmic Market Oracle.

This module provides functionality for managing GPU instances on ThunderCompute,
including instance creation, monitoring, and cost optimization.
"""

import os
import json
import time
import logging
import yaml
import requests
from src.trading.unified_mcts import MCTS, MCTSPredictor
from src.trading.modular_hierarchical_rl import ModularHierarchicalRLAgent, MCTSStrategicPlanner, PPOTacticalExecutor
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta
import subprocess

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ThunderComputeManager:
    """Manager for ThunderCompute GPU instances.
    
    This class provides functionality for:
    1. Creating and managing GPU instances on ThunderCompute
    2. Monitoring instance status and utilization
    3. Implementing cost optimization strategies
    4. Automating deployment of code and data
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialize the ThunderCompute manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = Path(config_path) if config_path else Path("config/gpu_instances.yaml")
        self.config = self._load_config()
        
        # API configuration
        self.api_key = self.config.get("thundercompute", {}).get("api_key", os.environ.get("THUNDERCOMPUTE_API_KEY"))
        self.api_url = self.config.get("thundercompute", {}).get("api_url", "https://api.thundercompute.com/v1")
        
        # Instance configurations
        self.instance_configs = self.config.get("instance_types", {})
        
        # SSH key path
        self.ssh_key_path = Path(self.config.get("thundercompute", {}).get("ssh_key_path", "~/.ssh/thundercompute_rsa")).expanduser()
        
        if not self.api_key:
            logger.warning("ThunderCompute API key not found in config or environment variables")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file.
        
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
    
    def _api_request(self, endpoint: str, method: str = "GET", data: Optional[Dict] = None) -> Dict:
        """Make an API request to ThunderCompute.
        
        Args:
            endpoint: API endpoint
            method: HTTP method
            data: Request data
            
        Returns:
            Response data
        """
        if not self.api_key:
            raise ValueError("ThunderCompute API key not configured")
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        url = f"{self.api_url}/{endpoint}"
        
        try:
            if method == "GET":
                response = requests.get(url, headers=headers)
            elif method == "POST":
                response = requests.post(url, headers=headers, json=data)
            elif method == "PUT":
                response = requests.put(url, headers=headers, json=data)
            elif method == "DELETE":
                response = requests.delete(url, headers=headers)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise
    
    def create_instance(self, instance_type: str, docker_image: Optional[str] = None, 
                       onstart_script: Optional[str] = None) -> Optional[str]:
        """Create a new GPU instance on ThunderCompute.
        
        Args:
            instance_type: Type of instance to create (e.g., 'rtx4090', 'a100')
            docker_image: Docker image to use
            onstart_script: Script to run on instance startup
            
        Returns:
            Instance ID if successful, None otherwise
        """
        # Get instance configuration
        instance_config = self.instance_configs.get(instance_type)
        if not instance_config:
            logger.error(f"Unknown instance type: {instance_type}")
            return None
        
        # Prepare instance creation request
        request_data = {
            "gpu_type": instance_config.get("gpu_type", "RTX 3080"),
            "cpu_cores": instance_config.get("cpu_cores", 4),
            "memory_gb": instance_config.get("memory_gb", 16),
            "disk_gb": instance_config.get("disk_gb", 100),
            "ssh_key": self._get_ssh_public_key(),
            "docker_image": docker_image or instance_config.get("docker_image", "nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04"),
            "onstart_script": onstart_script or instance_config.get("onstart_script")
        }
        
        try:
            # Create instance
            logger.info(f"Creating {instance_type} instance on ThunderCompute")
            response = self._api_request("instances", method="POST", data=request_data)
            
            instance_id = response.get("instance_id")
            if instance_id:
                logger.info(f"Created instance {instance_id}")
                return instance_id
            else:
                logger.error("Failed to create instance: No instance ID in response")
                return None
        except Exception as e:
            logger.error(f"Failed to create instance: {e}")
            return None
    
    def _get_ssh_public_key(self) -> str:
        """Get SSH public key for instance access.
        
        Returns:
            SSH public key content
        """
        public_key_path = self.ssh_key_path.with_suffix(".pub")
        
        try:
            if not public_key_path.exists():
                logger.error(f"SSH public key not found at {public_key_path}")
                raise FileNotFoundError(f"SSH public key not found at {public_key_path}")
            
            with open(public_key_path, "r") as f:
                return f.read().strip()
        except Exception as e:
            logger.error(f"Failed to read SSH public key: {e}")
            raise
    
    def list_instances(self) -> List[Dict]:
        """List all instances.
        
        Returns:
            List of instance dictionaries
        """
        try:
            response = self._api_request("instances")
            return response.get("instances", [])
        except Exception as e:
            logger.error(f"Failed to list instances: {e}")
            return []
    
    def get_instance(self, instance_id: str) -> Optional[Dict]:
        """Get instance details.
        
        Args:
            instance_id: Instance ID
            
        Returns:
            Instance details dictionary
        """
        try:
            response = self._api_request(f"instances/{instance_id}")
            return response
        except Exception as e:
            logger.error(f"Failed to get instance {instance_id}: {e}")
            return None
    
    def terminate_instance(self, instance_id: str) -> bool:
        """Terminate an instance.
        
        Args:
            instance_id: Instance ID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self._api_request(f"instances/{instance_id}", method="DELETE")
            logger.info(f"Terminated instance {instance_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to terminate instance {instance_id}: {e}")
            return False
    
    def get_instance_metrics(self, instance_id: str) -> Optional[Dict]:
        """Get instance metrics (GPU utilization, memory usage, etc.).
        
        Args:
            instance_id: Instance ID
            
        Returns:
            Metrics dictionary
        """
        try:
            response = self._api_request(f"instances/{instance_id}/metrics")
            return response
        except Exception as e:
            logger.error(f"Failed to get metrics for instance {instance_id}: {e}")
            return None
    
    def upload_data(self, instance_id: str, local_path: Union[str, Path], remote_path: str) -> bool:
        """Upload data to an instance.
        
        Args:
            instance_id: Instance ID
            local_path: Local file or directory path
            remote_path: Remote path on the instance
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get instance IP address
            instance = self.get_instance(instance_id)
            if not instance:
                logger.error(f"Instance {instance_id} not found")
                return False
            
            ip_address = instance.get("ip_address")
            if not ip_address:
                logger.error(f"No IP address found for instance {instance_id}")
                return False
            
            # Use scp to upload data
            local_path = Path(local_path)
            cmd = [
                "scp",
                "-i", str(self.ssh_key_path),
                "-r",
                "-o", "StrictHostKeyChecking=no",
                str(local_path),
                f"ubuntu@{ip_address}:{remote_path}"
            ]
            
            logger.info(f"Uploading {local_path} to {instance_id}:{remote_path}")
            subprocess.run(cmd, check=True)
            
            logger.info(f"Upload completed successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to upload data to instance {instance_id}: {e}")
            return False
    
    def run_command(self, instance_id: str, command: str) -> Optional[str]:
        """Run a command on an instance.
        
        Args:
            instance_id: Instance ID
            command: Command to run
            
        Returns:
            Command output if successful, None otherwise
        """
        try:
            # Get instance IP address
            instance = self.get_instance(instance_id)
            if not instance:
                logger.error(f"Instance {instance_id} not found")
                return None
            
            ip_address = instance.get("ip_address")
            if not ip_address:
                logger.error(f"No IP address found for instance {instance_id}")
                return None
            
            # Use ssh to run command
            cmd = [
                "ssh",
                "-i", str(self.ssh_key_path),
                "-o", "StrictHostKeyChecking=no",
                f"ubuntu@{ip_address}",
                command
            ]
            
            logger.info(f"Running command on instance {instance_id}: {command}")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            logger.info(f"Command executed successfully")
            return result.stdout
        except Exception as e:
            logger.error(f"Failed to run command on instance {instance_id}: {e}")
            return None
    
    def is_instance_idle(self, instance_id: str, threshold: float = 5.0, duration: int = 1800) -> bool:
        """Check if an instance is idle (low GPU utilization for extended period).
        
        Args:
            instance_id: Instance ID
            threshold: GPU utilization threshold percentage
            duration: Duration in seconds to consider idle
            
        Returns:
            True if instance is idle, False otherwise
        """
        try:
            metrics = self.get_instance_metrics(instance_id)
            if not metrics:
                return False
            
            # Check if GPU utilization is below threshold for the specified duration
            utilization_history = metrics.get("gpu_utilization_history", [])
            if not utilization_history:
                return False
            
            # Filter recent history based on duration
            current_time = time.time()
            recent_history = [entry for entry in utilization_history 
                             if current_time - entry.get("timestamp", 0) <= duration]
            
            if not recent_history:
                return False
            
            # Check if all recent utilization values are below threshold
            return all(entry.get("value", 100) < threshold for entry in recent_history)
        except Exception as e:
            logger.error(f"Failed to check if instance {instance_id} is idle: {e}")
            return False
    
    def estimate_cost(self, instance_type: str, hours: float) -> float:
        """Estimate cost for running an instance.
        
        Args:
            instance_type: Type of instance
            hours: Number of hours
            
        Returns:
            Estimated cost in USD
        """
        # Get instance configuration
        instance_config = self.instance_configs.get(instance_type)
        if not instance_config:
            logger.error(f"Unknown instance type: {instance_type}")
            return 0.0
        
        # Get hourly rate
        hourly_rate = instance_config.get("hourly_rate", 0.0)
        
        # Calculate cost
        return hourly_rate * hours


def get_thunder_compute_manager(config_path: Optional[Union[str, Path]] = None) -> ThunderComputeManager:
    """Get a ThunderCompute manager instance.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        ThunderCompute manager instance
    """
    return ThunderComputeManager(config_path)