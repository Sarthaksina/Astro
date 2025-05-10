"""VAST.ai GPU Instance Manager for Cosmic Market Oracle.

This module provides functionality for managing GPU instances on VAST.ai,
including creating, monitoring, and terminating instances based on workload requirements.
"""

import os
import json
import time
import logging
import requests
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VASTAIManager:
    """Manager for VAST.ai GPU instances.
    
    This class provides functionality for:
    1. Creating and managing GPU instances on VAST.ai
    2. Monitoring instance utilization and costs
    3. Automatically shutting down underutilized instances
    4. Synchronizing data between local and remote instances
    5. Deploying code to instances
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None, api_key: Optional[str] = None):
        """Initialize the VAST.ai manager.
        
        Args:
            config_path: Path to GPU instance configuration file
            api_key: VAST.ai API key (if not provided, will look for VAST_AI_API_KEY environment variable)
        """
        self.config_path = Path(config_path) if config_path else Path("config/gpu_instances.yaml")
        self.api_key = api_key or os.environ.get("VAST_AI_API_KEY")
        
        if not self.api_key:
            raise ValueError("VAST.ai API key not provided. Set VAST_AI_API_KEY environment variable or pass api_key parameter.")
        
        self.config = self._load_config()
        self.base_url = "https://console.vast.ai/api/v0"
        self.headers = {"Accept": "application/json", "Authorization": f"Bearer {self.api_key}"}
        
        # Create instances directory if it doesn't exist
        self.instances_dir = Path(self.config.get("instances_dir", "./instances"))
        self.instances_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing instances
        self.instances = self._load_instances()
        
        logger.info(f"Initialized VAST.ai manager with {len(self.instances)} existing instances")
    
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
    
    def _load_instances(self) -> Dict[str, Dict[str, Any]]:
        """Load information about existing instances.
        
        Returns:
            Dictionary of instance information keyed by instance ID
        """
        instances = {}
        
        try:
            for file_path in self.instances_dir.glob("*.json"):
                with open(file_path, "r") as f:
                    instance_info = json.load(f)
                    instance_id = instance_info.get("id")
                    if instance_id:
                        instances[str(instance_id)] = instance_info
        except Exception as e:
            logger.error(f"Error loading instances: {e}")
        
        return instances
    
    def _save_instance(self, instance_id: str, instance_info: Dict[str, Any]) -> None:
        """Save instance information to disk.
        
        Args:
            instance_id: Instance ID
            instance_info: Instance information dictionary
        """
        try:
            file_path = self.instances_dir / f"{instance_id}.json"
            with open(file_path, "w") as f:
                json.dump(instance_info, f, indent=2)
            
            # Update in-memory cache
            self.instances[instance_id] = instance_info
            
            logger.debug(f"Saved instance information for {instance_id}")
        except Exception as e:
            logger.error(f"Error saving instance {instance_id}: {e}")
    
    def _delete_instance_info(self, instance_id: str) -> None:
        """Delete instance information file.
        
        Args:
            instance_id: Instance ID
        """
        try:
            file_path = self.instances_dir / f"{instance_id}.json"
            if file_path.exists():
                file_path.unlink()
            
            # Remove from in-memory cache
            if instance_id in self.instances:
                del self.instances[instance_id]
            
            logger.debug(f"Deleted instance information for {instance_id}")
        except Exception as e:
            logger.error(f"Error deleting instance info {instance_id}: {e}")
    
    def search_instances(self, instance_type: str) -> List[Dict[str, Any]]:
        """Search for available instances on VAST.ai.
        
        Args:
            instance_type: Type of instance to search for (e.g., 'rtx4090', 'a100')
            
        Returns:
            List of available instances matching criteria
        """
        try:
            # Get instance type configuration
            if instance_type not in self.config.get("instance_types", {}):
                raise ValueError(f"Unknown instance type: {instance_type}")
            
            instance_config = self.config["instance_types"][instance_type]
            
            # Build search parameters
            params = {
                "gpu_name": instance_type.upper(),
                "num_gpus": instance_config.get("gpu_count", 1),
                "cpu_cores": instance_config.get("cpu_count", 4),
                "ram": instance_config.get("memory", 16),
                "disk_space": instance_config.get("disk", 50),
                "order": "score-",  # Sort by score descending
                "type": "on-demand" if not instance_config.get("spot_instance", True) else "bid"
            }
            
            # Make API request
            response = requests.get(f"{self.base_url}/offers", headers=self.headers, params=params)
            response.raise_for_status()
            
            # Filter by max hourly cost
            max_cost = instance_config.get("hourly_cost_max", float('inf'))
            offers = response.json().get("offers", [])
            filtered_offers = [offer for offer in offers if offer.get("dph_total", float('inf')) <= max_cost]
            
            logger.info(f"Found {len(filtered_offers)} available {instance_type} instances")
            return filtered_offers
        except Exception as e:
            logger.error(f"Error searching for instances: {e}")
            return []
    
    def create_instance(self, instance_type: str, docker_image: Optional[str] = None, 
                       onstart_script: Optional[str] = None) -> Optional[str]:
        """Create a new instance on VAST.ai.
        
        Args:
            instance_type: Type of instance to create (e.g., 'rtx4090', 'a100')
            docker_image: Docker image to use (defaults to instance type config)
            onstart_script: Script to run on instance startup
            
        Returns:
            Instance ID if successful, None otherwise
        """
        try:
            # Search for available instances
            offers = self.search_instances(instance_type)
            if not offers:
                logger.error(f"No available {instance_type} instances found")
                return None
            
            # Select the best offer (first in list, sorted by score)
            offer = offers[0]
            offer_id = offer.get("id")
            
            # Get instance configuration
            instance_config = self.config["instance_types"][instance_type]
            
            # Use provided docker image or default from config
            image = docker_image or instance_config.get("image", "nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04")
            
            # Create instance
            data = {
                "client_id": "cosmic_market_oracle",
                "image": image,
                "disk": instance_config.get("disk", 50),
                "label": f"Cosmic Market Oracle - {instance_type} - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                "onstart": onstart_script or ""
            }
            
            response = requests.put(f"{self.base_url}/instances/{offer_id}/", headers=self.headers, json=data)
            response.raise_for_status()
            
            # Get instance ID
            instance_data = response.json()
            instance_id = str(instance_data.get("id"))
            
            if not instance_id:
                logger.error("Failed to get instance ID from response")
                return None
            
            # Save instance information
            instance_info = {
                "id": instance_id,
                "type": instance_type,
                "created_at": datetime.now().isoformat(),
                "status": "starting",
                "image": image,
                "offer": offer,
                "instance_data": instance_data
            }
            
            self._save_instance(instance_id, instance_info)
            
            logger.info(f"Created {instance_type} instance with ID {instance_id}")
            return instance_id
        except Exception as e:
            logger.error(f"Error creating instance: {e}")
            return None
    
    def get_instance_status(self, instance_id: str) -> Dict[str, Any]:
        """Get status of an instance.
        
        Args:
            instance_id: Instance ID
            
        Returns:
            Instance status information
        """
        try:
            response = requests.get(f"{self.base_url}/instances/{instance_id}", headers=self.headers)
            response.raise_for_status()
            
            instance_data = response.json()
            
            # Update stored instance information
            if instance_id in self.instances:
                self.instances[instance_id]["status"] = instance_data.get("status", "unknown")
                self.instances[instance_id]["last_checked"] = datetime.now().isoformat()
                self.instances[instance_id]["instance_data"] = instance_data
                self._save_instance(instance_id, self.instances[instance_id])
            
            return instance_data
        except Exception as e:
            logger.error(f"Error getting instance status for {instance_id}: {e}")
            return {"status": "error", "error": str(e)}
    
    def stop_instance(self, instance_id: str) -> bool:
        """Stop an instance.
        
        Args:
            instance_id: Instance ID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            response = requests.put(f"{self.base_url}/instances/{instance_id}/stop", headers=self.headers)
            response.raise_for_status()
            
            # Update stored instance information
            if instance_id in self.instances:
                self.instances[instance_id]["status"] = "stopped"
                self.instances[instance_id]["stopped_at"] = datetime.now().isoformat()
                self._save_instance(instance_id, self.instances[instance_id])
            
            logger.info(f"Stopped instance {instance_id}")
            return True
        except Exception as e:
            logger.error(f"Error stopping instance {instance_id}: {e}")
            return False
    
    def destroy_instance(self, instance_id: str) -> bool:
        """Destroy an instance.
        
        Args:
            instance_id: Instance ID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            response = requests.delete(f"{self.base_url}/instances/{instance_id}", headers=self.headers)
            response.raise_for_status()
            
            # Delete instance information
            self._delete_instance_info(instance_id)
            
            logger.info(f"Destroyed instance {instance_id}")
            return True
        except Exception as e:
            logger.error(f"Error destroying instance {instance_id}: {e}")
            return False
    
    def monitor_instances(self) -> Dict[str, Dict[str, Any]]:
        """Monitor all instances and update their status.
        
        Returns:
            Dictionary of instance information keyed by instance ID
        """
        updated_instances = {}
        
        for instance_id, instance_info in self.instances.items():
            # Skip instances that are known to be destroyed
            if instance_info.get("status") == "destroyed":
                continue
                
            # Get current status
            status_data = self.get_instance_status(instance_id)
            updated_instances[instance_id] = self.instances[instance_id]
            
            # Check for underutilized instances
            if self._should_stop_instance(instance_id, status_data):
                logger.info(f"Instance {instance_id} is underutilized, stopping")
                self.stop_instance(instance_id)
        
        return updated_instances
    
    def _should_stop_instance(self, instance_id: str, status_data: Dict[str, Any]) -> bool:
        """Check if an instance should be stopped based on utilization.
        
        Args:
            instance_id: Instance ID
            status_data: Instance status data
            
        Returns:
            True if instance should be stopped, False otherwise
        """
        # Skip if instance is not running
        if status_data.get("status") != "running":
            return False
        
        # Get monitoring configuration
        monitoring_config = self.config.get("monitoring", {})
        utilization_threshold = monitoring_config.get("utilization_threshold", 0.1)
        
        # Get GPU utilization
        gpu_util = status_data.get("gpu_util", 1.0)  # Default to 100% if unknown
        
        # Check if instance has been running for at least 30 minutes
        instance_info = self.instances.get(instance_id, {})
        created_at = instance_info.get("created_at")
        
        if created_at:
            created_time = datetime.fromisoformat(created_at)
            min_runtime = timedelta(minutes=30)
            
            if datetime.now() - created_time < min_runtime:
                # Don't stop instances that just started
                return False
        
        # Check if utilization is below threshold
        return gpu_util < utilization_threshold
    
    def sync_data(self, instance_id: str, local_path: Union[str, Path], remote_path: str, 
                 direction: str = "upload") -> bool:
        """Sync data between local and remote instance.
        
        Args:
            instance_id: Instance ID
            local_path: Local path
            remote_path: Remote path
            direction: 'upload' or 'download'
            
        Returns:
            True if successful, False otherwise
        """
        # This is a placeholder for the actual implementation
        # In a real implementation, this would use rsync, scp, or a similar tool
        logger.info(f"Syncing data for instance {instance_id} ({direction})")
        return True
    
    def deploy_code(self, instance_id: str, git_repo: Optional[str] = None, 
                   local_path: Optional[Union[str, Path]] = None) -> bool:
        """Deploy code to an instance.
        
        Args:
            instance_id: Instance ID
            git_repo: Git repository URL (if deploying from git)
            local_path: Local path to code (if deploying from local)
            
        Returns:
            True if successful, False otherwise
        """
        # This is a placeholder for the actual implementation
        # In a real implementation, this would use git clone or rsync
        logger.info(f"Deploying code to instance {instance_id}")
        return True
    
    def get_cost_estimate(self, instance_id: str) -> float:
        """Get estimated cost for an instance.
        
        Args:
            instance_id: Instance ID
            
        Returns:
            Estimated cost in USD
        """
        try:
            instance_info = self.instances.get(instance_id)
            if not instance_info:
                return 0.0
            
            # Get instance data
            instance_data = instance_info.get("instance_data", {})
            hourly_rate = instance_data.get("dph_total", 0.0)
            
            # Calculate runtime
            created_at = instance_info.get("created_at")
            stopped_at = instance_info.get("stopped_at")
            
            if not created_at:
                return 0.0
            
            start_time = datetime.fromisoformat(created_at)
            end_time = datetime.fromisoformat(stopped_at) if stopped_at else datetime.now()
            
            runtime_hours = (end_time - start_time).total_seconds() / 3600
            
            # Calculate cost
            estimated_cost = hourly_rate * runtime_hours
            
            return estimated_cost
        except Exception as e:
            logger.error(f"Error calculating cost for instance {instance_id}: {e}")
            return 0.0
    
    def get_total_cost(self) -> float:
        """Get total estimated cost for all instances.
        
        Returns:
            Total estimated cost in USD
        """
        total_cost = 0.0
        
        for instance_id in self.instances:
            total_cost += self.get_cost_estimate(instance_id)
        
        return total_cost


def get_vast_ai_manager(config_path: Optional[Union[str, Path]] = None, 
                       api_key: Optional[str] = None) -> VASTAIManager:
    """Get a VAST.ai manager instance.
    
    Args:
        config_path: Path to GPU instance configuration file
        api_key: VAST.ai API key
        
    Returns:
        VASTAIManager instance
    """
    return VASTAIManager(config_path, api_key)