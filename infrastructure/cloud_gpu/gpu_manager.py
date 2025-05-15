# Cosmic Market Oracle - GPU Manager Module

"""
This module provides a unified interface for managing cloud GPU resources
across different providers (VAST.ai and ThunderStorm).

It handles instance selection, deployment, monitoring, and automatic shutdown
to optimize cost efficiency for different workloads.
"""

import os
import json
import time
import logging
from typing import Dict, List, Optional, Union, Any
import subprocess
import requests
from datetime import datetime, timedelta
import yaml

# Import provider-specific managers
from .vast_ai_manager import VastAIManager
from .thunder_compute_manager import ThunderComputeManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GPUManager:
    """Unified interface for managing cloud GPU resources."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the GPU manager with configuration.
        
        Args:
            config_path: Path to the GPU configuration YAML file
        """
        self.config = self._load_config(config_path)
        self.vast_manager = VastAIManager(self.config.get('vast_ai', {}))
        self.thunder_manager = ThunderComputeManager(self.config.get('thunder_compute', {}))
        
        # Track active instances
        self.active_instances = {}
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict:
        """
        Load configuration from file or use defaults.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        default_config = {
            'default_provider': 'vast_ai',
            'vast_ai': {
                'api_key': os.environ.get('VAST_AI_API_KEY', ''),
                'min_bid_price': 0.2,
                'max_bid_price': 0.8,
                'default_image': 'pytorch/pytorch:latest',
                'storage_gb': 50
            },
            'thunder_compute': {
                'api_key': os.environ.get('THUNDER_COMPUTE_API_KEY', ''),
                'default_image': 'pytorch/pytorch:latest',
                'storage_gb': 50
            },
            'workloads': {
                'data_preprocessing': {
                    'provider': 'vast_ai',
                    'gpu_type': ['RTX 3080', 'RTX 3090'],
                    'min_vram_gb': 10,
                    'cpu_cores': 8,
                    'ram_gb': 32,
                    'max_price': 0.4,
                    'docker_image': 'cosmic_market_oracle/preprocessing:latest'
                },
                'model_training': {
                    'provider': 'vast_ai',
                    'gpu_type': ['RTX 4090', 'A100'],
                    'min_vram_gb': 24,
                    'cpu_cores': 16,
                    'ram_gb': 64,
                    'max_price': 0.8,
                    'docker_image': 'cosmic_market_oracle/training:latest'
                },
                'reinforcement_learning': {
                    'provider': 'thunder_compute',
                    'gpu_type': ['A100'],
                    'min_vram_gb': 40,
                    'cpu_cores': 32,
                    'ram_gb': 128,
                    'max_price': 1.2,
                    'docker_image': 'cosmic_market_oracle/reinforcement:latest'
                }
            },
            'auto_shutdown': {
                'idle_threshold_minutes': 30,
                'max_runtime_hours': 24,
                'check_interval_minutes': 5
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = yaml.safe_load(f)
                    # Deep merge configs
                    self._deep_merge(default_config, user_config)
            except Exception as e:
                logger.error(f"Error loading config from {config_path}: {str(e)}")
                logger.warning("Using default configuration")
        
        return default_config
    
    def _deep_merge(self, base_dict: Dict, update_dict: Dict) -> None:
        """
        Deep merge two dictionaries, modifying base_dict in-place.
        
        Args:
            base_dict: Base dictionary to update
            update_dict: Dictionary with values to merge into base_dict
        """
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_merge(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def get_available_instances(self, provider: Optional[str] = None, workload_type: Optional[str] = None) -> List[Dict]:
        """
        Get available GPU instances from the specified provider.
        
        Args:
            provider: Provider name ('vast_ai' or 'thunder_compute')
            workload_type: Type of workload to get instances for
            
        Returns:
            List of available instances matching criteria
        """
        if provider is None:
            provider = self.config.get('default_provider', 'vast_ai')
        
        # Get workload-specific requirements if specified
        requirements = {}
        if workload_type and workload_type in self.config.get('workloads', {}):
            requirements = self.config['workloads'][workload_type].copy()
            # Override provider if specified in workload config
            if 'provider' in requirements:
                provider = requirements.pop('provider')
        
        if provider == 'vast_ai':
            return self.vast_manager.get_available_instances(**requirements)
        elif provider == 'thunder_compute':
            return self.thunder_manager.get_available_instances(**requirements)
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    def create_instance(self, workload_type: str, instance_id: Optional[str] = None, 
                       custom_requirements: Optional[Dict] = None) -> Dict:
        """
        Create a GPU instance for the specified workload.
        
        Args:
            workload_type: Type of workload to create instance for
            instance_id: Optional specific instance ID to create
            custom_requirements: Optional custom requirements overriding workload defaults
            
        Returns:
            Dictionary with instance details
        """
        if workload_type not in self.config.get('workloads', {}):
            raise ValueError(f"Unknown workload type: {workload_type}")
        
        # Get workload configuration
        workload_config = self.config['workloads'][workload_type].copy()
        provider = workload_config.pop('provider', self.config.get('default_provider', 'vast_ai'))
        
        # Override with custom requirements if provided
        if custom_requirements:
            workload_config.update(custom_requirements)
        
        # Create instance with appropriate provider
        if provider == 'vast_ai':
            instance = self.vast_manager.create_instance(instance_id, **workload_config)
        elif provider == 'thunder_compute':
            instance = self.thunder_manager.create_instance(instance_id, **workload_config)
        else:
            raise ValueError(f"Unknown provider: {provider}")
        
        # Track the active instance
        instance_key = f"{provider}_{instance['id']}"
        self.active_instances[instance_key] = {
            'provider': provider,
            'id': instance['id'],
            'workload_type': workload_type,
            'start_time': datetime.now(),
            'details': instance
        }
        
        return instance
    
    def terminate_instance(self, instance_key: str) -> bool:
        """
        Terminate a GPU instance.
        
        Args:
            instance_key: Key of the instance to terminate
            
        Returns:
            True if successful, False otherwise
        """
        if instance_key not in self.active_instances:
            logger.warning(f"Instance {instance_key} not found in active instances")
            return False
        
        instance = self.active_instances[instance_key]
        provider = instance['provider']
        instance_id = instance['id']
        
        success = False
        if provider == 'vast_ai':
            success = self.vast_manager.terminate_instance(instance_id)
        elif provider == 'thunder_compute':
            success = self.thunder_manager.terminate_instance(instance_id)
        
        if success:
            del self.active_instances[instance_key]
            logger.info(f"Instance {instance_key} terminated successfully")
        
        return success
    
    def get_instance_status(self, instance_key: str) -> Dict:
        """
        Get the status of a GPU instance.
        
        Args:
            instance_key: Key of the instance to check
            
        Returns:
            Dictionary with instance status
        """
        if instance_key not in self.active_instances:
            raise ValueError(f"Instance {instance_key} not found in active instances")
        
        instance = self.active_instances[instance_key]
        provider = instance['provider']
        instance_id = instance['id']
        
        if provider == 'vast_ai':
            status = self.vast_manager.get_instance_status(instance_id)
        elif provider == 'thunder_compute':
            status = self.thunder_manager.get_instance_status(instance_id)
        else:
            raise ValueError(f"Unknown provider: {provider}")
        
        # Update instance details
        self.active_instances[instance_key]['details'].update(status)
        
        return status
    
    def check_idle_instances(self) -> List[str]:
        """
        Check for idle instances that should be terminated.
        
        Returns:
            List of instance keys that were terminated due to idleness
        """
        terminated_instances = []
        idle_threshold = timedelta(minutes=self.config.get('auto_shutdown', {}).get('idle_threshold_minutes', 30))
        max_runtime = timedelta(hours=self.config.get('auto_shutdown', {}).get('max_runtime_hours', 24))
        
        for instance_key, instance in list(self.active_instances.items()):
            # Check if instance has exceeded max runtime
            runtime = datetime.now() - instance['start_time']
            if runtime > max_runtime:
                logger.info(f"Instance {instance_key} has exceeded maximum runtime of {max_runtime}, terminating")
                if self.terminate_instance(instance_key):
                    terminated_instances.append(instance_key)
                continue
            
            # Check if instance is idle
            status = self.get_instance_status(instance_key)
            if status.get('idle_time', 0) > idle_threshold.total_seconds():
                logger.info(f"Instance {instance_key} is idle for more than {idle_threshold}, terminating")
                if self.terminate_instance(instance_key):
                    terminated_instances.append(instance_key)
        
        return terminated_instances
    
    def start_auto_shutdown_monitor(self, check_interval: Optional[int] = None) -> None:
        """
        Start a background thread to monitor and shutdown idle instances.
        
        Args:
            check_interval: Interval in minutes between checks (default from config)
        """
        if check_interval is None:
            check_interval = self.config.get('auto_shutdown', {}).get('check_interval_minutes', 5)
        
        import threading
        
        def monitor_thread():
            while True:
                try:
                    terminated = self.check_idle_instances()
                    if terminated:
                        logger.info(f"Auto-shutdown terminated {len(terminated)} instances: {terminated}")
                except Exception as e:
                    logger.error(f"Error in auto-shutdown monitor: {str(e)}")
                
                # Sleep for the check interval
                time.sleep(check_interval * 60)
        
        # Start the monitor thread
        thread = threading.Thread(target=monitor_thread, daemon=True)
        thread.start()
        logger.info(f"Auto-shutdown monitor started with {check_interval} minute interval")
    
    def run_benchmark(self, instance_key: str, benchmark_type: str = 'matrix_multiply') -> Dict:
        """
        Run a benchmark on the specified instance.
        
        Args:
            instance_key: Key of the instance to benchmark
            benchmark_type: Type of benchmark to run
            
        Returns:
            Dictionary with benchmark results
        """
        if instance_key not in self.active_instances:
            raise ValueError(f"Instance {instance_key} not found in active instances")
        
        instance = self.active_instances[instance_key]
        provider = instance['provider']
        instance_id = instance['id']
        
        # Define benchmark command based on type
        benchmark_commands = {
            'matrix_multiply': 'python /app/infrastructure/scripts/benchmarks/matrix_multiply.py',
            'conv_network': 'python /app/infrastructure/scripts/benchmarks/conv_network.py',
            'transformer': 'python /app/infrastructure/scripts/benchmarks/transformer.py',
            'data_processing': 'python /app/infrastructure/scripts/benchmarks/data_processing.py',
            'reinforcement': 'python /app/infrastructure/scripts/benchmarks/reinforcement.py'
        }
        
        if benchmark_type not in benchmark_commands:
            raise ValueError(f"Unknown benchmark type: {benchmark_type}")
        
        command = benchmark_commands[benchmark_type]
        
        # Run benchmark on instance
        if provider == 'vast_ai':
            result = self.vast_manager.run_command(instance_id, command)
        elif provider == 'thunder_compute':
            result = self.thunder_manager.run_command(instance_id, command)
        else:
            raise ValueError(f"Unknown provider: {provider}")
        
        return result
    
    def sync_data(self, instance_key: str, local_path: str, remote_path: str, upload: bool = True) -> bool:
        """
        Sync data between local and remote instance.
        
        Args:
            instance_key: Key of the instance to sync with
            local_path: Local path
            remote_path: Remote path on the instance
            upload: If True, upload from local to remote, otherwise download
            
        Returns:
            True if successful, False otherwise
        """
        if instance_key not in self.active_instances:
            raise ValueError(f"Instance {instance_key} not found in active instances")
        
        instance = self.active_instances[instance_key]
        provider = instance['provider']
        instance_id = instance['id']
        
        if provider == 'vast_ai':
            return self.vast_manager.sync_data(instance_id, local_path, remote_path, upload)
        elif provider == 'thunder_compute':
            return self.thunder_manager.sync_data(instance_id, local_path, remote_path, upload)
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    def get_cost_estimate(self, workload_type: str, runtime_hours: float) -> Dict:
        """
        Get cost estimate for running a workload.
        
        Args:
            workload_type: Type of workload to estimate cost for
            runtime_hours: Estimated runtime in hours
            
        Returns:
            Dictionary with cost estimates for different providers
        """
        if workload_type not in self.config.get('workloads', {}):
            raise ValueError(f"Unknown workload type: {workload_type}")
        
        workload_config = self.config['workloads'][workload_type].copy()
        provider = workload_config.pop('provider', self.config.get('default_provider', 'vast_ai'))
        
        estimates = {}
        
        # Get estimate from VAST.ai
        try:
            vast_estimate = self.vast_manager.get_cost_estimate(workload_config, runtime_hours)
            estimates['vast_ai'] = vast_estimate
        except Exception as e:
            logger.error(f"Error getting VAST.ai cost estimate: {str(e)}")
            estimates['vast_ai'] = {'error': str(e)}
        
        # Get estimate from ThunderCompute
        try:
            thunder_estimate = self.thunder_manager.get_cost_estimate(workload_config, runtime_hours)
            estimates['thunder_compute'] = thunder_estimate
        except Exception as e:
            logger.error(f"Error getting ThunderCompute cost estimate: {str(e)}")
            estimates['thunder_compute'] = {'error': str(e)}
        
        # Add recommended provider
        if 'vast_ai' in estimates and 'thunder_compute' in estimates:
            if isinstance(estimates['vast_ai'], dict) and isinstance(estimates['thunder_compute'], dict):
                vast_cost = estimates['vast_ai'].get('total_cost', float('inf'))
                thunder_cost = estimates['thunder_compute'].get('total_cost', float('inf'))
                
                if vast_cost < thunder_cost:
                    estimates['recommended_provider'] = 'vast_ai'
                else:
                    estimates['recommended_provider'] = 'thunder_compute'
        
        return estimates


# Singleton instance
_gpu_manager_instance = None


def get_gpu_manager(config_path: Optional[str] = None) -> GPUManager:
    """
    Get or create the singleton GPU manager instance.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        GPUManager instance
    """
    global _gpu_manager_instance
    if _gpu_manager_instance is None:
        _gpu_manager_instance = GPUManager(config_path)
    return _gpu_manager_instance
