#!/usr/bin/env python3
"""
ThunderCompute Idle Instance Shutdown Script

This script monitors ThunderCompute GPU instances and automatically shuts down
instances that have been idle for a specified period to optimize costs.
It's designed to be run as a scheduled task or service.
"""

import argparse
import json
import logging
import sys
import time
from src.trading.unified_mcts import MCTS, MCTSPredictor
from src.trading.modular_hierarchical_rl import ModularHierarchicalRLAgent, MCTSStrategicPlanner, PPOTacticalExecutor
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

# Import ThunderCompute manager
from infrastructure.cloud_gpu.thunder_compute_manager import get_thunder_compute_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/thunder_shutdown_idle.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_CONFIG_PATH = Path("config/gpu_instances.yaml")
DEFAULT_IDLE_THRESHOLD = 5.0  # GPU utilization percentage
DEFAULT_IDLE_DURATION = 1800  # 30 minutes in seconds
DEFAULT_CHECK_INTERVAL = 300  # 5 minutes in seconds


def shutdown_idle_instances(config_path=None, idle_threshold=None, idle_duration=None, dry_run=False):
    """
    Monitor and shutdown idle ThunderCompute instances.
    
    Args:
        config_path: Path to configuration file
        idle_threshold: GPU utilization threshold percentage to consider idle
        idle_duration: Duration in seconds to consider idle
        dry_run: If True, don't actually shutdown instances, just log
        
    Returns:
        Number of instances shutdown
    """
    # Initialize ThunderCompute manager
    manager = get_thunder_compute_manager(config_path)
    
    # Load configuration
    config = manager._load_config()
    cost_optimization = config.get("cost_optimization", {})
    
    # Use provided parameters or defaults from config
    idle_threshold = idle_threshold or cost_optimization.get("idle_threshold_percent", DEFAULT_IDLE_THRESHOLD)
    idle_duration = idle_duration or cost_optimization.get("idle_duration_seconds", DEFAULT_IDLE_DURATION)
    
    # Get all instances
    instances = manager.list_instances()
    logger.info(f"Found {len(instances)} ThunderCompute instances")
    
    # Track shutdown instances
    shutdown_count = 0
    
    # Check each instance
    for instance in instances:
        instance_id = instance.get("instance_id")
        if not instance_id:
            continue
        
        # Skip instances that are not running
        status = instance.get("status")
        if status != "running":
            logger.info(f"Skipping instance {instance_id} with status {status}")
            continue
        
        # Check if instance is idle
        logger.info(f"Checking if instance {instance_id} is idle (threshold: {idle_threshold}%, duration: {idle_duration}s)")
        is_idle = manager.is_instance_idle(instance_id, idle_threshold, idle_duration)
        
        if is_idle:
            logger.warning(f"Instance {instance_id} has been idle for at least {idle_duration} seconds")
            
            # Get instance details for cost reporting
            instance_details = manager.get_instance(instance_id)
            instance_type = instance_details.get("instance_type", "unknown")
            gpu_type = instance_details.get("gpu_type", "unknown")
            start_time = instance_details.get("start_time")
            
            # Calculate runtime and cost
            if start_time:
                start_datetime = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                runtime_hours = (datetime.now(start_datetime.tzinfo) - start_datetime).total_seconds() / 3600
                
                # Get hourly rate from config
                instance_config = config.get("instance_types", {}).get(instance_type, {})
                hourly_rate = instance_config.get("hourly_rate", 0.0)
                estimated_cost = runtime_hours * hourly_rate
                
                logger.info(f"Instance {instance_id} runtime: {runtime_hours:.2f} hours, estimated cost: ${estimated_cost:.2f}")
            
            if dry_run:
                logger.info(f"[DRY RUN] Would shutdown idle instance {instance_id} ({gpu_type})")
            else:
                # Shutdown the instance
                logger.warning(f"Shutting down idle instance {instance_id} ({gpu_type})")
                success = manager.terminate_instance(instance_id)
                
                if success:
                    logger.info(f"Successfully shutdown instance {instance_id}")
                    shutdown_count += 1
                else:
                    logger.error(f"Failed to shutdown instance {instance_id}")
        else:
            logger.info(f"Instance {instance_id} is active")
    
    return shutdown_count


def main():
    """
    Main function for the idle instance shutdown script.
    """
    parser = argparse.ArgumentParser(description="ThunderCompute Idle Instance Shutdown Script")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--threshold", type=float, help="GPU utilization threshold percentage")
    parser.add_argument("--duration", type=int, help="Idle duration in seconds")
    parser.add_argument("--interval", type=int, default=DEFAULT_CHECK_INTERVAL, help="Check interval in seconds")
    parser.add_argument("--dry-run", action="store_true", help="Don't actually shutdown instances, just log")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    
    args = parser.parse_args()
    
    # Create logs directory if it doesn't exist
    Path("logs").mkdir(exist_ok=True)
    
    try:
        if args.once:
            # Run once
            shutdown_count = shutdown_idle_instances(
                config_path=args.config,
                idle_threshold=args.threshold,
                idle_duration=args.duration,
                dry_run=args.dry_run
            )
            logger.info(f"Shutdown {shutdown_count} idle instances")
        else:
            # Run continuously
            logger.info(f"Starting continuous monitoring with {args.interval}s interval")
            while True:
                try:
                    shutdown_count = shutdown_idle_instances(
                        config_path=args.config,
                        idle_threshold=args.threshold,
                        idle_duration=args.duration,
                        dry_run=args.dry_run
                    )
                    logger.info(f"Shutdown {shutdown_count} idle instances")
                except Exception as e:
                    logger.error(f"Error in monitoring cycle: {e}")
                
                # Wait for next check
                logger.info(f"Waiting {args.interval} seconds until next check")
                time.sleep(args.interval)
    except KeyboardInterrupt:
        logger.info("Shutdown script stopped by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())