#!/usr/bin/env python3
"""
ThunderCompute Instance Manager

This script provides utilities for managing ThunderCompute GPU instances,
including launching, monitoring, and automatically shutting down instances
to optimize costs.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Configuration
CONFIG_DIR = Path.home() / ".thundercompute"
CONFIG_FILE = CONFIG_DIR / "config.json"
API_KEY_FILE = CONFIG_DIR / "api_key.txt"
INSTANCE_TEMPLATES = {
    "deep_learning": {
        "gpu_type": "RTX 4090",
        "cpu_cores": 8,
        "memory": 32,
        "disk": 100,
        "image": "cosmic-market-oracle/deep-learning:latest"
    },
    "data_preprocessing": {
        "gpu_type": "RTX 3080",
        "cpu_cores": 4,
        "memory": 16,
        "disk": 200,
        "image": "cosmic-market-oracle/data-preprocessing:latest"
    },
    "reinforcement_learning": {
        "gpu_type": "A100",
        "gpu_count": 2,
        "cpu_cores": 16,
        "memory": 64,
        "disk": 200,
        "image": "cosmic-market-oracle/reinforcement-learning:latest"
    }
}


def setup_config():
    """Set up ThunderCompute configuration files."""
    if not CONFIG_DIR.exists():
        CONFIG_DIR.mkdir(parents=True)
    
    if not CONFIG_FILE.exists():
        config = {
            "api_endpoint": "https://api.thundercompute.com/v1",
            "ssh_key_path": str(Path.home() / ".ssh" / "thundercompute_rsa"),
            "default_region": "us-east",
            "storage_path": str(Path.home() / "thundercompute_data"),
            "cost_threshold": 10.0,  # USD per day
            "max_idle_time": 30  # minutes
        }
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Created default configuration at {CONFIG_FILE}")
    
    if not API_KEY_FILE.exists():
        api_key = input("Enter your ThunderCompute API key: ")
        with open(API_KEY_FILE, 'w') as f:
            f.write(api_key)
        os.chmod(API_KEY_FILE, 0o600)  # Secure permissions
        print(f"API key saved to {API_KEY_FILE}")


def load_config():
    """Load ThunderCompute configuration."""
    if not CONFIG_FILE.exists():
        setup_config()
    
    with open(CONFIG_FILE, 'r') as f:
        return json.load(f)


def get_api_key():
    """Get ThunderCompute API key."""
    if not API_KEY_FILE.exists():
        setup_config()
    
    with open(API_KEY_FILE, 'r') as f:
        return f.read().strip()


def launch_instance(template_name, name=None, spot=False, max_price=None):
    """Launch a new ThunderCompute instance based on a template."""
    if template_name not in INSTANCE_TEMPLATES:
        print(f"Error: Template '{template_name}' not found. Available templates: {list(INSTANCE_TEMPLATES.keys())}")
        return False
    
    config = load_config()
    template = INSTANCE_TEMPLATES[template_name]
    
    if name is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        name = f"{template_name}-{timestamp}"
    
    instance_config = {
        "name": name,
        "gpu_type": template["gpu_type"],
        "cpu_cores": template["cpu_cores"],
        "memory": template["memory"],
        "disk": template["disk"],
        "image": template["image"],
        "region": config["default_region"],
        "ssh_key_path": config["ssh_key_path"],
        "spot": spot
    }
    
    if "gpu_count" in template:
        instance_config["gpu_count"] = template["gpu_count"]
    
    if spot and max_price is not None:
        instance_config["max_price"] = max_price
    
    # In a real implementation, this would use the ThunderCompute API
    # For now, we'll just print the configuration
    print(f"Launching ThunderCompute instance with configuration:")
    print(json.dumps(instance_config, indent=2))
    
    # Simulate API call
    print(f"Instance '{name}' launched successfully!")
    return True


def list_instances():
    """List all running ThunderCompute instances."""
    # In a real implementation, this would use the ThunderCompute API
    # For now, we'll just print a message
    print("Listing ThunderCompute instances...")
    print("(This would show actual instances in a real implementation)")


def monitor_instance(instance_id):
    """Monitor a ThunderCompute instance's resource usage."""
    # In a real implementation, this would use the ThunderCompute API
    # For now, we'll just print a message
    print(f"Monitoring instance {instance_id}...")
    print("GPU Utilization: 78%")
    print("Memory Usage: 45%")
    print("Cost so far: $2.34")


def shutdown_idle_instances(idle_threshold_minutes=None):
    """Shutdown instances that have been idle for longer than the threshold."""
    config = load_config()
    threshold = idle_threshold_minutes or config["max_idle_time"]
    
    # In a real implementation, this would use the ThunderCompute API
    # For now, we'll just print a message
    print(f"Checking for instances idle for more than {threshold} minutes...")
    print("(This would shut down idle instances in a real implementation)")


def main():
    parser = argparse.ArgumentParser(description="ThunderCompute Instance Manager")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Setup command
    setup_parser = subparsers.add_parser("setup", help="Set up ThunderCompute configuration")
    
    # Launch command
    launch_parser = subparsers.add_parser("launch", help="Launch a new instance")
    launch_parser.add_argument("template", choices=INSTANCE_TEMPLATES.keys(), help="Instance template to use")
    launch_parser.add_argument("--name", help="Custom name for the instance")
    launch_parser.add_argument("--spot", action="store_true", help="Use spot instance for lower cost")
    launch_parser.add_argument("--max-price", type=float, help="Maximum hourly price for spot instance")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List running instances")
    
    # Monitor command
    monitor_parser = subparsers.add_parser("monitor", help="Monitor instance resource usage")
    monitor_parser.add_argument("instance_id", help="ID of the instance to monitor")
    
    # Shutdown command
    shutdown_parser = subparsers.add_parser("shutdown", help="Shutdown idle instances")
    shutdown_parser.add_argument("--idle-threshold", type=int, help="Idle time threshold in minutes")
    
    args = parser.parse_args()
    
    if args.command == "setup":
        setup_config()
    elif args.command == "launch":
        launch_instance(args.template, args.name, args.spot, args.max_price)
    elif args.command == "list":
        list_instances()
    elif args.command == "monitor":
        monitor_instance(args.instance_id)
    elif args.command == "shutdown":
        shutdown_idle_instances(args.idle_threshold)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()