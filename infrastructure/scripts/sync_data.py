#!/usr/bin/env python3
"""
Data Synchronization Tool for ThunderCompute

This script provides utilities for efficiently synchronizing data between
local environment and ThunderCompute GPU instances, using delta transfers
to minimize bandwidth usage and transfer time.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from src.trading.unified_mcts import MCTS, MCTSPredictor
from src.trading.modular_hierarchical_rl import ModularHierarchicalRLAgent, MCTSStrategicPlanner, PPOTacticalExecutor
from datetime import datetime
from pathlib import Path
import hashlib
import shutil

# Configuration
CONFIG_DIR = Path.home() / ".thundercompute"
CONFIG_FILE = CONFIG_DIR / "config.json"


def load_config():
    """Load ThunderCompute configuration."""
    if not CONFIG_FILE.exists():
        print(f"Error: Configuration file not found at {CONFIG_FILE}")
        print("Please run 'python instance_manager.py setup' first.")
        sys.exit(1)
    
    with open(CONFIG_FILE, 'r') as f:
        return json.load(f)


def calculate_file_hash(file_path, block_size=65536):
    """Calculate SHA-256 hash of a file for integrity verification.
    
    Args:
        file_path: Path to the file
        block_size: Size of blocks to read
        
    Returns:
        Hex digest of the file hash
    """
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for block in iter(lambda: f.read(block_size), b''):
            sha256.update(block)
    return sha256.hexdigest()


def sync_to_instance(local_path, instance_address, remote_path, exclude=None, dry_run=False):
    """Synchronize data from local to ThunderCompute instance.
    
    Args:
        local_path: Local directory or file to sync
        instance_address: SSH address of the instance (user@hostname)
        remote_path: Path on the remote instance
        exclude: List of patterns to exclude
        dry_run: If True, only show what would be transferred
        
    Returns:
        True if successful, False otherwise
    """
    config = load_config()
    ssh_key_path = config.get("ssh_key_path")
    
    # Ensure paths end with slash for directories
    local_path = str(Path(local_path))
    if os.path.isdir(local_path) and not local_path.endswith(os.sep):
        local_path += os.sep
    
    # Build rsync command
    cmd = ["rsync", "-avz", "--progress"]
    
    # Add SSH key if specified
    if ssh_key_path:
        cmd.extend(["-e", f"ssh -i {ssh_key_path}"])
    
    # Add exclude patterns
    if exclude:
        for pattern in exclude:
            cmd.extend(["--exclude", pattern])
    
    # Add dry run flag if specified
    if dry_run:
        cmd.append("--dry-run")
    
    # Add source and destination
    cmd.extend([local_path, f"{instance_address}:{remote_path}"])
    
    print(f"Syncing data to instance:\n{' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        print("Sync completed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error syncing data: {e}")
        return False


def sync_from_instance(instance_address, remote_path, local_path, exclude=None, dry_run=False):
    """Synchronize data from ThunderCompute instance to local.
    
    Args:
        instance_address: SSH address of the instance (user@hostname)
        remote_path: Path on the remote instance
        local_path: Local directory or file to sync to
        exclude: List of patterns to exclude
        dry_run: If True, only show what would be transferred
        
    Returns:
        True if successful, False otherwise
    """
    config = load_config()
    ssh_key_path = config.get("ssh_key_path")
    
    # Ensure local directory exists
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    # Build rsync command
    cmd = ["rsync", "-avz", "--progress"]
    
    # Add SSH key if specified
    if ssh_key_path:
        cmd.extend(["-e", f"ssh -i {ssh_key_path}"])
    
    # Add exclude patterns
    if exclude:
        for pattern in exclude:
            cmd.extend(["--exclude", pattern])
    
    # Add dry run flag if specified
    if dry_run:
        cmd.append("--dry-run")
    
    # Add source and destination
    cmd.extend([f"{instance_address}:{remote_path}", local_path])
    
    print(f"Syncing data from instance:\n{' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        print("Sync completed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error syncing data: {e}")
        return False


def setup_sync_config(local_data_dir, remote_data_dir):
    """Set up synchronization configuration.
    
    Args:
        local_data_dir: Local data directory
        remote_data_dir: Remote data directory
    """
    config = load_config()
    
    # Update config with sync directories
    config["local_data_dir"] = str(Path(local_data_dir).absolute())
    config["remote_data_dir"] = remote_data_dir
    
    # Save updated config
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Sync configuration updated:\n  Local: {config['local_data_dir']}\n  Remote: {config['remote_data_dir']}")


def create_sync_script():
    """Create a convenience script for frequent syncing."""
    config = load_config()
    
    if "local_data_dir" not in config or "remote_data_dir" not in config:
        print("Error: Sync configuration not set up. Run 'setup' command first.")
        return False
    
    script_path = Path("./sync_instance.sh")
    script_content = f"#!/bin/bash\n\n# Auto-generated sync script\n\n# Sync to instance\nsync_to() {{\n  python {__file__} to $INSTANCE_ADDRESS {config['remote_data_dir']} --from {config['local_data_dir']} $@\n}}\n\n# Sync from instance\nsync_from() {{\n  python {__file__} from $INSTANCE_ADDRESS {config['remote_data_dir']} --to {config['local_data_dir']} $@\n}}\n\n# Check if instance address is provided\nif [ -z "$INSTANCE_ADDRESS" ]; then\n  echo "Error: INSTANCE_ADDRESS environment variable not set"\n  echo "Usage: INSTANCE_ADDRESS=user@hostname ./sync_instance.sh [to|from] [additional args]"\n  exit 1\nfi\n\n# Parse command\nif [ "$1" = "to" ]; then\n  shift\n  sync_to $@\nelif [ "$1" = "from" ]; then\n  shift\n  sync_from $@\nelse\n  echo "Usage: INSTANCE_ADDRESS=user@hostname ./sync_instance.sh [to|from] [additional args]"\n  exit 1\nfi\n"
    
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    os.chmod(script_path, 0o755)  # Make executable
    
    print(f"Created sync script: {script_path}")
    print("Usage: INSTANCE_ADDRESS=user@hostname ./sync_instance.sh [to|from]")
    return True


def main():
    parser = argparse.ArgumentParser(description="ThunderCompute Data Synchronization Tool")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Setup command
    setup_parser = subparsers.add_parser("setup", help="Set up sync configuration")
    setup_parser.add_argument("local_dir", help="Local data directory")
    setup_parser.add_argument("remote_dir", help="Remote data directory")
    
    # Create script command
    create_script_parser = subparsers.add_parser("create-script", help="Create convenience sync script")
    
    # Sync to instance command
    to_parser = subparsers.add_parser("to", help="Sync data to instance")
    to_parser.add_argument("instance", help="Instance address (user@hostname)")
    to_parser.add_argument("remote_path", help="Path on the remote instance")
    to_parser.add_argument("--from", dest="local_path", required=True, help="Local path to sync from")
    to_parser.add_argument("--exclude", action="append", help="Patterns to exclude")
    to_parser.add_argument("--dry-run", action="store_true", help="Show what would be transferred without actually doing it")
    
    # Sync from instance command
    from_parser = subparsers.add_parser("from", help="Sync data from instance")
    from_parser.add_argument("instance", help="Instance address (user@hostname)")
    from_parser.add_argument("remote_path", help="Path on the remote instance")
    from_parser.add_argument("--to", dest="local_path", required=True, help="Local path to sync to")
    from_parser.add_argument("--exclude", action="append", help="Patterns to exclude")
    from_parser.add_argument("--dry-run", action="store_true", help="Show what would be transferred without actually doing it")
    
    args = parser.parse_args()
    
    if args.command == "setup":
        setup_sync_config(args.local_dir, args.remote_dir)
    
    elif args.command == "create-script":
        create_sync_script()
    
    elif args.command == "to":
        sync_to_instance(args.local_path, args.instance, args.remote_path, args.exclude, args.dry_run)
    
    elif args.command == "from":
        sync_from_instance(args.instance, args.remote_path, args.local_path, args.exclude, args.dry_run)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()