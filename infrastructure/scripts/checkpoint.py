#!/usr/bin/env python3
"""
Automatic Checkpointing System for ThunderCompute

This script provides utilities for automatically saving model checkpoints
and resuming training from checkpoints, optimized for ThunderCompute GPU instances.
It helps minimize costs by ensuring training can be resumed if instances are
shut down or preempted.
"""

import argparse
import json
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
import shutil

# Configuration
DEFAULT_CHECKPOINT_DIR = "./checkpoints"
DEFAULT_CHECKPOINT_INTERVAL = 15  # minutes


class CheckpointManager:
    """Manages model checkpoints for training jobs."""
    
    def __init__(self, checkpoint_dir=DEFAULT_CHECKPOINT_DIR, interval_minutes=DEFAULT_CHECKPOINT_INTERVAL):
        """Initialize the checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to store checkpoints
            interval_minutes: How often to save checkpoints (in minutes)
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.interval_minutes = interval_minutes
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Create metadata file if it doesn't exist
        self.metadata_file = self.checkpoint_dir / "metadata.json"
        if not self.metadata_file.exists():
            self._initialize_metadata()
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)
    
    def _initialize_metadata(self):
        """Initialize checkpoint metadata file."""
        metadata = {
            "job_id": f"job-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            "start_time": datetime.now().isoformat(),
            "checkpoints": [],
            "last_checkpoint": None,
            "total_training_time": 0,
            "current_epoch": 0,
            "current_batch": 0,
            "best_checkpoint": None,
            "best_metric": None
        }
        
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _load_metadata(self):
        """Load checkpoint metadata."""
        with open(self.metadata_file, 'r') as f:
            return json.load(f)
    
    def _save_metadata(self, metadata):
        """Save checkpoint metadata."""
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def save_checkpoint(self, model_state, optimizer_state=None, epoch=None, batch=None, metrics=None):
        """Save a model checkpoint.
        
        Args:
            model_state: The model state to save
            optimizer_state: The optimizer state (optional)
            epoch: Current epoch number
            batch: Current batch number
            metrics: Dictionary of evaluation metrics
        
        Returns:
            Path to the saved checkpoint
        """
        metadata = self._load_metadata()
        
        # Update metadata
        if epoch is not None:
            metadata["current_epoch"] = epoch
        if batch is not None:
            metadata["current_batch"] = batch
        
        # Create checkpoint filename
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        checkpoint_name = f"checkpoint-{metadata['current_epoch']:04d}-{timestamp}"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        # Save checkpoint data
        checkpoint_data = {
            "model_state": model_state,
            "optimizer_state": optimizer_state,
            "epoch": metadata["current_epoch"],
            "batch": metadata["current_batch"],
            "timestamp": timestamp,
            "metrics": metrics or {}
        }
        
        # In a real implementation, this would use framework-specific saving
        # For demonstration, we'll just create a JSON file
        with open(f"{checkpoint_path}.json", 'w') as f:
            json.dump({"checkpoint_info": "This is a placeholder for actual model data"}, f)
        
        print(f"Checkpoint saved to {checkpoint_path}")
        
        # Update metadata
        checkpoint_info = {
            "path": str(checkpoint_path),
            "epoch": metadata["current_epoch"],
            "batch": metadata["current_batch"],
            "timestamp": timestamp,
            "metrics": metrics or {}
        }
        
        metadata["checkpoints"].append(checkpoint_info)
        metadata["last_checkpoint"] = checkpoint_info
        
        # Update best checkpoint if applicable
        if metrics and "validation_loss" in metrics:
            if metadata["best_metric"] is None or metrics["validation_loss"] < metadata["best_metric"]:
                metadata["best_checkpoint"] = checkpoint_info
                metadata["best_metric"] = metrics["validation_loss"]
        
        self._save_metadata(metadata)
        return checkpoint_path
    
    def load_latest_checkpoint(self):
        """Load the latest checkpoint.
        
        Returns:
            Tuple of (model_state, optimizer_state, epoch, batch)
        """
        metadata = self._load_metadata()
        
        if metadata["last_checkpoint"] is None:
            print("No checkpoints found.")
            return None, None, 0, 0
        
        checkpoint_info = metadata["last_checkpoint"]
        checkpoint_path = checkpoint_info["path"]
        
        print(f"Loading checkpoint from {checkpoint_path}")
        
        # In a real implementation, this would use framework-specific loading
        # For demonstration, we'll just return placeholder data
        model_state = {"placeholder": "model_state"}
        optimizer_state = {"placeholder": "optimizer_state"}
        
        return model_state, optimizer_state, checkpoint_info["epoch"], checkpoint_info["batch"]
    
    def load_best_checkpoint(self):
        """Load the best checkpoint based on validation metrics.
        
        Returns:
            Tuple of (model_state, optimizer_state, epoch, batch)
        """
        metadata = self._load_metadata()
        
        if metadata["best_checkpoint"] is None:
            print("No best checkpoint found.")
            return self.load_latest_checkpoint()
        
        checkpoint_info = metadata["best_checkpoint"]
        checkpoint_path = checkpoint_info["path"]
        
        print(f"Loading best checkpoint from {checkpoint_path}")
        
        # In a real implementation, this would use framework-specific loading
        # For demonstration, we'll just return placeholder data
        model_state = {"placeholder": "model_state"}
        optimizer_state = {"placeholder": "optimizer_state"}
        
        return model_state, optimizer_state, checkpoint_info["epoch"], checkpoint_info["batch"]
    
    def start_checkpoint_scheduler(self):
        """Start a background thread to periodically save checkpoints."""
        import threading
        
        def checkpoint_worker():
            while True:
                time.sleep(self.interval_minutes * 60)
                print(f"Auto-saving checkpoint (interval: {self.interval_minutes} minutes)")
                # In a real implementation, this would get the current model state
                # For demonstration, we'll just use placeholder data
                model_state = {"placeholder": "model_state"}
                optimizer_state = {"placeholder": "optimizer_state"}
                
                metadata = self._load_metadata()
                self.save_checkpoint(
                    model_state, 
                    optimizer_state,
                    metadata["current_epoch"],
                    metadata["current_batch"]
                )
        
        thread = threading.Thread(target=checkpoint_worker, daemon=True)
        thread.start()
        print(f"Automatic checkpointing started (every {self.interval_minutes} minutes)")
    
    def _handle_shutdown(self, signum, frame):
        """Handle shutdown signals by saving a final checkpoint."""
        print("\nReceived shutdown signal. Saving final checkpoint...")
        # In a real implementation, this would get the current model state
        # For demonstration, we'll just use placeholder data
        model_state = {"placeholder": "model_state"}
        optimizer_state = {"placeholder": "optimizer_state"}
        
        metadata = self._load_metadata()
        self.save_checkpoint(
            model_state, 
            optimizer_state,
            metadata["current_epoch"],
            metadata["current_batch"]
        )
        print("Final checkpoint saved. Shutting down.")
        sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description="ThunderCompute Checkpoint Manager")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Initialize command
    init_parser = subparsers.add_parser("init", help="Initialize checkpoint directory")
    init_parser.add_argument("--dir", default=DEFAULT_CHECKPOINT_DIR, help="Checkpoint directory")
    
    # Save command
    save_parser = subparsers.add_parser("save", help="Save a checkpoint")
    save_parser.add_argument("--dir", default=DEFAULT_CHECKPOINT_DIR, help="Checkpoint directory")
    save_parser.add_argument("--epoch", type=int, required=True, help="Current epoch")
    save_parser.add_argument("--batch", type=int, default=0, help="Current batch")
    
    # Load command
    load_parser = subparsers.add_parser("load", help="Load a checkpoint")
    load_parser.add_argument("--dir", default=DEFAULT_CHECKPOINT_DIR, help="Checkpoint directory")
    load_parser.add_argument("--best", action="store_true", help="Load best checkpoint instead of latest")
    
    # Clean command
    clean_parser = subparsers.add_parser("clean", help="Clean old checkpoints")
    clean_parser.add_argument("--dir", default=DEFAULT_CHECKPOINT_DIR, help="Checkpoint directory")
    clean_parser.add_argument("--keep", type=int, default=5, help="Number of checkpoints to keep")
    
    args = parser.parse_args()
    
    if args.command == "init":
        checkpoint_manager = CheckpointManager(args.dir)
        print(f"Checkpoint directory initialized at {args.dir}")
    
    elif args.command == "save":
        checkpoint_manager = CheckpointManager(args.dir)
        # In a real implementation, this would get the current model state
        # For demonstration, we'll just use placeholder data
        model_state = {"placeholder": "model_state"}
        optimizer_state = {"placeholder": "optimizer_state"}
        
        checkpoint_manager.save_checkpoint(
            model_state,
            optimizer_state,
            args.epoch,
            args.batch
        )
    
    elif args.command == "load":
        checkpoint_manager = CheckpointManager(args.dir)
        if args.best:
            model_state, optimizer_state, epoch, batch = checkpoint_manager.load_best_checkpoint()
        else:
            model_state, optimizer_state, epoch, batch = checkpoint_manager.load_latest_checkpoint()
        
        print(f"Loaded checkpoint: epoch={epoch}, batch={batch}")
    
    elif args.command == "clean":
        checkpoint_manager = CheckpointManager(args.dir)
        metadata = checkpoint_manager._load_metadata()
        
        if len(metadata["checkpoints"]) <= args.keep:
            print(f"Only {len(metadata['checkpoints'])} checkpoints exist, nothing to clean.")
            return
        
        # Sort checkpoints by timestamp (newest first)
        checkpoints = sorted(
            metadata["checkpoints"],
            key=lambda x: x["timestamp"],
            reverse=True
        )
        
        # Keep the best checkpoint if it exists
        checkpoints_to_keep = checkpoints[:args.keep]
        if metadata["best_checkpoint"] is not None:
            best_path = metadata["best_checkpoint"]["path"]
            if not any(c["path"] == best_path for c in checkpoints_to_keep):
                checkpoints_to_keep.append(metadata["best_checkpoint"])
        
        # Remove old checkpoints
        keep_paths = [c["path"] for c in checkpoints_to_keep]
        for checkpoint in metadata["checkpoints"]:
            if checkpoint["path"] not in keep_paths:
                try:
                    checkpoint_path = Path(checkpoint["path"])
                    if checkpoint_path.exists():
                        os.remove(f"{checkpoint_path}.json")
                    print(f"Removed old checkpoint: {checkpoint_path}")
                except Exception as e:
                    print(f"Error removing checkpoint {checkpoint['path']}: {e}")
        
        # Update metadata
        metadata["checkpoints"] = [c for c in metadata["checkpoints"] if c["path"] in keep_paths]
        checkpoint_manager._save_metadata(metadata)
        print(f"Cleaned checkpoints, kept {len(checkpoints_to_keep)} newest/best checkpoints.")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()