#!/usr/bin/env python
# Cosmic Market Oracle - Configuration Utility

"""
Configuration utility for the Cosmic Market Oracle.

This module provides utilities for loading and accessing configuration settings
from YAML files, with support for environment variable substitution.
"""

import os
import yaml
from typing import Any, Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


class Config:
    """
    Configuration manager for the Cosmic Market Oracle.
    
    This class handles loading configuration from YAML files and provides
    methods for accessing configuration values with support for nested keys.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to the configuration YAML file.
        """
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict:
        """
        Load configuration from a YAML file.
        
        Returns:
            Dictionary containing configuration values.
        """
        if not os.path.exists(self.config_path):
            logger.warning(f"Configuration file not found: {self.config_path}")
            return {}
        
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Substitute environment variables
            config = self._substitute_env_vars(config)
            
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return {}
    
    def _substitute_env_vars(self, config: Dict) -> Dict:
        """
        Substitute environment variables in configuration values.
        
        Args:
            config: Configuration dictionary.
            
        Returns:
            Configuration with environment variables substituted.
        """
        if isinstance(config, dict):
            return {k: self._substitute_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._substitute_env_vars(item) for item in config]
        elif isinstance(config, str) and config.startswith("${") and config.endswith("}"):
            # Extract environment variable name
            env_var = config[2:-1]
            # Get value from environment, or use empty string if not found
            return os.environ.get(env_var, "")
        else:
            return config
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key.
        
        Args:
            key: Configuration key, can use dot notation for nested keys.
            default: Default value to return if key is not found.
            
        Returns:
            Configuration value, or default if not found.
        """
        # Handle nested keys with dot notation
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value.
        
        Args:
            key: Configuration key, can use dot notation for nested keys.
            value: Value to set.
        """
        # Handle nested keys with dot notation
        keys = key.split('.')
        config = self.config
        
        # Navigate to the nested dictionary
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
    
    def save(self, path: Optional[str] = None) -> None:
        """
        Save the configuration to a YAML file.
        
        Args:
            path: Path to save the configuration to. If None, uses the original path.
        """
        save_path = path or self.config_path
        
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Save configuration
            with open(save_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            
            logger.info(f"Configuration saved to {save_path}")
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")


def load_config(config_path: str) -> Dict:
    """
    Convenience function to load configuration from a YAML file.
    
    Args:
        config_path: Path to the configuration YAML file.
        
    Returns:
        Dictionary containing configuration values.
    """
    config = Config(config_path)
    return config.config


if __name__ == "__main__":
    import argparse
    import json
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Configuration utility for Cosmic Market Oracle")
    parser.add_argument("--config", required=True, help="Path to configuration file")
    parser.add_argument("--get", help="Get a configuration value by key")
    parser.add_argument("--set", help="Set a configuration value (requires --value)")
    parser.add_argument("--value", help="Value to set (used with --set)")
    parser.add_argument("--save", help="Save configuration to a file")
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config(args.config)
    
    # Get configuration value
    if args.get:
        value = config.get(args.get)
        print(json.dumps(value, indent=2))
    
    # Set configuration value
    if args.set and args.value:
        # Try to parse value as JSON, fall back to string if it fails
        try:
            value = json.loads(args.value)
        except json.JSONDecodeError:
            value = args.value
        
        config.set(args.set, value)
        print(f"Set {args.set} to {value}")
    
    # Save configuration
    if args.save:
        config.save(args.save)
        print(f"Configuration saved to {args.save}")
