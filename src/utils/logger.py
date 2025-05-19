#!/usr/bin/env python
# Cosmic Market Oracle - Logging Utility

"""
[DEPRECATED] Use logging_config.py instead.

Logging utility for the Cosmic Market Oracle.

This module provides a standardized logging setup for all components of the Cosmic Market Oracle,
ensuring consistent log formatting and handling across the application.
"""

import os
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


from .logging_config import setup_logging, get_logger

def setup_logger(name: str, level: int = logging.INFO, 
                log_dir: Optional[str] = None, 
                console: bool = True) -> logging.Logger:
    """
    Set up a logger with consistent formatting and handlers.
    
    Args:
        name: Name of the logger.
        level: Logging level (default: INFO).
        log_dir: Directory to store log files. If None, logs are only sent to console.
        console: Whether to log to console (default: True).
        
    Returns:
        Configured logger instance.
    """
    # Use centralized logging configuration
    if log_dir:
        return setup_logging(log_dir=log_dir, log_level=level)
    else:
        return get_logger(name)

# Re-export get_logger from logging_config
get_logger = get_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get an existing logger or create a new one with default settings.
    
    Args:
        name: Name of the logger.
        
    Returns:
        Logger instance.
    """
    logger = logging.getLogger(name)
    
    # If logger doesn't have handlers, set it up with defaults
    if not logger.handlers:
        # Get log directory from environment or use default
        log_dir = os.environ.get("LOG_DIR", "logs")
        return setup_logger(name, log_dir=log_dir)
    
    return logger


class LoggerContext:
    """
    Context manager for temporarily changing log level.
    
    This is useful for suppressing verbose logs in certain sections of code.
    """
    
    def __init__(self, logger: logging.Logger, level: int):
        """
        Initialize the context manager.
        
        Args:
            logger: The logger to modify.
            level: The temporary log level.
        """
        self.logger = logger
        self.level = level
        self.original_level = logger.level
    
    def __enter__(self):
        """Enter the context."""
        self.logger.setLevel(self.level)
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context and restore the original log level."""
        self.logger.setLevel(self.original_level)


def suppress_logs(logger: logging.Logger, level: int = logging.WARNING):
    """
    Create a context manager to temporarily suppress logs below the specified level.
    
    Args:
        logger: Logger to modify.
        level: Minimum level to log (default: WARNING).
        
    Returns:
        Context manager for log suppression.
    """
    return LoggerContext(logger, level)


if __name__ == "__main__":
    # Example usage
    logger = setup_logger("example", log_dir="logs")
    
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    # Suppress logs temporarily
    with suppress_logs(logger, logging.ERROR):
        logger.info("This info message will be suppressed")
        logger.warning("This warning message will be suppressed")
        logger.error("This error message will still be shown")
    
    # Log level is restored
    logger.info("This info message will be shown again")
