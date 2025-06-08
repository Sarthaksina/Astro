import logging
import logging.handlers
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

# Custom log levels for market events
MARKET_SIGNAL = 25
ASTRO_EVENT = 26

# Register custom log levels
logging.addLevelName(MARKET_SIGNAL, "MARKET_SIGNAL")
logging.addLevelName(ASTRO_EVENT, "ASTRO_EVENT")

class MarketLogger(logging.Logger):
    """
    Custom logger class with additional methods for market-specific logging.
    """
    def market_signal(self, msg, *args, **kwargs):
        """Log a market signal event"""
        if self.isEnabledFor(MARKET_SIGNAL):
            self._log(MARKET_SIGNAL, msg, args, **kwargs)

    def astro_event(self, msg, *args, **kwargs):
        """Log an astrological event"""
        if self.isEnabledFor(ASTRO_EVENT):
            self._log(ASTRO_EVENT, msg, args, **kwargs)

# Register the custom logger class
# This should be done before any loggers are instantiated if they are to use this class
logging.setLoggerClass(MarketLogger)

_default_logger_instance = None

def setup_logger(
    log_dir: str = "logs",
    log_level: int = logging.INFO,
    log_format: Optional[str] = None,
    log_file_prefix: str = "cosmic_market_oracle",
    logger_name: str = "cosmic_market_oracle"
) -> logging.Logger:
    """
    Configure logging for the Cosmic Market Oracle application.

    Args:
        log_dir: Directory to store log files
        log_level: Logging level (default: INFO)
        log_format: Custom log format string (optional)
        log_file_prefix: Prefix for log file names
        logger_name: The name of the logger to configure.

    Returns:
        Configured logger instance
    """
    global _default_logger_instance

    # Create log directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Generate log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"{log_file_prefix}_{timestamp}_{logger_name}.log"

    # Set up logging format
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s"
    formatter = logging.Formatter(log_format)

    # Configure file handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(formatter)

    # Configure console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Get the logger instance (it might be an instance of MarketLogger)
    logger = logging.getLogger(logger_name)

    # Check if handlers are already present to avoid duplication if called multiple times
    if not logger.handlers:
        logger.setLevel(log_level)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        logger.propagate = False # Avoid double-logging if root logger is also configured
        logger.info(f"Logging initialized for {logger_name}. Log file: {log_file}")
    else:
        logger.info(f"Logger {logger_name} already configured.")

    if logger_name == "cosmic_market_oracle": # Default logger
        _default_logger_instance = logger

    return logger

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Retrieves a logger instance.
    If name is None, it returns the default 'cosmic_market_oracle' logger,
    ensuring it's configured if it hasn't been already.
    """
    global _default_logger_instance

    if name is None:
        if _default_logger_instance is None:
            # Setup the default logger if it hasn't been explicitly set up.
            # This provides a basic default configuration.
            _default_logger_instance = setup_logger(logger_name="cosmic_market_oracle")
        return _default_logger_instance

    # For named loggers, assume they might need their own setup or rely on root.
    # For simplicity here, we'll just get the logger. If it's not configured,
    # it will propagate to the root logger (if configured).
    # A more sophisticated setup might involve configuring it if not already done.
    return logging.getLogger(name)

# Example of setting up a default logger when the module is imported,
# if that's a desired behavior. Otherwise, setup_logger should be called explicitly.
# setup_logger(logger_name="cosmic_market_oracle_default")

# Ensure the custom logger class is set *before* any loggers are created implicitly or explicitly
# if you want them to use MarketLogger by default when getLogger is called for the first time
# for a given name.
# Moved setLoggerClass to the top of the file.
