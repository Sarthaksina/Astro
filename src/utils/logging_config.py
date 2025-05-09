import logging
import logging.handlers
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

def setup_logging(
    log_dir: str = "logs",
    log_level: int = logging.INFO,
    log_format: Optional[str] = None,
    log_file_prefix: str = "cosmic_market_oracle"
) -> logging.Logger:
    """
    Configure logging for the Cosmic Market Oracle application.
    
    Args:
        log_dir: Directory to store log files
        log_level: Logging level (default: INFO)
        log_format: Custom log format string (optional)
        log_file_prefix: Prefix for log file names
        
    Returns:
        Configured logger instance
    """
    # Create log directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Generate log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"{log_file_prefix}_{timestamp}.log"
    
    # Set up logging format
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
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
    
    # Set up logger
    logger = logging.getLogger("cosmic_market_oracle")
    logger.setLevel(log_level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Log initial setup message
    logger.info(f"Logging initialized. Log file: {log_file}")
    
    return logger

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
        self.log(MARKET_SIGNAL, msg, *args, **kwargs)
    
    def astro_event(self, msg, *args, **kwargs):
        """Log an astrological event"""
        self.log(ASTRO_EVENT, msg, *args, **kwargs)

# Register the custom logger class
logging.setLoggerClass(MarketLogger)

def get_logger(name: str = "cosmic_market_oracle") -> MarketLogger:
    """
    Get a logger instance with the specified name.
    
    Args:
        name: Logger name (default: cosmic_market_oracle)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)