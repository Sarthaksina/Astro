"""
Database configuration - Redirects to consolidated TimescaleDB connector.
This file is kept for backward compatibility.
"""

# Import from the consolidated database utility
from src.utils.db.timescale_connector import TimescaleConnector, get_connector
from config.db_config import get_connection_string, get_db_params
from sqlalchemy.ext.declarative import declarative_base

# Create Base for backward compatibility with existing models
Base = declarative_base()

# For backward compatibility
def get_db():
    """Database dependency for FastAPI - redirects to TimescaleConnector"""
    connector = get_connector()
    return connector.get_session()

def get_engine():
    """Get database engine from TimescaleConnector"""
    connector = get_connector()
    return connector.engine

# Legacy aliases for existing code
DATABASE_URL = get_connection_string()
DB_PARAMS = get_db_params()

# Export commonly used items
__all__ = [
    'Base',
    'get_db', 
    'get_engine',
    'DATABASE_URL',
    'DB_PARAMS',
    'TimescaleConnector',
    'get_connector'
]