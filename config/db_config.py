#!/usr/bin/env python
"""
Centralized Database Configuration for Cosmic Market Oracle

Single source of truth for all database connection parameters.
"""

import os
from typing import Dict, Any

# Default database parameters
DEFAULT_DB_PARAMS = {
    'host': os.environ.get('DB_HOST', 'localhost'),
    'port': int(os.environ.get('DB_PORT', 5432)),
    'user': os.environ.get('DB_USER', 'postgres'),
    'password': os.environ.get('DB_PASSWORD', 'postgres'),
    'database': os.environ.get('DB_NAME', 'cosmic_oracle'),
    'schema': os.environ.get('DB_SCHEMA', 'cosmic_data')
}

def get_db_params() -> Dict[str, Any]:
    """Get database connection parameters."""
    return DEFAULT_DB_PARAMS.copy()

def get_connection_string() -> str:
    """Get database connection string."""
    params = get_db_params()
    return f"postgresql://{params['user']}:{params['password']}@{params['host']}:{params['port']}/{params['database']}"

# For backward compatibility
DB_PARAMS = DEFAULT_DB_PARAMS