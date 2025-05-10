#!/bin/bash
set -e

# Setup environment variables if not already set
export PYTHONPATH=/app:$PYTHONPATH

# Create directories for persistent storage if they don't exist
mkdir -p /app/data/models
mkdir -p /app/data/checkpoints
mkdir -p /app/data/logs

# Check if we need to run database migrations
if [ "$RUN_MIGRATIONS" = "true" ]; then
    echo "Running database migrations..."
    python -m src.utils.db_migrations
fi

# Execute the command passed to docker run
exec "$@"