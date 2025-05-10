#!/bin/bash
# Docker entrypoint script for ThunderCompute GPU instances

set -e

# Configure environment based on instance type
if [ -n "$INSTANCE_TYPE" ]; then
    echo "Configuring environment for instance type: $INSTANCE_TYPE"
    case "$INSTANCE_TYPE" in
        "deep_learning")
            # Configure for deep learning workloads
            export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
            export TF_FORCE_GPU_ALLOW_GROWTH=true
            ;;
        "data_preprocessing")
            # Configure for data preprocessing workloads
            export NUMBA_CACHE_DIR=/tmp/numba_cache
            export RAPIDS_MEMORY_LIMIT=75
            ;;
        "reinforcement_learning")
            # Configure for reinforcement learning workloads
            export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1
            export RAY_BACKEND_LOG_LEVEL=warning
            ;;
        *)
            echo "Unknown instance type: $INSTANCE_TYPE, using default configuration"
            ;;
    esac
fi

# Set up automatic checkpointing if enabled
if [ "$AUTO_CHECKPOINT" = "true" ]; then
    echo "Setting up automatic checkpointing"
    CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL:-15}
    nohup python -m checkpoint init --dir=${CHECKPOINT_DIR:-/workspace/checkpoints} > /var/log/checkpoint.log 2>&1 &
    echo "Automatic checkpointing configured with interval: $CHECKPOINT_INTERVAL minutes"
fi

# Set up automatic shutdown for idle instances if enabled
if [ "$AUTO_SHUTDOWN" = "true" ]; then
    echo "Setting up automatic shutdown for idle instances"
    IDLE_THRESHOLD=${IDLE_THRESHOLD:-30}
    nohup python -m shutdown_idle --threshold=$IDLE_THRESHOLD > /var/log/shutdown.log 2>&1 &
    echo "Automatic shutdown configured with idle threshold: $IDLE_THRESHOLD minutes"
fi

# Start GPU monitoring if enabled
if [ "$ENABLE_MONITORING" = "true" ]; then
    echo "Starting GPU monitoring"
    INSTANCE_ID=${INSTANCE_ID:-$(hostname)}
    MONITOR_INTERVAL=${MONITOR_INTERVAL:-60}
    nohup python -m monitor monitor $INSTANCE_ID --interval=$MONITOR_INTERVAL > /var/log/monitor.log 2>&1 &
    echo "GPU monitoring started for instance: $INSTANCE_ID"
fi

# Print GPU information
echo "GPU Information:"
nvidia-smi

# Execute the command passed to the entrypoint
exec "$@"