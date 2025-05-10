# Base Docker image for reinforcement learning workloads on ThunderCompute
# Optimized for multi-GPU setups (A100, etc.)

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    software-properties-common \
    build-essential \
    python3-dev \
    python3-pip \
    python3-setuptools \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python
RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir \
    torch==2.0.1+cu118 \
    torchvision==0.15.2+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118

# Install reinforcement learning libraries
RUN pip install --no-cache-dir \
    numpy \
    pandas \
    matplotlib \
    gym \
    gymnasium \
    stable-baselines3 \
    ray[rllib,tune,serve] \
    mlagents-envs \
    tensorboard \
    optuna \
    wandb \
    jupyterlab \
    ipywidgets \
    tqdm \
    pettingzoo \
    supersuit \
    dm-control \
    procgen \
    ale-py \
    autorom[accept-rom-license]

# Install multi-GPU support libraries
RUN pip install --no-cache-dir \
    horovod \
    mpi4py \
    dask-cuda \
    cupy-cuda11x

# Set up working directory
WORKDIR /workspace

# Add scripts for model checkpointing and instance management
COPY scripts/checkpoint.py /usr/local/bin/checkpoint
COPY scripts/shutdown_idle.py /usr/local/bin/shutdown_idle
COPY scripts/monitor.py /usr/local/bin/monitor

RUN chmod +x /usr/local/bin/checkpoint /usr/local/bin/shutdown_idle /usr/local/bin/monitor

# Set up entrypoint script
COPY scripts/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Configure environment for multi-GPU training
ENV RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1
ENV RAY_BACKEND_LOG_LEVEL=warning
ENV TF_FORCE_GPU_ALLOW_GROWTH=true

ENTRYPOINT ["/entrypoint.sh"]

# Default command
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]