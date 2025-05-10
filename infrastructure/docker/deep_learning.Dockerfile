# Base Docker image for deep learning workloads on ThunderCompute
# Optimized for RTX 4090 and A100 GPUs

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
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python
RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir \
    torch==2.0.1+cu118 \
    torchvision==0.15.2+cu118 \
    torchaudio==2.0.2 \
    --extra-index-url https://download.pytorch.org/whl/cu118

# Install TensorFlow
RUN pip install --no-cache-dir tensorflow==2.12.0

# Install astronomical libraries and other dependencies
RUN pip install --no-cache-dir \
    numpy \
    pandas \
    matplotlib \
    scikit-learn \
    jupyterlab \
    ipywidgets \
    tqdm \
    pyarrow \
    swisseph \
    skyfield \
    astropy \
    jplephem \
    ray[tune] \
    mlflow \
    optuna

# Set up working directory
WORKDIR /workspace

# Add scripts for model checkpointing and instance management
COPY scripts/checkpoint.py /usr/local/bin/checkpoint
COPY scripts/shutdown_idle.py /usr/local/bin/shutdown_idle

RUN chmod +x /usr/local/bin/checkpoint /usr/local/bin/shutdown_idle

# Set up entrypoint script
COPY scripts/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]

# Default command
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]