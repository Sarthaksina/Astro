# Base Docker image for data preprocessing workloads on ThunderCompute
# Optimized for cost-effective GPUs like RTX 3080

FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    software-properties-common \
    python3-dev \
    python3-pip \
    python3-setuptools \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python
RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Install data processing libraries
RUN pip install --no-cache-dir \
    numpy \
    pandas \
    dask[complete] \
    pyarrow \
    fastparquet \
    matplotlib \
    seaborn \
    scikit-learn \
    jupyterlab \
    ipywidgets \
    tqdm \
    swisseph \
    skyfield \
    astropy \
    jplephem \
    numba \
    vaex \
    modin[ray]

# Install RAPIDS for GPU-accelerated data processing (compatible with RTX 3080)
RUN pip install --no-cache-dir \
    cudf-cu11 \
    cuml-cu11 \
    cugraph-cu11 \
    --extra-index-url=https://pypi.nvidia.com

# Set up working directory
WORKDIR /workspace

# Add scripts for instance management
COPY scripts/shutdown_idle.py /usr/local/bin/shutdown_idle
COPY scripts/monitor.py /usr/local/bin/monitor

RUN chmod +x /usr/local/bin/shutdown_idle /usr/local/bin/monitor

# Set up entrypoint script
COPY scripts/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]

# Default command
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]