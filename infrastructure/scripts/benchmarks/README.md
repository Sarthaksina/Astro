# ThunderCompute GPU Benchmarking Suite

This directory contains benchmark scripts for evaluating the performance and cost-efficiency of different GPU types on ThunderCompute. These benchmarks help optimize instance selection for various workloads.

## Available Benchmarks

### Matrix Multiplication (`matrix_multiply.py`)
Benchmarks large matrix multiplication operations, measuring GFLOPS/s performance across different matrix sizes. This benchmark is particularly sensitive to memory bandwidth and raw computational power.

### Convolutional Network Training (`conv_network.py`)
Benchmarks training performance of a convolutional neural network on synthetic image data. Measures samples processed per second and is particularly relevant for deep learning workloads.

### Data Processing (`data_processing.py`)
Benchmarks common data processing operations like normalization, filtering, and aggregation on large datasets. This benchmark is relevant for data preprocessing pipelines.

### Transformer Model Training (`transformer.py`)
Benchmarks training performance of a transformer model on synthetic text data. Measures tokens processed per second and is particularly relevant for NLP workloads.

### Reinforcement Learning (`reinforcement.py`)
Benchmarks training a reinforcement learning agent in a simulated environment. Measures environment steps per second and is relevant for RL workloads.

## Running Benchmarks

These benchmark scripts are designed to be run by the main `benchmark.py` tool in the parent directory. Each script:

1. Creates a synthetic workload appropriate for the benchmark type
2. Performs warmup iterations to stabilize GPU performance
3. Measures execution time and calculates performance metrics
4. Outputs results in a standardized JSON format

## Output Format

Each benchmark outputs a JSON object with the following common fields:

- `device`: The GPU device name
- `performance`: The primary performance metric (varies by benchmark)
- `unit`: The unit of the performance metric
- `higher_is_better`: Boolean indicating if higher values are better

Additional benchmark-specific metrics are also included.

## Interpreting Results

The main `benchmark.py` tool processes these results to:

1. Calculate cost-efficiency metrics based on GPU hourly rates
2. Generate comparison reports across different GPU types
3. Visualize performance and cost-efficiency tradeoffs
4. Recommend optimal GPU types for specific workloads

## Requirements

These benchmarks require:

- Python 3.6+
- PyTorch with CUDA support
- NumPy
- Matplotlib (for visualization)

## Adding New Benchmarks

To add a new benchmark:

1. Create a new Python script in this directory
2. Implement the benchmark following the pattern in existing scripts
3. Ensure the script outputs results in the standardized JSON format
4. Add the benchmark to the `WORKLOADS` dictionary in `benchmark.py`