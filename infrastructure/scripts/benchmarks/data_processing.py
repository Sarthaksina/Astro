#!/usr/bin/env python3
"""
Data Processing Benchmark

This script benchmarks data processing performance on GPUs.
It measures the time taken to perform common data processing operations
and outputs performance metrics in a standardized JSON format.
"""

import json
import time
import numpy as np
import torch
from src.trading.unified_mcts import MCTS, MCTSPredictor
from src.trading.modular_hierarchical_rl import ModularHierarchicalRLAgent, MCTSStrategicPlanner, PPOTacticalExecutor

# Configure benchmark parameters
DATASET_SIZES = [1000000, 5000000, 10000000]  # Number of samples
FEATURE_DIM = 128  # Feature dimension
WARMUP_ITERATIONS = 2
BENCHMARK_ITERATIONS = 3


def benchmark_data_processing():
    """Run data processing benchmark on the current GPU.
    
    Returns:
        Dictionary containing benchmark results
    """
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("Error: CUDA is not available. This benchmark requires a GPU.")
        return {"error": "CUDA not available"}
    
    # Get device information
    device = torch.device("cuda")
    device_name = torch.cuda.get_device_name(0)
    print(f"Running benchmark on: {device_name}")
    
    results = {}
    total_samples_processed = 0
    total_time = 0
    
    # Run benchmarks for different dataset sizes
    for size in DATASET_SIZES:
        print(f"\nBenchmarking with {size} samples...")
        
        # Create synthetic dataset
        data = torch.rand(size, FEATURE_DIM, device=device)
        
        # Warmup runs
        for _ in range(WARMUP_ITERATIONS):
            torch.cuda.synchronize()
            # Common data processing operations
            _ = torch.nn.functional.normalize(data, dim=1)  # Normalization
            _ = torch.sort(data, dim=1)[0]  # Sorting
            _ = torch.topk(data, k=10, dim=1)[0]  # Top-k selection
            torch.cuda.synchronize()
        
        # Benchmark runs
        times = []
        for i in range(BENCHMARK_ITERATIONS):
            torch.cuda.synchronize()
            start_time = time.time()
            
            # Perform a series of common data processing operations
            # 1. Normalization
            normalized = torch.nn.functional.normalize(data, dim=1)
            
            # 2. Filtering
            mask = torch.sum(normalized > 0.5, dim=1) > (FEATURE_DIM // 4)
            filtered = normalized[mask]
            
            # 3. Aggregation
            means = torch.mean(filtered, dim=0)
            stds = torch.std(filtered, dim=0)
            
            # 4. Transformation
            transformed = (filtered - means) / (stds + 1e-8)
            
            # 5. Sorting and top-k
            _, indices = torch.sort(transformed, dim=1, descending=True)
            topk = torch.topk(transformed, k=min(10, FEATURE_DIM), dim=1)[0]
            
            torch.cuda.synchronize()
            end_time = time.time()
            
            iteration_time = end_time - start_time
            times.append(iteration_time)
            print(f"  Iteration {i+1}: {iteration_time:.4f} seconds")
        
        # Calculate statistics
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        samples_per_second = size / avg_time
        
        # Store results for this dataset size
        results[f"{size}_samples"] = {
            "avg_time": avg_time,
            "min_time": min_time,
            "max_time": max_time,
            "samples_per_second": samples_per_second
        }
        
        # Add to totals for overall performance metric
        total_samples_processed += size * BENCHMARK_ITERATIONS
        total_time += avg_time * BENCHMARK_ITERATIONS
    
    # Calculate overall performance metric
    overall_samples_per_second = total_samples_processed / total_time
    
    # Prepare final results
    benchmark_results = {
        "device": device_name,
        "dataset_sizes": DATASET_SIZES,
        "feature_dim": FEATURE_DIM,
        "results": results,
        "performance": overall_samples_per_second,  # Overall samples/second as the main performance metric
        "unit": "samples/second",
        "higher_is_better": True
    }
    
    return benchmark_results


def main():
    """Run the benchmark and output results as JSON."""
    results = benchmark_data_processing()
    print(json.dumps(results))


if __name__ == "__main__":
    main()