#!/usr/bin/env python3
"""
Matrix Multiplication Benchmark

This script benchmarks matrix multiplication performance on GPUs.
It measures the time taken to perform large matrix multiplications
and outputs performance metrics in a standardized JSON format.
"""

import json
import time
import numpy as np
import torch
from src.trading.unified_mcts import MCTS, MCTSPredictor
from src.trading.modular_hierarchical_rl import ModularHierarchicalRLAgent, MCTSStrategicPlanner, PPOTacticalExecutor

# Configure benchmark parameters
MATRIX_SIZES = [1024, 2048, 4096, 8192]  # Matrix dimensions to test
WARMUP_ITERATIONS = 3     # Number of warmup iterations
BENCHMARK_ITERATIONS = 5  # Number of iterations for actual benchmarking


def benchmark_matrix_multiply():
    """Run matrix multiplication benchmark on the current GPU.
    
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
    total_flops = 0
    total_time = 0
    
    # Run benchmarks for different matrix sizes
    for size in MATRIX_SIZES:
        print(f"\nBenchmarking {size}x{size} matrix multiplication...")
        
        # Create random matrices
        matrix_a = torch.rand(size, size, device=device)
        matrix_b = torch.rand(size, size, device=device)
        
        # Warmup runs
        for _ in range(WARMUP_ITERATIONS):
            torch.cuda.synchronize()
            _ = torch.matmul(matrix_a, matrix_b)
            torch.cuda.synchronize()
        
        # Benchmark runs
        times = []
        for i in range(BENCHMARK_ITERATIONS):
            torch.cuda.synchronize()
            start_time = time.time()
            _ = torch.matmul(matrix_a, matrix_b)
            torch.cuda.synchronize()
            end_time = time.time()
            
            iteration_time = end_time - start_time
            times.append(iteration_time)
            print(f"  Iteration {i+1}: {iteration_time:.4f} seconds")
        
        # Calculate statistics
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        # Calculate FLOPS (Floating Point Operations Per Second)
        # For matrix multiplication: 2 * N^3 - N^2 operations
        # We'll use the simplified 2 * N^3 for large matrices
        flops = 2 * (size ** 3)
        flops_per_second = flops / avg_time
        gflops_per_second = flops_per_second / 1e9
        
        # Store results for this matrix size
        results[f"{size}x{size}"] = {
            "avg_time": avg_time,
            "min_time": min_time,
            "max_time": max_time,
            "gflops_per_second": gflops_per_second
        }
        
        # Add to totals for overall performance metric
        total_flops += flops
        total_time += avg_time * BENCHMARK_ITERATIONS
    
    # Calculate overall performance metric
    overall_gflops_per_second = (total_flops / total_time) / 1e9
    
    # Prepare final results
    benchmark_results = {
        "device": device_name,
        "matrix_sizes": MATRIX_SIZES,
        "results": results,
        "performance": overall_gflops_per_second,  # Overall GFLOPS/s as the main performance metric
        "unit": "GFLOPS/s",
        "higher_is_better": True
    }
    
    return benchmark_results


def main():
    """Run the benchmark and output results as JSON."""
    results = benchmark_matrix_multiply()
    print(json.dumps(results))


if __name__ == "__main__":
    main()