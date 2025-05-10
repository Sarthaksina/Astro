#!/usr/bin/env python3
"""
ThunderCompute GPU Benchmarking Tool

This script provides utilities for benchmarking different GPU instance types
on ThunderCompute to evaluate cost-efficiency for various workloads.
Results help optimize instance selection for different tasks.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# Configuration
BENCHMARK_RESULTS_DIR = Path("./benchmark_results")
GPU_TYPES = [
    "RTX 3080",
    "RTX 4090",
    "A100"
]

# Benchmark workloads
WORKLOADS = {
    "matrix_multiply": {
        "description": "Matrix multiplication benchmark",
        "script": "benchmarks/matrix_multiply.py"
    },
    "conv_network": {
        "description": "Convolutional network training benchmark",
        "script": "benchmarks/conv_network.py"
    },
    "data_processing": {
        "description": "Data processing benchmark",
        "script": "benchmarks/data_processing.py"
    },
    "transformer": {
        "description": "Transformer model training benchmark",
        "script": "benchmarks/transformer.py"
    },
    "reinforcement": {
        "description": "Reinforcement learning benchmark",
        "script": "benchmarks/reinforcement.py"
    }
}

# Cost estimates (USD per hour)
HOURLY_RATES = {
    "RTX 3080": 0.60,
    "RTX 4090": 1.20,
    "A100": 2.50,
}


class GPUBenchmark:
    """Benchmarks GPU performance and cost-efficiency."""
    
    def __init__(self, results_dir=None):
        """Initialize the benchmark tool.
        
        Args:
            results_dir: Directory to store benchmark results
        """
        self.results_dir = Path(results_dir or BENCHMARK_RESULTS_DIR)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamp for this benchmark run
        self.timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_dir = self.results_dir / f"run_{self.timestamp}"
        self.run_dir.mkdir(exist_ok=True)
    
    def detect_gpu(self):
        """Detect the GPU type of the current instance.
        
        Returns:
            String representing the GPU type
        """
        try:
            # Try to use nvidia-smi to get GPU info
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                check=True
            )
            gpu_name = result.stdout.strip()
            
            # Map the GPU name to our standard types
            if "3080" in gpu_name:
                return "RTX 3080"
            elif "4090" in gpu_name:
                return "RTX 4090"
            elif "A100" in gpu_name:
                return "A100"
            else:
                return gpu_name
        except (subprocess.SubprocessError, FileNotFoundError):
            print("Warning: Could not detect GPU using nvidia-smi.")
            return "Unknown"
    
    def run_benchmark(self, workload, iterations=3):
        """Run a specific benchmark workload.
        
        Args:
            workload: Name of the workload to run
            iterations: Number of times to run the benchmark
            
        Returns:
            Dictionary containing benchmark results
        """
        if workload not in WORKLOADS:
            print(f"Error: Unknown workload '{workload}'. Available workloads: {list(WORKLOADS.keys())}")
            return None
        
        workload_info = WORKLOADS[workload]
        script_path = workload_info["script"]
        
        print(f"Running benchmark: {workload} ({workload_info['description']})")
        print(f"Script: {script_path}")
        print(f"Iterations: {iterations}")
        
        # Check if script exists
        if not os.path.exists(script_path):
            print(f"Error: Benchmark script not found at {script_path}")
            return None
        
        # Run the benchmark multiple times
        results = []
        for i in range(iterations):
            print(f"Iteration {i+1}/{iterations}...")
            
            # Run the benchmark script
            start_time = time.time()
            try:
                result = subprocess.run(
                    ["python", script_path],
                    capture_output=True,
                    text=True,
                    check=True
                )
                output = result.stdout.strip()
                
                # Parse the output (assuming it's JSON)
                try:
                    metrics = json.loads(output)
                except json.JSONDecodeError:
                    # If not JSON, just use the raw output
                    metrics = {"output": output}
                
                # Add timing information
                execution_time = time.time() - start_time
                metrics["execution_time"] = execution_time
                
                results.append(metrics)
                print(f"Execution time: {execution_time:.2f} seconds")
            except subprocess.CalledProcessError as e:
                print(f"Error running benchmark: {e}")
                print(f"Stderr: {e.stderr}")
                return None
        
        # Calculate average metrics
        avg_results = {}
        for key in results[0].keys():
            if isinstance(results[0][key], (int, float)):
                avg_results[key] = sum(r[key] for r in results) / len(results)
        
        # Add metadata
        gpu_type = self.detect_gpu()
        avg_results["gpu_type"] = gpu_type
        avg_results["workload"] = workload
        avg_results["timestamp"] = self.timestamp
        avg_results["iterations"] = iterations
        
        # Calculate cost efficiency if we have hourly rates
        if gpu_type in HOURLY_RATES and "execution_time" in avg_results:
            hourly_rate = HOURLY_RATES[gpu_type]
            execution_hours = avg_results["execution_time"] / 3600
            cost = hourly_rate * execution_hours
            avg_results["cost"] = cost
            
            # If the workload has a performance metric, calculate cost efficiency
            if "performance" in avg_results:
                avg_results["cost_efficiency"] = avg_results["performance"] / cost
        
        # Save results
        results_file = self.run_dir / f"{workload}_{gpu_type}.json"
        with open(results_file, 'w') as f:
            json.dump(avg_results, f, indent=2)
        
        print(f"Benchmark results saved to {results_file}")
        return avg_results
    
    def run_all_benchmarks(self, iterations=3):
        """Run all available benchmark workloads.
        
        Args:
            iterations: Number of times to run each benchmark
            
        Returns:
            Dictionary mapping workload names to benchmark results
        """
        results = {}
        for workload in WORKLOADS:
            print(f"\n{'='*50}\nRunning workload: {workload}\n{'='*50}\n")
            result = self.run_benchmark(workload, iterations)
            if result:
                results[workload] = result
        
        # Save combined results
        combined_file = self.run_dir / "combined_results.json"
        with open(combined_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nAll benchmark results saved to {combined_file}")
        return results
    
    def generate_report(self, results=None):
        """Generate a benchmark report with visualizations.
        
        Args:
            results: Dictionary of benchmark results (if None, load from files)
            
        Returns:
            Path to the generated report file
        """
        if results is None:
            # Load results from files
            results = {}
            for workload_file in self.run_dir.glob("*.json"):
                if workload_file.name != "combined_results.json":
                    with open(workload_file, 'r') as f:
                        workload_results = json.load(f)
                        workload = workload_results["workload"]
                        results[workload] = workload_results
        
        # Generate report file
        report_file = self.run_dir / f"benchmark_report.html"
        charts_dir = self.run_dir / "charts"
        charts_dir.mkdir(exist_ok=True)
        
        # Generate performance charts
        self._generate_performance_charts(results, charts_dir)
        
        # Generate HTML report
        with open(report_file, 'w') as f:
            f.write(f"<html>\n<head>\n<title>GPU Benchmark Report</title>\n")
            f.write("<style>\n")
            f.write("body { font-family: Arial, sans-serif; margin: 20px; }\n")
            f.write("table { border-collapse: collapse; width: 100%; }\n")
            f.write("th, td { padding: 8px; text-align: left; border: 1px solid #ddd; }\n")
            f.write("th { background-color: #f2f2f2; }\n")
            f.write("tr:nth-child(even) { background-color: #f9f9f9; }\n")
            f.write("h1, h2, h3 { color: #333; }\n")
            f.write("img { max-width: 100%; height: auto; margin: 10px 0; }\n")
            f.write("</style>\n")
            f.write("</head>\n<body>\n")
            f.write(f"<h1>GPU Benchmark Report</h1>\n")
            f.write(f"<p>Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>\n")
            f.write(f"<p>GPU Type: {self.detect_gpu()}</p>\n")
            
            # Performance charts section
            f.write("<h2>Performance Charts</h2>\n")
            f.write("<div class='charts'>\n")
            f.write(f"<img src='charts/execution_time.png' alt='Execution Time Comparison'>\n")
            f.write(f"<img src='charts/performance.png' alt='Performance Comparison'>\n")
            f.write(f"<img src='charts/cost_efficiency.png' alt='Cost Efficiency Comparison'>\n")
            f.write("</div>\n")
            
            # Summary table
            f.write("<h2>Summary</h2>\n")
            f.write("<table>\n")
            f.write("<tr><th>Workload</th><th>Execution Time (s)</th><th>Cost ($)</th><th>Performance</th><th>Cost Efficiency</th></tr>\n")
            
            for workload, result in results.items():
                f.write(f"<tr>")
                f.write(f"<td>{workload}</td>")
                f.write(f"<td>{result.get('execution_time', 'N/A'):.2f}</td>")
                f.write(f"<td>{result.get('cost', 'N/A'):.4f}</td>")
                f.write(f"<td>{result.get('performance', 'N/A')}</td>")
                f.write(f"<td>{result.get('cost_efficiency', 'N/A')}</td>")
                f.write(f"</tr>\n")
            
            f.write("</table>\n")
            
            # Detailed results
            f.write("<h2>Detailed Results</h2>\n")
            for workload, result in results.items():
                f.write(f"<h3>{workload}</h3>\n")
                f.write("<pre>\n")
                f.write(json.dumps(result, indent=2))
                f.write("\n</pre>\n")
            
            f.write("</body>\n</html>")
        
        print(f"Report generated: {report_file}")
        return report_file
        
    def _generate_performance_charts(self, results, charts_dir):
        """Generate performance comparison charts.
        
        Args:
            results: Dictionary of benchmark results
            charts_dir: Directory to save the charts
        """
        # Extract data for charts
        workloads = list(results.keys())
        execution_times = [results[w].get('execution_time', 0) for w in workloads]
        performances = [results[w].get('performance', 0) for w in workloads]
        cost_efficiencies = [results[w].get('cost_efficiency', 0) for w in workloads]
        
        # Set up plot style
        plt.style.use('ggplot')
        
        # Create execution time chart
        plt.figure(figsize=(10, 6))
        bars = plt.bar(workloads, execution_times, color='skyblue')
        plt.title('Execution Time by Workload')
        plt.xlabel('Workload')
        plt.ylabel('Execution Time (seconds)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.2f}s', ha='center', va='bottom')
        
        plt.savefig(charts_dir / 'execution_time.png', dpi=300)
        plt.close()
        
        # Create performance chart
        plt.figure(figsize=(10, 6))
        bars = plt.bar(workloads, performances, color='lightgreen')
        plt.title('Performance by Workload')
        plt.xlabel('Workload')
        plt.ylabel('Performance (workload specific units)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.2f}', ha='center', va='bottom')
        
        plt.savefig(charts_dir / 'performance.png', dpi=300)
        plt.close()
        
        # Create cost efficiency chart
        plt.figure(figsize=(10, 6))
        bars = plt.bar(workloads, cost_efficiencies, color='salmon')
        plt.title('Cost Efficiency by Workload')
        plt.xlabel('Workload')
        plt.ylabel('Cost Efficiency (performance/cost)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.2f}', ha='center', va='bottom')
        
        plt.savefig(charts_dir / 'cost_efficiency.png', dpi=300)
        plt.close()
    
    def compare_gpu_types(self, workload, gpu_results):
        """Compare performance and cost-efficiency across GPU types.
        
        Args:
            workload: Name of the workload to compare
            gpu_results: Dictionary mapping GPU types to benchmark results
            
        Returns:
            Path to the generated comparison report file
        """
        # Generate comparison report file
        report_file = self.run_dir / f"{workload}_comparison.html"
        charts_dir = self.run_dir / "comparison_charts"
        charts_dir.mkdir(exist_ok=True)
        
        # Extract metrics for comparison
        execution_times = []
        costs = []
        performances = []
        cost_efficiencies = []
        gpu_types = []
        
        for gpu_type, result in gpu_results.items():
            gpu_types.append(gpu_type)
            execution_times.append(result.get("execution_time", 0))
            costs.append(result.get("cost", 0))
            performances.append(result.get("performance", 0))
            cost_efficiencies.append(result.get("cost_efficiency", 0))
        
        # Generate comparison charts
        self._generate_comparison_charts(workload, gpu_types, execution_times, 
                                        performances, costs, cost_efficiencies, charts_dir)
        
        # Generate HTML report
        with open(report_file, 'w') as f:
            f.write(f"<html>\n<head>\n<title>GPU Comparison Report - {workload}</title>\n")
            f.write("<style>\n")
            f.write("body { font-family: Arial, sans-serif; margin: 20px; }\n")
            f.write("table { border-collapse: collapse; width: 100%; }\n")
            f.write("th, td { padding: 8px; text-align: left; border: 1px solid #ddd; }\n")
            f.write("th { background-color: #f2f2f2; }\n")
            f.write("tr:nth-child(even) { background-color: #f9f9f9; }\n")
            f.write("h1, h2, h3 { color: #333; }\n")
            f.write("img { max-width: 100%; height: auto; margin: 10px 0; }\n")
            f.write(".chart-container { display: flex; flex-wrap: wrap; justify-content: space-between; }\n")
            f.write(".chart { width: 48%; margin-bottom: 20px; }\n")
            f.write(".recommendation { background-color: #f0f7ff; padding: 15px; border-radius: 5px; margin-top: 20px; }\n")
            f.write(".best { color: #2e7d32; font-weight: bold; }\n")
            f.write("@media (max-width: 768px) { .chart { width: 100%; } }\n")
            f.write("</style>\n")
            f.write("</head>\n<body>\n")
            f.write(f"<h1>GPU Comparison Report - {workload}</h1>\n")
            f.write(f"<p>Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>\n")
            
            # Comparison charts
            f.write("<h2>Performance Comparison</h2>\n")
            f.write("<div class='chart-container'>\n")
            f.write("<div class='chart'>\n")
            f.write(f"<img src='comparison_charts/{workload}_execution_time.png' alt='Execution Time Comparison'>\n")
            f.write("</div>\n")
            f.write("<div class='chart'>\n")
            f.write(f"<img src='comparison_charts/{workload}_performance.png' alt='Performance Comparison'>\n")
            f.write("</div>\n")
            f.write("<div class='chart'>\n")
            f.write(f"<img src='comparison_charts/{workload}_cost.png' alt='Cost Comparison'>\n")
            f.write("</div>\n")
            f.write("<div class='chart'>\n")
            f.write(f"<img src='comparison_charts/{workload}_cost_efficiency.png' alt='Cost Efficiency Comparison'>\n")
            f.write("</div>\n")
            f.write("</div>\n")
            
            # Comparison table
            f.write("<h2>Comparison Table</h2>\n")
            f.write("<table>\n")
            f.write("<tr><th>GPU Type</th><th>Execution Time (s)</th><th>Cost ($)</th><th>Performance</th><th>Cost Efficiency</th></tr>\n")
            
            # Find best values for highlighting
            best_performance_idx = performances.index(max(performances))
            best_time_idx = execution_times.index(min(execution_times))
            best_cost_idx = costs.index(min(costs))
            best_efficiency_idx = cost_efficiencies.index(max(cost_efficiencies))
            
            for i, gpu_type in enumerate(gpu_types):
                f.write(f"<tr>")
                f.write(f"<td>{gpu_type}</td>")
                
                # Highlight best values
                time_class = " class='best'" if i == best_time_idx else ""
                cost_class = " class='best'" if i == best_cost_idx else ""
                perf_class = " class='best'" if i == best_performance_idx else ""
                eff_class = " class='best'" if i == best_efficiency_idx else ""
                
                f.write(f"<td{time_class}>{execution_times[i]:.2f}</td>")
                f.write(f"<td{cost_class}>{costs[i]:.4f}</td>")
                f.write(f"<td{perf_class}>{performances[i]:.2f}</td>")
                f.write(f"<td{eff_class}>{cost_efficiencies[i]:.2f}</td>")
                f.write(f"</tr>\n")
            
            f.write("</table>\n")
            
            # Recommendations
            f.write("<div class='recommendation'>\n")
            f.write("<h2>Recommendations</h2>\n")
            
            f.write("<p>Based on the benchmark results, here are our recommendations:</p>\n")
            f.write("<ul>\n")
            f.write(f"<li><strong>Best Performance:</strong> {gpu_types[best_performance_idx]} - Ideal for time-critical workloads where processing speed is the primary concern.</li>\n")
            f.write(f"<li><strong>Lowest Cost:</strong> {gpu_types[best_cost_idx]} - Best option for budget-constrained scenarios or workloads that don't require maximum performance.</li>\n")
            f.write(f"<li><strong>Best Cost-Efficiency:</strong> {gpu_types[best_efficiency_idx]} - Optimal balance between performance and cost, recommended for most production workloads.</li>\n")
            f.write("</ul>\n")
            
            # Add workload-specific recommendations
            f.write("<p><strong>Workload-Specific Recommendation:</strong> ")
            if workload == "matrix_multiply":
                f.write("For matrix multiplication operations, prioritize GPUs with high memory bandwidth and CUDA cores.")
            elif workload == "conv_network":
                f.write("For convolutional neural networks, GPUs with tensor cores provide significant acceleration for training and inference.")
            elif workload == "data_processing":
                f.write("For data processing tasks, consider GPUs with larger memory capacity to handle bigger datasets.")
            elif workload == "transformer":
                f.write("For transformer models, GPUs with tensor cores and higher memory bandwidth are crucial for optimal performance.")
            elif workload == "reinforcement":
                f.write("For reinforcement learning, GPUs with good balance between compute power and memory are recommended.")
            f.write("</p>\n")
            
            f.write("</div>\n")
            
            f.write("</body>\n</html>")
        
        print(f"Comparison report generated: {report_file}")
        return report_file
        
    def _generate_comparison_charts(self, workload, gpu_types, execution_times, performances, costs, cost_efficiencies, charts_dir):
        """Generate charts comparing different GPU types for a specific workload.
        
        Args:
            workload: Name of the workload being compared
            gpu_types: List of GPU types
            execution_times: List of execution times
            performances: List of performance metrics
            costs: List of costs
            cost_efficiencies: List of cost efficiency metrics
            charts_dir: Directory to save the charts
        """
        # Set up plot style
        plt.style.use('ggplot')
        colors = ['#4285F4', '#EA4335', '#FBBC05', '#34A853', '#8334A4']
        
        # Create execution time chart
        plt.figure(figsize=(10, 6))
        bars = plt.bar(gpu_types, execution_times, color=colors[:len(gpu_types)])
        plt.title(f'Execution Time Comparison - {workload}')
        plt.xlabel('GPU Type')
        plt.ylabel('Execution Time (seconds)')
        plt.xticks(rotation=0)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.2f}s', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(charts_dir / f'{workload}_execution_time.png', dpi=300)
        plt.close()
        
        # Create performance chart
        plt.figure(figsize=(10, 6))
        bars = plt.bar(gpu_types, performances, color=colors[:len(gpu_types)])
        plt.title(f'Performance Comparison - {workload}')
        plt.xlabel('GPU Type')
        plt.ylabel('Performance')
        plt.xticks(rotation=0)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(charts_dir / f'{workload}_performance.png', dpi=300)
        plt.close()
        
        # Create cost chart
        plt.figure(figsize=(10, 6))
        bars = plt.bar(gpu_types, costs, color=colors[:len(gpu_types)])
        plt.title(f'Cost Comparison - {workload}')
        plt.xlabel('GPU Type')
        plt.ylabel('Cost ($)')
        plt.xticks(rotation=0)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'${height:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(charts_dir / f'{workload}_cost.png', dpi=300)
        plt.close()
        
        # Create cost efficiency chart
        plt.figure(figsize=(10, 6))
        bars = plt.bar(gpu_types, cost_efficiencies, color=colors[:len(gpu_types)])
        plt.title(f'Cost Efficiency Comparison - {workload}')
        plt.xlabel('GPU Type')
        plt.ylabel('Cost Efficiency (performance/cost)')
        plt.xticks(rotation=0)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(charts_dir / f'{workload}_cost_efficiency.png', dpi=300)
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="ThunderCompute GPU Benchmarking Tool")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Run benchmarks")
    run_parser.add_argument("--workload", choices=WORKLOADS.keys(), help="Specific workload to benchmark")
    run_parser.add_argument("--iterations", type=int, default=3, help="Number of iterations for each benchmark")
    run_parser.add_argument("--output-dir", help="Directory to store benchmark results")
    
    # Report command
    report_parser = subparsers.add_parser("report", help="Generate benchmark report")
    report_parser.add_argument("--run-dir", required=True, help="Directory containing benchmark results")
    
    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare GPU types")
    compare_parser.add_argument("--workload", required=True, choices=WORKLOADS.keys(), help="Workload to compare")
    compare_parser.add_argument("--results-file", required=True, help="JSON file with comparison results")
    
    # Optimize command
    optimize_parser = subparsers.add_parser("optimize", help="Recommend optimal GPU for a workload")
    optimize_parser.add_argument("--workload", required=True, choices=WORKLOADS.keys(), help="Target workload")
    optimize_parser.add_argument("--results-file", required=True, help="JSON file with comparison results")
    optimize_parser.add_argument("--priority", choices=["performance", "cost", "efficiency"], default="efficiency", 
                               help="Optimization priority (default: efficiency)")
    optimize_parser.add_argument("--budget", type=float, help="Maximum hourly budget in USD")
    
    args = parser.parse_args()
    
    if args.command == "run":
        benchmark = GPUBenchmark(args.output_dir)
        
        if args.workload:
            benchmark.run_benchmark(args.workload, args.iterations)
        else:
            benchmark.run_all_benchmarks(args.iterations)
        
        benchmark.generate_report()
    
    elif args.command == "report":
        run_dir = Path(args.run_dir)
        if not run_dir.exists() or not run_dir.is_dir():
            print(f"Error: Run directory not found: {run_dir}")
            return
        
        benchmark = GPUBenchmark(run_dir.parent)
        benchmark.run_dir = run_dir
        benchmark.generate_report()
    
    elif args.command == "compare":
        if not os.path.exists(args.results_file):
            print(f"Error: Results file not found: {args.results_file}")
            return
        
        with open(args.results_file, 'r') as f:
            gpu_results = json.load(f)
        
        benchmark = GPUBenchmark()
        benchmark.compare_gpu_types(args.workload, gpu_results)
    
    elif args.command == "optimize":
        if not os.path.exists(args.results_file):
            print(f"Error: Results file not found: {args.results_file}")
            return
        
        with open(args.results_file, 'r') as f:
            gpu_results = json.load(f)
        
        # Extract metrics for optimization
        gpu_types = []
        execution_times = []
        costs = []
        performances = []
        cost_efficiencies = []
        
        for gpu_type, result in gpu_results.items():
            # Skip if this GPU doesn't have data for the requested workload
            if args.workload not in result:
                continue
                
            workload_result = result[args.workload]
            gpu_types.append(gpu_type)
            execution_times.append(workload_result.get("execution_time", 0))
            costs.append(workload_result.get("cost", 0))
            performances.append(workload_result.get("performance", 0))
            cost_efficiencies.append(workload_result.get("cost_efficiency", 0))
        
        if not gpu_types:
            print(f"Error: No data found for workload '{args.workload}' in the results file.")
            return
        
        # Filter by budget if specified
        if args.budget is not None:
            valid_indices = [i for i, cost in enumerate(costs) if 
                           HOURLY_RATES.get(gpu_types[i], float('inf')) <= args.budget]
            
            if not valid_indices:
                print(f"Error: No GPU types found within the specified budget of ${args.budget}/hour.")
                return
                
            # Filter all lists to only include GPUs within budget
            gpu_types = [gpu_types[i] for i in valid_indices]
            execution_times = [execution_times[i] for i in valid_indices]
            costs = [costs[i] for i in valid_indices]
            performances = [performances[i] for i in valid_indices]
            cost_efficiencies = [cost_efficiencies[i] for i in valid_indices]
        
        # Find optimal GPU based on priority
        if args.priority == "performance":
            best_idx = performances.index(max(performances))
            metric = "performance"
            value = performances[best_idx]
            unit = gpu_results[gpu_types[best_idx]][args.workload].get("unit", "")
        elif args.priority == "cost":
            best_idx = costs.index(min(costs))
            metric = "cost"
            value = costs[best_idx]
            unit = "$"
        else:  # efficiency
            best_idx = cost_efficiencies.index(max(cost_efficiencies))
            metric = "cost-efficiency"
            value = cost_efficiencies[best_idx]
            unit = "perf/$"
        
        # Print recommendation
        print("\n" + "=" * 60)
        print(f"OPTIMAL GPU RECOMMENDATION FOR {args.workload.upper()}")
        print("=" * 60)
        print(f"Optimization priority: {args.priority}")
        if args.budget is not None:
            print(f"Budget constraint: ${args.budget}/hour")
        print("\nRecommended GPU: " + "*" * 10 + f" {gpu_types[best_idx]} " + "*" * 10)
        print(f"\nMetrics for {gpu_types[best_idx]}:")
        print(f"  - Performance: {performances[best_idx]:.2f} {gpu_results[gpu_types[best_idx]][args.workload].get('unit', '')}")
        print(f"  - Cost: ${costs[best_idx]:.4f}")
        print(f"  - Cost-Efficiency: {cost_efficiencies[best_idx]:.2f} perf/$")
        print(f"  - Execution Time: {execution_times[best_idx]:.2f} seconds")
        
        # Print comparison to other options
        print("\nComparison to alternatives:")
        for i, gpu in enumerate(gpu_types):
            if i != best_idx:
                if args.priority == "performance":
                    diff = (performances[i] / performances[best_idx] - 1) * 100
                    print(f"  - {gpu}: {diff:.1f}% {'better' if diff > 0 else 'worse'} performance")
                elif args.priority == "cost":
                    diff = (costs[i] / costs[best_idx] - 1) * 100
                    print(f"  - {gpu}: {diff:.1f}% {'more' if diff > 0 else 'less'} expensive")
                else:  # efficiency
                    diff = (cost_efficiencies[i] / cost_efficiencies[best_idx] - 1) * 100
                    print(f"  - {gpu}: {diff:.1f}% {'better' if diff > 0 else 'worse'} efficiency")
        
        print("\nWorkload-specific recommendation:")
        if args.workload == "matrix_multiply":
            print("  For matrix multiplication operations, prioritize GPUs with high memory bandwidth and CUDA cores.")
        elif args.workload == "conv_network":
            print("  For convolutional neural networks, GPUs with tensor cores provide significant acceleration.")
        elif args.workload == "data_processing":
            print("  For data processing tasks, consider GPUs with larger memory capacity for bigger datasets.")
        elif args.workload == "transformer":
            print("  For transformer models, GPUs with tensor cores and higher memory bandwidth are crucial.")
        elif args.workload == "reinforcement":
            print("  For reinforcement learning, GPUs with good balance between compute power and memory are recommended.")
        print("=" * 60)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()