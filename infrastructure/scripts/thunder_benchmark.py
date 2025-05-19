#!/usr/bin/env python3
"""
ThunderCompute GPU Benchmarking Tool

This script automates the benchmarking of different GPU instance types on ThunderCompute
to evaluate their performance and cost-efficiency for various workloads.
Results help optimize instance selection for different tasks in the Cosmic Market Oracle project.
"""

import argparse
import json
import os
import sys
import time
from src.trading.unified_mcts import MCTS, MCTSPredictor
from src.trading.modular_hierarchical_rl import ModularHierarchicalRLAgent, MCTSStrategicPlanner, PPOTacticalExecutor
from datetime import datetime
from pathlib import Path
import subprocess
import logging
import yaml

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

# Import ThunderCompute manager
from infrastructure.cloud_gpu.thunder_compute_manager import get_thunder_compute_manager
from infrastructure.scripts.benchmark import GPUBenchmark, WORKLOADS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
CONFIG_PATH = Path("config/gpu_instances.yaml")
RESULTS_DIR = Path("benchmark_results/thundercompute")

class ThunderComputeBenchmark:
    """Benchmark different GPU instance types on ThunderCompute."""
    
    def __init__(self, config_path=None, results_dir=None):
        """Initialize the ThunderCompute benchmark tool.
        
        Args:
            config_path: Path to GPU instance configuration file
            results_dir: Directory to store benchmark results
        """
        self.config_path = Path(config_path or CONFIG_PATH)
        self.results_dir = Path(results_dir or RESULTS_DIR)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize ThunderCompute manager
        self.manager = get_thunder_compute_manager(self.config_path)
        
        # Create timestamp for this benchmark run
        self.timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_dir = self.results_dir / f"run_{self.timestamp}"
        self.run_dir.mkdir(exist_ok=True)
        
        logger.info(f"Initialized ThunderCompute benchmark tool")
    
    def _load_config(self):
        """Load GPU instance configuration.
        
        Returns:
            Configuration dictionary
        """
        try:
            if not self.config_path.exists():
                logger.error(f"Config file {self.config_path} not found")
                return {}
            
            with open(self.config_path, "r") as f:
                config = yaml.safe_load(f)
            
            logger.info(f"Loaded configuration from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {}
    
    def benchmark_instance_type(self, instance_type, workloads=None, iterations=3):
        """Benchmark a specific instance type on ThunderCompute.
        
        Args:
            instance_type: Type of instance to benchmark
            workloads: List of workloads to run (if None, run all)
            iterations: Number of iterations for each benchmark
            
        Returns:
            Dictionary containing benchmark results
        """
        # Get instance configuration
        instance_config = self.config.get("instance_types", {}).get(instance_type)
        if not instance_config:
            logger.error(f"Unknown instance type: {instance_type}")
            return None
        
        # Use all workloads if none specified
        if workloads is None:
            workloads = list(WORKLOADS.keys())
        
        # Validate workloads
        valid_workloads = [w for w in workloads if w in WORKLOADS]
        if len(valid_workloads) != len(workloads):
            invalid = set(workloads) - set(valid_workloads)
            logger.warning(f"Invalid workloads: {invalid}. Will be skipped.")
        
        if not valid_workloads:
            logger.error("No valid workloads specified")
            return None
        
        # Create instance
        logger.info(f"Creating {instance_type} instance for benchmarking")
        instance_id = self.manager.create_instance(instance_type)
        
        if not instance_id:
            logger.error(f"Failed to create {instance_type} instance")
            return None
        
        try:
            # Wait for instance to be ready
            logger.info(f"Waiting for instance {instance_id} to be ready")
            self._wait_for_instance(instance_id)
            
            # Upload benchmark scripts
            logger.info(f"Uploading benchmark scripts to instance {instance_id}")
            self._upload_benchmark_scripts(instance_id)
            
            # Run benchmarks
            results = {}
            for workload in valid_workloads:
                logger.info(f"Running {workload} benchmark on {instance_type} instance")
                workload_result = self._run_benchmark(instance_id, workload, iterations)
                if workload_result:
                    results[workload] = workload_result
            
            # Add instance metadata
            results["instance_type"] = instance_type
            results["gpu_type"] = instance_config.get("gpu_type")
            results["timestamp"] = self.timestamp
            
            # Save results
            results_file = self.run_dir / f"{instance_type}_results.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Benchmark results saved to {results_file}")
            return results
        
        finally:
            # Terminate instance
            logger.info(f"Terminating instance {instance_id}")
            self.manager.terminate_instance(instance_id)
    
    def _wait_for_instance(self, instance_id, timeout=300):
        """Wait for an instance to be ready.
        
        Args:
            instance_id: Instance ID
            timeout: Timeout in seconds
            
        Returns:
            True if instance is ready, False otherwise
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            instance = self.manager.get_instance(instance_id)
            if not instance:
                logger.error(f"Instance {instance_id} not found")
                return False
            
            status = instance.get("status")
            if status == "running":
                logger.info(f"Instance {instance_id} is ready")
                return True
            
            logger.info(f"Instance status: {status}. Waiting...")
            time.sleep(10)
        
        logger.error(f"Timeout waiting for instance {instance_id} to be ready")
        return False
    
    def _upload_benchmark_scripts(self, instance_id):
        """Upload benchmark scripts to an instance.
        
        Args:
            instance_id: Instance ID
            
        Returns:
            True if successful, False otherwise
        """
        # Create remote directory
        self.manager.run_command(instance_id, "mkdir -p ~/benchmarks")
        
        # Upload benchmark scripts
        scripts_dir = Path(__file__).parent / "benchmarks"
        return self.manager.upload_data(instance_id, scripts_dir, "~/benchmarks")
    
    def _run_benchmark(self, instance_id, workload, iterations):
        """Run a benchmark on an instance.
        
        Args:
            instance_id: Instance ID
            workload: Workload name
            iterations: Number of iterations
            
        Returns:
            Benchmark results dictionary
        """
        try:
            # Get workload script
            workload_info = WORKLOADS.get(workload)
            if not workload_info:
                logger.error(f"Unknown workload: {workload}")
                return None
            
            script_path = workload_info["script"]
            script_name = Path(script_path).name
            
            # Run benchmark command
            command = f"cd ~ && python benchmarks/{script_name} --iterations {iterations} --json"
            output = self.manager.run_command(instance_id, command)
            
            if not output:
                logger.error(f"Failed to run {workload} benchmark on instance {instance_id}")
                return None
            
            # Parse output
            try:
                result = json.loads(output)
                return result
            except json.JSONDecodeError:
                logger.error(f"Failed to parse benchmark output: {output}")
                return None
        except Exception as e:
            logger.error(f"Error running benchmark: {e}")
            return None
    
    def benchmark_all_instance_types(self, workloads=None, iterations=3):
        """Benchmark all configured instance types.
        
        Args:
            workloads: List of workloads to run (if None, run all)
            iterations: Number of iterations for each benchmark
            
        Returns:
            Dictionary mapping instance types to benchmark results
        """
        results = {}
        instance_types = self.config.get("instance_types", {}).keys()
        
        for instance_type in instance_types:
            logger.info(f"Benchmarking {instance_type} instance type")
            result = self.benchmark_instance_type(instance_type, workloads, iterations)
            if result:
                results[instance_type] = result
        
        # Save combined results
        combined_file = self.run_dir / "combined_results.json"
        with open(combined_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"All benchmark results saved to {combined_file}")
        return results
    
    def generate_comparison_report(self, results=None):
        """Generate a comparison report for different instance types.
        
        Args:
            results: Dictionary mapping instance types to benchmark results (if None, load from files)
            
        Returns:
            Path to the generated report file
        """
        if results is None:
            # Load results from files
            results = {}
            for results_file in self.run_dir.glob("*_results.json"):
                if results_file.name != "combined_results.json":
                    with open(results_file, 'r') as f:
                        instance_results = json.load(f)
                        instance_type = instance_results.get("instance_type")
                        if instance_type:
                            results[instance_type] = instance_results
        
        if not results:
            logger.error("No benchmark results found")
            return None
        
        # Create benchmark tool for report generation
        benchmark_tool = GPUBenchmark(self.run_dir)
        
        # Generate comparison reports for each workload
        report_paths = []
        for workload in WORKLOADS:
            # Collect results for this workload across instance types
            workload_results = {}
            for instance_type, instance_results in results.items():
                if workload in instance_results:
                    workload_results[instance_type] = instance_results[workload]
            
            if workload_results:
                # Generate comparison report
                report_path = benchmark_tool.compare_gpu_types(workload, workload_results)
                report_paths.append(report_path)
        
        # Generate overall report
        report_file = self.run_dir / "instance_comparison_report.html"
        with open(report_file, 'w') as f:
            f.write(f"<html>\n<head>\n<title>ThunderCompute Instance Comparison Report</title>\n")
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
            f.write(f"<h1>ThunderCompute Instance Comparison Report</h1>\n")
            f.write(f"<p>Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>\n")
            
            # Links to workload-specific reports
            f.write("<h2>Workload-Specific Reports</h2>\n")
            f.write("<ul>\n")
            for report_path in report_paths:
                report_name = report_path.name
                f.write(f"<li><a href='{report_name}'>{report_name}</a></li>\n")
            f.write("</ul>\n")
            
            # Summary table
            f.write("<h2>Performance Summary</h2>\n")
            f.write("<table>\n")
            f.write("<tr><th>Instance Type</th><th>GPU Type</th><th>Avg. Performance</th><th>Avg. Cost Efficiency</th><th>Hourly Rate</th></tr>\n")
            
            for instance_type, instance_results in results.items():
                # Get instance configuration
                instance_config = self.config.get("instance_types", {}).get(instance_type, {})
                gpu_type = instance_results.get("gpu_type", "Unknown")
                hourly_rate = instance_config.get("hourly_rate", "N/A")
                
                # Calculate average performance and cost efficiency across workloads
                performances = []
                cost_efficiencies = []
                
                for workload, workload_results in instance_results.items():
                    if workload not in ["instance_type", "gpu_type", "timestamp"]:
                        if "performance" in workload_results:
                            performances.append(workload_results["performance"])
                        if "cost_efficiency" in workload_results:
                            cost_efficiencies.append(workload_results["cost_efficiency"])
                
                avg_performance = sum(performances) / len(performances) if performances else "N/A"
                avg_cost_efficiency = sum(cost_efficiencies) / len(cost_efficiencies) if cost_efficiencies else "N/A"
                
                f.write(f"<tr>")
                f.write(f"<td>{instance_type}</td>")
                f.write(f"<td>{gpu_type}</td>")
                f.write(f"<td>{avg_performance if isinstance(avg_performance, str) else f'{avg_performance:.2f}'}</td>")
                f.write(f"<td>{avg_cost_efficiency if isinstance(avg_cost_efficiency, str) else f'{avg_cost_efficiency:.2f}'}</td>")
                f.write(f"<td>${hourly_rate}/hr</td>")
                f.write(f"</tr>\n")
            
            f.write("</table>\n")
            
            # Recommendations
            f.write("<h2>Recommendations</h2>\n")
            f.write("<p>Based on the benchmark results, here are our recommendations for different workloads:</p>\n")
            
            for workload in WORKLOADS:
                f.write(f"<h3>{workload}</h3>\n")
                f.write("<ul>\n")
                
                # Find best performance and cost efficiency for this workload
                best_performance = (None, 0)
                best_cost_efficiency = (None, 0)
                
                for instance_type, instance_results in results.items():
                    if workload in instance_results:
                        workload_results = instance_results[workload]
                        
                        if "performance" in workload_results and (best_performance[0] is None or workload_results["performance"] > best_performance[1]):
                            best_performance = (instance_type, workload_results["performance"])
                        
                        if "cost_efficiency" in workload_results and (best_cost_efficiency[0] is None or workload_results["cost_efficiency"] > best_cost_efficiency[1]):
                            best_cost_efficiency = (instance_type, workload_results["cost_efficiency"])
                
                if best_performance[0]:
                    f.write(f"<li><strong>Best Performance:</strong> {best_performance[0]}</li>\n")
                
                if best_cost_efficiency[0]:
                    f.write(f"<li><strong>Best Cost-Efficiency:</strong> {best_cost_efficiency[0]}</li>\n")
                
                f.write("</ul>\n")
            
            f.write("</body>\n</html>")
        
        logger.info(f"Comparison report generated: {report_file}")
        return report_file


def main():
    """Main function for ThunderCompute benchmarking."""
    parser = argparse.ArgumentParser(description="ThunderCompute GPU Benchmarking Tool")
    parser.add_argument("--config", help="Path to GPU instance configuration file")
    parser.add_argument("--results-dir", help="Directory to store benchmark results")
    parser.add_argument("--instance-type", help="Specific instance type to benchmark")
    parser.add_argument("--workload", action="append", help="Specific workload to run (can be specified multiple times)")
    parser.add_argument("--iterations", type=int, default=3, help="Number of iterations for each benchmark")
    parser.add_argument("--all", action="store_true", help="Benchmark all instance types")
    parser.add_argument("--report", action="store_true", help="Generate comparison report from existing results")
    
    args = parser.parse_args()
    
    # Create benchmark tool
    benchmark = ThunderComputeBenchmark(args.config, args.results_dir)
    
    if args.report:
        # Generate comparison report from existing results
        benchmark.generate_comparison_report()
    elif args.instance_type:
        # Benchmark specific instance type
        benchmark.benchmark_instance_type(args.instance_type, args.workload, args.iterations)
    elif args.all:
        # Benchmark all instance types
        benchmark.benchmark_all_instance_types(args.workload, args.iterations)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()