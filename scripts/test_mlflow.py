#!/usr/bin/env python
# Test MLflow logging

"""
Simple script to test MLflow logging and verify that the tracking server is working.
"""

import os
import sys
import numpy as np
import mlflow
from pathlib import Path
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("test_experiment")

# Create a simple test run
with mlflow.start_run(run_name="test_run"):
    # Log some parameters
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_param("batch_size", 32)
    mlflow.log_param("model_type", "test_model")
    
    # Log some metrics
    for i in range(10):
        mlflow.log_metric("loss", 1.0 - 0.1 * i, step=i)
        mlflow.log_metric("accuracy", 0.5 + 0.05 * i, step=i)
    
    # Create and log a simple plot
    fig, ax = plt.subplots()
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    ax.plot(x, y)
    ax.set_title("Test Plot")
    ax.set_xlabel("X")
    ax.set_ylabel("sin(X)")
    
    # Save the plot
    plot_path = "test_plot.png"
    plt.savefig(plot_path)
    
    # Log the plot as an artifact
    mlflow.log_artifact(plot_path)
    
    # Clean up
    os.remove(plot_path)
    
    print("Test run completed successfully!")
    print("Check the MLflow UI at http://localhost:5000")

if __name__ == "__main__":
    print("Running MLflow test...")
