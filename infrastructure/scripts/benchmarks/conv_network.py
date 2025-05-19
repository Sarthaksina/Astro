#!/usr/bin/env python3
"""
Convolutional Network Training Benchmark

This script benchmarks training performance of convolutional neural networks on GPUs.
It measures the time taken to train a CNN on a synthetic dataset
and outputs performance metrics in a standardized JSON format.
"""

import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from src.trading.unified_mcts import MCTS, MCTSPredictor
from src.trading.modular_hierarchical_rl import ModularHierarchicalRLAgent, MCTSStrategicPlanner, PPOTacticalExecutor

# Configure benchmark parameters
BATCH_SIZE = 64
INPUT_CHANNELS = 3
INPUT_SIZE = 224  # 224x224 images
NUM_CLASSES = 1000
EPOCHS = 3
WARMUP_ITERATIONS = 1


class SimpleCNN(nn.Module):
    """A simple CNN model for benchmarking."""
    
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(INPUT_CHANNELS, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * (INPUT_SIZE // 8) * (INPUT_SIZE // 8), 4096)
        self.fc2 = nn.Linear(4096, 1024)
        self.fc3 = nn.Linear(1024, NUM_CLASSES)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 256 * (INPUT_SIZE // 8) * (INPUT_SIZE // 8))
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


def generate_synthetic_data(batch_size, device):
    """Generate synthetic data for benchmarking.
    
    Args:
        batch_size: Number of samples in the batch
        device: Device to create the tensors on
        
    Returns:
        Tuple of (inputs, targets)
    """
    inputs = torch.randn(batch_size, INPUT_CHANNELS, INPUT_SIZE, INPUT_SIZE, device=device)
    targets = torch.randint(0, NUM_CLASSES, (batch_size,), device=device)
    return inputs, targets


def benchmark_cnn_training():
    """Run CNN training benchmark on the current GPU.
    
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
    
    # Create model and move to GPU
    model = SimpleCNN().to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    # Warmup
    print("Performing warmup iterations...")
    for _ in range(WARMUP_ITERATIONS):
        inputs, targets = generate_synthetic_data(BATCH_SIZE, device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    
    # Benchmark training
    print(f"\nBenchmarking CNN training for {EPOCHS} epochs...")
    torch.cuda.synchronize()
    start_time = time.time()
    
    total_batches = 50  # Number of batches per epoch
    total_samples = 0
    
    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        running_loss = 0.0
        
        for i in range(total_batches):
            # Generate synthetic data
            inputs, targets = generate_synthetic_data(BATCH_SIZE, device)
            
            # Forward + backward + optimize
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            total_samples += BATCH_SIZE
        
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1}: {epoch_time:.2f} seconds, Loss: {running_loss/total_batches:.4f}")
    
    torch.cuda.synchronize()
    total_time = time.time() - start_time
    
    # Calculate performance metrics
    samples_per_second = total_samples / total_time
    batches_per_second = (total_batches * EPOCHS) / total_time
    
    # Prepare final results
    benchmark_results = {
        "device": device_name,
        "model": "SimpleCNN",
        "batch_size": BATCH_SIZE,
        "input_size": f"{INPUT_SIZE}x{INPUT_SIZE}",
        "epochs": EPOCHS,
        "total_batches": total_batches * EPOCHS,
        "total_samples": total_samples,
        "total_time": total_time,
        "samples_per_second": samples_per_second,
        "batches_per_second": batches_per_second,
        "performance": samples_per_second,  # Samples/second as the main performance metric
        "unit": "samples/second",
        "higher_is_better": True
    }
    
    return benchmark_results


def main():
    """Run the benchmark and output results as JSON."""
    results = benchmark_cnn_training()
    print(json.dumps(results))


if __name__ == "__main__":
    main()