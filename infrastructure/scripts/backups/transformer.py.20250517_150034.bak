#!/usr/bin/env python3
"""
Transformer Model Training Benchmark

This script benchmarks training performance of transformer models on GPUs.
It measures the time taken to train a transformer model on synthetic data
and outputs performance metrics in a standardized JSON format.
"""

import json
import time
import torch
import torch.nn as nn
import torch.optim as optim

# Configure benchmark parameters
BATCH_SIZE = 32
SEQ_LENGTH = 512
VOCAB_SIZE = 30000
EMBED_DIM = 512
NUM_HEADS = 8
NUM_LAYERS = 6
EPOCHS = 2
WARMUP_ITERATIONS = 1


class TransformerEncoderModel(nn.Module):
    """A simple transformer encoder model for benchmarking."""
    
    def __init__(self):
        super(TransformerEncoderModel, self).__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        self.pos_encoder = nn.Embedding(SEQ_LENGTH, EMBED_DIM)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM,
            nhead=NUM_HEADS,
            dim_feedforward=EMBED_DIM * 4,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=NUM_LAYERS)
        self.output_layer = nn.Linear(EMBED_DIM, VOCAB_SIZE)
    
    def forward(self, x):
        # Create position indices
        positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0).expand(x.size(0), -1)
        
        # Combine token embeddings and position embeddings
        x = self.embedding(x) + self.pos_encoder(positions)
        
        # Pass through transformer encoder
        x = self.transformer_encoder(x)
        
        # Project to vocabulary size
        x = self.output_layer(x)
        return x


def generate_synthetic_data(batch_size, seq_length, device):
    """Generate synthetic data for benchmarking.
    
    Args:
        batch_size: Number of samples in the batch
        seq_length: Length of each sequence
        device: Device to create the tensors on
        
    Returns:
        Tuple of (inputs, targets)
    """
    inputs = torch.randint(0, VOCAB_SIZE, (batch_size, seq_length), device=device)
    targets = torch.randint(0, VOCAB_SIZE, (batch_size, seq_length), device=device)
    return inputs, targets


def benchmark_transformer_training():
    """Run transformer training benchmark on the current GPU.
    
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
    model = TransformerEncoderModel().to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    # Warmup
    print("Performing warmup iterations...")
    for _ in range(WARMUP_ITERATIONS):
        inputs, targets = generate_synthetic_data(BATCH_SIZE, SEQ_LENGTH, device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.view(-1, VOCAB_SIZE), targets.view(-1))
        loss.backward()
        optimizer.step()
    
    # Benchmark training
    print(f"\nBenchmarking transformer training for {EPOCHS} epochs...")
    torch.cuda.synchronize()
    start_time = time.time()
    
    total_batches = 20  # Number of batches per epoch
    total_tokens = 0
    
    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        running_loss = 0.0
        
        for i in range(total_batches):
            # Generate synthetic data
            inputs, targets = generate_synthetic_data(BATCH_SIZE, SEQ_LENGTH, device)
            
            # Forward + backward + optimize
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, VOCAB_SIZE), targets.view(-1))
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            total_tokens += BATCH_SIZE * SEQ_LENGTH
        
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1}: {epoch_time:.2f} seconds, Loss: {running_loss/total_batches:.4f}")
    
    torch.cuda.synchronize()
    total_time = time.time() - start_time
    
    # Calculate performance metrics
    tokens_per_second = total_tokens / total_time
    batches_per_second = (total_batches * EPOCHS) / total_time
    
    # Prepare final results
    benchmark_results = {
        "device": device_name,
        "model": "TransformerEncoder",
        "batch_size": BATCH_SIZE,
        "seq_length": SEQ_LENGTH,
        "embed_dim": EMBED_DIM,
        "num_heads": NUM_HEADS,
        "num_layers": NUM_LAYERS,
        "epochs": EPOCHS,
        "total_batches": total_batches * EPOCHS,
        "total_tokens": total_tokens,
        "total_time": total_time,
        "tokens_per_second": tokens_per_second,
        "batches_per_second": batches_per_second,
        "performance": tokens_per_second,  # Tokens/second as the main performance metric
        "unit": "tokens/second",
        "higher_is_better": True
    }
    
    return benchmark_results


def main():
    """Run the benchmark and output results as JSON."""
    results = benchmark_transformer_training()
    print(json.dumps(results))


if __name__ == "__main__":
    main()