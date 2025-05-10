# ThunderCompute GPU Infrastructure Setup Guide

This guide provides comprehensive instructions for setting up and managing GPU instances on ThunderCompute for the Cosmic Market Oracle project.

## Table of Contents

1. [Account Setup](#account-setup)
2. [SSH Key Configuration](#ssh-key-configuration)
3. [Instance Types and Selection](#instance-types-and-selection)
4. [Docker Environment](#docker-environment)
5. [Data Management](#data-management)
6. [Cost Optimization](#cost-optimization)
7. [Monitoring and Management](#monitoring-and-management)
8. [Workflow Integration](#workflow-integration)

## Account Setup

### Creating a ThunderCompute Account

1. Visit [ThunderCompute's website](https://thundercompute.com) and sign up for an account
2. Verify your email address and complete the registration process
3. Add a payment method to your account
4. Generate an API key from your account dashboard

### Setting Up API Access

```bash
# Run the instance manager setup command
python infrastructure/scripts/instance_manager.py setup
```

This will prompt you for your API key and create the necessary configuration files.

## SSH Key Configuration

Secure SSH access is essential for remote GPU instances:

```bash
# Generate a new SSH key pair for ThunderCompute
ssh-keygen -t rsa -b 4096 -f ~/.ssh/thundercompute_rsa

# Add the key to your ThunderCompute account
cat ~/.ssh/thundercompute_rsa.pub
```

Copy the public key output and add it to your ThunderCompute account dashboard.

## Instance Types and Selection

The project uses different GPU configurations for various workloads:

### Deep Learning Model Training

- **Recommended GPU**: RTX 4090 or A100
- **Configuration**: 8+ CPU cores, 32+ GB RAM
- **Use Case**: Training large neural networks, transformer models
- **Launch Command**:
  ```bash
  python infrastructure/scripts/instance_manager.py launch deep_learning
  ```

### Data Preprocessing

- **Recommended GPU**: RTX 3080
- **Configuration**: 4+ CPU cores, 16+ GB RAM
- **Use Case**: Feature engineering, data transformation, ETL pipelines
- **Launch Command**:
  ```bash
  python infrastructure/scripts/instance_manager.py launch data_preprocessing
  ```

### Reinforcement Learning

- **Recommended GPU**: Multiple A100 GPUs
- **Configuration**: 16+ CPU cores, 64+ GB RAM
- **Use Case**: Training RL agents, multi-agent simulations
- **Launch Command**:
  ```bash
  python infrastructure/scripts/instance_manager.py launch reinforcement_learning
  ```

## Docker Environment

All workloads use Docker containers for consistent environments:

### Building Docker Images

```bash
# Build the deep learning image
docker build -t cosmic-market-oracle/deep-learning:latest -f infrastructure/docker/deep_learning.Dockerfile .

# Push to your container registry (if using)
docker push cosmic-market-oracle/deep-learning:latest
```

### Docker Image Contents

The deep learning image includes:
- CUDA 11.8 with cuDNN 8
- PyTorch 2.0.1
- TensorFlow 2.12.0
- Astronomical libraries (swisseph, skyfield, astropy)
- ML tools (Ray Tune, MLflow, Optuna)

## Data Management

### Persistent Storage

ThunderCompute instances use the following storage strategy:

1. **Instance Storage**: Fast local SSD for temporary data
2. **Persistent Storage**: Mounted volumes for datasets and checkpoints
3. **Object Storage**: For archiving results and sharing between instances

### Data Synchronization

Use the following approach for efficient data transfers:

```bash
# Sync local data to instance (using delta transfers)
rsync -avz --progress --partial ./data/ username@instance-ip:/workspace/data/

# Sync results back to local machine
rsync -avz --progress --partial username@instance-ip:/workspace/results/ ./results/
```

## Cost Optimization

Implement these strategies to minimize GPU rental costs:

### Spot Instances

For non-critical workloads, use spot instances at reduced prices:

```bash
python infrastructure/scripts/instance_manager.py launch deep_learning --spot --max-price 0.80
```

### Automatic Shutdown

Instances automatically shut down when idle to prevent unnecessary charges:

```bash
# Configure idle shutdown threshold (in minutes)
export AUTO_SHUTDOWN=true
export IDLE_THRESHOLD=30
```

### Checkpointing

Regular checkpointing ensures work isn't lost if instances are preempted:

```bash
# Initialize checkpointing
python infrastructure/scripts/checkpoint.py init

# Configure automatic checkpointing
export AUTO_CHECKPOINT=true
export CHECKPOINT_INTERVAL=15  # minutes
```

## Monitoring and Management

### GPU Utilization Monitoring

Track GPU usage and costs with the monitoring tool:

```bash
# Start monitoring an instance
python infrastructure/scripts/monitor.py monitor instance-id

# Generate a report for the past 7 days
python infrastructure/scripts/monitor.py report instance-id --days 7
```

### Instance Management

Manage running instances with these commands:

```bash
# List all running instances
python infrastructure/scripts/instance_manager.py list

# Shutdown idle instances
python infrastructure/scripts/instance_manager.py shutdown --idle-threshold 30
```

## Workflow Integration

### Local Development to Cloud GPU

Follow this workflow for efficient development:

1. Develop and test code locally with small datasets
2. Push changes to version control
3. Launch appropriate ThunderCompute instance
4. Clone repository and sync necessary data
5. Run training or processing jobs
6. Monitor progress and costs
7. Retrieve results and shut down instance

### Continuous Integration

For automated workflows:

1. Set up CI/CD pipeline with GitHub Actions or similar
2. Configure workflow to launch ThunderCompute instances
3. Run tests and training jobs automatically
4. Collect results and metrics
5. Shut down instances after completion

## Troubleshooting

### Common Issues

1. **Connection Refused**: Check SSH key configuration and security groups
2. **Out of Memory**: Increase instance memory or optimize batch size
3. **Low GPU Utilization**: Check data loading pipeline and preprocessing
4. **Instance Termination**: Ensure proper checkpointing for spot instances

### Support

For ThunderCompute-specific issues, contact their support at support@thundercompute.com

For project-specific infrastructure questions, refer to the internal documentation or contact the DevOps team.