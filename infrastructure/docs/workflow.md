# ThunderCompute GPU Development Workflow

This document outlines the recommended workflow for developing and running machine learning workloads on ThunderCompute GPU instances for the Cosmic Market Oracle project.

## Development Workflow Overview

```
Local Development → Version Control → ThunderCompute Deployment → Results Retrieval
```

## Step 1: Local Development

1. **Set up local environment**
   - Create a virtual environment for local development
   - Install required libraries for development and testing
   - Use smaller datasets for initial development and testing

2. **Develop code locally**
   - Implement features and algorithms
   - Test with small sample data
   - Optimize code for GPU execution

3. **Prepare for cloud deployment**
   - Ensure code is modular and configurable
   - Add checkpointing capabilities
   - Implement logging and monitoring

## Step 2: Version Control Integration

1. **Commit changes to repository**
   ```bash
   git add .
   git commit -m "Implement feature X for model training"
   git push
   ```

2. **Tag versions for deployment**
   ```bash
   git tag -a v0.1.0 -m "Initial training pipeline"
   git push origin v0.1.0
   ```

## Step 3: ThunderCompute Deployment

1. **Launch appropriate instance**
   ```bash
   # For deep learning workloads
   python infrastructure/scripts/instance_manager.py launch deep_learning
   
   # For data preprocessing
   python infrastructure/scripts/instance_manager.py launch data_preprocessing
   
   # For reinforcement learning
   python infrastructure/scripts/instance_manager.py launch reinforcement_learning
   ```

2. **Connect to instance**
   ```bash
   ssh -i ~/.ssh/thundercompute_rsa user@instance-ip
   ```

3. **Clone repository and sync data**
   ```bash
   # Clone repository
   git clone https://github.com/your-org/cosmic-market-oracle.git
   cd cosmic-market-oracle
   
   # Sync data from local machine
   python infrastructure/scripts/sync_data.py from user@local-machine:/path/to/data --to /workspace/data
   ```

4. **Configure environment**
   ```bash
   # Set up automatic checkpointing
   export AUTO_CHECKPOINT=true
   export CHECKPOINT_INTERVAL=15  # minutes
   
   # Set up automatic shutdown for idle instances
   export AUTO_SHUTDOWN=true
   export IDLE_THRESHOLD=30  # minutes
   
   # Start GPU monitoring
   python infrastructure/scripts/monitor.py monitor $(hostname) --interval=60
   ```

5. **Run workload**
   ```bash
   # Example: Training a deep learning model
   python src/models/train_model.py \
       --data_dir=/workspace/data \
       --output_dir=/workspace/results \
       --config=configs/training_config.yaml
   ```

## Step 4: Monitoring and Management

1. **Monitor GPU utilization and costs**
   ```bash
   # From another terminal on your local machine
   ssh -i ~/.ssh/thundercompute_rsa user@instance-ip "cat /var/log/monitor.log"
   ```

2. **Check training progress**
   ```bash
   # View training logs
   ssh -i ~/.ssh/thundercompute_rsa user@instance-ip "tail -f /workspace/results/training.log"
   
   # Check TensorBoard (if running)
   ssh -i ~/.ssh/thundercompute_rsa -L 6006:localhost:6006 user@instance-ip
   # Then open http://localhost:6006 in your browser
   ```

## Step 5: Results Retrieval

1. **Sync results back to local machine**
   ```bash
   # From your local machine
   python infrastructure/scripts/sync_data.py from user@instance-ip:/workspace/results --to ./results
   ```

2. **Verify data integrity**
   ```bash
   # Check model checkpoints
   ls -la ./results/checkpoints
   
   # Verify logs and metrics
   cat ./results/metrics.json
   ```

3. **Shut down instance if no longer needed**
   ```bash
   # Using the instance manager
   python infrastructure/scripts/instance_manager.py shutdown --instance-id=instance-id
   ```

## Continuous Integration Workflow

For automated workflows, consider setting up CI/CD pipelines:

1. **GitHub Actions configuration**
   ```yaml
   # .github/workflows/train_model.yml
   name: Train Model on ThunderCompute
   
   on:
     push:
       branches: [ main ]
       paths:
         - 'src/models/**'
         - 'configs/**'
   
   jobs:
     train:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v2
         
         - name: Set up Python
           uses: actions/setup-python@v2
           with:
             python-version: 3.9
             
         - name: Install dependencies
           run: pip install -r requirements.txt
           
         - name: Launch ThunderCompute instance
           run: python infrastructure/scripts/instance_manager.py launch deep_learning
           
         # Additional steps for running training and syncing results
   ```

## Best Practices

1. **Cost Optimization**
   - Use spot instances for non-critical workloads
   - Schedule intensive training during lower-cost periods
   - Implement automatic shutdown for idle instances
   - Use checkpointing to resume interrupted jobs

2. **Data Management**
   - Use delta transfers to minimize bandwidth usage
   - Compress large datasets before transfer
   - Consider using object storage for shared datasets
   - Implement data versioning for reproducibility

3. **Resource Utilization**
   - Monitor GPU utilization to ensure efficient use
   - Optimize batch sizes for maximum throughput
   - Use mixed precision training when possible
   - Implement parallel data loading pipelines

4. **Security**
   - Use SSH keys for authentication
   - Keep API keys and credentials secure
   - Restrict access to instances with proper firewall rules
   - Regularly rotate credentials

## Troubleshooting

1. **Connection Issues**
   - Verify SSH key permissions (should be 600)
   - Check instance status in ThunderCompute dashboard
   - Ensure firewall rules allow SSH connections

2. **Performance Problems**
   - Check GPU utilization with monitoring tools
   - Verify data loading isn't bottlenecking training
   - Optimize code for better GPU utilization

3. **Data Sync Failures**
   - Check network connectivity
   - Verify disk space on both ends
   - Try smaller batches of files

4. **Instance Termination**
   - Ensure checkpointing is properly configured
   - Check if spot instance was preempted
   - Verify budget limits haven't been reached