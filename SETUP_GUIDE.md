# Cosmic Market Oracle - Setup Guide

This guide will help you set up and run the Cosmic Market Oracle project using Python 3.10 and leverage the MLflow UI for visualization.

## Environment Setup

### Python 3.10 Requirements

This project is optimized for Python 3.10. The requirements files have been updated with compatible package versions.

### Setting Up Your Environment

1. Create a virtual environment with Python 3.10:
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements-hyperopt.txt
   ```

## Running the MLflow UI

To view your model training results, hyperparameter optimization progress, and metrics, you'll need to run the MLflow UI:

1. Start the MLflow UI server:
   ```bash
   python scripts/setup_and_run_mlflow.py
   ```

2. Once running, open your browser and navigate to:
   ```
   http://localhost:5000
   ```

3. The MLflow UI will display:
   - Experiment runs
   - Hyperparameter values
   - Metrics (accuracy, loss, etc.)
   - Artifacts (model files, plots)

## Running Hyperparameter Optimization

To run the hyperparameter optimization and model training process:

1. Make sure your virtual environment is activated
2. Run the optimization script:
   ```bash
   python scripts/run_optimization.py --config config/model_config.yaml
   ```

3. This will:
   - Load your data
   - Run Optuna hyperparameter optimization
   - Train the model with the best parameters
   - Log results to MLflow
   - Save the trained model

## Viewing Results

After running optimization, you can view the results in the MLflow UI:

1. Start the MLflow UI if it's not already running
2. Navigate to the experiment page
3. Click on individual runs to see:
   - Parameter values used
   - Performance metrics
   - Learning curves
   - Prediction plots

## Using Advanced Architectures

The project includes several advanced model architectures:

1. **Transformer Models**: Capture long-range dependencies in time series data
2. **Graph Neural Networks (GNN)**: Model relationships between planetary positions
3. **Hybrid CNN-Transformer**: Combine local pattern detection with global context
4. **Neural ODE**: Model continuous-time dynamics
5. **Ensemble Models**: Combine multiple architectures for improved accuracy

To select a specific architecture, modify the `model_type` parameter in your configuration file.

## Leveraging MCP Tools

This project can leverage the Model Context Protocol (MCP) tools for enhanced capabilities:

1. **Brave Search API**: Use for market research and astrological data validation
   - The API key is already configured in your environment

2. **Memory MCP Server**: Store and retrieve important context about your models and data

To use these tools, make API calls from your Python code or use them through the Cascade interface.

## Troubleshooting

If you encounter issues:

1. **Package Compatibility**: Ensure you're using Python 3.10 with the updated requirements
2. **MLflow UI Not Showing Data**: Make sure you've run at least one training experiment
3. **Import Errors**: Check that your PYTHONPATH includes the project root directory

## Next Steps

1. Run a full hyperparameter optimization
2. Train the final model with the best parameters
3. Analyze results in the MLflow UI
4. Explore the advanced architectures to improve prediction accuracy
