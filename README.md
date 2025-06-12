# The Cosmic Market Oracle

An AI-powered financial forecasting system that integrates Vedic astrological principles with 200 years of market data to predict market inflection points.

## Project Overview

This pioneering project fuses historical Dow Jones Industrial Average (DJI) data with Vedic astrological positions through advanced AI techniques to create a revolutionary market prediction system. By analyzing cosmic patterns that potentially influence market psychology and economic cycles, we aim to develop a robust system capable of identifying market inflection points with unprecedented accuracy.

### Key Features

- **Historical Data Integration**: Combines 200 years of market data with precise planetary positions
- **Advanced Feature Engineering**: Converts complex astrological phenomena into ML-compatible inputs
- **Multi-Modal Deep Learning**: Integrates time series, cyclical, and symbolic data
- **Cloud GPU Infrastructure**: Optimized training on VAST.ai and ThunderCompute platforms
- **Interpretable Predictions**: Explains market forecasts through astrological reasoning
- **Modular Hierarchical RL**: Flexible framework for strategic planning and tactical execution

## Getting Started

### Prerequisites

- Python 3.10 (recommended, project is optimized for this version)
- PostgreSQL with TimescaleDB extension (for time-series data storage)
- GPU access for model training (local or cloud-based)

### Environment Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Sarthaksina/Astro.git
   cd Astro
   ```

2. Run the automated environment setup script:
   ```bash
   python scripts/setup/init_environment.py
   ```
   
   This script will:
   - Check Python version compatibility
   - Create a virtual environment
   - Install all dependencies
   - Set up environment variables
   - Create necessary data directories

3. Activate the virtual environment:
   ```bash
   # On Windows
   .\venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

4. Configure the environment variables in the `.env` file:
   ```
   # Database Configuration
   DB_USER=postgres
   DB_PASSWORD=your_password
   DB_HOST=localhost
   DB_PORT=5432
   DB_NAME=cosmic_market_oracle
   
   # Swiss Ephemeris Configuration
   EPHE_PATH=./data/ephemeris
   
   # API Configuration
   API_HOST=0.0.0.0
   API_PORT=8000
   
   # MLflow Configuration
   MLFLOW_TRACKING_URI=http://localhost:5000
   
   # Cloud GPU API Keys (if using cloud GPUs)
   VAST_AI_API_KEY=your_vast_ai_key
   THUNDER_COMPUTE_API_KEY=your_thunder_compute_key
   ```

5. Initialize the database:
   ```bash
   python scripts/setup/init_database.py
   ```

## Project Structure

```
├── src/                             # Source code
│   ├── astro_engine/                # Vedic astrological calculation engine
│   ├── data_acquisition/            # Data collection modules
│   ├── data_processing/             # Data cleaning and transformation
│   ├── feature_engineering/         # Feature creation from raw data
│   ├── models/                      # ML/DL model implementations
│   ├── api/                         # FastAPI implementation
│   ├── utils/                       # Core utilities
│   │   ├── config.py                # Configuration management
│   │   ├── logger.py                # Centralized logging
│   │   ├── file_io.py               # DataFrame saving utility
│   │   └── environment.py           # Environment checks (e.g., Python version)
│   └── visualization/               # Data visualization tools
│
├── tests/                           # Unit and integration tests
├── notebooks/                       # Jupyter notebooks for exploration
├── config/                          # Configuration files
├── scripts/                         # Operational scripts
│
├── requirements.txt                 # Python dependencies
├── setup.py                         # Package installation
└── README.md                        # Project documentation
```

## Core Components

1. **Vedic Astrological Engine**: Comprehensive calculation modules for planetary positions, nakshatras, and dashas
2. **Data Acquisition & Integration**: Modules for collecting market/astrological data and a unified system (`src/data_integration/db_manager.py`) for database schema and operations.
3. **Feature Engineering**: Conversion of astrological phenomena into ML-compatible inputs
4. **Model Development**: Implementation of ensemble models combining multiple AI paradigms
5. **Evaluation Framework**: Comprehensive testing across different market regimes
6. **Prediction Pipeline**: End-to-end system for generating market predictions
7. **API Server**: FastAPI-based interface for accessing predictions and data
8. **Cloud GPU Management**: Automated infrastructure for cost-efficient model training
9. **Modular Hierarchical RL Framework**: Flexible architecture for strategic planning and tactical execution in trading
10. **Core Utilities**: Centralized modules for logging, configuration, file I/O, and environment checks (`src/utils/`).

## Modular Hierarchical RL Framework

The project now features a unified modular hierarchical reinforcement learning framework that replaces several redundant implementations. This new architecture provides a clear separation between strategic planning and tactical execution components.

### Key Benefits

- **Modularity**: Easily swap different strategic planners or tactical executors
- **Maintainability**: Clear interfaces and separation of concerns
- **Extensibility**: Simple to add new algorithms or strategies
- **Performance**: Optimized implementations of MCTS and PPO algorithms

### Usage Example

```python
from src.trading.modular_hierarchical_rl import (
    ModularHierarchicalRLAgent,
    MCTSStrategicPlanner,
    PPOTacticalExecutor
)

# Create components
strategic_planner = MCTSStrategicPlanner(state_dim=state_dim, num_simulations=50)
tactical_executor = PPOTacticalExecutor(state_dim=state_dim, action_dim=action_dim)

# Create the agent
agent = ModularHierarchicalRLAgent(
    strategic_planner=strategic_planner,
    tactical_executor=tactical_executor
)

# Use the agent
action = agent.select_action(state)
```

For a complete demonstration, see `scripts/modular_hierarchical_rl_demo.py`.

## Development Workflow

1. Set up the development environment as described above
2. Refer to `TASK.md` for current development priorities
3. Follow the architectural guidelines in `PLANNING.md`
4. Create unit tests for all new functionality
5. Document code with Google-style docstrings

## Running the System

### Using the Unified Command-Line Interface

The Cosmic Market Oracle now features a unified command-line interface that works with Python 3.10. Use the provided batch script on Windows:

```bash
# Show available commands
run_with_python310

# Acquire data
run_with_python310 data --start_date=2023-01-01 --end_date=2023-12-31 --symbol=^DJI

# Run hyperparameter optimization
run_with_python310 optimize --config=config/model_config.yaml

# Generate predictions
run_with_python310 predict --start_date=2023-01-01 --end_date=2023-01-31 --symbol=^DJI --model=best_model

# Start MLflow UI for visualization
run_with_python310 mlflow
```

The MLflow UI will be available at http://localhost:5000

### Start the API Server

```bash
python -m src.api.app
```

The API will be available at http://localhost:8000 with interactive documentation at http://localhost:8000/docs

### Manage Cloud GPU Instances

```bash
# Start a new GPU instance for training
python scripts/setup/gpu_instance_manager.py start --instance_type=rtx4090 --job=training

# Monitor GPU utilization and automatically stop idle instances
python scripts/setup/gpu_instance_manager.py monitor --threshold=0.1 --interval=300
```

## Production Deployment

For detailed instructions on deploying the Cosmic Market Oracle in a production environment, please refer to the [Production Deployment Guide](docs/PRODUCTION_DEPLOYMENT.md).

Key production features include:

- **Containerization**: Docker-based deployment for consistency across environments
- **Load Balancing**: Nginx configuration for SSL termination and load distribution
- **Database Scaling**: TimescaleDB configuration for efficient time-series data storage
- **Monitoring**: Prometheus, Grafana, and Loki for comprehensive observability
- **Security**: API key authentication, rate limiting, and proper error handling
- **CI/CD**: Automated testing and deployment pipeline with GitHub Actions

## License

This project is proprietary and confidential. All rights reserved.

## Acknowledgments

- Swiss Ephemeris for high-precision astronomical calculations
- TimescaleDB for efficient time-series data storage
- VAST.ai and ThunderStorm for cloud GPU infrastructure
- Prometheus, Grafana, and Loki for monitoring infrastructure