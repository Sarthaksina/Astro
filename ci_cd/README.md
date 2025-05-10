# Cosmic Market Oracle CI/CD Infrastructure

This directory contains the CI/CD pipeline configuration and monitoring integration for the Cosmic Market Oracle project. The CI/CD infrastructure automates testing, model validation, and deployment processes while integrating with comprehensive monitoring systems.

## Components

### 1. Pipeline Configuration (`pipeline_config.py`)

The `CIPipelineConfig` class provides the core functionality for the CI/CD pipeline:

**Features:**
- Configuration management for different environments
- Test execution with coverage reporting
- Model validation against performance thresholds
- Deployment logic with environment-specific settings
- Results tracking and reporting

### 2. Monitoring Integration (`monitoring_integration.py`)

The `MonitoringIntegration` class connects the CI/CD pipeline with the model monitoring system:

**Features:**
- Automatic registration of deployed models with the monitoring system
- Performance metrics tracking over time
- Health checks for deployed models
- Alerting when metrics fall below thresholds

## GitHub Actions Workflows

The project uses GitHub Actions for automated CI/CD processes, defined in the `.github/workflows` directory:

### Continuous Integration (`ci.yml`)

This workflow runs on pushes to main/develop branches and pull requests, performing automated testing with code coverage reporting.

### Model Deployment (`deploy.yml`)

This workflow handles model validation and deployment to staging and production environments.

### GPU Tests (`gpu-tests.yml`)

Specialized workflow for running tests and benchmarks on GPU infrastructure.

## Usage Examples

### Running Tests Locally

```bash
# Run all tests
python -c "from ci_cd.pipeline_config import CIPipelineConfig; config = CIPipelineConfig(); config.run_tests('all')"

# Run only unit tests
python -c "from ci_cd.pipeline_config import CIPipelineConfig; config = CIPipelineConfig(); config.run_tests('unit')"
```

### Validating a Model Locally

```bash
python -c "from ci_cd.pipeline_config import CIPipelineConfig; \
          config = CIPipelineConfig(); \
          metrics = config.validate_model('models/my_model', 'my_model', '1.0')"
```

### Running the Complete Pipeline

```bash
python -c "from ci_cd.pipeline_config import CIPipelineConfig; \
          config = CIPipelineConfig(); \
          model_info = {'name': 'my_model', 'version': '1.0', 'path': 'models/my_model/1.0'}; \
          results = config.run_pipeline(model_info=model_info)"
```

### Monitoring a Deployed Model

```bash
python -c "from ci_cd.monitoring_integration import integrate_monitoring_with_pipeline; \
          result = integrate_monitoring_with_pipeline('my_model', '1.0', 'staging', \
                                                    {'accuracy': 0.85, 'precision': 0.82, \
                                                     'recall': 0.79, 'f1': 0.80})"
```

## Configuration

The CI/CD system uses configuration files:

- `ci_cd/config.yaml`: Main pipeline configuration
- `config/monitoring.json`: Monitoring system configuration

You can create default configurations by running:

```bash
python -m ci_cd.pipeline_config
python -m src.utils.monitoring.model_monitor
```