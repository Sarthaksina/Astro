# Cosmic Market Oracle CI/CD Infrastructure

This directory contains the GitHub Actions workflows and CI/CD configuration for the Cosmic Market Oracle project. The CI/CD infrastructure automates testing, model validation, and deployment processes while integrating with comprehensive monitoring systems.

## Workflow Files

### 1. Continuous Integration (`ci.yml`)

This workflow runs on pushes to main/develop branches and pull requests, performing automated testing with code coverage reporting.

**Features:**
- Runs on multiple Python versions (3.8, 3.9)
- Executes unit and integration tests
- Generates code coverage reports
- Uploads test results as artifacts

**Usage:**
Automatically triggered on push/PR events, but can also be manually triggered from the Actions tab in GitHub.

### 2. Model Deployment (`deploy.yml`)

This workflow handles model validation and deployment to staging and production environments.

**Features:**
- Triggered automatically after successful CI workflow runs
- Can be manually triggered with custom parameters
- Validates models against performance thresholds
- Deploys to staging automatically, production with approval
- Integrates with model monitoring system

**Manual Trigger Parameters:**
- `environment`: Deployment target (staging/production)
- `model_name`: Name of the model to deploy
- `model_version`: Version of the model to deploy

### 3. GPU Tests (`gpu-tests.yml`)

Specialized workflow for running tests and benchmarks on GPU infrastructure.

**Features:**
- Scheduled to run weekly
- Can be manually triggered with GPU configuration options
- Runs GPU-specific tests and performance benchmarks
- Reports GPU utilization metrics

**Manual Trigger Parameters:**
- `gpu_type`: GPU configuration to use (standard/high-memory/multi-gpu)

## Integration with Pipeline Configuration

The workflows integrate with the `CIPipelineConfig` class in `ci_cd/pipeline_config.py`, which provides:

- Configuration management for different environments
- Test execution with coverage reporting
- Model validation against performance thresholds
- Deployment logic with environment-specific settings
- Results tracking and reporting

## Monitoring Integration

The CI/CD system integrates with the monitoring infrastructure through `ci_cd/monitoring_integration.py`, which provides:

- Automatic registration of deployed models with the monitoring system
- Performance metrics tracking over time
- Data drift detection capabilities
- Health checks for deployed models
- Alerting when metrics fall below thresholds

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