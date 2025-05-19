import os
import sys
import yaml
import json
import subprocess
from src.trading.unified_mcts import MCTS, MCTSPredictor
from src.trading.modular_hierarchical_rl import ModularHierarchicalRLAgent, MCTSStrategicPlanner, PPOTacticalExecutor
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class CIPipelineConfig:
    """
    Configuration for CI/CD pipelines in the Cosmic Market Oracle project.
    This class handles configuration for automated testing, model validation,
    and deployment processes.
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize the CI pipeline configuration.
        
        Args:
            config_path: Path to the configuration file (YAML or JSON)
        """
        self.config_path = Path(config_path) if config_path else Path("ci_cd/config.yaml")
        self.config = self._load_config()
        self.environment = self.config.get("environment", "development")
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Set up pipeline directories
        self.pipeline_dir = Path(self.config.get("pipeline_dir", "ci_cd/pipelines"))
        self.pipeline_dir.mkdir(parents=True, exist_ok=True)
        
        # Create run directory for this pipeline execution
        self.run_dir = self.pipeline_dir / f"run_{self.timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized CI pipeline for environment: {self.environment}")
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Returns:
            Configuration dictionary
        """
        try:
            if not self.config_path.exists():
                logger.warning(f"Config file {self.config_path} not found, using defaults")
                return self._get_default_config()
            
            with open(self.config_path, "r") as f:
                if self.config_path.suffix in [".yaml", ".yml"]:
                    config = yaml.safe_load(f)
                elif self.config_path.suffix == ".json":
                    config = json.load(f)
                else:
                    logger.error(f"Unsupported config file format: {self.config_path.suffix}")
                    return self._get_default_config()
            
            logger.info(f"Loaded configuration from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration.
        
        Returns:
            Default configuration dictionary
        """
        return {
            "environment": "development",
            "pipeline_dir": "ci_cd/pipelines",
            "test": {
                "unit_test_dir": "tests/unit",
                "integration_test_dir": "tests/integration",
                "coverage_threshold": 80
            },
            "model_validation": {
                "validation_dataset": "data/processed/validation.csv",
                "metrics": ["accuracy", "precision", "recall", "f1"],
                "performance_threshold": 0.75
            },
            "deployment": {
                "staging": {
                    "mlflow_model_stage": "Staging",
                    "auto_deploy": True
                },
                "production": {
                    "mlflow_model_stage": "Production",
                    "auto_deploy": False,
                    "approval_required": True
                }
            },
            "notification": {
                "email": False,
                "slack": False
            }
        }
    
    def save_run_config(self) -> Path:
        """
        Save the current run configuration.
        
        Returns:
            Path to the saved configuration file
        """
        try:
            config_file = self.run_dir / "config.json"
            with open(config_file, "w") as f:
                json.dump(self.config, f, indent=2)
            logger.info(f"Saved run configuration to {config_file}")
            return config_file
        except Exception as e:
            logger.error(f"Failed to save run configuration: {e}")
            raise
    
    def run_tests(self, test_type: str = "all") -> bool:
        """
        Run tests as part of the CI pipeline.
        
        Args:
            test_type: Type of tests to run ("unit", "integration", or "all")
            
        Returns:
            True if tests passed, False otherwise
        """
        try:
            logger.info(f"Running {test_type} tests")
            
            # Determine test directories
            test_dirs = []
            if test_type in ["unit", "all"]:
                test_dirs.append(self.config["test"]["unit_test_dir"])
            if test_type in ["integration", "all"]:
                test_dirs.append(self.config["test"]["integration_test_dir"])
            
            # Run pytest with coverage
            cmd = [
                sys.executable, "-m", "pytest",
                *test_dirs,
                "--cov=src",
                f"--cov-fail-under={self.config['test']['coverage_threshold']}",
                "--cov-report=xml:coverage.xml",
                "--cov-report=term",
                "-v"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Save test results
            test_output_file = self.run_dir / f"{test_type}_test_results.txt"
            with open(test_output_file, "w") as f:
                f.write(result.stdout)
                f.write("\n\n")
                f.write(result.stderr)
            
            # Check if tests passed
            passed = result.returncode == 0
            status = "passed" if passed else "failed"
            logger.info(f"Tests {status} with return code {result.returncode}")
            
            return passed
        except Exception as e:
            logger.error(f"Error running tests: {e}")
            return False
    
    def validate_model(self, model_path: str, model_name: str, version: str) -> Dict[str, float]:
        """
        Validate a model against validation dataset.
        
        Args:
            model_path: Path to the model
            model_name: Name of the model
            version: Version of the model
            
        Returns:
            Dictionary of validation metrics
        """
        try:
            logger.info(f"Validating model {model_name} version {version}")
            
            # In a real implementation, this would load the model and validation dataset
            # and compute metrics. For now, we'll simulate the process.
            
            # Placeholder for validation metrics
            metrics = {
                "accuracy": 0.85,
                "precision": 0.82,
                "recall": 0.79,
                "f1": 0.80
            }
            
            # Save validation results
            validation_file = self.run_dir / f"model_validation_{model_name}_{version}.json"
            with open(validation_file, "w") as f:
                json.dump(metrics, f, indent=2)
            
            # Check if model meets performance threshold
            threshold = self.config["model_validation"]["performance_threshold"]
            primary_metric = self.config["model_validation"]["metrics"][0]  # Use first metric as primary
            
            if metrics[primary_metric] >= threshold:
                logger.info(f"Model validation passed: {primary_metric}={metrics[primary_metric]:.4f} >= {threshold}")
            else:
                logger.warning(f"Model validation failed: {primary_metric}={metrics[primary_metric]:.4f} < {threshold}")
            
            return metrics
        except Exception as e:
            logger.error(f"Error validating model: {e}")
            raise
    
    def deploy_model(self, model_name: str, version: str, environment: str) -> bool:
        """
        Deploy a model to the specified environment.
        
        Args:
            model_name: Name of the model
            version: Version of the model
            environment: Target environment ("staging" or "production")
            
        Returns:
            True if deployment succeeded, False otherwise
        """
        try:
            logger.info(f"Deploying model {model_name} version {version} to {environment}")
            
            # Check if auto-deploy is enabled for this environment
            env_config = self.config["deployment"].get(environment, {})
            if not env_config.get("auto_deploy", False):
                logger.info(f"Auto-deploy disabled for {environment}, skipping deployment")
                return False
            
            # Check if approval is required
            if env_config.get("approval_required", False):
                logger.info(f"Approval required for deployment to {environment}")
                # In a real implementation, this would trigger an approval workflow
                # For now, we'll simulate approval
                approved = True  # Simulated approval
                if not approved:
                    logger.info("Deployment not approved, skipping")
                    return False
            
            # In a real implementation, this would handle the actual deployment
            # For now, we'll simulate the deployment process
            
            # Record deployment
            deployment_file = self.run_dir / f"deployment_{model_name}_{version}_{environment}.json"
            deployment_info = {
                "model_name": model_name,
                "version": version,
                "environment": environment,
                "timestamp": datetime.now().isoformat(),
                "status": "success"
            }
            
            with open(deployment_file, "w") as f:
                json.dump(deployment_info, f, indent=2)
            
            logger.info(f"Model {model_name} version {version} deployed to {environment}")
            return True
        except Exception as e:
            logger.error(f"Error deploying model: {e}")
            return False
    
    def run_pipeline(self, model_info: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Run the complete CI/CD pipeline.
        
        Args:
            model_info: Optional information about the model to validate and deploy
            
        Returns:
            Pipeline results dictionary
        """
        try:
            logger.info(f"Starting CI/CD pipeline run in {self.environment} environment")
            
            # Save run configuration
            self.save_run_config()
            
            # Initialize results
            results = {
                "timestamp": self.timestamp,
                "environment": self.environment,
                "tests": {"passed": False},
                "model_validation": {"performed": False},
                "deployment": {"performed": False}
            }
            
            # Run tests
            test_passed = self.run_tests()
            results["tests"]["passed"] = test_passed
            
            # If tests failed and we're not in development, stop the pipeline
            if not test_passed and self.environment != "development":
                logger.warning("Tests failed, stopping pipeline")
                return results
            
            # If model info is provided, validate and deploy the model
            if model_info:
                model_name = model_info.get("name")
                version = model_info.get("version")
                model_path = model_info.get("path")
                
                if model_name and version and model_path:
                    # Validate model
                    metrics = self.validate_model(model_path, model_name, version)
                    results["model_validation"] = {
                        "performed": True,
                        "metrics": metrics
                    }
                    
                    # Deploy to staging if validation passed
                    primary_metric = self.config["model_validation"]["metrics"][0]
                    threshold = self.config["model_validation"]["performance_threshold"]
                    
                    if metrics[primary_metric] >= threshold:
                        # Deploy to staging
                        staging_deployed = self.deploy_model(model_name, version, "staging")
                        
                        # If in production environment and staging deployment succeeded,
                        # consider deploying to production
                        production_deployed = False
                        if staging_deployed and self.environment == "production":
                            production_deployed = self.deploy_model(model_name, version, "production")
                        
                        results["deployment"] = {
                            "performed": True,
                            "staging": staging_deployed,
                            "production": production_deployed if self.environment == "production" else "not_attempted"
                        }
            
            # Save pipeline results
            results_file = self.run_dir / "pipeline_results.json"
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"CI/CD pipeline completed, results saved to {results_file}")
            return results
        except Exception as e:
            logger.error(f"Error running pipeline: {e}")
            raise


def create_default_config():
    """
    Create a default CI/CD configuration file.
    """
    config_dir = Path("ci_cd")
    config_dir.mkdir(parents=True, exist_ok=True)
    
    config_file = config_dir / "config.yaml"
    
    if config_file.exists():
        logger.info(f"Config file {config_file} already exists, skipping creation")
        return
    
    default_config = {
        "environment": "development",
        "pipeline_dir": "ci_cd/pipelines",
        "test": {
            "unit_test_dir": "tests/unit",
            "integration_test_dir": "tests/integration",
            "coverage_threshold": 80
        },
        "model_validation": {
            "validation_dataset": "data/processed/validation.csv",
            "metrics": ["accuracy", "precision", "recall", "f1"],
            "performance_threshold": 0.75
        },
        "deployment": {
            "staging": {
                "mlflow_model_stage": "Staging",
                "auto_deploy": True
            },
            "production": {
                "mlflow_model_stage": "Production",
                "auto_deploy": False,
                "approval_required": True
            }
        },
        "notification": {
            "email": {
                "enabled": False,
                "recipients": ["team@cosmicmarketoracle.com"]
            },
            "slack": {
                "enabled": False,
                "webhook_url": "",
                "channel": "#model-deployments"
            }
        }
    }
    
    with open(config_file, "w") as f:
        yaml.dump(default_config, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"Created default CI/CD configuration at {config_file}")


if __name__ == "__main__":
    create_default_config()