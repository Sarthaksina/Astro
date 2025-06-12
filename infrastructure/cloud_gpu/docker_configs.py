"""Docker Configuration Generator for Cosmic Market Oracle GPU Workloads.

This module provides functionality for generating Docker configurations
for different GPU workloads, optimized for various machine learning tasks.
"""

import os
import yaml
# import logging # Removed
from src.utils.logger import get_logger # Added
from src.trading.unified_mcts import MCTS, MCTSPredictor
from src.trading.modular_hierarchical_rl import ModularHierarchicalRLAgent, MCTSStrategicPlanner, PPOTacticalExecutor
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from jinja2 import Environment, FileSystemLoader, select_autoescape

# Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s') # Removed
logger = get_logger(__name__) # Changed

class DockerConfigGenerator:
    """Generator for Docker configurations optimized for GPU workloads.
    
    This class provides functionality for:
    1. Generating Dockerfiles for different workload types
    2. Creating docker-compose configurations for multi-container setups
    3. Optimizing Docker configurations for GPU utilization
    4. Managing Docker volumes for persistent storage
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None, 
                 templates_dir: Optional[Union[str, Path]] = None):
        """Initialize the Docker configuration generator.
        
        Args:
            config_path: Path to GPU instance configuration file
            templates_dir: Path to Dockerfile templates directory
        """
        self.config_path = Path(config_path) if config_path else Path("config/gpu_instances.yaml")
        self.templates_dir = Path(templates_dir) if templates_dir else Path("infrastructure/cloud_gpu/templates")
        
        # Create templates directory if it doesn't exist
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize Jinja2 environment for templates
        self.jinja_env = Environment(
            loader=FileSystemLoader(self.templates_dir),
            autoescape=select_autoescape(['html', 'xml'])
        )
        
        logger.info(f"Initialized Docker configuration generator")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load GPU instance configuration.
        
        Returns:
            Configuration dictionary
        """
        try:
            if not self.config_path.exists():
                logger.warning(f"Config file {self.config_path} not found")
                return {}
            
            with open(self.config_path, "r") as f:
                config = yaml.safe_load(f)
            
            logger.info(f"Loaded configuration from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {}
    
    def generate_dockerfile(self, workload_type: str, output_path: Optional[Union[str, Path]] = None) -> Optional[str]:
        """Generate a Dockerfile for a specific workload type.
        
        Args:
            workload_type: Type of workload (e.g., 'training', 'preprocessing', 'inference')
            output_path: Path to write the Dockerfile to (optional)
            
        Returns:
            Dockerfile content if successful, None otherwise
        """
        try:
            # Define template variables based on workload type
            template_vars = self._get_template_vars(workload_type)
            
            # Load template
            template_name = f"Dockerfile.{workload_type}.j2"
            
            # Check if template exists, if not, create it
            template_path = self.templates_dir / template_name
            if not template_path.exists():
                self._create_default_template(workload_type)
            
            # Render template
            template = self.jinja_env.get_template(template_name)
            dockerfile_content = template.render(**template_vars)
            
            # Write to file if output path provided
            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_path, "w") as f:
                    f.write(dockerfile_content)
                
                logger.info(f"Generated Dockerfile for {workload_type} at {output_path}")
            
            return dockerfile_content
        except Exception as e:
            logger.error(f"Error generating Dockerfile for {workload_type}: {e}")
            return None
    
    def _get_template_vars(self, workload_type: str) -> Dict[str, Any]:
        """Get template variables for a specific workload type.
        
        Args:
            workload_type: Type of workload
            
        Returns:
            Dictionary of template variables
        """
        # Base variables for all workload types
        vars = {
            "base_image": "nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04",
            "python_version": "3.10",
            "workload_type": workload_type,
            "project_name": "cosmic_market_oracle"
        }
        
        # Workload-specific variables
        if workload_type == "training":
            vars.update({
                "cuda_version": "11.8",
                "pytorch_version": "2.0.1",
                "tensorflow_version": "2.12.0",
                "extra_packages": [
                    "scikit-learn",
                    "pandas",
                    "matplotlib",
                    "seaborn",
                    "jupyterlab",
                    "pyephem",
                    "skyfield",
                    "astropy",
                    "swisseph"
                ],
                "environment_variables": {
                    "PYTHONUNBUFFERED": "1",
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "CUDA_DEVICE_ORDER": "PCI_BUS_ID"
                },
                "expose_ports": ["8888", "6006"],  # Jupyter and TensorBoard
                "entrypoint": "python -m src.models.train"
            })
        elif workload_type == "preprocessing":
            vars.update({
                "cuda_version": "11.8",
                "pytorch_version": "2.0.1",
                "extra_packages": [
                    "pandas",
                    "numpy",
                    "scikit-learn",
                    "pyarrow",
                    "fastparquet",
                    "dask",
                    "pyephem",
                    "skyfield",
                    "astropy",
                    "swisseph"
                ],
                "environment_variables": {
                    "PYTHONUNBUFFERED": "1",
                    "PYTHONDONTWRITEBYTECODE": "1"
                },
                "expose_ports": [],
                "entrypoint": "python -m src.data_processing.preprocess"
            })
        elif workload_type == "inference":
            vars.update({
                "cuda_version": "11.8",
                "pytorch_version": "2.0.1",
                "tensorflow_version": "2.12.0",
                "extra_packages": [
                    "pandas",
                    "numpy",
                    "scikit-learn",
                    "fastapi",
                    "uvicorn",
                    "pyephem",
                    "skyfield",
                    "astropy",
                    "swisseph"
                ],
                "environment_variables": {
                    "PYTHONUNBUFFERED": "1",
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "MODEL_PATH": "/app/models"
                },
                "expose_ports": ["8000"],  # FastAPI
                "entrypoint": "uvicorn src.api.main:app --host 0.0.0.0 --port 8000"
            })
        elif workload_type == "rl":
            vars.update({
                "cuda_version": "11.8",
                "pytorch_version": "2.0.1",
                "extra_packages": [
                    "gym",
                    "stable-baselines3",
                    "pandas",
                    "numpy",
                    "matplotlib",
                    "jupyterlab",
                    "pyephem",
                    "skyfield",
                    "astropy",
                    "swisseph"
                ],
                "environment_variables": {
                    "PYTHONUNBUFFERED": "1",
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "CUDA_DEVICE_ORDER": "PCI_BUS_ID"
                },
                "expose_ports": ["8888", "6006"],  # Jupyter and TensorBoard
                "entrypoint": "python -m src.models.rl.train"
            })
        
        return vars
    
    def _create_default_template(self, workload_type: str) -> None:
        """Create a default Dockerfile template for a workload type.
        
        Args:
            workload_type: Type of workload
        """
        template_path = self.templates_dir / f"Dockerfile.{workload_type}.j2"
        
        # Basic template structure
        template_content = """# Base image with CUDA support
FROM {{ base_image }}

# Set environment variables
{% for key, value in environment_variables.items() %}
ENV {{ key }}={{ value }} \
{% endfor %}
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    wget \
    ca-certificates \
    python{{ python_version }} \
    python{{ python_version }}-dev \
    python3-pip \
    python3-setuptools \
    libpq-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python
RUN ln -sf /usr/bin/python{{ python_version }} /usr/bin/python3 && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install Python dependencies
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

{% if workload_type == 'training' or workload_type == 'inference' %}
# Install deep learning frameworks
RUN pip install --no-cache-dir \
    torch=={{ pytorch_version }}+cu{{ cuda_version.replace('.', '') }} \
    torchvision \
    torchaudio \
    -f https://download.pytorch.org/whl/torch_stable.html \
    tensorflow=={{ tensorflow_version }}
{% endif %}

# Install extra packages
{% if extra_packages %}
RUN pip install --no-cache-dir \
    {% for package in extra_packages %}
    {{ package }} {% if not loop.last %}\
    {% endif %}
    {% endfor %}
{% endif %}

# Create working directory
WORKDIR /app

# Copy project files
COPY . /app/

{% if expose_ports %}
# Expose ports
{% for port in expose_ports %}
EXPOSE {{ port }}
{% endfor %}
{% endif %}

# Set up entrypoint
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

# Default command
CMD ["{{ entrypoint }}"]
"""
        
        # Write template to file
        with open(template_path, "w") as f:
            f.write(template_content)
        
        logger.info(f"Created default template for {workload_type} at {template_path}")
    
    def generate_docker_compose(self, services: List[str], output_path: Optional[Union[str, Path]] = None) -> Optional[str]:
        """Generate a docker-compose.yml file for multiple services.
        
        Args:
            services: List of service names to include
            output_path: Path to write the docker-compose.yml to (optional)
            
        Returns:
            docker-compose.yml content if successful, None otherwise
        """
        try:
            # Create docker-compose configuration
            compose_config = {
                "version": "3.8",
                "services": {},
                "volumes": {}
            }
            
            # Add services
            for service in services:
                service_config = self._get_service_config(service)
                if service_config:
                    compose_config["services"][service] = service_config
            
            # Add common volumes
            compose_config["volumes"] = {
                "model_data": {},
                "checkpoint_data": {},
                "dataset_data": {}
            }
            
            # Convert to YAML
            compose_yaml = yaml.dump(compose_config, default_flow_style=False)
            
            # Write to file if output path provided
            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_path, "w") as f:
                    f.write(compose_yaml)
                
                logger.info(f"Generated docker-compose.yml at {output_path}")
            
            return compose_yaml
        except Exception as e:
            logger.error(f"Error generating docker-compose.yml: {e}")
            return None
    
    def _get_service_config(self, service_name: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific service.
        
        Args:
            service_name: Name of the service
            
        Returns:
            Service configuration dictionary
        """
        if service_name == "training":
            return {
                "build": {
                    "context": "..",
                    "dockerfile": "infrastructure/cloud_gpu/Dockerfile.training"
                },
                "container_name": "cosmic_training",
                "volumes": [
                    "../:/app",
                    "model_data:/app/data/models",
                    "checkpoint_data:/app/data/checkpoints",
                    "dataset_data:/app/data/datasets"
                ],
                "environment": [
                    "CUDA_VISIBLE_DEVICES=0,1,2,3",
                    "PYTHONPATH=/app"
                ],
                "deploy": {
                    "resources": {
                        "reservations": {
                            "devices": [{
                                "driver": "nvidia",
                                "count": "all",
                                "capabilities": ["gpu"]
                            }]
                        }
                    }
                }
            }
        elif service_name == "jupyter":
            return {
                "build": {
                    "context": "..",
                    "dockerfile": "infrastructure/cloud_gpu/Dockerfile.training"
                },
                "container_name": "cosmic_jupyter",
                "volumes": [
                    "../:/app",
                    "model_data:/app/data/models",
                    "checkpoint_data:/app/data/checkpoints",
                    "dataset_data:/app/data/datasets"
                ],
                "ports": [
                    "8888:8888"
                ],
                "environment": [
                    "JUPYTER_ENABLE_LAB=yes",
                    "PYTHONPATH=/app"
                ],
                "command": "jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='cosmic'",
                "deploy": {
                    "resources": {
                        "reservations": {
                            "devices": [{
                                "driver": "nvidia",
                                "count": 1,
                                "capabilities": ["gpu"]
                            }]
                        }
                    }
                }
            }
        elif service_name == "inference":
            return {
                "build": {
                    "context": "..",
                    "dockerfile": "infrastructure/cloud_gpu/Dockerfile.inference"
                },
                "container_name": "cosmic_inference",
                "volumes": [
                    "../:/app",
                    "model_data:/app/data/models"
                ],
                "ports": [
                    "8000:8000"
                ],
                "environment": [
                    "PYTHONPATH=/app"
                ],
                "deploy": {
                    "resources": {
                        "reservations": {
                            "devices": [{
                                "driver": "nvidia",
                                "count": 1,
                                "capabilities": ["gpu"]
                            }]
                        }
                    }
                }
            }
        elif service_name == "preprocessing":
            return {
                "build": {
                    "context": "..",
                    "dockerfile": "infrastructure/cloud_gpu/Dockerfile.preprocessing"
                },
                "container_name": "cosmic_preprocessing",
                "volumes": [
                    "../:/app",
                    "dataset_data:/app/data/datasets"
                ],
                "environment": [
                    "PYTHONPATH=/app"
                ],
                "deploy": {
                    "resources": {
                        "reservations": {
                            "devices": [{
                                "driver": "nvidia",
                                "count": 1,
                                "capabilities": ["gpu"]
                            }]
                        }
                    }
                }
            }
        
        return None


def get_docker_config_generator(config_path: Optional[Union[str, Path]] = None,
                              templates_dir: Optional[Union[str, Path]] = None) -> DockerConfigGenerator:
    """Get a Docker configuration generator instance.
    
    Args:
        config_path: Path to GPU instance configuration file
        templates_dir: Path to Dockerfile templates directory
        
    Returns:
        DockerConfigGenerator instance
    """
    return DockerConfigGenerator(config_path, templates_dir)