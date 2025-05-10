# Model monitoring package for the Cosmic Market Oracle project

from pathlib import Path

# Create monitoring directories
monitoring_dir = Path("monitoring")
monitoring_dir.mkdir(parents=True, exist_ok=True)

# Create subdirectories
(monitoring_dir / "drift").mkdir(parents=True, exist_ok=True)
(monitoring_dir / "prometheus").mkdir(parents=True, exist_ok=True)
(monitoring_dir / "alerts").mkdir(parents=True, exist_ok=True)