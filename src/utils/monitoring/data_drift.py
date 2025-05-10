import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path
from datetime import datetime
from scipy import stats

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class DataDriftDetector:
    """
    Data drift detection for the Cosmic Market Oracle project.
    
    This class provides functionality for:
    1. Detecting statistical drift in feature distributions
    2. Monitoring input data quality
    3. Alerting when significant drift is detected
    4. Tracking feature statistics over time
    """
    
    def __init__(self, reference_data: Optional[Union[str, Path, pd.DataFrame]] = None, 
                 config_path: Optional[Union[str, Path]] = None):
        """
        Initialize the data drift detector.
        
        Args:
            reference_data: Reference dataset or path to reference data
            config_path: Path to drift detection configuration file
        """
        # Load configuration
        self.config_path = Path(config_path) if config_path else Path("config/monitoring.json")
        self.config = self._load_config()
        
        # Set up monitoring directory
        self.monitoring_dir = Path(self.config.get("monitoring_dir", "monitoring"))
        self.drift_dir = self.monitoring_dir / "drift"
        self.drift_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize reference data
        self.reference_stats = {}
        if reference_data is not None:
            self.set_reference_data(reference_data)
        else:
            # Try to load from config
            ref_path = self.config.get("data_drift", {}).get("reference_dataset")
            if ref_path:
                try:
                    self.set_reference_data(ref_path)
                except Exception as e:
                    logger.error(f"Failed to load reference data from {ref_path}: {e}")
        
        # Initialize drift history
        self.drift_history: Dict[str, List[Dict[str, Any]]] = {}
        self._load_drift_history()
        
        logger.info("Initialized data drift detector")
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load drift detection configuration.
        
        Returns:
            Configuration dictionary
        """
        try:
            if not self.config_path.exists():
                logger.warning(f"Monitoring config file {self.config_path} not found, using defaults")
                return self._get_default_config()
            
            with open(self.config_path, "r") as f:
                config = json.load(f)
            
            logger.info(f"Loaded drift detection configuration from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load drift detection config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default drift detection configuration.
        
        Returns:
            Default configuration dictionary
        """
        return {
            "monitoring_dir": "monitoring",
            "data_drift": {
                "detection_method": "ks_test",
                "significance_level": 0.05,
                "min_samples_required": 100,
                "features_to_monitor": ["all"],
                "reference_dataset": "data/processed/reference.csv"
            },
            "alerts": {
                "enabled": True,
                "channels": ["log"],
                "throttle_period_seconds": 3600
            }
        }
    
    def _load_drift_history(self) -> None:
        """
        Load drift detection history if available.
        """
        try:
            drift_history_file = self.drift_dir / "drift_history.json"
            if drift_history_file.exists():
                with open(drift_history_file, "r") as f:
                    self.drift_history = json.load(f)
                logger.info("Loaded drift detection history")
        except Exception as e:
            logger.error(f"Error loading drift history: {e}")
    
    def save_drift_history(self) -> None:
        """
        Save current drift detection history to disk.
        """
        try:
            drift_history_file = self.drift_dir / "drift_history.json"
            with open(drift_history_file, "w") as f:
                json.dump(self.drift_history, f, indent=2)
            logger.info("Saved drift detection history")
        except Exception as e:
            logger.error(f"Error saving drift history: {e}")
    
    def set_reference_data(self, reference_data: Union[str, Path, pd.DataFrame]) -> None:
        """
        Set reference data for drift detection.
        
        Args:
            reference_data: Reference dataset or path to reference data
        """
        try:
            # Load reference data if path is provided
            if isinstance(reference_data, (str, Path)):
                reference_path = Path(reference_data)
                if not reference_path.exists():
                    raise FileNotFoundError(f"Reference data file not found: {reference_path}")
                
                if reference_path.suffix == ".csv":
                    reference_df = pd.read_csv(reference_path)
                elif reference_path.suffix in [".parquet", ".pq"]:
                    reference_df = pd.read_parquet(reference_path)
                else:
                    raise ValueError(f"Unsupported reference data format: {reference_path.suffix}")
            else:
                reference_df = reference_data
            
            # Calculate reference statistics
            self.reference_stats = self._calculate_statistics(reference_df)
            
            # Save reference statistics
            ref_stats_file = self.drift_dir / "reference_statistics.json"
            with open(ref_stats_file, "w") as f:
                # Convert numpy types to Python native types for JSON serialization
                serializable_stats = {}
                for feature, stats in self.reference_stats.items():
                    serializable_stats[feature] = {
                        k: float(v) if isinstance(v, (np.float32, np.float64)) else v
                        for k, v in stats.items()
                    }
                json.dump(serializable_stats, f, indent=2)
            
            logger.info(f"Set reference data with {len(reference_df)} samples and {len(reference_df.columns)} features")
        except Exception as e:
            logger.error(f"Error setting reference data: {e}")
            raise
    
    def _calculate_statistics(self, data: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Calculate statistics for each feature in the dataset.
        
        Args:
            data: DataFrame containing features
            
        Returns:
            Dictionary of feature statistics
        """
        stats_dict = {}
        
        for column in data.columns:
            column_data = data[column].dropna()
            
            # Skip columns with insufficient data
            min_samples = self.config.get("data_drift", {}).get("min_samples_required", 100)
            if len(column_data) < min_samples:
                logger.warning(f"Insufficient samples for feature {column}: {len(column_data)} < {min_samples}")
                continue
            
            # Calculate statistics based on data type
            if np.issubdtype(column_data.dtype, np.number):
                # Numerical feature
                stats_dict[column] = {
                    "type": "numerical",
                    "mean": float(column_data.mean()),
                    "std": float(column_data.std()),
                    "min": float(column_data.min()),
                    "25%": float(column_data.quantile(0.25)),
                    "50%": float(column_data.median()),
                    "75%": float(column_data.quantile(0.75)),
                    "max": float(column_data.max()),
                    "samples": len(column_data),
                    "missing": data[column].isna().sum()
                }
                
                # Store histogram data for distribution comparison
                hist, bin_edges = np.histogram(column_data, bins=20)
                stats_dict[column]["histogram"] = {
                    "counts": hist.tolist(),
                    "bin_edges": bin_edges.tolist()
                }
            else:
                # Categorical feature
                value_counts = column_data.value_counts(normalize=True).to_dict()
                stats_dict[column] = {
                    "type": "categorical",
                    "unique_values": len(value_counts),
                    "distribution": value_counts,
                    "samples": len(column_data),
                    "missing": data[column].isna().sum()
                }
        
        return stats_dict
    
    def detect_drift(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect drift between current data and reference data.
        
        Args:
            current_data: Current data to check for drift
            
        Returns:
            Dictionary with drift detection results
        """
        if not self.reference_stats:
            logger.error("Reference data not set. Cannot detect drift.")
            return {"error": "Reference data not set"}
        
        # Calculate statistics for current data
        current_stats = self._calculate_statistics(current_data)
        
        # Detect drift for each feature
        drift_results = {}
        drift_detected = False
        timestamp = datetime.now().isoformat()
        
        for feature, ref_stats in self.reference_stats.items():
            # Skip if feature not in current data
            if feature not in current_stats:
                logger.warning(f"Feature {feature} not found in current data")
                continue
            
            curr_stats = current_stats[feature]
            
            # Initialize feature result
            drift_results[feature] = {
                "type": ref_stats["type"],
                "drift_detected": False,
                "p_value": None,
                "method": self.config.get("data_drift", {}).get("detection_method", "ks_test"),
                "statistics": {
                    "reference": {
                        k: v for k, v in ref_stats.items() 
                        if k not in ["histogram", "distribution"]
                    },
                    "current": {
                        k: v for k, v in curr_stats.items() 
                        if k not in ["histogram", "distribution"]
                    }
                }
            }
            
            # Detect drift based on feature type
            if ref_stats["type"] == "numerical":
                p_value, drift_detected_feature = self._detect_numerical_drift(
                    feature, ref_stats, curr_stats)
            else:  # categorical
                p_value, drift_detected_feature = self._detect_categorical_drift(
                    feature, ref_stats, curr_stats)
            
            drift_results[feature]["drift_detected"] = drift_detected_feature
            drift_results[feature]["p_value"] = p_value
            
            if drift_detected_feature:
                drift_detected = True
        
        # Record drift detection results
        result_summary = {
            "timestamp": timestamp,
            "drift_detected": drift_detected,
            "features_checked": len(drift_results),
            "features_with_drift": sum(1 for f in drift_results.values() if f.get("drift_detected", False)),
            "details": drift_results
        }
        
        # Add to history
        if "summary" not in self.drift_history:
            self.drift_history["summary"] = []
        self.drift_history["summary"].append({
            "timestamp": timestamp,
            "drift_detected": drift_detected,
            "features_with_drift": result_summary["features_with_drift"]
        })
        
        # Save detailed results
        result_file = self.drift_dir / f"drift_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(result_file, "w") as f:
            json.dump(result_summary, f, indent=2)
        
        # Save drift history
        self.save_drift_history()
        
        # Log results
        if drift_detected:
            logger.warning(f"Data drift detected in {result_summary['features_with_drift']} features")
        else:
            logger.info("No data drift detected")
        
        return result_summary
    
    def _detect_numerical_drift(self, feature: str, ref_stats: Dict[str, Any], 
                               curr_stats: Dict[str, Any]) -> Tuple[float, bool]:
        """
        Detect drift for numerical features.
        
        Args:
            feature: Feature name
            ref_stats: Reference statistics
            curr_stats: Current statistics
            
        Returns:
            Tuple of (p_value, drift_detected)
        """
        method = self.config.get("data_drift", {}).get("detection_method", "ks_test")
        significance_level = self.config.get("data_drift", {}).get("significance_level", 0.05)
        
        # For demonstration, we'll use a simple approach comparing distributions
        # In a real implementation, this would use the actual data and statistical tests
        
        if method == "ks_test":
            # Simulate Kolmogorov-Smirnov test using histograms
            # In a real implementation, this would use the raw data
            ref_hist = np.array(ref_stats["histogram"]["counts"]) / sum(ref_stats["histogram"]["counts"])
            curr_hist = np.array(curr_stats["histogram"]["counts"]) / sum(curr_stats["histogram"]["counts"])
            
            # Calculate maximum difference between distributions (D statistic)
            d_statistic = np.max(np.abs(np.cumsum(ref_hist) - np.cumsum(curr_hist)))
            
            # Approximate p-value based on D statistic
            # This is a simplified approximation
            n = min(ref_stats["samples"], curr_stats["samples"])
            p_value = np.exp(-2 * n * d_statistic**2)
        elif method == "mean_diff":
            # Simple mean difference test
            mean_diff = abs(ref_stats["mean"] - curr_stats["mean"])
            std_pooled = np.sqrt((ref_stats["std"]**2 + curr_stats["std"]**2) / 2)
            
            # Standardized difference
            if std_pooled > 0:
                z_score = mean_diff / std_pooled
                p_value = 2 * (1 - stats.norm.cdf(z_score))  # Two-tailed test
            else:
                p_value = 1.0 if mean_diff == 0 else 0.0
        else:
            logger.warning(f"Unknown drift detection method: {method}. Using default.")
            # Default simple comparison
            mean_pct_change = abs(ref_stats["mean"] - curr_stats["mean"]) / abs(ref_stats["mean"]) if ref_stats["mean"] != 0 else float('inf')
            p_value = 0.01 if mean_pct_change > 0.2 else 0.5  # Simplified p-value
        
        # Determine if drift is detected
        drift_detected = p_value < significance_level
        
        return p_value, drift_detected
    
    def _detect_categorical_drift(self, feature: str, ref_stats: Dict[str, Any], 
                                 curr_stats: Dict[str, Any]) -> Tuple[float, bool]:
        """
        Detect drift for categorical features.
        
        Args:
            feature: Feature name
            ref_stats: Reference statistics
            curr_stats: Current statistics
            
        Returns:
            Tuple of (p_value, drift_detected)
        """
        method = self.config.get("data_drift", {}).get("detection_method", "chi2_test")
        significance_level = self.config.get("data_drift", {}).get("significance_level", 0.05)
        
        # For demonstration, we'll use a simple approach comparing distributions
        # In a real implementation, this would use the actual data and statistical tests
        
        ref_dist = ref_stats["distribution"]
        curr_dist = curr_stats["distribution"]
        
        # Get all unique categories
        all_categories = set(ref_dist.keys()) | set(curr_dist.keys())
        
        if method == "chi2_test":
            # Simulate Chi-square test
            # In a real implementation, this would use the raw data
            chi2_stat = 0
            for category in all_categories:
                ref_prob = ref_dist.get(category, 0)
                curr_prob = curr_dist.get(category, 0)
                
                # Calculate contribution to chi-square statistic
                if ref_prob > 0:  # Avoid division by zero
                    chi2_stat += ((curr_prob - ref_prob)**2) / ref_prob
            
            # Degrees of freedom: number of categories - 1
            df = len(all_categories) - 1
            if df > 0:
                p_value = 1 - stats.chi2.cdf(chi2_stat * ref_stats["samples"], df)
            else:
                p_value = 1.0
        elif method == "js_divergence":
            # Jensen-Shannon divergence
            # Create probability vectors with same categories
            ref_probs = np.array([ref_dist.get(cat, 0) for cat in all_categories])
            curr_probs = np.array([curr_dist.get(cat, 0) for cat in all_categories])
            
            # Ensure probabilities sum to 1
            if np.sum(ref_probs) > 0:
                ref_probs = ref_probs / np.sum(ref_probs)
            if np.sum(curr_probs) > 0:
                curr_probs = curr_probs / np.sum(curr_probs)
            
            # Calculate JS divergence
            m_probs = 0.5 * (ref_probs + curr_probs)
            js_div = 0.5 * (stats.entropy(ref_probs, m_probs) + stats.entropy(curr_probs, m_probs))
            
            # Convert to p-value (approximation)
            # JS divergence is between 0 and 1, with 0 meaning identical distributions
            p_value = np.exp(-js_div * 10)  # Simplified conversion
        else:
            logger.warning(f"Unknown drift detection method for categorical data: {method}. Using default.")
            # Default simple comparison - total variation distance
            tvd = 0
            for category in all_categories:
                ref_prob = ref_dist.get(category, 0)
                curr_prob = curr_dist.get(category, 0)
                tvd += abs(ref_prob - curr_prob)
            
            tvd = tvd / 2  # Normalize to [0, 1]
            p_value = 1 - tvd  # Simple conversion to p-value
        
        # Determine if drift is detected
        drift_detected = p_value < significance_level
        
        return p_value, drift_detected
    
    def get_drift_summary(self) -> Dict[str, Any]:
        """
        Get a summary of drift detection history.
        
        Returns:
            Dictionary with drift summary
        """
        if "summary" not in self.drift_history or not self.drift_history["summary"]:
            return {
                "drift_checks": 0,
                "drift_detected": 0,
                "last_check": None
            }
        
        summary = self.drift_history["summary"]
        return {
            "drift_checks": len(summary),
            "drift_detected": sum(1 for s in summary if s.get("drift_detected", False)),
            "last_check": summary[-1]["timestamp"],
            "recent_results": summary[-5:] if len(summary) >= 5 else summary
        }


if __name__ == "__main__":
    # Example usage
    from model_monitor import create_default_monitoring_config
    create_default_monitoring_config()
    
    # Create a sample reference dataset
    logger.info("Creating sample reference dataset for demonstration")
    np.random.seed(42)
    n_samples = 1000
    ref_data = pd.DataFrame({
        "feature1": np.random.normal(0, 1, n_samples),
        "feature2": np.random.normal(5, 2, n_samples),
        "category": np.random.choice(["A", "B", "C"], n_samples, p=[0.6, 0.3, 0.1])
    })
    
    # Initialize drift detector
    detector = DataDriftDetector(ref_data)
    
    # Create a sample current dataset with drift
    logger.info("Creating sample current dataset with drift")
    current_data = pd.DataFrame({
        "feature1": np.random.normal(0.5, 1.2, n_samples),  # Mean and std shifted
        "feature2": np.random.normal(5, 2, n_samples),     # No drift
        "category": np.random.choice(["A", "B", "C"], n_samples, p=[0.4, 0.4, 0.2])  # Distribution shifted
    })
    
    # Detect drift
    logger.info("Detecting drift")
    results = detector.detect_drift(current_data)
    
    # Print results
    logger.info(f"Drift detected: {results['drift_detected']}")
    logger.info(f"Features with drift: {results['features_with_drift']}")