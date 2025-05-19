"""
Market Regime Labeling System for the Cosmic Market Oracle.

This module provides tools for identifying and labeling different market regimes:
1. Bull markets
2. Bear markets
3. Sideways/consolidation markets
4. High volatility regimes
5. Low volatility regimes
6. Transitional periods

It uses multiple approaches including:
- Hidden Markov Models
- Clustering algorithms
- Rule-based approaches
- Expert validation framework
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from hmmlearn import hmm
import json
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MarketRegimeClassifier:
    """Classifier for detecting market regimes."""
    
    def __init__(self, method: str = "hmm", n_regimes: int = 3, data_dir: str = "data/regimes"):
        """
        Initialize the market regime classifier.
        
        Args:
            method: Classification method (hmm, kmeans, rule_based)
            n_regimes: Number of regimes to detect
            data_dir: Directory for storing regime data
        """
        self.method = method
        self.n_regimes = n_regimes
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialize models
        self.hmm_model = None
        self.kmeans_model = None
        self.scaler = StandardScaler()
        
        # Rule-based parameters
        self.rule_params = {
            "bull_threshold": 0.2,  # 20% up from recent low
            "bear_threshold": -0.2,  # 20% down from recent high
            "volatility_lookback": 20,
            "high_volatility_percentile": 75,
            "low_volatility_percentile": 25,
            "trend_ma_fast": 50,
            "trend_ma_slow": 200
        }
    
    def fit(self, market_data: pd.DataFrame) -> None:
        """
        Fit the regime classifier to market data.
        
        Args:
            market_data: DataFrame with market data (must include 'close' column)
        """
        if 'close' not in market_data.columns:
            raise ValueError("Market data must include 'close' column")
            
        # Prepare features
        features = self._prepare_features(market_data)
        
        # Scale features
        scaled_features = self.scaler.fit_transform(features)
        
        if self.method == "hmm":
            # Fit Hidden Markov Model
            self.hmm_model = hmm.GaussianHMM(
                n_components=self.n_regimes,
                covariance_type="full",
                n_iter=1000,
                random_state=42
            )
            self.hmm_model.fit(scaled_features)
            logger.info(f"Fitted HMM model with {self.n_regimes} regimes")
            
        elif self.method == "kmeans":
            # Fit K-means clustering
            self.kmeans_model = KMeans(
                n_clusters=self.n_regimes,
                random_state=42,
                n_init=10
            )
            self.kmeans_model.fit(scaled_features)
            logger.info(f"Fitted K-means model with {self.n_regimes} clusters")
            
        elif self.method == "rule_based":
            # No fitting needed for rule-based approach
            logger.info("Using rule-based regime classification")
            
        else:
            raise ValueError(f"Unsupported method: {self.method}")
    
    def predict(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """
        Predict market regimes for the given data.
        
        Args:
            market_data: DataFrame with market data
            
        Returns:
            DataFrame with original data and regime labels
        """
        if 'close' not in market_data.columns:
            raise ValueError("Market data must include 'close' column")
            
        # Prepare features
        features = self._prepare_features(market_data)
        
        # Scale features
        scaled_features = self.scaler.transform(features)
        
        # Create result DataFrame
        result = market_data.copy()
        
        if self.method == "hmm":
            if self.hmm_model is None:
                raise ValueError("Model not fitted. Call fit() first.")
                
            # Predict hidden states
            hidden_states = self.hmm_model.predict(scaled_features)
            result['regime'] = hidden_states
            
        elif self.method == "kmeans":
            if self.kmeans_model is None:
                raise ValueError("Model not fitted. Call fit() first.")
                
            # Predict clusters
            clusters = self.kmeans_model.predict(scaled_features)
            result['regime'] = clusters
            
        elif self.method == "rule_based":
            # Apply rule-based classification
            result['regime'] = self._apply_rule_based(market_data)
            
        else:
            raise ValueError(f"Unsupported method: {self.method}")
            
        # Add regime names
        result['regime_name'] = self._get_regime_names(result)
        
        return result
    
    def _prepare_features(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for regime classification.
        
        Args:
            market_data: DataFrame with market data
            
        Returns:
            DataFrame with features
        """
        df = market_data.copy()
        
        # Calculate returns
        df['returns'] = df['close'].pct_change()
        df['returns_5d'] = df['close'].pct_change(5)
        df['returns_20d'] = df['close'].pct_change(20)
        
        # Calculate volatility
        df['volatility_20d'] = df['returns'].rolling(20).std()
        
        # Calculate moving averages
        df['ma_50'] = df['close'].rolling(50).mean()
        df['ma_200'] = df['close'].rolling(200).mean()
        
        # Calculate MA ratio
        df['ma_ratio'] = df['ma_50'] / df['ma_200']
        
        # Calculate distance from moving averages
        df['dist_from_ma_50'] = (df['close'] - df['ma_50']) / df['ma_50']
        df['dist_from_ma_200'] = (df['close'] - df['ma_200']) / df['ma_200']
        
        # Select and clean features
        features = df[['returns', 'returns_5d', 'returns_20d', 
                      'volatility_20d', 'ma_ratio', 
                      'dist_from_ma_50', 'dist_from_ma_200']].copy()
        
        # Fill NaN values
        features = features.fillna(method='bfill').fillna(0)
        
        return features
    
    def _apply_rule_based(self, market_data: pd.DataFrame) -> pd.Series:
        """
        Apply rule-based regime classification.
        
        Args:
            market_data: DataFrame with market data
            
        Returns:
            Series with regime labels
        """
        df = market_data.copy()
        
        # Calculate returns
        df['returns'] = df['close'].pct_change()
        
        # Calculate moving averages
        fast_ma = df['close'].rolling(self.rule_params['trend_ma_fast']).mean()
        slow_ma = df['close'].rolling(self.rule_params['trend_ma_slow']).mean()
        
        # Calculate volatility
        lookback = self.rule_params['volatility_lookback']
        volatility = df['returns'].rolling(lookback).std()
        
        # Calculate drawdowns
        rolling_max = df['close'].rolling(252, min_periods=1).max()
        drawdown = (df['close'] / rolling_max) - 1
        
        # Calculate run-ups
        rolling_min = df['close'].rolling(252, min_periods=1).min()
        run_up = (df['close'] / rolling_min) - 1
        
        # Determine regimes
        regimes = pd.Series(index=df.index, dtype=int)
        
        # Bull market (fast MA > slow MA and run-up > threshold)
        bull_condition = (
            (fast_ma > slow_ma) & 
            (run_up > self.rule_params['bull_threshold'])
        )
        regimes[bull_condition] = 0  # Bull market
        
        # Bear market (fast MA < slow MA and drawdown < threshold)
        bear_condition = (
            (fast_ma < slow_ma) & 
            (drawdown < self.rule_params['bear_threshold'])
        )
        regimes[bear_condition] = 1  # Bear market
        
        # High volatility regime
        high_vol_threshold = volatility.quantile(
            self.rule_params['high_volatility_percentile'] / 100)
        high_vol_condition = volatility > high_vol_threshold
        regimes[high_vol_condition & ~bull_condition & ~bear_condition] = 2  # High volatility
        
        # Low volatility regime
        low_vol_threshold = volatility.quantile(
            self.rule_params['low_volatility_percentile'] / 100)
        low_vol_condition = volatility < low_vol_threshold
        regimes[low_vol_condition & ~bull_condition & ~bear_condition] = 3  # Low volatility
        
        # Sideways market (default)
        regimes[regimes.isna()] = 4  # Sideways
        
        return regimes
    
    def _get_regime_names(self, data: pd.DataFrame) -> pd.Series:
        """
        Get regime names based on regime labels.
        
        Args:
            data: DataFrame with regime labels
            
        Returns:
            Series with regime names
        """
        if self.method == "rule_based":
            # For rule-based, we have predefined names
            regime_map = {
                0: "Bull Market",
                1: "Bear Market",
                2: "High Volatility",
                3: "Low Volatility",
                4: "Sideways Market"
            }
            return data['regime'].map(regime_map)
            
        else:
            # For HMM and K-means, we need to infer names
            regimes = data['regime'].unique()
            regime_stats = {}
            
            for regime in regimes:
                regime_data = data[data['regime'] == regime]
                
                # Calculate statistics for this regime
                returns = regime_data['close'].pct_change().mean() * 252  # Annualized
                volatility = regime_data['close'].pct_change().std() * np.sqrt(252)  # Annualized
                
                regime_stats[regime] = {
                    'returns': returns,
                    'volatility': volatility
                }
            
            # Assign names based on statistics
            regime_names = {}
            
            # Sort regimes by returns
            sorted_by_returns = sorted(
                regime_stats.items(), 
                key=lambda x: x[1]['returns'], 
                reverse=True
            )
            
            # Assign bull/bear based on returns
            if len(sorted_by_returns) >= 2:
                regime_names[sorted_by_returns[0][0]] = "Bull Market"
                regime_names[sorted_by_returns[-1][0]] = "Bear Market"
            
            # Sort remaining regimes by volatility
            remaining = [r for r in regimes if r not in regime_names]
            if remaining:
                sorted_by_vol = sorted(
                    [(r, regime_stats[r]['volatility']) for r in remaining],
                    key=lambda x: x[1],
                    reverse=True
                )
                
                # Assign high/low volatility
                if len(sorted_by_vol) >= 2:
                    regime_names[sorted_by_vol[0][0]] = "High Volatility"
                    regime_names[sorted_by_vol[-1][0]] = "Low Volatility"
                
                # Any remaining regimes are sideways
                for r in remaining:
                    if r not in regime_names:
                        regime_names[r] = "Sideways Market"
            
            return data['regime'].map(regime_names)
    
    def save_model(self, filename: str):
        """
        Save the trained model.
        
        Args:
            filename: Name of the file
        """
        filepath = os.path.join(self.data_dir, filename)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model parameters
        params = {
            "method": self.method,
            "n_regimes": self.n_regimes,
            "rule_params": self.rule_params,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(params, f, indent=2)
            
        # Save scikit-learn models
        if self.method == "hmm" and self.hmm_model is not None:
            model_path = os.path.join(self.data_dir, f"{filename}_hmm.pkl")
            import joblib
            joblib.dump(self.hmm_model, model_path)
            
        elif self.method == "kmeans" and self.kmeans_model is not None:
            model_path = os.path.join(self.data_dir, f"{filename}_kmeans.pkl")
            import joblib
            joblib.dump(self.kmeans_model, model_path)
            
        # Save scaler
        scaler_path = os.path.join(self.data_dir, f"{filename}_scaler.pkl")
        import joblib
        joblib.dump(self.scaler, scaler_path)
        
        logger.info(f"Saved model to {filepath}")
    
    def load_model(self, filename: str):
        """
        Load a trained model.
        
        Args:
            filename: Name of the file
        """
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File {filepath} not found")
            
        # Load model parameters
        with open(filepath, 'r') as f:
            params = json.load(f)
            
        self.method = params["method"]
        self.n_regimes = params["n_regimes"]
        self.rule_params = params["rule_params"]
        
        # Load scikit-learn models
        if self.method == "hmm":
            model_path = os.path.join(self.data_dir, f"{filename}_hmm.pkl")
            if os.path.exists(model_path):
                import joblib
                self.hmm_model = joblib.load(model_path)
                
        elif self.method == "kmeans":
            model_path = os.path.join(self.data_dir, f"{filename}_kmeans.pkl")
            if os.path.exists(model_path):
                import joblib
                self.kmeans_model = joblib.load(model_path)
                
        # Load scaler
        scaler_path = os.path.join(self.data_dir, f"{filename}_scaler.pkl")
        if os.path.exists(scaler_path):
            import joblib
            self.scaler = joblib.load(scaler_path)
            
        logger.info(f"Loaded model from {filepath}")
    
    def plot_regimes(self, data: pd.DataFrame, save_path: Optional[str] = None):
        """
        Plot market data with regime labels.
        
        Args:
            data: DataFrame with market data and regime labels
            save_path: Optional path to save the plot
        """
        if 'regime' not in data.columns or 'regime_name' not in data.columns:
            raise ValueError("Data must include 'regime' and 'regime_name' columns")
            
        plt.figure(figsize=(15, 10))
        
        # Plot price
        ax1 = plt.subplot(2, 1, 1)
        ax1.plot(data.index, data['close'], label='Price', color='black')
        ax1.set_title('Market Regimes')
        ax1.set_ylabel('Price')
        ax1.legend()
        
        # Color the background based on regimes
        regimes = data['regime'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(regimes)))
        
        for i, regime in enumerate(regimes):
            regime_data = data[data['regime'] == regime]
            if len(regime_data) > 0:
                # Get first occurrence of regime name
                regime_name = regime_data['regime_name'].iloc[0]
                
                # Find contiguous segments
                segments = self._find_contiguous_segments(regime_data.index)
                
                for segment in segments:
                    start, end = segment
                    ax1.axvspan(start, end, alpha=0.3, color=colors[i], 
                              label=f"{regime_name}" if segment == segments[0] else "")
        
        # Remove duplicate labels
        handles, labels = ax1.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax1.legend(by_label.values(), by_label.keys())
        
        # Plot returns
        ax2 = plt.subplot(2, 1, 2, sharex=ax1)
        ax2.plot(data.index, data['close'].pct_change().rolling(20).mean() * 100, 
               label='20-day Rolling Returns (%)', color='blue')
        ax2.set_ylabel('Returns (%)')
        ax2.set_xlabel('Date')
        ax2.axhline(y=0, color='gray', linestyle='--')
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
    
    def _find_contiguous_segments(self, index: pd.DatetimeIndex) -> List[Tuple]:
        """
        Find contiguous segments in a DatetimeIndex.
        
        Args:
            index: DatetimeIndex to find segments in
            
        Returns:
            List of (start, end) tuples
        """
        if len(index) == 0:
            return []
            
        segments = []
        start = index[0]
        prev = index[0]
        
        for i in range(1, len(index)):
            curr = index[i]
            # Check if dates are consecutive
            if (curr - prev).days > 1:
                segments.append((start, prev))
                start = curr
            prev = curr
            
        # Add the last segment
        segments.append((start, prev))
        
        return segments


class RegimeValidator:
    """Expert validation framework for market regimes."""
    
    def __init__(self, data_dir: str = "data/regimes/validation"):
        """
        Initialize the regime validator.
        
        Args:
            data_dir: Directory for storing validation data
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.expert_validations = {}
    
    def add_expert_validation(self, 
                            expert_name: str, 
                            market_data: pd.DataFrame,
                            regime_labels: Dict[Tuple[datetime, datetime], str]):
        """
        Add expert validation for market regimes.
        
        Args:
            expert_name: Name of the expert
            market_data: DataFrame with market data
            regime_labels: Dictionary mapping (start_date, end_date) to regime name
        """
        # Convert dates to strings for JSON serialization
        labels = {}
        for (start, end), regime in regime_labels.items():
            start_str = start.isoformat() if isinstance(start, datetime) else start
            end_str = end.isoformat() if isinstance(end, datetime) else end
            labels[f"{start_str}_{end_str}"] = regime
            
        # Store validation
        self.expert_validations[expert_name] = {
            "labels": labels,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save to file
        filepath = os.path.join(self.data_dir, f"{expert_name}_validation.json")
        with open(filepath, 'w') as f:
            json.dump(self.expert_validations[expert_name], f, indent=2)
            
        logger.info(f"Added expert validation from {expert_name}")
    
    def load_expert_validations(self):
        """Load all expert validations from files."""
        self.expert_validations = {}
        
        if not os.path.exists(self.data_dir):
            return
            
        validation_files = [f for f in os.listdir(self.data_dir) 
                          if f.endswith("_validation.json")]
        
        for file in validation_files:
            expert_name = file.replace("_validation.json", "")
            filepath = os.path.join(self.data_dir, file)
            
            with open(filepath, 'r') as f:
                validation = json.load(f)
                self.expert_validations[expert_name] = validation
                
        logger.info(f"Loaded {len(self.expert_validations)} expert validations")
    
    def compare_with_model(self, 
                         classifier: MarketRegimeClassifier,
                         market_data: pd.DataFrame) -> Dict:
        """
        Compare model predictions with expert validations.
        
        Args:
            classifier: Trained MarketRegimeClassifier
            market_data: DataFrame with market data
            
        Returns:
            Dictionary with comparison metrics
        """
        if not self.expert_validations:
            self.load_expert_validations()
            
        if not self.expert_validations:
            raise ValueError("No expert validations available")
            
        # Get model predictions
        predictions = classifier.predict(market_data)
        
        # Compare with each expert
        results = {}
        
        for expert_name, validation in self.expert_validations.items():
            expert_labels = validation["labels"]
            
            # Convert string dates back to datetime
            expert_regimes = {}
            for date_range, regime in expert_labels.items():
                start_str, end_str = date_range.split("_")
                start = datetime.fromisoformat(start_str)
                end = datetime.fromisoformat(end_str)
                expert_regimes[(start, end)] = regime
                
            # Calculate agreement metrics
            agreement = self._calculate_agreement(
                predictions, expert_regimes, market_data.index)
            
            results[expert_name] = agreement
            
        # Calculate average agreement
        avg_agreement = sum(r["agreement_pct"] for r in results.values()) / len(results)
        
        return {
            "expert_comparisons": results,
            "average_agreement": avg_agreement
        }
    
    def _calculate_agreement(self, 
                           predictions: pd.DataFrame,
                           expert_regimes: Dict[Tuple[datetime, datetime], str],
                           dates: pd.DatetimeIndex) -> Dict:
        """
        Calculate agreement between model predictions and expert labels.
        
        Args:
            predictions: DataFrame with model predictions
            expert_regimes: Dictionary mapping (start_date, end_date) to regime name
            dates: DatetimeIndex of market data
            
        Returns:
            Dictionary with agreement metrics
        """
        # Create a Series with expert labels for each date
        expert_labels = pd.Series(index=dates, dtype=str)
        
        for (start, end), regime in expert_regimes.items():
            mask = (dates >= start) & (dates <= end)
            expert_labels[mask] = regime
            
        # Fill any gaps with "Unknown"
        expert_labels = expert_labels.fillna("Unknown")
        
        # Count agreements
        total_dates = len(dates)
        agreement_count = sum(
            predictions.loc[date, 'regime_name'] == expert_labels[date]
            for date in dates if date in predictions.index and date in expert_labels.index
        )
        
        agreement_pct = agreement_count / total_dates * 100
        
        # Create confusion matrix
        model_regimes = predictions['regime_name'].unique()
        expert_regime_names = expert_labels.unique()
        
        confusion = {}
        for model_regime in model_regimes:
            confusion[model_regime] = {}
            for expert_regime in expert_regime_names:
                model_mask = predictions['regime_name'] == model_regime
                expert_mask = expert_labels == expert_regime
                
                # Count dates where both masks are True
                overlap = sum(
                    model_mask.loc[date] and expert_mask.loc[date]
                    for date in dates if date in model_mask.index and date in expert_mask.index
                )
                
                confusion[model_regime][expert_regime] = overlap
                
        return {
            "agreement_count": agreement_count,
            "total_dates": total_dates,
            "agreement_pct": agreement_pct,
            "confusion_matrix": confusion
        }
    
    def plot_comparison(self, 
                      classifier: MarketRegimeClassifier,
                      market_data: pd.DataFrame,
                      expert_name: Optional[str] = None,
                      save_path: Optional[str] = None):
        """
        Plot comparison between model predictions and expert validations.
        
        Args:
            classifier: Trained MarketRegimeClassifier
            market_data: DataFrame with market data
            expert_name: Name of the expert to compare with (None for all)
            save_path: Optional path to save the plot
        """
        if not self.expert_validations:
            self.load_expert_validations()
            
        if not self.expert_validations:
            raise ValueError("No expert validations available")
            
        # Get model predictions
        predictions = classifier.predict(market_data)
        
        # Select experts to compare with
        if expert_name:
            if expert_name not in self.expert_validations:
                raise ValueError(f"Expert {expert_name} not found")
            experts = {expert_name: self.expert_validations[expert_name]}
        else:
            experts = self.expert_validations
            
        # Create plot
        n_experts = len(experts)
        plt.figure(figsize=(15, 5 + 3 * n_experts))
        
        # Plot price with model regimes
        ax1 = plt.subplot(n_experts + 1, 1, 1)
        ax1.plot(market_data.index, market_data['close'], label='Price', color='black')
        ax1.set_title('Model Regimes')
        ax1.set_ylabel('Price')
        
        # Color the background based on model regimes
        regimes = predictions['regime'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(regimes)))
        
        for i, regime in enumerate(regimes):
            regime_data = predictions[predictions['regime'] == regime]
            if len(regime_data) > 0:
                regime_name = regime_data['regime_name'].iloc[0]
                segments = classifier._find_contiguous_segments(regime_data.index)
                
                for segment in segments:
                    start, end = segment
                    ax1.axvspan(start, end, alpha=0.3, color=colors[i], 
                              label=f"{regime_name}" if segment == segments[0] else "")
        
        # Remove duplicate labels
        handles, labels = ax1.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax1.legend(by_label.values(), by_label.keys())
        
        # Plot expert validations
        for i, (name, validation) in enumerate(experts.items(), 1):
            ax = plt.subplot(n_experts + 1, 1, i + 1, sharex=ax1)
            ax.plot(market_data.index, market_data['close'], label='Price', color='black')
            ax.set_title(f'Expert: {name}')
            ax.set_ylabel('Price')
            
            # Convert string dates back to datetime and plot expert regimes
            expert_labels = validation["labels"]
            unique_regimes = set(expert_labels.values())
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_regimes)))
            color_map = {regime: colors[i] for i, regime in enumerate(unique_regimes)}
            
            for date_range, regime in expert_labels.items():
                start_str, end_str = date_range.split("_")
                start = datetime.fromisoformat(start_str)
                end = datetime.fromisoformat(end_str)
                
                if start in market_data.index and end in market_data.index:
                    ax.axvspan(start, end, alpha=0.3, color=color_map[regime], label=regime)
            
            # Remove duplicate labels
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys())
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
    
    def _find_contiguous_segments(self, index: pd.DatetimeIndex) -> List[Tuple]:
        """
        Find contiguous segments in a DatetimeIndex.
        
        Args:
            index: DatetimeIndex to find segments in
            
        Returns:
            List of (start, end) tuples
        """
        if len(index) == 0:
            return []
            
        segments = []
        start = index[0]
        prev = index[0]
        
        for i in range(1, len(index)):
            curr = index[i]
            # Check if dates are consecutive
            if (curr - prev).days > 1:
                segments.append((start, prev))
                start = curr
            prev = curr
            
        # Add the last segment
        segments.append((start, prev))
        
        return segments


# Example usage
if __name__ == "__main__":
    # Load sample market data
    import yfinance as yf
    
    # Download S&P 500 data
    sp500 = yf.download("^GSPC", start="2000-01-01", end="2022-12-31")
    
    # Train regime classifier using HMM
    hmm_classifier = MarketRegimeClassifier(method="hmm", n_regimes=4)
    hmm_classifier.fit(sp500)
    
    # Predict regimes
    hmm_regimes = hmm_classifier.predict(sp500)
    
    # Plot regimes
    hmm_classifier.plot_regimes(hmm_regimes)
    
    # Train regime classifier using rule-based approach
    rule_classifier = MarketRegimeClassifier(method="rule_based")
    rule_regimes = rule_classifier.predict(sp500)
    
    # Plot regimes
    rule_classifier.plot_regimes(rule_regimes)
