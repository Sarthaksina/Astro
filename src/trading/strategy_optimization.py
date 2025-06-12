# Cosmic Market Oracle - Strategy Optimization

"""
This module provides tools for optimizing trading strategies based on
Vedic astrological signals and market data.

It includes:
- Parameter optimization for trading strategies
- Strategy performance evaluation across different market regimes
- Optimization for different objectives (return, risk, Sharpe ratio)
- Cross-validation techniques for robust parameter estimation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Callable
from datetime import datetime, timedelta
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed

from src.trading.strategy_framework import BaseStrategy # VedicAstrologyStrategy removed
from src.trading.backtest import BacktestEngine
from src.utils.logger import get_logger # Changed to get_logger
from .signal_generator import ( # Added signal generator imports
    CombinedSignalGenerator,
    VedicNakshatraSignalGenerator,
    VedicYogaSignalGenerator,
    VedicDashaSignalGenerator
)
from .constants import DEFAULT_OPTIMIZATION_METRIC

# Configure logging
logger = get_logger("strategy_optimization") # Changed to get_logger

# Definition of _OptimizerStrategyWrapper to be added here
class _OptimizerStrategyWrapper(BaseStrategy):
    def __init__(self, name: str = "OptimizedStrategy", description: str = "Strategy with configured signal generator",
                 use_yogas: bool = True, use_nakshatras: bool = True, use_dashas: bool = True,
                 min_signal_strength: float = 0.6, **kwargs): # Added **kwargs to consume other potential params
        super().__init__(name, description)
        # kwargs will contain signal generator configuration
        self.combined_gen = CombinedSignalGenerator()
        self.combined_gen.generators = [] # Clear defaults
        self.combined_gen.weights = {}

        active_generator_details = []
        # Example default weights if all are active, will be normalized
        default_weights = {'yoga': 0.4, 'nakshatra': 0.3, 'dasha': 0.3}

        if use_yogas:
            active_generator_details.append( (VedicYogaSignalGenerator(), default_weights['yoga']) )
        if use_nakshatras:
            active_generator_details.append( (VedicNakshatraSignalGenerator(), default_weights['nakshatra']) )
        if use_dashas:
            active_generator_details.append( (VedicDashaSignalGenerator(), default_weights['dasha']) )

        total_weight_for_active = sum(details[1] for details in active_generator_details)
        if active_generator_details:
            for gen_instance, relative_weight in active_generator_details:
                self.combined_gen.generators.append(gen_instance)
                self.combined_gen.weights[gen_instance.name] = relative_weight / total_weight_for_active if total_weight_for_active > 0 else 0
        else:
            logger.warning(f"No signal generators selected for {name} in _OptimizerStrategyWrapper. Strategy will produce no signals.")
            pass

        self.combined_gen.min_signal_strength = min_signal_strength

    def generate_signals(self, market_data: pd.DataFrame, planetary_data: pd.DataFrame) -> pd.DataFrame:
        return self.combined_gen.generate_signals(market_data, planetary_data)

class StrategyOptimizer:
    """Base class for strategy optimization."""
    
    def __init__(self, market_data: pd.DataFrame, planetary_data: pd.DataFrame,
                initial_capital: float = 100000.0, commission: float = 0.001):
        """
        Initialize the strategy optimizer.
        
        Args:
            market_data: Market data for backtesting
            planetary_data: Planetary data for backtesting
            initial_capital: Initial capital for backtesting
            commission: Commission rate for backtesting
        """
        self.market_data = market_data
        self.planetary_data = planetary_data
        self.initial_capital = initial_capital
        self.commission = commission
        self.backtest_engine = BacktestEngine(initial_capital=initial_capital, commission=commission)
        self.best_params = {}
        self.best_metrics = {}
    
    def optimize(self, strategy_class, param_space: Dict, 
                objective: str = DEFAULT_OPTIMIZATION_METRIC, n_trials: int = 100,
                timeout: int = None) -> Dict:
        """
        Optimize strategy parameters.
        
        Args:
            strategy_class: Strategy class to optimize
            param_space: Parameter space to search
            objective: Objective metric to maximize
            n_trials: Number of optimization trials
            timeout: Timeout in seconds
            
        Returns:
            Dictionary with best parameters
        """
        # Create Optuna study
        study = optuna.create_study(direction="maximize")
        
        # Define objective function
        def objective_func(trial):
            # Sample parameters
            params = {}
            for param_name, param_range in param_space.items():
                if isinstance(param_range, tuple) and len(param_range) == 3:
                    if isinstance(param_range[0], int):
                        params[param_name] = trial.suggest_int(param_name, param_range[0], param_range[1], param_range[2])
                    else:
                        params[param_name] = trial.suggest_float(param_name, param_range[0], param_range[1], step=param_range[2])
                elif isinstance(param_range, tuple) and len(param_range) == 2:
                    if isinstance(param_range[0], int):
                        params[param_name] = trial.suggest_int(param_name, param_range[0], param_range[1])
                    else:
                        params[param_name] = trial.suggest_float(param_name, param_range[0], param_range[1])
                elif isinstance(param_range, list):
                    params[param_name] = trial.suggest_categorical(param_name, param_range)
                elif isinstance(param_range, bool):
                    params[param_name] = trial.suggest_categorical(param_name, [True, False])
            
            # Create strategy with sampled parameters
            strategy = strategy_class(**params)
            
            # Run backtest
            results = self.backtest_engine.run_backtest(strategy, self.market_data, self.planetary_data)
            
            # Get objective metric
            metrics = results["metrics"]
            if objective == "sharpe_ratio":
                return metrics.get("sharpe_ratio", 0)
            elif objective == "total_return":
                return metrics.get("total_return", 0)
            elif objective == "annualized_return":
                return metrics.get("annualized_return", 0)
            elif objective == "win_rate":
                return metrics.get("win_rate", 0)
            else:
                return metrics.get(objective, 0)
        
        # Run optimization
        study.optimize(objective_func, n_trials=n_trials, timeout=timeout)
        
        # Get best parameters
        self.best_params = study.best_params
        
        # Run backtest with best parameters
        strategy = strategy_class(**self.best_params)
        results = self.backtest_engine.run_backtest(strategy, self.market_data, self.planetary_data)
        self.best_metrics = results["metrics"]
        
        # Log results
        logger.info(f"Best parameters: {self.best_params}")
        logger.info(f"Best metrics: {self.best_metrics}")
        
        return self.best_params
    
    def plot_optimization_results(self, study, save_path: str = None):
        """
        Plot optimization results.
        
        Args:
            study: Optuna study
            save_path: Path to save the plot
        """
        # Plot optimization history
        plt.figure(figsize=(10, 6))
        optuna.visualization.matplotlib.plot_optimization_history(study)
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()
        
        # Plot parameter importance
        plt.figure(figsize=(10, 6))
        optuna.visualization.matplotlib.plot_param_importances(study)
        if save_path:
            plt.savefig(save_path.replace(".png", "_importance.png"))
        else:
            plt.show()
        plt.close()
    
    def cross_validate(self, strategy_class, params: Dict, n_splits: int = 5) -> Dict:
        """
        Perform cross-validation for a strategy.
        
        Args:
            strategy_class: Strategy class to validate
            params: Strategy parameters
            n_splits: Number of cross-validation splits
            
        Returns:
            Dictionary with cross-validation results
        """
        # Split data into n_splits
        total_days = len(self.market_data)
        split_size = total_days // n_splits
        
        # Initialize results
        cv_results = {
            "total_return": [],
            "annualized_return": [],
            "sharpe_ratio": [],
            "max_drawdown": [],
            "win_rate": []
        }
        
        # Run backtest for each split
        for i in range(n_splits):
            # Calculate split indices
            start_idx = i * split_size
            end_idx = (i + 1) * split_size if i < n_splits - 1 else total_days
            
            # Split data
            split_market_data = self.market_data.iloc[start_idx:end_idx]
            split_planetary_data = self.planetary_data.iloc[start_idx:end_idx]
            
            # Create strategy
            strategy = strategy_class(**params)
            
            # Run backtest
            results = self.backtest_engine.run_backtest(strategy, split_market_data, split_planetary_data)
            
            # Record metrics
            metrics = results["metrics"]
            for metric in cv_results.keys():
                cv_results[metric].append(metrics.get(metric, 0))
        
        # Calculate statistics
        cv_stats = {}
        for metric, values in cv_results.items():
            cv_stats[f"{metric}_mean"] = np.mean(values)
            cv_stats[f"{metric}_std"] = np.std(values)
            cv_stats[f"{metric}_min"] = np.min(values)
            cv_stats[f"{metric}_max"] = np.max(values)
        
        # Log results
        logger.info(f"Cross-validation results: {cv_stats}")
        
        return cv_stats


class VedicStrategyOptimizer(StrategyOptimizer):
    """Optimizer for Vedic astrology trading strategies."""
    
    def __init__(self, market_data: pd.DataFrame, planetary_data: pd.DataFrame,
                initial_capital: float = 100000.0, commission: float = 0.001):
        """
        Initialize the Vedic strategy optimizer.
        
        Args:
            market_data: Market data for backtesting
            planetary_data: Planetary data for backtesting
            initial_capital: Initial capital for backtesting
            commission: Commission rate for backtesting
        """
        super().__init__(market_data, planetary_data, initial_capital, commission)
    
    def optimize_vedic_strategy(self, n_trials: int = 100, timeout: int = None) -> Dict:
        """
        Optimize Vedic astrology strategy parameters.
        
        Args:
            n_trials: Number of optimization trials
            timeout: Timeout in seconds
            
        Returns:
            Dictionary with best parameters
        """
        # Define parameter space
        param_space = {
            "name": ["Optimized Vedic Strategy"],
            "min_signal_strength": (0.3, 0.9, 0.05),
            "use_yogas": [True, False],
            "use_nakshatras": [True, False],
            "use_dashas": [True, False]
        }
        
        # Run optimization
        best_params = self.optimize(_OptimizerStrategyWrapper, param_space,
                                   objective="sharpe_ratio", n_trials=n_trials, timeout=timeout)
        
        return best_params
    
    def analyze_market_regimes(self, strategy_class, params: Dict, n_regimes: int = 3) -> Dict:
        """
        Analyze strategy performance across different market regimes.
        
        Args:
            strategy_class: Strategy class to analyze
            params: Strategy parameters
            n_regimes: Number of market regimes to identify
            
        Returns:
            Dictionary with performance metrics for each regime
        """
        # Calculate market returns
        market_returns = self.market_data["Close"].pct_change().dropna()
        
        # Calculate rolling volatility
        rolling_vol = market_returns.rolling(window=20).std() * np.sqrt(252)
        
        # Identify market regimes based on volatility
        regimes = pd.qcut(rolling_vol, n_regimes, labels=False)
        regime_data = pd.DataFrame({"regime": regimes}, index=rolling_vol.index)
        
        # Merge with market data
        market_with_regimes = pd.merge(self.market_data, regime_data, left_index=True, right_index=True, how="left")
        
        # Initialize results
        regime_results = {}
        
        # Run backtest for each regime
        for regime in range(n_regimes):
            # Filter data for this regime
            regime_market_data = market_with_regimes[market_with_regimes["regime"] == regime].drop(columns=["regime"])
            regime_planetary_data = self.planetary_data.loc[regime_market_data.index]
            
            # Create strategy
            strategy = strategy_class(**params)
            
            # Run backtest
            results = self.backtest_engine.run_backtest(strategy, regime_market_data, regime_planetary_data)
            
            # Record metrics
            regime_results[f"regime_{regime}"] = results["metrics"]
        
        # Log results
        logger.info(f"Market regime analysis: {regime_results}")
        
        return regime_results
    
    def optimize_for_regime(self, regime_data: Dict[str, pd.DataFrame], 
                          n_trials: int = 50, timeout: int = None) -> Dict:
        """
        Optimize strategy parameters for specific market regimes.
        
        Args:
            regime_data: Dictionary with market and planetary data for each regime
            n_trials: Number of optimization trials
            timeout: Timeout in seconds
            
        Returns:
            Dictionary with best parameters for each regime
        """
        # Initialize results
        regime_params = {}
        
        # Optimize for each regime
        for regime, data in regime_data.items():
            # Set data for this regime
            self.market_data = data["market_data"]
            self.planetary_data = data["planetary_data"]
            
            # Define parameter space
            param_space = {
                "name": [f"Regime-Specific Vedic Strategy ({regime})"],
                "min_signal_strength": (0.3, 0.9, 0.05),
                "use_yogas": [True, False],
                "use_nakshatras": [True, False],
                "use_dashas": [True, False]
            }
            
            # Run optimization
            best_params = self.optimize(_OptimizerStrategyWrapper, param_space,
                                      objective="sharpe_ratio", n_trials=n_trials, timeout=timeout)
            
            # Record best parameters
            regime_params[regime] = best_params
        
        # Log results
        logger.info(f"Regime-specific optimization results: {regime_params}")
        
        return regime_params


class MultiObjectiveOptimizer(StrategyOptimizer):
    """Multi-objective optimizer for trading strategies."""
    
    def __init__(self, market_data: pd.DataFrame, planetary_data: pd.DataFrame,
                initial_capital: float = 100000.0, commission: float = 0.001):
        """
        Initialize the multi-objective optimizer.
        
        Args:
            market_data: Market data for backtesting
            planetary_data: Planetary data for backtesting
            initial_capital: Initial capital for backtesting
            commission: Commission rate for backtesting
        """
        super().__init__(market_data, planetary_data, initial_capital, commission)
    
    def optimize_multi_objective(self, strategy_class, param_space: Dict, 
                              objectives: List[str], n_trials: int = 100,
                              timeout: int = None) -> Dict:
        """
        Perform multi-objective optimization.
        
        Args:
            strategy_class: Strategy class to optimize
            param_space: Parameter space to search
            objectives: List of objectives to maximize
            n_trials: Number of optimization trials
            timeout: Timeout in seconds
            
        Returns:
            Dictionary with Pareto-optimal parameters
        """
        # Create Optuna study for multi-objective optimization
        study = optuna.create_study(directions=["maximize"] * len(objectives))
        
        # Define objective function
        def objective_func(trial):
            # Sample parameters
            params = {}
            for param_name, param_range in param_space.items():
                if isinstance(param_range, tuple) and len(param_range) == 3:
                    if isinstance(param_range[0], int):
                        params[param_name] = trial.suggest_int(param_name, param_range[0], param_range[1], param_range[2])
                    else:
                        params[param_name] = trial.suggest_float(param_name, param_range[0], param_range[1], step=param_range[2])
                elif isinstance(param_range, tuple) and len(param_range) == 2:
                    if isinstance(param_range[0], int):
                        params[param_name] = trial.suggest_int(param_name, param_range[0], param_range[1])
                    else:
                        params[param_name] = trial.suggest_float(param_name, param_range[0], param_range[1])
                elif isinstance(param_range, list):
                    params[param_name] = trial.suggest_categorical(param_name, param_range)
                elif isinstance(param_range, bool):
                    params[param_name] = trial.suggest_categorical(param_name, [True, False])
            
            # Create strategy with sampled parameters
            strategy = strategy_class(**params)
            
            # Run backtest
            results = self.backtest_engine.run_backtest(strategy, self.market_data, self.planetary_data)
            
            # Get objective metrics
            metrics = results["metrics"]
            return [metrics.get(objective, 0) for objective in objectives]
        
        # Run optimization
        study.optimize(objective_func, n_trials=n_trials, timeout=timeout)
        
        # Get Pareto-optimal solutions
        pareto_solutions = []
        for trial in study.best_trials:
            solution = {
                "params": trial.params,
                "values": trial.values
            }
            pareto_solutions.append(solution)
        
        # Log results
        logger.info(f"Found {len(pareto_solutions)} Pareto-optimal solutions")
        
        return pareto_solutions
    
    def plot_pareto_front(self, pareto_solutions: List[Dict], objectives: List[str], 
                        save_path: str = None):
        """
        Plot Pareto front for multi-objective optimization.
        
        Args:
            pareto_solutions: List of Pareto-optimal solutions
            objectives: List of objectives
            save_path: Path to save the plot
        """
        if len(objectives) == 2:
            # 2D Pareto front
            plt.figure(figsize=(10, 6))
            
            # Extract values for each objective
            x_values = [solution["values"][0] for solution in pareto_solutions]
            y_values = [solution["values"][1] for solution in pareto_solutions]
            
            # Plot Pareto front
            plt.scatter(x_values, y_values, c="blue", s=50)
            plt.xlabel(objectives[0])
            plt.ylabel(objectives[1])
            plt.title("Pareto Front")
            plt.grid(True)
            
            if save_path:
                plt.savefig(save_path)
            else:
                plt.show()
            plt.close()
        
        elif len(objectives) == 3:
            # 3D Pareto front
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")
            
            # Extract values for each objective
            x_values = [solution["values"][0] for solution in pareto_solutions]
            y_values = [solution["values"][1] for solution in pareto_solutions]
            z_values = [solution["values"][2] for solution in pareto_solutions]
            
            # Plot Pareto front
            ax.scatter(x_values, y_values, z_values, c="blue", s=50)
            ax.set_xlabel(objectives[0])
            ax.set_ylabel(objectives[1])
            ax.set_zlabel(objectives[2])
            ax.set_title("Pareto Front")
            
            if save_path:
                plt.savefig(save_path)
            else:
                plt.show()
            plt.close()
        
        else:
            logger.warning("Pareto front visualization is only supported for 2 or 3 objectives")


class ScenarioAnalysis:
    """Scenario analysis for trading strategies."""
    
    def __init__(self, strategy: BaseStrategy, market_data: pd.DataFrame, 
                planetary_data: pd.DataFrame, initial_capital: float = 100000.0,
                commission: float = 0.001):
        """
        Initialize the scenario analysis.
        
        Args:
            strategy: Trading strategy to analyze
            market_data: Market data for backtesting
            planetary_data: Planetary data for backtesting
            initial_capital: Initial capital for backtesting
            commission: Commission rate for backtesting
        """
        self.strategy = strategy
        self.market_data = market_data
        self.planetary_data = planetary_data
        self.initial_capital = initial_capital
        self.commission = commission
        self.backtest_engine = BacktestEngine(initial_capital=initial_capital, commission=commission)
    
    def run_scenario_analysis(self, scenarios: Dict[str, Dict]) -> Dict:
        """
        Run scenario analysis for different market conditions.
        
        Args:
            scenarios: Dictionary with scenario definitions
            
        Returns:
            Dictionary with scenario results
        """
        # Initialize results
        scenario_results = {}
        
        # Run backtest for each scenario
        for scenario_name, scenario_params in scenarios.items():
            # Create modified market data for this scenario
            scenario_market_data = self._modify_market_data(scenario_params)
            
            # Run backtest
            results = self.backtest_engine.run_backtest(
                self.strategy, scenario_market_data, self.planetary_data)
            
            # Record results
            scenario_results[scenario_name] = {
                "metrics": results["metrics"],
                "equity_curve": results["results"]["equity"]
            }
        
        # Log results
        logger.info(f"Scenario analysis results: {scenario_results}")
        
        return scenario_results
    
    def _modify_market_data(self, scenario_params: Dict) -> pd.DataFrame:
        """
        Modify market data based on scenario parameters.
        
        Args:
            scenario_params: Scenario parameters
            
        Returns:
            Modified market data
        """
        # Make a copy of the original data
        modified_data = self.market_data.copy()
        
        # Apply volatility adjustment
        if "volatility_factor" in scenario_params:
            vol_factor = scenario_params["volatility_factor"]
            
            # Calculate returns
            returns = modified_data["Close"].pct_change()
            
            # Calculate mean return
            mean_return = returns.mean()
            
            # Generate new returns with adjusted volatility
            adjusted_returns = mean_return + (returns - mean_return) * vol_factor
            
            # Reconstruct prices
            initial_price = modified_data["Close"].iloc[0]
            new_prices = initial_price * (1 + adjusted_returns).cumprod()
            
            # Replace Close prices
            modified_data["Close"] = new_prices
            
            # Adjust other price columns proportionally
            for col in ["Open", "High", "Low"]:
                if col in modified_data.columns:
                    ratio = modified_data[col] / modified_data["Close"]
                    modified_data[col] = new_prices * ratio
        
        # Apply trend adjustment
        if "trend_factor" in scenario_params:
            trend_factor = scenario_params["trend_factor"]
            
            # Add trend component to returns
            daily_trend = trend_factor / 252  # Annualized to daily
            
            # Calculate returns
            returns = modified_data["Close"].pct_change()
            
            # Add trend
            adjusted_returns = returns + daily_trend
            
            # Reconstruct prices
            initial_price = modified_data["Close"].iloc[0]
            new_prices = initial_price * (1 + adjusted_returns).cumprod()
            
            # Replace Close prices
            modified_data["Close"] = new_prices
            
            # Adjust other price columns proportionally
            for col in ["Open", "High", "Low"]:
                if col in modified_data.columns:
                    ratio = modified_data[col] / modified_data["Close"]
                    modified_data[col] = new_prices * ratio
        
        # Apply market crash scenario
        if "crash" in scenario_params and scenario_params["crash"]:
            crash_day = scenario_params.get("crash_day", len(modified_data) // 2)
            crash_magnitude = scenario_params.get("crash_magnitude", -0.2)
            recovery_days = scenario_params.get("recovery_days", 60)
            
            # Apply crash
            if crash_day < len(modified_data):
                # Calculate crash and recovery factors
                pre_crash_price = modified_data["Close"].iloc[crash_day]
                post_crash_price = pre_crash_price * (1 + crash_magnitude)
                
                # Apply crash
                modified_data.iloc[crash_day, modified_data.columns.get_loc("Close")] = post_crash_price
                
                # Apply recovery
                if crash_day + 1 < len(modified_data):
                    recovery_end = min(crash_day + recovery_days, len(modified_data))
                    recovery_days_actual = recovery_end - crash_day - 1
                    
                    if recovery_days_actual > 0:
                        # Calculate recovery path
                        recovery_factor = (modified_data["Close"].iloc[crash_day - 1] / post_crash_price) ** (1 / recovery_days_actual)
                        
                        # Apply recovery
                        for i in range(crash_day + 1, recovery_end):
                            day_from_crash = i - crash_day
                            recovery_progress = recovery_factor ** day_from_crash
                            modified_data.iloc[i, modified_data.columns.get_loc("Close")] = post_crash_price * recovery_progress
        
        return modified_data
    
    def plot_scenario_results(self, scenario_results: Dict, save_path: str = None):
        """
        Plot scenario analysis results.
        
        Args:
            scenario_results: Dictionary with scenario results
            save_path: Path to save the plot
        """
        # Plot equity curves
        plt.figure(figsize=(12, 8))
        
        for scenario_name, results in scenario_results.items():
            equity_curve = results["equity_curve"]
            plt.plot(equity_curve.index, equity_curve, label=scenario_name)
        
        plt.xlabel("Date")
        plt.ylabel("Equity")
        plt.title("Scenario Analysis - Equity Curves")
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()
        
        # Plot performance metrics
        metrics = ["total_return", "sharpe_ratio", "max_drawdown", "win_rate"]
        
        plt.figure(figsize=(12, 8))
        
        # Create data for bar chart
        scenarios = list(scenario_results.keys())
        metric_data = {metric: [scenario_results[scenario]["metrics"].get(metric, 0) for scenario in scenarios] for metric in metrics}
        
        # Plot each metric
        for i, metric in enumerate(metrics):
            plt.subplot(2, 2, i + 1)
            plt.bar(scenarios, metric_data[metric])
            plt.title(metric)
            plt.xticks(rotation=45)
            plt.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path.replace(".png", "_metrics.png"))
        else:
            plt.show()
        plt.close()
