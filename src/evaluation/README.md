# Evaluation and Validation Framework

## Overview

The Evaluation and Validation Framework provides comprehensive tools for assessing the performance of prediction models in the Cosmic Market Oracle system. It includes modules for calculating various metrics, validating models across different market regimes, visualizing results, and comparing against benchmarks.

## Components

### 1. Metrics Module (`metrics.py`)

This module provides a wide range of metrics for evaluating prediction models:

- **Regression Metrics**: RMSE, MAE, MAPE, R², etc.
- **Classification Metrics**: Accuracy, Precision, Recall, F1-score, etc.
- **Trading Metrics**: Profit/Loss, Sharpe Ratio, Maximum Drawdown, etc.
- **Robustness Metrics**: Statistical significance tests, sensitivity analysis, etc.

```python
from src.evaluation.metrics import PredictionMetrics

# Create metrics calculator
metrics = PredictionMetrics()

# Calculate regression metrics
regression_metrics = metrics.calculate_regression_metrics(predictions, actuals)

# Calculate classification metrics
classification_metrics = metrics.calculate_classification_metrics(
    predicted_classes, actual_classes
)

# Calculate trading metrics
trading_metrics = metrics.calculate_trading_metrics(signals, prices)

# Test statistical significance
significance = metrics.test_statistical_significance(predictions, actuals, baseline_predictions)
```

### 2. Validation Module (`validation.py`)

This module implements various validation methodologies for time series data:

- **Walk-Forward Validation**: Train on historical data, validate on future data
- **Cross-Market Validation**: Test model generalization across different markets
- **Regime-Based Validation**: Evaluate performance across different market regimes

```python
from src.evaluation.validation import WalkForwardValidator, CrossMarketValidator

# Create walk-forward validator
wf_validator = WalkForwardValidator(n_splits=5, test_size=30)

# Validate model
wf_results = wf_validator.validate(model_factory, X, y)

# Create cross-market validator
cm_validator = CrossMarketValidator(markets=['SPY', 'QQQ', 'IWM'])

# Validate model across markets
cm_results = cm_validator.validate(model_factory, data_dict)
```

### 3. Visualization Module (`visualization.py`)

This module provides tools for visualizing evaluation results:

- **Performance Plots**: Actual vs. Predicted, Error Distribution, etc.
- **Time Series Visualizations**: Prediction over time, Cumulative Returns, etc.
- **Comparative Visualizations**: Model comparison, Feature importance, etc.

```python
from src.evaluation.visualization import PredictionVisualizer

# Create visualizer
visualizer = PredictionVisualizer()

# Plot predictions vs. actuals
fig1 = visualizer.plot_predictions_vs_actuals(predictions, actuals)

# Plot error distribution
fig2 = visualizer.plot_error_distribution(predictions, actuals)

# Plot cumulative returns
fig3 = visualizer.plot_cumulative_returns(signals, prices)
```

### 4. Regime Visualizer (`regime_visualizer.py`)

This module provides specialized visualizations for comparing model performance across different market regimes:

- **Regime Performance Comparison**: Compare metrics across regimes
- **Regime Transition Analysis**: Analyze model behavior during regime transitions
- **Regime-Specific Error Analysis**: Identify regime-specific error patterns

```python
from src.evaluation.regime_visualizer import RegimeComparisonVisualizer

# Create regime visualizer
regime_visualizer = RegimeComparisonVisualizer()

# Plot performance across regimes
fig = regime_visualizer.plot_regime_performance(
    predictions_dict, actuals_dict, regime_labels
)
```

### 5. Benchmark Module (`benchmark.py`)

This module provides tools for comparing model performance against various benchmarks:

- **Statistical Baselines**: Naive forecasts, Moving Averages, Random Walk, etc.
- **Trading Strategies**: Buy-and-Hold, Moving Average Crossover, Mean Reversion, etc.
- **Benchmark Comparison**: Compare model performance against multiple benchmarks

```python
from src.evaluation.benchmark import BenchmarkComparer

# Create benchmark comparer
comparer = BenchmarkComparer()

# Set model predictions
comparer.set_model_predictions(model_predictions)

# Compare against statistical benchmarks
prediction_comparison = comparer.compare_predictions(actuals)

# Set model trading signals
comparer.set_model_signals(model_signals)

# Compare against trading strategies
strategy_comparison = comparer.compare_strategies(prices)

# Plot equity curve comparison
fig = comparer.plot_equity_curve_comparison(prices)
```

## Usage Examples

### Basic Model Evaluation

```python
import pandas as pd
import numpy as np
from src.evaluation.metrics import PredictionMetrics
from src.evaluation.visualization import PredictionVisualizer

# Load data
data = pd.read_csv('market_data.csv')
predictions = model.predict(data['X'])
actuals = data['y']

# Calculate metrics
metrics_calculator = PredictionMetrics()
results = metrics_calculator.calculate_regression_metrics(predictions, actuals)
print(f"RMSE: {results['rmse']:.4f}")
print(f"MAE: {results['mae']:.4f}")
print(f"Directional Accuracy: {results['directional_accuracy']:.2%}")

# Visualize results
visualizer = PredictionVisualizer()
fig1 = visualizer.plot_predictions_vs_actuals(predictions, actuals)
fig2 = visualizer.plot_error_distribution(predictions, actuals)
```

### Walk-Forward Validation

```python
from src.evaluation.validation import WalkForwardValidator

# Define model factory function
def model_factory():
    # Create and return a new model instance
    return MyModel()

# Create validator
validator = WalkForwardValidator(n_splits=5, test_size=30)

# Validate model
results = validator.validate(model_factory, X, y)

# Print results
for i, result in enumerate(results):
    print(f"Fold {i+1}: RMSE = {result['metrics']['rmse']:.4f}")

# Plot validation results
validator.plot_validation_results()
```

### Benchmark Comparison

```python
from src.evaluation.benchmark import BenchmarkComparer

# Create benchmark comparer
comparer = BenchmarkComparer()

# Set model predictions
comparer.set_model_predictions(model_predictions)

# Compare against statistical benchmarks
comparison = comparer.compare_predictions(actuals)

# Print comparison results
for name, result in comparison.items():
    print(f"{result['name']}: RMSE = {result['metrics']['rmse']:.4f}")

# Plot comparison
fig = comparer.plot_prediction_comparison(actuals, metric="rmse")
```

### Regime-Based Analysis

```python
from src.evaluation.regime_visualizer import RegimeComparisonVisualizer

# Create regime visualizer
visualizer = RegimeComparisonVisualizer()

# Define regime data
predictions_dict = {
    'bull_market': bull_predictions,
    'bear_market': bear_predictions,
    'sideways_market': sideways_predictions
}
actuals_dict = {
    'bull_market': bull_actuals,
    'bear_market': bear_actuals,
    'sideways_market': sideways_actuals
}
regime_labels = ['Bull Market', 'Bear Market', 'Sideways Market']

# Plot regime comparison
fig = visualizer.plot_regime_performance(
    predictions_dict, actuals_dict, regime_labels
)
```

## Integration with Multi-Agent Orchestration

The Evaluation and Validation Framework integrates seamlessly with the Multi-Agent Orchestration Network:

- **Agent Performance Evaluation**: Evaluate the performance of individual agents
- **Consensus Strategy Evaluation**: Compare different consensus strategies
- **Agent Contribution Analysis**: Analyze the contribution of each agent to the final prediction

```python
from src.trading.multi_agent_orchestration.orchestrator import Orchestrator
from src.evaluation.metrics import PredictionMetrics

# Create orchestrator
orchestrator = Orchestrator()

# Add agents
orchestrator.add_agent(market_regime_agent)
orchestrator.add_agent(signal_generator_agent)
orchestrator.add_agent(risk_management_agent)

# Process data and get predictions
orchestrator.start()
orchestrator.process_data(data)
predictions = orchestrator.get_predictions()

# Evaluate predictions
metrics_calculator = PredictionMetrics()
results = metrics_calculator.calculate_regression_metrics(predictions, actuals)
```

## Best Practices

1. **Use Multiple Metrics**: Don't rely on a single metric; use a combination of metrics to get a comprehensive view of model performance.

2. **Compare Against Benchmarks**: Always compare your model against simple benchmarks to ensure it provides real value.

3. **Validate Across Regimes**: Market behavior varies across different regimes; ensure your model performs well in all relevant market conditions.

4. **Consider Statistical Significance**: Test whether your model's performance is statistically significantly better than benchmarks.

5. **Visualize Results**: Visualizations provide insights that may not be apparent from metrics alone.

6. **Use Proper Validation**: Time series data requires special validation techniques like walk-forward validation to avoid look-ahead bias.

7. **Analyze Errors**: Understand where and why your model makes errors to identify areas for improvement.
