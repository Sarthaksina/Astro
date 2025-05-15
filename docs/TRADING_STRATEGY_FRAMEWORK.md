# Cosmic Market Oracle - Trading Strategy Framework

## Overview

The Trading Strategy Framework is a comprehensive system for developing, testing, and implementing trading strategies based on Vedic astrological signals and market data. This framework enables traders and researchers to leverage the power of Vedic astrology for financial market prediction and trading decision-making.

## Key Components

The framework consists of the following key components:

1. **Strategy Framework** - Base classes and implementations for trading strategies
2. **Signal Generators** - Components that generate trading signals based on different astrological factors
3. **Backtesting Engine** - Tools for testing strategies against historical data and evaluating performance
4. **Visualization** - Tools for visualizing strategy performance and trading signals

## Getting Started

### Prerequisites

- Python 3.10
- Required packages: pandas, numpy, matplotlib, seaborn
- Cosmic Market Oracle core modules

### Installation

The Trading Strategy Framework is included in the Cosmic Market Oracle project. No additional installation is required beyond the base project setup.

### Basic Usage

```python
from src.trading.strategy_framework import VedicAstrologyStrategy
from src.trading.backtest import BacktestEngine
import pandas as pd

# Load market and planetary data
market_data = pd.read_csv('market_data.csv', index_col='Date', parse_dates=True)
planetary_data = pd.read_csv('planetary_data.csv', index_col='Date', parse_dates=True)

# Create a strategy
strategy = VedicAstrologyStrategy(name="My Vedic Strategy")

# Configure strategy parameters
strategy.use_yogas = True
strategy.use_nakshatras = True
strategy.use_dashas = True
strategy.min_signal_strength = 0.7

# Create backtest engine
engine = BacktestEngine(initial_capital=100000.0, commission=0.001)

# Run backtest
results = engine.run_backtest(strategy, market_data, planetary_data, "SPY")

# Generate report
report_path = engine.generate_report("reports")

# Visualize results
engine.plot_results("reports/equity_curve.png")
engine.plot_drawdowns("reports/drawdowns.png")
```

## Strategy Framework

### BaseStrategy

The `BaseStrategy` class provides the foundation for all trading strategies. It includes methods for:

- Generating trading signals
- Calculating position size
- Applying risk management rules
- Executing trades
- Calculating performance metrics

### VedicAstrologyStrategy

The `VedicAstrologyStrategy` class extends `BaseStrategy` to implement a trading strategy based on Vedic astrological signals. It considers:

- Market trends derived from planetary positions
- Financial yogas (planetary combinations)
- Nakshatra (lunar mansion) analysis
- Vimshottari Dasha periods

#### Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `use_yogas` | Whether to use financial yogas | `True` |
| `use_nakshatras` | Whether to use nakshatra analysis | `True` |
| `use_dashas` | Whether to use dasha periods | `True` |
| `min_signal_strength` | Minimum signal strength to generate a trade | `0.6` |

## Signal Generators

The framework includes several signal generators that can be used individually or combined:

### VedicNakshatraSignalGenerator

Generates signals based on the Moon's position in different nakshatras (lunar mansions). Each nakshatra has different financial characteristics that can indicate bullish or bearish market conditions.

### VedicYogaSignalGenerator

Generates signals based on Vedic astrological yogas (planetary combinations). Certain yogas are associated with financial prosperity or difficulty.

### VedicDashaSignalGenerator

Generates signals based on Vimshottari Dasha periods. The planetary lord of the current dasha period influences financial outcomes.

### CombinedSignalGenerator

Combines signals from multiple generators with weighted aggregation to produce a more robust signal.

## Backtesting

### BacktestEngine

The `BacktestEngine` class provides tools for testing trading strategies against historical data. It includes:

- Running backtests for individual strategies
- Comparing multiple strategies
- Calculating performance metrics
- Generating reports
- Visualizing results

#### Performance Metrics

| Metric | Description |
|--------|-------------|
| Total Return | Total percentage return over the backtest period |
| Annualized Return | Return normalized to an annual basis |
| Volatility | Standard deviation of returns (annualized) |
| Sharpe Ratio | Risk-adjusted return (return / volatility) |
| Max Drawdown | Maximum percentage decline from peak to trough |
| Win Rate | Percentage of winning trades |
| Total Trades | Number of trades executed |
| Avg Trade Duration | Average holding period for trades |

### BacktestRunner

The `BacktestRunner` class provides a higher-level interface for running backtests with different configurations. It includes:

- Running backtests for Vedic astrology strategies
- Parameter sweeps to find optimal strategy configurations
- Visualizing results

## Examples

The `examples` directory contains sample scripts demonstrating how to use the framework:

- `vedic_trading_strategy_example.py` - A complete example of creating, backtesting, and visualizing a Vedic astrology trading strategy

## Advanced Usage

### Creating Custom Strategies

You can create custom strategies by extending the `BaseStrategy` class:

```python
from src.trading.strategy_framework import BaseStrategy
import pandas as pd

class MyCustomStrategy(BaseStrategy):
    def __init__(self, name="My Custom Strategy", description=""):
        super().__init__(name, description)
        # Initialize strategy parameters
        
    def generate_signals(self, market_data, planetary_data):
        # Implement your signal generation logic
        signals = pd.DataFrame(index=market_data.index)
        signals["signal"] = 0  # 0: no signal, 1: buy, -1: sell
        signals["strength"] = 0.0
        signals["reason"] = ""
        
        # Your custom logic here
        
        return signals
```

### Creating Custom Signal Generators

You can create custom signal generators by extending the `SignalGenerator` class:

```python
from src.trading.signal_generator import SignalGenerator
import pandas as pd

class MyCustomSignalGenerator(SignalGenerator):
    def __init__(self, name="My Custom Generator", description=""):
        super().__init__(name, description)
        # Initialize generator parameters
        
    def generate_signals(self, market_data, planetary_data):
        # Implement your signal generation logic
        signals = pd.DataFrame(index=market_data.index)
        signals["signal"] = 0  # 0: no signal, 1: buy, -1: sell
        signals["strength"] = 0.0
        signals["reason"] = ""
        
        # Your custom logic here
        
        return signals
```

## Best Practices

1. **Start Simple** - Begin with a single signal generator and gradually add complexity
2. **Validate Signals** - Use the SignalFilter to remove weak or conflicting signals
3. **Test Thoroughly** - Use the backtesting framework to test strategies across different market conditions
4. **Optimize Parameters** - Use parameter sweeps to find optimal strategy configurations
5. **Combine Signals** - Use the CombinedSignalGenerator to integrate multiple signal sources
6. **Monitor Performance** - Regularly evaluate strategy performance and make adjustments as needed

## Limitations and Considerations

- Past performance does not guarantee future results
- Backtesting may suffer from look-ahead bias or overfitting
- Market conditions change over time, requiring strategy adaptation
- Transaction costs and slippage can significantly impact real-world performance
- Vedic astrological signals should be used in conjunction with traditional market analysis

## Future Enhancements

- Integration with live trading platforms
- Machine learning-enhanced signal generation
- Additional Vedic astrological factors
- Portfolio optimization
- Risk management enhancements
- Real-time market monitoring

## Contributing

Contributions to the Trading Strategy Framework are welcome. Please follow the project's coding standards and submit pull requests for review.

## License

This project is licensed under the terms specified in the main Cosmic Market Oracle project.
