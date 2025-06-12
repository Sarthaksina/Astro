# src/trading/constants.py

"""Constants for the trading package."""

# market_simulator.py defaults
DEFAULT_INITIAL_CAPITAL = 100000.0
DEFAULT_COMMISSION_RATE = 0.001  # 0.1%
DEFAULT_SLIPPAGE_PERCENT = 0.0005 # 0.05%

# portfolio_construction.py defaults
MAX_PORTFOLIO_WEIGHT = 0.2  # Maximum weight for a single asset
MIN_PORTFOLIO_WEIGHT = 0.01 # Minimum weight for a non-zero position
RISK_FREE_RATE = 0.02 # Annual risk-free rate (e.g., for Sharpe ratio)

# strategy_optimization.py defaults
DEFAULT_OPTIMIZATION_METRIC = "sharpe_ratio"
DEFAULT_N_EPOCHS = 50 # For genetic algorithms or similar iterative optimizations
DEFAULT_POPULATION_SIZE = 100
DEFAULT_TOURNAMENT_SIZE = 5
DEFAULT_MUTATION_RATE = 0.1
DEFAULT_CROSSOVER_RATE = 0.7
