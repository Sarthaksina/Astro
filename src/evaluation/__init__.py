#!/usr/bin/env python
# Cosmic Market Oracle - Evaluation and Validation Framework

"""
Evaluation and Validation Framework for the Cosmic Market Oracle.

This package provides tools for evaluating and validating prediction models
across different market regimes, ensuring reliability and robustness.
"""

from src.evaluation.metrics import (
    PredictionMetrics,
    RegimeSpecificMetrics,
    TurningPointMetrics,
    RobustnessMetrics
)

from src.evaluation.validation import (
    WalkForwardValidator,
    CrossMarketValidator,
    TemporalValidator,
    StatisticalSignificanceTester
)

from src.evaluation.visualization import (
    PerformanceVisualizer,
    RegimeComparisonVisualizer,
    MetricDistributionPlotter,
    RobustnessHeatmapPlotter
)

from src.evaluation.benchmark import (
    BenchmarkComparer,
    BaselineBenchmarks,
    TradingStrategyBenchmarks
)

__all__ = [
    'PredictionMetrics',
    'RegimeSpecificMetrics',
    'TurningPointMetrics',
    'RobustnessMetrics',
    'WalkForwardValidator',
    'CrossMarketValidator',
    'TemporalValidator',
    'StatisticalSignificanceTester',
    'PerformanceVisualizer',
    'RegimeComparisonVisualizer',
    'MetricDistributionPlotter',
    'RobustnessHeatmapPlotter',
    'BenchmarkComparer',
    'BaselineBenchmarks',
    'TradingStrategyBenchmarks'
]
