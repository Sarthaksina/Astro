#!/usr/bin/env python
# Cosmic Market Oracle - Regime Visualizer Bridge

"""
Bridge module to redirect imports from regime_visualizer to visualization.

This module provides compatibility for code that still imports from the deprecated
regime_visualizer module, redirecting those imports to the consolidated visualization module.
"""

import warnings
from src.evaluation.visualization import RegimeVisualizer as _RegimeVisualizer

# Issue deprecation warning
warnings.warn(
    "The regime_visualizer module is deprecated and will be removed in a future version. "
    "Please use src.evaluation.visualization.RegimeVisualizer instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export the RegimeVisualizer class
RegimeComparisonVisualizer = _RegimeVisualizer
