# Cosmic Market Oracle - Portfolio Construction

"""
This module provides portfolio construction algorithms for trading strategies
based on Vedic astrological signals and market data.

It includes:
- Portfolio optimization using modern portfolio theory
- Risk allocation based on astrological strength
- Sector rotation strategies based on planetary positions
- Diversification rules with astrological constraints
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
import scipy.optimize as sco

from src.astro_engine.planetary_positions import PlanetaryCalculator
from src.trading.signal_generator import CombinedSignalGenerator
from src.utils.logger import setup_logger
# Import new constants for sector rotation
from ..astro_engine.constants import (
    PLANET_SECTOR_AFFINITIES, SIGN_SECTOR_AFFINITIES, ZODIAC_SIGN_NAMES,
    SUN, MOON, MERCURY, VENUS, MARS, JUPITER, SATURN, RAHU, KETU
)

from .constants import MAX_PORTFOLIO_WEIGHT, MIN_PORTFOLIO_WEIGHT, RISK_FREE_RATE

# Configure logging
logger = setup_logger("portfolio_construction")


class PortfolioConstructor:
    """Base class for portfolio construction algorithms."""
    
    def __init__(self, name: str = "Base Portfolio Constructor"):
        """
        Initialize the portfolio constructor.
        
        Args:
            name: Constructor name
        """
        self.name = name
        self.calculator = PlanetaryCalculator()
    
    def allocate_portfolio(self, signals: pd.DataFrame, market_data: Dict[str, pd.DataFrame], 
                         total_capital: float, max_positions: int = 10) -> Dict[str, float]:
        """
        Allocate portfolio based on signals and market data.
        
        Args:
            signals: DataFrame with trading signals for multiple assets
            market_data: Dictionary of market data DataFrames for each asset
            total_capital: Total capital to allocate
            max_positions: Maximum number of positions to take
            
        Returns:
            Dictionary with allocation for each asset {symbol: amount}
        """
        # Base implementation: equal allocation to assets with positive signals
        allocation = {}
        
        # Get assets with positive signals
        positive_signals = []
        for symbol in signals.columns:
            if signals[symbol].iloc[-1] > 0:
                positive_signals.append(symbol)
        
        # Limit to max_positions
        if len(positive_signals) > max_positions:
            # Sort by signal strength if available
            if "strength" in signals.index:
                strengths = {symbol: signals[symbol].loc["strength"] for symbol in positive_signals}
                positive_signals = sorted(strengths.keys(), key=lambda x: strengths[x], reverse=True)[:max_positions]
            else:
                positive_signals = positive_signals[:max_positions]
        
        # Equal allocation
        if positive_signals:
            amount_per_asset = total_capital / len(positive_signals)
            for symbol in positive_signals:
                allocation[symbol] = amount_per_asset
        
        return allocation


class MPTPortfolioConstructor(PortfolioConstructor):
    """Portfolio constructor using Modern Portfolio Theory with astrological constraints."""
    
    def __init__(self, name: str = "MPT Portfolio Constructor", 
                risk_aversion: float = 2.0, 
                min_weight: float = MIN_PORTFOLIO_WEIGHT,
                max_weight: float = MAX_PORTFOLIO_WEIGHT):
        """
        Initialize the MPT portfolio constructor.
        
        Args:
            name: Constructor name
            risk_aversion: Risk aversion parameter (higher = more conservative)
            min_weight: Minimum weight for any asset
            max_weight: Maximum weight for any asset
        """
        super().__init__(name)
        self.risk_aversion = risk_aversion
        self.min_weight = min_weight
        self.max_weight = max_weight
    
    def allocate_portfolio(self, signals: pd.DataFrame, market_data: Dict[str, pd.DataFrame], 
                         total_capital: float, max_positions: int = 10) -> Dict[str, float]:
        """
        Allocate portfolio based on signals and market data using MPT.
        
        Args:
            signals: DataFrame with trading signals for multiple assets
            market_data: Dictionary of market data DataFrames for each asset
            total_capital: Total capital to allocate
            max_positions: Maximum number of positions to take
            
        Returns:
            Dictionary with allocation for each asset {symbol: amount}
        """
        # Get assets with positive signals
        positive_signals = []
        for symbol in signals.columns:
            if signals[symbol].iloc[-1] > 0:
                positive_signals.append(symbol)
        
        # Limit to max_positions
        if len(positive_signals) > max_positions:
            # Sort by signal strength if available
            if "strength" in signals.index:
                strengths = {symbol: signals[symbol].loc["strength"] for symbol in positive_signals}
                positive_signals = sorted(strengths.keys(), key=lambda x: strengths[x], reverse=True)[:max_positions]
            else:
                positive_signals = positive_signals[:max_positions]
        
        # If no positive signals, return empty allocation
        if not positive_signals:
            return {}
        
        # Calculate returns and covariance matrix
        returns = {}
        for symbol in positive_signals:
            if "Close" in market_data[symbol].columns:
                returns[symbol] = market_data[symbol]["Close"].pct_change().dropna()
        
        # If we don't have returns for all assets, fall back to equal allocation
        if len(returns) != len(positive_signals):
            logger.warning("Missing return data for some assets, falling back to equal allocation")
            amount_per_asset = total_capital / len(positive_signals)
            return {symbol: amount_per_asset for symbol in positive_signals}
        
        # Create returns DataFrame
        returns_df = pd.DataFrame(returns)
        
        # Calculate expected returns (use signal strength to adjust if available)
        expected_returns = returns_df.mean()
        if "strength" in signals.index:
            for symbol in expected_returns.index:
                strength = signals[symbol].loc["strength"]
                expected_returns[symbol] *= (1 + strength)  # Adjust expected return by signal strength
        
        # Calculate covariance matrix
        cov_matrix = returns_df.cov()
        
        # Optimize portfolio
        weights = self._optimize_portfolio(expected_returns, cov_matrix)
        
        # Allocate capital based on weights
        allocation = {}
        for symbol, weight in weights.items():
            allocation[symbol] = total_capital * weight
        
        return allocation
    
    def _optimize_portfolio(self, expected_returns: pd.Series, cov_matrix: pd.DataFrame) -> Dict[str, float]:
        """
        Optimize portfolio weights using mean-variance optimization.
        
        Args:
            expected_returns: Expected returns for each asset
            cov_matrix: Covariance matrix of returns
            
        Returns:
            Dictionary with optimal weights for each asset
        """
        n_assets = len(expected_returns)
        
        # Define objective function (negative Sharpe ratio)
        def objective(weights):
            portfolio_return = np.sum(expected_returns * weights)
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return -(portfolio_return - RISK_FREE_RATE) / portfolio_volatility
        
        # Define constraints
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # Weights sum to 1
        
        # Define bounds
        bounds = tuple((self.min_weight, self.max_weight) for _ in range(n_assets))
        
        # Initial guess (equal weights)
        initial_weights = np.array([1/n_assets] * n_assets)
        
        # Optimize
        result = sco.minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        
        # Get optimal weights
        optimal_weights = result['x']
        
        # Create dictionary with weights
        weights = {}
        for i, symbol in enumerate(expected_returns.index):
            weights[symbol] = optimal_weights[i]
        
        return weights


class AstrologicalSectorRotation(PortfolioConstructor):
    """Portfolio constructor using sector rotation based on planetary positions."""
    
    def __init__(self, name: str = "Astrological Sector Rotation"):
        """Initialize the astrological sector rotation constructor."""
        super().__init__(name)
        
        # self.planet_sector_affinities and self.zodiac_sector_affinities are now removed.
        # They will be imported from astro_engine.constants.
        
        # Map sectors to ETFs or stocks (example mapping) - this remains local as it's UI/config specific
        self.sector_tickers = {
            "Technology": ["XLK", "QQQ", "MSFT", "AAPL"],
            "Energy": ["XLE", "CVX", "XOM"],
            "Financial Services": ["XLF", "JPM", "BAC"],
            "Healthcare": ["XLV", "JNJ", "PFE"],
            "Consumer Staples": ["XLP", "PG", "KO"],
            "Utilities": ["XLU", "NEE", "DUK"],
            "Real Estate": ["XLRE", "AMT", "SPG"],
            "Communication": ["XLC", "GOOG", "META"],
            "Industrial": ["XLI", "HON", "UNP"],
            "Basic Materials": ["XLB", "LIN", "APD"],
            "Gold": ["GLD", "NEM", "GOLD"],
            "Defense": ["ITA", "LMT", "RTX"],
            "Pharmaceuticals": ["PJP", "JNJ", "PFE"],
            "Alternative Energy": ["TAN", "ICLN", "NEE"],
            "Luxury Goods": ["LVMUY", "PPRUY", "TIF"],
            "Entertainment": ["DIS", "NFLX", "CMCSA"],
            "Transportation": ["IYT", "UNP", "FDX"],
            "International": ["EFA", "EEM", "VXUS"],
            "Speculative": ["ARKK", "BTCC", "COIN"]
        }
    
    def allocate_portfolio(self, signals: pd.DataFrame, market_data: Dict[str, pd.DataFrame], 
                         total_capital: float, max_positions: int = 10,
                         date: datetime = None) -> Dict[str, float]:
        """
        Allocate portfolio based on astrological sector rotation.
        
        Args:
            signals: DataFrame with trading signals for multiple assets
            market_data: Dictionary of market data DataFrames for each asset
            total_capital: Total capital to allocate
            max_positions: Maximum number of positions to take
            date: Date for planetary positions (default: current date)
            
        Returns:
            Dictionary with allocation for each asset {symbol: amount}
        """
        # Use current date if not provided
        if date is None:
            date = datetime.now()
        
        # Get planetary positions
        positions = self.calculator.calculate_positions(date) # Corrected method name
        
        # Determine favored sectors based on planetary positions
        favored_sectors = self._determine_favored_sectors(positions)
        
        # Get tickers for favored sectors
        favored_tickers = self._get_tickers_for_sectors(favored_sectors, max_positions)
        
        # Filter by available market data and signals
        available_tickers = list(market_data.keys())
        favored_tickers = [ticker for ticker in favored_tickers if ticker in available_tickers]
        
        # Adjust allocation based on signal strength if available
        allocation = {}
        if "strength" in signals.index:
            # Calculate total strength
            total_strength = 0
            ticker_strengths = {}
            for ticker in favored_tickers:
                if ticker in signals.columns:
                    strength = max(0, signals[ticker].loc["strength"])
                    ticker_strengths[ticker] = strength
                    total_strength += strength
            
            # Allocate based on relative strength
            if total_strength > 0:
                for ticker, strength in ticker_strengths.items():
                    allocation[ticker] = total_capital * (strength / total_strength)
            else:
                # Equal allocation if no strength information
                amount_per_asset = total_capital / len(favored_tickers)
                allocation = {ticker: amount_per_asset for ticker in favored_tickers}
        else:
            # Equal allocation if no strength information
            amount_per_asset = total_capital / len(favored_tickers)
            allocation = {ticker: amount_per_asset for ticker in favored_tickers}
        
        return allocation
    
    def _determine_favored_sectors(self, positions: Dict) -> List[str]:
        """
        Determine favored sectors based on planetary positions.
        
        Args:
            positions: Dictionary of planetary positions
            
        Returns:
            List of favored sectors
        """
        favored_sectors = {}
        
        # The `positions` dict from `self.calculator.calculate_positions(date)`
        # should have integer planet IDs as keys.
        for planet_id, position_details in positions.items():
            # Skip non-integer keys if any (e.g., 'date' if it's part of the dict)
            if not isinstance(planet_id, int):
                continue

            # Use .get for safer access, providing an empty list if key not found
            planet_affinities = PLANET_SECTOR_AFFINITIES.get(planet_id, [])
            if planet_affinities: # Check if the planet has defined affinities
                longitude = position_details["longitude"]
                # sign_idx will be 0 for Aries, 1 for Taurus, etc.
                sign_idx = int(longitude / 30)
                
                strength = 1.0
                if "dignity" in position_details: # Ensure 'dignity' key exists
                    dignity = position_details["dignity"]
                    if dignity == "exalted":
                        strength = 1.5
                    elif dignity == "debilitated":
                        strength = 0.5
                
                # Add planet's favored sectors
                for sector in planet_affinities:
                    favored_sectors[sector] = favored_sectors.get(sector, 0) + strength
                
                # Add sign's favored sectors
                # SIGN_SECTOR_AFFINITIES uses 0-11 index for Aries-Pisces
                sign_affinities = SIGN_SECTOR_AFFINITIES.get(sign_idx, [])
                if sign_affinities: # Check if the sign has defined affinities
                    for sector in sign_affinities:
                        favored_sectors[sector] = favored_sectors.get(sector, 0) + strength * 0.5  # Sign influence is half
        
        # Sort sectors by strength
        sorted_sectors = sorted(favored_sectors.keys(), key=favored_sectors.get, reverse=True)
        
        return sorted_sectors
    
    def _get_tickers_for_sectors(self, sectors: List[str], max_tickers: int) -> List[str]:
        """
        Get tickers for the specified sectors.
        
        Args:
            sectors: List of sectors
            max_tickers: Maximum number of tickers to return
            
        Returns:
            List of tickers
        """
        tickers = []
        
        # Add tickers for each sector
        for sector in sectors:
            if sector in self.sector_tickers:
                sector_tickers = self.sector_tickers[sector]
                for ticker in sector_tickers:
                    if ticker not in tickers:
                        tickers.append(ticker)
                        if len(tickers) >= max_tickers:
                            return tickers
        
        return tickers


class AstrologicalRiskParity(PortfolioConstructor):
    """Portfolio constructor using risk parity with astrological risk adjustments."""
    
    def __init__(self, name: str = "Astrological Risk Parity"):
        """Initialize the astrological risk parity constructor."""
        super().__init__(name)
    
    def allocate_portfolio(self, signals: pd.DataFrame, market_data: Dict[str, pd.DataFrame], 
                         total_capital: float, max_positions: int = 10,
                         planetary_data: pd.DataFrame = None) -> Dict[str, float]:
        """
        Allocate portfolio based on risk parity with astrological adjustments.
        
        Args:
            signals: DataFrame with trading signals for multiple assets
            market_data: Dictionary of market data DataFrames for each asset
            total_capital: Total capital to allocate
            max_positions: Maximum number of positions to take
            planetary_data: DataFrame with planetary data
            
        Returns:
            Dictionary with allocation for each asset {symbol: amount}
        """
        # Get assets with positive signals
        positive_signals = []
        for symbol in signals.columns:
            if signals[symbol].iloc[-1] > 0:
                positive_signals.append(symbol)
        
        # Limit to max_positions
        if len(positive_signals) > max_positions:
            # Sort by signal strength if available
            if "strength" in signals.index:
                strengths = {symbol: signals[symbol].loc["strength"] for symbol in positive_signals}
                positive_signals = sorted(strengths.keys(), key=lambda x: strengths[x], reverse=True)[:max_positions]
            else:
                positive_signals = positive_signals[:max_positions]
        
        # If no positive signals, return empty allocation
        if not positive_signals:
            return {}
        
        # Calculate volatility for each asset
        volatilities = {}
        for symbol in positive_signals:
            if "Close" in market_data[symbol].columns:
                returns = market_data[symbol]["Close"].pct_change().dropna()
                volatilities[symbol] = returns.std()
        
        # If we don't have volatility for all assets, fall back to equal allocation
        if len(volatilities) != len(positive_signals):
            logger.warning("Missing volatility data for some assets, falling back to equal allocation")
            amount_per_asset = total_capital / len(positive_signals)
            return {symbol: amount_per_asset for symbol in positive_signals}
        
        # Adjust volatilities based on astrological factors if planetary data is provided
        if planetary_data is not None:
            volatilities = self._adjust_volatilities(volatilities, planetary_data)
        
        # Calculate inverse volatility
        inverse_volatility = {symbol: 1 / vol for symbol, vol in volatilities.items()}
        
        # Calculate total inverse volatility
        total_inverse_volatility = sum(inverse_volatility.values())
        
        # Calculate weights
        weights = {symbol: inv_vol / total_inverse_volatility for symbol, inv_vol in inverse_volatility.items()}
        
        # Allocate capital based on weights
        allocation = {symbol: total_capital * weight for symbol, weight in weights.items()}
        
        return allocation
    
    def _adjust_volatilities(self, volatilities: Dict[str, float], 
                           planetary_data: pd.DataFrame) -> Dict[str, float]:
        """
        Adjust volatilities based on astrological factors.
        
        Args:
            volatilities: Dictionary of volatilities for each asset
            planetary_data: DataFrame with planetary data
            
        Returns:
            Dictionary of adjusted volatilities
        """
        adjusted_volatilities = volatilities.copy()
        
        # Get latest planetary data
        latest_data = planetary_data.iloc[-1]
        
        # Check for volatility factors
        if "market_volatility_factor" in latest_data:
            volatility_factor = latest_data["market_volatility_factor"]
            
            # Adjust all volatilities by the factor
            for symbol in adjusted_volatilities:
                adjusted_volatilities[symbol] *= volatility_factor
        
        # Check for specific planetary aspects that increase volatility
        if "high_volatility_configuration" in latest_data and latest_data["high_volatility_configuration"]:
            # Increase volatility for all assets
            for symbol in adjusted_volatilities:
                adjusted_volatilities[symbol] *= 1.2  # 20% increase
        
        return adjusted_volatilities


def create_portfolio_constructor(constructor_type: str = "basic") -> PortfolioConstructor:
    """
    Factory function to create a portfolio constructor.
    
    Args:
        constructor_type: Type of constructor to create
        
    Returns:
        Portfolio constructor instance
    """
    if constructor_type.lower() == "mpt":
        return MPTPortfolioConstructor()
    elif constructor_type.lower() == "sector_rotation":
        return AstrologicalSectorRotation()
    elif constructor_type.lower() == "risk_parity":
        return AstrologicalRiskParity()
    else:
        return PortfolioConstructor()
