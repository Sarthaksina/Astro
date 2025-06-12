"""
Phase Analysis Module for the Cosmic Market Oracle.

This module implements phase-aware correlation metrics and circular statistics
for analyzing relationships between cyclical variables, particularly useful for
astrological data analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional
import matplotlib.pyplot as plt
from scipy import stats
# import logging # Removed
from ..utils.logger import get_logger # Changed path for src

# Configure logging
# logging.basicConfig( # Removed
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
logger = get_logger(__name__) # Changed


def convert_to_radians(angles: np.ndarray, units: str = 'degrees') -> np.ndarray:
    """
    Convert angles to radians.
    
    Args:
        angles: Array of angles
        units: Units of input angles ('degrees' or 'cycles')
        
    Returns:
        Array of angles in radians
    """
    if units == 'degrees':
        return np.radians(angles)
    elif units == 'cycles':
        return angles * 2 * np.pi
    elif units == 'radians':
        return angles
    else:
        raise ValueError(f"Unknown angle units: {units}")


def convert_from_radians(angles: np.ndarray, units: str = 'degrees') -> np.ndarray:
    """
    Convert angles from radians to specified units.
    
    Args:
        angles: Array of angles in radians
        units: Target units ('degrees' or 'cycles')
        
    Returns:
        Array of angles in specified units
    """
    if units == 'degrees':
        return np.degrees(angles)
    elif units == 'cycles':
        return angles / (2 * np.pi)
    elif units == 'radians':
        return angles
    else:
        raise ValueError(f"Unknown angle units: {units}")


def mean_circular(angles: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
    """
    Calculate the circular mean of angles.
    
    Args:
        angles: Array of angles in radians
        weights: Optional weights for each angle
        
    Returns:
        Circular mean in radians
    """
    if weights is None:
        weights = np.ones_like(angles)
    
    sin_sum = np.sum(weights * np.sin(angles))
    cos_sum = np.sum(weights * np.cos(angles))
    
    return np.arctan2(sin_sum, cos_sum)


def std_circular(angles: np.ndarray) -> float:
    """
    Calculate the circular standard deviation of angles.
    
    Args:
        angles: Array of angles in radians
        
    Returns:
        Circular standard deviation
    """
    sin_mean = np.mean(np.sin(angles))
    cos_mean = np.mean(np.cos(angles))
    
    r = np.sqrt(sin_mean**2 + cos_mean**2)
    
    return np.sqrt(-2 * np.log(r))


def circular_correlation(alpha: np.ndarray, beta: np.ndarray) -> float:
    """
    Calculate the circular correlation coefficient between two sets of angles.
    
    Args:
        alpha: First set of angles in radians
        beta: Second set of angles in radians
        
    Returns:
        Circular correlation coefficient (-1 to 1)
    """
    # Center the angles by subtracting their circular means
    alpha_mean = mean_circular(alpha)
    beta_mean = mean_circular(beta)
    
    alpha_centered = alpha - alpha_mean
    beta_centered = beta - beta_mean
    
    # Calculate the correlation
    num = np.sum(np.sin(alpha_centered) * np.sin(beta_centered))
    den = np.sqrt(np.sum(np.sin(alpha_centered)**2) * np.sum(np.sin(beta_centered)**2))
    
    if den == 0:
        return 0.0
    
    return num / den


def phase_coupling(alpha: np.ndarray, beta: np.ndarray, n: int = 1, m: int = 1) -> float:
    """
    Calculate the phase coupling (n:m phase synchronization) between two sets of angles.
    
    Args:
        alpha: First set of angles in radians
        beta: Second set of angles in radians
        n: First harmonic coefficient
        m: Second harmonic coefficient
        
    Returns:
        Phase coupling strength (0 to 1)
    """
    # Calculate the phase differences
    phase_diff = (n * alpha - m * beta) % (2 * np.pi)
    
    # Calculate the mean vector length
    sin_mean = np.mean(np.sin(phase_diff))
    cos_mean = np.mean(np.cos(phase_diff))
    
    r = np.sqrt(sin_mean**2 + cos_mean**2)
    
    return r


def phase_difference_histogram(alpha: np.ndarray, beta: np.ndarray, n: int = 1, m: int = 1, 
                              bins: int = 36, density: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the histogram of phase differences between two sets of angles.
    
    Args:
        alpha: First set of angles in radians
        beta: Second set of angles in radians
        n: First harmonic coefficient
        m: Second harmonic coefficient
        bins: Number of histogram bins
        density: Whether to normalize the histogram
        
    Returns:
        Tuple of (bin_centers, histogram_values)
    """
    # Calculate the phase differences
    phase_diff = (n * alpha - m * beta) % (2 * np.pi)
    
    # Calculate histogram
    hist, bin_edges = np.histogram(phase_diff, bins=bins, range=(0, 2*np.pi), density=density)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    return bin_centers, hist


def phase_locking_value(alpha: np.ndarray, beta: np.ndarray, n: int = 1, m: int = 1) -> float:
    """
    Calculate the phase locking value (PLV) between two sets of angles.
    
    Args:
        alpha: First set of angles in radians
        beta: Second set of angles in radians
        n: First harmonic coefficient
        m: Second harmonic coefficient
        
    Returns:
        Phase locking value (0 to 1)
    """
    # Calculate the phase differences
    phase_diff = (n * alpha - m * beta) % (2 * np.pi)
    
    # Calculate the PLV
    plv = np.abs(np.mean(np.exp(1j * phase_diff)))
    
    return plv


def phase_lag_index(alpha: np.ndarray, beta: np.ndarray, n: int = 1, m: int = 1) -> float:
    """
    Calculate the phase lag index (PLI) between two sets of angles.
    
    Args:
        alpha: First set of angles in radians
        beta: Second set of angles in radians
        n: First harmonic coefficient
        m: Second harmonic coefficient
        
    Returns:
        Phase lag index (-1 to 1)
    """
    # Calculate the phase differences
    phase_diff = (n * alpha - m * beta) % (2 * np.pi)
    
    # Adjust to -pi to pi range
    phase_diff = (phase_diff + np.pi) % (2 * np.pi) - np.pi
    
    # Calculate the PLI
    pli = np.abs(np.mean(np.sign(phase_diff)))
    
    return pli


def weighted_phase_lag_index(alpha: np.ndarray, beta: np.ndarray, n: int = 1, m: int = 1) -> float:
    """
    Calculate the weighted phase lag index (wPLI) between two sets of angles.
    
    Args:
        alpha: First set of angles in radians
        beta: Second set of angles in radians
        n: First harmonic coefficient
        m: Second harmonic coefficient
        
    Returns:
        Weighted phase lag index (-1 to 1)
    """
    # Calculate the phase differences
    phase_diff = (n * alpha - m * beta) % (2 * np.pi)
    
    # Adjust to -pi to pi range
    phase_diff = (phase_diff + np.pi) % (2 * np.pi) - np.pi
    
    # Calculate the wPLI
    imag_coh = np.sin(phase_diff)
    wpli = np.abs(np.mean(imag_coh)) / np.mean(np.abs(imag_coh))
    
    return wpli


def phase_synchronization_index(alpha: np.ndarray, beta: np.ndarray, n: int = 1, m: int = 1) -> float:
    """
    Calculate the phase synchronization index between two sets of angles.
    
    Args:
        alpha: First set of angles in radians
        beta: Second set of angles in radians
        n: First harmonic coefficient
        m: Second harmonic coefficient
        
    Returns:
        Phase synchronization index (0 to 1)
    """
    # Calculate the phase differences
    phase_diff = (n * alpha - m * beta) % (2 * np.pi)
    
    # Calculate the Shannon entropy of the phase difference distribution
    bins = min(len(phase_diff) // 5, 36)  # Ensure enough samples per bin
    hist, _ = np.histogram(phase_diff, bins=bins, range=(0, 2*np.pi), density=True)
    hist = hist[hist > 0]  # Remove zeros to avoid log(0)
    
    entropy = -np.sum(hist * np.log(hist))
    max_entropy = np.log(bins)  # Maximum entropy for uniform distribution
    
    # Calculate the phase synchronization index
    psi = 1 - entropy / max_entropy
    
    return psi


def kuramoto_order_parameter(phases: np.ndarray) -> Tuple[float, float]:
    """
    Calculate the Kuramoto order parameter for a set of phases.
    
    Args:
        phases: Array of phases in radians
        
    Returns:
        Tuple of (order_parameter, mean_phase)
    """
    # Calculate the complex order parameter
    z = np.mean(np.exp(1j * phases))
    
    # Extract magnitude and phase
    r = np.abs(z)
    psi = np.angle(z)
    
    return r, psi


def rayleigh_test(angles: np.ndarray) -> Tuple[float, float]:
    """
    Perform Rayleigh test for non-uniformity of circular data.
    
    Args:
        angles: Array of angles in radians
        
    Returns:
        Tuple of (test_statistic, p_value)
    """
    n = len(angles)
    
    # Calculate the mean resultant length
    r_bar = np.abs(np.mean(np.exp(1j * angles)))
    
    # Calculate the test statistic
    z = n * r_bar**2
    
    # Calculate the p-value
    p_value = np.exp(-z)
    if n > 50:
        # Apply correction for large sample sizes
        p_value = np.exp(-z) * (1 + (2*z - z**2) / (4*n) - (24*z - 132*z**2 + 76*z**3 - 9*z**4) / (288*n**2))
    
    return z, p_value


def watson_williams_test(samples: List[np.ndarray]) -> Tuple[float, float]:
    """
    Perform Watson-Williams test for homogeneity of means between multiple samples.
    
    Args:
        samples: List of arrays of angles in radians
        
    Returns:
        Tuple of (F_statistic, p_value)
    """
    k = len(samples)  # Number of samples
    n = [len(s) for s in samples]  # Sample sizes
    N = sum(n)  # Total sample size
    
    # Calculate mean resultant length for each sample
    R = [np.abs(np.sum(np.exp(1j * s))) for s in samples]
    
    # Calculate overall mean resultant length
    R_total = np.abs(np.sum([np.sum(np.exp(1j * s)) for s in samples]))
    
    # Calculate the test statistic
    F = (N - k) * (sum(R) - R_total) / ((k - 1) * (N - sum(R)))
    
    # Calculate the p-value
    df1 = k - 1
    df2 = N - k
    p_value = 1 - stats.f.cdf(F, df1, df2)
    
    return F, p_value


def plot_phase_distribution(angles: np.ndarray, bins: int = 36, 
                           title: str = "Phase Distribution",
                           figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Plot the distribution of phases on a circular histogram (rose plot).
    
    Args:
        angles: Array of angles in radians
        bins: Number of bins
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='polar')
    
    # Calculate histogram
    hist, bin_edges = np.histogram(angles, bins=bins, range=(0, 2*np.pi))
    width = 2 * np.pi / bins
    
    # Plot bars
    bars = ax.bar(bin_edges[:-1], hist, width=width, alpha=0.8)
    
    # Set the direction of the zero angle
    ax.set_theta_zero_location("N")
    
    # Set the direction of increasing angles
    ax.set_theta_direction(-1)
    
    # Add mean direction
    mean_angle = mean_circular(angles)
    ax.arrow(mean_angle, 0, 0, max(hist) * 0.8, alpha=0.8, width=0.05,
             edgecolor='black', facecolor='red', lw=2, zorder=5)
    
    # Add title
    ax.set_title(title)
    
    return fig


def plot_phase_coupling(alpha: np.ndarray, beta: np.ndarray, n: int = 1, m: int = 1,
                       bins: int = 36, title: str = None,
                       figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Plot the phase coupling between two sets of angles.
    
    Args:
        alpha: First set of angles in radians
        beta: Second set of angles in radians
        n: First harmonic coefficient
        m: Second harmonic coefficient
        bins: Number of bins for histogram
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Calculate phase differences
    phase_diff = (n * alpha - m * beta) % (2 * np.pi)
    
    # Calculate phase coupling metrics
    plv = phase_locking_value(alpha, beta, n, m)
    psi = phase_synchronization_index(alpha, beta, n, m)
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    
    # Create polar histogram subplot
    ax1 = fig.add_subplot(121, projection='polar')
    
    # Calculate histogram
    hist, bin_edges = np.histogram(phase_diff, bins=bins, range=(0, 2*np.pi))
    width = 2 * np.pi / bins
    
    # Plot bars
    bars = ax1.bar(bin_edges[:-1], hist, width=width, alpha=0.8)
    
    # Set the direction of the zero angle
    ax1.set_theta_zero_location("N")
    
    # Set the direction of increasing angles
    ax1.set_theta_direction(-1)
    
    # Add mean direction
    mean_angle = mean_circular(phase_diff)
    ax1.arrow(mean_angle, 0, 0, max(hist) * 0.8, alpha=0.8, width=0.05,
             edgecolor='black', facecolor='red', lw=2, zorder=5)
    
    # Add title
    if title:
        ax1.set_title(f"{title}\nPhase Difference Distribution")
    else:
        ax1.set_title(f"{n}:{m} Phase Difference Distribution")
    
    # Create scatter plot subplot
    ax2 = fig.add_subplot(122)
    
    # Plot points on unit circle
    theta = np.linspace(0, 2*np.pi, 100)
    ax2.plot(np.cos(theta), np.sin(theta), 'k-', alpha=0.3)
    
    # Plot phase differences as points on unit circle
    ax2.scatter(np.cos(phase_diff), np.sin(phase_diff), alpha=0.5)
    
    # Plot mean vector
    mean_vector = np.mean(np.exp(1j * phase_diff))
    ax2.arrow(0, 0, np.real(mean_vector), np.imag(mean_vector), 
             head_width=0.05, head_length=0.1, fc='red', ec='red')
    
    # Add metrics as text
    ax2.text(0.05, 0.95, f"PLV: {plv:.3f}\nPSI: {psi:.3f}", 
             transform=ax2.transAxes, verticalalignment='top')
    
    # Set equal aspect ratio
    ax2.set_aspect('equal')
    
    # Set limits
    ax2.set_xlim(-1.1, 1.1)
    ax2.set_ylim(-1.1, 1.1)
    
    # Add grid
    ax2.grid(True)
    
    # Add labels
    ax2.set_xlabel("cos(φ)")
    ax2.set_ylabel("sin(φ)")
    ax2.set_title("Phase Differences on Unit Circle")
    
    # Adjust layout
    plt.tight_layout()
    
    return fig


def phase_difference_matrix(phases: Dict[str, np.ndarray]) -> pd.DataFrame:
    """
    Calculate the phase difference matrix for multiple phase variables.
    
    Args:
        phases: Dictionary of phase variables (name -> phases in radians)
        
    Returns:
        DataFrame with phase differences
    """
    names = list(phases.keys())
    n = len(names)
    
    # Initialize matrix
    matrix = np.zeros((n, n))
    
    # Calculate phase differences
    for i in range(n):
        for j in range(n):
            if i != j:
                matrix[i, j] = phase_locking_value(phases[names[i]], phases[names[j]])
    
    # Convert to DataFrame
    df = pd.DataFrame(matrix, index=names, columns=names)
    
    return df


def plot_phase_difference_matrix(matrix: pd.DataFrame, 
                               title: str = "Phase Locking Value Matrix",
                               figsize: Tuple[int, int] = (10, 8),
                               cmap: str = 'viridis',
                               annot: bool = True) -> plt.Figure:
    """
    Plot the phase difference matrix.
    
    Args:
        matrix: Phase difference matrix
        title: Plot title
        figsize: Figure size
        cmap: Colormap
        annot: Whether to annotate cells
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot matrix
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1)
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Phase Locking Value", rotation=-90, va="bottom")
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_xticklabels(matrix.columns)
    ax.set_yticklabels(matrix.index)
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Annotate cells
    if annot:
        for i in range(len(matrix.index)):
            for j in range(len(matrix.columns)):
                ax.text(j, i, f"{matrix.iloc[i, j]:.2f}",
                       ha="center", va="center", 
                       color="white" if matrix.iloc[i, j] > 0.5 else "black")
    
    # Add title
    ax.set_title(title)
    
    # Adjust layout
    fig.tight_layout()
    
    return fig


# Example usage
if __name__ == "__main__":
    # Generate sample data with phase coupling
    np.random.seed(42)
    
    # Create time index
    days = 365
    t = np.arange(days)
    
    # Create two phase variables with n:m coupling (2:1)
    phase1 = (2 * np.pi * t / 30) % (2 * np.pi)  # 30-day cycle
    phase2 = (np.pi * t / 30 + 0.5 * np.random.randn(days)) % (2 * np.pi)  # 60-day cycle with noise
    
    # Calculate circular correlation
    corr = circular_correlation(phase1, phase2)
    print(f"Circular correlation: {corr:.4f}")
    
    # Calculate phase coupling metrics
    plv = phase_locking_value(phase1, phase2, n=2, m=1)
    pli = phase_lag_index(phase1, phase2, n=2, m=1)
    psi = phase_synchronization_index(phase1, phase2, n=2, m=1)
    
    print(f"Phase Locking Value (2:1): {plv:.4f}")
    print(f"Phase Lag Index (2:1): {pli:.4f}")
    print(f"Phase Synchronization Index (2:1): {psi:.4f}")
    
    # Perform Rayleigh test
    z, p = rayleigh_test(phase1)
    print(f"Rayleigh test: z={z:.4f}, p={p:.4f}")
    
    # Plot phase distribution
    plot_phase_distribution(phase1, title="Phase Distribution of 30-day Cycle")
    
    # Plot phase coupling
    plot_phase_coupling(phase1, phase2, n=2, m=1, title="2:1 Phase Coupling")
    
    # Create phase difference matrix
    phases = {
        "30-day": phase1,
        "60-day": phase2,
        "Random": np.random.uniform(0, 2*np.pi, days)
    }
    
    matrix = phase_difference_matrix(phases)
    plot_phase_difference_matrix(matrix)
    
    plt.show()
