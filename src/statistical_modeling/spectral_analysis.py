"""
Spectral Analysis Module for the Cosmic Market Oracle.

This module implements frequency domain analysis and spectral coherence techniques
for detecting cyclical correlations in time series data, particularly useful for
astrological and financial data analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional
import matplotlib.pyplot as plt
from scipy import signal, stats
from scipy.fft import fft, ifft, fftfreq
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def compute_periodogram(x: np.ndarray, fs: float = 1.0, window: str = 'hann',
                      scaling: str = 'density', detrend: str = 'constant') -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the periodogram of a time series.
    
    Args:
        x: Time series data
        fs: Sampling frequency
        window: Window function to apply
        scaling: Scaling of the periodogram ('density' or 'spectrum')
        detrend: Detrending method ('constant', 'linear', or False)
        
    Returns:
        Tuple of (frequencies, power)
    """
    # Compute periodogram
    frequencies, power = signal.periodogram(
        x, fs=fs, window=window, scaling=scaling, detrend=detrend
    )
    
    return frequencies, power


def compute_welch(x: np.ndarray, fs: float = 1.0, nperseg: int = None,
                window: str = 'hann', scaling: str = 'density',
                detrend: str = 'constant') -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the Welch's periodogram of a time series.
    
    Args:
        x: Time series data
        fs: Sampling frequency
        nperseg: Length of each segment
        window: Window function to apply
        scaling: Scaling of the periodogram ('density' or 'spectrum')
        detrend: Detrending method ('constant', 'linear', or False)
        
    Returns:
        Tuple of (frequencies, power)
    """
    # Set default segment length if not provided
    if nperseg is None:
        nperseg = min(256, len(x))
    
    # Compute Welch's periodogram
    frequencies, power = signal.welch(
        x, fs=fs, nperseg=nperseg, window=window, 
        scaling=scaling, detrend=detrend
    )
    
    return frequencies, power


def compute_lombscargle(t: np.ndarray, x: np.ndarray, 
                      frequencies: Optional[np.ndarray] = None,
                      normalization: str = 'standard') -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the Lomb-Scargle periodogram for unevenly sampled data.
    
    Args:
        t: Time points
        x: Signal values
        frequencies: Frequencies to evaluate (if None, automatically determined)
        normalization: Normalization method ('standard', 'model', or 'log')
        
    Returns:
        Tuple of (frequencies, power)
    """
    # Set default frequencies if not provided
    if frequencies is None:
        # Determine reasonable frequency range
        min_freq = 1 / (t[-1] - t[0])  # Lowest frequency (full time span)
        max_freq = 1 / (2 * np.min(np.diff(np.sort(t))))  # Nyquist-like frequency
        frequencies = np.linspace(min_freq, max_freq, 1000)
    
    # Compute Lomb-Scargle periodogram
    power = signal.lombscargle(t, x, frequencies, normalization=normalization)
    
    return frequencies, power


def find_peaks(frequencies: np.ndarray, power: np.ndarray, 
             height: Optional[float] = None, 
             threshold: Optional[float] = None,
             distance: Optional[int] = None,
             prominence: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Find peaks in a periodogram.
    
    Args:
        frequencies: Frequency array
        power: Power spectrum
        height: Required height of peaks
        threshold: Required threshold of peaks
        distance: Required minimum distance between peaks
        prominence: Required prominence of peaks
        
    Returns:
        Tuple of (peak_frequencies, peak_powers, peak_properties)
    """
    # Find peaks
    peaks, properties = signal.find_peaks(
        power, height=height, threshold=threshold,
        distance=distance, prominence=prominence
    )
    
    # Extract peak frequencies and powers
    peak_frequencies = frequencies[peaks]
    peak_powers = power[peaks]
    
    return peak_frequencies, peak_powers, properties


def compute_cross_spectrum(x: np.ndarray, y: np.ndarray, fs: float = 1.0,
                         nperseg: int = None, window: str = 'hann',
                         scaling: str = 'density', detrend: str = 'constant') -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the cross-spectrum between two time series.
    
    Args:
        x: First time series
        y: Second time series
        fs: Sampling frequency
        nperseg: Length of each segment
        window: Window function to apply
        scaling: Scaling of the spectrum ('density' or 'spectrum')
        detrend: Detrending method ('constant', 'linear', or False)
        
    Returns:
        Tuple of (frequencies, cross_spectrum)
    """
    # Set default segment length if not provided
    if nperseg is None:
        nperseg = min(256, len(x), len(y))
    
    # Compute cross-spectrum
    frequencies, cross_spectrum = signal.csd(
        x, y, fs=fs, nperseg=nperseg, window=window,
        scaling=scaling, detrend=detrend
    )
    
    return frequencies, cross_spectrum


def compute_coherence(x: np.ndarray, y: np.ndarray, fs: float = 1.0,
                    nperseg: int = None, window: str = 'hann',
                    detrend: str = 'constant') -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the magnitude squared coherence between two time series.
    
    Args:
        x: First time series
        y: Second time series
        fs: Sampling frequency
        nperseg: Length of each segment
        window: Window function to apply
        detrend: Detrending method ('constant', 'linear', or False)
        
    Returns:
        Tuple of (frequencies, coherence)
    """
    # Set default segment length if not provided
    if nperseg is None:
        nperseg = min(256, len(x), len(y))
    
    # Compute coherence
    frequencies, coherence = signal.coherence(
        x, y, fs=fs, nperseg=nperseg, window=window, detrend=detrend
    )
    
    return frequencies, coherence


def compute_phase_coherence(x: np.ndarray, y: np.ndarray, fs: float = 1.0,
                          nperseg: int = None, window: str = 'hann',
                          detrend: str = 'constant') -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the phase coherence between two time series.
    
    Args:
        x: First time series
        y: Second time series
        fs: Sampling frequency
        nperseg: Length of each segment
        window: Window function to apply
        detrend: Detrending method ('constant', 'linear', or False)
        
    Returns:
        Tuple of (frequencies, phase_coherence)
    """
    # Set default segment length if not provided
    if nperseg is None:
        nperseg = min(256, len(x), len(y))
    
    # Compute cross-spectrum
    f, Pxy = signal.csd(
        x, y, fs=fs, nperseg=nperseg, window=window, detrend=detrend
    )
    
    # Compute auto-spectra
    _, Pxx = signal.csd(
        x, x, fs=fs, nperseg=nperseg, window=window, detrend=detrend
    )
    _, Pyy = signal.csd(
        y, y, fs=fs, nperseg=nperseg, window=window, detrend=detrend
    )
    
    # Compute phase coherence
    phase_coherence = np.abs(Pxy) / np.sqrt(Pxx * Pyy)
    
    return f, phase_coherence


def compute_wavelet_transform(x: np.ndarray, scales: np.ndarray, 
                            wavelet: str = 'morlet',
                            sampling_period: float = 1.0) -> np.ndarray:
    """
    Compute the continuous wavelet transform of a time series.
    
    Args:
        x: Time series data
        scales: Scales to use for the wavelet transform
        wavelet: Wavelet to use ('morlet', 'paul', 'dog')
        sampling_period: Sampling period of the time series
        
    Returns:
        Wavelet transform coefficients (complex array)
    """
    try:
        import pywt
    except ImportError:
        logger.error("PyWavelets (pywt) is required for wavelet transform.")
        raise
    
    # Compute continuous wavelet transform
    coef, _ = pywt.cwt(x, scales, wavelet, sampling_period)
    
    return coef


def compute_wavelet_coherence(x: np.ndarray, y: np.ndarray, 
                            scales: np.ndarray,
                            sampling_period: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the wavelet coherence between two time series.
    
    Args:
        x: First time series
        y: Second time series
        scales: Scales to use for the wavelet transform
        sampling_period: Sampling period of the time series
        
    Returns:
        Tuple of (wavelet coherence, phase difference)
    """
    try:
        import pywt
    except ImportError:
        logger.error("PyWavelets (pywt) is required for wavelet coherence.")
        raise
    
    # Compute wavelet transforms
    coef_x, _ = pywt.cwt(x, scales, 'morlet', sampling_period)
    coef_y, _ = pywt.cwt(y, scales, 'morlet', sampling_period)
    
    # Compute cross-wavelet transform
    cross_wavelet = coef_x * np.conj(coef_y)
    
    # Compute wavelet coherence
    smooth_x = np.abs(coef_x)**2
    smooth_y = np.abs(coef_y)**2
    
    # Simple smoothing (average over neighboring points)
    def smooth(z):
        z_smooth = np.zeros_like(z)
        for i in range(z.shape[0]):
            for j in range(z.shape[1]):
                i_min = max(0, i-1)
                i_max = min(z.shape[0], i+2)
                j_min = max(0, j-1)
                j_max = min(z.shape[1], j+2)
                z_smooth[i, j] = np.mean(z[i_min:i_max, j_min:j_max])
        return z_smooth
    
    smooth_cross = smooth(np.abs(cross_wavelet))
    smooth_x = smooth(smooth_x)
    smooth_y = smooth(smooth_y)
    
    # Compute coherence
    coherence = smooth_cross / np.sqrt(smooth_x * smooth_y)
    
    # Compute phase difference
    phase_diff = np.angle(cross_wavelet)
    
    return coherence, phase_diff


def compute_hilbert_transform(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the Hilbert transform of a time series.
    
    Args:
        x: Time series data
        
    Returns:
        Tuple of (analytic_signal, amplitude_envelope, instantaneous_phase)
    """
    # Compute analytic signal (signal + i*hilbert(signal))
    analytic_signal = signal.hilbert(x)
    
    # Compute amplitude envelope
    amplitude_envelope = np.abs(analytic_signal)
    
    # Compute instantaneous phase
    instantaneous_phase = np.angle(analytic_signal)
    
    return analytic_signal, amplitude_envelope, instantaneous_phase


def compute_instantaneous_frequency(phase: np.ndarray, fs: float = 1.0) -> np.ndarray:
    """
    Compute the instantaneous frequency from instantaneous phase.
    
    Args:
        phase: Instantaneous phase
        fs: Sampling frequency
        
    Returns:
        Instantaneous frequency
    """
    # Unwrap phase to avoid discontinuities
    unwrapped_phase = np.unwrap(phase)
    
    # Compute instantaneous frequency (derivative of phase)
    inst_freq = np.diff(unwrapped_phase) / (2.0 * np.pi) * fs
    
    # Pad to match original length
    inst_freq = np.pad(inst_freq, (0, 1), 'edge')
    
    return inst_freq


def compute_empirical_mode_decomposition(x: np.ndarray) -> List[np.ndarray]:
    """
    Compute the Empirical Mode Decomposition (EMD) of a time series.
    
    Args:
        x: Time series data
        
    Returns:
        List of Intrinsic Mode Functions (IMFs)
    """
    try:
        from PyEMD import EMD
    except ImportError:
        logger.error("PyEMD is required for Empirical Mode Decomposition.")
        raise
    
    # Initialize EMD
    emd = EMD()
    
    # Compute IMFs
    imfs = emd(x)
    
    return imfs


def compute_hilbert_huang_transform(x: np.ndarray, fs: float = 1.0) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Compute the Hilbert-Huang Transform of a time series.
    
    Args:
        x: Time series data
        fs: Sampling frequency
        
    Returns:
        Tuple of (imfs, instantaneous_amplitudes, instantaneous_frequencies)
    """
    # Compute EMD
    imfs = compute_empirical_mode_decomposition(x)
    
    # Initialize lists for results
    instantaneous_amplitudes = []
    instantaneous_frequencies = []
    
    # Compute Hilbert transform for each IMF
    for imf in imfs:
        _, amplitude, phase = compute_hilbert_transform(imf)
        frequency = compute_instantaneous_frequency(phase, fs)
        
        instantaneous_amplitudes.append(amplitude)
        instantaneous_frequencies.append(frequency)
    
    return imfs, instantaneous_amplitudes, instantaneous_frequencies


def plot_periodogram(frequencies: np.ndarray, power: np.ndarray, 
                   peak_frequencies: Optional[np.ndarray] = None,
                   peak_powers: Optional[np.ndarray] = None,
                   title: str = "Periodogram",
                   figsize: Tuple[int, int] = (10, 6),
                   xscale: str = 'linear',
                   yscale: str = 'log') -> plt.Figure:
    """
    Plot a periodogram with optional peak highlighting.
    
    Args:
        frequencies: Frequency array
        power: Power spectrum
        peak_frequencies: Frequencies of detected peaks
        peak_powers: Powers of detected peaks
        title: Plot title
        figsize: Figure size
        xscale: X-axis scale ('linear' or 'log')
        yscale: Y-axis scale ('linear' or 'log')
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot periodogram
    ax.plot(frequencies, power, 'b-')
    
    # Plot peaks if provided
    if peak_frequencies is not None and peak_powers is not None:
        ax.plot(peak_frequencies, peak_powers, 'ro')
        
        # Annotate peaks
        for i, (freq, pwr) in enumerate(zip(peak_frequencies, peak_powers)):
            period = 1 / freq if freq > 0 else float('inf')
            ax.annotate(f"Period: {period:.2f}", 
                       xy=(freq, pwr),
                       xytext=(5, 5),
                       textcoords='offset points')
    
    # Set scales
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    
    # Add labels
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Power')
    ax.set_title(title)
    ax.grid(True)
    
    return fig


def plot_coherence(frequencies: np.ndarray, coherence: np.ndarray,
                 title: str = "Coherence",
                 figsize: Tuple[int, int] = (10, 6),
                 xscale: str = 'linear') -> plt.Figure:
    """
    Plot coherence between two time series.
    
    Args:
        frequencies: Frequency array
        coherence: Coherence values
        title: Plot title
        figsize: Figure size
        xscale: X-axis scale ('linear' or 'log')
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot coherence
    ax.plot(frequencies, coherence, 'b-')
    
    # Set scale
    ax.set_xscale(xscale)
    
    # Add confidence level
    # For MSC, the 95% confidence level is approximately 1 - 0.05^(1/(n_segments-1))
    # where n_segments is typically nperseg / (nperseg/2)
    n_segments = 8  # Approximate default for Welch's method
    conf_level = 1 - 0.05**(1/(n_segments-1))
    ax.axhline(y=conf_level, color='r', linestyle='--', label=f'95% Confidence ({conf_level:.2f})')
    
    # Add labels
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Coherence')
    ax.set_title(title)
    ax.grid(True)
    ax.legend()
    
    return fig


def plot_wavelet_coherence(coherence: np.ndarray, phase_diff: np.ndarray,
                         scales: np.ndarray, times: np.ndarray,
                         title: str = "Wavelet Coherence",
                         figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Plot wavelet coherence between two time series.
    
    Args:
        coherence: Wavelet coherence
        phase_diff: Phase difference
        scales: Scales used for wavelet transform
        times: Time points
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create a time-scale mesh for plotting
    T, S = np.meshgrid(times, scales)
    
    # Plot coherence as a filled contour
    contourf = ax.contourf(T, S, coherence, 100, cmap='jet', vmin=0, vmax=1)
    
    # Add colorbar
    cbar = fig.colorbar(contourf, ax=ax)
    cbar.set_label('Coherence')
    
    # Plot phase arrows
    # Subsample for clarity
    arrow_spacing = max(1, len(times) // 20)
    phase_arrows = np.exp(1j * phase_diff)
    
    for i in range(0, coherence.shape[0], 2):
        for j in range(0, coherence.shape[1], arrow_spacing):
            if coherence[i, j] > 0.5:  # Only plot arrows where coherence is significant
                ax.arrow(times[j], scales[i],
                        np.real(phase_arrows[i, j]) * arrow_spacing,
                        np.imag(phase_arrows[i, j]) * 0.5,
                        head_width=0.2, head_length=0.2,
                        fc='k', ec='k', alpha=0.5)
    
    # Set y-axis to log scale and invert (smaller scales at top)
    ax.set_yscale('log')
    ax.invert_yaxis()
    
    # Add labels
    ax.set_xlabel('Time')
    ax.set_ylabel('Scale')
    ax.set_title(title)
    
    return fig


def plot_hilbert_huang_spectrum(imfs: List[np.ndarray], 
                              inst_freqs: List[np.ndarray],
                              inst_amps: List[np.ndarray],
                              times: np.ndarray,
                              title: str = "Hilbert-Huang Spectrum",
                              figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Plot the Hilbert-Huang spectrum.
    
    Args:
        imfs: List of Intrinsic Mode Functions
        inst_freqs: List of instantaneous frequencies for each IMF
        inst_amps: List of instantaneous amplitudes for each IMF
        times: Time points
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    
    # Plot IMFs
    for i, imf in enumerate(imfs):
        ax1.plot(times, imf + i*2, label=f'IMF {i+1}')
    
    # Add labels
    ax1.set_xlabel('Time')
    ax1.set_ylabel('IMF (offset for clarity)')
    ax1.set_title('Intrinsic Mode Functions')
    ax1.grid(True)
    
    # Create a time-frequency-amplitude plot
    # Initialize a 2D histogram-like array
    n_freq_bins = 100
    max_freq = max([np.max(freq) for freq in inst_freqs if len(freq) > 0])
    freq_bins = np.linspace(0, max_freq, n_freq_bins)
    
    hht = np.zeros((n_freq_bins, len(times)))
    
    for i, (freq, amp) in enumerate(zip(inst_freqs, inst_amps)):
        if len(freq) == 0 or len(amp) == 0:
            continue
            
        for j, (f, a) in enumerate(zip(freq, amp)):
            if j < len(times):
                # Find the frequency bin
                bin_idx = int(f / max_freq * (n_freq_bins - 1))
                if 0 <= bin_idx < n_freq_bins:
                    hht[bin_idx, j] += a
    
    # Plot the Hilbert-Huang spectrum
    im = ax2.imshow(hht, aspect='auto', origin='lower', 
                   extent=[times[0], times[-1], 0, max_freq],
                   cmap='jet', interpolation='bilinear')
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax2)
    cbar.set_label('Amplitude')
    
    # Add labels
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Hilbert-Huang Spectrum')
    
    # Adjust layout
    fig.tight_layout()
    fig.suptitle(title, y=1.02)
    
    return fig


# Example usage
if __name__ == "__main__":
    # Generate sample data with multiple frequency components
    np.random.seed(42)
    
    # Create time array
    days = 365
    fs = 1.0  # 1 sample per day
    t = np.arange(days) / fs
    
    # Create signal with multiple frequency components
    f1 = 1/30  # 30-day cycle
    f2 = 1/90  # 90-day cycle
    f3 = 1/365  # Annual cycle
    
    x = (
        1.0 * np.sin(2 * np.pi * f1 * t) +
        0.5 * np.sin(2 * np.pi * f2 * t) +
        2.0 * np.sin(2 * np.pi * f3 * t) +
        0.2 * np.random.randn(days)
    )
    
    # Create a second signal with some shared and some different components
    y = (
        0.8 * np.sin(2 * np.pi * f1 * t + np.pi/4) +  # Same 30-day cycle but phase-shifted
        0.7 * np.sin(2 * np.pi * f2 * t) +            # Same 90-day cycle
        1.5 * np.sin(2 * np.pi * (1/180) * t) +       # Different 180-day cycle
        0.2 * np.random.randn(days)
    )
    
    # Compute periodogram
    frequencies, power = compute_welch(x, fs=fs, nperseg=128)
    
    # Find peaks
    peak_frequencies, peak_powers, _ = find_peaks(
        frequencies, power, prominence=0.1
    )
    
    # Compute coherence
    freq_coh, coherence = compute_coherence(x, y, fs=fs, nperseg=128)
    
    # Plot results
    plot_periodogram(
        frequencies, power, peak_frequencies, peak_powers,
        title="Welch's Periodogram of Signal with Multiple Cycles"
    )
    
    plot_coherence(
        freq_coh, coherence,
        title="Coherence between Signals with Shared Cycles"
    )
    
    # Compute Hilbert transform
    analytic_signal, amplitude, phase = compute_hilbert_transform(x)
    inst_freq = compute_instantaneous_frequency(phase, fs)
    
    # Try to compute EMD if PyEMD is available
    try:
        imfs = compute_empirical_mode_decomposition(x)
        imfs_amps = []
        imfs_freqs = []
        
        for imf in imfs:
            _, amp, ph = compute_hilbert_transform(imf)
            freq = compute_instantaneous_frequency(ph, fs)
            imfs_amps.append(amp)
            imfs_freqs.append(freq)
        
        plot_hilbert_huang_spectrum(imfs, imfs_freqs, imfs_amps, t)
    except ImportError:
        logger.warning("PyEMD not available, skipping Hilbert-Huang Transform")
    
    # Try to compute wavelet coherence if PyWavelets is available
    try:
        import pywt
        scales = np.arange(1, 128)
        coherence, phase_diff = compute_wavelet_coherence(x, y, scales, sampling_period=1.0)
        plot_wavelet_coherence(coherence, phase_diff, scales, t)
    except ImportError:
        logger.warning("PyWavelets not available, skipping wavelet coherence")
    
    plt.show()
