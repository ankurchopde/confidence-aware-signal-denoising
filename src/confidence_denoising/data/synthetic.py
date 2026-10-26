"""
Synthetic signal generators and noise injection routines.

Supports linear periodic signals, modulated waveforms, non-linear signals,
and multiple real-world noise models (Gaussian, Impulse, Brown, and Interference).
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np
from scipy import signal


def generate_linear_sine_mix(
    duration: float = 1.0,
    fs: int = 1000,
    f1: float = 5.0,
    a1: float = 0.7,
    f2: float = 12.0,
    a2: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a two-tone linear sinusoidal mixture: a1*sin(2*pi*f1*t) + a2*sin(2*pi*f2*t).

    Args:
        duration: Signal duration in seconds.
        fs: Sampling rate in Hz.
        f1: First harmonic frequency (Hz).
        a1: First harmonic amplitude.
        f2: Second harmonic frequency (Hz).
        a2: Second harmonic amplitude.

    Returns:
        Tuple of (time_array, clean_signal).
    """
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    clean = a1 * np.sin(2 * np.pi * f1 * t) + a2 * np.sin(2 * np.pi * f2 * t)
    return t, clean


def generate_square_wave(
    duration: float = 1.0,
    fs: int = 1000,
    freq: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a periodic square waveform."""
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    clean = signal.square(2 * np.pi * freq * t)
    return t, clean


def generate_sawtooth_wave(
    duration: float = 1.0,
    fs: int = 1000,
    freq: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a periodic sawtooth waveform."""
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    clean = signal.sawtooth(2 * np.pi * freq * t)
    return t, clean


def generate_am_modulated(
    duration: float = 1.0,
    fs: int = 1000,
    carrier_freq: float = 10.0,
    mod_freq: float = 2.0,
    mod_index: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate an amplitude-modulated (AM) waveform:
    (1 + mod_index * sin(2*pi*mod_freq*t)) * sin(2*pi*carrier_freq*t).
    """
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    clean = (1.0 + mod_index * np.sin(2 * np.pi * mod_freq * t)) * np.sin(2 * np.pi * carrier_freq * t)
    return t, clean


def generate_cubic_nonlinear(
    duration: float = 1.0,
    fs: int = 1000,
    freq: float = 6.0,
    alpha: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a non-linearly distorted sinusoidal signal:
    sin(2*pi*f*t) + alpha * sin(2*pi*f*t)^3.
    """
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    base = np.sin(2 * np.pi * freq * t)
    clean = base + alpha * (base ** 3)
    return t, clean


def generate_am_nonlinear(
    duration: float = 1.0,
    fs: int = 1000,
    carrier_freq: float = 10.0,
    mod_freq: float = 2.0,
    mod_index: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a non-linear AM signal:
    (1 + mod_index * sin(2*pi*mod_freq*t)) * sin(2*pi*carrier_freq*t)^3.
    """
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    carrier = np.sin(2 * np.pi * carrier_freq * t)
    clean = (1.0 + mod_index * np.sin(2 * np.pi * mod_freq * t)) * (carrier ** 3)
    return t, clean


def generate_chaotic_nonlinear(
    duration: float = 1.0,
    fs: int = 1000,
    f1: float = 7.0,
    f2: float = 13.0,
    alpha: float = 0.4,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a chaotic/saturating non-linear signal:
    tanh(sin(2*pi*f1*t) + alpha * sin(2*pi*f2*t)).
    """
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    arg = np.sin(2 * np.pi * f1 * t) + alpha * np.sin(2 * np.pi * f2 * t)
    clean = np.tanh(arg)
    return t, clean


def generate_iot_chirp_ecg_suite(
    t_max: float = 10.0,
    num_samples: int = 50000,
) -> Dict[str, np.ndarray]:
    """
    Generate training suite including slow IoT sensor drifts, wideband chirps,
    and synthetic pseudo-ECG signals.
    """
    t = np.linspace(0, t_max, num_samples)
    iot_drift = 0.5 * np.sin(2 * np.pi * 0.2 * t) + 0.2 * np.sin(2 * np.pi * 1.2 * t)
    chirp_sig = signal.chirp(t, f0=1, f1=100, t1=t_max)
    ecg_sig = signal.square(2 * np.pi * 1.7 * t) * np.exp(-((t % 1) - 0.2) ** 2 / 0.01)
    am_slow = np.sin(2 * np.pi * 50 * t) * (1 + 0.2 * np.sin(2 * np.pi * 0.5 * t))

    return {
        "iot_drift": iot_drift,
        "chirp": chirp_sig,
        "ecg": ecg_sig,
        "am_slow": am_slow,
    }


def add_gaussian_noise(
    clean: np.ndarray,
    sigma: float = 0.4,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Add additive white Gaussian noise (AWGN)."""
    rng = np.random.RandomState(seed)
    return clean + sigma * rng.randn(len(clean))


def add_impulse_noise(
    clean: np.ndarray,
    ratio: float = 0.02,
    scale_range: Tuple[float, float] = (0.5, 1.5),
    seed: Optional[int] = None,
) -> np.ndarray:
    """Add sparse high-amplitude impulse spikes (shot noise)."""
    rng = np.random.RandomState(seed)
    n = len(clean)
    noisy = clean.copy()
    spike_count = max(1, int(n * ratio))
    idx = rng.choice(n, size=spike_count, replace=False)
    signs = rng.choice([-1.0, 1.0], size=spike_count)
    magnitudes = rng.uniform(scale_range[0], scale_range[1], size=spike_count)
    noisy[idx] += signs * magnitudes
    return noisy


def add_sine_interference(
    clean: np.ndarray,
    freq: float = 50.0,
    amplitude: float = 0.3,
    fs: int = 1000,
) -> np.ndarray:
    """Add periodic narrow-band harmonic interference (e.g. 50/60 Hz powerline hum)."""
    n = len(clean)
    t = np.linspace(0, n / fs, n, endpoint=False)
    interference = amplitude * np.sin(2 * np.pi * freq * t)
    return clean + interference


def add_brown_noise(
    clean: np.ndarray,
    amplitude: float = 0.3,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Add brown (Brownian / random walk) low-frequency drift noise."""
    rng = np.random.RandomState(seed)
    n = len(clean)
    white = rng.randn(n)
    brown = np.cumsum(white)
    max_val = np.max(np.abs(brown)) + 1e-12
    brown = (brown / max_val) * amplitude
    return clean + brown


def add_composite_nonlinear_noise(
    clean: np.ndarray,
    sigma_gauss: float = 0.3,
    sine_amp: float = 0.25,
    sine_freq: float = 50.0,
    impulse_ratio: float = 0.02,
    fs: int = 1000,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Add composite non-linear noise combining AWGN, harmonic 50Hz hum, and impulse spikes.
    """
    rng = np.random.RandomState(seed)
    n = len(clean)
    t = np.linspace(0, n / fs, n, endpoint=False)

    n_gauss = sigma_gauss * rng.randn(n)
    n_sine = sine_amp * np.sin(2 * np.pi * sine_freq * t)

    n_impulse = np.zeros(n)
    spike_count = max(1, int(n * impulse_ratio))
    idx = rng.choice(n, size=spike_count, replace=False)
    n_impulse[idx] = rng.choice([-1.0, 1.0], size=spike_count) * rng.uniform(0.5, 1.0, size=spike_count)

    total_noise = n_gauss + n_sine + n_impulse
    return clean + total_noise
