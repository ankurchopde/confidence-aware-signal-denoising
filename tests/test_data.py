"""
Unit tests for data generation, noise models, windowing, and reconstruction.
"""

import numpy as np
import pytest

from confidence_denoising.data.synthetic import (
    generate_linear_sine_mix,
    generate_square_wave,
    generate_sawtooth_wave,
    generate_am_modulated,
    generate_cubic_nonlinear,
    generate_am_nonlinear,
    generate_chaotic_nonlinear,
    add_gaussian_noise,
    add_impulse_noise,
    add_sine_interference,
    add_brown_noise,
    add_composite_nonlinear_noise,
)
from confidence_denoising.data.windowing import (
    standardize,
    extract_windows,
    normalize_windows,
    overlap_add_reconstruct,
    center_sample_reconstruct,
    build_window_dataset,
)


def test_signal_generators_output_shape():
    fs = 1000
    duration = 1.0
    expected_len = 1000

    generators = [
        generate_linear_sine_mix,
        generate_square_wave,
        generate_sawtooth_wave,
        generate_am_modulated,
        generate_cubic_nonlinear,
        generate_am_nonlinear,
        generate_chaotic_nonlinear,
    ]

    for gen in generators:
        t, sig = gen(duration=duration, fs=fs)
        assert len(t) == expected_len, f"{gen.__name__} time shape mismatch"
        assert len(sig) == expected_len, f"{gen.__name__} signal shape mismatch"
        assert not np.isnan(sig).any(), f"{gen.__name__} produced NaNs"
        assert not np.isinf(sig).any(), f"{gen.__name__} produced Infs"


def test_noise_injection():
    clean = np.ones(500, dtype=np.float64)

    # 1. Gaussian noise
    noisy_gauss = add_gaussian_noise(clean, sigma=0.2, seed=123)
    assert noisy_gauss.shape == (500,)
    assert not np.array_equal(clean, noisy_gauss)

    # 2. Impulse noise
    noisy_impulse = add_impulse_noise(clean, ratio=0.05, seed=123)
    assert noisy_impulse.shape == (500,)
    diff_count = np.sum(clean != noisy_impulse)
    assert diff_count > 0

    # 3. Sine interference
    noisy_sine = add_sine_interference(clean, freq=50.0, amplitude=0.5, fs=1000)
    assert noisy_sine.shape == (500,)

    # 4. Brown noise
    noisy_brown = add_brown_noise(clean, amplitude=0.4, seed=123)
    assert noisy_brown.shape == (500,)

    # 5. Composite noise
    noisy_comp = add_composite_nonlinear_noise(clean, seed=123)
    assert noisy_comp.shape == (500,)


def test_standardize_and_window_normalization():
    sig = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float64)
    std_sig = standardize(sig)
    assert np.isclose(np.mean(std_sig), 0.0, atol=1e-7)
    assert np.isclose(np.std(std_sig), 1.0, atol=1e-7)

    # Window normalization
    windows = np.array([
        [1.0, -2.0, 0.5],
        [0.1, -0.4, 0.2],
    ], dtype=np.float32)
    norm_w = normalize_windows(windows)
    assert norm_w.shape == windows.shape
    assert np.max(np.abs(norm_w[0])) <= 1.0 + 1e-6
    assert np.max(np.abs(norm_w[1])) <= 1.0 + 1e-6


def test_windowing_and_reconstruction_consistency():
    sig = np.sin(np.linspace(0, 10, 1000)).astype(np.float32)
    window_size = 256
    hop_size = 1

    windows = extract_windows(sig, window_size=window_size, hop_size=hop_size)
    assert windows.shape == (1000 - window_size + 1, window_size)

    # Overlap-add reconstruction
    recon_ola = overlap_add_reconstruct(
        windows,
        hop_size=hop_size,
        window_size=window_size,
        target_length=1000,
    )
    assert len(recon_ola) == 1000
    # Overlap add of original windows should recover original signal closely
    np.testing.assert_allclose(recon_ola, sig, atol=1e-5)


def test_build_window_dataset():
    s1 = np.sin(np.linspace(0, 50, 2000))
    s2 = np.cos(np.linspace(0, 50, 2000))

    x_win, y_clean, y_labels = build_window_dataset(
        sources=[s1, s2],
        window_size=256,
        hop_size=128,
        target_windows=50,
        seed=42,
    )
    assert x_win.shape[0] >= 50
    assert x_win.shape[1] == 256
    assert y_clean.shape == x_win.shape
    assert len(y_labels) == x_win.shape[0]
    assert set(np.unique(y_labels)).issubset({0, 1})
