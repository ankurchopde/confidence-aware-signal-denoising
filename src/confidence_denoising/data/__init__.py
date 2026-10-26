"""Data processing, signal generation, and windowing module."""

from confidence_denoising.data.synthetic import (
    generate_linear_sine_mix,
    generate_square_wave,
    generate_sawtooth_wave,
    generate_am_modulated,
    generate_cubic_nonlinear,
    generate_am_nonlinear,
    generate_chaotic_nonlinear,
    generate_iot_chirp_ecg_suite,
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
from confidence_denoising.data.loader import (
    load_first_numeric_csv,
    load_sensor_directory,
    prepare_train_test_split,
)

__all__ = [
    "generate_linear_sine_mix",
    "generate_square_wave",
    "generate_sawtooth_wave",
    "generate_am_modulated",
    "generate_cubic_nonlinear",
    "generate_am_nonlinear",
    "generate_chaotic_nonlinear",
    "generate_iot_chirp_ecg_suite",
    "add_gaussian_noise",
    "add_impulse_noise",
    "add_sine_interference",
    "add_brown_noise",
    "add_composite_nonlinear_noise",
    "standardize",
    "extract_windows",
    "normalize_windows",
    "overlap_add_reconstruct",
    "center_sample_reconstruct",
    "build_window_dataset",
    "load_first_numeric_csv",
    "load_sensor_directory",
    "prepare_train_test_split",
]
