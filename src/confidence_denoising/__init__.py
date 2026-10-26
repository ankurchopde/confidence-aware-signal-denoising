"""
Confidence-Aware AI for Sensor Signal Denoising.

A deep learning and adaptive signal processing framework designed for
noise suppression, signal-quality confidence estimation, and edge deployment.
"""

__version__ = "1.0.0"
__author__ = "Ankur Chopde, Parth Agalave, Saiprasad Kawthe, Archana Ratnaparkhi, Sagar Janokar"

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
)
from confidence_denoising.filtering.lms import lms_filter
from confidence_denoising.filtering.nlms import nlms_filter
from confidence_denoising.evaluation.metrics import (
    compute_snr_db,
    compute_mse,
    compute_rmse,
    compute_mae,
    evaluate_performance,
)

__all__ = [
    "__version__",
    "generate_linear_sine_mix",
    "generate_square_wave",
    "generate_sawtooth_wave",
    "generate_am_modulated",
    "generate_cubic_nonlinear",
    "generate_am_nonlinear",
    "generate_chaotic_nonlinear",
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
    "lms_filter",
    "nlms_filter",
    "compute_snr_db",
    "compute_mse",
    "compute_rmse",
    "compute_mae",
    "evaluate_performance",
]
