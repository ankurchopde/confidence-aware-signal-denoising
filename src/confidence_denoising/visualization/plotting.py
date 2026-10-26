"""
Publication-grade plotting and visualization routines.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np


def plot_4panel_comparison(
    clean: np.ndarray,
    noisy: np.ndarray,
    y_lms: np.ndarray,
    y_nlms: np.ndarray,
    y_model: np.ndarray,
    title: str = "Signal Comparison",
    snr_lms: Optional[float] = None,
    snr_nlms: Optional[float] = None,
    snr_model: Optional[float] = None,
    n_show: int = 500,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Generate a 4-panel vertical comparison plot:
        Panel 1: Input (Clean vs Noisy)
        Panel 2: LMS Filter Output
        Panel 3: NLMS Filter Output
        Panel 4: Proposed Confidence-Aware AI Model Output

    Args:
        clean: Ground truth clean signal.
        noisy: Contaminated input signal.
        y_lms: LMS filter output.
        y_nlms: NLMS filter output.
        y_model: AI model reconstructed output.
        title: Overall plot title.
        snr_lms: Computed SNR for LMS (dB).
        snr_nlms: Computed SNR for NLMS (dB).
        snr_model: Computed SNR for AI Model (dB).
        n_show: Number of samples to display (default 500).
        save_path: Path to save figure (optional).

    Returns:
        Matplotlib Figure object.
    """
    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)

    # 1. Input: Clean vs Noisy
    axes[0].plot(clean[:n_show], "k", lw=2, label="Clean Reference")
    axes[0].plot(noisy[:n_show], "gray", alpha=0.6, label="Noisy Input")
    axes[0].set_title(f"{title} — Input Signal", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Amplitude")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    # 2. LMS Filter
    lms_lbl = f"LMS Output (SNR={snr_lms:.2f} dB)" if snr_lms is not None else "LMS Output"
    axes[1].plot(clean[:n_show], "k", lw=2, alpha=0.5, label="Clean Reference")
    axes[1].plot(y_lms[:n_show], "orange", lw=1.5, label=lms_lbl)
    axes[1].set_ylabel("Amplitude")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    # 3. NLMS Filter
    nlms_lbl = f"NLMS Output (SNR={snr_nlms:.2f} dB)" if snr_nlms is not None else "NLMS Output"
    axes[2].plot(clean[:n_show], "k", lw=2, alpha=0.5, label="Clean Reference")
    axes[2].plot(y_nlms[:n_show], "blue", lw=1.5, label=nlms_lbl)
    axes[2].set_ylabel("Amplitude")
    axes[2].legend(loc="upper right")
    axes[2].grid(True, alpha=0.3)

    # 4. AI Model
    model_lbl = f"AI Model Output (SNR={snr_model:.2f} dB)" if snr_model is not None else "AI Model Output"
    axes[3].plot(clean[:n_show], "k", lw=2, alpha=0.5, label="Clean Reference")
    axes[3].plot(y_model[:n_show], "green", lw=1.8, label=model_lbl)
    axes[3].set_title("Confidence-Aware AI Model Output", fontsize=11, fontweight="bold")
    axes[3].set_ylabel("Amplitude")
    axes[3].set_xlabel("Sample Index")
    axes[3].legend(loc="upper right")
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(p), dpi=300, bbox_inches="tight")

    return fig


def plot_streaming_step(
    seg_clean: np.ndarray,
    seg_noisy: np.ndarray,
    pred_win: np.ndarray,
    start_idx: int,
    confidence: Optional[float] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Plot a single window snapshot during streaming evaluation.
    """
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(seg_clean, "k", lw=2, label="Clean Signal")
    ax.plot(seg_noisy, "gray", alpha=0.6, label="Noisy Window")
    ax.plot(pred_win, "g", lw=2, label="Model Prediction")

    conf_str = f" | Confidence: {confidence:.3f}" if confidence is not None else ""
    ax.set_title(f"Streaming Window at sample offset {start_idx}{conf_str}", fontsize=11)
    ax.set_xlabel("Window Sample Index")
    ax.set_ylabel("Amplitude")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(str(save_path), dpi=300, bbox_inches="tight")
    return fig


def plot_metrics_summary(
    results_by_signal: Dict[str, Dict[str, float]],
    metric_name: str = "snr_db",
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Plot grouped bar chart comparing LMS, NLMS, and Proposed AI Model across signals.
    """
    signals = list(results_by_signal.keys())
    lms_vals = [results_by_signal[s]["lms_" + metric_name] for s in signals]
    nlms_vals = [results_by_signal[s]["nlms_" + metric_name] for s in signals]
    model_vals = [results_by_signal[s]["model_" + metric_name] for s in signals]

    x = np.arange(len(signals))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width, lms_vals, width, label="LMS Filter", color="#f39c12")
    ax.bar(x, nlms_vals, width, label="NLMS Filter", color="#2980b9")
    ax.bar(x + width, model_vals, width, label="Confidence-Aware AI", color="#27ae60")

    ylabel = "Output SNR (dB)" if "snr" in metric_name else "Mean Squared Error (MSE)"
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f"Comparative Benchmark Performance ({ylabel}) across Signal Topologies", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(signals, rotation=15, ha="right", fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(str(save_path), dpi=300, bbox_inches="tight")
    return fig
