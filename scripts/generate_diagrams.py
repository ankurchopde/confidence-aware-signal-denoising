#!/usr/bin/env python3
"""
Generate publication-quality architecture and data-flow diagrams.
Outputs both vector SVG and high-resolution PNG (300 DPI) to figures/architecture/.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def create_system_architecture(output_dir: Path):
    fig, ax = plt.subplots(figsize=(14, 7), dpi=300)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis("off")

    # Title
    ax.text(7, 6.5, "Confidence-Aware AI Sensor Signal Denoising: System Architecture",
            fontsize=16, fontweight="bold", ha="center", va="center", color="#2c3e50")

    # Colors
    c_input = "#ecf0f1"
    c_prep = "#d5dbdb"
    c_cnn = "#3498db"
    c_head_reg = "#2ecc71"
    c_head_cls = "#e67e22"
    c_recon = "#9b59b6"
    c_eval = "#1abc9c"

    # Box Helper
    def draw_box(x, y, w, h, title, subtitle, bg_color, text_color="black", border_color="#7f8c8d"):
        box = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.1,rounding_size=0.15",
            facecolor=bg_color,
            edgecolor=border_color,
            linewidth=1.8,
            zorder=3
        )
        ax.add_patch(box)
        ax.text(x + w/2, y + h*0.62, title, fontsize=11, fontweight="bold", ha="center", va="center", color=text_color, zorder=4)
        ax.text(x + w/2, y + h*0.28, subtitle, fontsize=8.5, ha="center", va="center", color=text_color, zorder=4)

    # Draw Nodes
    draw_box(0.5, 3.5, 2.0, 1.2, "Sensor Input", "x(t) [Noisy]", "#e8f8f5", "#16a085", "#16a085")
    draw_box(3.0, 3.5, 2.2, 1.2, "Preprocessing", "Standardize + Frame\nWindow=256, Hop=128/1", "#eaeded", "#2c3e50")
    draw_box(5.8, 3.2, 2.6, 1.8, "1D-CNN Backbone", "Conv1D(32) → MaxPool\nConv1D(64) → MaxPool\nDense(128, ReLU)", "#ebf5fb", "#2980b9", "#2980b9")
    
    # Dual Heads
    draw_box(9.1, 4.3, 2.4, 1.3, "Denoising Head", "Dense(256, Linear)\nWindow Reconstruction", "#eafaf1", "#27ae60", "#27ae60")
    draw_box(9.1, 2.3, 2.4, 1.3, "Confidence Head", "Dense(64) → Dense(1, Sigmoid)\nQuality Score [0, 1]", "#fef5e7", "#d35400", "#d35400")

    # Reconstruction & Adaptation
    draw_box(12.0, 4.3, 1.6, 1.3, "Signal OLA", "Overlap-Add\nReconstruction", "#f4ecf7", "#8e44ad", "#8e44ad")
    draw_box(12.0, 2.3, 1.6, 1.3, "Adaptive Tuning", "η = η₀(0.5 + conf)\nOnline Update", "#fcf3cf", "#b7950b", "#b7950b")

    # Arrows
    arrow_kw = dict(arrowstyle="-|>", mutation_scale=15, lw=2, color="#34495e", zorder=2)
    ax.annotate("", xy=(3.0, 4.1), xytext=(2.5, 4.1), arrowprops=arrow_kw)
    ax.annotate("", xy=(5.8, 4.1), xytext=(5.2, 4.1), arrowprops=arrow_kw)
    
    # Split to dual heads
    ax.annotate("", xy=(9.1, 4.95), xytext=(8.4, 4.4), arrowprops=arrow_kw)
    ax.annotate("", xy=(9.1, 2.95), xytext=(8.4, 3.8), arrowprops=arrow_kw)

    # To OLA and Adaptation
    ax.annotate("", xy=(12.0, 4.95), xytext=(11.5, 4.95), arrowprops=arrow_kw)
    ax.annotate("", xy=(12.0, 2.95), xytext=(11.5, 2.95), arrowprops=arrow_kw)

    # Feedback loop arrow from Adaptation back to 1D-CNN
    ax.annotate("", xy=(7.1, 3.2), xytext=(12.8, 2.3),
                arrowprops=dict(arrowstyle="-|>", mutation_scale=12, lw=1.6, color="#e67e22",
                                connectionstyle="arc3,rad=-0.4", linestyle="--"))
    ax.text(10.0, 1.0, "Confidence-Guided Weight Updates (η scaled by confidence)",
            fontsize=9.5, color="#d35400", fontstyle="italic", ha="center")

    fig.savefig(output_dir / "system_architecture.png", bbox_inches="tight")
    fig.savefig(output_dir / "system_architecture.svg", bbox_inches="tight")
    plt.close(fig)
    print("Generated system_architecture.png and .svg")


def create_model_architecture(output_dir: Path):
    fig, ax = plt.subplots(figsize=(15, 6), dpi=300)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 6)
    ax.axis("off")

    ax.text(7.5, 5.5, "Detailed 1D-CNN Multi-Task Neural Network Architecture (580,481 Parameters)",
            fontsize=15, fontweight="bold", ha="center", color="#2c3e50")

    layers_info = [
        ("Input Window", "(256, 1)", "Raw Window Input", "#ecf0f1", "#7f8c8d"),
        ("Conv1D (k=9)", "(256, 32)", "32 Filters, Same, ReLU", "#d4e6f1", "#2980b9"),
        ("MaxPool1D", "(128, 32)", "Pool Size=2, Stride=2", "#e8f8f5", "#16a085"),
        ("Conv1D (k=7)", "(128, 64)", "64 Filters, Same, ReLU", "#d4e6f1", "#2980b9"),
        ("MaxPool1D", "(64, 64)", "Pool Size=2, Stride=2", "#e8f8f5", "#16a085"),
        ("Flatten", "(4096,)", "Vector Flattening", "#fcf3cf", "#b7950b"),
        ("Dense-128", "(128,)", "Shared Latent Space, ReLU", "#f5eef8", "#8e44ad"),
    ]

    for i, (name, shape, desc, bg, border) in enumerate(layers_info):
        x = 0.5 + i * 1.5
        y = 2.4
        box = patches.FancyBboxPatch((x, y), 1.25, 2.0, boxstyle="round,pad=0.08,rounding_size=0.1",
                                     facecolor=bg, edgecolor=border, lw=1.8, zorder=3)
        ax.add_patch(box)
        ax.text(x + 0.625, y + 1.5, name, fontsize=9.5, fontweight="bold", ha="center", va="center")
        ax.text(x + 0.625, y + 1.0, shape, fontsize=8.5, fontweight="bold", color="#2c3e50", ha="center", va="center")
        ax.text(x + 0.625, y + 0.45, desc, fontsize=7.0, color="#566573", ha="center", va="center", wrap=True)

        if i < len(layers_info) - 1:
            ax.annotate("", xy=(x + 1.5, y + 1.0), xytext=(x + 1.25, y + 1.0),
                        arrowprops=dict(arrowstyle="-|>", mutation_scale=12, lw=1.6, color="#34495e", zorder=2))

    # Dual Output Heads
    # Head 1: Denoising Regression Head
    x_h = 11.6
    box_reg = patches.FancyBboxPatch((x_h, 3.4), 2.8, 1.4, boxstyle="round,pad=0.08,rounding_size=0.1",
                                    facecolor="#eafaf1", edgecolor="#27ae60", lw=2, zorder=3)
    ax.add_patch(box_reg)
    ax.text(x_h + 1.4, 4.3, "Regression Head: reg_out", fontsize=10, fontweight="bold", color="#1e8449", ha="center")
    ax.text(x_h + 1.4, 3.85, "Dense(256, Linear) | Target: Clean Window", fontsize=8, color="#27ae60", ha="center")
    ax.text(x_h + 1.4, 3.55, "Loss: MSE (Loss Weight: 1.0)", fontsize=7.5, color="#145a32", ha="center")

    # Head 2: Confidence Classification Head
    box_cls = patches.FancyBboxPatch((x_h, 1.4), 2.8, 1.6, boxstyle="round,pad=0.08,rounding_size=0.1",
                                    facecolor="#fef5e7", edgecolor="#d35400", lw=2, zorder=3)
    ax.add_patch(box_cls)
    ax.text(x_h + 1.4, 2.55, "Confidence Head: cls_out", fontsize=10, fontweight="bold", color="#ba4a00", ha="center")
    ax.text(x_h + 1.4, 2.15, "Dense(64, ReLU) → Dense(1, Sigmoid)", fontsize=8, color="#d35400", ha="center")
    ax.text(x_h + 1.4, 1.75, "Loss: BCE (Loss Weight: 0.2)", fontsize=7.5, color="#7e3204", ha="center")
    ax.text(x_h + 1.4, 1.5, "Metric: Accuracy (Train: 96.9%, Val: 97.1%)", fontsize=7.0, fontstyle="italic", color="#7e3204", ha="center")

    # Arrows to heads
    ax.annotate("", xy=(x_h, 4.1), xytext=(10.75, 3.4),
                arrowprops=dict(arrowstyle="-|>", mutation_scale=12, lw=1.6, color="#27ae60"))
    ax.annotate("", xy=(x_h, 2.2), xytext=(10.75, 3.4),
                arrowprops=dict(arrowstyle="-|>", mutation_scale=12, lw=1.6, color="#d35400"))

    fig.savefig(output_dir / "model_architecture.png", bbox_inches="tight")
    fig.savefig(output_dir / "model_architecture.svg", bbox_inches="tight")
    plt.close(fig)
    print("Generated model_architecture.png and .svg")


def create_data_flow(output_dir: Path):
    fig, ax = plt.subplots(figsize=(14, 6), dpi=300)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis("off")

    ax.text(7, 5.5, "End-to-End Data Flow: Ingestion, Multi-Task Inference, and Reconstruction",
            fontsize=15, fontweight="bold", ha="center", color="#2c3e50")

    nodes = [
        (0.6, 2.4, 1.8, 1.6, "Sensor Data Source", "Real CSVs / Kaggle IoT\nSynthetic Multi-Tone\nSquare / Saw / AM / Chaos", "#f4f6f7", "#7f8c8d"),
        (2.9, 2.4, 1.8, 1.6, "Windowing & Norm", "Window Length = 256\nStandardization [μ=0, σ=1]\nPeak Scale: [-1, 1]", "#e8f6f3", "#1abc9c"),
        (5.2, 2.4, 2.0, 1.6, "Inference Engine", "Batch / Sliding Hop\nTensorFlow / TFLite\nCPU / Edge TPU / Pi", "#ebf5fb", "#3498db"),
        (7.7, 3.4, 2.0, 1.3, "Denoised Output", "Reconstructed Windows\n(N_win, 256)", "#eafaf1", "#2ecc71"),
        (7.7, 1.5, 2.0, 1.3, "Signal Confidence", "Quality Probability\nScore ∈ [0, 1]", "#fef9e7", "#f39c12"),
        (10.2, 3.4, 1.6, 1.3, "Overlap-Add", "OLA Window Fusion\nRescaling Factor", "#f5eef8", "#9b59b6"),
        (10.2, 1.5, 1.6, 1.3, "Gating / Alarm", "Noise Rejection\nAdaptive Tuning", "#fdedec", "#e74c3c"),
        (12.3, 2.4, 1.4, 1.6, "Clean Signal\nx_est(t)", "Evaluated vs Clean\nSNR: +12 to +16 dB\nMSE Reduction", "#d5f5e3", "#27ae60"),
    ]

    for (x, y, w, h, title, sub, bg, border) in nodes:
        box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.1",
                                     facecolor=bg, edgecolor=border, lw=1.8, zorder=3)
        ax.add_patch(box)
        ax.text(x + w/2, y + h*0.68, title, fontsize=9.5, fontweight="bold", ha="center")
        ax.text(x + w/2, y + h*0.28, sub, fontsize=7.5, color="#34495e", ha="center")

    arrow_kw = dict(arrowstyle="-|>", mutation_scale=12, lw=1.6, color="#34495e", zorder=2)
    ax.annotate("", xy=(2.9, 3.2), xytext=(2.4, 3.2), arrowprops=arrow_kw)
    ax.annotate("", xy=(5.2, 3.2), xytext=(4.7, 3.2), arrowprops=arrow_kw)
    ax.annotate("", xy=(7.7, 4.05), xytext=(7.2, 3.5), arrowprops=arrow_kw)
    ax.annotate("", xy=(7.7, 2.15), xytext=(7.2, 2.9), arrowprops=arrow_kw)
    ax.annotate("", xy=(10.2, 4.05), xytext=(9.7, 4.05), arrowprops=arrow_kw)
    ax.annotate("", xy=(10.2, 2.15), xytext=(9.7, 2.15), arrowprops=arrow_kw)
    ax.annotate("", xy=(12.3, 3.5), xytext=(11.8, 4.05), arrowprops=arrow_kw)
    ax.annotate("", xy=(12.3, 2.9), xytext=(11.8, 2.15), arrowprops=arrow_kw)

    fig.savefig(output_dir / "data_flow.png", bbox_inches="tight")
    fig.savefig(output_dir / "data_flow.svg", bbox_inches="tight")
    plt.close(fig)
    print("Generated data_flow.png and .svg")


def create_methodology_pipeline(output_dir: Path):
    fig, ax = plt.subplots(figsize=(14, 5.5), dpi=300)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5.5)
    ax.axis("off")

    ax.text(7, 4.9, "Research Methodology & Experimental Pipeline",
            fontsize=15, fontweight="bold", ha="center", color="#2c3e50")

    stages = [
        (0.5, 1.8, 2.2, 2.2, "Phase 1: Dataset Prep",
         "• Diverse sources: IoT, Chirp, ECG\n• Windowing: 256 samples, Hop=128\n• 6,000 paired windows (signal/noise)\n• Train/Test Split: 88% / 12%",
         "#ebf5fb", "#2980b9"),
        (3.2, 1.8, 2.3, 2.2, "Phase 2: Offline Training",
         "• Dual-Head 1D-CNN (580k params)\n• Multi-task loss: MSE + 0.2*BCE\n• Adam Optimizer (lr=1e-3, 25 epochs)\n• Val Acc: 97.1%, MSE: ~0.08",
         "#eafaf1", "#27ae60"),
        (6.0, 1.8, 2.4, 2.2, "Phase 3: Online Adapt",
         "• Simulated streaming windows\n• Adaptive LR: η = η₀*(0.5 + conf)\n• Dynamic weight updates\n• Safe against noise bursts",
         "#fef9e7", "#f39c12"),
        (8.9, 1.8, 2.2, 2.2, "Phase 4: Evaluation",
         "• Baselines: LMS & NLMS filters\n• 5 signal topologies, 4 noise types\n• Complex non-linear combinations\n• SNR & MSE metric verification",
         "#f5eef8", "#8e44ad"),
        (11.6, 1.8, 2.0, 2.2, "Phase 5: Edge Deploy",
         "• Float32 TFLite (2.3 MB)\n• Dynamic Range Quantization\n• Sub-millisecond latency\n• Raspberry Pi / MCU ready",
         "#e8f8f5", "#16a085"),
    ]

    for (x, y, w, h, title, sub, bg, border) in stages:
        box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1,rounding_size=0.12",
                                     facecolor=bg, edgecolor=border, lw=2.0, zorder=3)
        ax.add_patch(box)
        ax.text(x + w/2, y + h*0.82, title, fontsize=10, fontweight="bold", ha="center", color=border)
        ax.text(x + w/2, y + h*0.38, sub, fontsize=8.0, color="#2c3e50", ha="center", va="center")

    for i in range(len(stages) - 1):
        x_start = stages[i][0] + stages[i][2]
        x_end = stages[i+1][0]
        y_mid = 2.9
        ax.annotate("", xy=(x_end, y_mid), xytext=(x_start, y_mid),
                    arrowprops=dict(arrowstyle="-|>", mutation_scale=14, lw=2, color="#34495e", zorder=2))

    fig.savefig(output_dir / "pipeline_diagram.png", bbox_inches="tight")
    fig.savefig(output_dir / "pipeline_diagram.svg", bbox_inches="tight")
    plt.close(fig)
    print("Generated pipeline_diagram.png and .svg")


def main():
    out_dir = Path("figures/architecture")
    out_dir.mkdir(parents=True, exist_ok=True)
    create_system_architecture(out_dir)
    create_model_architecture(out_dir)
    create_data_flow(out_dir)
    create_methodology_pipeline(out_dir)


if __name__ == "__main__":
    main()
