"""
Sinh biểu đồ chuẩn bài báo khoa học cho Cột trụ 2 (Accuracy, Low-Data, Noise Robustness, Gating Dynamics)
Lưu tại:
  1. reports/figures/pillar2_accuracy_benchmark.png
  2. reports/figures/pillar2_gating_gradcam.png
"""

from __future__ import annotations
import json
import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "src")
from models.gr_quanv import extract_multiscale_qlut, GatedResidualQuanvNet


def plot_main_dashboard(res: dict, out_path: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), dpi=220)
    plt.subplots_adjust(hspace=0.34, wspace=0.25)

    # Panel A: 100% Full OCTMNIST (109,309 images) Fair-Play Model Comparison
    ax = axes[0, 0]
    oct_full = res["task_2_3_main_benchmarks"]["octmnist_full_109k"]["models"]
    model_names = [
        "Classical Baseline",
        "Pure Quanvolution",
        "Naive Residual Quanv",
        "Classical MS-Gated Control",
        "Lean GR-Quanv (1x1)",
        "Proposed GR-Quanv",
    ]
    short_labels = [
        "Classical\nConv2D",
        "Pure\nQuanv (d=1)",
        "Naive\nResidual",
        "Classical\nMS-Control",
        "Lean\nGR-Quanv (1x1)",
        "Proposed\nGR-Quanv",
    ]
    accs = [oct_full[m]["acc_mean"] * 100 for m in model_names]
    acc_stds = [oct_full[m]["acc_std"] * 100 for m in model_names]
    aucs = [oct_full[m]["auc_mean"] * 100 for m in model_names]

    x = np.arange(len(model_names))
    w = 0.36
    b1 = ax.bar(x - w / 2, accs, w, yerr=acc_stds, capsize=4, label="Test Accuracy (%)", color="#1f77b4", alpha=0.9)
    b2 = ax.bar(x + w / 2, aucs, w, label="Test ROC-AUC (%)", color="#ff7f0e", alpha=0.88)
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=8.5, fontweight="bold")
    ax.set_ylim(30, 102)
    ax.set_ylabel("Percentage (%)", fontweight="bold")
    ax.set_title("A. 100% Full OCTMNIST (109,309 imgs) — Fair-Play Linear Head", fontweight="bold", fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="lower right", frameon=True)
    for bar in b1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1.2, f"{h:.1f}%", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    # Panel B: Cross-Dataset Accuracy/AUC Summary (5 Seeds)
    ax = axes[0, 1]
    ds_keys = ["breastmnist", "pneumoniamnist", "octmnist_canonical_3500", "octmnist_full_109k"]
    ds_titles = ["BreastMNIST\n(ROC-AUC %)", "PneumoniaMNIST\n(Accuracy %)", "OCTMNIST 3.5K\n(Accuracy %)", "OCTMNIST 100%\n(Accuracy %)"]
    metrics_pick = ["auc_mean", "acc_mean", "acc_mean", "acc_mean"]

    c_vals, p_vals, g_vals = [], [], []
    for dk, mp in zip(ds_keys, metrics_pick):
        m_dict = res["task_2_3_main_benchmarks"][dk]["models"]
        c_vals.append(m_dict["Classical Baseline"][mp] * 100)
        p_vals.append(m_dict["Pure Quanvolution"][mp] * 100)
        g_vals.append(m_dict["Proposed GR-Quanv"][mp] * 100)

    xd = np.arange(len(ds_keys))
    bw = 0.25
    ax.bar(xd - bw, c_vals, bw, label="Classical Baseline", color="#4c72b0")
    ax.bar(xd, p_vals, bw, label="Pure Quanvolution", color="#dd8452")
    bars_g = ax.bar(xd + bw, g_vals, bw, label="Proposed GR-Quanv", color="#2ca02c")
    ax.set_xticks(xd)
    ax.set_xticklabels(ds_titles, fontsize=9, fontweight="bold")
    ax.set_ylim(30, 102)
    ax.set_ylabel("Score (%)", fontweight="bold")
    ax.set_title("B. Multi-Dataset Fair-Play Benchmark (5 Seeds Mean)", fontweight="bold", fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="lower right", frameon=True)
    for bar in bars_g:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1.0, f"{h:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold", color="#1b611b")

    # Panel C: Regime A — Extreme Low-Data / Few-Shot Learning Curve
    ax = axes[1, 0]
    fs_dict = res["task_2_4_few_shot"]
    fs_keys = list(fs_dict.keys())
    fs_labels = ["0.25%\n(240)", "0.5%\n(480)", "1.0%\n(960)", "5.0%\n(4800)", "10.0%\n(9600)"]
    for m_name, color, marker in [
        ("Classical Baseline", "#1f77b4", "o"),
        ("Pure Quanvolution", "#ff7f0e", "s"),
        ("Lean GR-Quanv (1x1)", "#9467bd", "^"),
        ("Proposed GR-Quanv", "#2ca02c", "D"),
    ]:
        ys = [fs_dict[k][m_name]["acc_mean"] * 100 for k in fs_keys]
        ystd = [fs_dict[k][m_name]["acc_std"] * 100 for k in fs_keys]
        ax.errorbar(range(len(fs_keys)), ys, yerr=ystd, label=m_name, color=color, marker=marker, linewidth=2.2, capsize=4)
    ax.set_xticks(range(len(fs_keys)))
    ax.set_xticklabels(fs_labels, fontweight="bold")
    ax.set_xlabel("Training Data Fraction (Total Images)", fontweight="bold")
    ax.set_ylabel("Test Accuracy (%)", fontweight="bold")
    ax.set_title("C. Tử huyệt 1: Kịch bản Thiếu dữ liệu (Low-Data / Few-Shot OCTMNIST)", fontweight="bold", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower right", frameon=True)

    # Panel D: Regime B — Clinical Speckle Noise Robustness
    ax = axes[1, 1]
    sp_dict = res["task_2_4_noise_robustness"]["speckle"]
    sp_keys = sorted(sp_dict.keys())
    sigmas = [float(k.split("_")[1]) for k in sp_keys]
    for m_name, color, marker in [
        ("Classical Baseline", "#d62728", "o"),
        ("Pure Quanvolution", "#ff7f0e", "s"),
        ("Lean GR-Quanv (1x1)", "#9467bd", "^"),
        ("Proposed GR-Quanv", "#2ca02c", "D"),
    ]:
        ys = [sp_dict[k][m_name]["acc"] * 100 for k in sp_keys]
        ax.plot(sigmas, ys, label=m_name, color=color, marker=marker, linewidth=2.4)
    ax.set_xlabel("Speckle Noise Standard Deviation (σ)", fontweight="bold")
    ax.set_ylabel("Test Accuracy (%)", fontweight="bold")
    ax.set_title("D. Tử huyệt 2: Kháng nhiễu Speckle Thiết bị Y tế (OCTMNIST)", fontweight="bold", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", frameon=True)

    fig.suptitle(
        "CỘT TRỤ 2: KIẾN TRÚC GR-QUANV — ĐỐI CHỨNG CÔNG BẰNG & 3 TỬ HUYỆT LÂM SÀNG (5 SEEDS)",
        fontsize=13, fontweight="bold", y=0.98
    )
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved main dashboard figure to {out_path}")


def plot_gating_dynamics(res: dict, out_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=220)
    gate_dict = res["task_2_4_gating_dynamics"]
    ds_list = ["breastmnist", "pneumoniamnist", "octmnist_canonical_3500", "octmnist_full_109k"]
    ds_names = ["BreastMNIST", "PneumoniaMNIST", "OCTMNIST (3.5K)", "OCTMNIST (100%)"]

    ax = axes[0]
    x = np.arange(len(ds_list))
    w = 0.18
    for ch in range(4):
        vals = [gate_dict[d]["per_channel_mean"][ch] for d in ds_list]
        ax.bar(x + (ch - 1.5) * w, vals, w, label=f"Gate α Channel {ch}")
    ax.axhline(0.5, color="black", linestyle="--", alpha=0.5, label="50/50 Equilibrium")
    ax.set_xticks(x)
    ax.set_xticklabels(ds_names, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Learned Quantum Gate Weight α ∈ [0, 1]", fontweight="bold")
    ax.set_title("A. Phân bố Trọng số Cổng Lượng tử α theo Kênh & Tập dữ liệu", fontweight="bold", fontsize=10.5)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", fontsize=8.5)

    # Panel B: Gaussian Noise Robustness
    ax = axes[1]
    ga_dict = res["task_2_4_noise_robustness"]["gaussian"]
    ga_keys = sorted(ga_dict.keys())
    sigmas = [float(k.split("_")[1]) for k in ga_keys]
    for m_name, color, marker in [
        ("Classical Baseline", "#d62728", "o"),
        ("Pure Quanvolution", "#ff7f0e", "s"),
        ("Lean GR-Quanv (1x1)", "#9467bd", "^"),
        ("Proposed GR-Quanv", "#2ca02c", "D"),
    ]:
        ys = [ga_dict[k][m_name]["auc"] * 100 for k in ga_keys]
        ax.plot(sigmas, ys, label=m_name, color=color, marker=marker, linewidth=2.3)
    ax.set_xlabel("Additive Gaussian Sensor Noise (σ)", fontweight="bold")
    ax.set_ylabel("Test ROC-AUC (%)", fontweight="bold")
    ax.set_title("B. Kháng nhiễu Cảm biến Gaussian (ROC-AUC trên OCTMNIST)", fontweight="bold", fontsize=10.5)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", fontsize=8.5)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved gating dynamics figure to {out_path}")


if __name__ == "__main__":
    with open("reports/pillar2_results.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    plot_main_dashboard(data, "reports/figures/pillar2_accuracy_benchmark.png")
    plot_gating_dynamics(data, "reports/figures/pillar2_gating_gradcam.png")
