"""
Ablation Study: Factor Isolation on Speckle Noise (PneumoniaMNIST, 5 Seeds)
==========================================================================
Isolates the contributions of:
  (1) Classical Control (Raw noisy input)
  (2) Classical Control + Input Phase Damping (Laplacian Filter)
  (3) GR-Quanv v2 (Raw noisy input, no Pre-LUT Phase Damping)
  (4) GR-Quanv v2 + Pre-LUT Phase Damping (Full System)

Evaluates at:
  sigma in [0.00, 0.05, 0.10, 0.15, 0.20] across all 5 seeds (42..46) with 5 noise draws.
Outputs:
  - reports/ablation_isolation_results.json
  - reports/figures/ablation_factor_isolation.png
"""

import os
import sys
import time
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from models.gr_quanv_wide import WideClassicalControl, WideGRQuanvNet
from models.gr_quanv import extract_multiscale_qlut, phase_damping_u8
from run_pillar2_wide import load_dataset_with_cache, SEEDS

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SIGMAS = [0.00, 0.05, 0.10, 0.15, 0.20]
NOISE_DRAWS = [0, 1, 2, 3, 4]


def speckle(imgs_norm, sigma, draw_idx):
    if sigma <= 0.0:
        return (imgs_norm * 255.0 + 0.5).astype(np.uint8)
    rng = np.random.default_rng(2027 + int(sigma * 1000) + draw_idx * 7)
    noise = rng.normal(0.0, sigma, size=imgs_norm.shape).astype(np.float32)
    noisy = imgs_norm + imgs_norm * noise
    return (np.clip(noisy, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def evaluate_batch(model, x_tensor, q_tensor, y_np):
    with torch.no_grad():
        preds = []
        bs = 256
        for i in range(0, len(x_tensor), bs):
            xb = x_tensor[i:i+bs]
            qb = q_tensor[i:i+bs] if q_tensor is not None else None
            logits = model(xb, qb)
            p = torch.softmax(logits, dim=1).cpu().numpy()
            preds.append(p)
        preds = np.concatenate(preds, axis=0)
        acc = float(np.mean(preds.argmax(axis=1) == y_np))
        # Binary F1 score
        y_pred = preds.argmax(axis=1)
        tp = np.sum((y_pred == 1) & (y_np == 1))
        fp = np.sum((y_pred == 1) & (y_np == 0))
        fn = np.sum((y_pred == 0) & (y_np == 1))
        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        f1 = float(2 * prec * rec / (prec + rec + 1e-9))
        return acc * 100.0, f1 * 100.0


def main():
    print("=" * 80)
    print("  ABLATION STUDY: FACTOR ISOLATION ON PNEUMONIAMNIST (5 SEEDS)")
    print("=" * 80)

    ds = "pneumoniamnist"
    D = load_dataset_with_cache(ds, interp4d=True)
    te_f = D["te"].astype(np.float32) / 255.0
    y_test = D["yte"].astype(np.int64)

    ckpt_dir = os.path.join(PROJECT_ROOT, "weights", "pillar2_wide")
    
    # Load 5-seed checkpoints
    models_ctrl = {}
    models_quanv = {}
    for s in SEEDS:
        cp_c = os.path.join(ckpt_dir, f"wide_v2_interp4d_{ds}_full_control_s{s}.pt")
        cp_q = os.path.join(ckpt_dir, f"wide_v2_interp4d_{ds}_full_grquanv_s{s}.pt")
        
        mc = WideClassicalControl(num_classes=2).to(DEV)
        mc.load_state_dict(torch.load(cp_c, map_location=DEV, weights_only=True))
        mc.eval()
        models_ctrl[s] = mc

        mq = WideGRQuanvNet(num_classes=2).to(DEV)
        mq.load_state_dict(torch.load(cp_q, map_location=DEV, weights_only=True))
        mq.eval()
        models_quanv[s] = mq

    conditions = [
        "1_ctrl_raw",          # Classical Control (No damping)
        "2_ctrl_damped",       # Classical Control + Phase Damping
        "3_grquanv_raw",       # GR-Quanv (No Pre-LUT Phase Damping)
        "4_grquanv_damped",    # GR-Quanv + Pre-LUT Phase Damping (Full proposed)
    ]

    results = {cond: {f"{sg:.2f}": [] for sg in SIGMAS} for cond in conditions}

    for sg in SIGMAS:
        draws = NOISE_DRAWS if sg > 0 else [0]
        print(f"\n--- Processing Sigma = {sg:.2f} ({len(draws)} noise draws) ---")

        draw_metrics = {cond: {s: [] for s in SEEDS} for cond in conditions}

        for dr in draws:
            noisy_u8 = speckle(te_f, sg, dr)

            # Raw tensors (normalized to [-1, 1])
            x_raw_t = torch.from_numpy(noisy_u8.astype(np.float32) / 127.5 - 1.0).unsqueeze(1).to(DEV)

            # Damped images (via phase_damping_u8)
            noisy_damped_u8 = phase_damping_u8(noisy_u8, beta=18.0, tau=0.024)
            x_damped_t = torch.from_numpy(noisy_damped_u8.astype(np.float32) / 127.5 - 1.0).unsqueeze(1).to(DEV)

            # Extract Q-LUT features without damping
            q_raw_t, _ = extract_multiscale_qlut(noisy_u8, interp4d=True, phase_damping=False)
            q_raw_t = q_raw_t.to(DEV)

            # Extract Q-LUT features with damping
            q_damped_t, _ = extract_multiscale_qlut(noisy_u8, interp4d=True, phase_damping=True)
            q_damped_t = q_damped_t.to(DEV)

            for s in SEEDS:
                # 1. Classical Control on RAW noisy image
                acc1, f11 = evaluate_batch(models_ctrl[s], x_raw_t, None, y_test)
                draw_metrics["1_ctrl_raw"][s].append((acc1, f11))

                # 2. Classical Control on DAMPED image (Ablation test: Does damping alone save classical?)
                acc2, f12 = evaluate_batch(models_ctrl[s], x_damped_t, None, y_test)
                draw_metrics["2_ctrl_damped"][s].append((acc2, f12))

                # 3. GR-Quanv on RAW Q-LUT features (No Phase Damping)
                acc3, f13 = evaluate_batch(models_quanv[s], x_raw_t, q_raw_t, y_test)
                draw_metrics["3_grquanv_raw"][s].append((acc3, f13))

                # 4. GR-Quanv on DAMPED Q-LUT features (Full system)
                acc4, f14 = evaluate_batch(models_quanv[s], x_damped_t, q_damped_t, y_test)
                draw_metrics["4_grquanv_damped"][s].append((acc4, f14))

        for cond in conditions:
            # Average across draws per seed
            seed_accs = [np.mean([item[0] for item in draw_metrics[cond][s]]) for s in SEEDS]
            seed_f1s = [np.mean([item[1] for item in draw_metrics[cond][s]]) for s in SEEDS]
            results[cond][f"{sg:.2f}"] = {
                "acc_mean": float(np.mean(seed_accs)),
                "acc_std": float(np.std(seed_accs)),
                "f1_mean": float(np.mean(seed_f1s)),
                "f1_std": float(np.std(seed_f1s)),
            }
            res_str = f"  {cond:<20} => ACC: {np.mean(seed_accs):5.2f}+-{np.std(seed_accs):4.2f}% | F1: {np.mean(seed_f1s):5.2f}%"
            print(res_str)

    # Save Results
    out_json = os.path.join(PROJECT_ROOT, "reports", "ablation_isolation_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Saved JSON] {out_json}")

    # Plot Figure
    fig_path = os.path.join(PROJECT_ROOT, "reports", "figures", "ablation_factor_isolation.png")
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=220)

    x_vals = np.array(SIGMAS)

    # Colors & Line styles
    c_map = {
        "1_ctrl_raw": ("#7f8c8d", "s--", "1. Classical Control (Raw Noisy)"),
        "2_ctrl_damped": ("#2980b9", "^-.", "2. Classical Control + Damping Filter"),
        "3_grquanv_raw": ("#e67e22", "v:", "3. GR-Quanv v2 (Raw Q-LUT, No Damping)"),
        "4_grquanv_damped": ("#c0392b", "o-", "4. GR-Quanv v2 + Pre-LUT Damping (Full)"),
    }

    for cond, (color, marker, label) in c_map.items():
        acc_m = [results[cond][f"{sg:.2f}"]["acc_mean"] for sg in SIGMAS]
        f1_m = [results[cond][f"{sg:.2f}"]["f1_mean"] for sg in SIGMAS]
        lw = 2.4 if "4_" in cond else 1.8
        ms = 7 if "4_" in cond else 6
        ax1.plot(x_vals, acc_m, marker, color=color, label=label, lw=lw, markersize=ms)
        ax2.plot(x_vals, f1_m, marker, color=color, label=label, lw=lw, markersize=ms)

    ax1.set_xlabel("Speckle Noise Amplitude (sigma)")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("(A) Accuracy Isolation across Noise Levels", fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.set_xticks(SIGMAS)
    ax1.legend(loc="lower left", fontsize=8.5, framealpha=0.95)

    ax2.set_xlabel("Speckle Noise Amplitude (sigma)")
    ax2.set_ylabel("F1 Score (%)")
    ax2.set_title("(B) F1 Score Isolation across Noise Levels", fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.set_xticks(SIGMAS)
    ax2.legend(loc="lower left", fontsize=8.5, framealpha=0.95)

    fig.suptitle("Ablation Study: Disentangling Classical Damping vs Quantum Geometric Confinement", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"[Saved Figure] {fig_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
