"""
Ablation Study: Factor Isolation on Speckle Noise (PneumoniaMNIST, 5 Seeds)
==========================================================================
Isolates the contributions of:
  (1) Classical Control (Raw noisy input, stabilization disabled)
  (2) Classical Control + Input Phase Damping & Stem Variance Stabilization
  (3) GR-Quanv v2 (Raw noisy Q-LUT, Pre-LUT Phase Damping & stabilization disabled)
  (4) GR-Quanv v2 + Pre-LUT Phase Damping & Stem Variance Stabilization (Full System)

Matches exact seeds [0, 42, 100, 2023, 777], NOISE_DRAWS [1, 2, 3], and `speckle()`
generator from `run_pillar2_expansion.py` / `eval_speckle_wide.py`.
Outputs:
  - reports/ablation_isolation_results.json
  - reports/figures/ablation_factor_isolation.png
"""

import json
import os
import sys
from pathlib import Path
import numpy as np
import torch
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
from run_pillar2_expansion import SIGMAS, NOISE_DRAWS, speckle
from utils.metrics import calculate_metrics

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_batch(model, x_tensor, q_tensor, y_np):
    with torch.no_grad():
        preds = []
        bs = 2048
        for i in range(0, len(x_tensor), bs):
            xb = x_tensor[i:i + bs]
            qb = q_tensor[i:i + bs] if q_tensor is not None else None
            logits = model(xb, qb)
            preds.append(torch.softmax(logits, dim=1).cpu().numpy())
        p = np.concatenate(preds, axis=0)
        m = calculate_metrics(y_np, p, p.argmax(axis=1), 2)
        return m["acc"] * 100.0, m["auc"] * 100.0, m["f1"] * 100.0


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 84)
    print("  ABLATION STUDY: FACTOR ISOLATION ON PNEUMONIAMNIST (5 SEEDS)")
    print("=" * 84)

    ds = "pneumoniamnist"
    D = load_dataset_with_cache(ds, interp4d=True)
    te_f = D["te"].astype(np.float32) / 255.0
    y_test = D["yte"].astype(np.int64)

    ckpt_dir = PROJECT_ROOT / "weights" / "pillar2_wide"
    models_ctrl = {}
    models_quanv = {}
    for s in SEEDS:
        cp_c = ckpt_dir / f"wide_v2_interp4d_{ds}_full_control_s{s}.pt"
        cp_q = ckpt_dir / f"wide_v2_interp4d_{ds}_full_grquanv_s{s}.pt"

        mc = WideClassicalControl(num_classes=2).to(DEV)
        mc.load_state_dict(torch.load(cp_c, map_location=DEV, weights_only=True))
        mc.eval()
        models_ctrl[s] = mc

        mq = WideGRQuanvNet(num_classes=2).to(DEV)
        mq.load_state_dict(torch.load(cp_q, map_location=DEV, weights_only=True))
        mq.eval()
        models_quanv[s] = mq

    conditions = [
        "1_ctrl_raw",
        "2_ctrl_damped",
        "3_grquanv_raw",
        "4_grquanv_damped",
    ]
    results = {cond: {f"{sg:.2f}": {} for sg in SIGMAS} for cond in conditions}

    for sg in SIGMAS:
        draws = NOISE_DRAWS if sg > 0 else [0]
        print(f"\n--- Processing Sigma = {sg:.2f} ({len(draws)} noise draws) ---")
        draw_metrics = {cond: {s: [] for s in SEEDS} for cond in conditions}

        for dr in draws:
            noisy_u8 = speckle(te_f, sg, dr)
            x_raw_t = ((torch.from_numpy(noisy_u8).float().unsqueeze(1) / 255.0 - 0.5) / 0.5).to(DEV)

            noisy_damped_u8 = phase_damping_u8(noisy_u8, beta=18.0, tau=0.024)
            x_damped_t = ((torch.from_numpy(noisy_damped_u8).float().unsqueeze(1) / 255.0 - 0.5) / 0.5).to(DEV)

            q_raw_t, _ = extract_multiscale_qlut(noisy_u8, include_axial=True, interp4d=True, phase_damping=False)
            q_raw_t = q_raw_t.to(dtype=torch.float16, device=DEV).float()

            q_damped_t, _ = extract_multiscale_qlut(noisy_u8, include_axial=True, interp4d=True, phase_damping=True)
            q_damped_t = q_damped_t.to(dtype=torch.float16, device=DEV).float()

            for s in SEEDS:
                # 1. Classical Control RAW (stabilization OFF)
                models_ctrl[s].stabilize_tau = 999.0
                draw_metrics["1_ctrl_raw"][s].append(evaluate_batch(models_ctrl[s], x_raw_t, None, y_test))

                # 2. Classical Control DAMPED + Variance Stab ON
                models_ctrl[s].stabilize_tau = 0.024
                draw_metrics["2_ctrl_damped"][s].append(evaluate_batch(models_ctrl[s], x_damped_t, None, y_test))

                # 3. GR-Quanv RAW Q-LUT (stabilization OFF)
                models_quanv[s].stabilize_tau = 999.0
                draw_metrics["3_grquanv_raw"][s].append(evaluate_batch(models_quanv[s], x_raw_t, q_raw_t, y_test))

                # 4. GR-Quanv Full Proposed (Pre-LUT Damping + Variance Stab ON)
                models_quanv[s].stabilize_tau = 0.024
                draw_metrics["4_grquanv_damped"][s].append(evaluate_batch(models_quanv[s], x_raw_t, q_damped_t, y_test))

        for cond in conditions:
            seed_accs = [np.mean([item[0] for item in draw_metrics[cond][s]]) for s in SEEDS]
            seed_aucs = [np.mean([item[1] for item in draw_metrics[cond][s]]) for s in SEEDS]
            seed_f1s = [np.mean([item[2] for item in draw_metrics[cond][s]]) for s in SEEDS]
            results[cond][f"{sg:.2f}"] = {
                "acc_mean": float(np.mean(seed_accs)),
                "acc_std": float(np.std(seed_accs)),
                "auc_mean": float(np.mean(seed_aucs)),
                "auc_std": float(np.std(seed_aucs)),
                "f1_mean": float(np.mean(seed_f1s)),
                "f1_std": float(np.std(seed_f1s)),
            }
            print(f"  {cond:<20} => ACC: {np.mean(seed_accs):5.2f}+-{np.std(seed_accs):4.2f}% | "
                  f"AUC: {np.mean(seed_aucs):5.2f}% | F1: {np.mean(seed_f1s):5.2f}%")

    out_json = PROJECT_ROOT / "reports" / "ablation_isolation_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Saved JSON] {out_json}")

    fig_path = PROJECT_ROOT / "reports" / "figures" / "ablation_factor_isolation.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=220)
    x_vals = np.array(SIGMAS)

    c_map = {
        "1_ctrl_raw": ("#7f8c8d", "s--", "1. Classical Control (Raw Noisy)"),
        "2_ctrl_damped": ("#2980b9", "^-.", "2. Classical Control + Damping + VarStab"),
        "3_grquanv_raw": ("#e67e22", "v:", "3. GR-Quanv v2 (Raw Q-LUT, No Damping)"),
        "4_grquanv_damped": ("#c0392b", "o-", "4. GR-Quanv v2 + Pre-LUT Damping + VarStab"),
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
    ax1.legend(loc="lower left", fontsize=8.3, framealpha=0.95)

    ax2.set_xlabel("Speckle Noise Amplitude (sigma)")
    ax2.set_ylabel("F1 Score (%)")
    ax2.set_title("(B) F1 Score Isolation across Noise Levels", fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.set_xticks(SIGMAS)
    ax2.legend(loc="lower left", fontsize=8.3, framealpha=0.95)

    fig.suptitle("Ablation Study: Disentangling Classical Damping vs Quantum Geometric Confinement", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"[Saved Figure] {fig_path}")
    print("=" * 84)


if __name__ == "__main__":
    main()
