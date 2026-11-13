"""
Task 2: Pareto Evaluation — INT8 Shift/Grid-LUT (256 KiB) vs Interp4D SIMD (1 MiB) vs Classical Control
========================================================================================================
Strictly separates accuracy and latency across the exact same model checkpoints (Zero Retraining):
  - Wide Classical Control (FP32)
  - Wide-GR-Quanv [INT8 Shift-LUT: p >> 4, 256 KiB table, 0 FLOPs]
  - Wide-GR-Quanv [INT8 Grid-LUT: round(p*15/255), 256 KiB table]
  - Wide-GR-Quanv [Interp4D SIMD: FP32 4D multilinear, 1,024 KiB table]

Outputs:
  - reports/pareto_int8_vs_interp4d.json
  - reports/figures/pareto_int8_vs_interp4d.png
"""

import json
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rust_qlut
import torch
from scipy import stats

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from models.gr_quanv import _build_axial_u8, _build_dilated_u8, phase_damping_u8
from models.gr_quanv_wide import WideClassicalControl, WideGRQuanvNet
from run_pillar2_expansion import DATASETS, NOISE_DRAWS, SEEDS, SIGMAS, load, speckle
from run_pillar2_wide import evaluate_wide
from utils.metrics import calculate_metrics

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_qlut_exact_mode(imgs_u8: np.ndarray, mode: str, phase_damping: bool = True) -> torch.Tensor:
    """mode in {'int8_shift', 'int8_grid', 'interp4d'}."""
    imgs_u8 = np.ascontiguousarray(imgs_u8, dtype=np.uint8)
    if phase_damping:
        imgs_u8 = phase_damping_u8(imgs_u8)
    N = imgs_u8.shape[0]
    inputs = [
        imgs_u8,
        _build_dilated_u8(imgs_u8, 2),
        _build_dilated_u8(imgs_u8, 3),
        _build_axial_u8(imgs_u8),
    ]
    par = N >= 64
    if mode == "interp4d":
        scales = []
        for u in inputs:
            s = np.empty((N, 4, 14, 14), dtype=np.float32)
            rust_qlut.forward_interp4d_nchw(u, s, parallel=par)
            scales.append(s)
        return torch.from_numpy(np.concatenate(scales, axis=1))
    else:
        use_shift = (mode == "int8_shift")
        scales = []
        for u in inputs:
            s = np.empty((N, 4, 14, 14), dtype=np.int8)
            rust_qlut.forward_int8_nchw(u, s, use_shift=use_shift, parallel=par)
            scales.append(s)
        return torch.from_numpy(np.concatenate(scales, axis=1)).to(torch.float32) / 127.0


def benchmark_cpu_latencies() -> dict:
    """Measure CPU Batch=1 & Batch=64 latencies for Q-LUT extractors and full PyTorch models."""
    img1 = np.random.default_rng(42).integers(0, 256, size=(1, 28, 28), dtype=np.uint8)
    out_i8 = np.empty((1, 4, 14, 14), dtype=np.int8)
    out_f32 = np.empty((1, 4, 14, 14), dtype=np.float32)

    # 1. Single-scale Q-LUT Rust kernel (Batch=1)
    for _ in range(500):
        rust_qlut.forward_int8_nchw(img1, out_i8, use_shift=True, parallel=False)
        rust_qlut.forward_interp4d_nchw(img1, out_f32, parallel=False)

    t0 = time.perf_counter()
    for _ in range(10000):
        rust_qlut.forward_int8_nchw(img1, out_i8, use_shift=True, parallel=False)
    lut1_int8_us = ((time.perf_counter() - t0) / 10000) * 1e6

    t0 = time.perf_counter()
    for _ in range(10000):
        rust_qlut.forward_interp4d_nchw(img1, out_f32, parallel=False)
    lut1_interp4d_us = ((time.perf_counter() - t0) / 10000) * 1e6

    # 2. Full End-to-End PyTorch CPU forward latency (Batch=1 and Batch=64)
    mc = WideClassicalControl(num_classes=2).eval().cpu()
    mq = WideGRQuanvNet(num_classes=2).eval().cpu()
    x1 = torch.randn(1, 1, 28, 28)
    x64 = torch.randn(64, 1, 28, 28)
    img64 = np.random.default_rng(7).integers(0, 256, size=(64, 28, 28), dtype=np.uint8)

    with torch.no_grad():
        for _ in range(50):
            _ = mc(x1)
            q_tmp = extract_qlut_exact_mode(img1, "int8_shift", phase_damping=False)
            _ = mq(x1, q_tmp)

        n_b1 = 500
        t0 = time.perf_counter()
        for _ in range(n_b1):
            _ = mc(x1)
        ctrl_b1_ms = ((time.perf_counter() - t0) / n_b1) * 1e3

        t0 = time.perf_counter()
        for _ in range(n_b1):
            q_s = extract_qlut_exact_mode(img1, "int8_shift", phase_damping=False)
            _ = mq(x1, q_s)
        quanv_int8_b1_ms = ((time.perf_counter() - t0) / n_b1) * 1e3

        t0 = time.perf_counter()
        for _ in range(n_b1):
            q_f = extract_qlut_exact_mode(img1, "interp4d", phase_damping=False)
            _ = mq(x1, q_f)
        quanv_interp4d_b1_ms = ((time.perf_counter() - t0) / n_b1) * 1e3

        n_b64 = 100
        t0 = time.perf_counter()
        for _ in range(n_b64):
            _ = mc(x64)
        ctrl_b64_ms = ((time.perf_counter() - t0) / n_b64) * 1e3 / 64.0

        t0 = time.perf_counter()
        for _ in range(n_b64):
            q_s64 = extract_qlut_exact_mode(img64, "int8_shift", phase_damping=False)
            _ = mq(x64, q_s64)
        quanv_int8_b64_ms = ((time.perf_counter() - t0) / n_b64) * 1e3 / 64.0

        t0 = time.perf_counter()
        for _ in range(n_b64):
            q_f64 = extract_qlut_exact_mode(img64, "interp4d", phase_damping=False)
            _ = mq(x64, q_f64)
        quanv_interp4d_b64_ms = ((time.perf_counter() - t0) / n_b64) * 1e3 / 64.0

    return {
        "lut_single_scale_int8_shift_us": round(lut1_int8_us, 3),
        "lut_single_scale_interp4d_us": round(lut1_interp4d_us, 3),
        "pytorch_e2e_b1_ctrl_ms": round(ctrl_b1_ms, 4),
        "pytorch_e2e_b1_grquanv_int8_ms": round(quanv_int8_b1_ms, 4),
        "pytorch_e2e_b1_grquanv_interp4d_ms": round(quanv_interp4d_b1_ms, 4),
        "pytorch_e2e_b64_per_img_ctrl_ms": round(ctrl_b64_ms, 4),
        "pytorch_e2e_b64_per_img_grquanv_int8_ms": round(quanv_int8_b64_ms, 4),
        "pytorch_e2e_b64_per_img_grquanv_interp4d_ms": round(quanv_interp4d_b64_ms, 4),
    }


def main():
    print("=" * 90)
    print("  TASK 2: PARETO BENCHMARK — INT8 (256 KiB) vs INTERP4D (1 MiB) vs CLASSICAL CONTROL")
    print("=" * 90)

    datasets = ["breastmnist", "pneumoniamnist", "octmnist"]
    modes = ["control", "int8_shift", "int8_grid", "interp4d"]
    ckpt_dir = PROJECT_ROOT / "weights" / "pillar2_wide"

    latencies = benchmark_cpu_latencies()
    print("[CPU Latencies]", json.dumps(latencies, indent=2))

    eval_data = {}
    for ds in datasets:
        print(f"\n=== Dataset: {ds.upper()} ===")
        npz_path = DATASETS[ds][0]
        D = load(npz_path)
        nc = int(len(np.unique(D["ytr"])))
        te_f = D["te"].astype(np.float32) / 255.0
        y_te = D["yte"]

        models_ctrl = {}
        models_quanv = {}
        for s in SEEDS:
            mc = WideClassicalControl(num_classes=nc).to(DEV)
            mc.load_state_dict(torch.load(ckpt_dir / f"wide_v2_interp4d_{ds}_full_control_s{s}.pt", map_location=DEV, weights_only=True))
            mc.eval()
            models_ctrl[s] = mc

            mq = WideGRQuanvNet(num_classes=nc).to(DEV)
            mq.load_state_dict(torch.load(ckpt_dir / f"wide_v2_interp4d_{ds}_full_grquanv_s{s}.pt", map_location=DEV, weights_only=True))
            mq.eval()
            models_quanv[s] = mq

        eval_data[ds] = {}
        for sg in SIGMAS:
            draws = NOISE_DRAWS if sg > 0 else [0]
            draw_res = {m: {s: [] for s in SEEDS} for m in modes}

            for dr in draws:
                noisy_u8 = speckle(te_f, sg, dr)
                xu8_gpu = torch.from_numpy(np.ascontiguousarray(noisy_u8, dtype=np.uint8)).to(DEV)

                q_modes = {
                    qm: extract_qlut_exact_mode(noisy_u8, qm, phase_damping=True).to(dtype=torch.float16, device=DEV)
                    for qm in ["int8_shift", "int8_grid", "interp4d"]
                }

                for s in SEEDS:
                    # Control
                    m_c = evaluate_wide(models_ctrl[s], xu8_gpu, q_modes["interp4d"], y_te, nc)
                    draw_res["control"][s].append(m_c)
                    # GR-Quanv modes
                    for qm in ["int8_shift", "int8_grid", "interp4d"]:
                        m_q = evaluate_wide(models_quanv[s], xu8_gpu, q_modes[qm], y_te, nc)
                        draw_res[qm][s].append(m_q)

            sg_key = f"{sg:.2f}"
            eval_data[ds][sg_key] = {}
            print(f"  -- Sigma = {sg_key} --")
            for mode in modes:
                seed_acc = [np.mean([x["acc"] for x in draw_res[mode][s]]) * 100.0 for s in SEEDS]
                seed_auc = [np.mean([x["auc"] for x in draw_res[mode][s]]) * 100.0 for s in SEEDS]
                seed_f1 = [np.mean([x["f1"] for x in draw_res[mode][s]]) * 100.0 for s in SEEDS]
                eval_data[ds][sg_key][mode] = {
                    "acc_mean": float(np.mean(seed_acc)),
                    "acc_std": float(np.std(seed_acc)),
                    "auc_mean": float(np.mean(seed_auc)),
                    "auc_std": float(np.std(seed_auc)),
                    "f1_mean": float(np.mean(seed_f1)),
                    "f1_std": float(np.std(seed_f1)),
                    "per_seed_acc": [float(v) for v in seed_acc],
                }
                print(f"    {mode:<12} | ACC: {np.mean(seed_acc):5.2f}+-{np.std(seed_acc):4.2f}% | "
                      f"AUC: {np.mean(seed_auc):5.2f}% | F1: {np.mean(seed_f1):5.2f}%")

            # Paired t-test: INT8 Shift vs Control and Interp4D vs Control
            ctrl_acc = np.array(eval_data[ds][sg_key]["control"]["per_seed_acc"])
            for qm in ["int8_shift", "interp4d"]:
                qm_acc = np.array(eval_data[ds][sg_key][qm]["per_seed_acc"])
                t_stat, p_val = stats.ttest_rel(qm_acc, ctrl_acc)
                eval_data[ds][sg_key][f"{qm}_vs_control"] = {
                    "delta_acc_pct": float(np.mean(qm_acc - ctrl_acc)),
                    "p_value": float(p_val),
                }

    out_payload = {
        "metadata": {
            "description": "Strict separation of INT8 Shift/Grid-LUT (256 KiB) vs Interp4D SIMD (1 MiB) vs Classical Control",
            "seeds": SEEDS,
            "sigmas": SIGMAS,
            "noise_draws": NOISE_DRAWS,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "cpu_latencies": latencies,
        "results": eval_data,
    }
    out_json = PROJECT_ROOT / "reports" / "pareto_int8_vs_interp4d.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2)
    print(f"\n[Saved JSON] {out_json}")

    # Plot Pareto & Noise comparison figure
    fig_path = PROJECT_ROOT / "reports" / "figures" / "pareto_int8_vs_interp4d.png"
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8), dpi=220)
    styles = {
        "control": ("#34495e", "s--", "Wide Classical Control (FP32)"),
        "int8_shift": ("#27ae60", "^-.", "GR-Quanv INT8 Shift-LUT (256 KiB, 0-FLOP)"),
        "int8_grid": ("#8e44ad", "d:", "GR-Quanv INT8 Grid-LUT (256 KiB)"),
        "interp4d": ("#c0392b", "o-", "GR-Quanv Interp4D SIMD (1,024 KiB)"),
    }
    x_s = np.array(SIGMAS)
    for ax, ds in zip(axes, datasets):
        for mode, (col, fmt, lbl) in styles.items():
            ys = [eval_data[ds][f"{sg:.2f}"][mode]["acc_mean"] for sg in SIGMAS]
            ax.plot(x_s, ys, fmt, color=col, label=lbl, lw=2.0, markersize=6)
        ax.set_title(ds.upper(), fontweight="bold")
        ax.set_xlabel("Speckle Noise (sigma)")
        ax.set_ylabel("Accuracy (%)")
        ax.set_xticks(SIGMAS)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(loc="lower left", fontsize=7.8, framealpha=0.92)
    fig.suptitle("Clean & Speckle Robustness: INT8 Shift-LUT (256 KiB) vs Interp4D (1 MiB) vs Control", fontweight="bold")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"[Saved Figure] {fig_path}")


if __name__ == "__main__":
    main()
