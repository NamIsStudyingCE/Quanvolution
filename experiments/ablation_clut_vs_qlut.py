"""Step 4 of C-LUT vs Q-LUT Isolated Ablation:
Statistical aggregation, paired t-tests, win-rates, hardware sanity check via rust_qlut,
and Markdown / CSV report generation (optionally orchestrating Steps 1-3).

Outputs ONLY inside results/ablation_clut/:
  - results/ablation_clut/ablation_summary.csv
  - results/ablation_clut/ablation_summary.md
  - results/ablation_clut/hardware_sanity.json
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import rust_qlut
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "results" / "ablation_clut"
CLUT_BIN_DIR = PROJECT_ROOT / "weights" / "ablation_clut"
WEIGHTS_DIR = PROJECT_ROOT / "weights"

MODEL_LABELS = {
    "ctrl": "Ctrl (Conv2d 0% LUT)",
    "clut_septrig": "CLUT-SepTrig",
    "clut_randmlp": "CLUT-RandMLP",
    "qlut": "Q-LUT (SU(16) L=2)",
}
BASELINE_KEYS = ["ctrl", "clut_septrig", "clut_randmlp"]


def paired_ttest_and_winrate(
    vals_a: List[float],
    vals_b: List[float],
) -> Tuple[float, float, int, int]:
    """Computes mean(a - b), paired two-sided t-test p-value, wins(a > b), total n."""
    a = np.asarray(vals_a, dtype=np.float64)
    b = np.asarray(vals_b, dtype=np.float64)
    diff = a - b
    mean_delta = float(np.mean(diff))
    wins = int(np.sum(diff > 1e-12))
    n = int(len(diff))
    if np.allclose(diff, 0.0, atol=1e-12):
        return mean_delta, 1.0, wins, n
    _, p_val = stats.ttest_rel(a, b)
    if np.isnan(p_val):
        p_val = 1.0
    return mean_delta, float(p_val), wins, n


def benchmark_rust_qlut_hardware() -> dict:
    """Measures rust_qlut SIMD kernel latency & verifies identical .bin byte footprints."""
    rng = np.random.RandomState(42)
    single_u8 = np.ascontiguousarray(rng.randint(0, 256, size=(1, 28, 28), dtype=np.uint8))
    batch_u8 = np.ascontiguousarray(rng.randint(0, 256, size=(100, 28, 28), dtype=np.uint8))

    out_i8_1 = np.empty((1, 4, 14, 14), dtype=np.int8)
    out_i8_b = np.empty((100, 4, 14, 14), dtype=np.int8)
    out_f32_1 = np.empty((1, 4, 14, 14), dtype=np.float32)
    out_f32_b = np.empty((100, 4, 14, 14), dtype=np.float32)

    # Warmup
    for _ in range(50):
        rust_qlut.forward_int8_nchw(single_u8, out_i8_1, use_shift=True, parallel=False)
        rust_qlut.forward_interp4d_nchw(single_u8, out_f32_1, parallel=False)

    n_runs = 500
    t0 = time.perf_counter()
    for _ in range(n_runs):
        rust_qlut.forward_int8_nchw(single_u8, out_i8_1, use_shift=True, parallel=False)
    lat_i8_1_ms = (time.perf_counter() - t0) * 1000.0 / n_runs

    t0 = time.perf_counter()
    for _ in range(n_runs):
        rust_qlut.forward_interp4d_nchw(single_u8, out_f32_1, parallel=False)
    lat_f32_1_ms = (time.perf_counter() - t0) * 1000.0 / n_runs

    n_batch_runs = 50
    t0 = time.perf_counter()
    for _ in range(n_batch_runs):
        rust_qlut.forward_int8_nchw(batch_u8, out_i8_b, use_shift=True, parallel=False)
    lat_i8_b_ms_per_img = (time.perf_counter() - t0) * 1000.0 / (n_batch_runs * 100)

    t0 = time.perf_counter()
    for _ in range(n_batch_runs):
        rust_qlut.forward_interp4d_nchw(batch_u8, out_f32_b, parallel=False)
    lat_f32_b_ms_per_img = (time.perf_counter() - t0) * 1000.0 / (n_batch_runs * 100)

    file_sizes = {
        "qlut_int8_shift_bytes": (WEIGHTS_DIR / "qlut_strongly_L2_4bit_int8_shift.bin").stat().st_size,
        "clut_septrig_int8_shift_bytes": (CLUT_BIN_DIR / "clut_septrig_4bit_int8_shift.bin").stat().st_size,
        "clut_randmlp_int8_shift_bytes": (CLUT_BIN_DIR / "clut_randmlp_4bit_int8_shift.bin").stat().st_size,
        "qlut_fp32_grid_bytes": (WEIGHTS_DIR / "qlut_strongly_L2_4bit_fp32_grid.bin").stat().st_size,
        "clut_septrig_fp32_grid_bytes": (CLUT_BIN_DIR / "clut_septrig_4bit_fp32_grid.bin").stat().st_size,
        "clut_randmlp_fp32_grid_bytes": (CLUT_BIN_DIR / "clut_randmlp_4bit_fp32_grid.bin").stat().st_size,
    }
    assert file_sizes["qlut_int8_shift_bytes"] == file_sizes["clut_septrig_int8_shift_bytes"] == 262144
    assert file_sizes["qlut_int8_shift_bytes"] == file_sizes["clut_randmlp_int8_shift_bytes"]
    assert file_sizes["qlut_fp32_grid_bytes"] == file_sizes["clut_septrig_fp32_grid_bytes"] == 1048576
    assert file_sizes["qlut_fp32_grid_bytes"] == file_sizes["clut_randmlp_fp32_grid_bytes"]

    hw = {
        "rust_qlut_int8_shift_single_ms": float(lat_i8_1_ms),
        "rust_qlut_interp4d_single_ms": float(lat_f32_1_ms),
        "rust_qlut_int8_shift_batch100_ms_per_img": float(lat_i8_b_ms_per_img),
        "rust_qlut_interp4d_batch100_ms_per_img": float(lat_f32_b_ms_per_img),
        "file_sizes": file_sizes,
        "note": (
            "rust_qlut compiles LUT bytes statically via include_bytes! without mutating rust_qlut/src. "
            "Because INT8 Shift (262,144 B) and Interp4D FP32 (1,048,576 B) binaries have byte-identical "
            "layouts and access patterns, C-LUT and Q-LUT share identical hardware latency and memory footprint."
        ),
    }
    with open(OUT_DIR / "hardware_sanity.json", "w", encoding="utf-8") as f:
        json.dump(hw, f, indent=2)
    return hw


def generate_reports() -> None:
    calib_path = OUT_DIR / "clut_calibration_stats.json"
    train_path = OUT_DIR / "train_results.json"
    eval_path = OUT_DIR / "eval_results.json"

    for p in (calib_path, eval_path):
        if not p.exists():
            raise FileNotFoundError(f"Required input missing: {p}")

    with open(calib_path, "r", encoding="utf-8") as f:
        calib = json.load(f)
    train_res = None
    if train_path.exists():
        with open(train_path, "r", encoding="utf-8") as f:
            train_res = json.load(f)
    with open(eval_path, "r", encoding="utf-8") as f:
        eval_res = json.load(f)

    hw = benchmark_rust_qlut_hardware()

    meta = eval_res["metadata"]
    seeds = [str(s) for s in meta["seeds"]]
    modes = meta["modes"]
    stems = meta["stems"]
    models_order = meta["models"]

    csv_rows: List[dict] = []
    md: List[str] = []
    md.append("# Isolated Ablation Study: Classical-LUT (`C-LUT`) vs. Quantum-LUT (`Q-LUT`)")
    md.append("")
    md.append("> **Scientific Question:** *Does the clinical speckle-noise robustness of `WideGRQuanvNet` stem from non-local $SU(16)$ quantum entanglement/unitary interference, or merely from 4-bit LUT quantization and bounded $[-1, 1]$ non-linearities?*")
    md.append("")
    md.append("## 1. LUT Output Distribution Calibration & Binary Parity")
    md.append("")
    md.append("| Table Variant | Pre-Act Gain | Mean | Std (`fp32_grid`) | Min | Max | Std vs `Q-LUT` | File Sizes (`INT8 Shift` / `Interp4D`) |")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    tstats = calib["tables"]
    q_std = tstats["qlut_strongly_L2"]["stats_fp32_grid"]["std"]
    for tname, pretty in [
        ("qlut_strongly_L2", "`Q-LUT` (`SU(16)` Strongly L=2)"),
        ("clut_septrig", "`CLUT-SepTrig` (Separable cos/sin)"),
        ("clut_randmlp", "`CLUT-RandMLP` (Ortho 4-16-4 GELU+Tanh)"),
    ]:
        st_fp = tstats[tname]["stats_fp32_grid"]
        gain_val = tstats[tname].get("gain")
        gain_str = f"`{gain_val:.4f}`" if isinstance(gain_val, (int, float)) else "`native`"
        rel_diff = (st_fp["std"] - q_std) / q_std * 100.0
        md.append(
            f"| {pretty} | {gain_str} | `{st_fp['mean']:+.5f}` | `{st_fp['std']:.5f}` | "
            f"`{st_fp['min']:+.4f}` | `{st_fp['max']:+.4f}` | `{rel_diff:+.2f}%` | `262,144 B` / `1,048,576 B` |"
        )

    parity = calib.get("rust_extractor_parity", {})
    md.append("")
    md.append(
        f"- **PyTorch-vs-`rust_qlut` Numerical Parity:** "
        f"`Interp4D max|diff| = {parity.get('max_diff_interp4d', float('nan')):.2e}`, "
        f"`INT8 Shift max|diff| = {parity.get('max_diff_int8_shift_lsb', float('nan')):.1f}` LSB."
    )
    md.append(
        f"- **Hardware Footprint & Latency Sanity (`rust_qlut` native CPU SIMD):** "
        f"`forward_int8_nchw (use_shift=True)` = `{hw['rust_qlut_int8_shift_single_ms']:.4f} ms/img` single (`{hw['rust_qlut_int8_shift_batch100_ms_per_img']:.4f} ms/img` batch-100); "
        f"`forward_interp4d_nchw` = `{hw['rust_qlut_interp4d_single_ms']:.4f} ms/img` single (`{hw['rust_qlut_interp4d_batch100_ms_per_img']:.4f} ms/img` batch-100). "
        f"Because `C-LUT` and `Q-LUT` compile into byte-identical `256 KiB` / `1,024 KiB` LUT arrays with identical gather memory layouts, "
        f"their hardware latency and RAM footprint are identical."
    )

    if train_res is not None and train_res.get("runs"):
        md.append("")
        md.append("## 2. Training Parity Verification (`Wide-Fusion v2`, 5 Seeds)")
        md.append("")
        md.append("| Dataset | Variant | Params (Exact) | Best Val (Mean ± Std) | Clean Test ACC (Mean ± Std) | Mean Train Sec/Seed |")
        md.append("| :--- | :--- | :---: | :---: | :---: | :---: |")
        grouped: Dict[Tuple[str, str], List[dict]] = {}
        for _, rec in train_res["runs"].items():
            grouped.setdefault((rec["ds"], rec["variant"]), []).append(rec)
        for (ds_name, var_name), items in sorted(grouped.items()):
            params_set = sorted({int(x["params"]) for x in items})
            vals = [x["best_val"] * 100.0 for x in items]
            accs = [x["clean_test"]["acc"] * 100.0 for x in items]
            secs = [x["train_sec"] for x in items]
            md.append(
                f"| `{ds_name}` | `{var_name}` | `{params_set}` | "
                f"`{np.mean(vals):.2f} ± {np.std(vals):.2f}%` | "
                f"`{np.mean(accs):.2f} ± {np.std(accs):.2f}%` | `{np.mean(secs):.1f}s` |"
            )

    md.append("")
    md.append("## 3. Controlled Clean & Speckle-Noise Evaluation (`mean ± std` across 5 seeds)")

    verdict_highlights: List[str] = []

    for ds, ds_payload in eval_res["datasets"].items():
        nc = ds_payload["nc"]
        n_test = ds_payload["n_test"]
        sigmas = ds_payload["sigmas"]
        md.append("")
        md.append(f"### Dataset: `{ds}` (`nc={nc}`, `n_test={n_test}`, seeds=`{ds_payload['seeds']}`)")

        for mode in modes:
            mode_label = "Interp4D (`fp32_grid.bin`, 1,024 KiB — Primary Training Parity)" if mode == "interp4d" else "INT8 Shift-LUT (`int8_shift.bin`, 256 KiB — Edge Deployment Kernel)"
            for stem in stems:
                stem_label = "Raw Stem (`phase_damping=False`, `stabilize_tau=999.0`)" if stem == "raw_stem" else "Stabilized Stem (`phase_damping=True` symmetric, `stabilize_tau=0.024`)"
                md.append("")
                md.append(f"#### [{mode_label}] — {stem_label}")
                md.append("")
                md.append("| $\\sigma$ | Model | Acc (%) | Macro-F1 (%) | AUC (%) | Mean $\\bar{\\alpha}(\\sigma)$ | $\\Delta$ Acc (`Q-LUT - M`) | Paired $p$ (Acc) | Win Rate (`Q-LUT > M`) |")
                md.append("| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

                stem_tbl = ds_payload["results"][mode][stem]
                for sg in sigmas:
                    sg_str = f"{sg:.2f}"
                    sg_tbl = stem_tbl[sg_str]
                    qlut_accs = [sg_tbl["qlut"]["per_seed"][s]["acc"] for s in seeds]
                    qlut_mf1s = [sg_tbl["qlut"]["per_seed"][s]["macro_f1"] for s in seeds]

                    for mk in models_order:
                        entry = sg_tbl[mk]
                        if mk == "qlut":
                            d_acc_str, p_acc_str, win_str = "—", "—", "—"
                            d_acc_val, p_acc_val, win_val, d_mf1_val, p_mf1_val = 0.0, 1.0, "5/5", 0.0, 1.0
                        else:
                            m_accs = [entry["per_seed"][s]["acc"] for s in seeds]
                            m_mf1s = [entry["per_seed"][s]["macro_f1"] for s in seeds]
                            d_acc, p_acc, wins, n_tot = paired_ttest_and_winrate(qlut_accs, m_accs)
                            d_mf1, p_mf1, _, _ = paired_ttest_and_winrate(qlut_mf1s, m_mf1s)
                            d_acc_str = f"`{d_acc:+.2f}%`"
                            p_acc_str = f"`{p_acc:.4f}`"
                            win_str = f"`{wins}/{n_tot}`"
                            d_acc_val, p_acc_val, win_val, d_mf1_val, p_mf1_val = d_acc, p_acc, f"{wins}/{n_tot}", d_mf1, p_mf1

                            if mode == "interp4d" and sg in (0.0, 0.15, 0.20):
                                verdict_highlights.append(
                                    f"- `{ds}` | `{stem}` | $\\sigma={sg_str}$ | `Q-LUT vs {mk}`: "
                                    f"$\\Delta$Acc=`{d_acc:+.2f}%` (`p={p_acc:.4f}`, win `{wins}/{n_tot}`), "
                                    f"$\\Delta$mF1=`{d_mf1:+.2f}%` (`p={p_mf1:.4f}`)"
                                )

                        md.append(
                            f"| `{sg_str}` | {MODEL_LABELS[mk]} | "
                            f"`{entry['acc_mean']:.2f} ± {entry['acc_std']:.2f}` | "
                            f"`{entry['macro_f1_mean']:.2f} ± {entry['macro_f1_std']:.2f}` | "
                            f"`{entry['auc_mean']:.2f}` | `{entry['alpha_mean_mean']:.3f}` | "
                            f"{d_acc_str} | {p_acc_str} | {win_str} |"
                        )

                        csv_rows.append(
                            {
                                "dataset": ds,
                                "lut_mode": mode,
                                "stem_condition": stem,
                                "sigma": sg_str,
                                "model": mk,
                                "acc_mean": f"{entry['acc_mean']:.4f}",
                                "acc_std": f"{entry['acc_std']:.4f}",
                                "macro_f1_mean": f"{entry['macro_f1_mean']:.4f}",
                                "macro_f1_std": f"{entry['macro_f1_std']:.4f}",
                                "auc_mean": f"{entry['auc_mean']:.4f}",
                                "auc_std": f"{entry['auc_std']:.4f}",
                                "alpha_mean": f"{entry['alpha_mean_mean']:.4f}",
                                "delta_qlut_acc": f"{d_acc_val:+.4f}" if mk != "qlut" else "",
                                "p_qlut_acc": f"{p_acc_val:.6f}" if mk != "qlut" else "",
                                "win_rate_acc": win_val if mk != "qlut" else "",
                                "delta_qlut_mf1": f"{d_mf1_val:+.4f}" if mk != "qlut" else "",
                                "p_qlut_mf1": f"{p_mf1_val:.6f}" if mk != "qlut" else "",
                            }
                        )

    md.append("")
    md.append("## 4. Key Pairwise Highlights (`Interp4D`, $\\sigma \\in \\{0.00, 0.15, 0.20\\}$)")
    md.append("")
    md.extend(verdict_highlights)

    csv_path = OUT_DIR / "ablation_summary.csv"
    fieldnames = [
        "dataset",
        "lut_mode",
        "stem_condition",
        "sigma",
        "model",
        "acc_mean",
        "acc_std",
        "macro_f1_mean",
        "macro_f1_std",
        "auc_mean",
        "auc_std",
        "alpha_mean",
        "delta_qlut_acc",
        "p_qlut_acc",
        "win_rate_acc",
        "delta_qlut_mf1",
        "p_qlut_mf1",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    md_path = OUT_DIR / "ablation_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    print(f"[Saved CSV] {csv_path}")
    print(f"[Saved MD ] {md_path}")
    print(f"[Saved HW ] {OUT_DIR / 'hardware_sanity.json'}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Orchestrate or report C-LUT vs Q-LUT ablation.")
    parser.add_argument("--run-all", action="store_true", help="Run Steps 1, 2, 3, then generate reports.")
    parser.add_argument("--datasets", nargs="+", default=["pneumoniamnist"])
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.run_all:
        py = sys.executable
        subprocess.run([py, str(PROJECT_ROOT / "scripts" / "generate_clut.py")], check=True)
        subprocess.run(
            [py, str(PROJECT_ROOT / "experiments" / "train_ablation_clut.py"), "--datasets", *args.datasets, "--resume"],
            check=True,
        )
        subprocess.run(
            [py, str(PROJECT_ROOT / "experiments" / "eval_ablation_clut.py"), "--datasets", *args.datasets],
            check=True,
        )

    generate_reports()


if __name__ == "__main__":
    main()
