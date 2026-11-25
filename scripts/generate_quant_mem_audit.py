"""Step 3: 4-bit Quantization Cost (INT8 Shift vs Interp4D vs PyTorch Conv2d) & Unified Memory Audit.

Reads strictly read-only from:
  - reports/pareto_int8_vs_interp4d.json
  - reports/highres_scalability_results.json
  - reports/medquanv_cli_verification.json
  - reports/medquanv_cli_verification_a3.json
  - reports/medquanv_cli_v3_verification.json
  - reports/rigorous_stats_summary.json
  - weights/*.bin (exact file byte sizes)

Writes ONLY to:
  - reports/quantization_and_memory_audit.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
WEIGHTS_DIR = PROJECT_ROOT / "weights"

OUT_MD = REPORTS_DIR / "quantization_and_memory_audit.md"


def _kib(n_bytes: int) -> float:
    return n_bytes / 1024.0


def _pct_l2(kib: float, l2_kib: float = 2048.0) -> float:
    return (kib / l2_kib) * 100.0


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    with open(REPORTS_DIR / "pareto_int8_vs_interp4d.json", "r", encoding="utf-8") as f:
        pareto = json.load(f)
    with open(REPORTS_DIR / "highres_scalability_results.json", "r", encoding="utf-8") as f:
        highres = json.load(f)
    with open(REPORTS_DIR / "medquanv_cli_verification.json", "r", encoding="utf-8") as f:
        cli_v2 = json.load(f)
    with open(REPORTS_DIR / "medquanv_cli_verification_a3.json", "r", encoding="utf-8") as f:
        cli_v2_a3 = json.load(f)
    with open(REPORTS_DIR / "medquanv_cli_v3_verification.json", "r", encoding="utf-8") as f:
        cli_v3 = json.load(f)
    with open(REPORTS_DIR / "rigorous_stats_summary.json", "r", encoding="utf-8") as f:
        rstats = json.load(f)

    # Index C_quant_4bit tests by (dataset, mode, sigma)
    q_tests: Dict[tuple, dict] = {}
    for t in rstats["tests"]:
        if t["section"] == "C_quant_4bit":
            q_tests[(t["dataset"], t["mode"], t["sigma"])] = t
        elif t["section"] == "C_quant_4bit_sym":
            q_tests[("sym", t["dataset"], t["stem"], t["sigma"])] = t

    # Exact byte verification from disk and Rust source structs (64-bit x86_64 & aarch64)
    lut_int8_shift_b = (WEIGHTS_DIR / "qlut_strongly_L2_4bit_int8_shift.bin").stat().st_size
    lut_int8_grid_b = (WEIGHTS_DIR / "qlut_strongly_L2_4bit_int8_grid.bin").stat().st_size
    lut_fp32_grid_b = (WEIGHTS_DIR / "qlut_strongly_L2_4bit_fp32_grid.bin").stat().st_size
    w_v2_file_b = (WEIGHTS_DIR / "wide_v2_pneumoniamnist_s42_rust.bin").stat().st_size
    w_v2_header_b = 16
    w_v2_payload_b = w_v2_file_b - w_v2_header_b
    assert lut_int8_shift_b == 262144
    assert lut_int8_grid_b == 262144
    assert lut_fp32_grid_b == 1048576
    assert w_v2_payload_b == 147778 * 4 == 591112

    # Exact static index table sizes in rust_qlut/src/lib.rs on 64-bit targets (sizeof(usize) == 8):
    # - INT8 Shift uses bitwise shift in patch_idx_shift_w() -> 0 B index table
    # - INT8 Grid uses GRID_MAP { s12, s8, s4, s0: [usize; 256] } -> 4 * 256 * 8 = 8,192 B = 8.00 KiB
    # - Interp4D uses INTERP_TABLE { lo0..lo3: [usize; 256], frac: [f32; 256] } -> 4*256*8 + 256*4 = 9,216 B = 9.00 KiB
    grid_map_b = 4 * 256 * 8
    interp_table_b = 4 * 256 * 8 + 256 * 4

    # Exact runtime activation buffer sizes in rust_qlut/src/bin/medquanv_cli.rs::forward_single (Batch=1, 28x28):
    # Peak live tensors during head.block1 (Conv3x3 48->48 @ 14x14 with residual skip):
    #   h_stem: 48 * 196 * 4 = 37,632 B (36.75 KiB)
    #   b1_h  : 48 * 196 * 4 = 37,632 B (36.75 KiB)
    #   b1_out: 48 * 196 * 4 = 37,632 B (36.75 KiB)
    #   -> Exact peak live tensors (3-buffer residual block) = 112,896 B = 110.25 KiB
    #   -> Minimal 2-buffer ping-pong (if b1_out fused in-place into h_stem) = 75,264 B = 73.50 KiB
    # Total non-reused local stack arrays declared across forward_single() + extract_multiscale_qlut():
    stack_arrays = [
        ("extract_multiscale_qlut: img_d2/d3/ax (3 × 28×28 u8)", 3 * 784),
        ("extract_multiscale_qlut: tmp_i8 (784 i8, INT8 only) + q_raw (16×196 f32)", 784 + 16 * 196 * 4),
        ("forward_single: x_01 + x_norm (2 × 784 f32)", 2 * 784 * 4),
        ("forward_single: f_c + q_bn + f_q + alpha (4 × 16×196 f32)", 4 * 16 * 196 * 4),
        ("forward_single: q_h1 + q_h2 + cat_buf + g_in + g_h1 (5 × 32×196 f32)", 5 * 32 * 196 * 4),
        ("forward_single: h_stem + b1_h + b1_out (3 × 48×196 f32)", 3 * 48 * 196 * 4),
        ("forward_single: p1 (48×49 f32) + b2_h + b2_out + b2_skip (3 × 64×49 f32) + gap (64 f32)", (48 * 49 + 3 * 64 * 49 + 64) * 4),
    ]
    total_stack_b = sum(sz for _, sz in stack_arrays)
    peak_live_3buf_b = 3 * 48 * 196 * 4
    peak_live_2buf_b = 2 * 48 * 196 * 4

    md: List[str] = []
    md.append("# Audit Report: 4-bit Input Quantization Cost (`INT8` vs `Interp4D` vs `Conv2d`) & Unified Memory Accounting")
    md.append("")
    md.append("> **Scope & Measurement Environment:** All latency measurements below were executed natively on **x86_64 host CPU** (not extrapolated to Jetson Nano). Previous Jetson Nano FPS extrapolation columns (`×0.912`) and PennyLane Python-simulator speedup ratios are intentionally excluded. Memory percentages relative to `2,048 KiB` (2 MB) are purely architectural reference ratios against the Cortex-A57 L2 cache size.")
    md.append("")

    # =========================================================================
    # PART 1: ACCURACY COST OF 4-BIT QUANTIZATION (INT8 vs INTERP4D)
    # =========================================================================
    md.append("## 1. Classification Accuracy Cost of 4-bit Input Discretization (`INT8 Shift` vs `Interp4D`)")
    md.append("")
    md.append("Discretizing 8-bit medical pixel intensities into 4-bit bins (`16` levels per channel, discarding the 4 least-significant bits, step size `6.67%`) removes fine soft-tissue micro-contrast. Below is the exact zero-retraining accuracy comparison across all 3 MedMNIST datasets (5 paired seeds `[0, 42, 100, 2023, 777]`, 3 speckle noise draws; source: `reports/pareto_int8_vs_interp4d.json` and `reports/rigorous_stats_summary.json`):")
    md.append("")
    md.append("| Dataset | $\\sigma$ | `INT8 Shift` 256 KiB (ACC %) | `INT8 Grid` 256 KiB (ACC %) | `Interp4D` 1,024 KiB (ACC %) | $\\Delta$ (`INT8 Shift` − `Interp4D`) | 95% CI | $p_{\\text{raw}}$ | $p_{\\text{adj,fam}}$ | TOST ($\\pm 2\\%$) |")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |")

    for ds in ["breastmnist", "pneumoniamnist", "octmnist"]:
        for sg in ["0.00", "0.05", "0.10", "0.15", "0.20"]:
            node = pareto["results"][ds][sg]
            i8s = node["int8_shift"]
            i8g = node["int8_grid"]
            i4d = node["interp4d"]
            st = q_tests[(ds, "int8_shift_vs_interp4d", sg)]
            verdict = "EQUIV" if st["tost_equivalent"] else ("DIFF" if st["p_raw"] < 0.05 else "INCONCL.")
            md.append(
                f"| **{ds}** | `{sg}` | {i8s['acc_mean']:.2f} ± {i8s['acc_std']:.2f} | "
                f"{i8g['acc_mean']:.2f} ± {i8g['acc_std']:.2f} | "
                f"{i4d['acc_mean']:.2f} ± {i4d['acc_std']:.2f} | "
                f"**{st['delta_mean']:+.2f}%** | [{st['ci95_lo']:+.2f}, {st['ci95_hi']:+.2f}] | "
                f"{st['p_raw']:.4f} | {st['p_adj_family']:.4f} | {verdict} (p={st['p_tost']:.3f}) |"
            )

    md.append("")
    md.append("### Symmetric Stabilized Stem (`ablation_clut`, 5 Seeds: Q-LUT `INT8 Shift` vs `Interp4D`)")
    md.append("")
    md.append("| Dataset | Stem | $\\sigma$ | `INT8 Shift` (ACC %) | `Interp4D` (ACC %) | $\\Delta$ (`INT8` − `Interp4D`) | 95% CI | $p_{\\text{raw}}$ | $p_{\\text{adj,fam}}$ | TOST ($\\pm 2\\%$) |")
    md.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |")
    for ds in ["pneumoniamnist", "breastmnist"]:
        for stem in ["stabilized_stem", "raw_stem"]:
            for sg in ["0.00", "0.05", "0.10", "0.15", "0.20"]:
                st = q_tests[("sym", ds, stem, sg)]
                verdict = "EQUIV" if st["tost_equivalent"] else ("DIFF" if st["p_raw"] < 0.05 else "INCONCL.")
                md.append(
                    f"| **{ds}** | `{stem}` | `{sg}` | {st['mean_a']:.2f} ± {st['std_a']:.2f} | "
                    f"{st['mean_b']:.2f} ± {st['std_b']:.2f} | **{st['delta_mean']:+.2f}%** | "
                    f"[{st['ci95_lo']:+.2f}, {st['ci95_hi']:+.2f}] | {st['p_raw']:.4f} | "
                    f"{st['p_adj_family']:.4f} | {verdict} (p={st['p_tost']:.3f}) |"
                )

    md.append("")
    md.append("**Key findings on 4-bit quantization cost:**")
    md.append("1. **Zero-retraining degradation is dataset-dependent and non-trivial on fine-contrast retinal OCT:** On **OCTMNIST** (`σ=0.00`, `0.05`, `0.10`), zero-shot `INT8 Shift` drops **−3.40%**, **−4.69%**, and **−2.53%** relative to `Interp4D`, failing TOST equivalence (`±2%`). On **PneumoniaMNIST**, the drop is smaller (**−0.45%** clean, **−1.41% to −1.74%** under $\\sigma \\ge 0.10$, statistically significant on `p_raw` under `pre_lut_damped`). On **BreastMNIST**, `INT8 Shift` and `Interp4D` stay within **±0.40%** across all noise levels and pass TOST `±2%` equivalence at all $\\sigma$.")
    md.append("2. **Clarification for Table 4.2:** Table 4.2 (`reports/ablation_isolation_results.json`) used **`Interp4D` (1,024 KiB)** exclusively, NOT `INT8` (256 KiB). Mixing `Interp4D` accuracy with `INT8` footprint/latency is invalid and is strictly separated here.")
    md.append("3. **Recoverability via QAT (`v3`):** Training with Straight-Through Estimator `INT8-QAT` (`quanv_v3_int8`, `reports/pillar3_v3_results.json`) restores clean OCTMNIST `INT8` accuracy from `80.64%` (`v2` zero-shot) to `82.90%` (`0.0000%` compilation mismatch vs deployed `256 KiB` table), though it still trails `ctrl_v3_light` at OCTMNIST $\\sigma=0.10$ (`51.23%` vs `57.32%`).")
    md.append("")

    # =========================================================================
    # PART 2: HONEST LATENCY COMPARISON vs PYTORCH CONV2D (NO PENNYLANE, NO JETSON EXTRAPOLATION)
    # =========================================================================
    md.append("## 2. Honest Latency Comparison: `rust_qlut` (`INT8` & `Interp4D`) vs. `PyTorch Conv2d` (x86_64 Host CPU)")
    md.append("")
    md.append("Measured on x86_64 host CPU (`reports/highres_scalability_results.json`, single 4-channel $2\\times 2$ layer). Ratios `< 1.0×` explicitly indicate that `rust_qlut` is **slower** than `PyTorch Conv2d`:")
    md.append("")
    md.append("| Resolution ($H\\times W$, stride) | Patches ($N_p$) | `INT8 Shift` (256 KiB) | `Interp4D` (1,024 KiB) | `PyTorch Conv2d` CPU | Ratio `Conv2d / INT8` | Ratio `Conv2d / Interp4D` |")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    for row in highres["resolutions"]:
        int8_ms = row["qlut_int8_ms"]
        i4d_ms = row["qlut_interp4d_ms"]
        pt_ms = row["pytorch_conv2d_ms"]
        r_int8 = row["pytorch_conv2d_us"] / row["qlut_int8_us"]
        r_i4d = row["pytorch_conv2d_us"] / row["qlut_interp4d_us"]
        i4d_str = f"{r_i4d:.2f}× (**{1.0/r_i4d:.2f}× slower**)" if r_i4d < 1.0 else f"{r_i4d:.2f}× faster"
        md.append(
            f"| `{row['label']}` | {row['num_patches']:,} | {row['qlut_int8_us']:.2f} µs ({int8_ms:.4f} ms) | "
            f"{row['qlut_interp4d_us']:.2f} µs ({i4d_ms:.4f} ms) | {row['pytorch_conv2d_us']:.2f} µs ({pt_ms:.4f} ms) | "
            f"**{r_int8:.2f}× faster** | {i4d_str} |"
        )
    md.append("")
    md.append("### Full-Network End-to-End Latency on x86_64 Host CPU (`28×28`, Batch=1)")
    md.append("")
    md.append("| Implementation & Model | LUT Mode | LUT Latency (4 scales) | End-to-End Latency | Throughput (x86_64) | Max Logit Diff vs PyTorch |")
    md.append("| :--- | :--- | :---: | :---: | :---: | :---: |")
    m_v2 = cli_v2["modes"]
    m_a3 = cli_v2_a3["modes"]
    m_v3 = cli_v3["models"]
    md.append(f"| `medquanv-cli` v2 (`WideGRQuanvNet`, 148.3K, unblocked) | `INT8 Shift` | {m_v2['int8_shift']['qlut_4scale_us']:.2f} µs | {m_v2['int8_shift']['e2e_us']:.1f} µs ({m_v2['int8_shift']['e2e_us']/1000:.2f} ms) | {m_v2['int8_shift']['e2e_fps']:.1f} FPS | `{m_v2['int8_shift']['max_logit_diff']:.2e}` |")
    md.append(f"| `medquanv-cli` v2 (`WideGRQuanvNet`, 148.3K, unblocked) | `Interp4D` | {m_v2['interp4d']['qlut_4scale_us']:.2f} µs | {m_v2['interp4d']['e2e_us']:.1f} µs ({m_v2['interp4d']['e2e_us']/1000:.2f} ms) | {m_v2['interp4d']['e2e_fps']:.1f} FPS | `{m_v2['interp4d']['max_logit_diff']:.2e}` |")
    md.append(f"| `medquanv-cli` v2 (`WideGRQuanvNet`, 148.3K, A3 opt) | `INT8 Shift` | {m_a3['int8_shift']['qlut_4scale_us']:.2f} µs | {m_a3['int8_shift']['e2e_us']:.1f} µs ({m_a3['int8_shift']['e2e_us']/1000:.2f} ms) | {m_a3['int8_shift']['e2e_fps']:.1f} FPS | `{m_a3['int8_shift']['max_logit_diff']:.2e}` |")
    md.append(f"| `medquanv-cli` v2 (`WideGRQuanvNet`, 148.3K, A3 opt) | `Interp4D` | {m_a3['interp4d']['qlut_4scale_us']:.2f} µs | {m_a3['interp4d']['e2e_us']:.1f} µs ({m_a3['interp4d']['e2e_us']/1000:.2f} ms) | {m_a3['interp4d']['e2e_fps']:.1f} FPS | `{m_a3['interp4d']['max_logit_diff']:.2e}` |")
    md.append(f"| `medquanv-cli --suite-v3`: `ctrl_v3_light` (39.5K) | None (`Conv2d`) | N/A | {m_v3['ctrl_v3_light']['e2e_us']:.1f} µs ({m_v3['ctrl_v3_light']['e2e_us']/1000:.2f} ms) | {m_v3['ctrl_v3_light']['e2e_fps']:.1f} FPS | `{m_v3['ctrl_v3_light']['max_logit_diff_clean']:.2e}` |")
    md.append(f"| `medquanv-cli --suite-v3`: `quanv_v3_int8` (39.3K) | `INT8 Shift` | {m_v3['quanv_v3_int8']['qlut_stage1_us']:.2f} µs | {m_v3['quanv_v3_int8']['e2e_us']:.1f} µs ({m_v3['quanv_v3_int8']['e2e_us']/1000:.2f} ms) | {m_v3['quanv_v3_int8']['e2e_fps']:.1f} FPS | `{m_v3['quanv_v3_int8']['max_logit_diff_clean']:.2e}` |")
    md.append(f"| `medquanv-cli --suite-v3`: `micro_ctrl` (2.08K) | None (`Conv2d`) | N/A | {m_v3['micro_ctrl']['e2e_us']:.2f} µs | {m_v3['micro_ctrl']['e2e_fps']:,.1f} FPS | `{m_v3['micro_ctrl']['max_logit_diff_clean']:.2e}` |")
    md.append(f"| `medquanv-cli --suite-v3`: `micro_quanv` (2.00K) | `INT8 Shift` | {m_v3['micro_quanv']['qlut_stage1_us']:.2f} µs | {m_v3['micro_quanv']['e2e_us']:.2f} µs | {m_v3['micro_quanv']['e2e_fps']:,.1f} FPS | `{m_v3['micro_quanv']['max_logit_diff_clean']:.2e}` |")
    md.append("")
    md.append("**Key latency observations:**")
    md.append("- At clinical high resolution (`1024×1024`, $s=2$), **`INT8 Shift` is `4.89×` faster** than `PyTorch Conv2d` (`0.2718 ms` vs `1.3300 ms`), whereas **`Interp4D` (the accuracy-preserving variant) is `4.87×` slower** (`6.4739 ms` vs `1.3300 ms`) due to 16-corner gather + 15 FMA interpolation ops per patch.")
    md.append("- At full-network End-to-End level (`28×28`, `v3`), `quanv_v3_int8` (`1.126 ms`) and `ctrl_v3_light` (`1.114 ms`) have effectively identical latency (`+1.1%` overhead for `quanv_v3_int8`), while `quanv_v3_int8` requires `+256 KiB` extra static memory for the LUT.")
    md.append("- **Jetson Nano status:** No physical Jetson Nano hardware measurement has been performed yet; all FPS numbers above are strictly x86_64 host CPU measurements.")
    md.append("")

    # =========================================================================
    # PART 3: UNIFIED MEMORY AUDIT (RESOLVING 833 / 835 / 836 / 891 KiB)
    # =========================================================================
    md.append("## 3. Unified Memory Footprint Audit (Resolving `833.26` / `835.32` / `836` / `891.32 KiB` Discrepancies)")
    md.append("")
    md.append("### 3.1. Root Cause of Previous Discrepancies")
    md.append("")
    md.append("| Legacy Figure | Where It Appeared | Exact Formula Used | Why It Differed |")
    md.append("| :---: | :--- | :--- | :--- |")
    md.append("| **`833.26 KiB`** | `medquanv_cli_verification*.json`, `TECHNICAL_REPORT_EDGE_QUANV.md` §2 | `256.00` (`LUT_INT8_SHIFT`) + `577.26` (`147,778` BN-folded `f32`) | **Correct static ROM/weight footprint for `INT8 Shift`** (excludes runtime stack activations). |")
    md.append("| **`835.32 KiB`** | `reports/highres_scalability_results.json` | `256.00` (`LUT`) + `579.32` (`148,306` **unfolded** PyTorch `f32`) | Used pre-folding PyTorch parameter count (`+528` BN parameters = `+2.06 KiB`). |")
    md.append("| **`836 KiB`** | Legacy `medquanv-cli` v0.2.0 console telemetry | Rounded `835.32 KiB` up to integer | Same unfolded weight count + integer rounding. |")
    md.append("| **`891.32 KiB`** | Legacy `d90ed19` §2 table & ASCII diagram | `256.00` + `579.32` (unfolded) + `8.00` (`GRID_MAP`) + `48.00` (est. runtime buf) | Mixed unfolded weights with `GRID_MAP` (unused by `INT8 Shift`) and a rough `48 KiB` estimate for activations; ASCII diagram merged `8 + 48 = 56 KiB` under `Runtime Buf`. |")
    md.append("")
    md.append("*(Minor code-comment bug identified in `rust_qlut/src/lib.rs:78`: the comment claims `GRID_MAP` is `2 KiB`, whereas on 64-bit x86_64/aarch64 `sizeof(usize) = 8`, so `4 × 256 × 8 = 8,192 bytes = 8.00 KiB`. Likewise `INTERP_TABLE` is `4 × 256 × 8 + 256 × 4 = 9,216 bytes = 9.00 KiB`.)*")
    md.append("")

    md.append("### 3.2. Authoritative Byte-Exact Memory Table (`WideGRQuanvNet v2`, Batch=1, `28×28`, 64-bit Target)")
    md.append("")
    md.append("Verified directly from on-disk binary weights (`weights/wide_v2_pneumoniamnist_s42_rust.bin = 591,128 B = 16 B header + 591,112 B payload`) and Rust source structs (`rust_qlut/src/lib.rs`, `rust_qlut/src/bin/medquanv_cli.rs`):")
    md.append("")

    # Compute exact mode totals
    w_kib = _kib(w_v2_payload_b)
    lut_i8_kib = _kib(lut_int8_shift_b)
    lut_f32_kib = _kib(lut_fp32_grid_b)
    grid_kib = _kib(grid_map_b)
    interp_kib = _kib(interp_table_b)
    peak3_kib = _kib(peak_live_3buf_b)
    peak2_kib = _kib(peak_live_2buf_b)
    stack_kib = _kib(total_stack_b)

    modes_mem = [
        ("WideGRQuanvNet v2 — `INT8 Shift-LUT`", lut_i8_kib, 0.0, w_kib, peak3_kib, stack_kib),
        ("WideGRQuanvNet v2 — `INT8 Grid-LUT`", lut_i8_kib, grid_kib, w_kib, peak3_kib, stack_kib),
        ("WideGRQuanvNet v2 — `Interp4D SIMD`", lut_f32_kib, interp_kib, w_kib, peak3_kib, stack_kib),
        ("`WideClassicalControl` v2 (0% LUT)", 0.0, 0.0, m_v3["ctrl_v2_full"]["weight_kib"], peak3_kib, stack_kib),
        ("`quanv_v3_int8` (v3, `INT8 Shift`)", lut_i8_kib, 0.0, m_v3["quanv_v3_int8"]["weight_kib"], None, None),
        ("`ctrl_v3_light` (v3, 0% LUT)", 0.0, 0.0, m_v3["ctrl_v3_light"]["weight_kib"], None, None),
        ("`micro_quanv` (v3, `INT8 Shift`)", lut_i8_kib, 0.0, m_v3["micro_quanv"]["weight_kib"], None, None),
        ("`micro_ctrl` (v3, 0% LUT)", 0.0, 0.0, m_v3["micro_ctrl"]["weight_kib"], None, None),
    ]

    md.append("| Configuration | `Q-LUT` (KiB) | Index Table (`GRID_MAP`/`INTERP`) | BN-Folded Weights (KiB) | **Static Total (KiB / % of 2 MB L2)** | Peak Live Activations (`head.block1`, 3-buf) | **Static + Peak Live Total (KiB / % of 2 MB L2)** |")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    for name, lk, ik, wk, pk, _ in modes_mem:
        static_tot = lk + ik + wk
        if pk is not None:
            full_tot = static_tot + pk
            pk_s = f"{pk:.2f} KiB"
            ft_s = f"**{full_tot:.2f} KiB** ({_pct_l2(full_tot):.2f}%)"
        else:
            pk_s = "*(smaller v3 head)*"
            ft_s = f"≥ {static_tot:.2f} KiB ({_pct_l2(static_tot):.2f}%)"
        md.append(
            f"| {name} | {lk:.2f} | {ik:.2f} | {wk:.2f} | **{static_tot:.2f} KiB** ({_pct_l2(static_tot):.2f}%) | {pk_s} | {ft_s} |"
        )

    md.append("")
    md.append("### 3.3. Detailed Breakdown of `medquanv_cli.rs` Runtime Buffers (`WideGRQuanvNet v2`, Batch=1)")
    md.append("")
    md.append("| Buffer Group in `medquanv_cli.rs` | Exact Bytes | Size (KiB) |")
    md.append("| :--- | :---: | :---: |")
    for label, nbytes in stack_arrays:
        md.append(f"| {label} | {nbytes:,} B | {_kib(nbytes):.2f} KiB |")
    md.append(f"| **Sum of all local stack arrays (unoptimized reuse in `forward_single`)** | **{total_stack_b:,} B** | **{stack_kib:.2f} KiB** |")
    md.append(f"| **Exact minimum live activation peak (`head.block1`: `h_stem + b1_h + b1_out`)** | **{peak_live_3buf_b:,} B** | **{peak3_kib:.2f} KiB** |")
    md.append(f"| **Minimal 2-buffer ping-pong peak (in-place residual add)** | **{peak_live_2buf_b:,} B** | **{peak2_kib:.2f} KiB** |")
    md.append("")
    md.append("### 3.4. Single Authoritative Memory Statement for Thesis Text")
    md.append("")
    static_i8 = lut_i8_kib + w_kib
    live_i8 = static_i8 + peak3_kib
    full_stack_i8 = static_i8 + stack_kib
    static_f32 = lut_f32_kib + interp_kib + w_kib
    live_f32 = static_f32 + peak3_kib
    md.append(
        f"- **Static model + table footprint (`WideGRQuanvNet v2`, `INT8 Shift-LUT`):** **`{static_i8:.2f} KiB`** "
        f"(`{lut_i8_kib:.2f} KiB` Q-LUT + `0.00 KiB` index table + `{w_kib:.2f} KiB` BN-folded FP32 weights; "
        f"`{_pct_l2(static_i8):.2f}%` of a `2,048 KiB` L2 cache). Including peak live activation tensors "
        f"(`{peak3_kib:.2f} KiB` across `head.block1`'s three `48×14×14` FP32 tensors), total active working set is "
        f"**`{live_i8:.2f} KiB`** (`{_pct_l2(live_i8):.2f}%` of `2,048 KiB`), or **`{full_stack_i8:.2f} KiB`** "
        f"(`{_pct_l2(full_stack_i8):.2f}%`) if summing every non-reused local stack array (`{stack_kib:.2f} KiB`) "
        f"in the current `extract_multiscale_qlut()` + `forward_single()` implementation."
    )
    md.append(
        f"- **Static model + table footprint (`WideGRQuanvNet v2`, `Interp4D`):** **`{static_f32:.2f} KiB`** "
        f"(`{lut_f32_kib:.2f} KiB` FP32 LUT + `{interp_kib:.2f} KiB` `INTERP_TABLE` + `{w_kib:.2f} KiB` BN-folded weights; "
        f"`{_pct_l2(static_f32):.2f}%` of `2,048 KiB`), rising to **`{live_f32:.2f} KiB`** (`{_pct_l2(live_f32):.2f}%`) "
        f"with `{peak3_kib:.2f} KiB` peak live activations."
    )
    md.append("")

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    print(f"[Saved] {OUT_MD}")


if __name__ == "__main__":
    main()
