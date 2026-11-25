# Audit Report: 4-bit Input Quantization Cost (`INT8` vs `Interp4D` vs `Conv2d`) & Unified Memory Accounting

> **Scope & Measurement Environment:** All latency measurements below were executed natively on **x86_64 host CPU** (not extrapolated to Jetson Nano). Previous Jetson Nano FPS extrapolation columns (`×0.912`) and PennyLane Python-simulator speedup ratios are intentionally excluded. Memory percentages relative to `2,048 KiB` (2 MB) are purely architectural reference ratios against the Cortex-A57 L2 cache size.

## 1. Classification Accuracy Cost of 4-bit Input Discretization (`INT8 Shift` vs `Interp4D`)

Discretizing 8-bit medical pixel intensities into 4-bit bins (`16` levels per channel, discarding the 4 least-significant bits, step size `6.67%`) removes fine soft-tissue micro-contrast. Below is the exact zero-retraining accuracy comparison across all 3 MedMNIST datasets (5 paired seeds `[0, 42, 100, 2023, 777]`, 3 speckle noise draws; source: `reports/pareto_int8_vs_interp4d.json` and `reports/rigorous_stats_summary.json`):

| Dataset | $\sigma$ | `INT8 Shift` 256 KiB (ACC %) | `INT8 Grid` 256 KiB (ACC %) | `Interp4D` 1,024 KiB (ACC %) | $\Delta$ (`INT8 Shift` − `Interp4D`) | 95% CI | $p_{\text{raw}}$ | $p_{\text{adj,fam}}$ | TOST ($\pm 2\%$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **breastmnist** | `0.00` | 84.49 ± 2.16 | 84.74 ± 2.12 | 84.74 ± 1.79 | **-0.26%** | [-1.81, +1.30] | 0.6702 | 0.6702 | EQUIV (p=0.018) |
| **breastmnist** | `0.05` | 84.06 ± 2.14 | 84.40 ± 1.68 | 84.49 ± 1.74 | **-0.43%** | [-1.35, +0.49] | 0.2663 | 0.5958 | EQUIV (p=0.004) |
| **breastmnist** | `0.10` | 83.42 ± 2.51 | 83.63 ± 2.40 | 83.50 ± 2.56 | **-0.09%** | [-0.39, +0.22] | 0.4766 | 0.5958 | EQUIV (p=0.000) |
| **breastmnist** | `0.15` | 82.74 ± 2.07 | 82.69 ± 1.97 | 82.61 ± 1.90 | **+0.13%** | [-0.27, +0.53] | 0.4263 | 0.5958 | EQUIV (p=0.000) |
| **breastmnist** | `0.20` | 80.34 ± 2.22 | 80.81 ± 2.63 | 80.73 ± 2.51 | **-0.38%** | [-0.86, +0.09] | 0.0876 | 0.4382 | EQUIV (p=0.000) |
| **pneumoniamnist** | `0.00` | 88.49 ± 1.50 | 88.46 ± 1.54 | 88.94 ± 1.63 | **-0.45%** | [-0.86, -0.04] | 0.0378 | 0.0465 | EQUIV (p=0.000) |
| **pneumoniamnist** | `0.05` | 89.59 ± 1.16 | 89.71 ± 1.19 | 89.96 ± 1.12 | **-0.36%** | [-0.46, -0.26] | 0.0005 | 0.0026 | EQUIV (p=0.000) |
| **pneumoniamnist** | `0.10` | 82.31 ± 4.56 | 82.04 ± 4.45 | 83.72 ± 3.84 | **-1.41%** | [-2.64, -0.18] | 0.0333 | 0.0465 | DIFF (p=0.127) |
| **pneumoniamnist** | `0.15` | 75.83 ± 6.84 | 75.54 ± 7.25 | 77.30 ± 6.50 | **-1.46%** | [-2.29, -0.64] | 0.0080 | 0.0200 | DIFF (p=0.073) |
| **pneumoniamnist** | `0.20` | 74.88 ± 6.82 | 74.73 ± 6.95 | 76.62 ± 5.90 | **-1.74%** | [-3.44, -0.04] | 0.0465 | 0.0465 | DIFF (p=0.347) |
| **octmnist** | `0.00` | 80.64 ± 2.37 | 81.22 ± 1.50 | 84.04 ± 2.45 | **-3.40%** | [-3.93, -2.87] | 0.0001 | 0.0003 | DIFF (p=0.999) |
| **octmnist** | `0.05` | 72.73 ± 1.59 | 73.54 ± 0.96 | 77.42 ± 1.11 | **-4.69%** | [-5.74, -3.65] | 0.0002 | 0.0006 | DIFF (p=0.999) |
| **octmnist** | `0.10` | 55.40 ± 2.07 | 55.08 ± 1.89 | 57.93 ± 1.58 | **-2.53%** | [-3.99, -1.07] | 0.0086 | 0.0143 | DIFF (p=0.814) |
| **octmnist** | `0.15` | 44.83 ± 2.49 | 44.52 ± 2.55 | 45.90 ± 2.21 | **-1.07%** | [-2.36, +0.21] | 0.0810 | 0.1012 | INCONCL. (p=0.058) |
| **octmnist** | `0.20` | 40.19 ± 2.32 | 40.12 ± 2.58 | 40.59 ± 2.17 | **-0.40%** | [-1.06, +0.26] | 0.1656 | 0.1656 | EQUIV (p=0.001) |

### Symmetric Stabilized Stem (`ablation_clut`, 5 Seeds: Q-LUT `INT8 Shift` vs `Interp4D`)

| Dataset | Stem | $\sigma$ | `INT8 Shift` (ACC %) | `Interp4D` (ACC %) | $\Delta$ (`INT8` − `Interp4D`) | 95% CI | $p_{\text{raw}}$ | $p_{\text{adj,fam}}$ | TOST ($\pm 2\%$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **pneumoniamnist** | `stabilized_stem` | `0.00` | 88.49 ± 1.50 | 88.94 ± 1.63 | **-0.45%** | [-0.86, -0.04] | 0.0378 | 0.1035 | EQUIV (p=0.000) |
| **pneumoniamnist** | `stabilized_stem` | `0.05` | 87.72 ± 1.48 | 88.02 ± 1.34 | **-0.30%** | [-0.76, +0.16] | 0.1444 | 0.2406 | EQUIV (p=0.000) |
| **pneumoniamnist** | `stabilized_stem` | `0.10` | 85.52 ± 1.36 | 85.80 ± 1.28 | **-0.28%** | [-0.81, +0.25] | 0.2189 | 0.2736 | EQUIV (p=0.000) |
| **pneumoniamnist** | `stabilized_stem` | `0.15` | 84.68 ± 1.43 | 84.78 ± 1.49 | **-0.10%** | [-0.37, +0.17] | 0.3804 | 0.3804 | EQUIV (p=0.000) |
| **pneumoniamnist** | `stabilized_stem` | `0.20` | 83.03 ± 0.91 | 83.39 ± 0.90 | **-0.35%** | [-0.68, -0.02] | 0.0414 | 0.1035 | EQUIV (p=0.000) |
| **pneumoniamnist** | `raw_stem` | `0.00` | 88.49 ± 1.50 | 88.94 ± 1.63 | **-0.45%** | [-0.86, -0.04] | 0.0378 | 0.0946 | EQUIV (p=0.000) |
| **pneumoniamnist** | `raw_stem` | `0.05` | 90.03 ± 1.00 | 90.37 ± 1.08 | **-0.34%** | [-0.58, -0.10] | 0.0161 | 0.0807 | EQUIV (p=0.000) |
| **pneumoniamnist** | `raw_stem` | `0.10` | 70.71 ± 8.37 | 71.29 ± 8.07 | **-0.59%** | [-1.29, +0.12] | 0.0812 | 0.1354 | EQUIV (p=0.003) |
| **pneumoniamnist** | `raw_stem` | `0.15` | 46.21 ± 3.48 | 46.12 ± 3.50 | **+0.09%** | [-0.32, +0.49] | 0.5870 | 0.6885 | EQUIV (p=0.000) |
| **pneumoniamnist** | `raw_stem` | `0.20` | 39.51 ± 0.60 | 39.49 ± 0.68 | **+0.02%** | [-0.12, +0.16] | 0.6885 | 0.6885 | EQUIV (p=0.000) |
| **breastmnist** | `stabilized_stem` | `0.00` | 84.49 ± 2.16 | 84.74 ± 1.79 | **-0.26%** | [-1.81, +1.30] | 0.6702 | 0.6702 | EQUIV (p=0.018) |
| **breastmnist** | `stabilized_stem` | `0.05` | 84.19 ± 2.20 | 84.53 ± 1.73 | **-0.34%** | [-1.29, +0.61] | 0.3739 | 0.6232 | EQUIV (p=0.004) |
| **breastmnist** | `stabilized_stem` | `0.10` | 84.23 ± 2.48 | 85.00 ± 2.80 | **-0.77%** | [-1.62, +0.08] | 0.0662 | 0.3309 | EQUIV (p=0.008) |
| **breastmnist** | `stabilized_stem` | `0.15` | 83.08 ± 1.77 | 82.95 ± 1.65 | **+0.13%** | [-0.42, +0.68] | 0.5529 | 0.6702 | EQUIV (p=0.000) |
| **breastmnist** | `stabilized_stem` | `0.20` | 81.97 ± 2.08 | 82.14 ± 1.98 | **-0.17%** | [-0.52, +0.17] | 0.2420 | 0.6050 | EQUIV (p=0.000) |
| **breastmnist** | `raw_stem` | `0.00` | 84.49 ± 2.16 | 84.74 ± 1.79 | **-0.26%** | [-1.81, +1.30] | 0.6702 | 0.8377 | EQUIV (p=0.018) |
| **breastmnist** | `raw_stem` | `0.05` | 83.85 ± 2.14 | 84.02 ± 1.63 | **-0.17%** | [-1.04, +0.70] | 0.6135 | 0.8377 | EQUIV (p=0.002) |
| **breastmnist** | `raw_stem` | `0.10` | 81.92 ± 2.85 | 81.88 ± 2.78 | **+0.04%** | [-0.59, +0.68] | 0.8605 | 0.8605 | EQUIV (p=0.001) |
| **breastmnist** | `raw_stem` | `0.15` | 79.40 ± 1.70 | 79.53 ± 1.70 | **-0.13%** | [-0.37, +0.11] | 0.2080 | 0.8154 | EQUIV (p=0.000) |
| **breastmnist** | `raw_stem` | `0.20` | 76.84 ± 1.54 | 77.05 ± 1.46 | **-0.21%** | [-0.74, +0.32] | 0.3262 | 0.8154 | EQUIV (p=0.000) |

**Key findings on 4-bit quantization cost:**
1. **Zero-retraining degradation is dataset-dependent and non-trivial on fine-contrast retinal OCT:** On **OCTMNIST** (`σ=0.00`, `0.05`, `0.10`), zero-shot `INT8 Shift` drops **−3.40%**, **−4.69%**, and **−2.53%** relative to `Interp4D`, failing TOST equivalence (`±2%`). On **PneumoniaMNIST**, the drop is smaller (**−0.45%** clean, **−1.41% to −1.74%** under $\sigma \ge 0.10$, statistically significant on `p_raw` under `pre_lut_damped`). On **BreastMNIST**, `INT8 Shift` and `Interp4D` stay within **±0.40%** across all noise levels and pass TOST `±2%` equivalence at all $\sigma$.
2. **Clarification for Table 4.2:** Table 4.2 (`reports/ablation_isolation_results.json`) used **`Interp4D` (1,024 KiB)** exclusively, NOT `INT8` (256 KiB). Mixing `Interp4D` accuracy with `INT8` footprint/latency is invalid and is strictly separated here.
3. **Recoverability via QAT (`v3`):** Training with Straight-Through Estimator `INT8-QAT` (`quanv_v3_int8`, `reports/pillar3_v3_results.json`) restores clean OCTMNIST `INT8` accuracy from `80.64%` (`v2` zero-shot) to `82.90%` (`0.0000%` compilation mismatch vs deployed `256 KiB` table), though it still trails `ctrl_v3_light` at OCTMNIST $\sigma=0.10$ (`51.23%` vs `57.32%`).

## 2. Honest Latency Comparison: `rust_qlut` (`INT8` & `Interp4D`) vs. `PyTorch Conv2d` (x86_64 Host CPU)

Measured on x86_64 host CPU (`reports/highres_scalability_results.json`, single 4-channel $2\times 2$ layer). Ratios `< 1.0×` explicitly indicate that `rust_qlut` is **slower** than `PyTorch Conv2d`:

| Resolution ($H\times W$, stride) | Patches ($N_p$) | `INT8 Shift` (256 KiB) | `Interp4D` (1,024 KiB) | `PyTorch Conv2d` CPU | Ratio `Conv2d / INT8` | Ratio `Conv2d / Interp4D` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `28x28 (s=2)` | 196 | 0.49 µs (0.0005 ms) | 4.93 µs (0.0049 ms) | 54.45 µs (0.0544 ms) | **111.80× faster** | 11.05× faster |
| `64x64 (s=2)` | 1,024 | 1.34 µs (0.0013 ms) | 25.55 µs (0.0256 ms) | 56.80 µs (0.0568 ms) | **42.23× faster** | 2.22× faster |
| `128x128 (s=2)` | 4,096 | 4.35 µs (0.0043 ms) | 102.62 µs (0.1026 ms) | 61.36 µs (0.0614 ms) | **14.11× faster** | 0.60× (**1.67× slower**) |
| `224x224 (s=2)` | 12,544 | 11.91 µs (0.0119 ms) | 285.46 µs (0.2855 ms) | 158.58 µs (0.1586 ms) | **13.32× faster** | 0.56× (**1.80× slower**) |
| `512x512 (s=2)` | 65,536 | 61.55 µs (0.0616 ms) | 1619.72 µs (1.6197 ms) | 250.10 µs (0.2501 ms) | **4.06× faster** | 0.15× (**6.48× slower**) |
| `1024x1024 (s=2)` | 262,144 | 271.79 µs (0.2718 ms) | 6473.90 µs (6.4739 ms) | 1329.98 µs (1.3300 ms) | **4.89× faster** | 0.21× (**4.87× slower**) |
| `1024x1024 (s=4)` | 65,536 | 66.25 µs (0.0662 ms) | 1637.05 µs (1.6370 ms) | 270.42 µs (0.2704 ms) | **4.08× faster** | 0.17× (**6.05× slower**) |

### Full-Network End-to-End Latency on x86_64 Host CPU (`28×28`, Batch=1)

| Implementation & Model | LUT Mode | LUT Latency (4 scales) | End-to-End Latency | Throughput (x86_64) | Max Logit Diff vs PyTorch |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `medquanv-cli` v2 (`WideGRQuanvNet`, 148.3K, unblocked) | `INT8 Shift` | 1.79 µs | 6937.0 µs (6.94 ms) | 144.2 FPS | `3.81e-06` |
| `medquanv-cli` v2 (`WideGRQuanvNet`, 148.3K, unblocked) | `Interp4D` | 20.08 µs | 6702.2 µs (6.70 ms) | 149.2 FPS | `4.29e-06` |
| `medquanv-cli` v2 (`WideGRQuanvNet`, 148.3K, A3 opt) | `INT8 Shift` | 1.14 µs | 2961.1 µs (2.96 ms) | 337.7 FPS | `3.81e-06` |
| `medquanv-cli` v2 (`WideGRQuanvNet`, 148.3K, A3 opt) | `Interp4D` | 5.50 µs | 2979.7 µs (2.98 ms) | 335.6 FPS | `4.29e-06` |
| `medquanv-cli --suite-v3`: `ctrl_v3_light` (39.5K) | None (`Conv2d`) | N/A | 1114.0 µs (1.11 ms) | 897.6 FPS | `5.25e-06` |
| `medquanv-cli --suite-v3`: `quanv_v3_int8` (39.3K) | `INT8 Shift` | 3.16 µs | 1126.4 µs (1.13 ms) | 887.8 FPS | `5.25e-06` |
| `medquanv-cli --suite-v3`: `micro_ctrl` (2.08K) | None (`Conv2d`) | N/A | 47.59 µs | 21,014.2 FPS | `3.81e-06` |
| `medquanv-cli --suite-v3`: `micro_quanv` (2.00K) | `INT8 Shift` | 3.16 µs | 46.17 µs | 21,656.7 FPS | `1.91e-06` |

**Key latency observations:**
- At clinical high resolution (`1024×1024`, $s=2$), **`INT8 Shift` is `4.89×` faster** than `PyTorch Conv2d` (`0.2718 ms` vs `1.3300 ms`), whereas **`Interp4D` (the accuracy-preserving variant) is `4.87×` slower** (`6.4739 ms` vs `1.3300 ms`) due to 16-corner gather + 15 FMA interpolation ops per patch.
- At full-network End-to-End level (`28×28`, `v3`), `quanv_v3_int8` (`1.126 ms`) and `ctrl_v3_light` (`1.114 ms`) have effectively identical latency (`+1.1%` overhead for `quanv_v3_int8`), while `quanv_v3_int8` requires `+256 KiB` extra static memory for the LUT.
- **Jetson Nano status:** No physical Jetson Nano hardware measurement has been performed yet; all FPS numbers above are strictly x86_64 host CPU measurements.

## 3. Unified Memory Footprint Audit (Resolving `833.26` / `835.32` / `836` / `891.32 KiB` Discrepancies)

### 3.1. Root Cause of Previous Discrepancies

| Legacy Figure | Where It Appeared | Exact Formula Used | Why It Differed |
| :---: | :--- | :--- | :--- |
| **`833.26 KiB`** | `medquanv_cli_verification*.json`, `TECHNICAL_REPORT_EDGE_QUANV.md` §2 | `256.00` (`LUT_INT8_SHIFT`) + `577.26` (`147,778` BN-folded `f32`) | **Correct static ROM/weight footprint for `INT8 Shift`** (excludes runtime stack activations). |
| **`835.32 KiB`** | `reports/highres_scalability_results.json` | `256.00` (`LUT`) + `579.32` (`148,306` **unfolded** PyTorch `f32`) | Used pre-folding PyTorch parameter count (`+528` BN parameters = `+2.06 KiB`). |
| **`836 KiB`** | Legacy `medquanv-cli` v0.2.0 console telemetry | Rounded `835.32 KiB` up to integer | Same unfolded weight count + integer rounding. |
| **`891.32 KiB`** | Legacy `d90ed19` §2 table & ASCII diagram | `256.00` + `579.32` (unfolded) + `8.00` (`GRID_MAP`) + `48.00` (est. runtime buf) | Mixed unfolded weights with `GRID_MAP` (unused by `INT8 Shift`) and a rough `48 KiB` estimate for activations; ASCII diagram merged `8 + 48 = 56 KiB` under `Runtime Buf`. |

*(Minor code-comment bug identified in `rust_qlut/src/lib.rs:78`: the comment claims `GRID_MAP` is `2 KiB`, whereas on 64-bit x86_64/aarch64 `sizeof(usize) = 8`, so `4 × 256 × 8 = 8,192 bytes = 8.00 KiB`. Likewise `INTERP_TABLE` is `4 × 256 × 8 + 256 × 4 = 9,216 bytes = 9.00 KiB`.)*

### 3.2. Authoritative Byte-Exact Memory Table (`WideGRQuanvNet v2`, Batch=1, `28×28`, 64-bit Target)

Verified directly from on-disk binary weights (`weights/wide_v2_pneumoniamnist_s42_rust.bin = 591,128 B = 16 B header + 591,112 B payload`) and Rust source structs (`rust_qlut/src/lib.rs`, `rust_qlut/src/bin/medquanv_cli.rs`):

| Configuration | `Q-LUT` (KiB) | Index Table (`GRID_MAP`/`INTERP`) | BN-Folded Weights (KiB) | **Static Total (KiB / % of 2 MB L2)** | Peak Live Activations (`head.block1`, 3-buf) | **Static + Peak Live Total (KiB / % of 2 MB L2)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| WideGRQuanvNet v2 — `INT8 Shift-LUT` | 256.00 | 0.00 | 577.26 | **833.26 KiB** (40.69%) | 110.25 KiB | **943.51 KiB** (46.07%) |
| WideGRQuanvNet v2 — `INT8 Grid-LUT` | 256.00 | 8.00 | 577.26 | **841.26 KiB** (41.08%) | 110.25 KiB | **951.51 KiB** (46.46%) |
| WideGRQuanvNet v2 — `Interp4D SIMD` | 1024.00 | 9.00 | 577.26 | **1610.26 KiB** (78.63%) | 110.25 KiB | **1720.51 KiB** (84.01%) |
| `WideClassicalControl` v2 (0% LUT) | 0.00 | 0.00 | 577.65 | **577.65 KiB** (28.21%) | 110.25 KiB | **687.90 KiB** (33.59%) |
| `quanv_v3_int8` (v3, `INT8 Shift`) | 256.00 | 0.00 | 150.90 | **406.90 KiB** (19.87%) | *(smaller v3 head)* | ≥ 406.90 KiB (19.87%) |
| `ctrl_v3_light` (v3, 0% LUT) | 0.00 | 0.00 | 151.79 | **151.79 KiB** (7.41%) | *(smaller v3 head)* | ≥ 151.79 KiB (7.41%) |
| `micro_quanv` (v3, `INT8 Shift`) | 256.00 | 0.00 | 7.29 | **263.29 KiB** (12.86%) | *(smaller v3 head)* | ≥ 263.29 KiB (12.86%) |
| `micro_ctrl` (v3, 0% LUT) | 0.00 | 0.00 | 7.68 | **7.68 KiB** (0.38%) | *(smaller v3 head)* | ≥ 7.68 KiB (0.38%) |

### 3.3. Detailed Breakdown of `medquanv_cli.rs` Runtime Buffers (`WideGRQuanvNet v2`, Batch=1)

| Buffer Group in `medquanv_cli.rs` | Exact Bytes | Size (KiB) |
| :--- | :---: | :---: |
| extract_multiscale_qlut: img_d2/d3/ax (3 × 28×28 u8) | 2,352 B | 2.30 KiB |
| extract_multiscale_qlut: tmp_i8 (784 i8, INT8 only) + q_raw (16×196 f32) | 13,328 B | 13.02 KiB |
| forward_single: x_01 + x_norm (2 × 784 f32) | 6,272 B | 6.12 KiB |
| forward_single: f_c + q_bn + f_q + alpha (4 × 16×196 f32) | 50,176 B | 49.00 KiB |
| forward_single: q_h1 + q_h2 + cat_buf + g_in + g_h1 (5 × 32×196 f32) | 125,440 B | 122.50 KiB |
| forward_single: h_stem + b1_h + b1_out (3 × 48×196 f32) | 112,896 B | 110.25 KiB |
| forward_single: p1 (48×49 f32) + b2_h + b2_out + b2_skip (3 × 64×49 f32) + gap (64 f32) | 47,296 B | 46.19 KiB |
| **Sum of all local stack arrays (unoptimized reuse in `forward_single`)** | **357,760 B** | **349.38 KiB** |
| **Exact minimum live activation peak (`head.block1`: `h_stem + b1_h + b1_out`)** | **112,896 B** | **110.25 KiB** |
| **Minimal 2-buffer ping-pong peak (in-place residual add)** | **75,264 B** | **73.50 KiB** |

### 3.4. Single Authoritative Memory Statement for Thesis Text

- **Static model + table footprint (`WideGRQuanvNet v2`, `INT8 Shift-LUT`):** **`833.26 KiB`** (`256.00 KiB` Q-LUT + `0.00 KiB` index table + `577.26 KiB` BN-folded FP32 weights; `40.69%` of a `2,048 KiB` L2 cache). Including peak live activation tensors (`110.25 KiB` across `head.block1`'s three `48×14×14` FP32 tensors), total active working set is **`943.51 KiB`** (`46.07%` of `2,048 KiB`), or **`1182.63 KiB`** (`57.75%`) if summing every non-reused local stack array (`349.38 KiB`) in the current `extract_multiscale_qlut()` + `forward_single()` implementation.
- **Static model + table footprint (`WideGRQuanvNet v2`, `Interp4D`):** **`1610.26 KiB`** (`1024.00 KiB` FP32 LUT + `9.00 KiB` `INTERP_TABLE` + `577.26 KiB` BN-folded weights; `78.63%` of `2,048 KiB`), rising to **`1720.51 KiB`** (`84.01%`) with `110.25 KiB` peak live activations.

