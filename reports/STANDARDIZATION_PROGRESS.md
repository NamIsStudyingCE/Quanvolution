# Standardization & Verification Progress Tracker

**Updated at**: 2026-11-13T17:38:00+07:00
**Constraint**: `weights/pillar2_wide/` (30 `v2` checkpoints) stay 100% LOCKED | New `v3` checkpoints in `weights/pillar3_v3/` | Zero Fabrication

## Phase A/B Execution Plan (Post-Standardization Optimization)
- [DONE] **Phase A (Zero Retraining on 30 locked `v2` checkpoints -> INSUFFICIENT, TRIGGERED PHASE B)**:
  - `Step A1` (`reports/threshold_a_results.json`):
    - Putting `x_damped` symmetrically into `c_stem` + `q_ms` (`grquanv_sym_damped`) fixed the PneumoniaMNIST collapse (`76.63% -> 83.39%` at $\sigma=0.20$, beating `ctrl_damped` `82.68%`), and BreastMNIST beat `ctrl_damped` across all 5 sigmas (`84.74%..82.14%` vs `84.10%..81.58%`). Adding `clamp_cweight` pushed PneumoniaMNIST $\sigma=0.20$ to `84.25 +- 0.82%`, OCTMNIST $\sigma=0.00..0.20$ to `84.50%..41.15%`. HOWEVER, PneumoniaMNIST $\sigma=0.00..0.15$ still lagged `ctrl_damped` (`89.46% / 88.71% / 86.81% / 85.93%` vs `90.22% / 89.48% / 87.79% / 86.65%`). And changing gating scale without retraining (`varpres_sqrt2`) was unstable across datasets (`92.50%` on Pneumonia clean, but crashed to `78.36%` on OCTMNIST and `81.79%` on BreastMNIST).
  - `Step A2` (`reports/threshold_a_results.json`):
    - Verified $16 \times 16$ Hermitian bilinear quantum matrix $M_k(\theta)$ against PennyLane `strongly_L2` (`max|diff| = 8.88e-16`), and compiled exact 16-level bin-expected MMSE `INT8 Shift-LUT (256 KiB)`. Because `export_bin.py` already used `(16b + 7.5)/255` bin midpoints (`exact_match = 90.42%`, `mean|diff| = 0.096 lsb`), MMSE-16 only improved OCTMNIST clean from `80.64% -> 80.82%` (still below `Control 82.52%` and `Interp4D 84.00%`) and PneumoniaMNIST `88.40%` vs `Control 90.22%`.
  - `Step A3` (`reports/medquanv_cli_verification_a3.json`):
    - Removed inner-loop bounds check in `conv3x3_nchw` + compiled with `-C target-cpu=native`: End-to-End `medquanv-cli` latency dropped **`2.38x`** (`6.94 ms -> 2.91-2.98 ms`, **`336-343 FPS`**) and `Interp4D Q-LUT(4x)` dropped `20.08 us -> 5.50 us`, while `MaxLogitDiff` remained `3.81e-6`. Still did not surpass Classical CNN's architectural FLOP count (`~15.5M`).
  - *Decision Gate Verdict*: **Ngưỡng A chưa đủ** để bản `INT8 256 KiB` thắng `Classical Control` trên OCTMNIST/PneumoniaMNIST hay vượt hẳn về FLOPs/tốc độ End-to-End $\implies$ **Mở khóa Phase B (`GR-Quanv v3`)**.
- [DONE] **Phase B (`Trainable Variational Q-LUT v3` — Separate checkpoints in `weights/pillar3_v3/`)**:
  - `Step B1` (**`[DONE]`**): Finalized `src/models/gr_quanv_v3.py` with exact differentiable PyTorch `SU(16)` `StronglyEntanglingLayers` + $M_k(\theta)$ (`diff = 5.43e-7` vs PennyLane), QAT 4-bit shift + INT8 STE (`compile_max_diff = 0.00e+00` vs compiled `256 KiB` LUT), zero-overhead 4D TV circuit smoothness regularizer, noise-isolated Stage-2 Q-LUT reading strictly from $w_q f_q$, `_dw_dilated_mixer(16)`, symmetric `ClassicalControlV3Light` (`4.07M` MACs, `39,470` params vs `GRQuanvNetV3` `4.03M` MACs, `39,250` params), and Micro-Edge `MicroEdgeNetV3` (`0.169M` vs `0.186M` MACs, ~`2K` params).
  - `Step B2` (**`[DONE]`**): Executed `experiments/run_pillar3_v3.py` across all 3 datasets (BreastMNIST, PneumoniaMNIST, OCTMNIST) x 5 paired seeds (`[0, 42, 100, 2023, 777]`) x 5 models (`ctrl_v2_full`, `ctrl_v3_light`, `quanv_v3_int8`, `micro_ctrl`, `micro_quanv`) with `compile_acc_diff = 0.0` across all 30 quantum runs (`reports/pillar3_v3_results.json`, `reports/figures/v3_vs_cnn_pareto.png`).
  - `Step B3` (**`[DONE]`**): Compiled learned circuits to raw `.bin` INT8 Q-LUT, folded/exported all 5 PneumoniaMNIST `s42` models via `experiments/export_v3_to_rust.py`, verified 100.00% Rust/PyTorch agreement on 624 real test images (`medquanv-cli --suite-v3`, `reports/medquanv_cli_v3_verification.json`).
  - `Step B4` (**`[DONE]`**): Updated `reports/TECHNICAL_REPORT_EDGE_QUANV.md` (Sections 7–9), verified progress logs, and committed cleanly to git.

## Task Checklist
- [DONE] **Task 1**: Tune `phase_damping_u8` (`src/models/gr_quanv.py`) with quadratic soft-start ramp.
  - Clean image pixel modification ($\sigma=0.00$): `0.0000%` across BreastMNIST, PneumoniaMNIST, OCTMNIST.
  - PneumoniaMNIST $\sigma=0.05$: improved from `87.42%` to **`89.96 +- 1.12%`** (`eval_speckle_wide.py` & `eval_ablation_isolation.py`).
  - Raw comparison (no damping, no var-stab) at $\sigma=0.10$: `3_grquanv_raw` = `71.30%` vs `1_ctrl_raw` = `60.60%` (`+10.70%` ACC, `+21.05%` F1).
  - Crucial honest ablation finding: when Classical Control ALSO receives full image-level `phase_damping_u8` (`2_ctrl_damped`: `86.65%` @ 0.15, `82.68%` @ 0.20), it outperforms `4_grquanv_damped` where `c_stem` received un-damped `x_raw_t` (`77.30%` @ 0.15, `76.62%` @ 0.20). Must report transparently!
  - Outputs: `reports/pillar2_wide_speckle_results.json`, `reports/pillar2_wide_results.json`, `reports/ablation_isolation_results.json`.
- [DONE] **Task 2**: Strictly separated `INT8 Shift/Grid-LUT (256 KiB, 0-FLOP Q-LUT)` vs `Interp4D SIMD (1,024 KiB, FP32)` vs `Wide Classical Control` across all 3 datasets, 5 seeds, 5 sigmas, 3 noise draws.
  - CPU PyTorch benchmark (`pareto_int8_vs_interp4d.json`): single-scale Q-LUT latency `INT8 Shift = 0.967 us` vs `Interp4D = 6.298 us` (`6.5x` faster at Q-LUT layer); however PyTorch End-to-End B=1 (`8.12 ms` vs `8.27 ms` vs `9.17 ms` Ctrl) and B=64/img (`0.826 ms` vs `1.057 ms` vs `0.828 ms` Ctrl) show that Python/Conv head dominates PyTorch CPU runtime.
  - Clean ACC (PneumoniaMNIST / BreastMNIST / OCTMNIST):
    - `Wide Classical Control`: `90.22%` / `84.10%` / `82.52%`
    - `INT8 Shift (256 KiB)`: `88.49%` / `84.49%` / `80.64%`
    - `INT8 Grid (256 KiB)`: `88.46%` / `84.74%` / `81.22%`
    - `Interp4D (1,024 KiB)`: `88.94%` / `84.74%` / `84.04%`
  - PneumoniaMNIST Speckle $\sigma=0.10..0.20$: `INT8 Shift` retains `82.31% / 75.83% / 74.88%` vs `Interp4D` `83.72% / 77.30% / 76.62%` vs `Control` `63.84% / 52.43% / 51.75%`. Note honest quantization drop: on OCTMNIST clean (`-1.88%`) and $\sigma=0.05$ (`-3.74%`), zero-shot `INT8` without QAT drops accuracy vs `Interp4D`/`Control`.
  - Outputs: `reports/pareto_int8_vs_interp4d.json`, `reports/figures/pareto_int8_vs_interp4d.png`.
- [DONE] **Task 3**: Exported all 148,306 trainable params (147,778 BN-folded `f32` floats = `577.26 KiB`, `591,128` bytes total) from `weights/pillar2_wide/wide_v2_interp4d_pneumoniamnist_full_grquanv_s42.pt` to `weights/wide_v2_pneumoniamnist_s42_rust.bin` and `weights/wide_v2_pneumoniamnist_s42_folded.bin`, plus 624 real PneumoniaMNIST test images + PyTorch reference logits to `data/processed/pneumoniamnist_test_rust_verify.bin`. Implemented full pure-Rust `WideGRQuanvNetV2` inside `rust_qlut/src/bin/medquanv_cli.rs` (0 Python/PyTorch/CUDA dependencies), compiled & executed (`reports/medquanv_cli_verification.json`):
  - `INT8 Shift-LUT (256 KiB, 0-FLOP)`: Rust ACC **`89.42%`** == PyTorch **`89.42%`**, `MaxLogitDiff = 3.81e-6`, `Q-LUT (4x) = 1.79 us`, `Full-Net E2E = 6,937.00 us` (`144.2 FPS`), Total L2 = `833.3 KiB` (`40.7%` of 2 MB L2).
  - `INT8 Grid-LUT (256 KiB)`: Rust ACC **`89.42%`** == PyTorch **`89.42%`**, `MaxLogitDiff = 4.77e-6`, `Q-LUT (4x) = 1.71 us`, `Full-Net E2E = 6,730.87 us` (`148.6 FPS`).
  - `Interp4D SIMD (1,024 KiB, FP32)`: Rust ACC **`90.22%`** == PyTorch **`90.22%`**, `MaxLogitDiff = 4.29e-6`, `Q-LUT (4x) = 20.08 us`, `Full-Net E2E = 6,702.22 us` (`149.2 FPS`), Total L2 = `1,601.3 KiB` (`78.2%` of 2 MB L2).
- [DONE] **Task 4**: Standardized `reports/TECHNICAL_REPORT_EDGE_QUANV.md` with strictly separated tables, real End-to-End telemetry, transparent negative findings, and concrete improvement roadmap.


---

## PHASE B: GR-QUANV V3 (TRAINABLE VARIATIONAL Q-LUT + INT8 QAT + LOW-FLOP ARCHITECTURE)
- [DONE] **Step B1**: Finalized & verified `src/models/gr_quanv_v3.py` (`verify_unitary_vs_pennylane = 5.4347e-7`, `compile_max_diff = 0.0e+00`) across `GRQuanvNetV3` (`4.03M` MACs, `39,250` params), `ClassicalControlV3Light` (`4.07M` MACs, `39,470` params), and `MicroEdgeNetV3` (`0.169M` / `0.186M` MACs, `~2K` params).
- [DONE] **Step B2a (BreastMNIST & PneumoniaMNIST, 5 seeds)**: Completed full 5-seed training/evaluation (`reports/pillar3_v3_results.json`):
  - BreastMNIST Clean: `quanv_v3_int8` (`85.00 ± 1.48%`) > `ctrl_v3_light` (`84.49 ± 1.94%`) > `ctrl_v2_full` (`84.10 ± 1.05%`), all `cmp_diff = 0.0e+00`.
  - PneumoniaMNIST Clean / `s0.20`: `quanv_v3_int8` (`89.36 ± 2.09%` / `84.13 ± 1.81%`) vs `ctrl_v3_light` (`89.23 ± 1.70%` / `83.55 ± 2.06%`) vs `ctrl_v2_full` (`90.22 ± 1.85%` / `82.68 ± 2.05%`); Micro-Edge `s0.20`: `micro_quanv` (`77.87 ± 1.85%`) vs `micro_ctrl` (`74.29 ± 3.66%`).
- [DONE] **Step B2b (OCTMNIST, 5 seeds)**: Completed `experiments/run_pillar3_v3.py --datasets octmnist` (`reports/pillar3_v3_results.json`, figure `reports/figures/v3_vs_cnn_pareto.png`):
  - Clean: `quanv_v3_int8` (**`82.90 ± 3.73%`**) > `ctrl_v2_full` (`82.52 ± 2.92%`, `+0.38%`) > `ctrl_v3_light` (`80.18 ± 2.57%`, `+2.72%`); recovers `+2.26%` over Phase A zero-shot `v2 INT8` (`80.64%`).
  - Speckle $\sigma = 0.10$ / $0.20$: `quanv_v3_int8` drops to `51.23 ± 2.28%` at $\sigma=0.10$ (below `ctrl_v3_light` `57.32%` and `ctrl_v2_full` `58.46%`, $p=0.0710/0.0521$), then recovers lead at $\sigma=0.20$ (`40.52 ± 1.36%` vs `38.24%`, $p=0.0090$). Remarkably, Micro-Edge `micro_quanv` (`0.17M` MACs) hits **`65.41 ± 2.38%`** at $\sigma=0.10$ vs `micro_ctrl` `51.72 ± 4.39%` (`+13.69%`, $p=0.0005$) and **`48.47 ± 2.73%`** at $\sigma=0.15$ vs `42.72 ± 1.84%` (`+5.75%`, $p=0.0058$), at a `-3.62%` clean trade-off (`75.10%` vs `78.72%`).
- [DONE] **Step B3 (Pure-Rust Native E2E Suite)**: Exported all 5 PneumoniaMNIST `s42` models via `experiments/export_v3_to_rust.py`, implemented `rust_qlut/src/bin/v3_suite.rs` (`medquanv-cli --suite-v3`), compiled with `target-cpu=native`, and verified 100.0% classification agreement on 624 clean + `s0.20` test images (`reports/medquanv_cli_v3_verification.json`):
  - `quanv_v3_int8`: `1,126.38 us` (`887.80 FPS`), `406.90 KiB` (`19.87%` L2), `max_diff = 5.25e-6` clean (`8.30x` faster E2E vs `ctrl_v2_full`).
  - `ctrl_v3_light`: `1,114.03 us` (`897.65 FPS`), `151.79 KiB` (`7.41%` L2), `max_diff = 5.25e-6` clean (latency parity ~1.1% vs `quanv_v3_int8`).
  - `ctrl_v2_full`: `9,352.83 us` (`106.92 FPS`), `577.65 KiB` (`28.21%` L2), `max_diff = 4.77e-6` clean.
  - `micro_quanv`: `46.18 us` (`21,656.70 FPS`), `263.29 KiB` (`12.86%` L2), `max_diff = 1.91e-6` clean (slightly faster than `micro_ctrl` `47.59 us` / `21,014.16 FPS`, `+11.54%` s42 `s0.20` ACC).
- [DONE] **Step B4 (Technical Report, Final Audit & Git Commit)**: Updated `reports/TECHNICAL_REPORT_EDGE_QUANV.md` (Sections 7–9), verified parity across all JSON/Rust artifacts, and committed all Phase A/B artifacts to git.
