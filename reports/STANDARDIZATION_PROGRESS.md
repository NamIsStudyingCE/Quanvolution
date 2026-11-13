# Standardization & Verification Progress Tracker

**Updated at**: 2026-11-13T13:16:00+07:00
**Constraint**: Zero Retraining (30 checkpoints in `weights/pillar2_wide/` are LOCKED) | Zero Fabrication

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
