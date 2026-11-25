"""One-Command Reproducibility Pipeline for MedMNIST Quanvolution Thesis Benchmarks.

Usage:
  python scripts/reproduce_all.py                # Verify locked data + Rust unit tests + regenerate all stats & audit reports (~12s)
  python scripts/reproduce_all.py --retrain-extra  # Also run/resume extra classical baselines (ctrl_tanh, ctrl_hardclip, ctrl_noise_aug)
  python scripts/reproduce_all.py --skip-cargo     # Skip Rust cargo test (if Rust toolchain is not installed)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

REQUIRED_LOCKED_FILES = [
    "results/ablation_clut/eval_results.json",
    "results/ablation_clut/extra_baselines.json",
    "reports/pareto_int8_vs_interp4d.json",
    "reports/highres_scalability_results.json",
    "reports/medquanv_cli_verification.json",
    "reports/medquanv_cli_verification_a3.json",
    "reports/medquanv_cli_v3_verification.json",
    "reports/pillar3_v3_results.json",
]


def run_step(title: str, cmd: list[str], cwd: Path = PROJECT_ROOT) -> None:
    print(f"\n{'=' * 78}\n  {title}\n  CMD: {' '.join(cmd)}\n{'=' * 78}", flush=True)
    t0 = time.perf_counter()
    res = subprocess.run(cmd, cwd=str(cwd))
    if res.returncode != 0:
        raise RuntimeError(f"Step failed ({res.returncode}): {title}")
    print(f"  [OK] Finished in {time.perf_counter() - t0:.1f}s", flush=True)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Reproduce all thesis statistical tests, audits, and Rust unit tests.")
    parser.add_argument("--retrain-extra", action="store_true", help="Run/resume experiments/run_extra_baselines.py before stats.")
    parser.add_argument("--skip-cargo", action="store_true", help="Skip `cargo test --no-default-features` in rust_qlut/.")
    args = parser.parse_args()

    t_all = time.perf_counter()
    py = sys.executable

    if args.retrain_extra:
        run_step(
            "Step 0: Train/Resume Extra Classical Baselines (5 seeds)",
            [py, "experiments/run_extra_baselines.py", "--datasets", "pneumoniamnist", "breastmnist", "--resume"],
        )

    print("\n[Check] Verifying required locked result files...")
    missing = [rel for rel in REQUIRED_LOCKED_FILES if not (PROJECT_ROOT / rel).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required result files: {missing}")
    print(f"  [OK] All {len(REQUIRED_LOCKED_FILES)} locked result files present.")

    if not args.skip_cargo:
        run_step(
            "Step 1: Rust Q-LUT Library Unit Tests (no Python dependency)",
            ["cargo", "test", "--no-default-features"],
            cwd=PROJECT_ROOT / "rust_qlut",
        )

    run_step(
        "Step 2: Rigorous Statistical Analysis (345 Paired Tests + TOST ±2% + BH FDR)",
        [py, "scripts/run_rigorous_stats.py"],
    )

    run_step(
        "Step 3: 4-bit Quantization Cost & Unified Memory Audit",
        [py, "scripts/generate_quant_mem_audit.py"],
    )

    print(f"\n{'=' * 78}")
    print(f"  ALL REPRODUCIBILITY CHECKS PASSED ({time.perf_counter() - t_all:.1f}s)")
    print("  Generated outputs:")
    print("    - reports/rigorous_stats_summary.json")
    print("    - reports/rigorous_stats_summary.md")
    print("    - reports/quantization_and_memory_audit.md")
    print(f"{'=' * 78}")


if __name__ == "__main__":
    main()
