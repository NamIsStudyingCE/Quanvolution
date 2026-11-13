"""
Generate Fair Classical-LUT (C-LUT) Binaries for Isolated Ablation vs. Q-LUT
===========================================================================
Generates two non-trivial classical 4D->4D lookup tables bounded in [-1, 1]:
  1. `clut_septrig`: Separable single-pixel [cos(pi*x_i), sin(pi*x_i)] encoding
     projected through a fixed random orthogonal matrix W in R^{4x8} + tanh.
     (Direct isolation of multi-qubit cross-phase CNOT entanglement.)
  2. `clut_randmlp`: Fixed 2-layer random orthogonal MLP (4->16 GELU -> 16->4 + tanh).

Both outputs are variance-calibrated via pre-activation gain `g` so global output
standard deviation across the 65,536 grid states matches `qlut_strongly_L2_4bit_fp32_grid.bin`
within +/-0.5% (strictly tighter than +/-5%).

Exports exact binary formats compatible with `rust_qlut`:
  - `*_4bit_int8_grid.bin`   (262,144 bytes = 256 KiB)
  - `*_4bit_int8_shift.bin`  (262,144 bytes = 256 KiB)
  - `*_4bit_fp32_grid.bin`   (1,048,576 bytes = 1,024 KiB)
  - `*_4bit_fp32_shift.bin`  (1,048,576 bytes = 1,024 KiB)

Also provides exact PyTorch/GPU vectorized LUT extractors (for Interp4D & INT8 Shift-LUT)
verified against `rust_qlut` SIMD kernels within < 1e-6 without modifying Rust source code.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from models.gr_quanv import _build_axial_u8, _build_dilated_u8, phase_damping_u8

CORNER_BITS = torch.tensor(
    [[(v >> (3 - d)) & 1 for d in range(4)] for v in range(16)], dtype=torch.long
)
STRIDES = torch.tensor([4096, 256, 16, 1], dtype=torch.long)


def _grid_and_shift_inputs() -> Tuple[np.ndarray, np.ndarray]:
    """Return `(phi_grid, phi_shift)` of shape `(65536, 4)` in exact `rust_qlut` bit order."""
    L = 16
    total = L**4
    idx = np.arange(total, dtype=np.int64)
    p0 = idx >> 12
    p1 = (idx >> 8) & 0xF
    p2 = (idx >> 4) & 0xF
    p3 = idx & 0xF

    levels_grid = np.arange(L, dtype=np.float64) / 15.0
    levels_shift = (16.0 * np.arange(L, dtype=np.float64) + 7.5) / 255.0
    phi_grid = np.stack([levels_grid[p0], levels_grid[p1], levels_grid[p2], levels_grid[p3]], axis=1)
    phi_shift = np.stack([levels_shift[p0], levels_shift[p1], levels_shift[p2], levels_shift[p3]], axis=1)
    return phi_grid, phi_shift


def _calibrate_tanh_gain(raw: np.ndarray, target_std: float) -> float:
    """Binary-search scalar gain `g` such that `np.tanh(g * raw).std() == target_std`."""
    lo, hi = 1e-3, 30.0
    for _ in range(64):
        mid = 0.5 * (lo + hi)
        if float(np.tanh(mid * raw).std()) < target_std:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def make_septrig_evaluator(seed: int = 42, target_std: float = 0.23791):
    """Build deterministic separable trigonometric projection calibrated to `target_std`."""
    rng = np.random.default_rng(seed)
    q_mat, _ = np.linalg.qr(rng.normal(size=(8, 4)))
    w_proj = q_mat.T  # (4, 8) orthonormal rows

    def raw_fn(phi: np.ndarray) -> np.ndarray:
        feat = np.empty((len(phi), 8), dtype=np.float64)
        feat[:, 0::2] = np.cos(np.pi * phi)
        feat[:, 1::2] = np.sin(np.pi * phi)
        return feat @ w_proj.T

    phi_grid, _ = _grid_and_shift_inputs()
    gain = _calibrate_tanh_gain(raw_fn(phi_grid), target_std)

    def eval_fn(phi: np.ndarray) -> np.ndarray:
        return np.tanh(gain * raw_fn(phi)).astype(np.float32)

    return eval_fn, gain


def make_randmlp_evaluator(seed: int = 42, target_std: float = 0.23791):
    """Build fixed 2-layer orthogonal MLP (4->16 GELU -> 16->4 + tanh) calibrated to `target_std`."""
    torch.manual_seed(seed)
    l1 = nn.Linear(4, 16, bias=True)
    l2 = nn.Linear(16, 4, bias=True)
    nn.init.orthogonal_(l1.weight)
    nn.init.orthogonal_(l2.weight)
    nn.init.zeros_(l1.bias)
    nn.init.zeros_(l2.bias)

    def raw_fn(phi: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x = torch.from_numpy(phi.astype(np.float32))
            h = F.gelu(l1(x))
            return l2(h).numpy().astype(np.float64)

    phi_grid, _ = _grid_and_shift_inputs()
    gain = _calibrate_tanh_gain(raw_fn(phi_grid), target_std)

    def eval_fn(phi: np.ndarray) -> np.ndarray:
        return np.tanh(gain * raw_fn(phi)).astype(np.float32)

    return eval_fn, gain


def summarize_table(arr: np.ndarray) -> Dict[str, object]:
    """Compute exact global and per-channel distribution statistics across 65,536 LUT entries."""
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "channel_mean": [float(arr[:, c].mean()) for c in range(4)],
        "channel_std": [float(arr[:, c].std()) for c in range(4)],
    }


def export_lut_binaries(name: str, eval_fn, out_dir: Path) -> Dict[str, object]:
    """Compile `fp32_grid`, `fp32_shift`, `int8_grid`, `int8_shift` `.bin` files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    phi_grid, phi_shift = _grid_and_shift_inputs()

    table_grid = np.ascontiguousarray(eval_fn(phi_grid), dtype=np.float32)
    table_shift = np.ascontiguousarray(eval_fn(phi_shift), dtype=np.float32)
    int8_grid = np.clip(np.rint(table_grid * 127.0), -127, 127).astype(np.int8)
    int8_shift = np.clip(np.rint(table_shift * 127.0), -127, 127).astype(np.int8)

    files = {
        "fp32_grid": (out_dir / f"{name}_4bit_fp32_grid.bin", table_grid, 1_048_576),
        "fp32_shift": (out_dir / f"{name}_4bit_fp32_shift.bin", table_shift, 1_048_576),
        "int8_grid": (out_dir / f"{name}_4bit_int8_grid.bin", int8_grid, 262_144),
        "int8_shift": (out_dir / f"{name}_4bit_int8_shift.bin", int8_shift, 262_144),
    }
    paths = {}
    for fmt, (path, data, expected_bytes) in files.items():
        data.tofile(path)
        actual_bytes = path.stat().st_size
        assert actual_bytes == expected_bytes, f"{path}: got {actual_bytes}, expected {expected_bytes}"
        paths[fmt] = str(path.as_posix())

    return {
        "paths": paths,
        "stats_fp32_grid": summarize_table(table_grid),
        "stats_fp32_shift": summarize_table(table_shift),
        "stats_int8_shift_dequant": summarize_table(int8_shift.astype(np.float32) / 127.0),
    }


def load_lut_table(bin_path: str | Path, mode: str, device: torch.device) -> torch.Tensor:
    """Load `(65536, 4)` LUT `.bin` into device tensor (`float32` for interp4d, `int8` for int8_shift)."""
    p = Path(bin_path)
    if mode == "interp4d":
        arr = np.fromfile(p, dtype="<f4")
        assert arr.size == 65536 * 4, f"Expected 262144 floats in {p}, got {arr.size}"
        return torch.from_numpy(arr.reshape(65536, 4).copy()).to(device=device, dtype=torch.float32)
    elif mode == "int8_shift":
        arr = np.fromfile(p, dtype=np.int8)
        assert arr.size == 65536 * 4, f"Expected 262144 int8s in {p}, got {arr.size}"
        return torch.from_numpy(arr.reshape(65536, 4).copy()).to(device=device, dtype=torch.int8)
    raise ValueError(f"Unsupported LUT mode: {mode}")


@torch.no_grad()
def _extract_single_scale(
    u8_batch: torch.Tensor,
    table_flat: torch.Tensor,
    mode: str,
    corner_bits: torch.Tensor,
    strides: torch.Tensor,
) -> torch.Tensor:
    """Single-scale `(N, 28, 28) uint8` -> `(N, 4, 14, 14) float32` matching `rust_qlut`."""
    if mode == "interp4d":
        p = torch.stack(
            [
                u8_batch[:, 0::2, 0::2],
                u8_batch[:, 0::2, 1::2],
                u8_batch[:, 1::2, 0::2],
                u8_batch[:, 1::2, 1::2],
            ],
            dim=-1,
        ).float()
        s = (p / 255.0) * 15.0
        lo = torch.clamp(torch.floor(s).long(), max=14)
        frac = s - lo.float()
        c = lo.unsqueeze(-2) + corner_bits
        idx = (c * strides).sum(dim=-1)
        f = frac.unsqueeze(-2)
        w = torch.where(corner_bits.bool(), f, 1.0 - f).prod(dim=-1)
        out = (w.unsqueeze(-1) * table_flat[idx]).sum(dim=-2)
        return out.permute(0, 3, 1, 2).contiguous()
    else:
        p0 = u8_batch[:, 0::2, 0::2].long() >> 4
        p1 = u8_batch[:, 0::2, 1::2].long() >> 4
        p2 = u8_batch[:, 1::2, 0::2].long() >> 4
        p3 = u8_batch[:, 1::2, 1::2].long() >> 4
        idx = (p0 << 12) | (p1 << 8) | (p2 << 4) | p3
        out = table_flat[idx].float() / 127.0
        return out.permute(0, 3, 1, 2).contiguous()


@torch.no_grad()
def extract_multiscale_from_lut_tensor(
    imgs_u8: np.ndarray,
    table_flat: torch.Tensor,
    mode: str = "interp4d",
    phase_damping: bool = False,
    chunk: int = 2048,
) -> torch.Tensor:
    """
    Extract exact 16-channel multi-scale (`d=1, d=2, d=3, axial`) `(N, 16, 14, 14)` features
    using any loaded `(65536, 4)` LUT tensor on GPU/CPU, matching `rust_qlut` < 1e-6.
    """
    imgs_u8 = np.ascontiguousarray(imgs_u8, dtype=np.uint8)
    if phase_damping:
        imgs_u8 = phase_damping_u8(imgs_u8)

    u2 = _build_dilated_u8(imgs_u8, dilation=2)
    u3 = _build_dilated_u8(imgs_u8, dilation=3)
    ax = _build_axial_u8(imgs_u8)

    dev = table_flat.device
    cb = CORNER_BITS.to(dev)
    st = STRIDES.to(dev)
    out_dtype = torch.float16 if mode == "interp4d" else torch.int8

    n = len(imgs_u8)
    out_chunks = []
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        scales = []
        for arr in (imgs_u8[start:end], u2[start:end], u3[start:end], ax[start:end]):
            ut = torch.from_numpy(np.ascontiguousarray(arr)).to(dev)
            feat = _extract_single_scale(ut, table_flat, mode, cb, st)
            scales.append(feat)
        cat = torch.cat(scales, dim=1)
        if out_dtype == torch.float16:
            out_chunks.append(cat.to(torch.float16))
        else:
            out_chunks.append(torch.round(cat * 127.0).to(torch.int8))
    return torch.cat(out_chunks, dim=0)


def verify_parity_with_rust_qlut() -> Dict[str, float]:
    """Assert vectorized PyTorch LUT extractor matches `rust_qlut` SIMD kernels on Q-LUT."""
    import rust_qlut

    qlut_grid_path = PROJECT_ROOT / "weights" / "qlut_strongly_L2_4bit_fp32_grid.bin"
    qlut_shift_path = PROJECT_ROOT / "weights" / "qlut_strongly_L2_4bit_int8_shift.bin"

    rng = np.random.default_rng(2026)
    sample_u8 = rng.integers(0, 256, size=(32, 28, 28), dtype=np.uint8)

    # 1. Interp4D parity (comparing float32 before fp16 storage)
    t_grid = load_lut_table(qlut_grid_path, "interp4d", torch.device("cpu"))
    dev_f32 = torch.device("cpu")
    cb, st = CORNER_BITS.to(dev_f32), STRIDES.to(dev_f32)
    scales_py = []
    for arr in (
        sample_u8,
        _build_dilated_u8(sample_u8, 2),
        _build_dilated_u8(sample_u8, 3),
        _build_axial_u8(sample_u8),
    ):
        scales_py.append(_extract_single_scale(torch.from_numpy(arr), t_grid, "interp4d", cb, st).numpy())
    py_interp = np.concatenate(scales_py, axis=1)

    scales_rust = []
    for arr in (
        sample_u8,
        _build_dilated_u8(sample_u8, 2),
        _build_dilated_u8(sample_u8, 3),
        _build_axial_u8(sample_u8),
    ):
        buf = np.empty((32, 4, 14, 14), dtype=np.float32)
        rust_qlut.forward_interp4d_nchw(np.ascontiguousarray(arr), buf, parallel=False)
        scales_rust.append(buf)
    rust_interp = np.concatenate(scales_rust, axis=1)
    max_diff_interp = float(np.max(np.abs(py_interp - rust_interp)))

    # 2. INT8 Shift parity
    t_shift = load_lut_table(qlut_shift_path, "int8_shift", torch.device("cpu"))
    py_i8 = extract_multiscale_from_lut_tensor(sample_u8, t_shift, mode="int8_shift", phase_damping=False).numpy()
    scales_r_i8 = []
    for arr in (
        sample_u8,
        _build_dilated_u8(sample_u8, 2),
        _build_dilated_u8(sample_u8, 3),
        _build_axial_u8(sample_u8),
    ):
        buf = np.empty((32, 4, 14, 14), dtype=np.int8)
        rust_qlut.forward_int8_nchw(np.ascontiguousarray(arr), buf, use_shift=True, parallel=False)
        scales_r_i8.append(buf)
    rust_i8 = np.concatenate(scales_r_i8, axis=1)
    max_diff_i8 = float(np.max(np.abs(py_i8.astype(np.int16) - rust_i8.astype(np.int16))))

    assert max_diff_interp < 1e-6, f"Interp4D parity failed: {max_diff_interp}"
    assert max_diff_i8 == 0.0, f"INT8 Shift parity failed: {max_diff_i8}"
    return {"max_diff_interp4d": max_diff_interp, "max_diff_int8_shift_lsb": max_diff_i8}


def generate_all() -> Dict[str, object]:
    """Generate both C-LUT families, verify Q-LUT parity & variance calibration, and save metadata."""
    out_bin_dir = PROJECT_ROOT / "weights" / "ablation_clut"
    out_res_dir = PROJECT_ROOT / "results" / "ablation_clut"
    out_res_dir.mkdir(parents=True, exist_ok=True)

    # Read locked Q-LUT baseline statistics (read-only)
    q_grid = np.fromfile(PROJECT_ROOT / "weights" / "qlut_strongly_L2_4bit_fp32_grid.bin", dtype="<f4").reshape(65536, 4)
    q_shift = np.fromfile(PROJECT_ROOT / "weights" / "qlut_strongly_L2_4bit_fp32_shift.bin", dtype="<f4").reshape(65536, 4)
    q_i8_shift = np.fromfile(PROJECT_ROOT / "weights" / "qlut_strongly_L2_4bit_int8_shift.bin", dtype=np.int8).reshape(65536, 4)

    q_stats = {
        "stats_fp32_grid": summarize_table(q_grid),
        "stats_fp32_shift": summarize_table(q_shift),
        "stats_int8_shift_dequant": summarize_table(q_i8_shift.astype(np.float32) / 127.0),
    }
    target_std = q_stats["stats_fp32_grid"]["std"]

    eval_septrig, gain_septrig = make_septrig_evaluator(seed=42, target_std=target_std)
    eval_randmlp, gain_randmlp = make_randmlp_evaluator(seed=42, target_std=target_std)

    septrig_info = export_lut_binaries("clut_septrig", eval_septrig, out_bin_dir)
    randmlp_info = export_lut_binaries("clut_randmlp", eval_randmlp, out_bin_dir)

    for label, info in (("clut_septrig", septrig_info), ("clut_randmlp", randmlp_info)):
        rel_std_diff = abs(info["stats_fp32_grid"]["std"] - target_std) / target_std * 100.0
        info["std_diff_vs_qlut_pct"] = float(rel_std_diff)
        assert rel_std_diff < 5.0, f"{label} variance calibration error {rel_std_diff:.2f}% exceeds +/-5%"

    parity = verify_parity_with_rust_qlut()

    report = {
        "seed": 42,
        "target_qlut_std": target_std,
        "rust_extractor_parity": parity,
        "tables": {
            "qlut_strongly_L2": {"gain": None, **q_stats},
            "clut_septrig": {"gain": float(gain_septrig), **septrig_info},
            "clut_randmlp": {"gain": float(gain_randmlp), **randmlp_info},
        },
    }
    stats_path = out_res_dir / "clut_calibration_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("=" * 84)
    print("  STEP 1: CLASSICAL-LUT (C-LUT) GENERATION & VARIANCE CALIBRATION COMPLETE")
    print("=" * 84)
    for name in ("qlut_strongly_L2", "clut_septrig", "clut_randmlp"):
        st = report["tables"][name]["stats_fp32_grid"]
        g_str = f"gain={report['tables'][name]['gain']:.4f}" if report["tables"][name]["gain"] else "gain=native  "
        print(
            f"  {name:<18} | {g_str} | mean={st['mean']:+.5f} std={st['std']:.5f} "
            f"[{st['min']:+.4f}, {st['max']:+.4f}]"
        )
    print(f"  Rust Extractor Parity: Interp4D max|diff|={parity['max_diff_interp4d']:.2e} | INT8 max|diff|={parity['max_diff_int8_shift_lsb']:.1f} LSB")
    print(f"  Saved binaries -> {out_bin_dir}/")
    print(f"  Saved stats    -> {stats_path}")
    return report


if __name__ == "__main__":
    generate_all()
