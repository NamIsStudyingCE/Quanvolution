"""
Phase A (Threshold A - Zero Retraining) Evaluation Script
=========================================================
Tests Argon's Step 1 (Symmetric Damping + Complementary Gating) and
Step 2 (Bin-Expected MMSE 16-level INT8 Shift-LUT) strictly WITHOUT modifying
or overwriting any of the 30 locked checkpoints in `weights/pillar2_wide/`.

Outputs:
  - reports/threshold_a_results.json
"""

import json
import os
import sys
from pathlib import Path
import numpy as np
import pennylane as qml
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from models import circuits as C
from models.gr_quanv import extract_multiscale_qlut, phase_damping_u8, _build_dilated_u8, _build_axial_u8
from models.gr_quanv_wide import WideClassicalControl, WideGRQuanvNet, _stabilize_stem_variance
from qlut.compiler import build_batched_qnode
from run_pillar2_wide import load_dataset_with_cache, SEEDS
from run_pillar2_expansion import SIGMAS, NOISE_DRAWS, DATASETS, speckle
from utils.metrics import calculate_metrics

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_exact_hermitian_observables_strongly_l2():
    """
    Computes exact 16x16 real symmetric matrices M_k = Re(U^dagger Z_k U)
    for C.strongly_w_2l, and verifies <x|M_k|x> == PennyLane QNode within 1e-6.
    """
    dev = qml.device("default.qubit", wires=4)

    @qml.qnode(dev)
    def ansatz_unitary_circuit():
        qml.StronglyEntanglingLayers(weights=C.strongly_w_2l, wires=range(4))
        return qml.state()

    U = np.asarray(qml.matrix(ansatz_unitary_circuit)(), dtype=np.complex128)  # (16, 16)

    def z_diag(q):
        # Wire 0 is MSB (stride 8), wire 3 is LSB (stride 1)
        vals = np.empty(16, dtype=np.complex128)
        for idx in range(16):
            bit = (idx >> (3 - q)) & 1
            vals[idx] = 1.0 if bit == 0 else -1.0
        return np.diag(vals)

    M = np.empty((4, 16, 16), dtype=np.float64)
    for k in range(4):
        Mk = U.conj().T @ z_diag(k) @ U
        M[k] = Mk.real

    # Verify against PennyLane batched QNode
    qnode = build_batched_qnode("strongly_L2")
    rng = np.random.default_rng(123)
    phi_test = rng.uniform(0.0, 1.0, size=(64, 4))
    ref = np.stack(qnode(phi_test), axis=-1)

    c = np.cos(0.5 * np.pi * phi_test)
    s = np.sin(0.5 * np.pi * phi_test)
    # Kronecker product over 4 qubits (wire 0 MSB -> wire 3 LSB)
    v0 = np.stack([c[:, 0], s[:, 0]], axis=1)  # (N, 2)
    v1 = np.stack([c[:, 1], s[:, 1]], axis=1)
    v2 = np.stack([c[:, 2], s[:, 2]], axis=1)
    v3 = np.stack([c[:, 3], s[:, 3]], axis=1)
    psi = np.einsum("na,nb,nc,nd->nabcd", v0, v1, v2, v3).reshape(-1, 16)
    pred = np.einsum("ni,kij,nj->nk", psi, M, psi)
    max_diff = float(np.max(np.abs(ref - pred)))
    assert max_diff < 1e-6, f"Hermitian observable verification failed: max_diff={max_diff}"
    print(f"[Verify M_k(theta)] Exact 16x16 bilinear match vs PennyLane QNode: max|diff| = {max_diff:.2e}")
    return M


def compile_mmse_bin_expected_shift_table(M: np.ndarray, offsets=None) -> np.ndarray:
    """
    Exact product-state density expectation over all pixel values in shift bin [16q, 16q+15]:
      rho_bin(q) = mean_{u in S_q} |x(u)><x(u)|,
      E[<Z_k>] = Tr(M_k (rho_0 (x) rho_1 (x) rho_2 (x) rho_3)).
    If offsets is None -> exact average over all 16 integer levels 0..15!
    Returns int8 table of shape (16, 16, 16, 16, 4) quantized * 127.
    """
    if offsets is None:
        offsets = np.arange(16, dtype=np.float64)
    else:
        offsets = np.asarray(offsets, dtype=np.float64)

    rho_1q = np.zeros((16, 2, 2), dtype=np.float64)
    for q in range(16):
        u = (16.0 * q + offsets) / 255.0
        v = np.stack([np.cos(0.5 * np.pi * u), np.sin(0.5 * np.pi * u)], axis=1)  # (K, 2)
        rho_1q[q] = (v.T @ v) / len(offsets)

    # Reshape M_k to (2,2,2,2, 2,2,2,2) -> (k, i0, i1, i2, i3, j0, j1, j2, j3)
    # Note: E[psi^T M_k psi] = sum_{i,j} M_k[i,j] * rho0[i0,j0] rho1[i1,j1] rho2[i2,j2] rho3[i3,j3]
    M_t = M.reshape(4, 2, 2, 2, 2, 2, 2, 2, 2)
    table = np.einsum(
        "kabcdefgh,pai,qbj,rck,sdl->pqrs" + "k",
        # Wait, avoid index letter clash!
        # Let's use unambiguous indices: k for output, a..d for ket, e..h for bra, w,x,y,z for q0..q3
        optimize=True,
    ) if False else np.einsum(
        "kabcdefgh,wae,xbf,ycg,zdh->wxyzk",
        M_t,
        rho_1q,
        rho_1q,
        rho_1q,
        rho_1q,
        optimize=True,
    )
    return np.clip(np.rint(table * 127.0), -127, 127).astype(np.int8)


def extract_multiscale_from_int8_table(u8: np.ndarray, table_i8: np.ndarray, phase_damping: bool) -> torch.Tensor:
    imgs_u8 = np.ascontiguousarray(u8, dtype=np.uint8)
    if phase_damping:
        imgs_u8 = phase_damping_u8(imgs_u8)
    u2 = _build_dilated_u8(imgs_u8, dilation=2)
    u3 = _build_dilated_u8(imgs_u8, dilation=3)
    u4 = _build_axial_u8(imgs_u8)
    scales = []
    for u in (imgs_u8, u2, u3, u4):
        p0 = (u[:, 0::2, 0::2] >> 4).astype(np.int64)
        p1 = (u[:, 0::2, 1::2] >> 4).astype(np.int64)
        p2 = (u[:, 1::2, 0::2] >> 4).astype(np.int64)
        p3 = (u[:, 1::2, 1::2] >> 4).astype(np.int64)
        q_nhwc = table_i8[p0, p1, p2, p3]  # (N, 14, 14, 4) int8
        q_nchw = np.transpose(q_nhwc, (0, 3, 1, 2))
        scales.append(q_nchw)
    cat = np.concatenate(scales, axis=1)
    return torch.from_numpy(cat).to(torch.float32) / 127.0


@torch.no_grad()
def forward_grquanv_variant(model: WideGRQuanvNet, x_c: torch.Tensor, q_ms: torch.Tensor, mode: str) -> torch.Tensor:
    q_used = q_ms[:, : model.in_q_channels, :, :]
    f_c = _stabilize_stem_variance(x_c, model.norm_c(model.c_stem(x_c)), False, tau=0.024)
    f_q = model.norm_q(model.q_mixer(q_used))
    alpha = torch.sigmoid(model.gate(torch.cat([f_c, f_q], dim=1)))

    if mode == "standard":
        f_fused = torch.cat([f_c, alpha * f_q], dim=1)
    elif mode == "clamp_cweight":
        # Argon Step 1 formula: suppress classical branch as alpha increases above clean baseline (~0.44)
        c_weight = torch.clamp(1.0 - (alpha - 0.44) * 8.0, min=0.2, max=1.0)
        f_fused = torch.cat([c_weight * f_c, alpha * f_q], dim=1)
    elif mode == "varpres_sqrt2":
        # Argon Sec 1.2 Eq. 1A formula
        c_w = torch.sqrt(torch.clamp(2.0 * (1.0 - alpha * alpha), min=1e-4))
        q_w = np.sqrt(2.0) * alpha
        f_fused = torch.cat([c_w * f_c, q_w * f_q], dim=1)
    else:
        raise ValueError(mode)
    return model.head(f_fused)


@torch.no_grad()
def eval_model_acc(fn_logits, y_np: np.ndarray, nc: int, bs: int = 2048):
    probs = fn_logits(bs)
    m = calculate_metrics(y_np, probs, probs.argmax(axis=1), nc)
    return m["acc"] * 100.0, m["auc"] * 100.0, m["f1"] * 100.0


@torch.no_grad()
def run_step_a1_all_datasets(ckpt_dir: Path):
    print("\n" + "=" * 88)
    print("  STEP A1: SYMMETRIC DAMPING & COMPLEMENTARY GATING (ZERO RETRAINING, ALL 3 DATASETS)")
    print("=" * 88)
    variants = [
        "ctrl_damped",
        "grquanv_current_raw_cstem",
        "grquanv_sym_damped",
        "grquanv_sym_clamp_cweight",
        "grquanv_sym_varpres_sqrt2",
    ]
    all_res = {}
    for ds, nc in [("breastmnist", 2), ("pneumoniamnist", 2), ("octmnist", 4)]:
        D = load_dataset_with_cache(ds, interp4d=True)
        te_f = D["te"].astype(np.float32) / 255.0
        y_test = D["yte"].astype(np.int64)

        models_ctrl, models_q = {}, {}
        for s in SEEDS:
            mc = WideClassicalControl(num_classes=nc).to(DEV)
            mc.load_state_dict(torch.load(ckpt_dir / f"wide_v2_interp4d_{ds}_full_control_s{s}.pt", map_location=DEV, weights_only=True))
            mc.eval()
            models_ctrl[s] = mc

            mq = WideGRQuanvNet(num_classes=nc).to(DEV)
            mq.load_state_dict(torch.load(ckpt_dir / f"wide_v2_interp4d_{ds}_full_grquanv_s{s}.pt", map_location=DEV, weights_only=True))
            mq.eval()
            models_q[s] = mq

        res = {v: {f"{sg:.2f}": {} for sg in SIGMAS} for v in variants}
        print(f"\n=== Dataset: {ds.upper()} ===")

        for sg in SIGMAS:
            draws = NOISE_DRAWS if sg > 0 else [0]
            acc_map = {v: {s: [] for s in SEEDS} for v in variants}
            auc_map = {v: {s: [] for s in SEEDS} for v in variants}
            f1_map = {v: {s: [] for s in SEEDS} for v in variants}

            for d in draws:
                noisy_u8 = D["te"] if sg == 0.0 else speckle(te_f, sg, d)
                damped_u8 = phase_damping_u8(noisy_u8)
                x_raw_t = torch.from_numpy((noisy_u8.astype(np.float32) / 255.0 - 0.5) / 0.5).unsqueeze(1).to(DEV)
                x_damp_t = torch.from_numpy((damped_u8.astype(np.float32) / 255.0 - 0.5) / 0.5).unsqueeze(1).to(DEV)
                q_damp, _ = extract_multiscale_qlut(damped_u8, include_axial=True, interp4d=True, phase_damping=False)
                q_damp = q_damp.to(DEV)

                for s in SEEDS:
                    def eval_fn(fn):
                        ps = []
                        for i in range(0, len(y_test), 2048):
                            ps.append(torch.softmax(fn(i, min(i + 2048, len(y_test))), dim=1).cpu().numpy())
                        p = np.concatenate(ps, axis=0)
                        m = calculate_metrics(y_test, p, p.argmax(1), nc)
                        return m["acc"] * 100.0, m["auc"] * 100.0, m["f1"] * 100.0

                    mc = models_ctrl[s]
                    mq = models_q[s]

                    for v, call_fn in [
                        ("ctrl_damped", lambda a, b: mc(x_damp_t[a:b])),
                        ("grquanv_current_raw_cstem", lambda a, b: mq(x_raw_t[a:b], q_damp[a:b])),
                        ("grquanv_sym_damped", lambda a, b: mq(x_damp_t[a:b], q_damp[a:b])),
                        ("grquanv_sym_clamp_cweight", lambda a, b: forward_grquanv_variant(mq, x_damp_t[a:b], q_damp[a:b], "clamp_cweight")),
                        ("grquanv_sym_varpres_sqrt2", lambda a, b: forward_grquanv_variant(mq, x_damp_t[a:b], q_damp[a:b], "varpres_sqrt2")),
                    ]:
                        ac, au, f1 = eval_fn(call_fn)
                        acc_map[v][s].append(ac)
                        auc_map[v][s].append(au)
                        f1_map[v][s].append(f1)

            print(f"  -- Sigma = {sg:.2f} --")
            for v in variants:
                sa = [np.mean(acc_map[v][s]) for s in SEEDS]
                su = [np.mean(auc_map[v][s]) for s in SEEDS]
                sf = [np.mean(f1_map[v][s]) for s in SEEDS]
                res[v][f"{sg:.2f}"] = {
                    "acc_mean": float(np.mean(sa)),
                    "acc_std": float(np.std(sa)),
                    "auc_mean": float(np.mean(su)),
                    "f1_mean": float(np.mean(sf)),
                }
                print(f"    {v:<27} | ACC: {np.mean(sa):5.2f}+-{np.std(sa):4.2f}% | AUC: {np.mean(su):5.2f}% | F1: {np.mean(sf):5.2f}%")
        all_res[ds] = res

    return all_res


def run_step_a2_mmse_lut(ckpt_dir: Path, table_mid: np.ndarray, table_mmse16: np.ndarray, table_mmse4pt: np.ndarray):
    print("\n" + "=" * 88)
    print("  STEP A2: BIN-EXPECTED MMSE INT8 SHIFT-LUT (256 KiB, ZERO RETRAINING)")
    print("=" * 88)
    diff_mid_vs_16 = np.abs(table_mid.astype(np.int16) - table_mmse16.astype(np.int16))
    diff_mid_vs_4pt = np.abs(table_mid.astype(np.int16) - table_mmse4pt.astype(np.int16))
    print(f"  [LUT Compare] Midpoint vs 16-level Exact MMSE: mean|diff|={diff_mid_vs_16.mean():.3f} lsb, max|diff|={diff_mid_vs_16.max()} lsb, exact_match={(diff_mid_vs_16==0).mean()*100:.2f}%")
    print(f"  [LUT Compare] Midpoint vs 4-point Gauss MMSE : mean|diff|={diff_mid_vs_4pt.mean():.3f} lsb, max|diff|={diff_mid_vs_4pt.max()} lsb")

    res = {}
    for ds, nc in [("breastmnist", 2), ("pneumoniamnist", 2), ("octmnist", 4)]:
        D = load_dataset_with_cache(ds, interp4d=True)
        te_f = D["te"].astype(np.float32) / 255.0
        y_test = D["yte"].astype(np.int64)

        models_q = {}
        for s in SEEDS:
            mq = WideGRQuanvNet(num_classes=nc).to(DEV)
            mq.load_state_dict(torch.load(ckpt_dir / f"wide_v2_interp4d_{ds}_full_grquanv_s{s}.pt", map_location=DEV, weights_only=True))
            mq.eval()
            models_q[s] = mq

        res[ds] = {}
        print(f"\n=== Dataset: {ds.upper()} ===")
        for sg in [0.00, 0.05, 0.10, 0.20]:
            draws = NOISE_DRAWS if sg > 0 else [0]
            modes = ["int8_midpoint", "int8_mmse16", "int8_mmse4pt", "interp4d"]
            acc_map = {m: {s: [] for s in SEEDS} for m in modes}

            for d in draws:
                noisy_u8 = D["te"] if sg == 0.0 else speckle(te_f, sg, d)
                x_raw_t = torch.from_numpy((noisy_u8.astype(np.float32) / 255.0 - 0.5) / 0.5).unsqueeze(1).to(DEV)
                q_mid = extract_multiscale_from_int8_table(noisy_u8, table_mid, phase_damping=True).to(DEV)
                q_m16 = extract_multiscale_from_int8_table(noisy_u8, table_mmse16, phase_damping=True).to(DEV)
                q_m4 = extract_multiscale_from_int8_table(noisy_u8, table_mmse4pt, phase_damping=True).to(DEV)
                q_i4, _ = extract_multiscale_qlut(noisy_u8, include_axial=True, interp4d=True, phase_damping=True)
                q_i4 = q_i4.to(DEV)

                q_dict = {"int8_midpoint": q_mid, "int8_mmse16": q_m16, "int8_mmse4pt": q_m4, "interp4d": q_i4}
                for m_name in modes:
                    q_t = q_dict[m_name]
                    for s in SEEDS:
                        mq = models_q[s]
                        ps = []
                        with torch.no_grad():
                            for i in range(0, len(y_test), 2048):
                                ps.append(torch.softmax(mq(x_raw_t[i:i + 2048], q_t[i:i + 2048]), dim=1).cpu().numpy())
                        p = np.concatenate(ps, axis=0)
                        met = calculate_metrics(y_test, p, p.argmax(1), nc)
                        acc_map[m_name][s].append(met["acc"] * 100.0)

            res[ds][f"{sg:.2f}"] = {}
            line = f"  sigma={sg:.2f} | "
            parts = []
            for m_name in modes:
                sa = [np.mean(acc_map[m_name][s]) for s in SEEDS]
                res[ds][f"{sg:.2f}"][m_name] = {"acc_mean": float(np.mean(sa)), "acc_std": float(np.std(sa))}
                parts.append(f"{m_name}: {np.mean(sa):.2f}%")
            print(line + " | ".join(parts))

    return res


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    M = build_exact_hermitian_observables_strongly_l2()
    table_mid = np.fromfile(PROJECT_ROOT / "weights" / "qlut_strongly_L2_4bit_int8_shift.bin", dtype=np.int8).reshape(16, 16, 16, 16, 4)
    table_mmse16 = compile_mmse_bin_expected_shift_table(M, offsets=None)
    table_mmse4pt = compile_mmse_bin_expected_shift_table(M, offsets=[2.0, 6.0, 9.0, 13.0])

    ckpt_dir = PROJECT_ROOT / "weights" / "pillar2_wide"
    res_a1 = run_step_a1_all_datasets(ckpt_dir)
    res_a2 = run_step_a2_mmse_lut(ckpt_dir, table_mid, table_mmse16, table_mmse4pt)

    out_path = PROJECT_ROOT / "reports" / "threshold_a_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"step_a1_symmetry_gating": res_a1, "step_a2_mmse_lut": res_a2}, f, indent=2)
    print(f"\n[Saved] {out_path}")


if __name__ == "__main__":
    main()
