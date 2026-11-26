"""Lightweight CPU-only inference & telemetry backend for MedQuanv Workbench.

Reads strictly read-only from test splits (skipping train/val), checkpoints, LUT binaries,
and recorded JSON/TXT hardware reports. Never initializes CUDA or heavy ML utilities.
"""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

torch.set_num_threads(2)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.gr_quanv import phase_damping_u8  # noqa: E402
from models.gr_quanv_wide import (  # noqa: E402
    DEFAULT_DROP_C_PROB,
    WideClassicalControl,
    WideGRQuanvNet,
    _apply_branch_droppath,
    _stabilize_stem_variance,
)
from scripts.generate_clut import extract_multiscale_from_lut_tensor, load_lut_table  # noqa: E402

DEV = torch.device("cpu")

MODEL_SPECS: List[Tuple[str, str, str]] = [
    ("qlut", "Q-LUT (Lượng tử 4-qubit)", "lut"),
    ("clut_randmlp", "CLUT-RandMLP (Bảng tra ngẫu nhiên)", "lut"),
    ("clut_septrig", "CLUT-SepTrig (Bảng tra lượng giác)", "lut"),
    ("ctrl", "CNN cổ điển (Conv2d gốc)", "conv"),
    ("ctrl_tanh", "CNN + tanh [-1, 1]", "conv"),
    ("ctrl_hardclip", "CNN + cắt cứng [-1, 1]", "conv"),
    ("ctrl_noise_aug", "CNN + học kèm nhiễu", "conv"),
]

CLASS_NAMES_MAP: Dict[str, List[str]] = {
    "pneumoniamnist": ["Bình thường", "Viêm phổi"],
    "breastmnist": ["Ác tính", "Lành tính"],
}


def speckle(img_f: np.ndarray, sigma: float, draw: int) -> np.ndarray:
    """Exact multiplicative speckle noise matching experiments/run_pillar2_expansion.py."""
    if sigma == 0.0:
        return np.ascontiguousarray((img_f * 255).round().astype(np.uint8))
    rng = np.random.RandomState(1000 + draw)
    nf = np.clip(img_f + img_f * rng.normal(0, sigma, img_f.shape), 0, 1)
    return np.ascontiguousarray((nf * 255).round().astype(np.uint8))


class WideClassicalControlBounded(WideClassicalControl):
    """WideClassicalControl with tanh or hardclip [-1, 1] bound on the 16-ch conv branches."""

    def __init__(
        self,
        num_classes: int = 2,
        drop_c_prob: float = DEFAULT_DROP_C_PROB,
        bound_mode: str = "tanh",
    ):
        super().__init__(num_classes=num_classes, drop_c_prob=drop_c_prob)
        self.bound_mode = bound_mode

    def forward_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        f_c = _stabilize_stem_variance(
            x,
            self.norm_c(self.c_stem(x)),
            self.training,
            tau=getattr(self, "stabilize_tau", 0.024),
        )
        f_raw = torch.cat([b(x)[:, :, :14, :14] for b in self.classical_branches], dim=1)
        if self.bound_mode == "tanh":
            f_raw = torch.tanh(f_raw)
        elif self.bound_mode == "hardclip":
            f_raw = torch.clamp(f_raw, -1.0, 1.0)
        f_ms = self.norm_ms(self.mixer(f_raw))
        alpha = torch.sigmoid(self.gate(torch.cat([f_c, f_ms], dim=1)))
        self._track(alpha)
        f_c_mod = _apply_branch_droppath(f_c, self.drop_c_prob, self.training)
        f_fused = torch.cat([f_c_mod, alpha * f_ms], dim=1)
        return f_fused, alpha


class WorkbenchBackend:
    """Lazy-caching CPU-only read-only backend for 7-branch inference and thesis reports."""

    def __init__(self) -> None:
        self._test_splits: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._luts: Dict[Tuple[str, str], torch.Tensor] = {}
        self._models: Dict[Tuple[str, str, int], torch.nn.Module] = {}

    def _get_test_split(self, ds: str) -> Tuple[np.ndarray, np.ndarray]:
        if ds not in self._test_splits:
            with np.load(str(PROJECT_ROOT / "data" / f"{ds}.npz")) as d:
                te = np.ascontiguousarray(d["test_images"], dtype=np.uint8)
                yte = d["test_labels"].flatten().astype(np.int64)
            self._test_splits[ds] = (te, yte)
        return self._test_splits[ds]

    def get_num_samples(self, ds: str) -> int:
        te, _ = self._get_test_split(ds)
        return int(len(te))

    def _get_lut(self, model_key: str, lut_mode: str) -> torch.Tensor:
        key = (model_key, lut_mode)
        if key not in self._luts:
            suffix = "fp32_grid.bin" if lut_mode == "interp4d" else "int8_shift.bin"
            if model_key == "qlut":
                path = PROJECT_ROOT / "weights" / f"qlut_strongly_L2_4bit_{suffix}"
            else:
                path = PROJECT_ROOT / "weights" / "ablation_clut" / f"{model_key}_4bit_{suffix}"
            self._luts[key] = load_lut_table(path, mode=lut_mode, device=DEV)
        return self._luts[key]

    def _get_model(self, ds: str, model_key: str, seed: int = 42) -> torch.nn.Module:
        key = (ds, model_key, seed)
        if key in self._models:
            return self._models[key]

        nc = 2
        if model_key == "ctrl":
            cp = PROJECT_ROOT / "weights" / "pillar2_wide" / f"wide_v2_interp4d_{ds}_full_control_s{seed}.pt"
            m: torch.nn.Module = WideClassicalControl(num_classes=nc, drop_c_prob=DEFAULT_DROP_C_PROB).to(DEV)
        elif model_key == "qlut":
            cp = PROJECT_ROOT / "weights" / "pillar2_wide" / f"wide_v2_interp4d_{ds}_full_grquanv_s{seed}.pt"
            m = WideGRQuanvNet(num_classes=nc, drop_c_prob=DEFAULT_DROP_C_PROB).to(DEV)
        elif model_key in ("clut_septrig", "clut_randmlp"):
            cp = PROJECT_ROOT / "weights" / "ablation_clut_models" / f"ablation_{model_key}_{ds}_full_s{seed}.pt"
            m = WideGRQuanvNet(num_classes=nc, drop_c_prob=DEFAULT_DROP_C_PROB).to(DEV)
        elif model_key == "ctrl_tanh":
            cp = PROJECT_ROOT / "weights" / "ablation_clut_extra" / f"extra_{model_key}_{ds}_full_s{seed}.pt"
            m = WideClassicalControlBounded(nc, drop_c_prob=DEFAULT_DROP_C_PROB, bound_mode="tanh").to(DEV)
        elif model_key == "ctrl_hardclip":
            cp = PROJECT_ROOT / "weights" / "ablation_clut_extra" / f"extra_{model_key}_{ds}_full_s{seed}.pt"
            m = WideClassicalControlBounded(nc, drop_c_prob=DEFAULT_DROP_C_PROB, bound_mode="hardclip").to(DEV)
        elif model_key == "ctrl_noise_aug":
            cp = PROJECT_ROOT / "weights" / "ablation_clut_extra" / f"extra_{model_key}_{ds}_full_s{seed}.pt"
            m = WideClassicalControl(nc, drop_c_prob=DEFAULT_DROP_C_PROB).to(DEV)
        else:
            raise ValueError(f"Unknown model_key: {model_key}")

        m.load_state_dict(torch.load(cp, map_location=DEV, weights_only=True))
        m.eval()
        self._models[key] = m
        return m

    @torch.no_grad()
    def _extract_raw_4ch(
        self,
        model_key: str,
        model: torch.nn.Module,
        input_u8_1: np.ndarray,
        x_t: torch.Tensor,
        lut_mode: str,
    ) -> Tuple[Optional[torch.Tensor], np.ndarray]:
        if model_key in ("qlut", "clut_septrig", "clut_randmlp"):
            tbl = self._get_lut(model_key, lut_mode)
            q_buf = extract_multiscale_from_lut_tensor(
                input_u8_1, tbl, mode=lut_mode, phase_damping=False, chunk=1
            )
            q_t = (q_buf.float() / 127.0) if q_buf.dtype == torch.int8 else q_buf.float()
            return q_t, q_t[0, 0:4].numpy()

        f_raw0 = model.classical_branches[0](x_t)[:, :, :14, :14]
        bound_mode = getattr(model, "bound_mode", "none")
        if bound_mode == "tanh":
            f_raw0 = torch.tanh(f_raw0)
        elif bound_mode == "hardclip":
            f_raw0 = torch.clamp(f_raw0, -1.0, 1.0)
        return None, f_raw0[0].numpy()

    @torch.no_grad()
    def inspect_sample(
        self,
        ds: str,
        sample_idx: int,
        sigma: float,
        use_damping: bool,
        lut_mode: str,
        primary_key: str = "qlut",
        compare_key: str = "ctrl_tanh",
        seed: int = 42,
        noise_draw: int = 1,
    ) -> dict:
        te, yte = self._get_test_split(ds)
        n_test = len(te)
        idx = max(0, min(int(sample_idx), n_test - 1))

        clean_u8 = te[idx : idx + 1]
        true_label = int(yte[idx])
        class_names = CLASS_NAMES_MAP[ds]

        if sigma > 0.0:
            clean_f = clean_u8.astype(np.float32) / 255.0
            noisy_u8 = speckle(clean_f, float(sigma), noise_draw)
        else:
            noisy_u8 = clean_u8.copy()

        input_u8 = phase_damping_u8(noisy_u8, beta=18.0, tau=0.024) if use_damping else noisy_u8
        x_t = ((torch.from_numpy(np.ascontiguousarray(input_u8)).float().unsqueeze(1) / 255.0) - 0.5) / 0.5
        tau_val = 0.024 if use_damping else 999.0

        predictions: List[dict] = []
        primary_fmaps = None
        compare_fmaps = None
        primary_alpha_map = None

        for mk, label, _ in MODEL_SPECS:
            m = self._get_model(ds, mk, seed=seed)
            m.stabilize_tau = tau_val
            q_t, fmaps4 = self._extract_raw_4ch(mk, m, input_u8, x_t, lut_mode)

            f_fused, alpha = m.forward_features(x_t, q_t) if q_t is not None else m.forward_features(x_t)
            logits = m.head(f_fused)
            probs = torch.softmax(logits, dim=1)[0].numpy()
            pred_idx = int(np.argmax(probs))
            alpha_mean = float(alpha.mean().item())

            if mk == primary_key:
                primary_fmaps = fmaps4
                primary_alpha_map = alpha[0].mean(dim=0).numpy()
            if mk == compare_key:
                compare_fmaps = fmaps4

            predictions.append(
                {
                    "key": mk,
                    "label": label,
                    "pred_idx": pred_idx,
                    "pred_name": class_names[pred_idx],
                    "prob_pos": float(probs[1]),
                    "prob_pred": float(probs[pred_idx]),
                    "correct": pred_idx == true_label,
                    "alpha_mean": alpha_mean,
                }
            )

        return {
            "ds": ds,
            "sample_idx": idx,
            "n_test": n_test,
            "sigma": float(sigma),
            "use_damping": bool(use_damping),
            "lut_mode": lut_mode,
            "true_label": true_label,
            "true_name": class_names[true_label],
            "class_names": class_names,
            "img_clean": clean_u8[0],
            "img_noisy": noisy_u8[0],
            "img_input": input_u8[0],
            "primary_key": primary_key,
            "compare_key": compare_key,
            "primary_fmaps": primary_fmaps,
            "compare_fmaps": compare_fmaps,
            "alpha_map": primary_alpha_map,
            "predictions": predictions,
        }


def _read_json(rel: str) -> Optional[dict]:
    p = PROJECT_ROOT / rel
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_text(rel: str) -> str:
    p = PROJECT_ROOT / rel
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8", errors="replace")


def load_recorded_data() -> Dict[str, Any]:
    return {
        "jetson_v2_10w": _read_json("reports/jetson_nano_hardware/jetson_v2_10w.json"),
        "jetson_v2_5w": _read_json("reports/jetson_nano_hardware/jetson_v2_5w.json"),
        "jetson_v3_10w": _read_json("reports/jetson_nano_hardware/jetson_v3_10w.json"),
        "jetson_v3_5w": _read_json("reports/jetson_nano_hardware/jetson_v3_5w.json"),
        "jetson_raw_logs": _read_json("reports/jetson_nano_hardware/jetson_raw_logs.json"),
        "jetson_rss": _read_text("reports/jetson_nano_hardware/jetson_v2_10w_rss.txt"),
        "tegrastats_10w": _read_text("reports/jetson_nano_hardware/tegrastats_10w.txt"),
        "tegrastats_5w": _read_text("reports/jetson_nano_hardware/tegrastats_5w.txt"),
        "x86_v2": _read_json("reports/medquanv_cli_verification_a3.json"),
        "x86_v3": _read_json("reports/medquanv_cli_v3_verification.json"),
        "rigorous_stats": _read_json("reports/rigorous_stats_summary.json"),
        "pareto": _read_json("reports/pareto_int8_vs_interp4d.json"),
        "highres": _read_json("reports/highres_scalability_results.json"),
    }


def run_live_jetson_probe(
    host: str = "192.168.110.51",
    user: str = "nguyen",
    password: str = "123456",
    power_mode: int = 0,
) -> dict:
    """Runs a quick live Jetson Nano B01 probe via SSH without overwriting recorded reports."""
    import paramiko

    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(host, username=user, password=password, timeout=10)

    mode_flag = 0 if power_mode == 0 else 1
    cmd = (
        f"echo {password} | sudo -S nvpmodel -m {mode_flag} 2>/dev/null; "
        f"echo {password} | sudo -S nvpmodel -q 2>/dev/null | grep -E 'NV Power Mode|^[01]$'; "
        "cd /home/nguyen/medquanv_bench && "
        "./rust_qlut/target/release/medquanv-cli --bench-resolutions; "
        f"echo {password} | sudo -S timeout 2 tegrastats --interval 500 || true"
    )
    _, o, e = c.exec_command(cmd, timeout=45)
    out = o.read().decode(errors="replace")
    err = e.read().decode(errors="replace")
    rc = o.channel.recv_exit_status()
    if mode_flag != 0:
        c.exec_command(f"echo {password} | sudo -S nvpmodel -m 0 2>/dev/null")
    c.close()
    return {"rc": rc, "stdout": out, "stderr": err}
