"""Step 3 of C-LUT vs Q-LUT Isolated Ablation:
Controlled clean + speckle robustness evaluation across all 4 models:
  1. ctrl         (Locked WideClassicalControl, 0% LUT)
  2. clut_septrig (Classical Separable Trig Basis LUT)
  3. clut_randmlp (Classical Random Orthogonal MLP LUT)
  4. qlut         (Locked WideGRQuanvNet, StronglyEntanglingLayers L=2 Q-LUT)

Evaluated strictly on identical corrupted image tensors across:
- Stem conditions:
    * raw_stem        (phase_damping=False, stabilize_tau=999.0)
    * stabilized_stem (symmetric phase_damping_u8 on BOTH c_stem and LUT branches, stabilize_tau=0.024)
- LUT precision modes:
    * interp4d        (FP32 4D multilinear grid, 1,024 KiB - matches training protocol)
    * int8_shift      (INT8 midpoint shift LUT, 256 KiB - matches AVX2/NEON rust_qlut deployment)

Mandatory guardrails:
- Reads locked checkpoints ONLY (read-only) from weights/pillar2_wide/.
- Writes evaluation metrics ONLY to results/ablation_clut/eval_results.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_pillar2_expansion import NOISE_DRAWS, SEEDS, SIGMAS, speckle
from experiments.run_pillar2_wide import DATASETS, DEV, load

SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.gr_quanv import phase_damping_u8  # noqa: E402
from models.gr_quanv_wide import WideClassicalControl, WideGRQuanvNet  # noqa: E402
from scripts.generate_clut import (  # noqa: E402
    extract_multiscale_from_lut_tensor,
    load_lut_table,
)
from utils.metrics import calculate_metrics  # noqa: E402

LOCKED_WIDE_DIR = PROJECT_ROOT / "weights" / "pillar2_wide"
CLUT_MODEL_DIR = PROJECT_ROOT / "weights" / "ablation_clut_models"
CLUT_BIN_DIR = PROJECT_ROOT / "weights" / "ablation_clut"
WEIGHTS_DIR = PROJECT_ROOT / "weights"
OUT_DIR = PROJECT_ROOT / "results" / "ablation_clut"

MODELS_ORDER = ["ctrl", "clut_septrig", "clut_randmlp", "qlut"]
STEM_CONDS = ["raw_stem", "stabilized_stem"]
LUT_MODES = ["interp4d", "int8_shift"]


def _bin_path(model_key: str, lut_mode: str) -> Path:
    suffix = "fp32_grid.bin" if lut_mode == "interp4d" else "int8_shift.bin"
    if model_key == "qlut":
        return WEIGHTS_DIR / f"qlut_strongly_L2_4bit_{suffix}"
    if model_key in ("clut_septrig", "clut_randmlp"):
        return CLUT_BIN_DIR / f"{model_key}_4bit_{suffix}"
    raise ValueError(f"Unknown LUT model_key={model_key}")


def _ckpt_path(model_key: str, ds: str, seed: int) -> Path:
    if model_key == "ctrl":
        return LOCKED_WIDE_DIR / f"wide_v2_interp4d_{ds}_full_control_s{seed}.pt"
    if model_key == "qlut":
        return LOCKED_WIDE_DIR / f"wide_v2_interp4d_{ds}_full_grquanv_s{seed}.pt"
    if model_key in ("clut_septrig", "clut_randmlp"):
        return CLUT_MODEL_DIR / f"ablation_{model_key}_{ds}_full_s{seed}.pt"
    raise ValueError(f"Unknown model_key={model_key}")


def _u8_to_x_tensor(imgs_u8: np.ndarray) -> torch.Tensor:
    xu8 = torch.from_numpy(np.ascontiguousarray(imgs_u8, dtype=np.uint8)).to(DEV)
    return (xu8.float().unsqueeze(1) / 255.0 - 0.5) / 0.5


def _deq_q(q_buf: torch.Tensor) -> torch.Tensor:
    if q_buf.dtype == torch.int8:
        return q_buf.float() / 127.0
    return q_buf.float()


@torch.no_grad()
def evaluate_single_draw(
    model: nn.Module,
    x_tensor: torch.Tensor,
    q_buf: Optional[torch.Tensor],
    y_np: np.ndarray,
    nc: int,
    bs: int = 2048,
) -> Dict[str, float]:
    model.eval()
    probs = []
    alpha_weighted_sum = 0.0
    n_total = len(x_tensor)
    for i in range(0, n_total, bs):
        xb = x_tensor[i : i + bs]
        qb = _deq_q(q_buf[i : i + bs]) if q_buf is not None else None
        logits = model(xb, qb)
        probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        if getattr(model, "last_alpha_mean", None) is not None:
            alpha_weighted_sum += float(np.mean(model.last_alpha_mean)) * len(xb)

    p = np.concatenate(probs, axis=0)
    y_pred = p.argmax(axis=1)
    m = calculate_metrics(y_np, p, y_pred, nc)
    macro_f1 = float(f1_score(y_np, y_pred, average="macro"))
    return {
        "acc": float(m["acc"] * 100.0),
        "macro_f1": float(macro_f1 * 100.0),
        "f1": float(m["f1"] * 100.0),
        "auc": float(m["auc"] * 100.0),
        "alpha_mean": float(alpha_weighted_sum / max(1, n_total)),
    }


def _load_models(ds: str, nc: int, seeds: List[int]) -> Dict[str, Dict[int, nn.Module]]:
    loaded: Dict[str, Dict[int, nn.Module]] = {k: {} for k in MODELS_ORDER}
    expected_lut_params = 148306 if nc == 2 else None
    for s in seeds:
        for mk in MODELS_ORDER:
            cp = _ckpt_path(mk, ds, s)
            if not cp.exists():
                raise FileNotFoundError(f"Missing checkpoint for {mk} {ds} seed={s}: {cp}")
            if mk == "ctrl":
                model: nn.Module = WideClassicalControl(num_classes=nc).to(DEV)
            else:
                model = WideGRQuanvNet(num_classes=nc).to(DEV)
                if expected_lut_params is not None:
                    n_p = sum(p.numel() for p in model.parameters())
                    assert n_p == expected_lut_params, f"Param mismatch on {mk}: {n_p} != {expected_lut_params}"
            model.load_state_dict(torch.load(cp, map_location=DEV, weights_only=True))
            model.eval()
            loaded[mk][s] = model
    return loaded


def evaluate_dataset(
    ds: str,
    seeds: List[int],
    sigmas: List[float],
    lut_modes: List[str],
    stem_conds: List[str],
) -> dict:
    D = load(DATASETS[ds][0])
    te_u8 = D["te"]
    te_f = te_u8.astype(np.float32) / 255.0
    y_test = D["yte"].astype(np.int64)
    nc = int(len(np.unique(y_test)))

    models = _load_models(ds, nc, seeds)

    # Preload LUT lookup tables per precision mode cleanly.
    tables: Dict[str, Dict[str, torch.Tensor]] = {}
    for mode in lut_modes:
        tables[mode] = {
            mk: load_lut_table(_bin_path(mk, mode), mode=mode, device=DEV)
            for mk in ["clut_septrig", "clut_randmlp", "qlut"]
        }

    # Storage: results[lut_mode][stem_cond][sigma_str][model_key]["per_seed"][str(seed)] = {acc, macro_f1, auc, alpha_mean}
    ds_res: dict = {mode: {stem: {} for stem in stem_conds} for mode in lut_modes}

    # Evaluate ctrl once per (stem, sigma) because ctrl has no LUT dependence; copy symmetrically into each lut_mode view.
    for stem in stem_conds:
        tau_val = 999.0 if stem == "raw_stem" else 0.024
        use_damp = stem == "stabilized_stem"
        print(f"\n=== {ds.upper()} | stem={stem} (phase_damping={use_damp}, stabilize_tau={tau_val}) ===")

        for sg in sigmas:
            sg_str = f"{sg:.2f}"
            draws = NOISE_DRAWS if sg > 0 else [0]
            # draw_buf[mode][mk][seed] -> list of draw metric dicts
            draw_buf: Dict[str, Dict[str, Dict[int, List[Dict[str, float]]]]] = {
                mode: {mk: {s: [] for s in seeds} for mk in MODELS_ORDER} for mode in lut_modes
            }

            for dr in draws:
                noisy_u8 = te_u8 if sg == 0.0 else speckle(te_f, sg, dr)
                input_u8 = phase_damping_u8(noisy_u8, beta=18.0, tau=0.024) if use_damp else noisy_u8
                x_t = _u8_to_x_tensor(input_u8)

                # Classical Control evaluated once per (stem, sg, draw), cloned cleanly into all mode tables
                ctrl_draw_res: Dict[int, Dict[str, float]] = {}
                for s in seeds:
                    models["ctrl"][s].stabilize_tau = tau_val
                    ctrl_draw_res[s] = evaluate_single_draw(models["ctrl"][s], x_t, None, y_test, nc)

                for mode in lut_modes:
                    for s in seeds:
                        draw_buf[mode]["ctrl"][s].append(ctrl_draw_res[s])

                    for mk in ["clut_septrig", "clut_randmlp", "qlut"]:
                        q_buf = extract_multiscale_from_lut_tensor(
                            input_u8,
                            tables[mode][mk],
                            mode=mode,
                            phase_damping=False,  # damping already applied symmetrically via input_u8
                        )
                        for s in seeds:
                            models[mk][s].stabilize_tau = tau_val
                            draw_buf[mode][mk][s].append(evaluate_single_draw(models[mk][s], x_t, q_buf, y_test, nc))

            for mode in lut_modes:
                ds_res[mode][stem][sg_str] = {}
                print(f"  -- sigma={sg_str} | mode={mode} --")
                for mk in MODELS_ORDER:
                    per_seed: Dict[str, Dict[str, float]] = {}
                    for s in seeds:
                        items = draw_buf[mode][mk][s]
                        per_seed[str(s)] = {
                            metric: float(np.mean([it[metric] for it in items]))
                            for metric in ["acc", "macro_f1", "f1", "auc", "alpha_mean"]
                        }
                    agg = {}
                    for metric in ["acc", "macro_f1", "f1", "auc", "alpha_mean"]:
                        vals = [per_seed[str(s)][metric] for s in seeds]
                        agg[f"{metric}_mean"] = float(np.mean(vals))
                        agg[f"{metric}_std"] = float(np.std(vals))
                    agg["per_seed"] = per_seed
                    ds_res[mode][stem][sg_str][mk] = agg
                    print(
                        f"    {mk:<14} | ACC={agg['acc_mean']:5.2f}±{agg['acc_std']:4.2f}% | "
                        f"mF1={agg['macro_f1_mean']:5.2f}±{agg['macro_f1_std']:4.2f}% | "
                        f"AUC={agg['auc_mean']:5.2f}% | alpha={agg['alpha_mean_mean']:.3f}"
                    )

    return {
        "nc": nc,
        "n_test": int(len(y_test)),
        "seeds": seeds,
        "sigmas": [float(s) for s in sigmas],
        "results": ds_res,
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Evaluate C-LUT vs Q-LUT vs Ctrl under clean & speckle noise.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["pneumoniamnist"],
        choices=["pneumoniamnist", "breastmnist", "octmnist"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--sigmas", nargs="+", type=float, default=SIGMAS)
    parser.add_argument("--modes", nargs="+", default=LUT_MODES, choices=LUT_MODES)
    parser.add_argument("--stems", nargs="+", default=STEM_CONDS, choices=STEM_CONDS)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = OUT_DIR / "eval_results.json"

    t0 = time.perf_counter()
    payload: dict = {
        "metadata": {
            "seeds": args.seeds,
            "sigmas": args.sigmas,
            "noise_draws": NOISE_DRAWS,
            "modes": args.modes,
            "stems": args.stems,
            "models": MODELS_ORDER,
            "device": str(DEV),
        },
        "datasets": {},
    }

    for ds in args.datasets:
        payload["datasets"][ds] = evaluate_dataset(
            ds=ds,
            seeds=args.seeds,
            sigmas=args.sigmas,
            lut_modes=args.modes,
            stem_conds=args.stems,
        )

    payload["metadata"]["elapsed_sec"] = float(time.perf_counter() - t0)
    tmp = out_json.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(out_json)
    print(f"\n[Saved] {out_json} (elapsed={payload['metadata']['elapsed_sec']:.1f}s)")


if __name__ == "__main__":
    main()
