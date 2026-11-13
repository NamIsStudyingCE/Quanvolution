"""
Phase B (GR-Quanv v3) Full 5-Seed Evaluation Across BreastMNIST, PneumoniaMNIST, OCTMNIST
=========================================================================================
Strictly compares (Clean-Only Training, Symmetric Phase Damping, Paired 5 Seeds):
  1. ctrl_v2_full    : Locked WideClassicalControl v2 (18.90M MACs, 148.4K params)
  2. ctrl_v3_light   : Fair-Play Equal-MAC/Topology ClassicalControlV3Light (4.07M MACs, 39.5K params)
  3. quanv_v3_int8   : Proposed GRQuanvNetV3 (Trainable Variational Q-LUT 256 KiB INT8 QAT, 4.03M MACs, 39.3K params)
  4. micro_ctrl      : Micro-Edge Classical Control (0.186M MACs, 2.08K params)
  5. micro_quanv     : Micro-Edge Trainable INT8 Q-LUT (0.169M MACs, 2.00K params + 256 KiB LUT)

Outputs (Resumable per (ds, kind, seed)):
  - Weights: weights/pillar3_v3/{ds}_{kind}_s{seed}.pt
  - Compiled INT8 Shift-LUT binaries (seed=42): weights/v3_{ds}_s42_int8_shift.bin
  - Results JSON: reports/pillar3_v3_results.json
  - Pareto & Degradation Figure: reports/figures/v3_vs_cnn_pareto.png
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import stats

torch.set_float32_matmul_precision("high")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from models.gr_quanv import phase_damping_u8
from models.gr_quanv_resblock import GPUMedicalAugment
from models.gr_quanv_v3 import (
    ClassicalControlV3Light,
    GRQuanvNetV3,
    MicroEdgeNetV3,
    count_inference_macs,
)
from models.gr_quanv_wide import WideClassicalControl
from run_pillar2_expansion import DATASETS, NOISE_DRAWS, SEEDS, SIGMAS, speckle
from run_pillar2_wide import WARMUP_EPOCHS, WIDE_HP, load_dataset_with_cache
from utils.metrics import calculate_metrics

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_DIR = PROJECT_ROOT / "weights" / "pillar3_v3"
OUT_JSON = PROJECT_ROOT / "reports" / "pillar3_v3_results.json"
FIG_PATH = PROJECT_ROOT / "reports" / "figures" / "v3_vs_cnn_pareto.png"

ALL_KINDS = ["ctrl_v2_full", "ctrl_v3_light", "quanv_v3_int8", "micro_ctrl", "micro_quanv"]
TRAINABLE_KINDS = ["ctrl_v3_light", "quanv_v3_int8", "micro_ctrl", "micro_quanv"]


def build_v3_model(kind: str, nc: int) -> nn.Module:
    if kind == "ctrl_v2_full":
        return WideClassicalControl(num_classes=nc)
    if kind == "ctrl_v3_light":
        return ClassicalControlV3Light(num_classes=nc, drop_c_prob=0.20, width1=64, width2=96)
    if kind == "quanv_v3_int8":
        return GRQuanvNetV3(num_classes=nc, n_tables=1, qat_int8=True, drop_c_prob=0.20, use_stage2_qlut=True, width1=64, width2=96)
    if kind == "micro_ctrl":
        return MicroEdgeNetV3(num_classes=nc, classical=True)
    if kind == "micro_quanv":
        return MicroEdgeNetV3(num_classes=nc, classical=False)
    raise ValueError(f"Unknown kind: {kind}")


def make_opt_and_sched(model: nn.Module, epochs: int, lr: float, wd: float):
    q_params, other_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "qlut.theta" in name:
            q_params.append(p)
        else:
            other_params.append(p)
    groups = [{"params": other_params, "lr": lr, "weight_decay": wd}]
    if q_params:
        groups.append({"params": q_params, "lr": lr * 4.0, "weight_decay": 0.0})
    opt = optim.AdamW(groups)
    main = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs - WARMUP_EPOCHS))
    warm = optim.lr_scheduler.LinearLR(opt, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_EPOCHS)
    sched = optim.lr_scheduler.SequentialLR(opt, [warm, main], milestones=[WARMUP_EPOCHS])
    return opt, sched


@torch.no_grad()
def eval_split(model: nn.Module, xu8_gpu: torch.Tensor, y_np: np.ndarray, nc: int, compiled_tbl=None, bs: int = 1024):
    model.eval()
    probs = []
    is_qlut = isinstance(model, GRQuanvNetV3) or (isinstance(model, MicroEdgeNetV3) and not model.classical)
    for i in range(0, len(xu8_gpu), bs):
        x = (xu8_gpu[i:i + bs].float().unsqueeze(1) / 255.0 - 0.5) / 0.5
        logits = model(x, compiled_tables=compiled_tbl) if is_qlut else model(x)
        probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
    p = np.concatenate(probs, axis=0)
    return calculate_metrics(y_np, p, p.argmax(1), nc)


def val_score_fn(ds: str, mv: dict) -> float:
    if ds in ("breastmnist", "pneumoniamnist"):
        return 0.5 * float(mv["auc"]) + 0.5 * float(mv["bacc"])
    return float(mv["acc"])


def evaluate_model_all_sigmas(model: nn.Module, D: dict, nc: int, compiled_tbl=None) -> dict:
    te_f = D["te"].astype(np.float32) / 255.0
    yte_np = D["yte"]
    m_clean = eval_split(model, D["g_te"][0], yte_np, nc, compiled_tbl=compiled_tbl)
    speckle_res = {"0.00": {"acc": float(m_clean["acc"] * 100.0), "f1": float(m_clean["f1"] * 100.0), "bacc": float(m_clean["bacc"] * 100.0), "auc": float(m_clean["auc"] * 100.0)}}
    for sg in SIGMAS:
        if sg == 0.0:
            continue
        accs, f1s, baccs = [], [], []
        for dr in NOISE_DRAWS:
            damped_u8 = phase_damping_u8(speckle(te_f, sg, dr))
            mm = eval_split(model, torch.from_numpy(damped_u8).to(DEV), yte_np, nc, compiled_tbl=compiled_tbl)
            accs.append(mm["acc"] * 100.0)
            f1s.append(mm["f1"] * 100.0)
            baccs.append(mm["bacc"] * 100.0)
        speckle_res[f"{sg:.2f}"] = {
            "acc": float(np.mean(accs)),
            "f1": float(np.mean(f1s)),
            "bacc": float(np.mean(baccs)),
            "draw_accs": [float(a) for a in accs],
        }
    return {"clean": {k: float(v) for k, v in m_clean.items()}, "speckle": speckle_res}


def train_and_eval_one(ds: str, kind: str, seed: int, D: dict) -> dict:
    ytr_np, yva_np, yte_np = D["ytr"], D["yva"], D["yte"]
    nc = int(len(np.unique(ytr_np)))
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CKPT_DIR / f"{ds}_{kind}_s{seed}.pt"

    if kind == "ctrl_v2_full":
        locked_cp = PROJECT_ROOT / "weights" / "pillar2_wide" / f"wide_v2_interp4d_{ds}_full_control_s{seed}.pt"
        model = build_v3_model(kind, nc).to(DEV)
        model.load_state_dict(torch.load(locked_cp, map_location=DEV, weights_only=True))
        cx = count_inference_macs(model)
        ev = evaluate_model_all_sigmas(model, D, nc)
        ev["cx"] = cx
        ev["train_sec"] = 0.0
        ev["best_ep"] = -1
        ev["checkpoint"] = str(locked_cp.relative_to(PROJECT_ROOT)).replace("\\", "/")
        print(f"  [{ds:<14} | {kind:<14} | s{seed:<4}] 0.00={ev['speckle']['0.00']['acc']:5.2f}% 0.10={ev['speckle']['0.10']['acc']:5.2f}% 0.20={ev['speckle']['0.20']['acc']:5.2f}% (locked)", flush=True)
        return ev

    epochs, bs, lr, wd = WIDE_HP[ds]
    _, _, tau, _ = DATASETS[ds]
    torch.manual_seed(seed); np.random.seed(seed); torch.cuda.manual_seed_all(seed)
    model = build_v3_model(kind, nc).to(DEV)
    cx = count_inference_macs(model)
    is_qlut = isinstance(model, GRQuanvNetV3) or (isinstance(model, MicroEdgeNetV3) and not model.classical)

    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=DEV, weights_only=True))
        train_s, best_ep = 0.0, -1
    else:
        opt, sched = make_opt_and_sched(model, epochs, lr, wd)
        crit = nn.CrossEntropyLoss()
        counts = np.bincount(ytr_np, minlength=nc)
        logp = torch.log(torch.tensor(counts / counts.sum(), dtype=torch.float32, device=DEV) + 1e-8)

        g = torch.Generator(device=DEV); g.manual_seed(seed)
        aug = GPUMedicalAugment()
        xtr_u8, xva_u8 = D["g_tr"][0], D["g_va"][0]
        ytr = torch.from_numpy(ytr_np).to(DEV)
        N = len(ytr_np)

        best_sc, best_ep, best_state = -1.0, -1, None
        t0 = time.perf_counter()
        use_amp = (DEV.type == "cuda")
        for ep in range(epochs):
            xn = (xtr_u8.float().unsqueeze(1) / 255.0 - 0.5) / 0.5
            xe_u8 = torch.from_numpy(np.concatenate([aug(xn[i:i + 8192], g) for i in range(0, len(xn), 8192)])).to(DEV)
            model.train()
            perm = torch.randperm(N, device=DEV, generator=g)
            for i in range(0, N, bs):
                idx = perm[i:i + bs]
                if len(idx) < 2:
                    continue
                xb = (xe_u8[idx].float().unsqueeze(1) / 255.0 - 0.5) / 0.5
                opt.zero_grad()
                if use_amp:
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        logits = model(xb)
                else:
                    logits = model(xb)
                loss = crit(logits.float() + tau * logp, ytr[idx])
                if is_qlut:
                    loss = loss + 0.002 * model.cached_smoothness_loss()
                loss.backward()
                opt.step()
            sched.step()
            mv = eval_split(model, xva_u8, yva_np, nc)
            sc = val_score_fn(ds, mv)
            if sc > best_sc:
                best_sc, best_ep = sc, ep
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        train_s = time.perf_counter() - t0
        model.load_state_dict(best_state)
        torch.save(best_state, ckpt_path)

    compiled_tbl = None
    compile_max_diff = 0.0
    if is_qlut:
        tbl_np = model.qlut.compile_int8_shift_tables()
        compiled_tbl = torch.from_numpy(tbl_np).to(DEV)
        m_qat = eval_split(model, D["g_te"][0], yte_np, nc, compiled_tbl=None)
        m_cmp = eval_split(model, D["g_te"][0], yte_np, nc, compiled_tbl=compiled_tbl)
        compile_max_diff = float(abs(m_qat["acc"] - m_cmp["acc"]))
        if seed == 42 and kind == "quanv_v3_int8":
            bin_path = PROJECT_ROOT / "weights" / f"v3_{ds}_s42_int8_shift.bin"
            tbl_np[0].astype(np.int8).tofile(bin_path)

    ev = evaluate_model_all_sigmas(model, D, nc, compiled_tbl=compiled_tbl)
    ev["cx"] = cx
    ev["train_sec"] = float(train_s)
    ev["best_ep"] = int(best_ep)
    ev["compile_acc_diff"] = compile_max_diff
    ev["checkpoint"] = str(ckpt_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    print(
        f"  [{ds:<14} | {kind:<14} | s{seed:<4}] 0.00={ev['speckle']['0.00']['acc']:5.2f}% "
        f"0.05={ev['speckle']['0.05']['acc']:5.2f}% 0.10={ev['speckle']['0.10']['acc']:5.2f}% "
        f"0.15={ev['speckle']['0.15']['acc']:5.2f}% 0.20={ev['speckle']['0.20']['acc']:5.2f}% | "
        f"ep*={best_ep:2d} ({train_s:.1f}s) cmp_diff={compile_max_diff:.1e}",
        flush=True,
    )
    return ev


def aggregate_and_stats(data: dict):
    agg = {}
    paired_tests = {}
    for ds, ds_dict in data.get("per_seed", {}).items():
        agg[ds] = {}
        for kind, seeds_dict in ds_dict.items():
            if len(seeds_dict) == 0:
                continue
            agg[ds][kind] = {"cx": next(iter(seeds_dict.values()))["cx"], "speckle": {}}
            for sg in [f"{s:.2f}" for s in SIGMAS]:
                accs = [seeds_dict[str(s)]["speckle"][sg]["acc"] for s in SEEDS if str(s) in seeds_dict]
                f1s = [seeds_dict[str(s)]["speckle"][sg]["f1"] for s in SEEDS if str(s) in seeds_dict]
                baccs = [seeds_dict[str(s)]["speckle"][sg]["bacc"] for s in SEEDS if str(s) in seeds_dict]
                agg[ds][kind]["speckle"][sg] = {
                    "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs, ddof=1) if len(accs) > 1 else 0.0),
                    "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s, ddof=1) if len(f1s) > 1 else 0.0),
                    "bacc_mean": float(np.mean(baccs)), "bacc_std": float(np.std(baccs, ddof=1) if len(baccs) > 1 else 0.0),
                    "n": len(accs),
                }

        paired_tests[ds] = {}
        pairs = [
            ("quanv_v3_int8_vs_ctrl_v2_full", "quanv_v3_int8", "ctrl_v2_full"),
            ("quanv_v3_int8_vs_ctrl_v3_light", "quanv_v3_int8", "ctrl_v3_light"),
            ("micro_quanv_vs_micro_ctrl", "micro_quanv", "micro_ctrl"),
        ]
        for label, ka, kb in pairs:
            if ka not in ds_dict or kb not in ds_dict:
                continue
            common = [str(s) for s in SEEDS if str(s) in ds_dict[ka] and str(s) in ds_dict[kb]]
            if len(common) < 2:
                continue
            paired_tests[ds][label] = {}
            for sg in [f"{s:.2f}" for s in SIGMAS]:
                va = np.array([ds_dict[ka][s]["speckle"][sg]["acc"] for s in common])
                vb = np.array([ds_dict[kb][s]["speckle"][sg]["acc"] for s in common])
                diff = va - vb
                if np.allclose(diff, 0.0):
                    p_t = 1.0
                else:
                    _, p_t = stats.ttest_rel(va, vb)
                paired_tests[ds][label][sg] = {
                    "mean_diff": float(np.mean(diff)),
                    "std_diff": float(np.std(diff, ddof=1)),
                    "p_ttest": float(p_t),
                }
    data["aggregated"] = agg
    data["stats"] = paired_tests


def plot_pareto_and_degradation(data: dict):
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    agg = data.get("aggregated", {})
    datasets = [d for d in ["breastmnist", "pneumoniamnist", "octmnist"] if d in agg]
    if not datasets:
        return

    fig, axes = plt.subplots(1, len(datasets), figsize=(5.5 * len(datasets), 4.6), dpi=160)
    if len(datasets) == 1:
        axes = [axes]

    styles = {
        "ctrl_v2_full": ("WideClassicalControl v2 (18.9M MAC)", "#4c566a", "o", "--"),
        "ctrl_v3_light": ("ClassicalControlV3Light (4.07M MAC)", "#2e86ab", "s", "-."),
        "quanv_v3_int8": ("GR-Quanv v3 INT8-QAT (4.03M MAC, 256KiB)", "#d62828", "^", "-"),
        "micro_ctrl": ("Micro-Classical (0.19M MAC)", "#f77f00", "v", ":"),
        "micro_quanv": ("Micro-Quanv INT8-QAT (0.17M MAC)", "#7209b7", "D", "-"),
    }
    sig_vals = [float(s) for s in SIGMAS]

    for ax, ds in zip(axes, datasets):
        for kind, (lbl, col, mk, ls) in styles.items():
            if kind not in agg[ds]:
                continue
            ys = [agg[ds][kind]["speckle"][f"{s:.2f}"]["acc_mean"] for s in sig_vals]
            es = [agg[ds][kind]["speckle"][f"{s:.2f}"]["acc_std"] for s in sig_vals]
            ax.errorbar(sig_vals, ys, yerr=es, label=lbl, color=col, marker=mk, linestyle=ls, linewidth=1.8, capsize=3)
        ax.set_title(ds.upper(), fontweight="bold")
        ax.set_xlabel("Speckle Noise Intensity $\\sigma$")
        ax.set_ylabel("Accuracy (%)")
        ax.grid(True, alpha=0.3)
        ax.set_xticks(sig_vals)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=3, fontsize=8.5, frameon=True)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(FIG_PATH, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved Figure] {FIG_PATH}", flush=True)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["breastmnist", "pneumoniamnist", "octmnist"])
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--kinds", nargs="+", default=ALL_KINDS)
    args = parser.parse_args()

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    if OUT_JSON.exists():
        with open(OUT_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {
            "metadata": {
                "version": "pillar3_v3_qat_int8_256kib",
                "device": str(DEV),
                "sigmas": SIGMAS,
                "noise_draws": NOISE_DRAWS,
                "seeds": SEEDS,
            },
            "per_seed": {},
            "aggregated": {},
            "stats": {},
        }

    for ds in args.datasets:
        print(f"\n=======================================================", flush=True)
        print(f"=== PHASE B 5-SEED BENCHMARK: {ds.upper()} ===", flush=True)
        print(f"=======================================================", flush=True)
        D = load_dataset_with_cache(ds, interp4d=True)
        data["per_seed"].setdefault(ds, {})
        for kind in args.kinds:
            data["per_seed"][ds].setdefault(kind, {})
            for seed in args.seeds:
                if str(seed) in data["per_seed"][ds][kind]:
                    ev = data["per_seed"][ds][kind][str(seed)]
                    print(f"  [{ds:<14} | {kind:<14} | s{seed:<4}] CACHED 0.00={ev['speckle']['0.00']['acc']:5.2f}% 0.20={ev['speckle']['0.20']['acc']:5.2f}%", flush=True)
                    continue
                ev = train_and_eval_one(ds, kind, seed, D)
                data["per_seed"][ds][kind][str(seed)] = ev
                aggregate_and_stats(data)
                with open(OUT_JSON, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)

    aggregate_and_stats(data)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    plot_pareto_and_degradation(data)
    print(f"\n[DONE] Saved full 5-seed Phase B results to {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
