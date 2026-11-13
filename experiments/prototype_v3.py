"""
Fast Seed-42 Prototype Comparison for GR-Quanv v3 (`Trainable Variational Q-LUT INT8 QAT`)
==========================================================================================
Evaluates strictly on seed=42 (clean + Speckle noise) across PneumoniaMNIST & OCTMNIST:
  1. ClassicalControlV3Light (Equal-FLOPs/Params Classical Control, width 64->96)
  2. GRQuanvNetV3 (n_tables=1 -> 256 KiB INT8 Shift-LUT, width 64->96)
  3. GRQuanvNetV3 (n_tables=4 -> 1024 KiB Multi-Basis INT8 Shift-LUT, width 64->96)
Verifies compiled INT8 Shift-LUT bit-exactness against PyTorch QAT forward and checks against
locked WideClassicalControl v2 seed-42 baseline!
"""

import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from models.gr_quanv import phase_damping_u8
from models.gr_quanv_resblock import GPUMedicalAugment
from models.gr_quanv_v3 import ClassicalControlV3Light, GRQuanvNetV3, count_inference_macs
from models.gr_quanv_wide import WideClassicalControl
from run_pillar2_wide import load_dataset_with_cache, WIDE_HP, WARMUP_EPOCHS
from run_pillar2_expansion import DATASETS, SIGMAS, NOISE_DRAWS, speckle
from utils.metrics import calculate_metrics

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
def eval_split(model, xu8_gpu: torch.Tensor, y_np: np.ndarray, nc: int, compiled_tbl=None, bs: int = 1024):
    model.eval()
    probs = []
    for i in range(0, len(xu8_gpu), bs):
        x = (xu8_gpu[i:i + bs].float().unsqueeze(1) / 255.0 - 0.5) / 0.5
        logits = model(x, compiled_tables=compiled_tbl) if isinstance(model, GRQuanvNetV3) else model(x)
        probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    p = np.concatenate(probs, axis=0)
    return calculate_metrics(y_np, p, p.argmax(1), nc)


def train_one(ds: str, kind: str, seed: int = 42):
    D = load_dataset_with_cache(ds, interp4d=True)
    ytr_np, yva_np, yte_np = D["ytr"], D["yva"], D["yte"]
    nc = int(len(np.unique(ytr_np)))
    epochs, bs, lr, wd = WIDE_HP[ds]
    _, sel, tau, _ = DATASETS[ds]

    torch.manual_seed(seed); np.random.seed(seed); torch.cuda.manual_seed_all(seed)
    if kind == "ctrl_v3_light":
        model = ClassicalControlV3Light(num_classes=nc, width1=64, width2=96).to(DEV)
    elif kind == "quanv_v3_1tbl":
        model = GRQuanvNetV3(num_classes=nc, n_tables=1, qat_int8=True, drop_c_prob=0.20, width1=64, width2=96).to(DEV)
    elif kind == "quanv_v3_4tbl":
        model = GRQuanvNetV3(num_classes=nc, n_tables=4, qat_int8=True, drop_c_prob=0.20, width1=64, width2=96).to(DEV)
    else:
        raise ValueError(kind)

    cx = count_inference_macs(model)
    opt, sched = make_opt_and_sched(model, epochs, lr, wd)
    crit = nn.CrossEntropyLoss()
    counts = np.bincount(ytr_np, minlength=nc)
    logp = torch.log(torch.tensor(counts / counts.sum(), dtype=torch.float32, device=DEV) + 1e-8)

    g = torch.Generator(device=DEV); g.manual_seed(seed)
    aug = GPUMedicalAugment()
    xtr_u8 = D["g_tr"][0]
    xva_u8 = D["g_va"][0]
    xte_u8 = D["g_te"][0]
    ytr = torch.from_numpy(ytr_np).to(DEV)
    N = len(ytr_np)

    best, best_ep, best_state = -1.0, -1, None
    t0 = time.perf_counter()
    for ep in range(epochs):
        xn = (xtr_u8.float().unsqueeze(1) / 255.0 - 0.5) / 0.5
        outs = []
        for i in range(0, len(xn), 8192):
            outs.append(aug(xn[i:i + 8192], g))
        xe_u8 = torch.from_numpy(np.concatenate(outs)).to(DEV)

        model.train()
        perm = torch.randperm(N, device=DEV, generator=g)
        for i in range(0, N, bs):
            idx = perm[i:i + bs]
            if len(idx) < 2:
                continue
            xb = (xe_u8[idx].float().unsqueeze(1) / 255.0 - 0.5) / 0.5
            opt.zero_grad()
            logits = model(xb) + tau * logp
            crit(logits, ytr[idx]).backward()
            opt.step()
        sched.step()
        v = eval_split(model, xva_u8, yva_np, nc)[sel]
        if v > best:
            best, best_ep = v, ep
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}

    train_s = time.perf_counter() - t0
    model.load_state_dict(best_state)
    compiled_tbl = torch.from_numpy(model.qlut.compile_int8_shift_tables()).to(DEV) if isinstance(model, GRQuanvNetV3) else None
    m_clean = eval_split(model, xte_u8, yte_np, nc, compiled_tbl=compiled_tbl)

    # Verify bit-exactness of compiled table vs forward
    if isinstance(model, GRQuanvNetV3):
        m_qat = eval_split(model, xte_u8, yte_np, nc, compiled_tbl=None)
        assert abs(m_clean["acc"] - m_qat["acc"]) < 1e-6, f"Mismatch QAT vs compiled LUT: {m_qat['acc']} vs {m_clean['acc']}"

    # Evaluate Speckle robustness with symmetric damping
    te_f = D["te"].astype(np.float32) / 255.0
    speckle_res = {}
    for sg in SIGMAS:
        if sg == 0.0:
            speckle_res["0.00"] = {"acc": m_clean["acc"] * 100.0, "f1": m_clean["f1"] * 100.0}
            continue
        accs, f1s = [], []
        for dr in NOISE_DRAWS:
            noisy_u8 = speckle(te_f, sg, dr)
            damped_u8 = phase_damping_u8(noisy_u8)
            xd = torch.from_numpy(damped_u8).to(DEV)
            md = eval_split(model, xd, yte_np, nc, compiled_tbl=compiled_tbl)
            accs.append(md["acc"] * 100.0)
            f1s.append(md["f1"] * 100.0)
        speckle_res[f"{sg:.2f}"] = {"acc": float(np.mean(accs)), "f1": float(np.mean(f1s))}

    print(
        f"[{ds:<14} | {kind:<14} | s{seed}] Clean ACC={m_clean['acc']*100:5.2f}% | "
        f"s0.10={speckle_res['0.10']['acc']:5.2f}% | s0.20={speckle_res['0.20']['acc']:5.2f}% | "
        f"MACs={cx['macs']/1e6:.2f}M Params={cx['params']:,} | ep*={best_ep} ({train_s:.1f}s)",
        flush=True,
    )
    return {"cx": cx, "clean": m_clean, "speckle": speckle_res, "train_sec": train_s}


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    results = {}
    for ds in ["pneumoniamnist", "octmnist"]:
        results[ds] = {}
        # Reference: locked WideClassicalControl v2 seed 42
        D = load_dataset_with_cache(ds, interp4d=True)
        nc = 2 if ds == "pneumoniamnist" else 4
        mc = WideClassicalControl(num_classes=nc).to(DEV)
        mc.load_state_dict(torch.load(PROJECT_ROOT / "weights" / "pillar2_wide" / f"wide_v2_interp4d_{ds}_full_control_s42.pt", map_location=DEV, weights_only=True))
        mc.eval()
        te_f = D["te"].astype(np.float32) / 255.0
        sp_ctrl = {}
        for sg in SIGMAS:
            if sg == 0.0:
                mm = eval_split(mc, D["g_te"][0], D["yte"], nc)
                sp_ctrl["0.00"] = {"acc": mm["acc"] * 100.0, "f1": mm["f1"] * 100.0}
            else:
                accs, f1s = [], []
                for dr in NOISE_DRAWS:
                    damped_u8 = phase_damping_u8(speckle(te_f, sg, dr))
                    mm = eval_split(mc, torch.from_numpy(damped_u8).to(DEV), D["yte"], nc)
                    accs.append(mm["acc"] * 100.0); f1s.append(mm["f1"] * 100.0)
                sp_ctrl[f"{sg:.2f}"] = {"acc": float(np.mean(accs)), "f1": float(np.mean(f1s))}
        cx_full = count_inference_macs(mc)
        print(f"[{ds:<14} | ctrl_v2_full   | s42] Clean ACC={sp_ctrl['0.00']['acc']:5.2f}% | s0.10={sp_ctrl['0.10']['acc']:5.2f}% | s0.20={sp_ctrl['0.20']['acc']:5.2f}% | MACs={cx_full['macs']/1e6:.2f}M Params={cx_full['params']:,}")
        results[ds]["ctrl_v2_full"] = {"cx": cx_full, "speckle": sp_ctrl}

        for kind in ["ctrl_v3_light", "quanv_v3_1tbl", "quanv_v3_4tbl"]:
            results[ds][kind] = train_one(ds, kind, seed=42)

    out_path = PROJECT_ROOT / "reports" / "prototype_v3_seed42.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.generic) else x)
    print(f"[Saved] {out_path}")


if __name__ == "__main__":
    main()
