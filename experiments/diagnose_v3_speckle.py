"""
Step 2 Diagnostic: Pinpointing & Fixing Speckle Noise Sensitivity (s=0.05..0.20) in GRQuanvNetV3.
Tests on PneumoniaMNIST & OCTMNIST (seed=42) with clean-only training (zero noise leakage).
"""
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from models.gr_quanv import _LAP_KERNEL, phase_damping_u8
from models.gr_quanv_wide import WideClassicalControl, _stabilize_stem_variance
from models.gr_quanv_v3 import (
    ClassicalControlV3Light,
    DSResBlock,
    GRQuanvNetV3,
    count_inference_macs,
)
from run_pillar2_expansion import DATASETS, NOISE_DRAWS, SIGMAS, calculate_metrics, speckle
from run_pillar2_wide import GPUMedicalAugment, WARMUP_EPOCHS, WIDE_HP, load_dataset_with_cache
from prototype_v3 import DEV, eval_split, make_opt_and_sched


class GRQuanvV3Candidate(GRQuanvNetV3):
    """Flexible diagnostic variant of GRQuanvNetV3 for clean s42 ablation."""

    def __init__(
        self,
        num_classes: int = 2,
        n_tables: int = 1,
        spatial_mixer: bool = False,
        test_alpha_boost: float = 0.0,
        block2_std: bool = False,
        rand_cdrop_min: float = 0.5,
        drop_c_prob: float = 0.20,
        c_only_noise_clamp: bool = True,
        width1: int = 64,
        width2: int = 96,
    ):
        super().__init__(
            num_classes=num_classes,
            n_tables=n_tables,
            qat_int8=True,
            drop_c_prob=drop_c_prob,
            use_stage2_qlut=True,
            width1=width1,
            width2=width2,
        )
        self.test_alpha_boost = test_alpha_boost
        self.rand_cdrop_min = rand_cdrop_min
        self.c_only_noise_clamp = c_only_noise_clamp

        if spatial_mixer:
            self.q_mixer = nn.Sequential(
                nn.Conv2d(16, 16, kernel_size=3, padding=1, groups=16, bias=False),
                nn.Conv2d(16, 16, kernel_size=1, bias=False),
                nn.BatchNorm2d(16),
            )
            self.s2_proj = nn.Sequential(
                nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32, bias=False),
                nn.Conv2d(32, 4, kernel_size=1, bias=False),
                nn.BatchNorm2d(4),
            )

        if block2_std:
            self.block2 = nn.Sequential(
                nn.Conv2d(width1, width2, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(width2),
                nn.ReLU(inplace=True),
                nn.Conv2d(width2, width2, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(width2),
                nn.ReLU(inplace=True),
            )

    def forward_features(self, x: torch.Tensor, compiled_tables=None):
        x_01 = torch.clamp(x * 0.5 + 0.5, 0.0, 1.0)
        if compiled_tables is None:
            M_all = self.qlut.get_observables()
            diff_tables = self.qlut.build_diff_tables(M_all)
        else:
            M_all, diff_tables = None, None
        q1 = self._extract_q_features(x_01, compiled_tables=compiled_tables, diff_tables=diff_tables)
        f_q = self.q_mixer(self.norm_q_in(q1))

        f_c = _stabilize_stem_variance(x, self.norm_c(self.c_stem(x)), self.training, tau=self.stabilize_tau)

        alpha = torch.sigmoid(self.gate(torch.cat([f_c, f_q], dim=1)))
        if (not self.training) and self.test_alpha_boost > 0.0:
            lap = _LAP_KERNEL.to(x_01.device, x_01.dtype)
            hp_score = F.conv2d(F.pad(x_01, (1, 1, 1, 1), mode="reflect"), lap).abs().mean(dim=(1, 2, 3), keepdim=True)
            excess = torch.clamp((hp_score - self.stabilize_tau) / self.stabilize_tau, min=0.0, max=4.0)
            if (excess > 0).any():
                alpha = torch.clamp(alpha + self.test_alpha_boost * excess, max=0.95)

        self._track(alpha)
        if self.training and self.drop_c_prob > 0.0:
            mask = torch.rand(x.shape[0], 1, 1, 1, device=x.device) < self.drop_c_prob
            if self.rand_cdrop_min < 0.5:
                rvals = torch.empty_like(mask, dtype=x.dtype).uniform_(self.rand_cdrop_min, 0.85)
                c_drop = torch.where(mask, rvals, torch.ones_like(mask, dtype=x.dtype))
            else:
                c_drop = torch.where(mask, torch.full_like(mask, 0.5, dtype=x.dtype), torch.ones_like(mask, dtype=x.dtype))
        elif (not self.training) and self.c_only_noise_clamp:
            # Gentle calibrated clamp floored at 0.50 (matching c_drop=0.50 seen during training!)
            lap = _LAP_KERNEL.to(x_01.device, x_01.dtype)
            hp_score = F.conv2d(F.pad(x_01, (1, 1, 1, 1), mode="reflect"), lap).abs().mean(dim=(1, 2, 3), keepdim=True)
            excess = torch.clamp((hp_score - self.stabilize_tau) / self.stabilize_tau, min=0.0, max=6.0)
            soft = excess * torch.clamp((excess - 0.15) / 0.50, min=0.0, max=1.0)
            c_drop = torch.clamp(1.0 / (1.0 + 0.15 * soft), min=0.50)
        else:
            c_drop = 1.0

        theta_g = (0.5 * math.pi) * alpha
        w_c = math.sqrt(2.0) * torch.cos(theta_g) * c_drop
        w_q = math.sqrt(2.0) * torch.sin(theta_g)
        f_fused = torch.cat([w_c * f_c, w_q * f_q], dim=1)

        if self.use_stage2_qlut:
            M0 = M_all[0] if M_all is not None else None
            q2 = self._extract_stage2_qlut(f_fused, compiled_tables=compiled_tables, M0=M0)
            feat = torch.cat([f_fused, q2], dim=1)
        else:
            feat = f_fused
        return feat, alpha


def run_candidate(ds: str, name: str, model_fn, lam_tv: float = 0.0, seed: int = 42):
    D = load_dataset_with_cache(ds, interp4d=True)
    ytr_np, yva_np, yte_np = D["ytr"], D["yva"], D["yte"]
    nc = int(len(np.unique(ytr_np)))
    epochs, bs, lr, wd = WIDE_HP[ds]
    _, sel, tau, _ = DATASETS[ds]

    torch.manual_seed(seed); np.random.seed(seed); torch.cuda.manual_seed_all(seed)
    model = model_fn(nc).to(DEV)
    cx = count_inference_macs(model)
    opt, sched = make_opt_and_sched(model, epochs, lr, wd)
    crit = nn.CrossEntropyLoss()
    counts = np.bincount(ytr_np, minlength=nc)
    logp = torch.log(torch.tensor(counts / counts.sum(), dtype=torch.float32, device=DEV) + 1e-8)

    g = torch.Generator(device=DEV); g.manual_seed(seed)
    aug = GPUMedicalAugment()
    xtr_u8, xva_u8, xte_u8 = D["g_tr"][0], D["g_va"][0], D["g_te"][0]
    ytr = torch.from_numpy(ytr_np).to(DEV)
    N = len(ytr_np)

    best, best_ep, best_state = -1.0, -1, None
    t0 = time.perf_counter()
    for ep in range(epochs):
        xn = (xtr_u8.float().unsqueeze(1) / 255.0 - 0.5) / 0.5
        outs = [aug(xn[i:i + 8192], g) for i in range(0, len(xn), 8192)]
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
            loss = crit(logits, ytr[idx])
            if lam_tv > 0.0:
                loss = loss + lam_tv * model.qlut.smoothness_loss()
            loss.backward()
            opt.step()
        sched.step()
        v = eval_split(model, xva_u8, yva_np, nc)[sel]
        if v > best:
            best, best_ep = v, ep
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}

    train_s = time.perf_counter() - t0
    model.load_state_dict(best_state)
    compiled_tbl = torch.from_numpy(model.qlut.compile_int8_shift_tables()).to(DEV)

    te_f = D["te"].astype(np.float32) / 255.0
    sp = {}
    for sg in SIGMAS:
        if sg == 0.0:
            mm = eval_split(model, xte_u8, yte_np, nc, compiled_tbl=compiled_tbl)
            sp["0.00"] = mm["acc"] * 100.0
        else:
            accs = []
            for dr in NOISE_DRAWS:
                damped_u8 = phase_damping_u8(speckle(te_f, sg, dr))
                mm = eval_split(model, torch.from_numpy(damped_u8).to(DEV), yte_np, nc, compiled_tbl=compiled_tbl)
                accs.append(mm["acc"] * 100.0)
            sp[f"{sg:.2f}"] = float(np.mean(accs))

    print(
        f"[{ds:<14} | {name:<22}] "
        f"0.00={sp['0.00']:5.2f} 0.05={sp['0.05']:5.2f} 0.10={sp['0.10']:5.2f} 0.15={sp['0.15']:5.2f} 0.20={sp['0.20']:5.2f} | "
        f"MACs={cx['macs']/1e6:.2f}M Params={cx['params']:,} ep*={best_ep} ({train_s:.1f}s)",
        flush=True,
    )
    return {"cx": cx, "sp": sp, "ep": best_ep, "train_s": train_s}


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    out = {}
    configs = [
        ("d1_sym_exact", lambda nc: GRQuanvV3Candidate(nc, spatial_mixer=False, test_alpha_boost=0.0, block2_std=False, rand_cdrop_min=0.5, drop_c_prob=0.20, c_only_noise_clamp=False), 0.0),
        ("d2_sym_dwmix_tv005", lambda nc: GRQuanvV3Candidate(nc, spatial_mixer=True, test_alpha_boost=0.0, block2_std=False, rand_cdrop_min=0.5, drop_c_prob=0.20, c_only_noise_clamp=False), 0.005),
        ("d3_floor05_dwmix_tv005", lambda nc: GRQuanvV3Candidate(nc, spatial_mixer=True, test_alpha_boost=0.0, block2_std=False, rand_cdrop_min=0.5, drop_c_prob=0.20, c_only_noise_clamp=True), 0.005),
    ]
    for ds in ["breastmnist", "pneumoniamnist"]:
        out[ds] = {}
        # Reference locked ctrl_v2_full seed=42
        D = load_dataset_with_cache(ds, interp4d=True)
        nc = 2
        mc = WideClassicalControl(num_classes=nc).to(DEV)
        mc.load_state_dict(torch.load(PROJECT_ROOT / "weights" / "pillar2_wide" / f"wide_v2_interp4d_{ds}_full_control_s42.pt", map_location=DEV, weights_only=True))
        mc.eval()
        te_f = D["te"].astype(np.float32) / 255.0
        sp_c = {}
        for sg in SIGMAS:
            if sg == 0.0:
                sp_c["0.00"] = eval_split(mc, D["g_te"][0], D["yte"], nc)["acc"] * 100.0
            else:
                sp_c[f"{sg:.2f}"] = float(np.mean([
                    eval_split(mc, torch.from_numpy(phase_damping_u8(speckle(te_f, sg, dr))).to(DEV), D["yte"], nc)["acc"] * 100.0
                    for dr in NOISE_DRAWS
                ]))
        print(f"[{ds:<14} | {'ctrl_v2_full':<22}] 0.00={sp_c['0.00']:5.2f} 0.05={sp_c['0.05']:5.2f} 0.10={sp_c['0.10']:5.2f} 0.15={sp_c['0.15']:5.2f} 0.20={sp_c['0.20']:5.2f}", flush=True)
        out[ds]["ctrl_v2_full"] = {"sp": sp_c}

        for cname, fn, tv in configs:
            out[ds][cname] = run_candidate(ds, cname, fn, lam_tv=tv, seed=42)

    p = PROJECT_ROOT / "reports" / "diagnose_v3_speckle_s42.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("[Saved]", p)


if __name__ == "__main__":
    main()
