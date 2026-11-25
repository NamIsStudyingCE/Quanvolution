"""Step 1: Train & Evaluate Missing Classical Baselines under Exact Wide-Fusion v2 Protocol.

Implements the missing baselines requested by the thesis advisor:
  1. ctrl_tanh      : WideClassicalControl with torch.tanh() bounding on the 16-ch multi-scale conv output
                      (isolates whether robustness stems from [-1, 1] saturation vs quantum entanglement).
  2. ctrl_hardclip  : WideClassicalControl with torch.clamp(..., -1, 1) hard clipping on the 16-ch branch.
  3. ctrl_noise_aug : WideClassicalControl trained with multiplicative speckle noise augmentation
                      (p=0.5 per image, sigma ~ Uniform(0, 0.20)) to match realistic deployment practice.

Strict guardrails:
- Does NOT modify weights/pillar2_wide/, weights/ablation_clut_models/, or existing ablation_clut JSONs.
- Saves checkpoints ONLY to weights/ablation_clut_extra/.
- Saves train + clean/speckle evaluation results ONLY to results/ablation_clut/extra_baselines.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_pillar2_expansion import NOISE_DRAWS, SEEDS, SIGMAS, speckle
from experiments.run_pillar2_wide import (
    DATASETS,
    DEFAULT_DROP_C_PROB,
    DEV,
    GPUMedicalAugment,
    WIDE_HP,
    load,
    make_scheduler,
)

SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.gr_quanv import phase_damping_u8  # noqa: E402
from models.gr_quanv_wide import (  # noqa: E402
    WideClassicalControl,
    _apply_branch_droppath,
    _stabilize_stem_variance,
)
from utils.metrics import calculate_metrics  # noqa: E402

CKPT_DIR = PROJECT_ROOT / "weights" / "ablation_clut_extra"
OUT_JSON = PROJECT_ROOT / "results" / "ablation_clut" / "extra_baselines.json"
LOCKED_WIDE_DIR = PROJECT_ROOT / "weights" / "pillar2_wide"

EXPECTED_CTRL_NC2_PARAMS = 148406
VARIANTS = ["ctrl_tanh", "ctrl_hardclip", "ctrl_noise_aug"]
STEM_CONDS = ["raw_stem", "stabilized_stem"]


class WideClassicalControlBounded(WideClassicalControl):
    """Exact WideClassicalControl (148,406 params for nc=2) with bounded [-1, 1] activation
    applied strictly to the 16-ch multi-scale conv output `f_raw` (1-to-1 counterpart to Q-LUT)."""

    def __init__(
        self,
        num_classes: int = 2,
        drop_c_prob: float = DEFAULT_DROP_C_PROB,
        bound_mode: str = "tanh",
    ):
        super().__init__(num_classes=num_classes, drop_c_prob=drop_c_prob)
        if bound_mode not in ("tanh", "hardclip", "none"):
            raise ValueError(f"Unsupported bound_mode={bound_mode}")
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


def build_extra_model(variant: str, nc: int) -> nn.Module:
    if variant == "ctrl_tanh":
        return WideClassicalControlBounded(nc, drop_c_prob=DEFAULT_DROP_C_PROB, bound_mode="tanh")
    if variant == "ctrl_hardclip":
        return WideClassicalControlBounded(nc, drop_c_prob=DEFAULT_DROP_C_PROB, bound_mode="hardclip")
    if variant == "ctrl_noise_aug":
        return WideClassicalControl(nc, drop_c_prob=DEFAULT_DROP_C_PROB)
    raise ValueError(f"Unknown variant: {variant}")


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def _u8_to_x_tensor(imgs_u8: np.ndarray) -> torch.Tensor:
    xu8 = torch.from_numpy(np.ascontiguousarray(imgs_u8, dtype=np.uint8)).to(DEV)
    return (xu8.float().unsqueeze(1) / 255.0 - 0.5) / 0.5


def _augment_epoch_u8(
    xu8_tr: torch.Tensor,
    aug: GPUMedicalAugment,
    gen: torch.Generator,
    apply_speckle_aug: bool = False,
    speckle_prob: float = 0.5,
    speckle_sigma_max: float = 0.20,
    aug_chunk: int = 8192,
) -> torch.Tensor:
    """Applies identical GPUMedicalAugment as Wide-Fusion v2, and optionally multiplicative speckle noise."""
    outs = []
    xn = (xu8_tr.float().unsqueeze(1) / 255.0 - 0.5) / 0.5
    for i in range(0, len(xn), aug_chunk):
        outs.append(aug(xn[i : i + aug_chunk], gen))
    xe_u8 = np.ascontiguousarray(np.concatenate(outs, axis=0), dtype=np.uint8)
    xe_gpu = torch.from_numpy(xe_u8).to(DEV)

    if apply_speckle_aug:
        # Multiplicative speckle noise: clip(I + I * N(0, sigma_i), 0, 1) with p=0.5, sigma_i ~ U(0, 0.20)
        xf = xe_gpu.float() / 255.0
        n = xf.shape[0]
        apply_mask = (torch.rand((n, 1, 1), device=DEV, generator=gen) < speckle_prob).float()
        sigmas = torch.rand((n, 1, 1), device=DEV, generator=gen) * speckle_sigma_max
        noise = torch.randn(xf.shape, device=DEV, generator=gen) * (sigmas * apply_mask)
        xf_noisy = torch.clamp(xf + xf * noise, 0.0, 1.0)
        xe_gpu = torch.round(xf_noisy * 255.0).to(torch.uint8)

    return xe_gpu


@torch.no_grad()
def evaluate_tensor(
    model: nn.Module,
    x_tensor: torch.Tensor,
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
        logits = model(xb, None)
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


def train_single_run(
    variant: str,
    ds: str,
    seed: int,
    data_dict: dict,
    xtr_gpu: torch.Tensor,
    xva_norm: torch.Tensor,
    xte_norm: torch.Tensor,
    ckpt_path: Path,
) -> dict:
    assert str(ckpt_path.resolve().parent) == str(CKPT_DIR.resolve())
    assert str(LOCKED_WIDE_DIR.resolve()) not in str(ckpt_path.resolve())

    epochs, bs, lr, wd = WIDE_HP[ds]
    _, sel, tau, _ = DATASETS[ds]

    ytr_np = data_dict["ytr"].astype(np.int64)
    yva_np = data_dict["yva"].astype(np.int64)
    yte_np = data_dict["yte"].astype(np.int64)
    nc = int(len(np.unique(ytr_np)))

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    model = build_extra_model(variant, nc).to(DEV)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert total_params == trainable_params
    if nc == 2:
        assert total_params == EXPECTED_CTRL_NC2_PARAMS, (
            f"Expected {EXPECTED_CTRL_NC2_PARAMS} params for nc=2, got {total_params}"
        )

    ysub = torch.from_numpy(ytr_np).to(DEV)
    counts = np.bincount(ytr_np, minlength=nc)
    logp = torch.log(torch.tensor(counts / counts.sum(), dtype=torch.float32, device=DEV) + 1e-8)

    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = make_scheduler(opt, epochs)
    crit = nn.CrossEntropyLoss()
    g = torch.Generator(device=DEV)
    g.manual_seed(seed)
    aug = GPUMedicalAugment()

    use_speckle_aug = variant == "ctrl_noise_aug"
    n_train = len(ytr_np)
    best_val, best_ep, best_state = -1.0, -1, None
    t0 = time.perf_counter()

    for ep in range(epochs):
        xe_u8_gpu = _augment_epoch_u8(
            xtr_gpu,
            aug,
            g,
            apply_speckle_aug=use_speckle_aug,
            speckle_prob=0.5,
            speckle_sigma_max=0.20,
        )
        model.train()
        perm = torch.randperm(n_train, device=DEV, generator=g)
        for i in range(0, n_train, bs):
            idx = perm[i : i + bs]
            if len(idx) < 2:
                continue
            x = (xe_u8_gpu[idx].float().unsqueeze(1) / 255.0 - 0.5) / 0.5
            opt.zero_grad()
            logits = model(x, None) + tau * logp
            crit(logits, ysub[idx]).backward()
            opt.step()
        sched.step()

        # Select strictly on clean validation split (same criterion as all locked models)
        val_metrics = evaluate_tensor(model, xva_norm, yva_np, nc)
        val_score = val_metrics[sel] / 100.0
        if val_score > best_val:
            best_val, best_ep = val_score, ep
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}

    model.load_state_dict(best_state)
    clean_metrics = evaluate_tensor(model, xte_norm, yte_np, nc)
    elapsed = time.perf_counter() - t0

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: t.cpu() for k, t in best_state.items()}, ckpt_path)

    return {
        "variant": variant,
        "ds": ds,
        "seed": seed,
        "nc": nc,
        "params": total_params,
        "sel_metric": sel,
        "best_val": float(best_val),
        "best_epoch": int(best_ep),
        "train_sec": float(elapsed),
        "clean_test": clean_metrics,
        "ckpt_path": str(ckpt_path.relative_to(PROJECT_ROOT)),
    }


def evaluate_variants_on_dataset(
    ds: str,
    variants: List[str],
    seeds: List[int],
    sigmas: List[float],
    stem_conds: List[str],
) -> dict:
    data_dict = load(DATASETS[ds][0])
    te_u8 = data_dict["te"]
    te_f = te_u8.astype(np.float32) / 255.0
    y_test = data_dict["yte"].astype(np.int64)
    nc = int(len(np.unique(y_test)))

    models: Dict[str, Dict[int, nn.Module]] = {v: {} for v in variants}
    for v in variants:
        for s in seeds:
            cp = CKPT_DIR / f"extra_{v}_{ds}_full_s{s}.pt"
            if not cp.exists():
                raise FileNotFoundError(f"Missing checkpoint: {cp}")
            m = build_extra_model(v, nc).to(DEV)
            m.load_state_dict(torch.load(cp, map_location=DEV, weights_only=True))
            m.eval()
            models[v][s] = m

    ds_res: dict = {stem: {} for stem in stem_conds}

    for stem in stem_conds:
        tau_val = 999.0 if stem == "raw_stem" else 0.024
        use_damp = stem == "stabilized_stem"
        print(f"\n=== EVAL {ds.upper()} | stem={stem} (phase_damping={use_damp}, stabilize_tau={tau_val}) ===")

        for sg in sigmas:
            sg_str = f"{sg:.2f}"
            draws = NOISE_DRAWS if sg > 0 else [0]
            draw_buf: Dict[str, Dict[int, List[Dict[str, float]]]] = {
                v: {s: [] for s in seeds} for v in variants
            }

            for dr in draws:
                noisy_u8 = te_u8 if sg == 0.0 else speckle(te_f, sg, dr)
                input_u8 = phase_damping_u8(noisy_u8, beta=18.0, tau=0.024) if use_damp else noisy_u8
                x_t = _u8_to_x_tensor(input_u8)

                for v in variants:
                    for s in seeds:
                        models[v][s].stabilize_tau = tau_val
                        draw_buf[v][s].append(evaluate_tensor(models[v][s], x_t, y_test, nc))

            ds_res[stem][sg_str] = {}
            print(f"  -- sigma={sg_str} --")
            for v in variants:
                per_seed: Dict[str, Dict[str, float]] = {}
                for s in seeds:
                    items = draw_buf[v][s]
                    per_seed[str(s)] = {
                        metric: float(np.mean([it[metric] for it in items]))
                        for metric in ["acc", "macro_f1", "f1", "auc", "alpha_mean"]
                    }
                agg: Dict[str, object] = {}
                for metric in ["acc", "macro_f1", "f1", "auc", "alpha_mean"]:
                    vals = [per_seed[str(s)][metric] for s in seeds]
                    agg[f"{metric}_mean"] = float(np.mean(vals))
                    agg[f"{metric}_std"] = float(np.std(vals))
                agg["per_seed"] = per_seed
                ds_res[stem][sg_str][v] = agg
                print(
                    f"    {v:<16} | ACC={agg['acc_mean']:5.2f}±{agg['acc_std']:4.2f}% | "
                    f"mF1={agg['macro_f1_mean']:5.2f}±{agg['macro_f1_std']:4.2f}% | "
                    f"AUC={agg['auc_mean']:5.2f}%"
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

    parser = argparse.ArgumentParser(description="Train & evaluate missing classical baselines.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["pneumoniamnist", "breastmnist"],
        choices=["pneumoniamnist", "breastmnist"],
    )
    parser.add_argument("--variants", nargs="+", default=VARIANTS, choices=VARIANTS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--resume", action="store_true", help="Skip already-trained checkpoints.")
    args = parser.parse_args()

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    if OUT_JSON.exists() and args.resume:
        with open(OUT_JSON, "r", encoding="utf-8") as f:
            payload = json.load(f)
    else:
        payload = {
            "metadata": {
                "seeds": args.seeds,
                "sigmas": [float(s) for s in SIGMAS],
                "noise_draws": NOISE_DRAWS,
                "stems": STEM_CONDS,
                "variants": args.variants,
                "speckle_train_aug": {"prob": 0.5, "sigma_range": [0.0, 0.20]},
                "device": str(DEV),
            },
            "train_runs": {},
            "datasets": {},
        }

    t_all = time.perf_counter()
    print("=" * 88)
    print("  STEP 1: MISSING CLASSICAL BASELINES (ctrl_tanh, ctrl_hardclip, ctrl_noise_aug)")
    print(f"  Device: {DEV} | Datasets: {args.datasets} | Variants: {args.variants} | Seeds: {args.seeds}")
    print("=" * 88)

    for ds in args.datasets:
        data_dict = load(DATASETS[ds][0])
        xtr_gpu = torch.from_numpy(np.ascontiguousarray(data_dict["tr"], dtype=np.uint8)).to(DEV)
        xva_norm = _u8_to_x_tensor(data_dict["va"])
        xte_norm = _u8_to_x_tensor(data_dict["te"])

        for variant in args.variants:
            for seed in args.seeds:
                key = f"{ds}|{variant}|s{seed}"
                ckpt_path = CKPT_DIR / f"extra_{variant}_{ds}_full_s{seed}.pt"
                if args.resume and key in payload["train_runs"] and ckpt_path.exists():
                    print(f"  [Skip Train] {key}")
                    continue

                res = train_single_run(
                    variant=variant,
                    ds=ds,
                    seed=seed,
                    data_dict=data_dict,
                    xtr_gpu=xtr_gpu,
                    xva_norm=xva_norm,
                    xte_norm=xte_norm,
                    ckpt_path=ckpt_path,
                )
                payload["train_runs"][key] = res
                _atomic_write_json(OUT_JSON, payload)

                ct = res["clean_test"]
                print(
                    f"  [Train] {key:<34} | ep={res['best_epoch']:2d} | val={res['best_val']*100:5.2f}% | "
                    f"test ACC={ct['acc']:5.2f}% AUC={ct['auc']:5.2f}% mF1={ct['macro_f1']:5.2f}% | "
                    f"{res['train_sec']:4.1f}s",
                    flush=True,
                )

    for ds in args.datasets:
        payload["datasets"][ds] = evaluate_variants_on_dataset(
            ds=ds,
            variants=args.variants,
            seeds=args.seeds,
            sigmas=SIGMAS,
            stem_conds=STEM_CONDS,
        )
        _atomic_write_json(OUT_JSON, payload)

    payload["metadata"]["elapsed_sec"] = float(time.perf_counter() - t_all)
    _atomic_write_json(OUT_JSON, payload)
    print(f"\n[Saved] {OUT_JSON} (total elapsed={payload['metadata']['elapsed_sec']:.1f}s)")


if __name__ == "__main__":
    main()
