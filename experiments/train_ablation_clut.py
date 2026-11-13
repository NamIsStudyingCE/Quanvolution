"""Step 2 of C-LUT vs Q-LUT Isolated Ablation:
Identical Wide-Fusion v2 training protocol (5 seeds) for Classical-LUT variants.

Mandatory guardrails:
- Reads C-LUT binaries ONLY from weights/ablation_clut/.
- Saves checkpoints ONLY to weights/ablation_clut_models/.
- Saves training telemetry ONLY to results/ablation_clut/train_results.json.
- Never mutates weights/pillar2_wide/, Q-LUT binaries, or locked benchmark outputs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import experiments.* before appending src/ so ./experiments is not shadowed by src/experiments.
from experiments.run_pillar2_wide import (
    DATASETS,
    DEFAULT_DROP_C_PROB,
    DEV,
    GPUMedicalAugment,
    WIDE_HP,
    deq_wide,
    evaluate_wide,
    load,
    make_scheduler,
)

SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.gr_quanv_wide import build_model  # noqa: E402
from scripts.generate_clut import (  # noqa: E402
    extract_multiscale_from_lut_tensor,
    load_lut_table,
)

SEEDS: List[int] = [0, 42, 100, 2023, 777]
EXPECTED_NC2_PARAMS = 148306

LUT_DIR = PROJECT_ROOT / "weights" / "ablation_clut"
CKPT_DIR = PROJECT_ROOT / "weights" / "ablation_clut_models"
OUT_DIR = PROJECT_ROOT / "results" / "ablation_clut"
LOCKED_WIDE_DIR = PROJECT_ROOT / "weights" / "pillar2_wide"


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def _jsonable(v):
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, list):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {k: _jsonable(val) for k, val in v.items()}
    return v


def _extract_to_gpu(
    imgs_u8: np.ndarray,
    table_flat: torch.Tensor,
    chunk: int = 2048,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xu8_gpu = torch.from_numpy(np.ascontiguousarray(imgs_u8, dtype=np.uint8)).to(DEV)
    q_gpu = extract_multiscale_from_lut_tensor(
        imgs_u8,
        table_flat,
        mode="interp4d",
        phase_damping=False,
        chunk=chunk,
    )
    return xu8_gpu, q_gpu


def _augment_epoch_clut(
    xu8_tr: torch.Tensor,
    aug: GPUMedicalAugment,
    gen: torch.Generator,
    table_flat: torch.Tensor,
    aug_chunk: int = 8192,
) -> Tuple[torch.Tensor, torch.Tensor]:
    outs = []
    xn = (xu8_tr.float().unsqueeze(1) / 255.0 - 0.5) / 0.5
    for i in range(0, len(xn), aug_chunk):
        outs.append(aug(xn[i : i + aug_chunk], gen))
    xe_u8 = np.ascontiguousarray(np.concatenate(outs, axis=0), dtype=np.uint8)
    return _extract_to_gpu(xe_u8, table_flat)


@torch.no_grad()
def _eval_with_macro_f1(
    model: nn.Module,
    xu8: torch.Tensor,
    qbuf: torch.Tensor,
    y_np: np.ndarray,
    nc: int,
    bs: int = 2048,
) -> Dict[str, float]:
    m = evaluate_wide(model, xu8, qbuf, y_np, nc, bs=bs)
    model.eval()
    preds = []
    for i in range(0, len(xu8), bs):
        x, q = deq_wide(xu8[i : i + bs], qbuf[i : i + bs])
        preds.append(model(x, q).argmax(dim=1).cpu().numpy())
    y_pred = np.concatenate(preds, axis=0)
    m["macro_f1"] = float(f1_score(y_np, y_pred, average="macro"))
    return {k: float(v) for k, v in m.items()}


def train_single_clut_run(
    variant: str,
    ds: str,
    seed: int,
    data_dict: dict,
    xtr_gpu: torch.Tensor,
    xva_gpu: torch.Tensor,
    qva_gpu: torch.Tensor,
    xte_gpu: torch.Tensor,
    qte_gpu: torch.Tensor,
    table_flat: torch.Tensor,
    ckpt_path: Path,
) -> dict:
    """Trains one WideGRQuanvNet run with a fixed C-LUT under identical Wide-Fusion v2 parity."""
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

    model = build_model("grquanv", nc, drop_c_prob=DEFAULT_DROP_C_PROB).to(DEV)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert total_params == trainable_params
    if nc == 2:
        assert total_params == EXPECTED_NC2_PARAMS, (
            f"Expected {EXPECTED_NC2_PARAMS} params for nc=2, got {total_params}"
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

    N = len(ytr_np)
    best_val, best_ep, best_state = -1.0, -1, None
    t0 = time.perf_counter()

    for ep in range(epochs):
        xe_gpu, qe_gpu = _augment_epoch_clut(xtr_gpu, aug, g, table_flat)
        model.train()
        perm = torch.randperm(N, device=DEV, generator=g)
        for i in range(0, N, bs):
            idx = perm[i : i + bs]
            if len(idx) < 2:
                continue
            x, q = deq_wide(xe_gpu[idx], qe_gpu[idx])
            opt.zero_grad()
            logits = model(x, q) + tau * logp
            crit(logits, ysub[idx]).backward()
            opt.step()
        sched.step()
        val_score = evaluate_wide(model, xva_gpu, qva_gpu, yva_np, nc)[sel]
        if val_score > best_val:
            best_val, best_ep = val_score, ep
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}

    model.load_state_dict(best_state)
    clean_metrics = _eval_with_macro_f1(model, xte_gpu, qte_gpu, yte_np, nc)
    elapsed = time.perf_counter() - t0

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: t.cpu() for k, t in best_state.items()}, ckpt_path)

    gate_info = _jsonable(model.get_gate_summary()) if hasattr(model, "get_gate_summary") else None
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
        "gate": gate_info,
        "ckpt_path": str(ckpt_path.relative_to(PROJECT_ROOT)),
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Train WideGRQuanvNet on Classical-LUT variants (5 seeds).")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["pneumoniamnist"],
        choices=["pneumoniamnist", "breastmnist", "octmnist"],
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["clut_septrig", "clut_randmlp"],
        choices=["clut_septrig", "clut_randmlp"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--resume", action="store_true", help="Skip runs already saved on disk.")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = OUT_DIR / "train_results.json"

    if out_json.exists():
        with open(out_json, "r", encoding="utf-8") as f:
            payload = json.load(f)
    else:
        payload = {"runs": {}}

    print("=" * 84)
    print("  STEP 2: TRAINING CLASSICAL-LUT (C-LUT) BASELINES UNDER IDENTICAL WIDE-FUSION V2")
    print(f"  Device: {DEV} | Datasets: {args.datasets} | Variants: {args.variants} | Seeds: {args.seeds}")
    print("=" * 84)

    for ds in args.datasets:
        data_dict = load(DATASETS[ds][0])
        xtr_gpu = torch.from_numpy(np.ascontiguousarray(data_dict["tr"], dtype=np.uint8)).to(DEV)
        xva_gpu = torch.from_numpy(np.ascontiguousarray(data_dict["va"], dtype=np.uint8)).to(DEV)
        xte_gpu = torch.from_numpy(np.ascontiguousarray(data_dict["te"], dtype=np.uint8)).to(DEV)

        for variant in args.variants:
            lut_bin = LUT_DIR / f"{variant}_4bit_fp32_grid.bin"
            table_flat = load_lut_table(lut_bin, mode="interp4d", device=DEV)
            _, qva_gpu = _extract_to_gpu(data_dict["va"], table_flat)
            _, qte_gpu = _extract_to_gpu(data_dict["te"], table_flat)

            for seed in args.seeds:
                key = f"{ds}|{variant}|s{seed}"
                ckpt_path = CKPT_DIR / f"ablation_{variant}_{ds}_full_s{seed}.pt"
                if args.resume and key in payload["runs"] and ckpt_path.exists():
                    print(f"  [Skip] {key} already completed.")
                    continue

                res = train_single_clut_run(
                    variant=variant,
                    ds=ds,
                    seed=seed,
                    data_dict=data_dict,
                    xtr_gpu=xtr_gpu,
                    xva_gpu=xva_gpu,
                    qva_gpu=qva_gpu,
                    xte_gpu=xte_gpu,
                    qte_gpu=qte_gpu,
                    table_flat=table_flat,
                    ckpt_path=ckpt_path,
                )
                payload["runs"][key] = res
                _atomic_write_json(out_json, payload)

                ct = res["clean_test"]
                alpha_str = f"{res['gate']['overall_alpha']:.3f}" if res.get("gate") else "n/a"
                print(
                    f"  {key:<32} | ep={res['best_epoch']:2d} | val={res['best_val']*100:5.2f}% | "
                    f"test ACC={ct['acc']*100:5.2f}% AUC={ct['auc']*100:5.2f}% "
                    f"mF1={ct['macro_f1']*100:5.2f}% | alpha={alpha_str} | {res['train_sec']:5.1f}s"
                )

    print(f"\n[Saved] {out_json}")


if __name__ == "__main__":
    main()
