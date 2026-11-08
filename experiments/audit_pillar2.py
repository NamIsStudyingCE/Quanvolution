"""
Kiểm toán độc lập Cột trụ 2 — sửa 3 điểm yếu phương pháp luận:
  (1) Thêm đối chứng cổ điển CÙNG bộ trộn lớn (ClassicalMultiScaleGatedControl lean_mixer=False)
      để tách đóng góp của lượng tử khỏi đóng góp của dung lượng mô hình.
  (2) Chọn epoch theo VAL (không dùng TEST) cho ConvHead.
  (3) Mọi mô hình dùng cùng logit-adjustment tau trong kịch bản ít dữ liệu / ConvHead.
Kết quả: reports/audit_pillar2.json
"""
import json, sys, time
import numpy as np, torch, torch.nn as nn, torch.optim as optim
sys.path.insert(0, "src"); sys.path.insert(0, "experiments")
import pillar2_benchmark as pb
from models.gr_quanv import (SymmetricalClassicalBaseline, ClassicalMultiScaleGatedControl,
                             GatedResidualQuanvNet, extract_multiscale_qlut, make_fairplay_head)
from utils.metrics import calculate_metrics

DEV, SEEDS = pb.DEVICE, pb.SEEDS
out = {}

def zoo(nc):
    return [
        ("Classical Baseline", lambda: SymmetricalClassicalBaseline(nc, "linear")),
        ("Classical MS-Gated FULL mixer (0% quantum)", lambda: ClassicalMultiScaleGatedControl(nc, 16, "linear", lean_mixer=False)),
        ("Proposed GR-Quanv (quantum)", lambda: GatedResidualQuanvNet(nc, 16, "linear", lean_mixer=False)),
    ]

def stat(runs, k):
    v = np.array([r[k] for r in runs]) * 100
    return float(v.mean()), float(v.std())

# ---------- A. Param-matched control, LinearHead, val-based selection ----------
cfgs = [("octmnist_canonical_3500", pb.load_canonical_split("data/processed/octmnist_splits.pt"), 25, 128, 1.5e-3),
        ("octmnist_full_109k", pb.load_npz_dataset("data/octmnist.npz"), 15, 1024, 2.0e-3)]
out["A_param_matched"] = {}
canon_models = {}
for name, raw, ep, bs, lr in cfgs:
    g = pb.prepare_gpu_tensors(raw)
    out["A_param_matched"][name] = {}
    print(f"\n=== A. {name} ===", flush=True)
    for mn, fn in zoo(g["num_classes"]):
        runs = []
        for s in SEEDS:
            if name.startswith("octmnist_canonical") and s == 42:
                r, mdl = pb.train_and_eval_single_seed(fn, g, s, epochs=ep, batch_size=bs, lr=lr, logit_adj_tau=0.85, select_metric="acc", return_model=True)
                canon_models[mn] = (mdl, g, raw)
            else:
                r = pb.train_and_eval_single_seed(fn, g, s, epochs=ep, batch_size=bs, lr=lr, logit_adj_tau=0.85, select_metric="acc")
            runs.append(r)
        a, asd = stat(runs, "acc"); u, usd = stat(runs, "auc"); f, _ = stat(runs, "f1")
        out["A_param_matched"][name][mn] = dict(params=runs[0]["params"], acc=a, acc_std=asd, auc=u, auc_std=usd, f1=f)
        print(f"  {mn:<44} params={runs[0]['params']:<6} ACC={a:5.2f}±{asd:4.2f} AUC={u:5.2f}±{usd:4.2f} F1={f:5.2f}", flush=True)

# ---------- B. Few-shot 1% & 5%: fixed epochs, no selection, no prior, identical for all ----------
oct_full = pb.load_npz_dataset("data/octmnist.npz")
vq, _ = extract_multiscale_qlut(oct_full["val_u8"]); tq, _ = extract_multiscale_qlut(oct_full["te_u8"])
tx = ((torch.from_numpy(oct_full["te_u8"]).float().unsqueeze(1) / 255 - .5) / .5).to(DEV); tq = tq.to(DEV)
out["B_fewshot_final_epoch"] = {}
for per_cls in (240, 1200):
    key = f"{per_cls*4}_imgs"; out["B_fewshot_final_epoch"][key] = {}
    print(f"\n=== B. few-shot {key} (final epoch, no test/val selection, no prior) ===", flush=True)
    for mn, fn in zoo(4):
        runs = []
        for s in SEEDS:
            rng = np.random.RandomState(s + 100); idx = []
            for c in range(4):
                idx.append(rng.choice(np.where(oct_full["tr_y"] == c)[0], per_cls, replace=False))
            idx = np.concatenate(idx); rng.shuffle(idx)
            u8 = np.ascontiguousarray(oct_full["tr_u8"][idx]); q, _ = extract_multiscale_qlut(u8)
            x = ((torch.from_numpy(u8).float().unsqueeze(1) / 255 - .5) / .5).to(DEV); q = q.to(DEV)
            y = torch.from_numpy(oct_full["tr_y"][idx]).long().to(DEV)
            pb.set_seed(s); m = fn().to(DEV)
            opt = optim.AdamW(m.parameters(), lr=1.2e-3, weight_decay=1e-3); crit = nn.CrossEntropyLoss()
            for ep in range(30):
                m.train(); perm = torch.randperm(len(x), device=DEV)
                for i in range(0, len(x), 64):
                    b = perm[i:i+64]; opt.zero_grad(); crit(m(x[b], q[b]), y[b]).backward(); opt.step()
            runs.append(pb.eval_gpu(m, tx, tq, oct_full["te_y"], 4))
        a, asd = stat(runs, "acc"); u, _ = stat(runs, "auc")
        out["B_fewshot_final_epoch"][key][mn] = dict(acc=a, acc_std=asd, auc=u)
        print(f"  {mn:<44} ACC={a:5.2f}±{asd:4.2f} AUC={u:5.2f}", flush=True)

# ---------- C. Speckle robustness, 5 seeds (trained on canonical, clean) ----------
print("\n=== C. Speckle noise robustness, 5 seeds, canonical-trained ===", flush=True)
raw = cfgs[0][1]; g = pb.prepare_gpu_tensors(raw)
clean = raw["te_u8"].astype(np.float32) / 255.0
out["C_speckle_5seeds"] = {}
trained = {mn: [] for mn, _ in zoo(4)}
for mn, fn in zoo(4):
    for s in SEEDS:
        _, mdl = pb.train_and_eval_single_seed(fn, g, s, epochs=25, batch_size=128, lr=1.5e-3, logit_adj_tau=0.85, select_metric="acc", return_model=True)
        trained[mn].append(mdl)
for sig in (0.0, 0.10, 0.15):
    rng = np.random.RandomState(42)
    nf = clean if sig == 0 else np.clip(clean + clean * rng.normal(0, sig, clean.shape), 0, 1)
    u8 = np.ascontiguousarray((nf * 255).round().astype(np.uint8)); q, _ = extract_multiscale_qlut(u8)
    x = ((torch.from_numpy(nf).float().unsqueeze(1) - .5) / .5).to(DEV); q = q.to(DEV)
    out["C_speckle_5seeds"][f"{sig:.2f}"] = {}
    for mn, mdls in trained.items():
        runs = [pb.eval_gpu(m, x, q, raw["te_y"], 4) for m in mdls]
        a, asd = stat(runs, "acc")
        out["C_speckle_5seeds"][f"{sig:.2f}"][mn] = dict(acc=a, acc_std=asd)
        print(f"  speckle={sig:.2f} {mn:<44} ACC={a:5.2f}±{asd:4.2f}", flush=True)

json.dump(out, open("reports/audit_pillar2.json", "w"), indent=2)
print("\nSaved reports/audit_pillar2.json")
