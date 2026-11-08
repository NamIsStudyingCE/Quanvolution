"""
BƯỚC 3 & 4: Huấn luyện + đánh giá đa tập dữ liệu (5 seeds) và stress-test nhiễu đốm.
- Giai đoạn linear  : head BN->ReLU->Linear (không augment)         -> so baseline đa tập.
- Giai đoạn resblock: ResidualQuanvHead + Medical Augmentation        -> thử phá trần accuracy.
- Mọi mô hình trong cùng giai đoạn dùng CÙNG head, CÙNG augment, CÙNG siêu tham số, CÙNG seed.
- Chọn epoch CHỈ theo val; test được đánh giá đúng 1 lần ở epoch tốt nhất theo val.
- Có thể tiếp tục khi bị ngắt: mỗi (tập, head, mô hình, seed) là một khóa riêng trong JSON, đã có thì bỏ qua.
Dùng:  python experiments/run_pillar2_expansion.py --stage all
"""
import argparse, json, os, sys, time
import numpy as np, torch, torch.nn as nn, torch.optim as optim
from scipy import stats

sys.path.insert(0, "src"); sys.path.insert(0, "experiments")
import pillar2_state as st
from models.gr_quanv import extract_multiscale_qlut
from models.gr_quanv_resblock import build_model, MODEL_KINDS, GPUMedicalAugment
from utils.metrics import calculate_metrics

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = [0, 42, 100, 2023, 777]
SIGMAS = [0.0, 0.05, 0.10, 0.15, 0.20]
NOISE_DRAWS = [1, 2, 3]

# name: (npz, sel_metric, tau, {head: (epochs, bs, lr, wd)})
DATASETS = {
    "breastmnist": ("data/breastmnist.npz", "auc", 0.35,
                    {"linear": (40, 32, 1.0e-3, 1e-4), "resblock": (60, 32, 1.0e-3, 5e-4)}),
    "pneumoniamnist": ("data/pneumoniamnist.npz", "auc", 0.50,
                       {"linear": (30, 128, 1.2e-3, 1e-4), "resblock": (40, 128, 1.5e-3, 5e-4)}),
    "octmnist": ("data/octmnist.npz", "acc", 0.85,
                 {"linear": (15, 1024, 2.0e-3, 1e-4), "resblock": (14, 256, 2.0e-3, 5e-4)}),
}


def load(npz):
    d = np.load(npz)
    g = lambda k: np.ascontiguousarray(d[k], dtype=np.uint8)
    y = lambda k: d[k].flatten().astype(np.int64)
    return dict(tr=g("train_images"), ytr=y("train_labels"), va=g("val_images"), yva=y("val_labels"),
                te=g("test_images"), yte=y("test_labels"))


def to_gpu(u8):
    """uint8 (N,28,28) -> (x_u8 GPU, q_int8 GPU) bằng Rust Q-LUT."""
    q, _ = extract_multiscale_qlut(np.ascontiguousarray(u8), include_axial=True)
    q8 = torch.round(q * 127.0).to(torch.int8)
    return torch.from_numpy(np.ascontiguousarray(u8)).to(DEV), q8.to(DEV)


def deq(xu8, q8):
    x = (xu8.float().unsqueeze(1) / 255.0 - 0.5) / 0.5
    return x, q8.float() / 127.0


@torch.no_grad()
def evaluate(model, xu8, q8, y_np, nc, bs=2048):
    model.eval(); probs = []
    for i in range(0, len(xu8), bs):
        x, q = deq(xu8[i:i + bs], q8[i:i + bs])
        probs.append(torch.softmax(model(x, q), 1).cpu().numpy())
    p = np.concatenate(probs)
    return calculate_metrics(y_np, p, p.argmax(1), nc)


def augment_epoch(xu8_tr, aug, gen, chunk=8192):
    outs = []
    xn = (xu8_tr.float().unsqueeze(1) / 255.0 - 0.5) / 0.5
    for i in range(0, len(xn), chunk):
        outs.append(aug(xn[i:i + chunk], gen))
    return to_gpu(np.concatenate(outs))


def train_run(ds, head, kind, seed, ckpt_path):
    npz, sel, tau, hp = DATASETS[ds]
    epochs, bs, lr, wd = hp[head]
    D = DATA[ds]; nc = int(len(np.unique(D["ytr"])))
    torch.manual_seed(seed); np.random.seed(seed); torch.cuda.manual_seed_all(seed)
    model = build_model(kind, nc, head).to(DEV)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ytr = torch.from_numpy(D["ytr"]).to(DEV)
    logp = torch.log(torch.tensor(np.bincount(D["ytr"], minlength=nc) / len(D["ytr"]), dtype=torch.float32, device=DEV) + 1e-8)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()
    g = torch.Generator(device=DEV); g.manual_seed(seed)
    use_aug = head == "resblock"; aug = GPUMedicalAugment()
    xtr, qtr = D["g_tr"]
    best, best_state, best_ep = -1.0, None, -1
    t0 = time.perf_counter()
    for ep in range(epochs):
        if use_aug:
            xe, qe = augment_epoch(D["g_tr"][0], aug, g)
        else:
            xe, qe = xtr, qtr
        model.train(); N = len(ytr); perm = torch.randperm(N, device=DEV, generator=g)
        for i in range(0, N, bs):
            idx = perm[i:i + bs]
            if len(idx) < 2:
                continue
            x, q = deq(xe[idx], qe[idx])
            opt.zero_grad()
            logits = model(x, q) + tau * logp
            crit(logits, ytr[idx]).backward(); opt.step()
        sched.step()
        v = evaluate(model, *D["g_va"], D["yva"], nc)[sel]
        if v > best:
            best, best_ep = v, ep
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
    model.load_state_dict(best_state)
    m = evaluate(model, *D["g_te"], D["yte"], nc)  # test: đúng 1 lần
    m.update(params=params, best_val=float(best), best_epoch=best_ep, train_sec=time.perf_counter() - t0)
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save({k: t.cpu() for k, t in best_state.items()}, ckpt_path)
    return m


def agg(rs, k):
    v = np.array([r[k] for r in rs]) * 100
    return float(v.mean()), float(v.std())


def ckpt_file(a, ds, head, kind, seed):
    return os.path.join(a.ckpt_dir, f"{ds}_{head}_{kind}_s{seed}.pt")


def stage_train(a, R):
    for ds in a.datasets:
        for head in a.heads:
            for name, kind in MODEL_KINDS:
                for seed in a.seeds:
                    key = f"{ds}|{head}|{kind}|{seed}"
                    if key in R["runs"]:
                        continue
                    r = train_run(ds, head, kind, seed, ckpt_file(a, ds, head, kind, seed))
                    R["runs"][key] = {k: float(v) for k, v in r.items()}
                    st.atomic_dump_json(R, a.results)
                    print(f"[{ds}|{head}|{name}|s{seed}] ACC={r['acc']*100:.2f} AUC={r['auc']*100:.2f} F1={r['f1']*100:.2f} "
                          f"params={int(r['params'])} ep*={int(r['best_epoch'])} {r['train_sec']:.0f}s", flush=True)
            print(f"== {ds}/{head} summary ==", flush=True)
            for name, kind in MODEL_KINDS:
                rs = [R["runs"][f"{ds}|{head}|{kind}|{s}"] for s in a.seeds if f"{ds}|{head}|{kind}|{s}" in R["runs"]]
                if rs:
                    (am, asd), (um, usd), (fm, _) = agg(rs, "acc"), agg(rs, "auc"), agg(rs, "f1")
                    print(f"   {name:<32} ACC={am:5.2f}±{asd:4.2f} AUC={um:5.2f}±{usd:4.2f} F1={fm:5.2f}", flush=True)
    full = all(f"{ds}|{h}|{k}|{s}" in R["runs"] for ds in DATASETS for h in ("linear", "resblock")
               for _, k in MODEL_KINDS for s in SEEDS)
    if full:
        st.mark("step3_multidataset_training", True, results=a.results)


def speckle(te_f, sigma, draw):
    if sigma == 0.0:
        return np.ascontiguousarray((te_f * 255).round().astype(np.uint8))
    rng = np.random.RandomState(1000 + draw)
    nf = np.clip(te_f + te_f * rng.normal(0, sigma, te_f.shape), 0, 1)
    return np.ascontiguousarray((nf * 255).round().astype(np.uint8))


def stage_speckle(a, R):
    R.setdefault("speckle", {})
    for ds in a.datasets:
        D = DATA[ds]; nc = int(len(np.unique(D["ytr"]))); te_f = D["te"].astype(np.float32) / 255.0
        for head in a.heads:
            for sg in SIGMAS:
                for name, kind in MODEL_KINDS:
                    key = f"{ds}|{head}|{kind}|{sg:.2f}"
                    if key in R["speckle"]:
                        continue
                    per_seed = []
                    for seed in a.seeds:
                        cp = ckpt_file(a, ds, head, kind, seed)
                        if not os.path.exists(cp):
                            per_seed = None; break
                        m = build_model(kind, nc, head).to(DEV); m.load_state_dict(torch.load(cp, map_location=DEV, weights_only=True))
                        accs, aucs = [], []
                        for dr in (NOISE_DRAWS if sg > 0 else [0]):
                            xu8, q8 = to_gpu(speckle(te_f, sg, dr))
                            r = evaluate(m, xu8, q8, D["yte"], nc); accs.append(r["acc"]); aucs.append(r["auc"])
                        per_seed.append(dict(seed=seed, acc=float(np.mean(accs)), auc=float(np.mean(aucs))))
                    if per_seed is None:
                        print(f"skip {key}: missing checkpoints"); continue
                    R["speckle"][key] = per_seed
                    st.atomic_dump_json(R, a.results)
            print(f"== speckle {ds}/{head} ==", flush=True)
            for sg in SIGMAS:
                row = []
                for name, kind in MODEL_KINDS:
                    ps = R["speckle"].get(f"{ds}|{head}|{kind}|{sg:.2f}")
                    if ps:
                        row.append(f"{kind}={np.mean([p['acc'] for p in ps])*100:5.2f}±{np.std([p['acc'] for p in ps])*100:4.2f}")
                print(f"   sigma={sg:.2f}  " + "  ".join(row), flush=True)
    full = all(f"{ds}|{h}|{k}|{sg:.2f}" in R["speckle"] for ds in DATASETS for h in ("linear", "resblock")
               for _, k in MODEL_KINDS for sg in SIGMAS)
    if full:
        st.mark("step4_speckle_stress", True, results=a.results)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", default="all", choices=["train", "speckle", "all"])
    p.add_argument("--datasets", default="breastmnist,pneumoniamnist,octmnist")
    p.add_argument("--heads", default="linear,resblock")
    p.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    p.add_argument("--results", default="reports/pillar2_expansion_results.json")
    p.add_argument("--ckpt-dir", default="weights/pillar2_expansion")
    a = p.parse_args()
    a.datasets, a.heads, a.seeds = a.datasets.split(","), a.heads.split(","), [int(s) for s in a.seeds.split(",")]
    global DATA
    DATA = {}
    for ds in a.datasets:
        D = load(DATASETS[ds][0]); D["g_tr"], D["g_va"], D["g_te"] = to_gpu(D["tr"]), to_gpu(D["va"]), to_gpu(D["te"])
        DATA[ds] = D
    if os.path.exists(a.results):
        with open(a.results, "r", encoding="utf-8") as f:
            R = json.load(f)
    else:
        R = {"metadata": dict(seeds=SEEDS, sigmas=SIGMAS, noise_draws=NOISE_DRAWS, datasets={k: v[1:3] for k, v in DATASETS.items()},
                              hparams={k: v[3] for k, v in DATASETS.items()}, device=str(DEV)), "runs": {}}
    if a.stage in ("train", "all"):
        stage_train(a, R)
    if a.stage in ("speckle", "all"):
        stage_speckle(a, R)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
