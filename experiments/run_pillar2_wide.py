"""
Cột trụ 2 - Tác vụ 2 & 3: Kiến trúc Wide-Fusion `gr_quanv_wide.py`.
===================================================================
- Stage `train`  : full-data, 3 mô hình wide (Baseline / Control / GR-Quanv), 5 seeds,
                   3 epochs Linear Warmup + Cosine, epoch sâu hơn (Pneumonia 50, OCT 25).
- Stage `lowdata`: sample-efficiency 5%/10%/20% train của Pneumonia & OCT (ảnh sạch),
                   Wide Classical Control vs Proposed Wide-GR-Quanv.
- Stage `report` : tổng hợp bảng Mean±Std (5 seeds) + t-test ghép cặp theo seed +
                   biểu đồ vào reports/figures/.
Bố trí thí nghiệm: cùng head/mix/gate/augment/siêu tham số, chọn epoch CHỈ theo val,
test đánh giá đúng 1 lần. Resumable: mỗi (stage, ds, cfg, kind, seed) là 1 khóa JSON,
đã có thì bỏ qua; ghi atomic sau từng run (không làm lại từ đầu khi bị ngắt).
Dùng: python experiments/run_pillar2_wide.py --stage all
"""
import argparse, json, os, sys, time, warnings
import numpy as np, torch, torch.nn as nn, torch.optim as optim

warnings.filterwarnings("ignore", message=".*does not have many workers.*")
warnings.filterwarnings("ignore", message=".*was not necessary and is being deprecated.*")

sys.path.insert(0, "src"); sys.path.insert(0, "experiments")
import pillar2_state as st
from models.gr_quanv_resblock import GPUMedicalAugment
from models.gr_quanv_wide import build_model as build_wide, WIDE_MODEL_KINDS
from run_pillar2_expansion import DATASETS, load, to_gpu, deq, evaluate, augment_epoch, SEEDS

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA = {}   # gán trong main(); để module còn dùng được khi smoke test từ ngoài
LOWDATA_FS = [0.05, 0.10, 0.20]          # 5% / 10% / 20% train
LOWDATA_DS = ["pneumoniamnist", "octmnist"]
LOWDATA_KINDS = ["control", "grquanv"]   # đối chứng và đề xuất
WARMUP_EPOCHS = 3

# (epochs, bs, lr, wd): giữ bs/lr/wd của pipeline resblock gốc, chỉ tăng epoch + warmup.
WIDE_HP = {
    "breastmnist":    (60, 32, 1.0e-3, 5e-4),
    "pneumoniamnist": (50, 128, 1.5e-3, 5e-4),
    "octmnist":       (25, 256, 2.0e-3, 5e-4),
}


def make_scheduler(opt, epochs: int) -> optim.lr_scheduler.SequentialLR:
    main = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs - WARMUP_EPOCHS))
    warm = optim.lr_scheduler.LinearLR(opt, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_EPOCHS)
    return optim.lr_scheduler.SequentialLR(opt, [warm, main], milestones=[WARMUP_EPOCHS])


def stratified_subset(y: np.ndarray, frac: float, seed: int) -> np.ndarray:
    """Mẫu gợi nhớ-pi cổ điển giữ nguyên tỷ lệ lớp, cố định theo (seed, frac)."""
    rng = np.random.default_rng(seed)
    keep = []
    for c in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == c))
        keep.append(idx[: max(1, int(round(len(idx) * frac)))])
    return np.sort(np.concatenate(keep))


def cfg_frac(cfg: str) -> float:
    return 1.0 if cfg == "full" else float(cfg[1:]) / 100.0


def _jsonable(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    return v


def train_run(ds: str, cfg: str, kind: str, seed: int, ckpt_path: str) -> dict:
    """Huấn luyện 1 mô hình wide trên 1 (tập, cấu hình, seed). cfg = 'full' | 'r5' | 'r10' | 'r20'."""
    epochs, bs, lr, wd = WIDE_HP[ds]
    _, sel, tau, _ = DATASETS[ds]
    D = DATA[ds]
    ytr_np = D["ytr"]
    nc = int(len(np.unique(ytr_np)))
    if cfg == "full":
        keep = np.arange(len(ytr_np))
    else:
        keep = stratified_subset(ytr_np, cfg_frac(cfg), seed)
    ysub_np = ytr_np[keep]

    torch.manual_seed(seed); np.random.seed(seed); torch.cuda.manual_seed_all(seed)
    model = build_wide(kind, nc).to(DEV)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    ysub = torch.from_numpy(ysub_np).to(DEV)
    counts = np.bincount(ysub_np, minlength=nc)
    logp = torch.log(torch.tensor(counts / counts.sum(), dtype=torch.float32, device=DEV) + 1e-8)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = make_scheduler(opt, epochs)
    crit = nn.CrossEntropyLoss()
    g = torch.Generator(device=DEV); g.manual_seed(seed)
    aug = GPUMedicalAugment()
    xtr_sub = D["g_tr"][0][keep]        # uint8 trên GPU, Q-LUT tính lại trên ảnh đã augment
    N = len(ysub)
    best, best_state, best_ep = -1.0, None, -1
    t0 = time.perf_counter()
    for ep in range(epochs):
        xe, qe = augment_epoch(xtr_sub, aug, g)
        model.train()
        perm = torch.randperm(N, device=DEV, generator=g)
        for i in range(0, N, bs):
            idx = perm[i:i + bs]
            if len(idx) < 2:
                continue
            x, q = deq(xe[idx], qe[idx])
            opt.zero_grad()
            logits = model(x, q) + tau * logp
            crit(logits, ysub[idx]).backward()
            opt.step()
        sched.step()
        v = evaluate(model, *D["g_va"], D["yva"], nc)[sel]
        if v > best:
            best, best_ep = v, ep
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
    model.load_state_dict(best_state)
    m = evaluate(model, *D["g_te"], D["yte"], nc)  # test: đúng 1 lần
    m.update(params=params, best_val=float(best), best_epoch=best_ep,
             train_sec=time.perf_counter() - t0, cfg=cfg, frac=cfg_frac(cfg), kind=kind, ds=ds, seed=seed)
    if hasattr(model, "get_gate_summary"):
        m["gate"] = {k: _jsonable(v) for k, v in model.get_gate_summary().items()}
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save({k: t.cpu() for k, t in best_state.items()}, ckpt_path)
    return m


def ckpt_file(a, ds, cfg, kind, seed):
    return os.path.join(a.ckpt_dir, f"wide_{ds}_{cfg}_{kind}_s{seed}.pt")


def stage_loop(a, R, datasets, kinds, cfgs):
    for ds in datasets:
        for cfg in cfgs:
            frac_lab = "100%" if cfg == "full" else f"{cfg_frac(cfg)*100:g}%"
            for name, kind in kinds:
                for seed in a.seeds:
                    key = f"{ds}|{cfg}|{kind}|{seed}"
                    if key in R["runs"]:
                        continue
                    r = train_run(ds, cfg, kind, seed, ckpt_file(a, ds, cfg, kind, seed))
                    R["runs"][key] = {k: _jsonable(v) for k, v in r.items()}
                    st.atomic_dump_json(R, a.results)
                    print(f"[{ds}|{cfg}({frac_lab})|{name}|s{seed}] ACC={r['acc']*100:.2f} AUC={r['auc']*100:.2f} "
                          f"F1={r['f1']*100:.2f} params={int(r['params'])} ep*={int(r['best_epoch'])} "
                          f"{r['train_sec']:.0f}s", flush=True)
            print(f"== {ds}/{cfg} ({frac_lab}) summary ==", flush=True)
            for name, kind in kinds:
                rs = [R["runs"][f"{ds}|{cfg}|{kind}|{s}"] for s in a.seeds if f"{ds}|{cfg}|{kind}|{s}" in R["runs"]]
                if rs:
                    accs = np.array([r["acc"] for r in rs]) * 100
                    aucs = np.array([r["auc"] for r in rs]) * 100
                    f1s = np.array([r["f1"] for r in rs]) * 100
                    print(f"   {name:<34} ACC={accs.mean():5.2f}±{accs.std():4.2f} AUC={aucs.mean():5.2f}±{aucs.std():4.2f} F1={f1s.mean():5.2f}", flush=True)


def run_stage(a, R, datasets, kinds, cfgs):
    stage_loop(a, R, datasets, kinds, cfgs)


def agg_runs(R, ds, cfg, kind, seeds):
    rs = [R["runs"][f"{ds}|{cfg}|{kind}|{s}"] for s in seeds if f"{ds}|{cfg}|{kind}|{s}" in R["runs"]]
    if not rs:
        return None
    out = {}
    for k in ("acc", "auc", "f1"):
        v = np.array([r[k] for r in rs]) * 100
        out[k] = (float(v.mean()), float(v.std()))
    out["params"] = int(rs[0]["params"])
    out["n"] = len(rs)
    return out


def stage_report(a, R):
    from scipy import stats
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker

    st_obj = R.setdefault("stats", {})
    lines = ["# Báo cáo Wide-Fusion (Tác vụ 1-3): gỡ nút thắt 4 kênh, warmup+cosine, sample-efficiency", "",
             f"Nguồn: `{a.results}`; mã: `experiments/run_pillar2_wide.py`, `src/models/gr_quanv_wide.py`. "
             f"5 seeds {a.seeds}, chọn epoch theo val, test đánh giá 1 lần, warmup {WARMUP_EPOCHS} epochs + cosine.",
             ""]

    clean_table = []
    for ds in a.datasets:
        rows = []
        for name, kind in WIDE_MODEL_KINDS:
            s = agg_runs(R, ds, "full", kind, a.seeds)
            if s is None:
                continue
            rows.append((name, kind, s))
        if len(rows) < 3:
            continue
        lines.append(f"## Sạch (full train) - {ds} - WideResidualHead")
        lines += ["| Mô hình | Tham số | ACC (%) | AUC (%) | F1 (%) |", "|---|---|---|---|---|"]
        for name, _, s in rows:
            lines.append(f"| {name} | {s['params']} | {s['acc'][0]:.2f} ± {s['acc'][1]:.2f} | "
                         f"{s['auc'][0]:.2f} ± {s['auc'][1]:.2f} | {s['f1'][0]:.2f} ± {s['f1'][1]:.2f} |")
        s_g = agg_runs(R, ds, "full", "grquanv", a.seeds)
        s_c = agg_runs(R, ds, "full", "control", a.seeds)
        acc_g = np.array([R["runs"][f"{ds}|full|grquanv|{s}"]["acc"] for s in a.seeds]) * 100
        acc_c = np.array([R["runs"][f"{ds}|full|control|{s}"]["acc"] for s in a.seeds]) * 100
        auc_g = np.array([R["runs"][f"{ds}|full|grquanv|{s}"]["auc"] for s in a.seeds]) * 100
        auc_c = np.array([R["runs"][f"{ds}|full|control|{s}"]["auc"] for s in a.seeds]) * 100
        p_acc = float(stats.ttest_rel(acc_g, acc_c).pvalue)
        p_auc = float(stats.ttest_rel(auc_g, auc_c).pvalue)
        lines += ["", f"**ΔACC** = {acc_g.mean()-acc_c.mean():+.2f} điểm (p = {p_acc:.3f}); "
                      f"**ΔAUC** = {auc_g.mean()-auc_c.mean():+.2f} điểm (p = {p_auc:.3f}) (t-test ghép cặp n=5).", ""]
        st_obj[f"{ds}|full"] = dict(dacc=float(acc_g.mean() - acc_c.mean()), p_acc=p_acc,
                                    dauc=float(auc_g.mean() - auc_c.mean()), p_auc=p_auc)
        clean_table.append((ds, rows, acc_g.mean() - acc_c.mean(), p_acc))

    lines.append("## Kịch bản ít dữ liệu (ảnh sạch; train 5/10/20%)")
    for ds in LOWDATA_DS:
        lines += ["", f"### {ds}", "| Tỷ lệ train | Wide Control ACC (%) | Wide-GR-Quanv ACC (%) | ΔACC | p |", "|---|---|---|---|---|"]
        st_obj[f"{ds}|lowdata"] = {}
        for cfg in ("r5", "r10", "r20"):
            c, g = agg_runs(R, ds, cfg, "control", a.seeds), agg_runs(R, ds, cfg, "grquanv", a.seeds)
            if not c or not g:
                continue
            acc_c = np.array([R["runs"][f"{ds}|{cfg}|control|{s}"]["acc"] for s in a.seeds]) * 100
            acc_g = np.array([R["runs"][f"{ds}|{cfg}|grquanv|{s}"]["acc"] for s in a.seeds]) * 100
            p = float(stats.ttest_rel(acc_g, acc_c).pvalue)
            d = acc_g.mean() - acc_c.mean()
            lines.append(f"| {cfg_frac(cfg)*100:g}% | {c['acc'][0]:.2f} ± {c['acc'][1]:.2f} | "
                         f"{g['acc'][0]:.2f} ± {g['acc'][1]:.2f} | {d:+.2f} | {p:.3f} |")
            st_obj[f"{ds}|lowdata"][cfg] = dict(dacc=float(d), p=p,
                                                acc_control=[float(v) for v in acc_c],
                                                acc_grquanv=[float(v) for v in acc_g])
        s_g, s_c = agg_runs(R, ds, "full", "grquanv", a.seeds), agg_runs(R, ds, "full", "control", a.seeds)
        if s_g and s_c:
            lines.append(f"| 100% (full) | {s_c['acc'][0]:.2f} ± {s_c['acc'][1]:.2f} | {s_g['acc'][0]:.2f} ± {s_g['acc'][1]:.2f} | "
                         f"{s_g['acc'][0]-s_c['acc'][0]:+.2f} | - |")
        lines.append("")

    alpha_rows = []
    for _, kind in WIDE_MODEL_KINDS:
        if kind == "classical":
            continue
        for ds in a.datasets:
            rs = [R["runs"][k] for k in R["runs"] if k.startswith(f"{ds}|full|{kind}|")]
            vals = [r.get("gate", {}).get("overall_alpha") for r in rs if isinstance(r.get("gate", {}).get("overall_alpha"), (int, float))]
            if vals:
                alpha_rows.append((ds, kind, float(np.mean(vals))))
    if alpha_rows:
        lines += ["## Cổng alpha trung bình sau huấn luyện (trên test, mean 5 seeds)", "", "| Tập | Mô hình | alpha |", "|---|---|---|"]
        for ds, kind, v in alpha_rows:
            lines.append(f"| {ds} | {kind} | {v:.3f} |")
        lines.append("")

    os.makedirs(os.path.dirname(a.report) or ".", exist_ok=True)
    with open(a.report, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    st.atomic_dump_json(R, a.results)

    # ---- Biểu đồ ----
    fig_dir = a.fig_dir
    os.makedirs(fig_dir, exist_ok=True)
    colors = ["#5b7fb9", "#c96a5a", "#3f8a5f"]

    if clean_table:
        fig, ax = plt.subplots(figsize=(10, 4.2))
        xs = np.arange(len(clean_table)); w = 0.26
        for j, (name, kind, _) in enumerate(clean_table[0][1]):
            ms, ss = [], []
            for _, rows, _, _ in clean_table:
                row = next(v for _, k, v in rows if k == kind)
                ms.append(row["acc"][0])
                ss.append(row["acc"][1])
            ax.bar(xs + (j - 1) * w, ms, w, yerr=ss, capsize=3, label=name, color=colors[j])
        ax.set_xticks(xs); ax.set_xticklabels([c[0] for c in clean_table])
        ax.set_ylabel("Test ACC sạch (%)"); ax.set_title("Wide-Fusion: Clean Test ACC (5 seeds ± std)")
        ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "pillar2_wide_clean_comparison.png"), dpi=150); plt.close(fig)

    lowdata_plottable = [(ds, cfg) for cfg in ("r5", "r10", "r20", "full") for ds in LOWDATA_DS
                         if agg_runs(R, ds, cfg, "control", a.seeds) and agg_runs(R, ds, cfg, "grquanv", a.seeds)]
    if lowdata_plottable:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.0), sharey=False)
        frac_ticks = [0.05, 0.10, 0.20, 1.0]
        for ax, ds in zip(axes, LOWDATA_DS):
            for kind, label, cl in (("control", "Wide Classical Control", "#c96a5a"), ("grquanv", "Proposed Wide-GR-Quanv", "#3f8a5f")):
                pts = []
                for fr in frac_ticks:
                    cfg = "full" if fr == 1.0 else f"r{int(round(fr*100))}"
                    s = agg_runs(R, ds, cfg, kind, a.seeds)
                    if s:
                        pts.append((fr, s["acc"][0], s["acc"][1]))
                if pts:
                    fx, fy, fe = zip(*pts)
                    ax.errorbar(fx, fy, yerr=fe, capsize=3, marker="o", label=label, color=cl)
            ax.set_xscale("log"); ax.set_xticks(frac_ticks)
            ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax.set_xlabel("Tỷ lệ train (%)"); ax.set_title(ds); ax.grid(alpha=0.3); ax.legend(fontsize=8)
        axes[0].set_ylabel("Test ACC sạch (%)")
        fig.suptitle("Sample-efficiency: Wide-GR-Quanv vs Control (5 seeds ± std)")
        fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "pillar2_wide_lowdata.png"), dpi=150); plt.close(fig)
    print(f"[report] markdown -> {a.report}; figures -> {fig_dir}/", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", default="all", choices=["train", "lowdata", "report", "all"])
    p.add_argument("--datasets", default="breastmnist,pneumoniamnist,octmnist")
    p.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    p.add_argument("--results", default="reports/pillar2_wide_results.json")
    p.add_argument("--report", default="reports/pillar2_wide_report.md")
    p.add_argument("--ckpt-dir", default="weights/pillar2_wide")
    p.add_argument("--fig-dir", default="reports/figures")
    a = p.parse_args()
    a.datasets = a.datasets.split(",")
    a.seeds = [int(s) for s in a.seeds.split(",")]

    global DATA
    DATA = {}
    if a.stage != "report":
        for ds in a.datasets:
            D = load(DATASETS[ds][0])
            D["g_tr"], D["g_va"], D["g_te"] = to_gpu(D["tr"]), to_gpu(D["va"]), to_gpu(D["te"])
            DATA[ds] = D
    if os.path.exists(a.results):
        with open(a.results, "r", encoding="utf-8") as f:
            R = json.load(f)
        R.setdefault("runs", {})
    else:
        R = {"metadata": dict(seeds=a.seeds, lowdata_fs=LOWDATA_FS, warmup=WARMUP_EPOCHS,
                              hparams=WIDE_HP, device=str(DEV), architecture="gr_quanv_wide (fusion 32ch, gate bias -1.5)"),
             "stats": {}, "runs": {}}

    if a.stage in ("train", "all"):
        run_stage(a, R, a.datasets, WIDE_MODEL_KINDS, ["full"])
        st.mark("wide_train_full", True, artifact="experiments/run_pillar2_wide.py", results=a.results)
    if a.stage in ("lowdata", "all"):
        lists = [d for d in a.datasets if d in LOWDATA_DS]
        run_stage(a, R, lists, [("Wide Classical Control (0% quantum)", "control"),
                                ("Proposed Wide-GR-Quanv", "grquanv")],
                  [f"r{int(round(f * 100))}" for f in LOWDATA_FS])
        st.mark("wide_lowdata", True, artifact="experiments/run_pillar2_wide.py --stage lowdata", results=a.results)
    if a.stage in ("report", "all"):
        stage_report(a, R)
        st.mark("wide_report", True, artifact="experiments/run_pillar2_wide.py --stage report", results=a.results)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
