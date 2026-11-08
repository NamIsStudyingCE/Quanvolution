"""
BƯỚC 5: Biểu đồ + báo cáo Markdown từ reports/pillar2_expansion_results.json
 -> reports/figures/multidataset_comparison.png, reports/figures/speckle_robustness_curve.png
 -> reports/pillar2_expansion_report.md
Kiểm định: t-test ghép cặp theo seed (GR-Quanv vs Control cổ điển cùng dung lượng) + Cohen's d.
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, "experiments")
import pillar2_state as st

R = json.load(open("reports/pillar2_expansion_results.json", encoding="utf-8"))
SEEDS, SIGMAS = R["metadata"]["seeds"], R["metadata"]["sigmas"]
DS = ["breastmnist", "pneumoniamnist", "octmnist"]
HEADS = {"linear": "Linear head (không augment)", "resblock": "ResidualQuanvHead + Augment"}
KINDS = [("classical", "Classical Baseline"), ("control", "Classical Control (0% quantum)"), ("grquanv", "Proposed GR-Quanv")]
COL = {"classical": "#9aa0a6", "control": "#4285f4", "grquanv": "#ea4335"}


def runs(ds, h, k, key):
    return np.array([R["runs"][f"{ds}|{h}|{k}|{s}"][key] for s in SEEDS if f"{ds}|{h}|{k}|{s}" in R["runs"]]) * 100


def fmt(v):
    return f"{v.mean():.2f} ± {v.std():.2f}"


def paired(a, b):
    if len(a) < 3 or len(a) != len(b):
        return float("nan"), float("nan")
    d = a - b
    t, p = stats.ttest_rel(a, b)
    return float(d.mean()), float(p)


os.makedirs("reports/figures", exist_ok=True)

# ---- Figure 1: multi-dataset bars
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
for r, metric in enumerate(["acc", "auc"]):
    for c, h in enumerate(HEADS):
        ax = axes[r, c]; w = 0.26
        for j, (k, nm) in enumerate(KINDS):
            m = [runs(d, h, k, metric).mean() if len(runs(d, h, k, metric)) else 0 for d in DS]
            s = [runs(d, h, k, metric).std() if len(runs(d, h, k, metric)) else 0 for d in DS]
            ax.bar(np.arange(3) + (j - 1) * w, m, w, yerr=s, capsize=3, color=COL[k], label=nm)
        ax.set_xticks(range(3)); ax.set_xticklabels(["BreastMNIST", "PneumoniaMNIST", "OCTMNIST"])
        ax.set_ylim(40, 100); ax.set_ylabel(metric.upper() + " (%)"); ax.set_title(f"{metric.upper()} - {HEADS[h]}")
        ax.grid(axis="y", alpha=.3)
axes[0, 0].legend(fontsize=8, loc="lower left")
plt.tight_layout(); plt.savefig("reports/figures/multidataset_comparison.png", dpi=300); plt.close()

# ---- Figure 2: speckle curves
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for r, h in enumerate(HEADS):
    for c, d in enumerate(DS):
        ax = axes[r, c]
        for k, nm in KINDS:
            ys = []
            for sg in SIGMAS:
                ps = R.get("speckle", {}).get(f"{d}|{h}|{k}|{sg:.2f}")
                ys.append([p["acc"] * 100 for p in ps] if ps else [np.nan] * len(SEEDS))
            ys = np.array(ys)
            ax.errorbar(SIGMAS, np.nanmean(ys, 1), yerr=np.nanstd(ys, 1), marker="o", capsize=3, color=COL[k], label=nm)
        ax.set_title(f"{d} | {h}"); ax.set_xlabel("Speckle sigma"); ax.set_ylabel("Accuracy (%)"); ax.grid(alpha=.3)
axes[0, 0].legend(fontsize=8)
plt.tight_layout(); plt.savefig("reports/figures/speckle_robustness_curve.png", dpi=300); plt.close()

# ---- Report
L = ["# Báo cáo mở rộng Cột trụ 2: đa tập dữ liệu, Residual head, kháng nhiễu đốm",
     "",
     "Nguồn: `reports/pillar2_expansion_results.json`; mã: `experiments/run_pillar2_expansion.py`. "
     f"5 seeds {SEEDS}, chọn epoch theo val, test đánh giá 1 lần, mọi mô hình cùng head/augment/siêu tham số.",
     "Tham số: Control cổ điển luôn >= GR-Quanv (đối chứng không bị thiệt).", ""]
for h, hn in HEADS.items():
    L += [f"## Kết quả sạch - {hn}", "", "| Tập | Mô hình | Tham số | ACC (%) | AUC (%) | F1 (%) |", "|---|---|---|---|---|---|"]
    for d in DS:
        for k, nm in KINDS:
            key = f"{d}|{h}|{k}|{SEEDS[0]}"
            if key in R["runs"]:
                L.append(f"| {d} | {nm} | {int(R['runs'][key]['params'])} | {fmt(runs(d,h,k,'acc'))} | {fmt(runs(d,h,k,'auc'))} | {fmt(runs(d,h,k,'f1'))} |")
    L += ["", "GR-Quanv so với Control cổ điển (t-test ghép cặp theo seed, n=5):", "",
          "| Tập | ΔACC (điểm) | p (ACC) | ΔAUC (điểm) | p (AUC) |", "|---|---|---|---|---|"]
    for d in DS:
        a, b = runs(d, h, "grquanv", "acc"), runs(d, h, "control", "acc")
        au, bu = runs(d, h, "grquanv", "auc"), runs(d, h, "control", "auc")
        da, pa = paired(a, b); du, pu = paired(au, bu)
        L.append(f"| {d} | {da:+.2f} | {pa:.3f} | {du:+.2f} | {pu:.3f} |")
    L.append("")
L += ["## Kháng nhiễu đốm (huấn luyện trên ảnh sạch, test có nhiễu; mỗi sigma trung bình 3 lần lấy mẫu nhiễu)", ""]
for h, hn in HEADS.items():
    for d in DS:
        L += [f"### {d} - {hn}", "", "| sigma | Classical Baseline | Control (0% quantum) | GR-Quanv | Δ(GR-Control) | p |", "|---|---|---|---|---|---|"]
        for sg in SIGMAS:
            cells, arrs = [], {}
            for k, _ in KINDS:
                ps = R.get("speckle", {}).get(f"{d}|{h}|{k}|{sg:.2f}")
                arrs[k] = np.array([p["acc"] for p in ps]) * 100 if ps else None
                cells.append(fmt(arrs[k]) if ps else "n/a")
            if arrs["grquanv"] is not None and arrs["control"] is not None:
                dd, pp = paired(arrs["grquanv"], arrs["control"])
                L.append(f"| {sg:.2f} | " + " | ".join(cells) + f" | {dd:+.2f} | {pp:.3f} |")
        L.append("")
L += ["## Biểu đồ", "", "![multi](figures/multidataset_comparison.png)", "", "![speckle](figures/speckle_robustness_curve.png)", ""]
open("reports/pillar2_expansion_report.md", "w", encoding="utf-8").write("\n".join(L))
st.mark("step5_report_figures", True)
print("Report + figures written")
