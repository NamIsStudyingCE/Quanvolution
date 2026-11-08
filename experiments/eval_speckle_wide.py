"""
Đánh giá Độ bền Nhiễu đốm (Speckle Noise Stress-Test) cho Kiến trúc Wide-Fusion v2.
================================================================================
Mô hình đánh giá:
  - Wide Classical Baseline (16ch stem + WideResidualHead)
  - Wide Classical Control (16ch multi-dilation classical conv + WideResidualHead)
  - Proposed Wide-GR-Quanv (16ch Q-LUT Interp4D SIMD + Spatial-Channel Gate + WideResidualHead)

Tập dữ liệu: BreastMNIST, PneumoniaMNIST, OCTMNIST (Ảnh kiểm tra - Test Set).
Mức nhiễu đốm: sigma in [0.00, 0.05, 0.10, 0.15, 0.20], mỗi sigma > 0 có 3 noise draws.
Checkpoints: weights/pillar2_wide/wide_v2_interp4d_{ds}_full_{kind}_s{seed}.pt (5 seeds).

Đầu ra:
  - JSON kết quả: reports/pillar2_wide_speckle_results.json
  - Cập nhật reports/pillar2_wide_results.json với khóa "speckle" và "speckle_stats"
  - Biểu đồ suy giảm: reports/figures/wide_v2_speckle_degradation.png
"""
import argparse, json, os, sys, time
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import torch
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "src")
sys.path.insert(0, "experiments")

import pillar2_state as st
from run_pillar2_expansion import DATASETS, load, SEEDS, SIGMAS, NOISE_DRAWS, speckle
from run_pillar2_wide import load_dataset_with_cache, to_gpu_wide, evaluate_wide, build_wide, WIDE_MODEL_KINDS

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_speckle_bench(datasets=None, seeds=None, sigmas=None, ckpt_dir="weights/pillar2_wide",
                      out_json="reports/pillar2_wide_speckle_results.json",
                      master_results="reports/pillar2_wide_results.json",
                      fig_path="reports/figures/wide_v2_speckle_degradation.png"):
    if datasets is None:
        datasets = ["breastmnist", "pneumoniamnist", "octmnist"]
    if seeds is None:
        seeds = SEEDS
    if sigmas is None:
        sigmas = SIGMAS

    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)

    results = {
        "metadata": {
            "version": "v2_interp4d_bias0.0_prebn_drop0.15",
            "sigmas": sigmas,
            "noise_draws": NOISE_DRAWS,
            "seeds": seeds,
            "datasets": datasets,
            "device": str(DEV),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "raw_draws": {},
        "per_seed": {},
        "aggregated": {},
        "stats": {},
        "gate_dynamics": {},
    }

    # Load master results if available to append
    master_data = None
    if os.path.exists(master_results):
        try:
            with open(master_results, "r", encoding="utf-8") as f:
                master_data = json.load(f)
        except Exception as e:
            print(f"[WARN] Khong the doc master_results: {e}")

    for ds in datasets:
        print(f"\n=======================================================")
        print(f"=== SPECKLE NOISE EVALUATION: {ds.upper()} ===")
        print(f"=======================================================")
        D = load_dataset_with_cache(ds, interp4d=True)
        nc = int(len(np.unique(D["ytr"])))
        te_f = D["te"].astype(np.float32) / 255.0
        n_test = len(te_f)

        # Pre-load all models into memory to avoid repeated disk reads
        models = {}
        for _, kind in WIDE_MODEL_KINDS:
            models[kind] = {}
            for s in seeds:
                cp = os.path.join(ckpt_dir, f"wide_v2_interp4d_{ds}_full_{kind}_s{s}.pt")
                if not os.path.exists(cp):
                    print(f"[WARN] Thieu checkpoint: {cp}")
                    continue
                m = build_wide(kind, nc).to(DEV)
                m.load_state_dict(torch.load(cp, map_location=DEV, weights_only=True))
                m.eval()
                models[kind][s] = m

        for sg in sigmas:
            draws = NOISE_DRAWS if sg > 0 else [0]
            print(f"\n--- [Dataset: {ds}] Sigma = {sg:.2f} ({len(draws)} noise draws) ---")

            draw_metrics = {kind: {s: [] for s in seeds if s in models[kind]} for _, kind in WIDE_MODEL_KINDS}
            gate_alphas = {s: [] for s in seeds if s in models["grquanv"]}

            for dr in draws:
                t0_draw = time.perf_counter()
                noisy_u8 = speckle(te_f, sg, dr)
                # Compute Q-LUT features dynamically for noisy image
                xu8, qbuf = to_gpu_wide(noisy_u8, interp4d=True)
                extract_ms = (time.perf_counter() - t0_draw) * 1000

                for _, kind in WIDE_MODEL_KINDS:
                    for s in seeds:
                        if s not in models[kind]:
                            continue
                        m = models[kind][s]
                        metrics = evaluate_wide(m, xu8, qbuf, D["yte"], nc)
                        draw_metrics[kind][s].append(metrics)
                        if kind == "grquanv" and hasattr(m, "get_gate_summary"):
                            g_sum = m.get_gate_summary()
                            if g_sum["overall_alpha"] is not None:
                                gate_alphas[s].append(g_sum["overall_alpha"])

            # Compute per-seed averages across draws
            for _, kind in WIDE_MODEL_KINDS:
                key = f"{ds}|{kind}|{sg:.2f}"
                seed_list = []
                for s in seeds:
                    if s not in draw_metrics[kind] or not draw_metrics[kind][s]:
                        continue
                    m_list = draw_metrics[kind][s]
                    acc = float(np.mean([m["acc"] for m in m_list]))
                    auc = float(np.mean([m["auc"] for m in m_list]))
                    f1 = float(np.mean([m["f1"] for m in m_list]))
                    seed_list.append({"seed": s, "acc": acc, "auc": auc, "f1": f1})
                results["per_seed"][key] = seed_list

                # Aggregate across seeds
                if seed_list:
                    accs = np.array([item["acc"] for item in seed_list]) * 100.0
                    aucs = np.array([item["auc"] for item in seed_list]) * 100.0
                    f1s = np.array([item["f1"] for item in seed_list]) * 100.0
                    results["aggregated"][key] = {
                        "acc_mean": float(accs.mean()),
                        "acc_std": float(accs.std()),
                        "auc_mean": float(aucs.mean()),
                        "auc_std": float(aucs.std()),
                        "f1_mean": float(f1s.mean()),
                        "f1_std": float(f1s.std()),
                        "n_seeds": len(seed_list),
                    }
                    print(f"  {kind:<10} => ACC: {accs.mean():5.2f}+-{accs.std():4.2f}% | AUC: {aucs.mean():5.2f}+-{aucs.std():4.2f}% | F1: {f1s.mean():5.2f}%")

            # Track gate dynamics
            if gate_alphas:
                all_alpha_vals = [a for sub in gate_alphas.values() for a in sub]
                mean_alpha = float(np.mean(all_alpha_vals)) if all_alpha_vals else 0.0
                results["gate_dynamics"][f"{ds}|{sg:.2f}"] = {
                    "mean_alpha": mean_alpha,
                    "per_seed_alpha": {s: float(np.mean(gate_alphas[s])) for s in gate_alphas if gate_alphas[s]},
                }
                print(f"  [Gate alpha] GR-Quanv alpha: {mean_alpha:.4f}")

            # Statistical significance (paired t-test: control vs grquanv)
            ctrl_key = f"{ds}|control|{sg:.2f}"
            gq_key = f"{ds}|grquanv|{sg:.2f}"
            if ctrl_key in results["per_seed"] and gq_key in results["per_seed"]:
                ctrl_accs = [item["acc"] for item in results["per_seed"][ctrl_key]]
                gq_accs = [item["acc"] for item in results["per_seed"][gq_key]]
                if len(ctrl_accs) == len(gq_accs) and len(ctrl_accs) >= 2:
                    diff = np.array(gq_accs) - np.array(ctrl_accs)
                    tt = stats.ttest_rel(gq_accs, ctrl_accs)
                    stat_key = f"{ds}|{sg:.2f}"
                    results["stats"][stat_key] = {
                        "delta_acc_pct": float(diff.mean() * 100.0),
                        "t_stat": float(tt.statistic),
                        "p_value": float(tt.pvalue),
                        "significant_005": bool(tt.pvalue < 0.05),
                        "significant_001": bool(tt.pvalue < 0.01),
                    }
                    p_sym = "***" if tt.pvalue < 0.001 else ("**" if tt.pvalue < 0.01 else ("*" if tt.pvalue < 0.05 else "ns"))
                    print(f"  [T-Test] GR - Ctrl: Delta={diff.mean()*100:+.2f}% | t={tt.statistic:+.3f} | p={tt.pvalue:.4f} ({p_sym})")

    # Save results to JSON
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[INFO] Saved speckle noise results to: {out_json}")

    # Merge into master results
    if master_data is not None:
        master_data["speckle"] = results["per_seed"]
        master_data["speckle_aggregated"] = results["aggregated"]
        master_data["speckle_stats"] = results["stats"]
        master_data["speckle_gate_dynamics"] = results["gate_dynamics"]
        with open(master_results, "w", encoding="utf-8") as f:
            json.dump(master_data, f, indent=2, ensure_ascii=False)
        print(f"[INFO] Updated master results: {master_results}")

    # Plot degradation figure
    plot_speckle_curves(results, fig_path)

    return results


def plot_speckle_curves(results, fig_path):
    datasets = results["metadata"]["datasets"]
    sigmas = results["metadata"]["sigmas"]
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.0 * len(datasets), 4.2), sharey=False)
    if len(datasets) == 1:
        axes = [axes]

    styles = {
        "classical": ("Wide Classical Baseline", "#7f7f7f", "--", "s"),
        "control": ("Wide Classical Control (0% Q)", "#d62728", "-.", "^"),
        "grquanv": ("Proposed Wide-GR-Quanv", "#2ca02c", "-", "o"),
    }

    for ax, ds in zip(axes, datasets):
        for kind, (label, color, ls, marker) in styles.items():
            means, stds = [], []
            for sg in sigmas:
                key = f"{ds}|{kind}|{sg:.2f}"
                if key in results["aggregated"]:
                    means.append(results["aggregated"][key]["acc_mean"])
                    stds.append(results["aggregated"][key]["acc_std"])
                else:
                    means.append(np.nan)
                    stds.append(np.nan)
            ax.errorbar(sigmas, means, yerr=stds, capsize=3, label=label, color=color,
                        linestyle=ls, marker=marker, linewidth=1.8, markersize=5)

        # Plot gate alpha on secondary axis
        ax_alpha = ax.twinx()
        alphas = []
        for sg in sigmas:
            dyn = results["gate_dynamics"].get(f"{ds}|{sg:.2f}")
            if dyn:
                alphas.append(dyn["mean_alpha"])
            else:
                alphas.append(np.nan)
        ax_alpha.plot(sigmas, alphas, color="#1f77b4", linestyle=":", marker="x", alpha=0.7, label=r"Gate $\alpha$")
        ax_alpha.set_ylabel(r"Gate weight $\alpha$", color="#1f77b4", fontsize=9)
        ax_alpha.tick_params(axis="y", labelcolor="#1f77b4")
        ax_alpha.set_ylim([0.2, 0.7])

        ax.set_title(f"MedMNIST: {ds.upper()}", fontsize=11, fontweight="bold")
        ax.set_xlabel(r"Speckle Noise Level $\sigma$", fontsize=10)
        ax.set_ylabel("Test Accuracy (%)", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_xticks(sigmas)
        if ds == datasets[0]:
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax_alpha.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc="lower left", fontsize=8)

    fig.suptitle("Wide-Fusion v2: Speckle Noise Robustness & Gate Dynamics (5 seeds ± std)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved speckle degradation figure to: {fig_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default="breastmnist,pneumoniamnist,octmnist")
    p.add_argument("--sigmas", default="0.0,0.05,0.10,0.15,0.20")
    p.add_argument("--seeds", default="0,42,100,2023,777")
    p.add_argument("--ckpt-dir", default="weights/pillar2_wide")
    p.add_argument("--results", default="reports/pillar2_wide_speckle_results.json")
    p.add_argument("--master-results", default="reports/pillar2_wide_results.json")
    p.add_argument("--fig-path", default="reports/figures/wide_v2_speckle_degradation.png")
    a = p.parse_args()

    ds_list = [d.strip() for d in a.datasets.split(",") if d.strip()]
    sg_list = [float(s.strip()) for s in a.sigmas.split(",") if s.strip()]
    sd_list = [int(s.strip()) for s in a.seeds.split(",") if s.strip()]

    run_speckle_bench(datasets=ds_list, seeds=sd_list, sigmas=sg_list, ckpt_dir=a.ckpt_dir,
                      out_json=a.results, master_results=a.master_results, fig_path=a.fig_path)
