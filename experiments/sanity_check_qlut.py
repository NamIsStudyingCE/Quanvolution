"""
Sanity check Q-LUT  (Step 3 của MVE)
====================================
Đo THẬT (không bịa số) trên N ảnh ngẫu nhiên của tập TEST MedMNIST:

  A. Fidelity : đặc trưng Q-LUT so với Ground Truth = đường chạy NAIVE gốc của repo (PennyLane, float liên tục)
                MAE, RMSE, nRMSE (RMSE/std GT), Cosine similarity trung bình theo ảnh, max |err|, P99 |err|
  B. Latency  : ms/ảnh & FPS của các engine; 2 chế độ (batch=1 và batch=N) x 2 cấu hình luồng (mặc định / 1 luồng)

Chạy:
    python -m experiments.sanity_check_qlut --datasets breastmnist octmnist --n-images 100
Kết quả: reports/qlut_mve_results.json|md  và  reports/figures/qlut_fidelity_latency_benchmark.png
"""
import argparse
import json
import os
import platform
import statistics
import sys
import time

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from src.qlut.compiler import compile_qlut, load_qlut  # noqa: E402
from src.qlut.inference_engine import (ExactBatchedQuanv2d, PennyLaneQuanv2d,  # noqa: E402
                                       QLUT_Quanv2d)

REPORT_DIR = os.path.join(REPO_ROOT, "reports")
FIG_DIR = os.path.join(REPORT_DIR, "figures")


# ------------------------------------------------------------------ data
def load_test_images(name: str, n: int, seed: int) -> torch.Tensor:
    path = os.path.join(REPO_ROOT, "data", f"{name}.npz")
    if os.path.exists(path):
        imgs = np.load(path)["test_images"]
    else:  # fallback: để medmnist tự tải
        import medmnist
        cls = {"breastmnist": medmnist.BreastMNIST, "octmnist": medmnist.OCTMNIST}[name]
        imgs = cls(split="test", download=True, size=28).imgs
    rng = np.random.default_rng(seed)
    sel = rng.choice(len(imgs), size=min(n, len(imgs)), replace=False)
    return torch.from_numpy(imgs[sel].astype(np.float32) / 255.0).unsqueeze(1)  # (n,1,28,28) in [0,1]


# ------------------------------------------------------------------ fidelity
def fidelity(pred: torch.Tensor, gt: torch.Tensor) -> dict:
    err = (pred - gt).abs()
    cos = torch.nn.functional.cosine_similarity(pred.flatten(1), gt.flatten(1), dim=1)
    rmse = torch.sqrt(((pred - gt) ** 2).mean()).item()
    return {
        "MAE": err.mean().item(),
        "RMSE": rmse,
        "nRMSE": rmse / gt.std().item(),
        "Cosine": cos.mean().item(),
        "MaxAbsErr": err.max().item(),
        "P99AbsErr": torch.quantile(err.flatten(), 0.99).item(),
    }


# ------------------------------------------------------------------ latency
def numpy_lut_nearest(table_flat: np.ndarray, n_bits: int):
    """Biến thể thuần NumPy (ít overhead hơn PyTorch ở batch nhỏ) — để thấy rõ phần overhead framework."""
    L = 1 << n_bits
    sh = np.array([3 * n_bits, 2 * n_bits, n_bits, 0], dtype=np.int64)

    def run(x: np.ndarray) -> np.ndarray:  # x (B,1,28,28) float32
        B = x.shape[0]
        p = x.reshape(B, 14, 2, 14, 2).transpose(0, 1, 3, 2, 4).reshape(B, 196, 4)
        q = np.rint(p * (L - 1)).astype(np.int64)
        idx = (q << sh).sum(-1)
        return table_flat[idx].transpose(0, 2, 1).reshape(B, 4, 14, 14)

    return run


def time_calls(fn, inputs: list, warmup: int) -> list:
    """Trả về list thời gian (ms) cho mỗi lần gọi fn(inputs[i]); warmup lần đầu không tính."""
    for i in range(warmup):
        fn(inputs[i % len(inputs)])
    ts = []
    for inp in inputs:
        t0 = time.perf_counter_ns()
        fn(inp)
        ts.append((time.perf_counter_ns() - t0) / 1e6)
    return ts


def summarize_latency(ts_ms: list, imgs_per_call: int) -> dict:
    med = statistics.median(ts_ms)
    p95 = float(np.percentile(ts_ms, 95))
    return {"ms_per_img_median": med / imgs_per_call, "ms_per_img_p95": p95 / imgs_per_call,
            "FPS_median": imgs_per_call / (med / 1000.0), "calls": len(ts_ms)}


def bench_engines(engines: dict, x: torch.Tensor, warmup: int, batch_repeats: int) -> dict:
    """engines: name -> (callable_on_tensor, is_torch). Trả về {name: {'b1': {...}, 'bN': {...}}}."""
    out = {}
    singles = [x[i:i + 1] for i in range(x.shape[0])]
    for name, (fn, kind) in engines.items():
        conv = (lambda t: fn(t.numpy())) if kind == "numpy" else fn
        r = {}
        r["b1"] = summarize_latency(time_calls(conv, singles, warmup), 1)
        if kind == "naive":  # naive vốn chạy từng ảnh tuần tự: batch=N không khác -> dùng lại số b1
            r["bN"] = dict(r["b1"])
        else:
            r["bN"] = summarize_latency(time_calls(conv, [x] * batch_repeats, min(warmup, 3)), x.shape[0])
        out[name] = r
        print(f"    {name:<28} b1: {r['b1']['ms_per_img_median']:.4f} ms/img | "
              f"bN: {r['bN']['ms_per_img_median']:.4f} ms/img", flush=True)
    return out


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["breastmnist", "octmnist"])
    ap.add_argument("--n-images", type=int, default=100)
    ap.add_argument("--circuit", default="strongly_L2")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--batch-repeats", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    os.makedirs(FIG_DIR, exist_ok=True)
    tables = {}
    for b in (4, 5):
        try:
            tables[b] = load_qlut(b, args.circuit)
        except FileNotFoundError:
            tables[b], _ = compile_qlut(b, args.circuit, save=True)

    default_threads = torch.get_num_threads()
    results = {"meta": {"circuit": args.circuit, "n_images": args.n_images, "seed": args.seed,
                        "cpu": platform.processor(), "platform": platform.platform(),
                        "torch": torch.__version__, "torch_threads_default": default_threads,
                        "table_MiB": {f"{b}bit": tables[b].numel() * 4 / 2 ** 20 for b in tables},
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")},
               "datasets": {}}

    conv = nn.Conv2d(1, 4, kernel_size=2, stride=2).eval()
    for ds in args.datasets:
        print(f"\n=== {ds} (n={args.n_images}) ===", flush=True)
        x = load_test_images(ds, args.n_images, args.seed)

        # ---- Ground truth = đường chạy naive gốc của repo
        naive = PennyLaneQuanv2d(args.circuit)
        t0 = time.perf_counter()
        gt = naive(x)
        print(f"  [GT] naive PennyLane trên {x.shape[0]} ảnh: {time.perf_counter() - t0:.1f}s", flush=True)

        exact_vec = ExactBatchedQuanv2d(args.circuit)
        variants = {
            "QLUT-4bit-nearest": QLUT_Quanv2d(tables[4], 4, args.circuit, "nearest"),
            "QLUT-4bit-interp": QLUT_Quanv2d(tables[4], 4, args.circuit, "interp"),
            "QLUT-5bit-nearest": QLUT_Quanv2d(tables[5], 5, args.circuit, "nearest"),
            "QLUT-5bit-interp": QLUT_Quanv2d(tables[5], 5, args.circuit, "interp"),
        }
        fid = {"ExactBatched(PennyLane vec)": fidelity(exact_vec(x), gt)}
        for k, m in variants.items():
            fid[k] = fidelity(m(x), gt)
        for k, v in fid.items():
            print(f"  [fidelity] {k:<28} MAE={v['MAE']:.5f} RMSE={v['RMSE']:.5f} "
                  f"nRMSE={v['nRMSE']:.4f} cos={v['Cosine']:.6f} max={v['MaxAbsErr']:.4f}")

        # ---- Latency (CPU)
        lat = {}
        for tag, nthreads in (("threads_default", default_threads), ("threads_1", 1)):
            torch.set_num_threads(nthreads)
            print(f"  [latency] torch threads = {nthreads}", flush=True)
            engines = {
                "PennyLane naive (repo)": (naive, "naive"),
                "PennyLane vectorized": (exact_vec, "torch"),
                "Classical Conv2d 2x2": (lambda t: conv(t), "torch"),
                "QLUT-4bit-nearest (torch)": (variants["QLUT-4bit-nearest"], "torch"),
                "QLUT-4bit-interp (torch)": (variants["QLUT-4bit-interp"], "torch"),
                "QLUT-5bit-nearest (torch)": (variants["QLUT-5bit-nearest"], "torch"),
                "QLUT-4bit-nearest (numpy)": (numpy_lut_nearest(tables[4].reshape(-1, 4).numpy(), 4), "numpy"),
            }
            if tag == "threads_1":  # naive & vectorized-PennyLane không phụ thuộc torch threads: bỏ để tiết kiệm thời gian
                for k in ("PennyLane naive (repo)", "PennyLane vectorized"):
                    engines.pop(k)
            with torch.no_grad():
                lat[tag] = bench_engines(engines, x, args.warmup, args.batch_repeats)
        torch.set_num_threads(default_threads)
        results["datasets"][ds] = {"fidelity": fid, "latency": lat}

    # ---- lưu + báo cáo
    with open(os.path.join(REPORT_DIR, "qlut_mve_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    md = render_markdown(results)
    with open(os.path.join(REPORT_DIR, "qlut_mve_results.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print("\n" + md)
    if not args.no_plot:
        plot(results)
    print(f"\nĐã lưu: reports/qlut_mve_results.json|md, reports/figures/qlut_fidelity_latency_benchmark.png")


def render_markdown(res: dict) -> str:
    L = [f"# Q-LUT MVE — kết quả đo thật ({res['meta']['timestamp']})", "",
         f"Circuit `{res['meta']['circuit']}`, {res['meta']['n_images']} ảnh test/dataset, "
         f"seed {res['meta']['seed']}, CPU: {res['meta']['cpu']}, torch {res['meta']['torch']} "
         f"(threads mặc định {res['meta']['torch_threads_default']}). Ground Truth = PennyLane naive của repo.", ""]
    for ds, d in res["datasets"].items():
        L += [f"## {ds} — Fidelity (so với Ground Truth)", "",
              "| Phương pháp | MAE | RMSE | nRMSE | Cosine | Max abs err | P99 abs err |", "|---|---|---|---|---|---|---|"]
        for k, v in d["fidelity"].items():
            L.append(f"| {k} | {v['MAE']:.5f} | {v['RMSE']:.5f} | {v['nRMSE']:.4f} | {v['Cosine']:.6f} | "
                     f"{v['MaxAbsErr']:.4f} | {v['P99AbsErr']:.4f} |")
        for tag, lat in d["latency"].items():
            L += ["", f"### {ds} — Latency CPU ({tag})", "",
                  "| Engine | b=1 ms/ảnh (median) | b=1 p95 | b=1 FPS | b=N ms/ảnh | b=N FPS |", "|---|---|---|---|---|---|"]
            for k, v in lat.items():
                L.append(f"| {k} | {v['b1']['ms_per_img_median']:.4f} | {v['b1']['ms_per_img_p95']:.4f} | "
                         f"{v['b1']['FPS_median']:.0f} | {v['bN']['ms_per_img_median']:.4f} | {v['bN']['FPS_median']:.0f} |")
            if "PennyLane naive (repo)" in lat:
                base = lat["PennyLane naive (repo)"]["b1"]["ms_per_img_median"]
                L += ["", "Speedup so với PennyLane naive (b=1, median): " + ", ".join(
                    f"{k}: x{base / v['b1']['ms_per_img_median']:.0f}" for k, v in lat.items() if k != "PennyLane naive (repo)")]
        L.append("")
    L += [f"Kích thước bảng: " + ", ".join(f"{k}={v:.2f} MiB" for k, v in res["meta"]["table_MiB"].items())]
    return "\n".join(L)


def plot(res: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dss = list(res["datasets"])
    fig, ax = plt.subplots(1, 3, figsize=(19, 5.6))
    fkeys = ["QLUT-4bit-nearest", "QLUT-4bit-interp", "QLUT-5bit-nearest", "QLUT-5bit-interp"]
    w = 0.8 / len(dss)
    for i, ds in enumerate(dss):
        f = res["datasets"][ds]["fidelity"]
        ax[0].bar(np.arange(len(fkeys)) + i * w, [f[k]["MAE"] for k in fkeys], w, label=ds)
        for j, k in enumerate(fkeys):
            ax[1].bar(j + i * w, 1 - f[k]["Cosine"], w, color=f"C{i}")
    for a, t in ((ax[0], "MAE vs PennyLane ground truth"), (ax[1], "1 - Cosine similarity")):
        a.set_xticks(np.arange(len(fkeys)) + w * (len(dss) - 1) / 2)
        a.set_xticklabels([k.replace("QLUT-", "") for k in fkeys], rotation=15)
        a.set_yscale("log"); a.set_title(t); a.grid(axis="y", alpha=.3)
    ax[0].legend()

    ds0 = dss[0]
    lat = res["datasets"][ds0]["latency"]["threads_default"]
    names = list(lat)
    vals = [lat[k]["b1"]["ms_per_img_median"] for k in names]
    colors = ["#c0392b" if "naive" in k or "vectorized" in k else "#7f8c8d" if "Conv2d" in k else "#27ae60" for k in names]
    ax[2].barh(names, vals, color=colors)
    ax[2].set_xscale("log"); ax[2].invert_yaxis()
    ax[2].set_xlabel("ms / ảnh (median, batch=1, CPU, log)")
    ax[2].set_title(f"Latency — {ds0}")
    for y, v in enumerate(vals):
        ax[2].text(v * 1.1, y, f"{v:.4f}", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "qlut_fidelity_latency_benchmark.png"), dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
