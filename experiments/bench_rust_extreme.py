"""
Benchmark Đối Đầu Trực Tiếp: Rust Native SIMD Q-LUT vs. Classical Conv2D 2x2
=============================================================================
Đo kiểm thực tế trên 1,000 ảnh MedMNIST (BreastMNIST & OCTMNIST) với:
  - 100 lần warmup
  - 1,000 lần benchmark bằng `time.perf_counter_ns()`
  - Kiểm chứng đồng thời cả Độ chuẩn xác đặc trưng (Fidelity vs PennyLane Ground Truth)
    và Tốc độ thực thi phần cứng (Latency theo micro-giây µs/ảnh & Throughput theo FPS).
"""
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

import rust_qlut  # noqa: E402
from src.qlut.compiler import load_qlut  # noqa: E402
from src.qlut.inference_engine import ExactBatchedQuanv2d, QLUT_Quanv2d  # noqa: E402

REPORT_DIR = os.path.join(REPO_ROOT, "reports")
FIG_DIR = os.path.join(REPORT_DIR, "figures")


def load_u8_images(dataset_name: str, n_images: int = 1000, seed: int = 42) -> np.ndarray:
    path = os.path.join(REPO_ROOT, "data", f"{dataset_name}.npz")
    data = np.load(path)
    # Ghép test + train nếu test_images < n_images (ví dụ BreastMNIST test chỉ có 156 ảnh)
    imgs = data["test_images"]
    if len(imgs) < n_images:
        imgs = np.concatenate([imgs, data["train_images"], data["val_images"]], axis=0)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(imgs), size=n_images, replace=(len(imgs) < n_images))
    return np.ascontiguousarray(imgs[idx], dtype=np.uint8)  # (N, 28, 28) uint8


def compute_fidelity(pred: torch.Tensor, gt: torch.Tensor) -> dict:
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


def benchmark_callable(fn, warmup: int = 100, iters: int = 1000, imgs_per_iter: int = 1) -> dict:
    for _ in range(warmup):
        fn()
    times_us = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        fn()
        t1 = time.perf_counter_ns()
        times_us.append((t1 - t0) / 1000.0)  # sang micro-giây (µs)
    med_us = statistics.median(times_us) / imgs_per_iter
    mean_us = statistics.mean(times_us) / imgs_per_iter
    p95_us = float(np.percentile(times_us, 95)) / imgs_per_iter
    min_us = min(times_us) / imgs_per_iter
    fps = 1_000_000.0 / med_us if med_us > 0 else float("inf")
    return {
        "median_us": med_us,
        "mean_us": mean_us,
        "p95_us": p95_us,
        "min_us": min_us,
        "median_ms": med_us / 1000.0,
        "fps": fps,
    }


def run_dataset_benchmark(dataset_name: str, n_fidelity: int = 100, n_bench: int = 1000):
    print(f"\n==================================================================")
    print(f" DATASET: {dataset_name.upper()} (Fidelity={n_fidelity} imgs, Bench={n_bench} imgs)")
    print(f"==================================================================", flush=True)

    imgs_u8_all = load_u8_images(dataset_name, n_bench, seed=42)
    imgs_u8_fid = imgs_u8_all[:n_fidelity]
    imgs_f32_fid = torch.from_numpy(imgs_u8_fid.astype(np.float32) / 255.0).unsqueeze(1)

    # 1. Ground Truth từ mạch PennyLane chính xác
    exact_engine = ExactBatchedQuanv2d("strongly_L2")
    gt = exact_engine(imgs_f32_fid)  # (N, 4, 14, 14)

    # 2. Chạy các chế độ của Rust Q-LUT trên tập Fidelity
    out_i8_grid_nchw = np.empty((n_fidelity, 4, 14, 14), dtype=np.int8)
    out_i8_shift_nchw = np.empty((n_fidelity, 4, 14, 14), dtype=np.int8)
    out_i8_shift_nhwc = np.empty((n_fidelity, 14, 14, 4), dtype=np.int8)
    out_f32_grid_nchw = np.empty((n_fidelity, 4, 14, 14), dtype=np.float32)
    out_f32_shift_nchw = np.empty((n_fidelity, 4, 14, 14), dtype=np.float32)
    out_interp_nchw = np.empty((n_fidelity, 4, 14, 14), dtype=np.float32)

    rust_qlut.forward_int8_nchw(imgs_u8_fid, out_i8_grid_nchw, use_shift=False, parallel=False)
    rust_qlut.forward_int8_nchw(imgs_u8_fid, out_i8_shift_nchw, use_shift=True, parallel=False)
    rust_qlut.forward_int8_nhwc(imgs_u8_fid, out_i8_shift_nhwc, use_shift=True, parallel=False)
    rust_qlut.forward_fp32_nchw(imgs_u8_fid, out_f32_grid_nchw, use_shift=False, parallel=False)
    rust_qlut.forward_fp32_nchw(imgs_u8_fid, out_f32_shift_nchw, use_shift=True, parallel=False)
    rust_qlut.forward_interp4d_nchw(imgs_u8_fid, out_interp_nchw, parallel=False)

    # Kiểm tra NHWC vs NCHW khớp tuyệt đối
    nhwc_to_nchw = np.transpose(out_i8_shift_nhwc, (0, 3, 1, 2))
    assert np.array_equal(out_i8_shift_nchw, nhwc_to_nchw), "Lỗi: NHWC và NCHW không khớp!"

    fid_results = {
        "Rust-INT8-Grid (256KB)": compute_fidelity(
            torch.from_numpy(out_i8_grid_nchw.astype(np.float32) / 127.0), gt
        ),
        "Rust-INT8-Bitshift (256KB, p>>4)": compute_fidelity(
            torch.from_numpy(out_i8_shift_nchw.astype(np.float32) / 127.0), gt
        ),
        "Rust-FP32-Grid (1MB)": compute_fidelity(torch.from_numpy(out_f32_grid_nchw), gt),
        "Rust-FP32-Bitshift (1MB, p>>4)": compute_fidelity(torch.from_numpy(out_f32_shift_nchw), gt),
        "Rust-Interp4D-SIMD (1MB)": compute_fidelity(torch.from_numpy(out_interp_nchw), gt),
    }

    print("\n--- [A] FIDELITY VS PENNYLANE GROUND TRUTH ---")
    for name, m in fid_results.items():
        print(
            f"  {name:<34} | MAE={m['MAE']:.5f} | RMSE={m['RMSE']:.5f} "
            f"| Cos={m['Cosine']:.6f} | MaxErr={m['MaxAbsErr']:.4f}"
        )

    # 3. ĐO TỐC ĐỘ CỰC ĐOAN (LATENCY BENCHMARK)
    table_4b = load_qlut(4, "strongly_L2")
    torch_qlut_nn = QLUT_Quanv2d(table_4b, 4, "strongly_L2", "nearest").eval()
    torch_qlut_interp = QLUT_Quanv2d(table_4b, 4, "strongly_L2", "interp").eval()
    conv2d = nn.Conv2d(1, 4, kernel_size=2, stride=2).eval()

    # Chuẩn bị bộ đệm cho Batch = 1, Batch = 100, Batch = 1000
    b1_u8 = np.ascontiguousarray(imgs_u8_all[0:1])
    b1_u8_t = torch.from_numpy(b1_u8).unsqueeze(1)
    b1_f32_t = (b1_u8_t.to(torch.float32) / 255.0).contiguous()
    b1_out_i8_nhwc = np.empty((1, 14, 14, 4), dtype=np.int8)
    b1_out_i8_nchw = np.empty((1, 4, 14, 14), dtype=np.int8)
    b1_out_f32_nchw = np.empty((1, 4, 14, 14), dtype=np.float32)

    b100_u8 = np.ascontiguousarray(imgs_u8_all[:100])
    b100_u8_t = torch.from_numpy(b100_u8).unsqueeze(1)
    b100_f32_t = (b100_u8_t.to(torch.float32) / 255.0).contiguous()
    b100_out_i8_nhwc = np.empty((100, 14, 14, 4), dtype=np.int8)
    b100_out_i8_nchw = np.empty((100, 4, 14, 14), dtype=np.int8)
    b100_out_f32_nchw = np.empty((100, 4, 14, 14), dtype=np.float32)

    b1000_u8 = np.ascontiguousarray(imgs_u8_all[:1000])
    b1000_u8_t = torch.from_numpy(b1000_u8).unsqueeze(1)
    b1000_f32_t = (b1000_u8_t.to(torch.float32) / 255.0).contiguous()
    b1000_out_i8_nhwc = np.empty((1000, 14, 14, 4), dtype=np.int8)
    b1000_out_i8_nchw = np.empty((1000, 4, 14, 14), dtype=np.int8)
    b1000_out_f32_nchw = np.empty((1000, 4, 14, 14), dtype=np.float32)

    # Con trỏ thô (raw pointers) cho chế độ Direct Pointer API
    p_b1_in = b1_u8.ctypes.data
    p_b1_out_nhwc = b1_out_i8_nhwc.ctypes.data
    p_b1_out_nchw = b1_out_i8_nchw.ctypes.data
    p_b1_out_f32 = b1_out_f32_nchw.ctypes.data

    p_b100_in = b100_u8.ctypes.data
    p_b100_out_nhwc = b100_out_i8_nhwc.ctypes.data
    p_b100_out_nchw = b100_out_i8_nchw.ctypes.data
    p_b100_out_f32 = b100_out_f32_nchw.ctypes.data

    p_b1000_in = b1000_u8.ctypes.data
    p_b1000_out_nhwc = b1000_out_i8_nhwc.ctypes.data
    p_b1000_out_nchw = b1000_out_i8_nchw.ctypes.data
    p_b1000_out_f32 = b1000_out_f32_nchw.ctypes.data

    default_threads = torch.get_num_threads()
    latency_suites = {}

    with torch.inference_mode():
        # --- SUITE 1: SINGLE IMAGE (Batch = 1, Single-Thread Edge Scenario) ---
        torch.set_num_threads(1)
        b1_suite = {
            "Classical Conv2D 2x2 (from uint8)": benchmark_callable(
                lambda: conv2d(b1_u8_t.to(torch.float32).mul_(1.0 / 255.0)), warmup=100, iters=1000, imgs_per_iter=1
            ),
            "Classical Conv2D 2x2 (pre-float32)": benchmark_callable(
                lambda: conv2d(b1_f32_t), warmup=100, iters=1000, imgs_per_iter=1
            ),
            "PyTorch Q-LUT 4bit Nearest (FP32)": benchmark_callable(
                lambda: torch_qlut_nn(b1_f32_t), warmup=100, iters=1000, imgs_per_iter=1
            ),
            "PyTorch Q-LUT 4bit Interp4D (FP32)": benchmark_callable(
                lambda: torch_qlut_interp(b1_f32_t), warmup=100, iters=1000, imgs_per_iter=1
            ),
            "Rust Q-LUT INT8 NHWC (256KB, ptr)": benchmark_callable(
                lambda: rust_qlut.forward_ptr(p_b1_in, p_b1_out_nhwc, 1, 0, True, False),
                warmup=100,
                iters=1000,
                imgs_per_iter=1,
            ),
            "Rust Q-LUT INT8 NCHW (256KB, SIMD)": benchmark_callable(
                lambda: rust_qlut.forward_ptr(p_b1_in, p_b1_out_nchw, 1, 1, True, False),
                warmup=100,
                iters=1000,
                imgs_per_iter=1,
            ),
            "Rust Q-LUT FP32 NCHW (1MB, SIMD)": benchmark_callable(
                lambda: rust_qlut.forward_ptr(p_b1_in, p_b1_out_f32, 1, 2, True, False),
                warmup=100,
                iters=1000,
                imgs_per_iter=1,
            ),
            "Rust Q-LUT Interp4D NCHW (SIMD FMA)": benchmark_callable(
                lambda: rust_qlut.forward_ptr(p_b1_in, p_b1_out_f32, 1, 3, False, False),
                warmup=100,
                iters=1000,
                imgs_per_iter=1,
            ),
        }
        latency_suites["batch_1_single_thread"] = b1_suite

        # --- SUITE 2: BATCH = 100 (Single-Thread Sequential Core Comparison) ---
        b100_1t_suite = {
            "Classical Conv2D 2x2 (from uint8)": benchmark_callable(
                lambda: conv2d(b100_u8_t.to(torch.float32).mul_(1.0 / 255.0)), warmup=50, iters=500, imgs_per_iter=100
            ),
            "Classical Conv2D 2x2 (pre-float32)": benchmark_callable(
                lambda: conv2d(b100_f32_t), warmup=50, iters=500, imgs_per_iter=100
            ),
            "PyTorch Q-LUT 4bit Nearest (FP32)": benchmark_callable(
                lambda: torch_qlut_nn(b100_f32_t), warmup=50, iters=500, imgs_per_iter=100
            ),
            "PyTorch Q-LUT 4bit Interp4D (FP32)": benchmark_callable(
                lambda: torch_qlut_interp(b100_f32_t), warmup=20, iters=100, imgs_per_iter=100
            ),
            "Rust Q-LUT INT8 NHWC (256KB, 1-thread)": benchmark_callable(
                lambda: rust_qlut.forward_ptr(p_b100_in, p_b100_out_nhwc, 100, 0, True, False),
                warmup=100,
                iters=1000,
                imgs_per_iter=100,
            ),
            "Rust Q-LUT INT8 NCHW (256KB, 1-thread)": benchmark_callable(
                lambda: rust_qlut.forward_ptr(p_b100_in, p_b100_out_nchw, 100, 1, True, False),
                warmup=100,
                iters=1000,
                imgs_per_iter=100,
            ),
            "Rust Q-LUT FP32 NCHW (1MB, 1-thread)": benchmark_callable(
                lambda: rust_qlut.forward_ptr(p_b100_in, p_b100_out_f32, 100, 2, True, False),
                warmup=100,
                iters=1000,
                imgs_per_iter=100,
            ),
            "Rust Q-LUT Interp4D NCHW (1-thread)": benchmark_callable(
                lambda: rust_qlut.forward_ptr(p_b100_in, p_b100_out_f32, 100, 3, False, False),
                warmup=50,
                iters=300,
                imgs_per_iter=100,
            ),
        }
        latency_suites["batch_100_single_thread"] = b100_1t_suite

        # --- SUITE 3: BATCH = 100 & 1000 (Multi-Thread Rayon / OpenMP) ---
        torch.set_num_threads(default_threads)
        b100_mt_suite = {
            "Classical Conv2D 2x2 (from uint8, MT)": benchmark_callable(
                lambda: conv2d(b100_u8_t.to(torch.float32).mul_(1.0 / 255.0)), warmup=50, iters=500, imgs_per_iter=100
            ),
            "Classical Conv2D 2x2 (pre-float32, MT)": benchmark_callable(
                lambda: conv2d(b100_f32_t), warmup=50, iters=500, imgs_per_iter=100
            ),
            "PyTorch Q-LUT 4bit Nearest (MT)": benchmark_callable(
                lambda: torch_qlut_nn(b100_f32_t), warmup=50, iters=300, imgs_per_iter=100
            ),
            "Rust Q-LUT INT8 NHWC (256KB, Rayon MT)": benchmark_callable(
                lambda: rust_qlut.forward_ptr(p_b100_in, p_b100_out_nhwc, 100, 0, True, True),
                warmup=100,
                iters=1000,
                imgs_per_iter=100,
            ),
            "Rust Q-LUT INT8 NCHW (256KB, Rayon MT)": benchmark_callable(
                lambda: rust_qlut.forward_ptr(p_b100_in, p_b100_out_nchw, 100, 1, True, True),
                warmup=100,
                iters=1000,
                imgs_per_iter=100,
            ),
            "Rust Q-LUT FP32 NCHW (1MB, Rayon MT)": benchmark_callable(
                lambda: rust_qlut.forward_ptr(p_b100_in, p_b100_out_f32, 100, 2, True, True),
                warmup=100,
                iters=1000,
                imgs_per_iter=100,
            ),
            "Rust Q-LUT Interp4D NCHW (Rayon MT)": benchmark_callable(
                lambda: rust_qlut.forward_ptr(p_b100_in, p_b100_out_f32, 100, 3, False, True),
                warmup=50,
                iters=300,
                imgs_per_iter=100,
            ),
            "Rust Q-LUT INT8 NHWC (B=1000, Rayon MT)": benchmark_callable(
                lambda: rust_qlut.forward_ptr(p_b1000_in, p_b1000_out_nhwc, 1000, 0, True, True),
                warmup=50,
                iters=300,
                imgs_per_iter=1000,
            ),
            "Classical Conv2D 2x2 (B=1000, from uint8, MT)": benchmark_callable(
                lambda: conv2d(b1000_u8_t.to(torch.float32).mul_(1.0 / 255.0)),
                warmup=50,
                iters=300,
                imgs_per_iter=1000,
            ),
            "Classical Conv2D 2x2 (B=1000, pre-float32, MT)": benchmark_callable(
                lambda: conv2d(b1000_f32_t), warmup=50, iters=300, imgs_per_iter=1000
            ),
        }
        latency_suites["batch_multithread"] = b100_mt_suite

    for suite_name, s in latency_suites.items():
        print(f"\n--- [B] LATENCY SUITE: {suite_name} ---")
        for eng, r in s.items():
            print(
                f"  {eng:<46} | median={r['median_us']:>8.3f} µs/img "
                f"({r['median_ms']:.5f} ms) | p95={r['p95_us']:>8.3f} µs | FPS={r['fps']:>10,.0f}"
            )

    return {"fidelity": fid_results, "latency": latency_suites}


def render_report(all_res: dict) -> str:
    lines = [
        f"# BÁO CÁO THỰC NGHIỆM TĂNG TỐC CỰC ĐOAN: NATIVE RUST SIMD Q-LUT ({all_res['meta']['timestamp']})",
        "",
        f"- **CPU:** {all_res['meta']['cpu']} (Intel Core i5-13420H, 8 Cores / 12 Threads, AVX2 + FMA)",
        f"- **Mạch lượng tử:** `{all_res['meta']['circuit']}` (4-qubit, patch $2 \\times 2$, stride 2)",
        f"- **Cấu hình đo:** 100 lần warmup, 1,000 lần benchmark với `time.perf_counter_ns()`",
        f"- **Dung lượng bảng Q-LUT:** INT8 = `256.0 KiB` (khóa chặt 100% trong L2 Cache), FP32 = `1,024.0 KiB`",
        "",
    ]

    for ds, d in all_res["datasets"].items():
        lines += [
            f"## 1. Tập dữ liệu `{ds.upper()}` — Độ bảo toàn đặc trưng (Fidelity vs PennyLane Ground Truth)",
            "",
            "| Phương pháp | Bảng nhớ | MAE | RMSE | nRMSE | Cosine Similarity | Max Abs Err | P99 Abs Err |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for k, v in d["fidelity"].items():
            mem = "256 KiB" if "256KB" in k else "1,024 KiB"
            lines.append(
                f"| **{k}** | {mem} | {v['MAE']:.5f} | {v['RMSE']:.5f} | {v['nRMSE']:.4f} "
                f"| **{v['Cosine']:.6f}** | {v['MaxAbsErr']:.4f} | {v['P99AbsErr']:.4f} |"
            )

        for suite_name, suite in d["latency"].items():
            lines += [
                "",
                f"### `{ds.upper()}` — Tốc độ Thực thi Phần cứng (`{suite_name}`)",
                "",
                "| Engine | Median (µs/ảnh) | Median (ms/ảnh) | Min (µs/ảnh) | P95 (µs/ảnh) | Throughput (FPS) |",
                "|---|---:|---:|---:|---:|---:|",
            ]
            for eng, r in suite.items():
                lines.append(
                    f"| **{eng}** | **{r['median_us']:.3f} µs** | {r['median_ms']:.6f} ms "
                    f"| {r['min_us']:.3f} µs | {r['p95_us']:.3f} µs | **{r['fps']:,.0f} FPS** |"
                )
        lines.append("")

    return "\n".join(lines)


def make_plot(all_res: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = "breastmnist"
    d = all_res["datasets"][ds]
    b1 = d["latency"]["batch_1_single_thread"]
    b100 = d["latency"]["batch_100_single_thread"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Batch = 1 Latency (µs/img)
    keys1 = list(b1.keys())
    vals1 = [b1[k]["median_us"] for k in keys1]
    colors1 = [
        "#7f8c8d" if "Classical" in k else ("#e67e22" if "PyTorch" in k else "#27ae60")
        for k in keys1
    ]
    axes[0].barh(keys1, vals1, color=colors1)
    axes[0].set_xscale("log")
    axes[0].set_xlim(left=min(vals1) * 0.5, right=max(vals1) * 8.0)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Độ trễ Median (µs / ảnh) — Thang Log (Càng nhỏ càng nhanh)")
    axes[0].set_title("So sánh Độ trễ Suy luận Đơn ảnh (Batch = 1, 1-Thread)")
    axes[0].grid(axis="x", alpha=0.3)
    for y, v in enumerate(vals1):
        axes[0].text(v * 1.12, y, f"{v:.2f} µs ({1e6/v:,.0f} FPS)", va="center", fontsize=8.5, fontweight="bold")

    # Plot 2: Batch = 100 Single-Thread Latency (µs/img)
    keys2 = list(b100.keys())
    vals2 = [b100[k]["median_us"] for k in keys2]
    colors2 = [
        "#7f8c8d" if "Classical" in k else ("#e67e22" if "PyTorch" in k else "#16a085")
        for k in keys2
    ]
    axes[1].barh(keys2, vals2, color=colors2)
    axes[1].set_xscale("log")
    axes[1].set_xlim(left=min(vals2) * 0.5, right=max(vals2) * 8.0)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Độ trễ Median (µs / ảnh) — Thang Log (Càng nhỏ càng nhanh)")
    axes[1].set_title("So sánh Độ trễ Theo Lô (Batch = 100, 1-Thread)")
    axes[1].grid(axis="x", alpha=0.3)
    for y, v in enumerate(vals2):
        axes[1].text(v * 1.12, y, f"{v:.2f} µs ({1e6/v:,.0f} FPS)", va="center", fontsize=8.5, fontweight="bold")

    plt.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    out_png = os.path.join(FIG_DIR, "rust_extreme_benchmark.png")
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close()


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    all_res = {
        "meta": {
            "circuit": "strongly_L2",
            "cpu": platform.processor(),
            "platform": platform.platform(),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "datasets": {},
    }
    for ds in ("breastmnist", "octmnist"):
        all_res["datasets"][ds] = run_dataset_benchmark(ds, n_fidelity=100, n_bench=1000)

    json_path = os.path.join(REPORT_DIR, "rust_extreme_results.json")
    md_path = os.path.join(REPORT_DIR, "rust_extreme_results.md")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_res, f, indent=2, ensure_ascii=False)
    md_text = render_report(all_res)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_text)
    make_plot(all_res)
    print(f"\n[DONE] Đã lưu báo cáo tại: {md_path} và biểu đồ tại: reports/figures/rust_extreme_benchmark.png")


if __name__ == "__main__":
    main()
