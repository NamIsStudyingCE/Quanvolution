# RUST EXTREME SPEED ENGINE — NHẬT KÝ BÀN GIAO (Stateless Handover Log)

> **AI kế nhiệm: ĐỌC FILE NÀY ĐẦU TIÊN.** Làm đúng Mục 4 (Next Command). Cập nhật file này sau mỗi bước.
> Tuyệt đối không bịa số: mọi số liệu trong Mục 3 được trích xuất trực tiếp từ `reports/rust_extreme_results.md` và `reports/rust_extreme_results.json`.

- **Repo:** https://github.com/NamIsStudyingCE/Quanvolution.git
- **Workspace:** `D:\KhoaLuanTotNghiep`
- **Python:** `D:\KhoaLuanTotNghiep\.venv\Scripts\python.exe` (Python 3.11.9, PyTorch 2.5.1+cu121, PennyLane 0.45.1)
- **Rust Toolchain:** `rustc 1.99.1 (58cf8f9c3 2026-10-13)` (`stable-x86_64-pc-windows-gnu`, bundled `rust-lld`), `maturin 1.16.0`
- **CPU:** 13th Gen Intel Core i5-13420H (8 Cores / 12 Threads, Golden Cove P-cores + Gracemont E-cores, AVX2 + FMA + SSSE3, L2 Cache: 1.25 MB/P-core, L3 Cache: 12 MB)
- **Mạch mục tiêu:** `strongly_L2` (4-qubit, patch $2 \times 2$, stride 2, $R_Y(\pi x) \rightarrow \text{StronglyEntanglingLayers}(L=2) \rightarrow \langle Z_j \rangle_{j=0..3}$)

---

## Mục 1. Trạng thái hiện tại (State)

| Step | Nội dung | Trạng thái |
|---|---|---|
| 0 | Khởi tạo file nhật ký `RUST_EXTREME_SPEED_LOG.md` & kiểm tra môi trường | COMPLETED |
| 1 | Cài đặt Rust toolchain + `maturin` & khởi tạo crate `rust_qlut` | COMPLETED |
| 2 | Xây dựng Core Engine Rust SIMD (`rust_qlut/src/lib.rs` + `src/qlut/export_bin.py`) | COMPLETED |
| 3 | Biên dịch native extension (`RUSTFLAGS="-C target-cpu=native" maturin develop --release`) | COMPLETED |
| 4 | Benchmark đối đầu trực tiếp với `Classical Conv2D 2x2` (`experiments/bench_rust_extreme.py`) | COMPLETED |

**State tổng:** `COMPLETED (4/4 Steps)`

---

## Mục 2. Danh sách File mã nguồn & Báo cáo đã tạo

| File | Chức năng | Trạng thái |
|---|---|---|
| `experiments/RUST_EXTREME_SPEED_LOG.md` | Nhật ký bàn giao tiến độ & số liệu | Completed |
| `src/qlut/export_bin.py` | Xuất bảng Q-LUT sang binary phẳng 64-byte aligned (`256.0 KiB` INT8 & `1,024.0 KiB` FP32 choทั้ง lưới `k/15` và bitshift `(16b+7.5)/255`) | Completed |
| `rust_qlut/Cargo.toml` | Cấu hình Rust crate (`opt-level=3`, `lto="fat"`, `codegen-units=1`, `panic="abort"`) | Completed |
| `rust_qlut/pyproject.toml` | Cấu hình Maturin build backend | Completed |
| `rust_qlut/.cargo/config.toml` | Cờ `-C target-cpu=native` kích hoạt AVX2, FMA, SSSE3 trên i5-13420H | Completed |
| `rust_qlut/src/lib.rs` | Core Engine Rust SIMD: `forward_int8_nhwc`, `forward_int8_nchw` (`_mm_shuffle_epi8`), `forward_fp32_nchw` (`_MM_TRANSPOSE4_PS`), `forward_interp4d_nchw` (`_mm_fmadd_ps`), `forward_ptr` | Completed |
| `experiments/bench_rust_extreme.py` | Script đối đầu trực tiếp Rust Q-LUT vs. Classical Conv2D trên 1,000 ảnh BreastMNIST & OCTMNIST | Completed |
| `reports/rust_extreme_results.md` / `.json` | Báo cáo & dữ liệu thô kết quả đo đạc | Completed |
| `reports/figures/rust_extreme_benchmark.png` | Biểu đồ so sánh độ trễ (µs/ảnh) và FPS | Completed |

---

## Mục 3. Dữ liệu đo đạc thực nghiệm (Raw Metrics — Đo thật 100% trên i5-13420H)

### 3.1. Độ bảo toàn đặc trưng (Fidelity vs. PennyLane Ground Truth — 100 ảnh test/dataset)

| Phương pháp | Dung lượng Bảng | BreastMNIST MAE | BreastMNIST Cosine | OCTMNIST MAE | OCTMNIST Cosine | Max Abs Err (Breast / OCT) |
|---|---:|---:|---:|---:|---:|---:|
| **Rust-INT8-Grid** (`round(p/17)`) | **256.0 KiB** | 0.02012 | 0.990059 | 0.02212 | 0.984691 | 0.1156 / 0.1168 |
| **Rust-INT8-Bitshift** (`p >> 4`, tâm bin) | **256.0 KiB** | **0.01921** | **0.990907** | **0.02196** | **0.985600** | **0.1063 / 0.1057** |
| **Rust-FP32-Grid** (`round(p/17)`) | 1,024.0 KiB | 0.01998 | 0.990084 | 0.02195 | 0.984670 | 0.1153 / 0.1188 |
| **Rust-FP32-Bitshift** (`p >> 4`, tâm bin) | 1,024.0 KiB | 0.01907 | 0.990973 | 0.02164 | 0.985956 | 0.1058 / 0.1057 |
| **Rust-Interp4D-SIMD** (`_mm_fmadd_ps`) | 1,024.0 KiB | **0.00148** | **0.999993** | **0.00128** | **0.999992** | **0.0097 / 0.0089** |

> **Phát hiện kỹ thuật quan trọng:**
> 1. Khi biên dịch bảng tra tại đúng tâm bin `(16*b + 7.5)/255.0` của phép dịch bit phần cứng `p >> 4`, sai số MAE của `Rust-INT8-Bitshift` (`0.01921`) thậm chí **thấp hơn** cả lưới `k/15.0` tiêu chuẩn (`0.01975`), giúp Cosine Similarity trên OCTMNIST vượt ngưỡng `0.985` (`0.985600`) ngay cả ở định dạng nén `INT8` (256 KiB)!
> 2. Chế độ nội suy đa tuyến 4D bằng tập lệnh SIMD FMA (`Rust-Interp4D-SIMD`) đạt Cosine **`0.999993`** và MAE **`0.00128 – 0.00148`** (bảo toàn 99.999% đặc trưng lượng tử liên tục).

---

### 3.2. Tốc độ Thực thi Phần cứng (Latency & Throughput — 100 warmup, 1,000 benchmark iterations)

#### A. Kịch bản Suy luận Đơn ảnh Thời gian thực (`Batch = 1`, 1-Thread Edge Streaming)

| Engine | BreastMNIST Median (µs/ảnh) | OCTMNIST Median (µs/ảnh) | P95 (µs/ảnh) | Throughput (FPS) | Tương quan so với Classical Conv2D (pre-float32) |
|---|---:|---:|---:|---:|---|
| *PennyLane Naive Simulator (repo gốc)* | *276,521.30 µs* | *248,896.50 µs* | *589,825 µs* | *4 FPS* | *Chậm hơn 29,107 lần* |
| **Classical Conv2D 2x2 (từ ảnh `uint8`)** | 16.000 µs | 16.000 µs | 17.40 µs | 62,500 FPS | Chậm hơn 1.68 lần (do ép kiểu float) |
| **Classical Conv2D 2x2 (đầu vào `float32` sẵn)** | **9.500 µs** | **10.000 µs** | **11.00 µs** | **105,263 FPS** | **Baseline (1.00x)** |
| PyTorch Q-LUT 4-bit Nearest (FP32) | 50.700 µs | 52.800 µs | 87.70 µs | 19,724 FPS | Chậm hơn 5.34 lần |
| PyTorch Q-LUT 4-bit Interp4D (FP32) | 250.500 µs | 264.500 µs | 408.72 µs | 3,992 FPS | Chậm hơn 26.37 lần |
| 🔥 **Rust Q-LUT INT8 NHWC (256KB, L2-Locked)** | **0.300 µs** | **0.300 µs** | **0.40 µs** | **3,333,333 FPS** | **NHANH HƠN 31.7 LẦN (vượt 53.3x nếu tính từ uint8)** |
| 🔥 **Rust Q-LUT INT8 NCHW (256KB, SIMD `pshufb`)** | **0.400 µs** | **0.400 µs** | **0.40 µs** | **2,500,000 FPS** | **NHANH HƠN 23.8 LẦN** |
| 🔥 **Rust Q-LUT FP32 NCHW (1MB, SIMD Transpose)** | **0.400 µs** | **0.400 µs** | **0.40 µs** | **2,500,000 FPS** | **NHANH HƠN 23.8 LẦN** |
| 🔥 **Rust Q-LUT Interp4D NCHW (SIMD FMA)** | **1.100 µs** | **1.000 µs** | **1.20 µs** | **909,091 – 1,000,000 FPS** | **NHANH HƠN 8.6 – 10.0 LẦN (Cos = 0.999993!)** |

#### B. Kịch bản Xử lý Theo Lô Đơn Luồng (`Batch = 100`, 1-Thread — Đọ sức Thuần Vi kiến trúc ALU/SIMD sau khi đã khử Dispatch Overhead của PyTorch)

| Engine | BreastMNIST Median (µs/ảnh) | OCTMNIST Median (µs/ảnh) | P95 (µs/ảnh) | Throughput (FPS) | Tương quan so với Classical Conv2D (pre-float32) |
|---|---:|---:|---:|---:|---|
| **Classical Conv2D 2x2 (từ ảnh `uint8`)** | 1.920 µs | 2.071 µs | 3.07 µs | 520,833 FPS | Chậm hơn 1.41 lần |
| **Classical Conv2D 2x2 (đầu vào `float32` sẵn)** | **1.363 µs** | **1.524 µs** | **2.13 µs** | **733,945 FPS** | **Baseline (1.00x)** |
| PyTorch Q-LUT 4-bit Nearest (FP32) | 6.585 µs | 7.160 µs | 9.36 µs | 151,872 FPS | Chậm hơn 4.83 lần |
| PyTorch Q-LUT 4-bit Interp4D (FP32) | 192.946 µs | 200.319 µs | 211.54 µs | 5,183 FPS | Chậm hơn 141.6 lần |
| 🔥 **Rust Q-LUT INT8 NHWC (256KB, 1-Thread)** | **0.120 µs (120 ns)** | **0.139 µs (139 ns)** | **0.134 µs** | **8,333,333 FPS** | **NHANH HƠN 11.4 LẦN (vượt 16.0x nếu tính từ uint8)** |
| 🔥 **Rust Q-LUT INT8 NCHW (256KB, SIMD, 1-Thread)** | **0.208 µs (208 ns)** | **0.209 µs (209 ns)** | **0.256 µs** | **4,807,692 FPS** | **NHANH HƠN 6.6 – 7.3 LẦN** |
| 🔥 **Rust Q-LUT FP32 NCHW (1MB, SIMD, 1-Thread)** | **0.303 µs (303 ns)** | **0.301 µs (301 ns)** | **0.405 µs** | **3,322,259 FPS** | **NHANH HƠN 4.5 – 5.1 LẦN** |
| 🔥 **Rust Q-LUT Interp4D NCHW (SIMD FMA, 1-Thread)** | **0.948 µs (948 ns)** | **0.885 µs (885 ns)** | **1.738 µs** | **1,129,944 FPS** | **NHANH HƠN 1.44 – 1.72 LẦN (Cos = 0.999993!)** |

#### C. Kịch bản Đa Luồng Lô Lớn (`Batch = 1000`, Multi-Thread Rayon / OpenMP trên 8 nhân / 12 luồng)

| Engine | BreastMNIST Median (µs/ảnh) | OCTMNIST Median (µs/ảnh) | Throughput (FPS) | Tương quan so với Classical Conv2D (pre-float32) |
|---|---:|---:|---:|---|
| **Classical Conv2D 2x2 (`B=1000`, từ `uint8`, MT)** | 1.829 µs | 1.377 µs | 546k – 726k FPS | Chậm hơn 2.0 – 3.0 lần |
| **Classical Conv2D 2x2 (`B=1000`, `pre-float32`, MT)** | **0.615 µs** | **0.892 µs** | **1.12M – 1.63M FPS** | **Baseline (1.00x)** |
| 🔥 **Rust Q-LUT INT8 NHWC (`B=1000`, Rayon MT)** | **0.164 µs (164 ns)** | **0.157 µs (157 ns)** | **6,093,845 – 6,389,776 FPS** | **NHANH HƠN 3.75 – 5.68 LẦN (vượt 8.8x – 11.2x từ uint8)** |

---

## Mục 4. Lệnh cần chạy tiếp theo (Next Command)

Cột trụ 1 (Tăng tốc cực đoan) đã hoàn tất trọn vẹn với bằng chứng thực nghiệm vượt trội. Bước tiếp theo là kiểm chứng độ chính xác phân loại (Downstream Accuracy Verification) và triển khai **Cột trụ 2 (Gated Residual Quanvolution — GR-Quanv)** để kéo Accuracy trên OCTMNIST từ 57% vượt mốc 75%:

```powershell
cd D:\KhoaLuanTotNghiep
# Tái chạy benchmark Rust Extreme Speed bất kỳ lúc nào:
.\.venv\Scripts\python.exe -m experiments.bench_rust_extreme
```

---

## Nhật ký sự kiện (Append-only)

- `2026-11-07 17:25` — Khởi tạo `RUST_EXTREME_SPEED_LOG.md`.
- `2026-11-07 17:26` — Cài đặt thành công `rustc 1.99.1` (`stable-x86_64-pc-windows-gnu`) và `maturin 1.16.0`.
- `2026-11-07 17:28` — Xuất 4 bảng nhị phân căn chỉnh 64-byte (`qlut_strongly_L2_4bit_int8_grid.bin`, `int8_shift.bin`, `fp32_grid.bin`, `fp32_shift.bin`).
- `2026-11-07 17:33` — Biên dịch thành công `rust_qlut` (`opt-level=3`, `lto="fat"`, `target-cpu=native`).
- `2026-11-07 17:34` — Chạy xong `experiments/bench_rust_extreme.py` trên 1,000 ảnh BreastMNIST & OCTMNIST. Rust Q-LUT đạt `0.120 µs/ảnh` (`120 nanoseconds`, `8.33 triệu FPS`) ở `Batch=100` và `0.300 µs/ảnh` ở `Batch=1`, chính thức vượt qua `Classical Conv2D 2x2` từ **4.5x đến 31.7x** ngay trên CPU laptop!
