# Q-LUT MVE — NHẬT KÝ BÀN GIAO (Stateless Handover Log)

> **AI kế nhiệm: ĐỌC FILE NÀY ĐẦU TIÊN.** Làm đúng Mục 4 (Next Command). Cập nhật file này sau mỗi bước.
> Không bịa số: mọi số trong Mục 3 được copy từ `reports/qlut_mve_results.md|json` (output thật).

Repo: https://github.com/NamIsStudyingCE/Quanvolution.git — Workspace: `D:\KhoaLuanTotNghiep`
Python: `D:\KhoaLuanTotNghiep\.venv\Scripts\python.exe` (torch 2.5.1+cu121, pennylane 0.45.1, medmnist 3.0.2)
Máy: Intel i5-13420H (8 core / 12 thread), Windows. torch threads mặc định = 8.
Mạch gốc: `src/models/circuits.py` (patch 2x2, stride 2, RY(pi*x) -> ansatz -> <Z_j> j=0..3).
Mạch dùng cho MVE: **`strongly_L2`** (Fixed Strongly L2 — "quán quân PR-AUC" theo README).
Lưu ý: `*.pt` bị `.gitignore` -> weights/qlut_*.pt KHÔNG được push; tái tạo bằng compiler (5-bit ~9 s).
Quy ước chỉ số: index = (p0<<3n)|(p1<<2n)|(p2<<n)|p3; p = round(x*(L-1)), L=2^n; pixel row-major trong patch 2x2.
Tên file weights: `weights/qlut_{circuit}_{n}bit.pt`.

## Mục 1. Trạng thái hiện tại (State)

| Step | Nội dung | Trạng thái |
|---|---|---|
| 0 | Khảo sát repo + tạo log | COMPLETED |
| 1 | `src/qlut/compiler.py` | COMPLETED |
| 2 | `src/qlut/inference_engine.py` | COMPLETED |
| 3 | `experiments/sanity_check_qlut.py` (fidelity + latency) | COMPLETED (chạy full 100 ảnh x 2 dataset) |
| 4 | Tổng hợp kết luận | COMPLETED (Mục 5) |

**State tổng:** `MVE_COMPLETED` — chưa commit lên git.

## Mục 2. File đã tạo

| File | Chức năng |
|---|---|
| `experiments/Q_LUT_MVE_LOG.md` | Nhật ký này |
| `src/qlut/__init__.py` | package marker |
| `src/qlut/compiler.py` | `compile_qlut`, `compile_qlut_from_fn` (hỗ trợ mạch trainable đóng băng), `load_qlut`, `verify_against_reference`; CLI |
| `src/qlut/inference_engine.py` | `QLUT_Quanv2d` (nearest/interp, không vòng lặp), `ExactBatchedQuanv2d` (PennyLane vector hoá), `PennyLaneQuanv2d` (naive của repo) |
| `experiments/sanity_check_qlut.py` | Đo fidelity + latency, xuất md/json/png |
| `reports/qlut_mve_results.md` / `.json` | Kết quả thô |
| `reports/figures/qlut_fidelity_latency_benchmark.png` | Biểu đồ |
| `weights/qlut_strongly_L2_{4,5}bit.pt` | Bảng đã biên dịch (gitignored) |

## Mục 3. Dữ liệu đo đạc thực nghiệm (Raw Metrics) — 2026-11-07, 100 ảnh test/dataset, seed 42

Ground Truth = `PennyLaneQuanv2d` (đường naive gốc của repo). Compile: 4-bit 0.55 s / 1.000 MiB; 5-bit 8.71 s / 16.000 MiB. Bảng ≡ QNode gốc tại điểm lưới: max|diff| ≈ 2.6e-8.

### 3.1 Fidelity

| Phương pháp | BreastMNIST MAE | BreastMNIST Cos | OCTMNIST MAE | OCTMNIST Cos | nRMSE (Breast/OCT) |
|---|---|---|---|---|---|
| ExactBatched (PennyLane vec) | 0 | 1.000000 | 0 | 1.000000 | 0 / 0 |
| QLUT-4bit-nearest | 0.01975 | 0.990519 | 0.02156 | 0.984833 | 0.137 / 0.174 |
| QLUT-4bit-interp | 0.00148 | 0.999993 | 0.00128 | 0.999992 | 0.0102 / 0.0102 |
| QLUT-5bit-nearest | 0.00961 | 0.997763 | 0.01040 | 0.996608 | 0.066 / 0.083 |
| QLUT-5bit-interp | 0.00035 | 1.000000 | 0.00030 | 1.000000 | 0.0024 / 0.0024 |

Max abs err: 4b-NN 0.110/0.114; 4b-interp 0.0088/0.0083; 5b-NN 0.059/0.056; 5b-interp 0.0021/0.0021.

### 3.2 Latency CPU (median, ms/ảnh) — BreastMNIST (OCTMNIST gần như giống hệt, xem file md)

| Engine | b=1 (8 threads) | b=100 (8 threads) | b=1 (1 thread) | b=100 (1 thread) |
|---|---|---|---|---|
| PennyLane naive (repo) | 276.52 | 276.52 | – | – |
| PennyLane vectorized | 4.239 | 1.566 | – | – |
| Classical Conv2d 2x2 (4 filter) | 0.0548 | 0.0018 | 0.0120 | 0.0017 |
| QLUT-4bit-nearest (torch) | 0.1112 | 0.0416 | 0.0566 | 0.0071 |
| QLUT-4bit-interp (torch) | 0.3274 | 0.1338 | 0.2597 | 0.1974 |
| QLUT-5bit-nearest (torch) | 0.1495 | 0.0370 | 0.0598 | 0.0074 |
| QLUT-4bit-nearest (numpy) | 0.0149 | 0.0074 | 0.0149 | 0.0058 |

Speedup so với naive (b=1): torch-4b-NN x2,487 (OCT x2,527); numpy-4b-NN x18,558 (OCT x16,593); torch-4b-interp x845 (OCT x814); PennyLane vectorized x65.

## Mục 4. Next Command

Bước kế tiếp (chưa làm): kiểm tra tác động lên ĐỘ CHÍNH XÁC PHÂN LOẠI — huấn luyện cùng bộ phân loại (BN-ReLU-Linear) trên đặc trưng exact vs Q-LUT (4b-NN, 4b-interp) cho BreastMNIST/OCTMNIST, vài seed. Cần viết `experiments/qlut_accuracy_check.py` (dùng `ExactBatchedQuanv2d` để tính đặc trưng exact cho toàn bộ train set, nhanh hơn precompute cũ).

```powershell
cd D:\KhoaLuanTotNghiep
.\.venv\Scripts\python.exe -m experiments.sanity_check_qlut --datasets breastmnist octmnist --n-images 100   # tái chạy MVE
```

## Mục 5. Kết luận khách quan (đối chiếu kỳ vọng ban đầu)

| Kỳ vọng ban đầu | Kết quả thật | Đánh giá |
|---|---|---|
| Cos 4-bit NN > 0.985 | 0.9905 (Breast) / **0.9848 (OCT)** | Breast đạt; OCT trượt 0.0002 — ở rìa |
| Cos 4-bit interp > 0.998 | 0.99999 | **ĐẠT** |
| MAE 4-bit < 0.04 | 0.020–0.022 (NN), 0.0013–0.0015 (interp) | **ĐẠT** |
| 5-bit NN: Cos > 0.999, MAE < 0.008 | Cos 0.9966–0.9978, MAE 0.0096–0.0104 | **KHÔNG ĐẠT** (NN 5-bit không đủ; phải dùng interp) |
| Latency < 0.1 ms/ảnh | torch-NN b=1: 0.099–0.111 (rìa); b=100: 0.033–0.042; numpy: 0.015 | Đạt ở batch/numpy; interp (0.13–0.33) **không đạt** |
| Nhanh hơn simulator > 2,000 lần | x2,487–2,527 (torch), x16.6k–18.6k (numpy) | **ĐẠT** |
| **"Nhanh ngang/hơn Conv2D cổ điển"** | Conv2d nhanh hơn: b=100 ~0.0017 vs 0.007–0.04 ms (4–25 lần); b=1 1-thread 0.012 vs 0.057 | **KHÔNG ĐẠT — claim này SAI trên CPU đa dụng**, cần bỏ khỏi báo cáo |
| Bảng nằm trong cache | 1 MiB / 16 MiB (chưa đo cache-miss trực tiếp) | Kích thước đúng; chưa chứng minh bằng đo |

**Thông điệp trung thực rút ra:**
1. Q-LUT thu hẹp khoảng cách với simulator từ ~700x (so CNN) xuống cùng bậc độ lớn với Conv2d (chậm hơn ~4–25x ở batch, ~5x ở b=1), nhanh hơn simulator 2.5k–18k lần. Đây là kết quả mạnh nhưng KHÔNG phải "nhanh hơn CNN".
2. Fidelity: 4-bit + nội suy (cos 0.99999, MAE ~0.0014) là lựa chọn bảo toàn tốt; NN 4-bit sai ~2% và OCT có cos thấp hơn. Nội suy tốn ~3–5x thời gian so với NN trong bản torch hiện tại (chưa tối ưu).
3. Phần overhead lớn của torch ở batch nhỏ (numpy NN nhanh hơn torch NN ~4–7x ở b=1). Để chứng minh LUT có thể nhanh hơn MAC thật sự cần bản C/SIMD (không có FLOP nhân) — chưa làm, đừng claim trước khi đo.
4. MVE mới chỉ chứng minh FIDELITY + SPEED của bước trích đặc trưng. CHƯA chứng minh độ chính xác phân loại không đổi, và CHƯA đề cập GR-Quanv. Giới hạn: LUT chỉ áp dụng cho mạch tĩnh/đã đóng băng.

## Nhật ký sự kiện (append-only)

- 2026-11-07 14:28 — Khảo sát repo xong, tạo log.
- 2026-11-07 14:28 — Step 1 xong: compiler OK, verify khớp QNode gốc.
- 2026-11-07 14:29 — Step 2 xong: self-test OK (ordering đúng, interp ≡ nearest tại điểm lưới).
- 2026-11-07 14:32 — Step 3 xong: chạy full, kết quả ở Mục 3; Mục 5 ghi rõ ngưỡng nào không đạt.
