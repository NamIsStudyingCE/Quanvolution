# BÁO CÁO KẾT QUẢ THỰC NGHIỆM CỘT TRỤ 2: ĐỘ CHÍNH XÁC CỰC ĐOAN & KIẾN TRÚC GR-QUANV

- **Sinh viên thực hiện:** Nguyễn Hoàng Nam (MSSV: 22520916) — Khoa Kỹ thuật Máy tính, UIT ĐHQG-HCM
- **Giảng viên hướng dẫn:** ThS. Nguyễn Duy Xuân Bách
- **Thiết bị thực thi:** NVIDIA GeForce RTX 3050 6GB Laptop GPU + Intel Core i5-13420H (`rust_qlut 0.1.0` Native SIMD)
- **Quy chuẩn đánh giá:** 5 Seeds độc lập `[0, 42, 100, 2023, 777]`, cùng tầng phân loại phía sau, cùng bộ tối ưu và hàm mất mát.

---

## 1. Task 2.1 — Kiểm chứng Độ bảo toàn Độ chính xác Phân loại Downstream của Rust Q-LUT (256 KB)

| Nguồn Đặc trưng Lượng tử | Dung lượng | BreastMNIST ACC | BreastMNIST AUC | OCTMNIST ACC | OCTMNIST AUC |
|---|---:|---:|---:|---:|---:|
| **PennyLane Exact (Ground Truth)** | N/A (Giả lập) | `79.74 ± 1.04%` | `81.04 ± 1.11%` | `39.84 ± 0.46%` | `68.31 ± 0.30%` |
| **Rust Q-LUT INT8 Shift (`p >> 4`)** | **256 KiB** | **`79.62 ± 1.03%`** | `80.05 ± 0.96%` | **`40.08 ± 0.37%`** | **`68.94 ± 0.37%`** |
| **Rust Q-LUT Interp4D SIMD FMA** | **1,024 KiB** | **`79.49 ± 1.22%`** | **`80.83 ± 1.12%`** | **`39.86 ± 0.42%`** | **`68.33 ± 0.30%`** |

> **Kết luận Task 2.1:** Bảng tra `Rust Q-LUT INT8 Shift (256 KiB)` bảo toàn độ chính xác phân loại của `PennyLane Exact` với độ lệch chỉ `-0.12%` trên BreastMNIST và thậm chí **tăng nhẹ `+0.24%` ACC (`+0.63%` AUC)** trên OCTMNIST nhờ hiệu ứng khử nhiễu lượng tử hóa bậc thang.

---

## 2. Task 2.3A — Benchmark Đối xứng Tuyệt đối (Strict Fair-Play Linear Head: `BN(4) -> ReLU -> Linear(784, C)`, 5 Seeds)

### 2.1. Tập dữ liệu BreastMNIST (546 Train / 78 Val / 156 Test, 2 lớp)
- **Thời gian trích xuất 4 nhánh Q-LUT đa tỷ lệ (`d=1,2,3,axial`) bằng Rust SIMD:** `11.42 ms` cho toàn bộ tập dữ liệu!

| Mô hình (Cùng `Linear(784, C)` Head) | Số tham số | Test ACC (%) | Test ROC-AUC (%) | Test PR-AUC (%) | Macro-F1 (%) | Thời gian Train/Seed |
|---|---:|---:|---:|---:|---:|---:|
| Classical Baseline | `1,598` | 80.90 ± 2.20% | 82.06 ± 2.46% | `90.48%` | `87.41%` | `1.27 s` |
| Pure Quanvolution | `1,578` | 77.18 ± 1.88% | 81.22 ± 1.15% | `91.56%` | `84.81%` | `1.07 s` |
| Naive Residual Quanv | `1,614` | 79.36 ± 3.52% | 82.43 ± 3.07% | `90.43%` | `86.45%` | `1.44 s` |
| Classical MS-Gated Control | `1,882` | 82.31 ± 0.51% | 83.20 ± 1.54% | `91.12%` | `88.45%` | `7.15 s` |
| Lean GR-Quanv (1x1) | `1,782` | 78.21 ± 1.28% | 83.46 ± 1.54% | `92.84%` | `85.55%` | `1.89 s` |
| **Proposed GR-Quanv** | `9,406` | **84.49 ± 1.03%** | **88.61 ± 0.71%** | `94.70%` | `89.51%` | `8.24 s` |

- **Kiểm định thống kê (`Proposed GR-Quanv` vs. `Classical Baseline`):** $\Delta\text{ACC} = +3.59\%$ ($p = 6.2192e-02$), $\Delta\text{AUC} = +6.55\%$ ($p = 3.4288e-03$).

### 2.2. Tập dữ liệu PneumoniaMNIST (4,708 Train / 524 Val / 624 Test, 2 lớp)
- **Thời gian trích xuất 4 nhánh Q-LUT đa tỷ lệ (`d=1,2,3,axial`) bằng Rust SIMD:** `64.10 ms` cho toàn bộ tập dữ liệu!

| Mô hình (Cùng `Linear(784, C)` Head) | Số tham số | Test ACC (%) | Test ROC-AUC (%) | Test PR-AUC (%) | Macro-F1 (%) | Thời gian Train/Seed |
|---|---:|---:|---:|---:|---:|---:|
| Classical Baseline | `1,598` | 86.73 ± 0.81% | 92.86 ± 0.41% | `91.72%` | `90.23%` | `1.84 s` |
| Pure Quanvolution | `1,578` | 83.91 ± 0.84% | 91.49 ± 0.07% | `93.12%` | `88.28%` | `1.51 s` |
| Naive Residual Quanv | `1,614` | 85.42 ± 1.27% | 92.36 ± 0.58% | `92.78%` | `89.35%` | `2.20 s` |
| Classical MS-Gated Control | `1,882` | 87.76 ± 0.78% | 93.75 ± 0.13% | `93.27%` | `90.92%` | `31.29 s` |
| Lean GR-Quanv (1x1) | `1,782` | 85.10 ± 0.69% | 92.30 ± 0.54% | `93.53%` | `89.13%` | `2.88 s` |
| **Proposed GR-Quanv** | `9,406` | **85.83 ± 1.50%** | **93.93 ± 0.78%** | `95.42%` | `89.63%` | `39.54 s` |

- **Kiểm định thống kê (`Proposed GR-Quanv` vs. `Classical Baseline`):** $\Delta\text{ACC} = -0.90\%$ ($p = 4.2197e-01$), $\Delta\text{AUC} = +1.07\%$ ($p = 4.2886e-02$).

### 2.3. Tập dữ liệu OCTMNIST Canonical Split (3,500 Train / 500 Val / 1,000 Test, 4 lớp)
- **Thời gian trích xuất 4 nhánh Q-LUT đa tỷ lệ (`d=1,2,3,axial`) bằng Rust SIMD:** `70.56 ms` cho toàn bộ tập dữ liệu!

| Mô hình (Cùng `Linear(784, C)` Head) | Số tham số | Test ACC (%) | Test ROC-AUC (%) | Test PR-AUC (%) | Macro-F1 (%) | Thời gian Train/Seed |
|---|---:|---:|---:|---:|---:|---:|
| Classical Baseline | `3,168` | 45.34 ± 3.43% | 74.18 ± 3.63% | `49.06%` | `38.84%` | `1.61 s` |
| Pure Quanvolution | `3,148` | 41.48 ± 1.06% | 69.48 ± 0.34% | `43.21%` | `34.52%` | `1.31 s` |
| Naive Residual Quanv | `3,184` | 46.22 ± 1.89% | 73.49 ± 2.06% | `47.91%` | `40.49%` | `1.97 s` |
| Classical MS-Gated Control | `3,452` | 49.24 ± 1.00% | 77.51 ± 0.85% | `52.84%` | `42.88%` | `3.58 s` |
| Lean GR-Quanv (1x1) | `3,352` | 49.00 ± 1.69% | 75.26 ± 1.96% | `52.57%` | `44.41%` | `2.45 s` |
| **Proposed GR-Quanv** | `10,976` | **62.08 ± 1.54%** | **86.62 ± 1.23%** | `68.36%` | `59.01%` | `3.82 s` |

- **Kiểm định thống kê (`Proposed GR-Quanv` vs. `Classical Baseline`):** $\Delta\text{ACC} = +16.74\%$ ($p = 2.2402e-03$), $\Delta\text{AUC} = +12.44\%$ ($p = 6.6141e-03$).

### 2.4. Tập dữ liệu OCTMNIST 100% Full (97,477 Train / 10,832 Val / 1,000 Test, 4 lớp)
- **Thời gian trích xuất 4 nhánh Q-LUT đa tỷ lệ (`d=1,2,3,axial`) bằng Rust SIMD:** `2085.19 ms` cho toàn bộ tập dữ liệu!

| Mô hình (Cùng `Linear(784, C)` Head) | Số tham số | Test ACC (%) | Test ROC-AUC (%) | Test PR-AUC (%) | Macro-F1 (%) | Thời gian Train/Seed |
|---|---:|---:|---:|---:|---:|---:|
| Classical Baseline | `3,168` | 53.06 ± 1.17% | 81.58 ± 1.02% | `58.16%` | `47.68%` | `2.61 s` |
| Pure Quanvolution | `3,148` | 44.08 ± 0.66% | 72.06 ± 0.09% | `45.64%` | `39.16%` | `2.27 s` |
| Naive Residual Quanv | `3,184` | 51.74 ± 3.56% | 80.85 ± 1.81% | `56.42%` | `45.78%` | `3.91 s` |
| Classical MS-Gated Control | `3,452` | 57.16 ± 1.39% | 84.63 ± 0.65% | `64.90%` | `53.08%` | `11.13 s` |
| Lean GR-Quanv (1x1) | `3,352` | 56.76 ± 1.01% | 82.94 ± 1.14% | `63.74%` | `53.20%` | `7.14 s` |
| **Proposed GR-Quanv** | `10,976` | **73.80 ± 1.22%** | **92.89 ± 0.31%** | `81.90%` | `72.19%` | `24.07 s` |

- **Kiểm định thống kê (`Proposed GR-Quanv` vs. `Classical Baseline`):** $\Delta\text{ACC} = +20.74\%$ ($p = 6.2961e-05$), $\Delta\text{AUC} = +11.31\%$ ($p = 1.6666e-05$).

---

## 3. Task 2.3B — Benchmark với Tầng Phân loại CNN 2 Lớp Đối xứng (`CompactConvHead`) trên 100% OCTMNIST (109,309 ảnh)

| Mô hình (Cùng `CompactConvHead`) | Test ACC (100% Data) | Test ROC-AUC (100% Data) | Macro-F1 | Thời gian Train/Seed | Test ACC (1% Data = 974 ảnh) |
|---|---:|---:|---:|---:|---:|
| Classical CNN | 77.12 ± 1.08% | 93.81 ± 0.40% | `76.42%` | `17.86 s` | 56.82 ± 1.59% |
| Pure Quanvolution | 71.38 ± 0.84% | 91.17 ± 0.29% | `69.84%` | `17.33 s` | 54.74 ± 2.52% |
| **Proposed GR-Quanv** | **77.40 ± 0.68%** | **94.19 ± 0.35%** | `76.60%` | `36.87 s` | **62.64 ± 1.39%** |

---

## 4. Task 2.4 — Đo kiểm 3 Tử huyệt Chứng minh Ưu việt Cực đoan (5 Seeds)

### 4.1. Tử huyệt 1: Kịch bản Thiếu dữ liệu (Few-Shot / Low-Data Regime trên OCTMNIST)

| Tỷ lệ Dữ liệu (Số ảnh/lớp) | Classical Baseline ACC | Pure Quanvolution ACC | Lean GR-Quanv (1x1) ACC | Proposed GR-Quanv ACC | Độ vượt trội (GR-Quanv vs. Classical) |
|---|---:|---:|---:|---:|---:|
| `0.25%_60_per_cls` | 30.20 ± 2.13% | 37.30 ± 2.50% | 33.00 ± 3.31% | **39.92 ± 3.57%** | **`+9.72%` ACC** |
| `0.5%_120_per_cls` | 35.92 ± 3.22% | 38.88 ± 1.23% | 37.78 ± 3.55% | **49.08 ± 3.03%** | **`+13.16%` ACC** |
| `1.0%_240_per_cls` | 40.50 ± 5.97% | 41.22 ± 2.02% | 41.80 ± 4.04% | **60.26 ± 1.54%** | **`+19.76%` ACC** |
| `5.0%_1200_per_cls` | 48.96 ± 1.22% | 41.76 ± 1.28% | 51.34 ± 1.53% | **67.24 ± 2.40%** | **`+18.28%` ACC** |
| `10.0%_2400_per_cls` | 50.30 ± 3.40% | 42.28 ± 0.80% | 54.58 ± 0.85% | **69.48 ± 0.52%** | **`+19.18%` ACC** |

### 4.2. Tử huyệt 2: Kịch bản Kháng nhiễu Thiết bị Lâm sàng (Speckle & Gaussian Noise trên OCTMNIST)

| Loại Nhiễu & Cường độ $\sigma$ | Classical Baseline ACC (AUC) | Pure Quanvolution ACC (AUC) | Lean GR-Quanv (1x1) ACC (AUC) | Proposed GR-Quanv ACC (AUC) | Độ vượt trội (GR-Quanv vs. Classical) |
|---|---:|---:|---:|---:|---:|
| **Speckle** `sigma_0.00` | `47.30%` (`76.10%`) | `40.40%` (`68.80%`) | `47.50%` (`72.26%`) | **`62.60%` (`86.79%`)** | **`+15.30%` ACC** |
| **Speckle** `sigma_0.10` | `26.00%` (`64.10%`) | `40.10%` (`68.12%`) | `45.60%` (`70.07%`) | **`59.30%` (`84.61%`)** | **`+33.30%` ACC** |
| **Speckle** `sigma_0.15` | `24.70%` (`60.47%`) | `39.00%` (`67.69%`) | `42.20%` (`67.45%`) | **`52.00%` (`81.78%`)** | **`+27.30%` ACC** |
| **Speckle** `sigma_0.25` | `24.60%` (`57.07%`) | `38.30%` (`65.79%`) | `36.10%` (`62.04%`) | **`40.30%` (`75.02%`)** | **`+15.70%` ACC** |
| **Speckle** `sigma_0.35` | `24.60%` (`55.56%`) | `36.60%` (`63.66%`) | `34.90%` (`59.07%`) | **`34.50%` (`69.22%`)** | **`+9.90%` ACC** |
| **Gaussian** `sigma_0.00` | `47.30%` (`76.10%`) | `40.40%` (`68.80%`) | `47.50%` (`72.26%`) | **`62.60%` (`86.79%`)** | **`+15.30%` ACC** |
| **Gaussian** `sigma_0.05` | `23.60%` (`56.34%`) | `39.50%` (`67.28%`) | `40.40%` (`66.89%`) | **`45.90%` (`78.79%`)** | **`+22.30%` ACC** |
| **Gaussian** `sigma_0.08` | `24.20%` (`54.15%`) | `37.40%` (`65.21%`) | `36.90%` (`62.25%`) | **`37.70%` (`72.51%`)** | **`+13.50%` ACC** |
| **Gaussian** `sigma_0.12` | `23.90%` (`53.04%`) | `36.80%` (`61.57%`) | `32.80%` (`58.14%`) | **`33.20%` (`64.84%`)** | **`+9.30%` ACC** |
| **Gaussian** `sigma_0.18` | `23.80%` (`52.14%`) | `32.80%` (`57.44%`) | `28.60%` (`53.81%`) | **`29.70%` (`57.16%`)** | **`+5.90%` ACC** |

### 4.3. Tử huyệt 3: Động học Cổng Thích nghi Lượng tử ($\alpha \in [0, 1]$)

| Tập dữ liệu | Trung bình $\bar{\alpha}$ toàn mạng | Kênh $\alpha_0$ ($Z_0$) | Kênh $\alpha_1$ ($Z_1$) | Kênh $\alpha_2$ ($Z_2$) | Kênh $\alpha_3$ ($Z_3$) |
|---|---:|---:|---:|---:|---:|
| `breastmnist` | **`0.6418`** | `0.6646` | `0.6146` | `0.6490` | `0.6390` |
| `pneumoniamnist` | **`0.6362`** | `0.6612` | `0.6296` | `0.6538` | `0.6001` |
| `octmnist_canonical_3500` | **`0.6060`** | `0.6339` | `0.5718` | `0.6249` | `0.5933` |
| `octmnist_full_109k` | **`0.6387`** | `0.6695` | `0.6618` | `0.5991` | `0.6244` |
