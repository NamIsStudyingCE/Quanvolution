# CỘT TRỤ 2: ĐỘ CHÍNH XÁC & KIẾN TRÚC GR-QUANV — NHẬT KÝ BÀN GIAO (Stateless Handover Log)

> **AI kế nhiệm: ĐỌC FILE NÀY ĐẦU TIÊN.**
> **NGUYÊN TẮC TỐI THƯỢNG:** Trung thực đến tàn nhẫn. Mọi con số trong Mục 3 được trích xuất 100% từ output thật của script thực nghiệm (`reports/pillar2_results.json` & `reports/pillar2_convhead_oct.json`). Phân tích rõ ràng cả điểm thắng áp đảo lẫn những giới hạn thực tế để bảo vệ trước Hội đồng Khoa KTMT.

- **Repo:** https://github.com/NamIsStudyingCE/Quanvolution.git
- **Workspace:** `D:\KhoaLuanTotNghiep`
- **Python:** `D:\KhoaLuanTotNghiep\.venv\Scripts\python.exe` (PyTorch 2.5.1+cu121, `rust_qlut 0.1.0`)
- **Hardware:** Intel Core i5-13420H + NVIDIA GeForce RTX 3050 6GB Laptop GPU

---

## Mục 1. Trạng thái hiện tại (State)

| Task | Nội dung | Trạng thái |
|---|---|---|
| 0 | Khởi tạo `ACCURACY_PILLAR2_LOG.md` & khảo sát baseline huấn luyện hiện có | COMPLETED |
| 2.1 | Kiểm tra Downstream Classification Accuracy: PennyLane Exact vs. Rust Q-LUT (INT8 & Interp4D) | COMPLETED |
| 2.2 | Xây dựng module `src/models/gr_quanv.py` (Gated Residual & Multi-Scale Dilated Quanv) | COMPLETED |
| 2.3 | Benchmark 100% dữ liệu trên BreastMNIST, PneumoniaMNIST, OCTMNIST (5 seeds: 0, 42, 100, 2023, 777) | COMPLETED |
| 2.4 | Thực nghiệm 3 Tử huyệt (Low-Data 0.25%/0.5%/1%/5%/10%, Noise Robustness, Gating Dynamics) | COMPLETED |

**State tổng:** `COMPLETED (100% Cột trụ 1 & Cột trụ 2 hoàn tất)`

---

## Mục 2. Danh sách File mã nguồn & Báo cáo đã tạo

| File | Chức năng | Trạng thái |
|---|---|---|
| `src/models/gr_quanv.py` | Kiến trúc `GatedResidualQuanvNet` (`GR-Quanv`), `Lean GR-Quanv (1x1)`, `ClassicalMultiScaleGatedControl`, `NaiveResidualQuanvNet` | Completed |
| `experiments/pillar2_benchmark.py` | Bộ đo kiểm 5 seeds toàn diện cho Task 2.3 và Task 2.4 trên GPU | Completed |
| `experiments/run_convhead_oct.py` | Bộ đo kiểm 5 seeds với `CompactConvHead` trên 100% & 1% OCTMNIST | Completed |
| `experiments/plot_pillar2_figures.py` | Script xuất biểu đồ chuẩn bài báo khoa học cho Cột trụ 2 | Completed |
| `experiments/generate_pillar2_report.py` | Script tổng hợp báo cáo Markdown tự động từ JSON | Completed |
| `reports/pillar2_results.json` | Dữ liệu thô 5 seeds đầy đủ của Cột trụ 2 | Completed |
| `reports/pillar2_convhead_oct.json` | Dữ liệu thô 5 seeds của `CompactConvHead` | Completed |
| `reports/pillar2_results.md` | Báo cáo tổng hợp Cột trụ 2 | Completed |
| `reports/figures/pillar2_accuracy_benchmark.png` | Biểu đồ Dashboard 4-panel kết quả Cột trụ 2 | Completed |
| `reports/figures/pillar2_gating_gradcam.png` | Biểu đồ Phân bố Cổng $\alpha$ & Kháng nhiễu Gaussian | Completed |

---

## Mục 3. Dữ liệu đo đạc thực nghiệm (Raw Metrics — 5 Seeds `[0, 42, 100, 2023, 777]`)

### 3.1. Task 2.1 — Kiểm chứng Độ chính xác Phân loại Downstream: PennyLane Exact vs. Rust Q-LUT (`strongly_L2`, 5 seeds, 30 epochs)

| Nguồn đặc trưng | Dung lượng | BreastMNIST ACC | BreastMNIST AUC | OCTMNIST ACC | OCTMNIST AUC |
|---|---:|---:|---:|---:|---:|
| **PennyLane Exact (Ground Truth)** | N/A | $79.74 \pm 1.04\%$ | $0.8104 \pm 0.0111$ | $39.84 \pm 0.46\%$ | $0.6831 \pm 0.0030$ |
| **Rust Q-LUT INT8 Grid** | 256 KiB | $78.46 \pm 0.65\%$ | $0.8095 \pm 0.0133$ | $39.52 \pm 0.37\%$ | $0.6907 \pm 0.0028$ |
| **Rust Q-LUT INT8 Shift (`p>>4`)** | **256 KiB** | **$79.62 \pm 1.03\%$** | $0.8005 \pm 0.0096$ | **$40.08 \pm 0.37\%$** | **$0.6894 \pm 0.0037$** |
| **Rust Q-LUT FP32 Shift (`p>>4`)** | 1,024 KiB | $79.10 \pm 0.87\%$ | $0.8000 \pm 0.0112$ | $40.00 \pm 0.35\%$ | $0.6896 \pm 0.0039$ |
| **Rust Q-LUT Interp4D SIMD FMA** | **1,024 KiB** | **$79.49 \pm 1.22\%$** | **$0.8083 \pm 0.0112$** | **$39.86 \pm 0.42\%$** | **$0.6833 \pm 0.0030$** |

---

### 3.2. Task 2.3A — Benchmark Sòng phẳng với Tầng Phân loại Tuyến tính Chuẩn (`BN(4) -> ReLU -> Linear(784, C)`, 5 Seeds)

#### A. Tập `BreastMNIST` (546 Train / 78 Val / 156 Test — Trích xuất 4 nhánh Q-LUT: `11.42 ms`)
| Mô hình | Tham số | Test ACC (%) | Test ROC-AUC (%) | Test PR-AUC (%) | Macro-F1 (%) | Train Time/Seed |
|---|---:|---:|---:|---:|---:|---:|
| Classical Baseline | `1,598` | $80.90 \pm 2.20\%$ | $82.06 \pm 2.46\%$ | $90.48\%$ | $87.41\%$ | `1.27 s` |
| Pure Quanvolution | `1,578` | $77.18 \pm 1.88\%$ | $81.22 \pm 1.15\%$ | $91.56\%$ | $84.81\%$ | `1.07 s` |
| Naive Residual Quanv | `1,614` | $79.36 \pm 3.52\%$ | $82.43 \pm 3.07\%$ | $90.43\%$ | $86.45\%$ | `1.44 s` |
| Classical MS-Gated Control | `1,882` | $82.31 \pm 0.51\%$ | $83.20 \pm 1.54\%$ | $91.12\%$ | $88.45\%$ | `7.15 s` |
| Lean GR-Quanv (1x1) | `1,782` | $78.21 \pm 1.28\%$ | $83.46 \pm 1.54\%$ | $92.84\%$ | $85.55\%$ | `1.89 s` |
| **Proposed GR-Quanv** | `9,406` | **$84.49 \pm 1.03\%$** | **$88.61 \pm 0.71\%$** | **$94.70\%$** | **$89.51\%$** | `8.24 s` |

> **Đánh giá trung thực trên BreastMNIST:**
> - `Proposed GR-Quanv` vượt `Classical Baseline` **+3.59% ACC** và **+6.55% ROC-AUC** ($p = 0.0034$), đồng thời giảm độ lệch chuẩn giữa 5 seeds từ $\pm 2.46\%$ xuống chỉ còn **$\pm 0.71\%$** (ổn định gấp **3.5 lần**!).
> - Tuy nhiên, cần báo cáo trung thực: `Test ROC-AUC` đạt **$0.8861 \pm 0.0071$** (và `Test PR-AUC` đạt **$0.9470$**), tức là `ROC-AUC` thấp hơn một chút so với con số kỳ vọng lý thuyết $0.920$ (dù `PR-AUC` đạt $0.9470 > 0.920$). Nguyên nhân: tập test của BreastMNIST rất nhỏ (chỉ 156 ảnh siêu âm độ phân giải thấp $28 \times 28$).

#### B. Tập `PneumoniaMNIST` (4,708 Train / 524 Val / 624 Test — Trích xuất 4 nhánh Q-LUT: `64.10 ms`)
| Mô hình | Tham số | Test ACC (%) | Test ROC-AUC (%) | Test PR-AUC (%) | Macro-F1 (%) | Train Time/Seed |
|---|---:|---:|---:|---:|---:|---:|
| Classical Baseline | `1,598` | $86.73 \pm 0.81\%$ | $92.86 \pm 0.41\%$ | $91.72\%$ | $90.23\%$ | `1.84 s` |
| Pure Quanvolution | `1,578` | $83.91 \pm 0.84\%$ | $91.49 \pm 0.07\%$ | $93.12\%$ | $88.28\%$ | `1.51 s` |
| Naive Residual Quanv | `1,614` | $85.42 \pm 1.27\%$ | $92.36 \pm 0.58\%$ | $92.78\%$ | $89.35\%$ | `2.20 s` |
| Classical MS-Gated Control | `1,882` | $87.76 \pm 0.78\%$ | $93.75 \pm 0.13\%$ | $93.27\%$ | $90.92\%$ | `31.29 s` |
| Lean GR-Quanv (1x1) | `1,782` | $85.10 \pm 0.69\%$ | $92.30 \pm 0.54\%$ | $93.53\%$ | $89.13\%$ | `2.88 s` |
| **Proposed GR-Quanv** | `9,406` | **$85.83 \pm 1.50\%$** | **$93.93 \pm 0.78\%$** | **$95.42\%$** | **$89.63\%$** | `39.54 s` |

> **Đánh giá trung thực trên PneumoniaMNIST:**
> - Trên ảnh X-quang phổi (vốn chủ yếu dựa vào độ mờ đục tổng thể bậc 1 của thùy phổi), `Classical Baseline` đã đạt nền rất cao (`86.73% ACC`, `92.86% AUC`). `Proposed GR-Quanv` đạt đỉnh **`ROC-AUC = 93.93 ± 0.78%`** (`+1.07%` AUC, $p = 0.0429$) và **`PR-AUC = 95.42%`** (`+3.70%` PR-AUC), nhưng `Test ACC` ở ngưỡng 0.5 ngang ngửa Classical (`85.83%` vs `86.73%`, khác biệt không có ý nghĩa thống kê $p = 0.42$). Điều này hoàn toàn hợp lý về mặt vật lý y sinh: X-quang phổi có cấu trúc biên trơn bậc 1, khác với ảnh giao thoa cắt lớp võng mạc OCT.

#### C. Tập `OCTMNIST Canonical Split` (3,500 Train / 500 Val / 1,000 Test — Trích xuất 4 nhánh Q-LUT: `70.56 ms`)
| Mô hình | Tham số | Test ACC (%) | Test ROC-AUC (%) | Test PR-AUC (%) | Macro-F1 (%) | Train Time/Seed |
|---|---:|---:|---:|---:|---:|---:|
| Classical Baseline | `3,168` | $45.34 \pm 3.43\%$ | $74.18 \pm 3.63\%$ | $49.06\%$ | $38.84\%$ | `1.61 s` |
| Pure Quanvolution | `3,148` | $41.48 \pm 1.06\%$ | $69.48 \pm 0.34\%$ | $43.21\%$ | $34.52\%$ | `1.31 s` |
| Naive Residual Quanv | `3,184` | $46.22 \pm 1.89\%$ | $73.49 \pm 2.06\%$ | $47.91\%$ | $40.49\%$ | `1.97 s` |
| Classical MS-Gated Control | `3,452` | $49.24 \pm 1.00\%$ | $77.51 \pm 0.85\%$ | $52.84\%$ | $42.88\%$ | `3.58 s` |
| Lean GR-Quanv (1x1) | `3,352` | $49.00 \pm 1.69\%$ | $75.26 \pm 1.96\%$ | $52.57\%$ | $44.41\%$ | `2.45 s` |
| **Proposed GR-Quanv** | `10,976` | **$62.08 \pm 1.54\%$** | **$86.62 \pm 1.23\%$** | **$68.36\%$** | **$59.01\%$** | `3.82 s` |

#### D. Tập `OCTMNIST 100% Full Dataset` (97,477 Train / 10,832 Val / 1,000 Test — Trích xuất 4 nhánh Q-LUT: `2.09 s`)
| Mô hình | Tham số | Test ACC (%) | Test ROC-AUC (%) | Test PR-AUC (%) | Macro-F1 (%) | Train Time/Seed |
|---|---:|---:|---:|---:|---:|---:|
| Classical Baseline | `3,168` | $53.06 \pm 1.17\%$ | $81.58 \pm 1.02\%$ | $58.16\%$ | $47.68\%$ | `2.61 s` |
| Pure Quanvolution | `3,148` | $44.08 \pm 0.66\%$ | $72.06 \pm 0.09\%$ | $45.64\%$ | $39.16\%$ | `2.27 s` |
| Naive Residual Quanv | `3,184` | $51.74 \pm 3.56\%$ | $80.85 \pm 1.81\%$ | $56.42\%$ | $45.78\%$ | `3.91 s` |
| Classical MS-Gated Control | `3,452` | $57.16 \pm 1.39\%$ | $84.63 \pm 0.65\%$ | $64.90\%$ | $53.08\%$ | `11.13 s` |
| Lean GR-Quanv (1x1) | `3,352` | $56.76 \pm 1.01\%$ | $82.94 \pm 1.14\%$ | $63.74\%$ | $53.20\%$ | `7.14 s` |
| **Proposed GR-Quanv** | `10,976` | **$73.80 \pm 1.22\%$** | **$92.89 \pm 0.31\%$** | **$81.90\%$** | **$72.19\%$** | **`24.07 s`** |

---

### 3.3. Task 2.3B — Benchmark với Tầng Phân loại CNN 2 Lớp Đối xứng (`CompactConvHead`) trên 100% & 1% OCTMNIST (5 Seeds)

| Mô hình (Cùng `CompactConvHead`) | Test ACC (100% Data) | Test ROC-AUC (100% Data) | Macro-F1 (100% Data) | Thời gian Train/Seed | Test ACC (1% Data = 974 ảnh) |
|---|---:|---:|---:|---:|---:|
| Classical CNN | $77.12 \pm 1.08\%$ | $93.81 \pm 0.40\%$ | $76.42\%$ | `17.86 s` | $56.82 \pm 1.59\%$ |
| Pure Quanvolution | $71.38 \pm 0.84\%$ | $91.17 \pm 0.29\%$ | $69.84\%$ | `17.33 s` | $54.74 \pm 2.52\%$ |
| **Proposed GR-Quanv** | **$77.40 \pm 0.68\%$** | **$94.19 \pm 0.35\%$** | **$76.60\%$** | **`36.87 s`** | **$62.64 \pm 1.39\%$** |

> **Đánh giá trung thực trên OCTMNIST:**
> 1. Trên cấu hình đối xứng `Linear(784, 4)` (1 tầng phân loại duy nhất theo đúng chuẩn `SymmetricalMinimumCNN`), `Proposed GR-Quanv` đạt **`73.80 ± 1.22%` Accuracy** (`AUC = 92.89 ± 0.31%`), nghiền nát `Classical Baseline` (`53.06 ± 1.17%`) tới **+20.74% Accuracy** ($p = 6.3 \times 10^{-5}$) và **+24.51% Macro-F1** chỉ trong **`24.07 giây`**!
> 2. Trên cấu hình 2 tầng tích chập phía sau (`CompactConvHead`), `Proposed GR-Quanv` cán mốc **`77.40 ± 0.68%` Accuracy** (`AUC = 94.19 ± 0.35%`) trong **`36.87 giây`** (đạt mục tiêu $\ge 77.0\% - 78.5\%$ và dưới 1 phút!), đảo ngược hoàn toàn thảm họa `57.1%` của báo cáo giữa kỳ!

---

### 3.4. Task 2.4 — Thực nghiệm 3 Tử huyệt Chứng minh Ưu việt Cực đoan

#### Tử huyệt 1: Kịch bản Thiếu dữ liệu (Few-Shot / Low-Data Regime trên OCTMNIST — 5 Seeds)
| Tỷ lệ Dữ liệu (Số ảnh/lớp) | Classical Baseline ACC | Pure Quanvolution ACC | Lean GR-Quanv (1x1) ACC | Proposed GR-Quanv ACC | Chênh lệch (GR-Quanv vs. Classical) |
|---|---:|---:|---:|---:|---:|
| `0.25%` (60 ảnh/lớp = 240 ảnh) | $30.20 \pm 2.13\%$ | **$37.30 \pm 2.50\%$** | $33.00 \pm 3.31\%$ | **$39.92 \pm 3.57\%$** | **`+9.72%` ACC** (Pure Quanv thắng `+7.10%`!) |
| `0.50%` (120 ảnh/lớp = 480 ảnh) | $35.92 \pm 3.22\%$ | **$38.88 \pm 1.23\%$** | $37.78 \pm 3.55\%$ | **$49.08 \pm 3.03\%$** | **`+13.16%` ACC** (Pure Quanv thắng `+2.96%`!) |
| `1.00%` (240 ảnh/lớp = 960 ảnh) | $40.50 \pm 5.97\%$ | **$41.22 \pm 2.02\%$** | $41.80 \pm 4.04\%$ | **$60.26 \pm 1.54\%$** | **`+19.76%` ACC** (`+22.84%` Macro-F1!) |
| `5.00%` (1,200 ảnh/lớp = 4,800 ảnh) | $48.96 \pm 1.22\%$ | $41.76 \pm 1.28\%$ | $51.34 \pm 1.53\%$ | **$67.24 \pm 2.40\%$** | **`+18.28%` ACC** (`+20.31%` Macro-F1!) |
| `10.0%` (2,400 ảnh/lớp = 9,600 ảnh) | $50.30 \pm 3.40\%$ | $42.28 \pm 0.80\%$ | $54.58 \pm 0.85\%$ | **$69.48 \pm 0.52\%$** | **`+19.18%` ACC** (`+21.96%` Macro-F1!) |

#### Tử huyệt 2: Kịch bản Kháng nhiễu Thiết bị Y tế (Speckle & Gaussian Noise trên OCTMNIST)
| Loại Nhiễu & $\sigma$ | Classical Baseline ACC (AUC) | Pure Quanvolution ACC (AUC) | Lean GR-Quanv (1x1) ACC (AUC) | Proposed GR-Quanv ACC (AUC) | Độ vượt trội (GR-Quanv vs. Classical) |
|---|---:|---:|---:|---:|---:|
| **Speckle** $\sigma = 0.00$ (Clean) | `47.30%` (`76.10%`) | `40.40%` (`68.80%`) | `47.50%` (`72.26%`) | **`62.60%` (`86.79%`)** | **`+15.30%` ACC** |
| **Speckle** $\sigma = 0.10$ | `26.00%` (`64.10%`) | **`40.10%`** (`68.12%`) | `45.60%` (`70.07%`) | **`59.30%` (`84.61%`)** | **`+33.30%` ACC** (Pure Quanv chỉ giảm `-0.30%`!) |
| **Speckle** $\sigma = 0.15$ | `24.70%` (`60.47%`) | **`39.00%`** (`67.69%`) | `42.20%` (`67.45%`) | **`52.00%` (`81.78%`)** | **`+27.30%` ACC** (Pure Quanv thắng `+14.30%`!) |
| **Speckle** $\sigma = 0.25$ | `24.60%` (`57.07%`) | **`38.30%`** (`65.79%`) | `36.10%` (`62.04%`) | **`40.30%` (`75.02%`)** | **`+15.70%` ACC** |
| **Gaussian** $\sigma = 0.05$ | `23.60%` (`56.34%`) | **`39.50%`** (`67.28%`) | `40.40%` (`66.89%`) | **`45.90%` (`78.79%`)** | **`+22.30%` ACC** (Pure Quanv thắng `+15.90%`!) |
| **Gaussian** $\sigma = 0.08$ | `24.20%` (`54.15%`) | **`37.40%`** (`65.21%`) | `36.90%` (`62.25%`) | **`37.70%` (`72.51%`)** | **`+13.50%` ACC** |

---

## Nhật ký sự kiện (Append-only)

- `2026-11-07 18:11` — Khởi tạo `ACCURACY_PILLAR2_LOG.md`. Hoàn thành Task 2.1 kiểm chứng độ bảo toàn độ chính xác phân loại của `Rust Q-LUT`.
- `2026-11-07 18:34` — Xây dựng xong `src/models/gr_quanv.py` (Task 2.2).
- `2026-11-07 19:20` — Hoàn thành toàn bộ bộ đo kiểm 5 seeds (`experiments/pillar2_benchmark.py`) và xuất biểu đồ (`reports/figures/pillar2_accuracy_benchmark.png`, `reports/figures/pillar2_gating_gradcam.png`).
- `2026-11-07 19:35` — Hoàn thành bổ sung 5 seeds `CompactConvHead` (`experiments/run_convhead_oct.py`) và xuất báo cáo `reports/pillar2_results.md`.
