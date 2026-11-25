# Quanvolutional Neural Networks for Edge Medical Imaging (`MedMNIST`)

Nghiên cứu thực nghiệm, bóc tách nhân quả và bộ công cụ benchmark tái lập cho lớp **Quanvolutional Layer (4-qubit cố định)** trên bộ dữ liệu ảnh y tế **MedMNIST** (`PneumoniaMNIST`, `BreastMNIST`, `OCTMNIST`).

---

## 1. Tổng quan & Kết luận Thực nghiệm (Scientific & Engineering Summary)

1. **Bằng chứng thực nghiệm trả lời Câu hỏi Nghiên cứu A (Tính tương đương Bảng tra):**
   Toàn bộ lớp quanvolution 4-qubit với mạch cố định ($L=2$ tầng `StronglyEntanglingLayers`) có thể biên dịch trước thành bảng tra tĩnh 65,536 phần tử chiếm **`256.00 KiB`** (`INT8`) hoặc **`1,024.00 KiB`** (`FP32 Interp4D`, độ tương đồng cosine **`0.999993`** so với mô phỏng giải tích). Tại thời điểm suy luận **không còn phép tính lượng tử nào trong hệ thống** — mọi hiệu ứng quan sát được bắt nguồn từ hàm phi tuyến bị chặn $[-1, 1]$ mà bảng tra biểu diễn, phù hợp với phê phán của *Bowles, Ahmed & Schuld* về các kiến trúc QML quy mô nhỏ mô phỏng cổ điển được.
2. **Kết quả đối chứng 7 nhánh & Kiểm định thống kê (`345` kiểm định, 5 Seeds, TOST $\pm 2\%$, Benjamini–Hochberg FDR):**
   - Trong bóc tách 4 nhánh gốc trên `PneumoniaMNIST`, khả năng kháng nhiễu đốm đến chủ yếu từ bộ lọc tiền xử lý `phase_damping_u8` (`+41.49%` tại $\sigma=0.15$, $p_{\text{adj,fam}}=0.0010$; `+42.48%` tại $\sigma=0.20$, $p_{\text{adj,fam}}<0.0001$); khi so với baseline công bằng `(2) Ctrl + Damping`, nhánh `(4) Quanv + PreLUT` thấp hơn ở 4/5 mức nhiễu (`p_adj,fam` $\ge 0.2390$).
   - Khi đối chứng mở rộng 7 nhánh (`Ctrl Conv2d`, `Conv2d + tanh [-1,1]`, `Conv2d + hard-clip [-1,1]`, `Conv2d + NoiseAug`, `CLUT-SepTrig`, `CLUT-RandMLP`, `Q-LUT`) trên `PneumoniaMNIST` và `BreastMNIST`, `Q-LUT` **không tạo ra lợi thế độ chính xác có ý nghĩa thống kê** (thấp hơn `Conv2d + tanh` trên 4/5 mức nhiễu Pneumonia Stabilized với $p_{\text{adj,fam}} \in [0.0125, 0.0152]$; thấp hơn `Conv2d + NoiseAug` ở Raw Stem $\sigma \ge 0.15$ với $p_{\text{adj,glob}} \le 0.0017$; không vượt `CLUT-RandMLP` / `CLUT-SepTrig`).
3. **Sản phẩm phần mềm benchmark tái lập:**
   Thư viện `rust_qlut` (SIMD + giao diện C-ABI), công cụ kiểm chứng `medquanv-cli` (`max_logit_diff < 5.3e-6` so với PyTorch trên 624 ảnh test thật), và kịch bản tái lập một lệnh [`scripts/reproduce_all.py`](scripts/reproduce_all.py).

---

## 2. Kết quả Benchmark Chính (Đã kiểm toán & Thống nhất)

### A. Hiệu năng CPU `x86_64` & Kiểm toán Bộ nhớ (`reports/quantization_and_memory_audit.md`)
*(Lưu ý: Toàn bộ đo đạc độ trễ dưới đây thực hiện trên CPU `x86_64`; không so sánh với PennyLane Python và không ngoại suy FPS sang Jetson Nano. Việc đo trực tiếp trên bo mạch NVIDIA Jetson Nano B01 đang chờ hoàn tất phần cứng cấp nguồn.)*

- **So sánh 1 tầng $2\times 2$ (`1024×1024`, $s=2$) với `PyTorch Conv2d` (`x86_64`):**
  - `rust_qlut` `INT8 Shift` (`256 KiB`): `0.2718 ms` — **`4.89×` nhanh hơn** `PyTorch Conv2d` (`1.3300 ms`).
  - `rust_qlut` `Interp4D` (`1,024 KiB`): `6.4739 ms` — **`4.87×` chậm hơn** `PyTorch Conv2d` (`1.3300 ms`).
- **Tác động rời rạc hóa 4-bit (`INT8 Shift` vs `Interp4D`, Zero-retraining):**
  - `BreastMNIST`: Đạt tương đương **TOST $\pm 2\%$ trên 5/5 mức nhiễu** ($|\Delta| \le 0.43\%$, $p_{\text{TOST}} \le 0.0178$).
  - `PneumoniaMNIST`: Giảm `-0.45%` trên ảnh sạch (`EQUIV`, $p_{\text{TOST}} = 0.0002$); đạt `EQUIV` 5/5 dưới `stabilized_stem` đối xứng ($p_{\text{TOST}} \le 0.0004$).
  - `OCTMNIST` (vi tương phản võng mạc): Giảm `-3.40% / -4.69% / -2.53%` tại $\sigma=0.00 / 0.05 / 0.10$ ($p_{\text{adj,fam}} \le 0.0143$, không đạt TOST).
- **Dung lượng bộ nhớ thống nhất (`WideGRQuanvNet v2`, 64-bit, trọng số đã gộp BN `577.26 KiB`):**
  - `INT8 Shift-LUT`: **`833.26 KiB` tĩnh** (`256.00` LUT + `0.00` index + `577.26` weights = `40.69%` L2 2 MB); cộng đỉnh kích hoạt sống `110.25 KiB` = **`943.51 KiB`**.
  - `Interp4D SIMD`: **`1,610.26 KiB` tĩnh** (`1,024.00` LUT + `9.00` `INTERP_TABLE` + `577.26` weights = `78.63%` L2); cộng đỉnh kích hoạt sống = **`1,720.51 KiB`**.
  - Bộ `v3` (`--suite-v3`, `28×28`, `x86_64`): `quanv_v3_int8` (`1.126 ms`, `887.8 FPS`, `406.90 KiB`) có độ trễ tương đương `ctrl_v3_light` (`1.114 ms`, `897.6 FPS`, `151.79 KiB`).

### B. Tóm tắt Đối chứng 7 Nhánh (5 Seeds, `Accuracy %`, `Interp4D` cho các LUT)
Nguồn: [`reports/rigorous_stats_summary.md`](reports/rigorous_stats_summary.md).

| Mô hình (`Params`) | Pneumonia Raw ($\sigma=0.00/0.10/0.20$) | Pneumonia Stabilized ($\sigma=0.00/0.10/0.20$) | Breast Raw ($\sigma=0.00/0.10/0.20$) | Breast Stabilized ($\sigma=0.00/0.10/0.20$) |
| :--- | :---: | :---: | :---: | :---: |
| `Ctrl (Conv2d)` (`148.4K`) | `90.22 / 60.60 / 40.20` | `90.22 / 87.79 / 82.68` | `84.10 / 81.37 / 76.07` | `84.10 / 83.21 / 81.58` |
| `Conv2d + tanh [-1,1]` (`148.4K`) | `90.71 / 63.39 / 39.69` | **`90.71 / 88.16 / 85.02`** | `84.23 / 81.50 / 75.43` | `84.23 / 84.19 / 82.26` |
| `Conv2d + clip [-1,1]` (`148.4K`) | `90.00 / 64.63 / 40.37` | `90.00 / 87.45 / 83.21` | `84.87 / 81.24 / 75.47` | **`84.87 / 85.73 / 84.74`** |
| `Conv2d + NoiseAug` (`148.4K`) | **`90.77 / 89.54 / 87.67`** | `90.77 / 87.40 / 85.76` | **`85.51 / 84.96 / 82.31`** | `85.51 / 83.46 / 81.50` |
| `CLUT-SepTrig` (`148.3K`) | `90.10 / 72.29 / 41.46` | `90.10 / 86.88 / 82.74` | `84.87 / 82.74 / 77.44` | `84.87 / 83.16 / 80.73` |
| `CLUT-RandMLP` (`148.3K`) | `90.48 / 71.53 / 46.69` | `90.48 / 87.71 / 82.52` | `85.77 / 83.25 / 77.69` | `85.77 / 85.73 / 83.21` |
| `Q-LUT SU(16) L=2` (`148.3K`) | `88.94 / 71.29 / 39.49` | `88.94 / 85.80 / 83.39` | `84.74 / 81.88 / 77.05` | `84.74 / 85.00 / 82.14` |

---

## 3. Tái lập Thực nghiệm Bằng Một Lệnh (One-Command Reproducibility)

Chạy kiểm tra toàn bộ kết quả khóa, unit test Rust, 345 kiểm định thống kê (TOST + BH FDR) và báo cáo kiểm toán bộ nhớ/lượng tử hóa:

```bash
python scripts/reproduce_all.py
```

Các tùy chọn khác:
```bash
# Chạy/tiếp tục huấn luyện & đánh giá lại 3 baseline cổ điển mới (yêu cầu dữ liệu gốc + GPU/CPU)
python scripts/reproduce_all.py --retrain-extra

# Chỉ chạy lại kiểm định thống kê & audit (nếu máy chưa cài Rust/cargo)
python scripts/reproduce_all.py --skip-cargo
```

---

## 4. Cấu trúc Thư mục & Báo cáo Chính

- [`reports/TECHNICAL_REPORT_EDGE_QUANV.md`](reports/TECHNICAL_REPORT_EDGE_QUANV.md): Báo cáo kỹ thuật đã chuẩn hóa theo nhận xét GVHD.
- [`reports/rigorous_stats_summary.md`](reports/rigorous_stats_summary.md): Bảng đầy đủ 345 kiểm định cặp, 95%/90% CI, TOST $\pm 2\%$ và BH FDR.
- [`reports/quantization_and_memory_audit.md`](reports/quantization_and_memory_audit.md): Kiểm toán chi tiết lượng tử hóa 4-bit và bộ nhớ (`833.26 / 943.51 KiB`).
- `rust_qlut/`: Thư viện Rust (`Q-LUT` + `WideResidualHead`) và CLI `medquanv-cli` (`cargo test --no-default-features`).
- `experiments/`: Script huấn luyện & đánh giá (`run_pillar2_wide.py`, `run_pillar3_v3.py`, `train_ablation_clut.py`, `eval_ablation_clut.py`, `run_extra_baselines.py`).
- `scripts/`: `reproduce_all.py`, `run_rigorous_stats.py`, `generate_quant_mem_audit.py`, `generate_clut.py`, `build_jetson_release.sh`.
- `deploy/jetson_nano/` & `dist/jetson_nano/`: Bộ khung triển khai và đo kiểm cho NVIDIA Jetson Nano B01 *(chờ đo trên bo mạch thật)*.
