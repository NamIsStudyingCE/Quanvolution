# Quanvolutional Neural Networks for Edge Medical Imaging (`MedMNIST`)

Nghiên cứu, đánh giá thực nghiệm và tối ưu hóa triển khai biên (**Edge Inference**) cho mạng **Quanvolutional Neural Network** trên bộ dữ liệu ảnh y tế **MedMNIST** (`PneumoniaMNIST`, `BreastMNIST`, `OCTMNIST`).

---

## 1. Tổng quan & Trạng thái Hiện tại (Honest Engineering Summary)

Dự án đi qua 3 giai đoạn chính:
1. **Giai đoạn 1 (SOICT v4 / GD0–GD3 Baseline):** Đánh giá mô hình Quanvolution chuẩn (PennyLane CPU simulator) trên `BreastMNIST`, `PneumoniaMNIST`, `OCTMNIST`. Kết quả cho thấy mô phỏng mạch lượng tử trên CPU có độ trễ lớn (`~220.22 ms/ảnh` trên `OCTMNIST`), chưa chứng minh được ưu thế vượt trội so với Classical CNN thuần.
2. **Giai đoạn 2 (`Q-LUT` + `rust_qlut` Edge Engine & `WideGRQuanvNet` v2/v3):**
   - Giải quyết triệt để nút thắt cổ chai $O(4^n)$ của mạch lượng tử 4-qubit cố định bằng **Quantum Look-Up Table (`Q-LUT`)** kết hợp lõi suy luận **Rust (`rust_qlut`)** tối ưu bộ nhớ đệm L2 và **SIMD**.
   - Độ trễ tầng lượng tử giảm từ mức hàng trăm `ms` xuống **`0.0007 ms/ảnh`** (`INT8 Shift-LUT`, 256 KiB) và **`0.0056 ms/ảnh`** (`Interp4D`, 1,024 KiB) trong `hardware_sanity.json`, với tổng dung lượng mô hình + bảng tra `INT8 Shift-LUT` đạt **`833.26 KiB`** (`40.7%` của 2 MB L2 Cache).
   - Binary `medquanv-cli` đạt độ chính xác khớp PyTorch (`max_logit_diff < 5e-6`): `89.42%` (`INT8 Shift`) và `90.22%` (`Interp4D`) trên 624 ảnh test `PneumoniaMNIST`, tốc độ end-to-end `~335–338 FPS` trên CPU x86_64.
   - Hỗ trợ nén mô hình `v3` (`quanv_v3_int8`: `39,250` params vs `ctrl_v3_light`: `39,470` params; `micro_quanv`: `2,002` params vs `micro_ctrl`: `2,078` params).
3. **Giai đoạn 3 (Bóc tách nhân quả `C-LUT vs Q-LUT`):**
   - Thực hiện thí nghiệm đối chứng cô lập (5 seeds trên `PneumoniaMNIST` & `BreastMNIST`, xem [`results/ablation_clut/ablation_summary.md`](results/ablation_clut/ablation_summary.md)) giữa `Q-LUT`, `CLUT-SepTrig`, `CLUT-RandMLP`, và `Ctrl Conv2d` ở cả hai chế độ `Raw Stem` và `Stabilized Stem Symmetric`.
   - **Kết luận trung thực:** `Q-LUT` cố định 4-qubit giải quyết bài toán **tốc độ & bộ nhớ biên** khi đóng gói thành LUT, nhưng **không vượt trội về độ chính xác** so với `C-LUT` đã hiệu chuẩn phương sai hoặc `Ctrl Conv2d` được ổn định đối xứng.

---

## 2. Kết quả Thực nghiệm Đã Khóa (Locked Benchmarks)

### A. Hiệu năng Vi kiến trúc & Parity (`medquanv-cli` / `rust_qlut`)
Nguồn: [`reports/medquanv_cli_verification_a3.json`](reports/medquanv_cli_verification_a3.json) và [`results/ablation_clut/hardware_sanity.json`](results/ablation_clut/hardware_sanity.json).

| Thành phần / Chế độ | Độ trễ LUT (`µs`) | End-to-End (`ms/ảnh`) | Throughput (`FPS`) | Accuracy (`PneumoniaMNIST`) | Max Logit Diff vs PyTorch |
|---|---:|---:|---:|---:|---:|
| `rust_qlut` (`INT8 Shift-LUT`, 256 KiB) | `1.139 µs` (`0.66 µs` single) | `2.961 ms` | `337.71` | `89.4231%` | `3.81e-6` |
| `rust_qlut` (`Interp4D`, 1,024 KiB) | `5.503 µs` (`5.60 µs` single) | `2.980 ms` | `335.61` | `90.2244%` | `4.29e-6` |

- Tổng bộ nhớ tĩnh `INT8 Shift-LUT` + `WideResidualHead`: **`833.26 KiB`** (`40.7%` dung lượng L2 Cache 2 MB).

### B. Đối chứng Cô lập `C-LUT vs Q-LUT` (5 Seeds)
Nguồn: [`results/ablation_clut/ablation_summary.md`](results/ablation_clut/ablation_summary.md).
- Trên `PneumoniaMNIST` và `BreastMNIST`, khi tách biệt hoàn toàn yếu tố lượng tử hóa 4-bit bị chặn $[-1, 1]$ và bộ lọc ổn định pha (`Stabilized Stem`), `Q-LUT` không tạo ra khác biệt có ý nghĩa thống kê vượt trội so với `CLUT-SepTrig`, `CLUT-RandMLP`, hoặc `Ctrl Conv2d`.

---

## 3. Cấu trúc Thư mục Chính

- `rust_qlut/`: Thư viện Rust (`Q-LUT` + `WideResidualHead`) và CLI `medquanv-cli` hỗ trợ `--benchmark`.
- `experiments/`: Script huấn luyện và đánh giá (`train_ablation_clut.py`, `eval_ablation_clut.py`, `ablation_clut_vs_qlut.py`, ...).
- `scripts/`:
  - `generate_clut.py`: Sinh bảng tra cổ điển đối chứng (`CLUT-SepTrig`, `CLUT-RandMLP`).
  - `build_jetson_release.sh`: Script biên dịch chéo cho `aarch64-unknown-linux-gnu` (Cortex-A57).
- `results/ablation_clut/`: Toàn bộ số liệu và báo cáo bóc tách `C-LUT vs Q-LUT` đã khóa.
- `reports/`: Kết quả kiểm thử parity `medquanv_cli_verification_a3.json`, `pillar3_v3_results.json`, v.v.
- `deploy/jetson_nano/` & `dist/jetson_nano/`: Bộ khung triển khai cho NVIDIA Jetson Nano (`run_on_board_bench.sh`, `README_JETSON.md`, `rust_qlut.h`). *(Lưu ý: Đây là bộ script/header chuẩn bị sẵn cho cross-build; chưa biên dịch và đo trực tiếp trên bo mạch Jetson Nano thật).*

---

## 4. Kiểm thử & Biên dịch Nhanh

Chạy unit test cho `rust_qlut` (không cần feature phụ thuộc Python):
```bash
cd rust_qlut
cargo test --no-default-features
```
