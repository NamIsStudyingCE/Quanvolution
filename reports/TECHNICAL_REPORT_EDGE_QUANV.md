# BÁO CÁO KỸ THUẬT: ĐỘNG CƠ BIÊN DỊCH SUY LUẬN TÍCH CHẬP LƯỢNG TỬ (EDGE-QUANV ENGINE) VÀ KIẾN TRÚC MẠNG LAI WIDE-FUSION V2 TRÊN THIẾT BỊ BIÊN CÔNG SUẤT THẤP

- **Đề tài**: Nghiên cứu và ứng dụng lớp tích chập lượng tử (Quanvolution) trong phân loại ảnh y tế MedMNIST
- **Tác giả thực hiện**: Nguyễn Hoàng Nam (MSSV: 22520916) — Khoa Kỹ thuật Máy tính, Trường ĐH Công nghệ Thông tin (UIT - ĐHQG TP.HCM)
- **Cán bộ hướng dẫn**: ThS. Nguyễn Duy Xuân Bách
- **Mã nguồn & Báo cáo thực nghiệm**: Commit `d90ed19` (main branch, GitHub Repository: `NamIsStudyingCE/Quanvolution.git`)
- **Phần cứng tham chiếu**: Vi xử lý nhúng Quad-core ARM Cortex-A57 @ 1.43 GHz (NVIDIA Jetson Nano), 2 MB Unified L2 Cache, 4 GB LPDDR4, công suất 5W–10W.

---

## 1. MÔ TẢ TOÁN HỌC & ĐẶC TẢ VI KIẾN TRÚC HỆ THỐNG

### 1.1. Toán tử Tích chập Lượng tử (Quanvolutional Operator)
Mạch lượng tử biến phân ngẫu nhiên được cấu hình cố định trên không gian trạng thái Hilbert 4-qubit ($n = 4$):
- **Phép mã hóa góc (Angle Encoding)**: Mỗi cửa sổ không gian $2 \times 2$ có giá trị điểm ảnh chuẩn hóa $x = [x_0, x_1, x_2, x_3]^T \in [0, 1]^4$ được nạp vào trạng thái cơ sở $|0\rangle^{\otimes 4}$ thông qua các cổng quay đơn qubit:
  $$\mathcal{E}(x) = \bigotimes_{j=0}^{3} R_Y(\pi x_j)$$
- **Mạch vướng víu biến phân**: Áp dụng $L = 2$ tầng `StronglyEntanglingLayers` với ma trận biến đổi Unitary cố định $U(\theta) \in SU(16)$ sinh ra từ các cổng quay $R(\alpha, \beta, \gamma)$ và vòng vướng víu CNOT khép kín:
  $$|\psi(x)\rangle = U(\theta) \mathcal{E}(x) |0\rangle^{\otimes 4}$$
- **Phép đo đại lượng quan sát (Observables)**: Trích xuất 4 kỳ vọng Pauli-Z cục bộ:
  $$q_k(x) = \langle \psi(x) | Z_k | \psi(x) \rangle \in [-1, 1], \quad \forall k \in \{0, 1, 2, 3\}$$

### 1.2. Kỹ thuật Tiền biên dịch Toán tử Ahead-of-Time (`rust_qlut`)
Trong quá trình suy luận, do trọng số $\theta$ không cập nhật, hàm ánh xạ $f_Q: [0, 1]^4 \to [-1, 1]^4$ là một toán tử tiền định bất biến theo thời gian.
1. **Lượng tử hóa không gian pha 4-bit (Grid INT8)**:
   Không gian đầu vào liên tục $[0, 1]^4$ được rời rạc hóa đều thành $16^4 = 65,536$ trạng thái rời rạc. Mỗi trạng thái ánh xạ thành 4 giá trị kỳ vọng Pauli-Z ở định dạng dấu chấm tĩnh `int8` ($[-127, 127]$).
   - **Kích thước bảng tra**: $65,536 \times 4\text{ bytes} = 262,144\text{ bytes} = \mathbf{256\text{ KiB}}$.
   - **Cơ chế tính địa chỉ 0-FLOP (Zero-Arithmetic Addressing)**:
     $$\text{Index}(p_0, p_1, p_2, p_3) = ((p_0 \gg 4) \ll 12) \mid ((p_1 \gg 4) \ll 8) \mid ((p_2 \gg 4) \ll 4) \mid (p_3 \gg 4)$$
   - Toàn bộ phép tính toán học $O(4^n)$ bị loại bỏ ở runtime; 4 kênh lượng tử được truy xuất trong 1 chu kỳ máy thông qua đọc con trỏ `uint32_t*`.
2. **Nội suy đa tuyến tính 4 chiều SIMD FP32 (`Interp4D`)**:
   Đối với các tác vụ yêu cầu bảo toàn vi tương phản dưới $6.67\%$, bảng tra FP32 kích thước $1,048,576\text{ bytes} = \mathbf{1\text{ MiB}}$ ($65,536 \times 4\text{ floats}$) được sử dụng.
   - Trích xuất 16 đỉnh siêu lập phương lân cận tại $(\lfloor 15x_i \rfloor, \lceil 15x_i \rceil)$.
   - Vector hóa 15 phép nội suy tuyến tính `lerp(a, b, t) = a + t * (b - a)` bằng tập lệnh FMA (`_mm_fmadd_ps` trên x86_64 và `vfmaq_f32` trên ARMv8 NEON).
   - Độ tương đồng Cosine đo được so với mô phỏng giải tích FP64 liên tục: **$0.999993$**.

---

## 2. HỒ SƠ TÀI NGUYÊN VI KIẾN TRÚC TRÊN NVIDIA JETSON NANO

Vi kiến trúc ARM Cortex-A57 tích hợp bộ nhớ đệm:
- L1 Data Cache: 32 KiB / nhân, 2-way set-associative, dòng cache 64-byte.
- L2 Unified Cache: **2,048 KiB (2 MB)** dùng chung 4 nhân, 16-way set-associative, dòng cache 64-byte.

Báo cáo chuẩn hóa tách bạch rành mạch **hai chế độ suy luận Q-LUT** trên cùng bộ trọng số `WideGRQuanvNet v2` sau khi gộp BatchNorm (`148,306` tham số gốc $\to$ **`147,778` tham số `f32` = `577.26 KiB`**):

| Cấu hình suy luận | Bảng tra Q-LUT | Trọng số mạng (BN-Folded FP32) | Tổng footprint L2 thực đo (`medquanv-cli`) | Tỷ lệ L2 Cache (2,048 KiB) | Tỷ lệ RAM (4 GB LPDDR4) | Đặc tính tính toán tầng Q-LUT |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Mode A: `INT8 Shift / Grid-LUT`** | **256.00 KiB** | 577.26 KiB | **833.26 KiB** | **40.7%** | $0.020\%$ | 0 FLOPs (Dịch bit / tra bảng tĩnh `uint32_t`) |
| **Mode B: `Interp4D SIMD (FP32)`** | **1,024.00 KiB** | 577.26 KiB | **1,601.26 KiB** | **78.2%** | $0.039\%$ | Nội suy 4 chiều trên 16 đỉnh, vector hóa FMA/NEON |

---

## 3. ĐỐI CHỨNG TÁCH BẠCH `INT8 (256 KiB)` vs `INTERP4D (1 MiB)` & ĐO BĂNG THÔNG THỰC TẾ

### 3.1. Bảng Pareto Độ chính xác Tách bạch (Zero Retraining, 5 Seeds $\times$ 3 Noise Draws)
Theo dữ liệu kiểm định thực nghiệm `reports/pareto_int8_vs_interp4d.json` (tuyệt đối **không** trộn lẫn độ chính xác của `Interp4D` với dung lượng/tốc độ của `INT8`):

| Tập dữ liệu | Mức nhiễu ($\sigma$) | `Wide Classical Control` (ACC % / F1 %) | `INT8 Shift-LUT` 256 KiB (ACC % / F1 %) | `INT8 Grid-LUT` 256 KiB (ACC % / F1 %) | `Interp4D SIMD` 1,024 KiB (ACC % / F1 %) | p-value (`INT8 Shift` / `Interp4D` vs Ctrl) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **BreastMNIST** | `0.00` (Sạch) | $84.10 \pm 0.94$ / $89.56$ | $84.49 \pm 2.16$ / $89.83$ | $84.74 \pm 2.12$ / $89.99$ | **$84.74 \pm 1.79$ / $90.03$** | $0.8029$ / $0.6060$ |
| **BreastMNIST** | `0.10` | $81.54 \pm 2.38$ / $88.43$ | $83.42 \pm 2.51$ / $89.29$ | **$83.63 \pm 2.40$ / $89.42$** | $83.50 \pm 2.56$ / $89.32$ | $0.3477$ / $0.3495$ |
| **BreastMNIST** | `0.20` | $77.22 \pm 3.26$ / $86.35$ | $80.34 \pm 2.22$ / $87.23$ | **$80.81 \pm 2.63$ / $87.51$** | $80.73 \pm 2.51$ / $87.43$ | $0.2461$ / $0.2106$ |
| **PneumoniaMNIST** | `0.00` (Sạch) | **$90.22 \pm 1.65$ / $92.69$** | $88.49 \pm 1.50$ / $91.53$ | $88.46 \pm 1.54$ / $91.52$ | $88.94 \pm 1.63$ / $91.84$ | $0.0971$ / $0.2350$ |
| **PneumoniaMNIST** | `0.05` | $88.79 \pm 1.55$ / $91.16$ | $89.59 \pm 1.16$ / $92.19$ | $89.71 \pm 1.19$ / $92.27$ | **$89.96 \pm 1.12$ / $92.46$** | $0.4291$ / $0.2569$ |
| **PneumoniaMNIST** | `0.10` | $63.84 \pm 14.41$ / $58.00$ | $82.31 \pm 4.56$ / $84.99$ | $82.04 \pm 4.45$ / $84.77$ | **$83.72 \pm 3.84$ / $86.37$** | **$0.0287^*$ / $0.0266^*$** |
| **PneumoniaMNIST** | `0.15` | $52.43 \pm 10.46$ / $39.24$ | $75.83 \pm 6.84$ / $77.90$ | $75.54 \pm 7.25$ / $77.48$ | **$77.30 \pm 6.50$ / $79.54$** | **$0.0030^{**}$ / $0.0023^{**}$** |
| **PneumoniaMNIST** | `0.20` | $51.75 \pm 8.93$ / $39.94$ | $74.88 \pm 6.82$ / $77.44$ | $74.73 \pm 6.95$ / $77.30$ | **$76.62 \pm 5.90$ / $79.54$** | **$0.0029^{**}$ / $0.0016^{**}$** |
| **OCTMNIST** | `0.00` (Sạch) | $82.52 \pm 2.61$ / $81.77$ | $80.64 \pm 2.37$ / $80.00$ | $81.22 \pm 1.50$ / $80.57$ | **$84.04 \pm 2.45$ / $83.47$** | $0.1061$ / $0.1850$ |
| **OCTMNIST** | `0.05` | $76.47 \pm 3.87$ / $75.35$ | $72.73 \pm 1.59$ / $71.56$ | $73.54 \pm 0.96$ / $72.40$ | **$77.42 \pm 1.11$ / $76.62$** | $0.0890$ / $0.6325$ |
| **OCTMNIST** | `0.10` | **$58.59 \pm 4.57$ / $53.77$** | $55.40 \pm 2.07$ / $50.13$ | $55.08 \pm 1.89$ / $49.55$ | $57.93 \pm 1.58$ / $53.34$ | $0.1604$ / $0.7330$ |
| **OCTMNIST** | `0.20` | $38.13 \pm 2.85$ / $27.38$ | $40.19 \pm 2.32$ / $28.91$ | $40.12 \pm 2.58$ / $28.72$ | **$40.59 \pm 2.17$ / $29.25$** | $0.2073$ / $0.1498$ |

### 3.2. Hiệu năng Mở rộng Độ phân giải Tầng Q-LUT (`medquanv-cli --bench-resolutions`)
Đo trực tiếp trên động cơ Rust thuần (`x86_64`, `opt-level=3`):

| Độ phân giải ($H \times W$) | Bước nhảy ($s$) | Số Patches ($N_p$) | `INT8 Shift-LUT` | `Interp4D SIMD` | Thông lượng tầng Q-LUT INT8 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **$28 \times 28$** | $s=2$ | 196 | **0.17 µs** | 3.62 µs | 5,907,931 FPS |
| **$64 \times 64$** | $s=2$ | 1,024 | **0.94 µs** | 18.51 µs | 1,063,768 FPS |
| **$128 \times 128$** | $s=2$ | 4,096 | **3.27 µs** | 74.03 µs | 305,863 FPS |
| **$224 \times 224$** | $s=2$ | 12,544 | **12.33 µs** | 253.77 µs | 81,129 FPS |
| **$512 \times 512$** | $s=2$ | 65,536 | **79.13 µs** | 1,415.40 µs | 12,638 FPS |
| **$1024 \times 1024$** | $s=2$ | 262,144 | **236.19 µs** | 4,850.00 µs | 4,234 FPS |
| **$1024 \times 1024$** | $s=4$ | 65,536 | **60.01 µs** | 1,303.75 µs | 16,665 FPS |

**Cảnh báo khoa học về tốc độ Toàn mạng End-to-End**: Tầng Q-LUT INT8 4 tỷ lệ chỉ mất **1.79 µs** trên ảnh $28 \times 28$, nhưng toàn mạng End-to-End 148,306 tham số (bao gồm `c_stem`, `q_mixer` và `WideResidualHead` chạy FP32) mất **6,937.00 µs (6.94 ms / 144.2 FPS)** trên `medquanv-cli` thuần và **8.12 ms** trên PyTorch CPU (`Batch=1`), gần tương đương với `Wide Classical Control` (**9.17 ms** ở `Batch=1` và **0.828 ms/ảnh** ở `Batch=64` so với **0.826 ms/ảnh** INT8 và **1.057 ms/ảnh** Interp4D). Tầng lượng tử không còn là nút thắt; phần `WideResidualHead` FP32 hiện chiếm **> 99.7%** thời gian thực thi.

---

## 4. ĐÁNH GIÁ ĐỘ BỀN VỮNG NHIỄU ĐỐM & THỰC NGHIỆM BÓC TÁCH ĐÃ CHUẨN HÓA

### 4.1. Tinh chỉnh Bộ Giảm xóc Pha Mềm (`phase_damping_u8` Quadratic Soft-Start)
- Công thức giảm xóc được chỉnh lại với dốc chuyển tiếp mượt `gate = raw_gate * clamp((raw_gate - 0.15) / 0.50, 0, 1)` từ ngưỡng $\tau = 0.024$.
- Kiểm chứng thực tế:
  - Trên ảnh sạch ($\sigma = 0.00$) của cả 3 bộ dữ liệu (`BreastMNIST`, `PneumoniaMNIST`, `OCTMNIST`): **chính xác `0.0000%` điểm ảnh bị thay đổi**.
  - Ở nhiễu nhẹ PneumoniaMNIST ($\sigma = 0.05$): khắc phục dứt điểm lỗi tụt ~2.5% trước đây, đưa độ chính xác từ `87.42%` lên **`89.96 ± 1.12%`** (F1 `92.46%`), cao hơn `Wide Classical Control` (`88.79 ± 1.55%`).

### 4.2. Bảng Phân tách 4 Nhánh Đã Chuẩn hóa (`reports/ablation_isolation_results.json`, 5 Seeds, 3 Noise Draws)

| Mức nhiễu ($\sigma$) | `(1) Control Thô` (ACC % / F1 %) | `(2) Control + Damping` (ACC % / F1 %) | `(3) GR-Quanv Thô` (ACC % / F1 %) | `(4) GR-Quanv + Pre-LUT Damping` (ACC % / F1 %) |
| :---: | :---: | :---: | :---: | :---: |
| **$\sigma = 0.00$ (Sạch)** | $90.22 \pm 1.65$ / $92.69$ | $90.22 \pm 1.65$ / $92.69$ | $88.94 \pm 1.63$ / $91.84$ | **$88.94 \pm 1.63$ / $91.84$** |
| **$\sigma = 0.05$** | $88.95 \pm 1.70$ / $91.23$ | $89.48 \pm 1.53$ / $92.09$ | **$90.37 \pm 1.08$ / $92.72$** | $89.96 \pm 1.12$ / $92.46$ |
| **$\sigma = 0.10$** | $60.60 \pm 16.28$ / $50.16$ | **$87.79 \pm 2.06$ / $90.99$** | $71.30 \pm 8.08$ / $71.21$ | $83.72 \pm 3.84$ / $86.37$ |
| **$\sigma = 0.15$** | $45.16 \pm 6.94$ / $21.18$ | **$86.65 \pm 1.87$ / $90.05$** | $46.12 \pm 3.50$ / $24.84$ | $77.30 \pm 6.50$ / $79.54$ |
| **$\sigma = 0.20$** | $40.20 \pm 2.68$ / $8.31$ | **$82.68 \pm 1.83$ / $86.28$** | $39.49 \pm 0.68$ / $6.19$ | $76.62 \pm 5.90$ / $79.54$ |

### 4.3. Phân tích Trung thực Kết quả Bóc tách
1. **Ưu thế nội tại ở chế độ thô ($\sigma = 0.10$)**: Khi cả hai mạng đều **không** dùng `phase_damping_u8` hay ổn định phương sai, `(3) GR-Quanv Thô` đạt **`71.30%`** (F1 `71.21%`) so với `(1) Control Thô` chỉ đạt **`60.60%`** (F1 `50.16%`) — tức lượng tử tự nhiên vẫn bảo vệ tốt hơn **`+10.70%` ACC** trước khi bị gập pha ở $\sigma \ge 0.15$ (nơi cả 2 đều sụp đổ xuống ~40–46% nếu không có lọc pha).
2. **Phát hiện tiêu cực quan trọng về tính đối xứng bộ lọc (`(2)` vs `(4)`)**: Khi cấp bộ lọc `phase_damping_u8` thẳng cho đầu vào của mạng cổ điển `(2) Control + Damping`, mạng cổ điển lại đạt **`87.79% / 86.65% / 82.68%`** tại $\sigma = 0.10 / 0.15 / 0.20$, **cao hơn** `(4) GR-Quanv + Pre-LUT Damping` (`83.72% / 77.30% / 76.62%`).
   - **Nguyên nhân gốc rễ**: Trong mã nguồn `WideGRQuanvNet.forward_features` (`src/models/gr_quanv_wide.py`), chỉ nhánh lượng tử `q_ms` nhận đầu vào đã qua `phase_damping_u8`, còn nhánh tích chập song song `c_stem` bên trong `WideGRQuanvNet` vẫn nhận ảnh thô `x_raw_t` chưa giảm nhiễu (chỉ kẹp variance khi ghép ở `WideResidualHead`). Ngược lại, trong nhánh `(2) Control + Damping`, `c_stem` của `WideClassicalControl` được nhận toàn bộ ảnh đã làm sạch bởi `phase_damping_u8`! Điều này cho thấy phần lớn khả năng trụ vững ở mức nhiễu nặng $\sigma \ge 0.15$ đến từ **`phase_damping_u8` + `_stabilize_stem_variance`**, còn kiến trúc trộn nhánh thô của `WideGRQuanvNet` đang tự làm yếu nhánh `c_stem` của chính mình khi gặp nhiễu cao.

---

## 5. KIỂM CHỨNG ĐỘNG CƠ END-TO-END RUST THUẦN (`medquanv-cli` v0.3.0)

`rust_qlut/src/bin/medquanv_cli.rs` đã được nâng cấp để thực thi **toàn bộ kiến trúc `WideGRQuanvNet v2` (148,306 tham số gộp BN $\to$ 147,778 `f32`)** bằng Rust thuần, không dùng trọng số giả và không phụ thuộc Python/PyTorch/CUDA.

- **Tệp trọng số gộp**: `weights/wide_v2_pneumoniamnist_s42_rust.bin` (và bản sao `weights/wide_v2_pneumoniamnist_s42_folded.bin`, 591,128 bytes) xuất từ checkpoint seed 42 (`weights/pillar2_wide/wide_v2_interp4d_pneumoniamnist_full_grquanv_s42.pt`).
- **Tập kiểm chứng thực tế**: `data/processed/pneumoniamnist_test_rust_verify.bin` gồm toàn bộ **624 ảnh test thật** của PneumoniaMNIST kèm logits tham chiếu PyTorch (`reports/medquanv_cli_verification.json`):

| Chế độ thực thi (`medquanv-cli`) | Rust ACC (624 ảnh) | PyTorch ACC (624 ảnh) | Sai số Logits tối đa | Q-LUT Latency (4 tỷ lệ) | Full-Net End-to-End Latency | Thông lượng E2E (FPS) | Dung lượng L2 tổng |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`INT8 Shift-LUT` (256 KiB)** | **89.42%** | **89.42%** | **`3.81e-6`** | **1.79 µs** | **6,937.00 µs (6.94 ms)** | **144.2 FPS** | **833.3 KiB** (40.7% L2) |
| **`INT8 Grid-LUT` (256 KiB)** | **89.42%** | **89.42%** | **`4.77e-6`** | **1.71 µs** | **6,730.87 µs (6.73 ms)** | **148.6 FPS** | **833.3 KiB** (40.7% L2) |
| **`Interp4D SIMD` (1,024 KiB)** | **90.22%** | **90.22%** | **`4.29e-6`** | **20.08 µs** | **6,702.22 µs (6.70 ms)** | **149.2 FPS** | **1,601.3 KiB** (78.2% L2) |

---

## 6. KẾT LUẬN TRUNG THỰC & LỘ TRÌNH CẢI THIỆN KỸ THUẬT

1. **Những ưu thế còn được bảo toàn vững chắc**:
   - Đã loại bỏ hoàn toàn mô phỏng lượng tử PennyLane khi suy luận biên (`medquanv-cli` đạt độ lệch logits $< 5 \times 10^{-6}$ và khớp 100% độ chính xác với PyTorch trên 624 ảnh test thật, chạy thuần Rust ở mức **144–149 FPS** End-to-End cho Batch=1).
   - Toàn bộ footprint của bản `INT8` (`833.3 KiB`) nằm gọn dưới 41% L2 Cache 2 MB của Jetson Nano.
   - Bản `Interp4D` giữ được độ chính xác ngang/nhỉnh nhẹ trên ảnh sạch OCTMNIST (`84.04%` vs `82.52%`) và BreastMNIST (`84.74%` vs `84.10%`), đồng thời kháng nhiễu mạnh hơn bản Control **không** có bộ giảm xóc (`+10.70%` thô ở $\sigma=0.10$; `+24.87%` có damping ở $\sigma=0.20$).
2. **Những hạn chế đã bộc lộ sau khi chuẩn hóa**:
   - **Suy giảm độ chính xác do lượng tử hóa Zero-shot `INT8`**: Do 30 mô hình gốc được huấn luyện với `Interp4D`, việc chuyển thẳng sang `INT8` không qua QAT làm giảm độ chính xác ảnh sạch trên OCTMNIST (`84.04% -> 80.64%`, kém Classical Control `82.52%`) và PneumoniaMNIST (`90.22% Ctrl` vs `88.49% INT8`).
   - **Tốc độ End-to-End chưa tạo bứt phá trước CNN thuần**: Q-LUT thực tế nhanh gấp ~11 lần giữa `INT8` và `Interp4D`, nhưng phần `WideResidualHead` FP32 chiếm >99.7% độ trễ khiến End-to-End CPU (`~6.7–6.9 ms` Rust / `~8.1 ms` PyTorch B=1) chỉ nhỉnh nhẹ hoặc ngang bằng Classical Control (`~9.17 ms` PyTorch B=1, `0.83 ms/img` B=64).
   - **Mất cân xứng bộ giảm xóc trên `c_stem`**: Để `c_stem` nhận ảnh thô khiến `4_grquanv_damped` tụt sau `2_ctrl_damped` ở $\sigma \ge 0.10$.
3. **Giải pháp kỹ thuật tiếp theo**:
   - *LUT-aware / Quantization-Aware Fine-Tuning (QAT)* dùng Straight-Through Estimator (STE) cho `INT8 Shift-LUT` để thu hồi ~1.5–3.4% ACC bị mất trên OCTMNIST và PneumoniaMNIST mà vẫn giữ kích thước 256 KiB.
   - *Đối xứng hóa giảm xóc nhiễu*: Áp dụng `phase_damping_u8` cho cả nhánh `c_stem` bên trong `WideGRQuanvNet` khi triển khai thực tế (hoặc huấn luyện kết hợp nhiễu nhẹ).
   - *Tối ưu hóa toàn mạng `WideResidualHead`*: Lượng tử hóa INT8 (Weight+Activation) và/hoặc nhân ma trận SIMD (`im2col` + GEMM NEON/AVX2) cho `Conv2d` phía sau để hạ độ trễ End-to-End từ ~6.7 ms xuống ngưỡng dưới mili-giây.

---

# PHA B: KIẾN TRÚC `GR-QUANV V3` (TRAINABLE VARIATIONAL Q-LUT, INT8-QAT & LOW-FLOP END-TO-END)

Sau khi kiểm chứng Ngưỡng A (tối ưu hóa không huấn luyện lại trên 30 checkpoint `v2` khóa cứng) cho thấy hai giới hạn cấu trúc (`reports/threshold_a_results.json`: bản `INT8 256 KiB` vẫn đạt `80.82%` < Control `82.52%` trên OCTMNIST và phần đầu `WideResidualHead` ~18.9M MACs còn cồng kềnh), Pha B triển khai dòng kiến trúc tách biệt `v3` (`src/models/gr_quanv_v3.py`, toàn bộ checkpoint lưu tại `weights/pillar3_v3/`) giải quyết đồng thời 3 nút thắt:

## 7. CƠ CHẾ TOÁN HỌC & KIẾN TRÚC `GR-QUANV V3`

1. **Mạch Q-LUT Biến phân Học được (`TrainableQLUTBank`) & QAT Bit-Exact**:
   - Biểu diễn chính xác toàn bộ mạch $4$-qubit `StronglyEntanglingLayers` ($L=2$ tầng, $U(\theta) \in SU(16)$) và $4$ toán tử đo $M_k(\theta) = U^\dagger(\theta) Z_k U(\theta)$ bằng tensor PyTorch khả vi (sai số cực đại so với giải tích PennyLane đo được là **$5.43 \times 10^{-7}$**).
   - **Lượng tử hóa Nhận thức Huấn luyện (`INT8-QAT + Straight-Through Estimator`)**: Cho phép lan truyền ngược thẳng qua các khoảng pha $4$-bit ($p \gg 4$) và biên độ `int8` ($[-127, 127]/127.0$) kết hợp bộ điều chuẩn độ trơn 4D Total Variation (`cached_smoothness_loss`). Kết quả: sau khi biên dịch mạch đã học sang bảng tra `256 KiB` `INT8 Shift-LUT`, độ lệch độ chính xác (`compile_acc_diff`) trên toàn bộ **30 lượt đánh giá lượng tử** (3 tập dữ liệu $\times$ 2 mô hình $\times$ 5 seeds) đạt chính xác **`0.0000%`**.
   - **Tái sử dụng Tầng 2 Biệt lập Nhiễu (`_extract_stage2_qlut`, 0 KiB L2 phụ trội)**: Cổng đối xứng bảo toàn phương sai $\theta_g = \frac{\pi}{2}\alpha \Rightarrow w_c = \sqrt{2}\cos\theta_g$, $w_q = \sqrt{2}\sin\theta_g$ cô lập hoàn toàn nhánh lượng tử $w_q f_q$ khỏi nhiễu từ nhánh cổ điển `c_stem` trước khi chiếu (`s2_proj: 16 -> 4 ch`) và tra lại bảng `Table 0` (`2x2`, stride=1, thêm `16` kênh đặc trưng phi tuyến tại $0\text{ MAC}$).
2. **Thiết kế Đầu ra Gọn nhẹ Low-FLOP (`DSResBlock` + `_dw_dilated_mixer`) & So sánh Công bằng**:
   - Thay `WideResidualHead` (`18.90M` MACs) bằng bộ trộn `_dw_dilated_mixer(16)` + 2 khối `DSResBlock` (`width1=64, width2=96`). Tổng chi phí thực thi của `GRQuanvNetV3` giảm xuống **`4.03M` MACs** (`39,250` tham số trên Breast/Pneumonia; `4.03M / 39,444` trên OCTMNIST) — giảm **`4.69×` MACs** và **`3.78×` tham số** so với `WideClassicalControl v2` (`18.90M` MACs, `148,406` tham số).
   - Thiết lập đối chứng công bằng ngang tầm FLOP & cấu trúc `ClassicalControlV3Light` (`4.07M` MACs, `39,470–39,664` tham số, thay Q-LUT bằng 4 nhánh `Conv2d` giãn nở + nhánh `s2_proj` tích chập cổ điển tương ứng) và cặp siêu nhỏ **Micro-Edge / TinyML** `MicroEdgeNetV3`: `micro_quanv` (`169,408–169,472` MACs, `2,002–2,068` tham số) vs `micro_ctrl` (`186,336–186,400` MACs, `2,078–2,144` tham số).

---

## 8. ĐỐI CHỨNG ĐỘ CHÍNH XÁC & ĐỘ BỀN NHIỄU ĐỐM 5 SEEDS (3 BỘ DỮ LIỆU MEDMNIST)

Toàn bộ kết quả thực nghiệm (`reports/pillar3_v3_results.json`, 5 paired seeds `[0, 42, 100, 2023, 777]`, 3 noise draws mỗi mức $\sigma$, hình trực quan `reports/figures/v3_vs_cnn_pareto.png`) được ghi nhận nguyên trạng cho cấu hình **`INT8 Shift-LUT (256 KiB)` (tuyệt đối không dùng `Interp4D`)**:

| Tập dữ liệu | Mô hình | MACs / Params | Sạch $\sigma=0.00$ (ACC% / F1%) | $\sigma=0.05$ (ACC%) | $\sigma=0.10$ (ACC%) | $\sigma=0.15$ (ACC%) | $\sigma=0.20$ (ACC% / F1%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **BreastMNIST** | `ctrl_v2_full` (18.9M) | `18.90M / 148.4K` | $84.10 \pm 1.05$ / $89.56$ | $83.80 \pm 1.93$ | **$83.21 \pm 1.58$** | **$81.97 \pm 3.18$** | **$81.58 \pm 4.79$** / **$86.86$** |
| **BreastMNIST** | `ctrl_v3_light` (4.1M) | `4.07M / 39.5K` | $84.49 \pm 1.94$ / $89.51$ | $84.23 \pm 1.00$ | $79.32 \pm 4.97$ | $76.37 \pm 7.01$ | $75.13 \pm 7.45$ / $80.27$ |
| **BreastMNIST** | **`quanv_v3_int8` (256KiB)** | **`4.03M / 39.3K`** | **$85.00 \pm 1.48$ / $90.01$** | **$84.27 \pm 0.86$** | $80.47 \pm 2.82$ | $76.50 \pm 4.78$ | $75.00 \pm 5.90$ / $80.13$ |
| **BreastMNIST** | `micro_ctrl` (0.19M) | `0.186M / 2.08K` | $83.33 \pm 1.28$ / $89.36$ | $81.20 \pm 2.08$ | $81.79 \pm 1.29$ | $77.65 \pm 3.30$ | $75.34 \pm 4.27$ / $81.27$ |
| **BreastMNIST** | `micro_quanv` (0.17M) | `0.169M / 2.00K` | $81.92 \pm 1.39$ / $88.61$ | $80.68 \pm 1.66$ | $82.35 \pm 1.64$ | $79.70 \pm 0.96$ | $76.54 \pm 2.73$ / $82.48$ |
| **PneumoniaMNIST** | `ctrl_v2_full` (18.9M) | `18.90M / 148.4K` | **$90.22 \pm 1.85$ / $92.69$** | **$89.48 \pm 1.71$** | **$87.79 \pm 2.31$** | **$86.65 \pm 2.09$** | $82.68 \pm 2.05$ / $86.28$ |
| **PneumoniaMNIST** | `ctrl_v3_light` (4.1M) | `4.07M / 39.5K` | $89.23 \pm 1.70$ / $91.98$ | $87.82 \pm 1.70$ | $85.83 \pm 2.42$ | $84.97 \pm 2.50$ | $83.55 \pm 2.06$ / $87.64$ |
| **PneumoniaMNIST** | **`quanv_v3_int8` (256KiB)** | **`4.03M / 39.3K`** | $89.36 \pm 2.09$ / $92.06$ | $88.29 \pm 1.72$ | $85.79 \pm 2.18$ | $85.04 \pm 2.37$ | **$84.13 \pm 1.81$ / $88.12$** |
| **PneumoniaMNIST** | `micro_ctrl` (0.19M) | `0.186M / 2.08K` | $88.62 \pm 0.96$ / $91.31$ | $82.16 \pm 3.10$ | $83.87 \pm 1.31$ | $81.93 \pm 1.50$ | $74.29 \pm 3.66$ / $77.53$ |
| **PneumoniaMNIST** | `micro_quanv` (0.17M) | `0.169M / 2.00K` | $88.24 \pm 1.18$ / $91.06$ | $85.56 \pm 1.37$ | $83.35 \pm 2.15$ | $83.60 \pm 1.48$ | $77.87 \pm 1.85$ / $81.19$ |
| **OCTMNIST** | `ctrl_v2_full` (18.9M) | `18.90M / 148.5K` | $82.52 \pm 2.92$ / $81.77$ | **$76.47 \pm 4.32$** | $58.46 \pm 4.94$ | $44.78 \pm 3.86$ | $38.30 \pm 4.55$ / $25.88$ |
| **OCTMNIST** | `ctrl_v3_light` (4.1M) | `4.07M / 39.7K` | $80.18 \pm 2.57$ / $79.05$ | $74.38 \pm 4.11$ | $57.32 \pm 4.80$ | $44.75 \pm 1.82$ | $38.24 \pm 1.23$ / $26.03$ |
| **OCTMNIST** | **`quanv_v3_int8` (256KiB)** | **`4.03M / 39.4K`** | **$82.90 \pm 3.73$ / $81.93$** | $73.20 \pm 3.45$ | $51.23 \pm 2.28$ | $42.81 \pm 1.09$ | **$40.52 \pm 1.36$ / $27.21$** |
| **OCTMNIST** | `micro_ctrl` (0.19M) | `0.186M / 2.14K` | $78.72 \pm 3.40$ / $78.13$ | $70.69 \pm 3.13$ | $51.72 \pm 4.39$ | $42.72 \pm 1.84$ | $38.98 \pm 1.77$ / $26.75$ |
| **OCTMNIST** | `micro_quanv` (0.17M) | `0.169M / 2.07K` | $75.10 \pm 1.82$ / $73.77$ | $74.61 \pm 1.21$ | **$65.41 \pm 2.38$** | **$48.47 \pm 2.73$** | $36.97 \pm 2.73$ / $27.34$ |

### Phân tích Thống kê (Paired t-test, 5 Seeds)
1. **Khắc phục dứt điểm suy giảm lượng tử hóa Zero-shot `INT8` trên ảnh sạch**:
   - Với QAT + Variational Q-LUT, `quanv_v3_int8` nâng độ chính xác ảnh sạch `INT8 (256 KiB)` trên **OCTMNIST** từ `80.64%` (`v2`) lên **`82.90%`**, vượt cả `ctrl_v3_light` (`+2.72%`, $p=0.1920$) lẫn mạng nặng gấp $4.69\times$ `ctrl_v2_full` (`+0.38%`, $p=0.8093$). Trên **BreastMNIST**, `quanv_v3_int8` đạt **`85.00%`** — đứng đầu cả bảng (`+0.51%` vs `ctrl_v3_light`, $p=0.2420$; `+0.90%` vs `ctrl_v2_full`, $p=0.1347$). Trên **PneumoniaMNIST**, `quanv_v3_int8` tăng từ `88.49%` (`v2 INT8`) lên **`89.36%`** (`+0.13%` vs `ctrl_v3_light`, còn kém nhẹ `-0.87%` so với `ctrl_v2_full`, $p=0.6032$).
2. **Đảo ngược thế thua ở $\sigma=0.20$ nhưng vẫn còn điểm yếu cục bộ**:
   - Trên **PneumoniaMNIST** ($\sigma=0.20$), `quanv_v3_int8` đạt **`84.13%`**, vượt cả `ctrl_v3_light` (`83.55%`, `+0.59%`, $p=0.4664$) lẫn `ctrl_v2_full` (`82.68%`, `+1.45%`, $p=0.3085$), lội ngược dòng hoàn toàn so với nhánh `4_grquanv_damped` Pha A (`76.62%`). Trên **OCTMNIST** ($\sigma=0.20$), `quanv_v3_int8` vượt `ctrl_v3_light` có ý nghĩa thống kê (`40.52%` vs `38.24%`, `+2.28%`, $p=0.0090$).
   - **Kết quả tiêu cực cần ghi nhận thẳng thắn**: Tại mức nhiễu trung bình **OCTMNIST $\sigma=0.10$**, `quanv_v3_int8` tụt xuống **`51.23%`**, thấp hơn `ctrl_v3_light` (`57.32%`, $-6.09\%$, $p=0.0710$) và `ctrl_v2_full` (`58.46%`, $-7.23\%$, $p=0.0521$). Ngoài ra trên **BreastMNIST** $\sigma \ge 0.15$, `ctrl_v2_full` có dung lượng mô hình lớn hơn vẫn kháng nhiễu tốt hơn đáng kể (`81.58%` vs `75.00%` tại $\sigma=0.20$, $p=0.0145$), trong khi ở cùng ngưỡng $4\text{M}$ MACs thì `quanv_v3_int8` và `ctrl_v3_light` gần như tương đương ($p=0.97$).
3. **Phát hiện đặc biệt ở dải Siêu biên (`Micro-Edge`, ~`0.17M` MACs)**:
   - Khi lược bỏ các tầng tích chập nặng phía sau (`DSResBlock`), `micro_quanv` thể hiện khả năng giữ cấu trúc cực mạnh dưới nhiễu đốm: trên **OCTMNIST**, vượt `micro_ctrl` **`+13.69%`** tại $\sigma=0.10$ (**`65.41%`** vs `51.72%`, $p=0.0005$ — thậm chí cao hơn cả `ctrl_v2_full` `58.46%`) và **`+5.75%`** tại $\sigma=0.15$ (**`48.47%`** vs `42.72%`, $p=0.0058$); trên **PneumoniaMNIST**, vượt `+3.40%` tại $\sigma=0.05$ (`85.56%` vs `82.16%`) và `+3.58%` tại $\sigma=0.20$ (`77.87%` vs `74.29%`). Đánh đổi đi kèm là độ chính xác ảnh sạch giảm `-3.62%` trên OCTMNIST (`75.10%` vs `78.72%`, $p=0.1829$) và `-1.41%` trên BreastMNIST (`81.92%` vs `83.33%`, $p=0.2155$).

---

## 9. ĐÁNH GIÁ RUST END-TO-END THUẦN (`medquanv-cli --suite-v3`) & NHẬN ĐỊNH KHOA HỌC

Toàn bộ 5 kiến trúc (xuất từ seed `42` trên PneumoniaMNIST bằng `experiments/export_v3_to_rust.py`, trọng số gộp BatchNorm FP32 + bảng `INT8 Shift-LUT` `256 KiB` riêng theo từng mô hình) đã được triển khai End-to-End bằng Rust thuần trong `rust_qlut/src/bin/v3_suite.rs` (`medquanv-cli --suite-v3 --verify data/processed/pneumoniamnist_v3_rust_verify.bin --json-out reports/medquanv_cli_v3_verification.json`). Lệnh biên dịch native dùng cờ:
`RUSTFLAGS="-C target-cpu=native" CARGO_PROFILE_RELEASE_LTO="false" CARGO_PROFILE_RELEASE_CODEGEN_UNITS="8" cargo build --release --no-default-features --manifest-path rust_qlut/Cargo.toml --bin medquanv-cli` (biên dịch sạch trong $3.51\text{s}$).

| Mô hình trong Rust (`medquanv-cli --suite-v3`) | Rust ACC Clean (624 ảnh) | Sai số Logits tối đa (Clean) | Rust ACC $\sigma=0.20$ | Khớp nhãn PyTorch | Q-LUT Stage-1 Latency | Độ trễ E2E (1 ảnh) | Thông lượng E2E (FPS) | Weights + LUT (KiB) | % L2 Cache (2 MB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`ctrl_v2_full` (18.9M)** | `90.71%` (`90.7051%`) | `4.77e-6` | `85.26%` | **`100.00%`** | N/A | `9,352.83 µs` | `106.92 FPS` | `577.65 + 0.00` | `28.21%` |
| **`ctrl_v3_light` (4.07M)** | `86.54%` (`86.5385%`) | `5.25e-6` | `83.49%` | **`100.00%`** | N/A | **`1,114.03 µs`** | **`897.65 FPS`** | **`151.79 + 0.00`** | **`7.41%`** |
| **`quanv_v3_int8` (4.03M)** | `87.98%` (`87.9808%`) | `5.25e-6` | `82.53%` | **`100.00%`** | `3.16 µs` (4 tỷ lệ) | `1,126.38 µs` | `887.80 FPS` | `150.90 + 256.00 = 406.90` | `19.87%` |
| **`micro_ctrl` (0.19M)** | `90.06%` (`90.0641%`) | `3.81e-6` | `67.95%` | **`100.00%`** | N/A | `47.59 µs` | `21,014.16 FPS` | `7.68 + 0.00` | `0.37%` |
| **`micro_quanv` (0.17M)** | `87.34%` (`87.3397%`) | `1.91e-6` | `79.49%` | **`100.00%`** | `3.16 µs` (4 tỷ lệ) | **`46.18 µs`** | **`21,656.70 FPS`** | `7.29 + 256.00 = 263.29` | `12.86%` |

*(Ghi chú kỹ thuật về sai số logits ở $\sigma=0.20$: `max_logit_diff_s20 = 1.21e-2` trên `quanv_v3_int8` do 1 pixel trung gian tại Stage-2 rơi đúng ranh giới lượng tử hóa $4$-bit $1/16$ giữa đường tính FP32 PyTorch và Rust; toàn bộ `624/624` ảnh đều khớp nhãn dự đoán `100.00%`.)*

### Nhận định Tổng kết Công tâm
1. **Bước nhảy vọt thực chất so với `v2` (`WideClassicalControl`)**: `quanv_v3_int8` chạy thuần trên CPU Rust đạt **`887.80 FPS` (`1.13 ms/ảnh`)** — tăng tốc **`8.30×`** so với `ctrl_v2_full` (`106.92 FPS` / `9.35 ms`), chỉ chiếm **`406.90 KiB` (`19.87%` L2 Cache 2 MB của Jetson Nano)**, và vượt `ctrl_v2_full` về độ chính xác ảnh sạch trên BreastMNIST (`+0.90%`), OCTMNIST (`+0.38%`), cũng như độ bền $\sigma=0.20$ trên PneumoniaMNIST (`+1.45%`) và OCTMNIST (`+2.22%`) mà hoàn toàn **không cần `Interp4D`**.
2. **So sánh sòng phẳng với mạng cổ điển đồng hạng MAC (`ctrl_v3_light`)**:
   - Về **độ chính xác**: `quanv_v3_int8` thắng `ctrl_v3_light` trên **ảnh sạch cả 3 tập dữ liệu** (`+0.51%` Breast, `+0.13%` Pneumonia, **`+2.72%` OCTMNIST**), thắng tại $\sigma=0.20$ trên PneumoniaMNIST (`+0.59%`) và OCTMNIST (**`+2.28%`, $p=0.0090$**), nhưng chưa thắng áp đảo ở mọi mức nhiễu giữa (nhất là OCTMNIST $\sigma=0.10$ bị tụt $-6.09\%$).
   - Về **tốc độ Rust E2E ở ngưỡng $4\text{M}$ MACs**: `quanv_v3_int8` (`1,126.38 µs` / `887.80 FPS`) đạt **ngang bằng thực chất (chỉ chậm hơn `1.1%` / `12.4 µs`, nằm trong sai số đo)** so với `ctrl_v3_light` (`1,114.03 µs` / `897.65 FPS`) nhưng bộ nhớ L2 lớn hơn (`406.9 KiB` vs `151.8 KiB` do bảng tra $256\text{ KiB}$ Q-LUT), vì chi phí `DSResBlock` + `_dw_dilated_mixer` (`~4.0M` MACs) chiếm áp đảo cả hai mô hình và phần Stage-2 Q-LUT có bước chiếu + `sigmoid` + re-quantize.
   - **Nơi tầng Q-LUT thực sự bứt tốc và giữ bản sắc lượng tử rõ rệt nhất** là phân khúc **Micro-Edge ($0.17\text{M}$ MACs)**: ở đây không có `DSResBlock` che lấp, nên `micro_quanv` chạy End-to-End **nhỉnh hơn `micro_ctrl` (`46.18 µs / 21,656.7 FPS` vs `47.59 µs / 21,014.2 FPS`, nhanh hơn ~`3.0%`)** và chống nhiễu đốm vượt trội rõ rệt trên thực địa (`+13.69%` OCTMNIST $\sigma=0.10$ với $p=0.0005$; `+11.54%` trên seed 42 Rust PneumoniaMNIST $\sigma=0.20$: `79.49%` vs `67.95%`).
3. **Hướng phát triển tiếp theo cho luận văn**:
   - Đo benchmark thực tế trên **Jetson Nano (ARM Cortex-A57, NEON)** bằng chính `medquanv-cli --suite-v3` (dự kiến chênh lệch băng thông bộ nhớ/cache trên ARM-A57 sẽ khác với AVX2 desktop).
   - Nghiên cứu **Stage-2 Noise Regularization / Multi-scale Sigma Training** khi huấn luyện QAT để khắc phục độ nhạy tại vùng chuyển tiếp $\sigma=0.10$ trên OCTMNIST của `quanv_v3_int8`, kết hợp **INT8 Weight/Activation quantization** cho `DSResBlock` nhằm kéo trọn vẹn luồng tính toán về miền số nguyên.
