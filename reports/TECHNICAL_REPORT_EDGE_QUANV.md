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
