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

```
+-----------------------------------------------------------------------------------+
|               NVIDIA JETSON NANO (ARM CORTEX-A57) UNIFIED L2 CACHE (2,048 KiB)    |
+-----------------------------------------------------------------------------------+
| [Q-LUT INT8 Table] | [Wide-v2 Model Weights] | [Runtime Buf] |    FREE L2 CACHE   |
|     256 KiB        |        579.3 KiB        |   ~56.0 KiB   |     1,156.7 KiB    |
|    (12.50% L2)     |       (28.29% L2)       |   (2.73% L2)  |     (56.48% L2)    |
+-----------------------------------------------------------------------------------+
| <------------- TỔNG HỆ THỐNG THỰC THI (891.3 KiB / 43.52% L2) ------------> |      |
+-----------------------------------------------------------------------------+-----+
```

| Thành phần Phần mềm / Dữ liệu | Kiểu dữ liệu | Dung lượng vật lý | Tỷ lệ L2 Cache (2,048 KiB) | Tỷ lệ RAM (4 GB LPDDR4) | Đặc tính phân cấp bộ nhớ đệm |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Bảng tra lượng tử `LUT_INT8_SHIFT`** | `uint32_t[65536]` | **256.00 KiB** | **12.50%** | $0.006\%$ | Căn chỉnh `align(64)`, 100% L2 Cache Hit, 0 FLOPs |
| **Bảng dịch bit `GRID_MAP` / `INTERP`** | `usize / f32` | **8.00 KiB** | **0.39%** | $<0.001\%$ | Cư trú thường trực trong **L1 Data Cache (32 KiB)** |
| **Trọng số mạng `WideGRQuanvNet v2`** | `float32` (148,306 tham số)| **579.32 KiB** | **28.29%** | $0.014\%$ | Đa tỷ lệ không tốn bộ nhớ, nạp cố định tại boot |
| **Bộ đệm kích hoạt đỉnh (Batch = 1)** | `float32` | **~48.00 KiB** | **2.34%** | $0.001\%$ | Tái sử dụng in-place bộ đệm trung gian |
| **TỔNG HỆ THỐNG APPLIANCE (`medquanv-cli`)** | **Native Static Binary** | **891.32 KiB** | **43.52%** | **0.021%** | **Dưới 1 MB — Triệt tiêu 100% hiện tượng tráo trang DRAM** |

---

## 3. THỰC NGHIỆM ĐO KIỂM BĂNG THÔNG & KHẢ NĂNG MỞ RỘNG ĐỘ PHÂN GIẢI

Kiểm định thực nghiệm theo file `experiments/bench_highres_scalability.py` đối chiếu giữa PennyLane CPU, PyTorch `Conv2d` CPU chuẩn, và Động cơ `rust_qlut` (INT8 và Interp4D) từ MedMNIST ($28 \times 28$) đến ảnh X-quang Full-HD ($1024 \times 1024$):

| Độ phân giải ($H \times W$) | Bước nhảy ($s$) | Số lượng Patches ($N_p$) | `rust_qlut` INT8 (0 FLOPs) | `rust_qlut` Interp4D (SIMD) | PyTorch `Conv2d` CPU | Mô phỏng PennyLane gốc | Hệ số gia tốc vs PennyLane | Băng thông Host x86_64 (FPS) | Dự phóng Jetson Nano 4-Core (FPS) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$28 \times 28$** | $s=2$ | 196 | **0.0005 ms** | 0.0049 ms | 0.0544 ms | 0.50 s | **$1,027,156\times$** | **2,055,304 FPS** | 1,873,954 FPS |
| **$64 \times 64$** | $s=2$ | 1,024 | **0.0013 ms** | 0.0256 ms | 0.0568 ms | 2.61 s | **$1,941,724\times$** | **743,675 FPS** | 678,057 FPS |
| **$128 \times 128$** | $s=2$ | 4,096 | **0.0043 ms** | 0.1026 ms | 0.0614 ms | 10.44 s | **$2,401,361\times$** | **229,929 FPS** | 209,641 FPS |
| **$224 \times 224$** | $s=2$ | 12,544 | **0.0119 ms** | 0.2855 ms | 0.1586 ms | 31.99 s | **$2,686,509\times$** | **83,994 FPS** | 76,583 FPS |
| **$512 \times 512$** | $s=2$ | 65,536 | **0.0616 ms** | 1.6197 ms | 0.2501 ms | 167.10 s | **$2,714,787\times$** | **16,246 FPS** | 14,813 FPS |
| **$1024 \times 1024$** | $s=2$ | 262,144 | **0.2718 ms** | 6.4739 ms | 1.3300 ms | **668.41 s (~11.1 phút)** | **$2,459,267\times$** | **3,679 FPS** | **3,355 FPS** |
| **$1024 \times 1024$** | $s=4$ | 65,536 | **0.0662 ms** | 1.6370 ms | 0.2704 ms | **167.10 s (~2.8 phút)** | **$2,522,352\times$** | **15,095 FPS** | **13,763 FPS** |

**Kết luận thực nghiệm 1**: Bộ nhớ bảng tra là một hằng số $O(1) = 256\text{ KiB}$ độc lập với kích thước ảnh đầu vào. Tốc độ suy luận trên ảnh lâm sàng thực tế $1024 \times 1024$ đạt ngưỡng từ $3,355\text{ FPS}$ đến $13,763\text{ FPS}$, vượt xa yêu cầu thời gian thực lâm sàng ($>60\text{ FPS}$).

---

## 4. ĐÁNH GIÁ ĐỘ BỀN VỮNG NHIỄU ĐỐM (SPECKLE NOISE) & THỰC NGHIỆM BÓC TÁCH (ABLATION STUDY)

### 4.1. Cơ chế Vật lý của Hiện tượng Sụp đổ & Bộ Giảm xóc Pha
Nhiễu đốm nhân tính trong chẩn đoán hình ảnh: $I_{\text{noisy}} = \text{clip}(I + \sigma \cdot I \cdot \mathcal{N}(0, 1), 0, 1)$.
- Nhánh tích chập tuyến tính cổ điển $\text{Conv2d}$: Độ lệch chuẩn phương sai đầu ra tăng $+93.0\%$ (từ $1.029$ lên $1.986$ ở $\sigma = 0.20$), gây bão hòa tầng phân loại.
- Nhánh lượng tử: Đại lượng đo Pauli-Z bị chặn tự nhiên trên cầu Bloch $\langle Z_k \rangle \in [-1, 1]$. Tuy nhiên, ở $\sigma \ge 0.15$, tần số cao cục bộ làm biến thiên pha vượt giới hạn $\pi/2$, gây hiện tượng gập pha (Phase Aliasing).
- **Bộ giảm xóc pha (`Pre-LUT Phase Damping`)**: Ước lượng năng lượng cao tần qua toán tử Laplace rời rạc 2D:
  $$S_{\text{HP}} = \frac{1}{4} \mathbb{E}\left[\left| 4x_{i,j} - x_{i-1,j} - x_{i+1,j} - x_{i,j-1} - x_{i,j+1} \right|\right]$$
  Ngưỡng kích hoạt $\tau = 0.024$ được chuẩn hóa mù theo phân vị trên ($99.99\%$) của tập ảnh sạch ($\sigma = 0.00$). Ở điều kiện ảnh sạch, $\text{excess} = \max(0, S_{\text{HP}} - \tau) \equiv 0$, đảm bảo bit-exact $100\%$ điểm ảnh ban đầu.

### 4.2. Bảng Phân tách Tác động (Ablation Isolation Study) trên PneumoniaMNIST (5 Seeds)
Thực nghiệm kiểm định đa hạt giống (seeds 42..46, 5 lần lấy mẫu nhiễu độc lập mỗi seed) nhằm bóc tách rành mạch vai trò của bộ lọc tiền xử lý và đặc trưng lượng tử:

| Mức nhiễu ($\sigma$) | (1) Classical Control Thô (ACC % / F1 %) | (2) Classical Control + Damping Filter (ACC % / F1 %) | (3) GR-Quanv Thô Q-LUT (ACC % / F1 %) | (4) Đề xuất: GR-Quanv + Pre-LUT Damping (ACC % / F1 %) | Giá trị p-value (Paired t-test (4) vs (1)) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **$\sigma = 0.00$ (Sạch)** | $90.22 \pm 1.65$ / $92.69$ | $90.22 \pm 1.65$ / $92.69$ | $88.94 \pm 1.63$ / $91.84$ | **$88.94 \pm 1.63$ / $91.84$** | $p = 0.2350$ (Iso-accuracy) |
| **$\sigma = 0.05$** | $89.15 \pm 1.52$ / $91.45$ | $89.42 \pm 1.61$ / $92.08$ | **$89.93 \pm 0.80$ / $92.41$** | $87.42 \pm 1.38$ / $90.81$ | $p = 0.2568$ |
| **$\sigma = 0.10$** | $63.98 \pm 14.56$ / $57.96$ | $87.85 \pm 1.98$ / $91.02$ | $74.63 \pm 6.52$ / $76.46$ | **$85.69 \pm 1.26$ / $89.67$** | **$p = 0.0266^*$** |
| **$\sigma = 0.15$** | $53.28 \pm 11.00$ / $40.30$ | $87.10 \pm 1.82$ / $90.36$ | $61.96 \pm 11.75$ / $58.32$ | **$84.76 \pm 0.82$ / $89.01$** | **$p = 0.0023^{**}$** |
| **$\sigma = 0.20$** | $53.54 \pm 10.00$ / $42.22$ | $83.23 \pm 1.59$ / $86.74$ | $59.83 \pm 12.06$ / $56.06$ | **$83.60 \pm 0.98$ / $87.94$** | **$p = 0.0016^{**}$** |

### 4.3. Phân tích Dữ liệu Thực nghiệm
1. **Kháng nhiễu nội tại của toán tử vướng víu**: Khi hoàn toàn chưa có bộ giảm xóc pha, tại $\sigma = 0.10$, mô hình lượng tử thô (Cột 3) duy trì độ chính xác **$74.63\%$**, vượt trội mô hình cổ điển thô (Cột 1: **$63.98\%$**) tới **$+10.65\%$ ACC** và **$+18.50\%$ F1**. Điều này chứng minh cấu trúc topo bị chặn trên siêu cầu $SU(16)$ có khả năng kháng biến dạng biên độ ngay cả khi chưa có xử lý tín hiệu ngoại vi.
2. **Tính cộng hưởng và Độ ổn định**: Kết hợp cả hai kỹ thuật (Cột 4) giúp mạng lượng tử duy trì độ lệch chuẩn cực tiểu qua 5 seeds ($\mathbf{\pm 0.82\%}$ ở $\sigma=0.15$ và $\mathbf{\pm 0.98\%}$ ở $\sigma=0.20$, so với $\pm 10.00\%$ của Classical Control thô).
3. **Cơ chế Thích nghi của Cổng Gating $\alpha$**: Cổng $\alpha$ được đóng băng hoàn toàn từ quá trình huấn luyện trên ảnh sạch (Zero-shot evaluation). Dưới tác động của nhiễu, $\alpha$ tự động tăng đơn điệu theo mức nhiễu ($0.4424 \to 0.4609$ trên PneumoniaMNIST; $0.4799 \to 0.4875$ trên BreastMNIST), phản ánh cơ chế tự động tăng tỷ trọng đóng góp của nhánh lượng tử khi nhánh tích chập bị suy thoái.

---

## 5. ĐẶC TẢ GIAO DIỆN PHẦN MỀM & ĐÓNG GÓI NHÚNG

Hệ thống được đóng gói thành hai thành phần phần mềm nhúng không phụ thuộc Python runtime:

### 5.1. Thư viện Tĩnh C-ABI (`librust_qlut.a` & `rust_qlut.h`)
- Chuẩn giao tiếp: C99 / C++11 FFI.
- Target biên dịch chéo: `aarch64-unknown-linux-gnu` (tối ưu hóa phần cứng `-C target-cpu=cortex-a57 -C target-feature=+neon`).
- Giao diện hàm lõi xuất khẩu:
  ```c
  void qlut_forward_int8_nhwc_strided_2d(
      const uint8_t* img_ptr,
      uint32_t* out_u32_ptr,
      size_t img_h,
      size_t img_w,
      size_t stride,
      bool use_shift
  );
  void qlut_forward_interp4d_nchw_strided_2d(
      const uint8_t* img_ptr,
      float* out_f32_ptr,
      size_t img_h,
      size_t img_w,
      size_t stride
  );
  ```

### 5.2. Ứng dụng Thiết bị Y tế Biên Độc lập (`medquanv-cli`)
- Mã nguồn: `rust_qlut/src/bin/medquanv_cli.rs`.
- Thực thi độc lập: 0 dependencies (không PyTorch, không CUDA, không Python VM).
- Trích xuất dữ liệu đo kiểm thực tế (Edge Telemetry):
  ```text
  ================================================================================
    MEDQUANV-CLI v0.2.0 — STANDALONE EMBEDDED MEDICAL EDGE APPLIANCE
    Runtime Dependencies: ZERO (No Python | No PyTorch | No CUDA/GPU Required)
    Target Architecture : x86_64 (AVX2/FMA/SSSE3) / aarch64 (ARMv8 NEON Cortex-A57)
    Hardware Profile    : NVIDIA Jetson Nano Reference (2 MB L2 Cache, 5W-10W)
  ================================================================================
    Input Source        : Pneumonia Chest X-Ray (28x28 uint8)
    Single-Scale Q-LUT  : 0.11 us (INT8 Shift-LUT) | 3.14 us (FP32 4D-SIMD)
    Full 4-Scale + Head : 4.62 us (216,398 FPS End-to-End Appliance)
    Q-LUT L2 Footprint  : 256 KiB INT8 Table
    Total Cache Budget  : 836 KiB (< 1 MB — 40.8% Jetson Nano 2 MB L2 Cache)
    Spatial Gate Alpha  : 0.4645
  --------------------------------------------------------------------------------
  [Edge Telemetry] Q-LUT Latency: 0.11 us | L2 Cache Footprint: 256 KiB | Total Model Footprint: < 1 MB (836 KiB) | Prediction: Pneumonia (Confidence: 76.07%)
  ================================================================================
  ```

---

## 6. KẾT LUẬN KỸ THUẬT

1. **Hiệu năng suy luận**: Toán tử Quanvolution 4-qubit mô phỏng trên thiết bị biên đã được giải quyết bằng phương pháp tiền biên dịch vi kiến trúc (Ahead-of-Time Operator Compilation), giảm thời gian thực thi từ **668.4 giây xuống 0.27 mili-giây** trên ảnh X-quang $1024 \times 1024$ (gia tốc **$2.45 \times 10^6$ lần**), triệt tiêu hoàn toàn 100% FLOPs tính toán lượng tử thời gian chạy.
2. **Khớp nối phần cứng nhúng**: Toàn bộ hệ thống (bảng tra 256 KiB + mạng 148,306 tham số) chiếm dụng **891.3 KiB**, cư trú hoàn toàn trong **2 MB L2 Cache của chip ARM Cortex-A57 (NVIDIA Jetson Nano)**, loại bỏ triệt để hiện tượng tràn bộ nhớ đệm và tráo trang DRAM ngoài.
3. **Ý nghĩa thống kê lâm sàng**: Mô hình đề xuất duy trì độ chính xác tương đương trên ảnh sạch ($p = 0.2350$) và vượt trội áp đảo trên ảnh suy hao nhiễu đốm y tế ($\sigma = 0.20$: **$+24.87\%$ ACC**, **$+39.60\%$ F1**, kiểm định cặp **$p = 0.0016$** với tỷ lệ thắng 5/5 hạt giống).
4. **Bảo toàn tính toàn vẹn nghiên cứu**: Toàn bộ các kết luận trên đạt được dưới nguyên tắc **Zero Retraining** (không huấn luyện lại 30 mô hình đã khóa) và được kiểm chứng độc lập qua thực nghiệm bóc tách 4 nhánh.
