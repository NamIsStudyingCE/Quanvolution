# BÁO CÁO KỸ THUẬT: PHÂN TÍCH THỰC NGHIỆM LỚP TÍCH CHẬP LƯỢNG TỬ 4-QUBIT CỐ ĐỊNH (`Q-LUT`) VÀ ĐÁNH GIÁ TRIỂN KHAI BIÊN TRÊN MEDMNIST

- **Đề tài**: Nghiên cứu và ứng dụng lớp tích chập lượng tử (Quanvolution) trong phân loại ảnh y tế MedMNIST
- **Tác giả thực hiện**: Nguyễn Hoàng Nam (MSSV: 22520916) — Khoa Kỹ thuật Máy tính, Trường ĐH Công nghệ Thông tin (UIT - ĐHQG TP.HCM)
- **Cán bộ hướng dẫn**: ThS. Nguyễn Duy Xuân Bách
- **Phạm vi đo đạc hiện tại**: Toàn bộ kết quả suy luận thời gian thực trong báo cáo này được đo trực tiếp trên **CPU máy chủ `x86_64`** (không ngoại suy sang phần cứng khác; số đo trực tiếp trên **NVIDIA Jetson Nano B01** đang chờ bổ sung sau khi hoàn tất cấp nguồn và chạy đo trên bo mạch vật lý).

---

## 1. TÓM TẮT ĐÓNG GÓP & KHUNG ĐÁNH GIÁ KHOA HỌC

Báo cáo này giải quyết trực tiếp **Câu hỏi nghiên cứu A** của đề cương: *Lớp quanvolution 4-qubit cố định có thực sự trích xuất được đặc trưng lượng tử hữu ích hơn mạng tích chập cổ điển không, hay hiệu ứng quan sát được (nếu có) chỉ bắt nguồn từ phép biến đổi phi tuyến bị chặn trong $[-1, 1]$?*

1. **Bằng chứng thực nghiệm về tính tương đương bảng tra (Lookup-Table Equivalence):**
   Toàn bộ lớp quanvolution 4-qubit với mạch cố định ($L=2$ tầng `StronglyEntanglingLayers`) có thể biên dịch trước thành bảng tra 65,536 phần tử chiếm **`256.00 KiB`** (`INT8`) hoặc **`1,024.00 KiB`** (`FP32 Interp4D`, độ tương đồng cosine **`0.999993`** so với mô phỏng giải tích). Do đó, tại thời điểm suy luận **không còn phép tính cơ học lượng tử nào trong hệ thống** — mọi hiệu ứng hiệu năng (nếu có) đều xuất phát từ dạng hàm phi tuyến bị chặn $[-1, 1]$, trực tiếp minh chứng thực nghiệm cho luận điểm của *Bowles, Ahmed & Schuld* về các kiến trúc QML quy mô nhỏ có thể mô phỏng cổ điển tầm thường.
2. **Kết quả đối chứng mở rộng 7 nhánh & Kiểm định thống kê chặt chẽ (`345` kiểm định):**
   Khi so sánh công bằng trên 5 hạt giống (`[0, 42, 100, 2023, 777]`) với đầy đủ baseline cổ điển (`Conv2d`, `Conv2d + tanh [-1,1]`, `Conv2d + hard-clip [-1,1]`, `Conv2d + NoiseAug`, `CLUT-SepTrig`, `CLUT-RandMLP`), kết hợp kiểm định cặp, kiểm định tương đương **TOST ($\pm 2.0\%$, $\alpha=0.05$)** và hiệu chỉnh đa kiểm định **Benjamini–Hochberg (FDR)**, lớp `Q-LUT` cố định **không tạo ra lợi thế độ chính xác có ý nghĩa thống kê** trước các đối thủ cổ điển đồng hạng.
3. **Sản phẩm phần mềm benchmark tái lập:**
   Thư viện Rust `rust_qlut` (SIMD + giao diện C-ABI `librust_qlut.a` / `rust_qlut.h`), công cụ CLI kiểm chứng độc lập `medquanv-cli` (sai số logits tối đa $< 5.3 \times 10^{-6}$ so với PyTorch trên 624 ảnh test thật), và kịch bản tái lập một lệnh (`scripts/reproduce_all.py`).

---

## 2. MÔ TẢ TOÁN HỌC & CÀI ĐẶT BẢNG TRA (`rust_qlut`)

### 2.1. Toán tử 4-Qubit Cố định
Mỗi cửa sổ ảnh $2 \times 2$ có giá trị chuẩn hóa $x = [x_0, x_1, x_2, x_3]^T \in [0, 1]^4$ được mã hóa góc bằng $\mathcal{E}(x) = \bigotimes_{j=0}^{3} R_Y(\pi x_j)$, đi qua mạch đơn vị cố định $U(\theta)$ ($L = 2$ tầng `StronglyEntanglingLayers`), và đo 4 kỳ vọng Pauli-Z cục bộ:
$$q_k(x) = \langle \psi(x) | Z_k | \psi(x) \rangle \in [-1, 1], \quad k \in \{0, 1, 2, 3\}$$

### 2.2. Biên dịch Trước thành Bảng Tra Tĩnh
Do $\theta$ cố định trong giai đoạn suy luận, $f_Q: [0, 1]^4 \to [-1, 1]^4$ là một hàm tiền định 4 biến:
- **`INT8 Shift-LUT` (`256.00 KiB`):** Rời rạc hóa đều mỗi kênh thành 16 mức (4 bit cao, bỏ 4 bit thấp nhất của ảnh 8-bit), tổng $16^4 = 65,536$ trạng thái $\times 4$ byte `int8`. Tính chỉ mục trực tiếp bằng phép dịch bit (không dùng bảng chỉ mục phụ `GRID_MAP`).
- **`INT8 Grid-LUT` (`256.00 KiB` + `8.00 KiB` `GRID_MAP`):** Làm tròn gần nhất `round(p * 15 / 255)` qua bảng ánh xạ tĩnh `GRID_MAP` (trên kiến trúc 64-bit, `4 × 256 × 8 = 8,192 B = 8.00 KiB`).
- **`Interp4D SIMD` (`1,024.00 KiB` + `9.00 KiB` `INTERP_TABLE`):** Lưu lưới `float32` $16^4 \times 4$ và nội suy đa tuyến tính 4D trên 16 đỉnh lân cận bằng tập lệnh FMA (`_mm_fmadd_ps` trên x86_64 / `vfmaq_f32` trên ARMv8 NEON), đạt độ tương đồng cosine `0.999993` so với giải tích FP64.

---

## 3. KIỂM TOÁN BỘ NHỚ THỐNG NHẤT & HIỆU NĂNG SO VỚI `PyTorch Conv2d`

Nguồn dữ liệu chi tiết: [`reports/quantization_and_memory_audit.md`](quantization_and_memory_audit.md).

### 3.1. Bảng Dung lượng Bộ nhớ Thống nhất (64-bit, `Batch = 1`, Ảnh `28×28`)
Kiểm toán trực tiếp từ tệp nhị phân trọng số gộp BatchNorm (`weights/wide_v2_pneumoniamnist_s42_rust.bin`: `147,778` phần tử `f32` = `577.26 KiB`, thay vì `148,306` tham số chưa gộp = `579.32 KiB` ở bản nháp cũ) và cấu trúc bộ đệm trong `rust_qlut/src/lib.rs`, `rust_qlut/src/bin/medquanv_cli.rs`:

| Cấu hình mô hình | Bảng `Q-LUT` (KiB) | Bảng chỉ mục (`GRID_MAP` / `INTERP`) | Trọng số gộp BN (KiB) | **Tổng tĩnh (KiB / % của 2 MB L2)** | Đỉnh kích hoạt sống (`head.block1`, 3-buf) | **Tổng tĩnh + Đỉnh sống (KiB / % L2)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `WideGRQuanvNet v2` — `INT8 Shift` | `256.00` | `0.00` | `577.26` | **`833.26 KiB`** (`40.69%`) | `110.25 KiB` | **`943.51 KiB`** (`46.07%`) |
| `WideGRQuanvNet v2` — `INT8 Grid` | `256.00` | `8.00` | `577.26` | **`841.26 KiB`** (`41.08%`) | `110.25 KiB` | **`951.51 KiB`** (`46.46%`) |
| `WideGRQuanvNet v2` — `Interp4D` | `1,024.00` | `9.00` | `577.26` | **`1,610.26 KiB`** (`78.63%`) | `110.25 KiB` | **`1,720.51 KiB`** (`84.01%`) |
| `WideClassicalControl v2` (0% LUT) | `0.00` | `0.00` | `577.65` | **`577.65 KiB`** (`28.21%`) | `110.25 KiB` | **`687.90 KiB`** (`33.59%`) |
| `quanv_v3_int8` (`v3`, `INT8 Shift`) | `256.00` | `0.00` | `150.90` | **`406.90 KiB`** (`19.87%`) | *(nhỏ hơn v2)* | $\ge$ `406.90 KiB` (`19.87%`) |
| `ctrl_v3_light` (`v3`, 0% LUT) | `0.00` | `0.00` | `151.79` | **`151.79 KiB`** (`7.41%`) | *(nhỏ hơn v2)* | $\ge$ `151.79 KiB` (`7.41%`) |
| `micro_quanv` (`v3`, `INT8 Shift`) | `256.00` | `0.00` | `7.29` | **`263.29 KiB`** (`12.86%`) | *(nhỏ hơn v2)* | $\ge$ `263.29 KiB` (`12.86%`) |
| `micro_ctrl` (`v3`, 0% LUT) | `0.00` | `0.00` | `7.68` | **`7.68 KiB`** (`0.38%`) | *(nhỏ hơn v2)* | $\ge$ `7.68 KiB` (`0.38%`) |

*(Lưu ý: Nếu cộng toàn bộ các mảng stack cục bộ chưa tái sử dụng trong `extract_multiscale_qlut()` + `forward_single()` là `349.38 KiB`, tổng bộ nhớ cấp phát tức thời của bản `v2 INT8 Shift` là `1,182.63 KiB` = `57.75%` của `2,048 KiB`. Các tỷ lệ `% L2` ở trên là tỷ số tham chiếu so với `2,048 KiB`.)*

### 3.2. Đối chiếu Tốc độ Trung thực với `PyTorch Conv2d` trên `x86_64`
Không sử dụng tỷ số gia tốc so với bộ mô phỏng nghiên cứu PennyLane Python. So sánh trực tiếp 1 tầng 4 kênh $2 \times 2$ với `PyTorch Conv2d` trên CPU `x86_64` (`reports/highres_scalability_results.json`):

| Độ phân giải ($H \times W$, stride) | Số patches ($N_p$) | `INT8 Shift` (256 KiB) | `Interp4D` (1,024 KiB) | `PyTorch Conv2d` CPU | Tỷ lệ `Conv2d / INT8` | Tỷ lệ `Conv2d / Interp4D` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `28×28 (s=2)` | 196 | `0.49 µs` | `4.93 µs` | `54.45 µs` | `111.80×` nhanh hơn | `11.05×` nhanh hơn |
| `64×64 (s=2)` | 1,024 | `1.34 µs` | `25.55 µs` | `56.80 µs` | `42.23×` nhanh hơn | `2.22×` nhanh hơn |
| `128×128 (s=2)` | 4,096 | `4.35 µs` | `102.62 µs` | `61.36 µs` | `14.11×` nhanh hơn | **`1.67×` chậm hơn** |
| `224×224 (s=2)` | 12,544 | `11.91 µs` | `285.46 µs` | `158.58 µs` | `13.32×` nhanh hơn | **`1.80×` chậm hơn** |
| `512×512 (s=2)` | 65,536 | `61.55 µs` | `1,619.72 µs` | `250.10 µs` | `4.06×` nhanh hơn | **`6.48×` chậm hơn** |
| `1024×1024 (s=2)` | 262,144 | `271.79 µs` (`0.2718 ms`) | `6,473.90 µs` (`6.4739 ms`) | `1,329.98 µs` (`1.3300 ms`) | **`4.89×` nhanh hơn** | **`4.87×` chậm hơn** |

- **Nhận xét:** Phiên bản `INT8 Shift` nhanh hơn `PyTorch Conv2d` `4.89×` ở `1024×1024`, nhưng phiên bản giữ được độ chính xác (`Interp4D`) lại **chậm hơn `Conv2d` `4.87×`** do chi phí gom 16 đỉnh và nội suy 15 phép FMA cho mỗi cửa sổ.
- **Độ trễ End-to-End thuần Rust (`medquanv-cli`, `28×28`, `x86_64`, `Batch=1`, parity `< 5.3e-6` trên 624 ảnh test):**
  - `WideGRQuanvNet v2` (tối ưu A3): `INT8 Shift` đạt `2.96 ms` (`337.7 FPS`), `Interp4D` đạt `2.98 ms` (`335.6 FPS`).
  - Bộ `v3` (`--suite-v3`): `quanv_v3_int8` (`1.126 ms`, `887.8 FPS`, `406.90 KiB`) gần như **ngang bằng** `ctrl_v3_light` (`1.114 ms`, `897.6 FPS`, `151.79 KiB`) nhưng tốn thêm `256 KiB` bộ nhớ bảng tra; ở ngưỡng siêu nhỏ `micro_quanv` (`46.17 µs`, `21,656.7 FPS`) tương đương `micro_ctrl` (`47.59 µs`, `21,014.2 FPS`).
- **Trạng thái đo phần cứng NVIDIA Jetson Nano B01 (`Model P3450`):** `[PENDING — Chưa có số liệu đo trên bo mạch vật lý; toàn bộ cột ngoại suy cũ đã được loại bỏ]`.

---

## 4. PHÂN TÍCH ĐỘ BỀN NHIỄU ĐỐM, BÓC TÁCH 7 NHÁNH & KIỂM ĐỊNH THỐNG KÊ

Nguồn đầy đủ (`345` kiểm định cặp, 95% CI, 90% CI, TOST $\pm 2\%$, Benjamini–Hochberg FDR): [`reports/rigorous_stats_summary.md`](rigorous_stats_summary.md).

### 4.1. Đính chính Bảng Phân tách 4 Nhánh (PneumoniaMNIST, 5 Seeds, `Interp4D`)
Tái tính read-only từ các checkpoint khóa `weights/pillar2_wide/` để thu đủ 5 cặp seed cho 4 nhánh gốc (`(1) Ctrl Raw`, `(2) Ctrl + Damping`, `(3) Quanv Raw`, `(4) Quanv + Pre-LUT Damping`):

| Mức nhiễu ($\sigma$) | `(1) Ctrl Raw` | `(2) Ctrl + Damping` | `(3) Quanv Raw` | `(4) Quanv + PreLUT` | $\Delta$ `(4)−(2)` ($p_{\text{raw}}$ / $p_{\text{adj,fam}}$) | $\Delta$ `(3)−(1)` ($p_{\text{raw}}$ / $p_{\text{adj,fam}}$) | $\Delta$ `(2)−(1)` Riêng Damping ($p_{\text{adj,fam}}$) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `0.00` | `90.22 ± 1.65` | `90.22 ± 1.65` | `88.94 ± 1.63` | `88.94 ± 1.63` | `-1.28%` (`0.2350` / `0.2937`) | `-1.28%` (`0.2350` / `0.3916`) | `0.00%` (`1.0000`, `EQUIV`) |
| `0.05` | `88.95 ± 1.70` | `89.48 ± 1.53` | `90.37 ± 1.08` | `89.96 ± 1.12` | `+0.48%` (`0.6055` / `0.6055`) | `+1.42%` (`0.2117` / `0.3916`) | `+0.52%` (`0.8521`) |
| `0.10` | `60.60 ± 16.28` | `87.79 ± 2.06` | `71.29 ± 8.07` | `83.72 ± 3.84` | `-4.07%` (`0.1918` / `0.2937`) | `+10.69%` (`0.1600` / `0.3916`) | `+27.19%` (`0.0591`) |
| `0.15` | `45.16 ± 6.94` | `86.65 ± 1.87` | `46.12 ± 3.50` | `77.30 ± 6.50` | `-9.35%` (`0.0610` / `0.2390`) | `+0.96%` (`0.7769` / `0.7769`) | **`+41.49%` (`0.0010`)** |
| `0.20` | `40.20 ± 2.68` | `82.68 ± 1.83` | `39.49 ± 0.68` | `76.62 ± 5.90` | `-6.06%` (`0.0956` / `0.2390`) | `-0.72%` (`0.6128` / `0.7660`) | **`+42.48%` (`<0.0001`)** |

**Kết luận trung thực:**
1. **Bộ lọc tiền xử lý (`phase_damping_u8`) mới là tác nhân chính tạo khả năng kháng nhiễu** (`+27.19%` đến `+42.48%` ở $\sigma \ge 0.10$, có ý nghĩa thống kê rất mạnh tại $\sigma=0.15, 0.20$).
2. Khi so với baseline công bằng `(2) Ctrl + Damping`, nhánh `(4) Quanv + Pre-LUT Damping` **thấp hơn ở 4/5 mức nhiễu** (không mang lại đóng góp cộng thêm).
3. Ở chế độ thô `(3) vs (1)` tại $\sigma=0.10$, mức chênh $+10.69\%$ có 95% CI $[-6.54\%, +27.93\%]$ nên **không có ý nghĩa thống kê** ($p_{\text{raw}} = 0.1600$, $p_{\text{adj,fam}} = 0.3916$).

### 4.2. Bảng Ablation Mở rộng 7 Nhánh (Đối xứng trên cả `PneumoniaMNIST` và `BreastMNIST`, 5 Seeds)
Bổ sung đầy đủ 3 nhánh cổ điển đối chứng: **LUT ngẫu nhiên cố định** (`CLUT-RandMLP`, `CLUT-SepTrig`), **`Conv2d + tanh / hard-clip [-1, 1]`**, và **`Conv2d + Speckle Noise Augmentation`** (`results/ablation_clut/eval_results.json` & `extra_baselines.json`):

| Mô hình (`Params`) | Pneumonia Raw ($\sigma=0.00 / 0.10 / 0.20$) | Pneumonia Stabilized ($\sigma=0.00 / 0.10 / 0.20$) | Breast Raw ($\sigma=0.00 / 0.10 / 0.20$) | Breast Stabilized ($\sigma=0.00 / 0.10 / 0.20$) |
| :--- | :---: | :---: | :---: | :---: |
| `Ctrl (Conv2d)` (`148.4K`) | `90.22 / 60.60 / 40.20` | `90.22 / 87.79 / 82.68` | `84.10 / 81.37 / 76.07` | `84.10 / 83.21 / 81.58` |
| `Conv2d + tanh [-1,1]` (`148.4K`) | `90.71 / 63.39 / 39.69` | **`90.71 / 88.16 / 85.02`** | `84.23 / 81.50 / 75.43` | `84.23 / 84.19 / 82.26` |
| `Conv2d + clip [-1,1]` (`148.4K`) | `90.00 / 64.63 / 40.37` | `90.00 / 87.45 / 83.21` | `84.87 / 81.24 / 75.47` | **`84.87 / 85.73 / 84.74`** |
| `Conv2d + NoiseAug` (`148.4K`) | **`90.77 / 89.54 / 87.67`** | `90.77 / 87.40 / 85.76` | **`85.51 / 84.96 / 82.31`** | `85.51 / 83.46 / 81.50` |
| `CLUT-SepTrig` (`148.3K`) | `90.10 / 72.29 / 41.46` | `90.10 / 86.88 / 82.74` | `84.87 / 82.74 / 77.44` | `84.87 / 83.16 / 80.73` |
| `CLUT-RandMLP` (`148.3K`) | `90.48 / 71.53 / 46.69` | `90.48 / 87.71 / 82.52` | `85.77 / 83.25 / 77.69` | `85.77 / 85.73 / 83.21` |
| `Q-LUT SU(16) L=2` (`148.3K`) | `88.94 / 71.29 / 39.49` | `88.94 / 85.80 / 83.39` | `84.74 / 81.88 / 77.05` | `84.74 / 85.00 / 82.14` |

**Kết quả kiểm định cặp, TOST ($\pm 2\%$) và BH FDR:**
- **`Q-LUT` vs `CLUT-RandMLP` / `CLUT-SepTrig`:** `Q-LUT` không thắng có ý nghĩa thống kê ở bất kỳ mức nhiễu nào ($p_{\text{adj,fam}} \ge 0.1636$); đạt tương đương TOST $\pm 2\%$ tại Pneumonia Stabilized $\sigma=0.20$ vs `CLUT-RandMLP` ($p_{\text{TOST}} = 0.0338$), $\sigma=0.05$ vs `CLUT-SepTrig` ($p_{\text{TOST}} = 0.0474$), và Breast Raw $\sigma=0.15$ vs `CLUT-SepTrig` ($p_{\text{TOST}} = 0.0311$). Các điểm có phương sai lớn ghi nhận trung thực là `INCONCLUSIVE` do $n=5$ ($df=4$).
- **`Q-LUT` vs `Conv2d + tanh [-1, 1]`:** Trên Pneumonia Stabilized, `Q-LUT` **thấp hơn có ý nghĩa thống kê ở family-FDR** tại 4/5 mức nhiễu ($\Delta = -1.76\%$ đến $-2.72\%$, $p_{\text{adj,fam}} \in [0.0125, 0.0152]$). Trên BreastMNIST sạch, hai mô hình đạt tương đương TOST $\pm 2\%$ ($p_{\text{TOST}} = 0.0271$).
- **`Q-LUT` vs `Conv2d + NoiseAug` (Raw Stem):** Huấn luyện CNN có chèn nhiễu đốm đạt `89.54% / 88.77% / 87.67%` trên PneumoniaMNIST và `84.96% / 83.85% / 82.31%` trên BreastMNIST mà **không cần bộ lọc damping**, vượt `Q-LUT` có ý nghĩa thống kê ở $\sigma \ge 0.15$ ($p_{\text{adj,glob}} \le 0.0017$ trên Pneumonia; $p_{\text{adj,fam}} \le 0.0480$ trên Breast).

---

## 5. ẢNH HƯỞNG CỦA LƯỢNG TỬ HÓA ĐẦU VÀO 4-BIT (`INT8` vs `Interp4D`)

Từ [`reports/pareto_int8_vs_interp4d.json`](pareto_int8_vs_interp4d.json) và [`reports/quantization_and_memory_audit.md`](quantization_and_memory_audit.md) (5 seeds, đánh giá không huấn luyện lại từ mô hình `v2` học trên `Interp4D`):
- **BreastMNIST:** `INT8 Shift` và `Interp4D` đạt tương đương **TOST $\pm 2\%$ trên cả 5/5 mức nhiễu** ($|\Delta| \le 0.43\%$, $p_{\text{TOST}} \le 0.0178$).
- **PneumoniaMNIST:** Sạch giảm nhẹ $-0.45\%$ (đạt TOST `EQUIV`, $p_{\text{TOST}} = 0.0002$); dưới chế độ `stabilized_stem` đối xứng đạt TOST `EQUIV` trên cả 5/5 mức nhiễu ($p_{\text{TOST}} \le 0.0004$).
- **OCTMNIST (ảnh cắt lớp võng mạc có vi tương phản mịn):** Rời rạc hóa 4-bit không huấn luyện lại làm giảm đáng kể độ chính xác tại $\sigma=0.00$ (`-3.40%`, $p_{\text{adj,fam}}=0.0003$), $\sigma=0.05$ (`-4.69%`, $p_{\text{adj,fam}}=0.0006$), và $\sigma=0.10$ (`-2.53%`, $p_{\text{adj,fam}}=0.0143$), không đạt TOST $\pm 2\%$.
- **Khả năng phục hồi bằng QAT (`v3`, `reports/pillar3_v3_results.json`):** Huấn luyện nhận thức lượng tử (`quanv_v3_int8`, `39.3K` params, `4.03M` MACs) khôi phục độ chính xác sạch `INT8` trên OCTMNIST từ `80.64%` lên `82.90%` với độ lệch biên dịch `0.0000%`, tuy nhiên vẫn thấp hơn `ctrl_v3_light` tại OCTMNIST $\sigma=0.10$ (`51.23%` vs `57.32%`).

---

## 6. KẾT LUẬN

1. Lớp quanvolution 4-qubit cố định tương đương thực nghiệm với bảng tra cổ điển `256 KiB` (`INT8`) hoặc `1,024 KiB` (`Interp4D`, cosine `0.999993`); tại thời điểm suy luận không tồn tại lợi thế tính toán lượng tử.
2. Trên các bộ dữ liệu ảnh y tế quy mô nhỏ (`MedMNIST`), khi đối chiếu công bằng với `Conv2d`, `Conv2d + tanh`, `Conv2d + NoiseAug`, và bảng tra phi tuyến cổ điển `CLUT-RandMLP` / `CLUT-SepTrig`, `Q-LUT` không đem lại lợi thế độ chính xác có ý nghĩa thống kê.
3. Trong thực hành kỹ thuật hiện tại, nghiên cứu lượng tử cho thị giác máy tính vẫn mang tính nghiên cứu nền tảng (*research-based*); với bài toán triển khai nhúng ảnh y tế quy mô nhỏ, mạng CNN cổ điển kết hợp tăng cường dữ liệu nhiễu hoặc lọc tiền xử lý vẫn là lựa chọn đơn giản, ít tốn bộ nhớ và hiệu quả hơn.
4. Toàn bộ thư viện Rust (`rust_qlut`), giao diện C-ABI, công cụ `medquanv-cli` và bộ kịch bản tái lập (`scripts/reproduce_all.py`) được công khai như một **công cụ benchmark tái lập** phục vụ kiểm chứng độc lập.
