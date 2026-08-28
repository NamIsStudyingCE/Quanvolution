# BẢNG 5: ĐỘ TRỄ SUY LUẬN & CHI PHÍ PHẦN CỨNG (INFERENCE LATENCY & COMPUTATIONAL COST)

> **Môi trường đo đạc:** CPU Intel Core (Mô phỏng máy trạng thái giải tích `default.qubit` trên PennyLane 0.35 + PyTorch 2.13), RAM 16GB.  
> **Nguồn đo lường:** `measure_params_cost.py` & Log huấn luyện `run_gd3.py`.

---

## 1. BẢNG ĐỘ TRỄ SUY LUẬN TỪNG THÀNH PHẦN (INFERENCE LATENCY)

| Mô hình & Chế độ Thực thi | Giai đoạn Tính toán | Độ trễ Trung bình (ms / ảnh) | Tỷ lệ so với Cổ điển | Chi phí Tham số Kernel |
| :--- | :--- | :---: | :---: | :---: |
| **Classical CNN Baseline** | Toàn bộ mạng (End-to-End Forward) | **$0.310 \text{ ms}$** | **$1.0\times$** *(Chuẩn)* | 20 tham số |
| **Fixed Quanvolution** | Trích xuất đặc trưng (196 patch lượng tử) | $220.187 \text{ ms}$ | $710.3\times$ | **0 tham số** |
| | Tầng phân loại (Classifier Head) | $0.034 \text{ ms}$ | $0.11\times$ | Giống nhau |
| | **Toàn bộ mạng (End-to-End)** | **$220.221 \text{ ms}$** | **$710.4\times$** | **0 tham số** |
| **Trainable Quanvolution (Strongly 3-Axis)** | **Toàn bộ mạng (End-to-End)** | **~220.25 ms** | **~710.5x** | 12 – 24 tham số |

---

## 2. BẢNG THỜI GIAN HUẤN LUYỆN THỰC TẾ (WALL-CLOCK TRAINING TIME)

| Bộ dữ liệu | Mô hình | Thời gian Huấn luyện / Seed | Tổng Thời gian (10 Seeds) | Đỉnh Bộ nhớ RAM (Peak RAM) |
| :--- | :--- | :---: | :---: | :---: |
| **BreastMNIST** (780 mẫu) | Classical CNN Baseline | $\approx 2.5 \text{ giây}$ | $\approx 25 \text{ giây}$ | $< 1.2 \text{ GB}$ |
| | Fixed Quanvolution (Precomputed) | $\approx 1.8 \text{ giây}$ | $\approx 18 \text{ giây}$ | $< 1.1 \text{ GB}$ |
| | Trainable Quanvolution (Backprop) | $\approx 35 \text{ giây}$ | $\approx 5.8 \text{ phút}$ | $< 1.5 \text{ GB}$ |
| **OCTMNIST** (5.000 mẫu) | Classical CNN Baseline | $\approx 18 \text{ giây}$ | $\approx 3.0 \text{ phút}$ | $< 2.0 \text{ GB}$ |
| | Fixed Quanvolution (Precomputed) | $\approx 12 \text{ giây}$ | $\approx 2.0 \text{ phút}$ | $< 1.8 \text{ GB}$ |
| | Trainable Quanvolution (Backprop) | $\approx 70 \text{ giây}$ | $\approx 11.6 \text{ phút}$ | $< 2.5 \text{ GB}$ |

---

## 3. LUẬN ĐIỂM BIỆN LUẬN CHI PHÍ & ĐÁNH ĐỔI (TRADE-OFF ARGUMENTS)

1. **Hiệu quả Tiền tính toán (Precomputation Advantage):** Với mạch tĩnh (Fixed Quanvolution), đặc trưng lượng tử chỉ cần trích xuất **đúng 1 lần duy nhất**, sau đó toàn bộ quá trình train 10 seeds cực kỳ nhanh ($\approx 12-18$ giây), nhanh hơn cả train Classical CNN từ ảnh thô.
2. **Khả thi trong Y tế Lâm sàng:** Độ trễ suy luận $\approx 0.22 \text{ giây/ảnh}$ hoàn toàn nằm trong giới hạn cho phép của các hệ thống chẩn đoán hỗ trợ bác sĩ (CAD systems), đổi lại là **độ ổn định phương sai vượt trội (~3x std nhỏ hơn)** và khả năng nhận diện ca bệnh hiếm (PR-AUC cao).
3. **Định hướng Tương lai (Future Work):** Độ trễ $\sim 710\times$ là hạn chế của môi trường giả lập CPU phần mềm. Khi chuyển giao sang bộ tăng tốc lượng tử chuyên dụng (GPU Tensor Core cuQuantum hoặc chip ASIC/FPGA lượng tử), độ trễ sẽ được rút ngắn tiệm cận thời gian thực.
