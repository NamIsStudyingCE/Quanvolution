# BẢNG 2: PHÂN BỐ THAM SỐ MÔ HÌNH VÀ TÍNH CÔNG BẰNG KIẾN TRÚC

> **Đo đạc tự động:** Trích xuất trực tiếp từ `measure_params_cost.py` trên PyTorch 2.13 & PennyLane.

| Tập dữ liệu | Mô hình | Feature-Extractor Params (Kernel + BatchNorm) | Classifier-Head Params (`Linear 784 → K`) | TỔNG SỐ THAM SỐ HUẤN LUYỆN | Ghi chú & Ý nghĩa Kiến trúc |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **BreastMNIST**<br>*(2 lớp, $L=2$)* | **Classical CNN Baseline** | 28 (Conv 20 + BN 8) | 1.570 | 1.598 | Baseline cổ điển đối xứng $1:1$ |
| | **Fixed Quanvolution** | **8 (Kernel 0 + BN 8)** | 1.570 | **1.578** | **0 tham số học ở kernel** (*Quantum Inductive Bias*) |
| | **Trainable Basic ($L=2$)** | 16 (Kernel 8 + BN 8) | 1.570 | 1.586 | Xoay 1 trục $RY(\theta)$, 8 tham số lượng tử |
| | **Trainable Strongly ($L=2$)** | 32 (Kernel 24 + BN 8) | 1.570 | 1.602 | Xoay 3 trục $(RX, RY, RZ)$, 24 tham số lượng tử |
| **OCTMNIST**<br>*(4 lớp, $L=1$)* | **Classical CNN Baseline** | 28 (Conv 20 + BN 8) | 3.140 | 3.168 | Baseline cổ điển đối xứng $1:1$ |
| | **Fixed Quanvolution** | **8 (Kernel 0 + BN 8)** | 3.140 | **3.148** | **0 tham số học ở kernel** |
| | **Trainable Basic ($L=1$)** | 12 (Kernel 4 + BN 8) | 3.140 | 3.152 | Xoay 1 trục $RY(\theta)$, 4 tham số lượng tử |
| | **Trainable Strongly ($L=1$)** | 20 (Kernel 12 + BN 8) | 3.140 | 3.160 | Xoay 3 trục $(RX, RY, RZ)$, 12 tham số lượng tử |

### 💡 Luận điểm Thảo luận Khoa học (Discussion Points):
1. **Classifier Head giống nhau 100%:** Phần `Linear 784 → K` chiếm >98% tổng tham số và được giữ đồng nhất tuyệt đối giữa hai phía, cô lập hoàn toàn biến số can thiệp ở tầng phân loại cuối.
2. **Hiệu quả tham số tầng đặc trưng:** Mạch lượng tử tĩnh chỉ cần **0 tham số học** mà vẫn trích xuất được không gian đặc trưng phi tuyến có tính phân tách cao, tiết kiệm 100% chi phí huấn luyện vi phân cho kernel.
