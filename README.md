# Quanvolutional Neural Networks for Medical Image Classification

> **Đề tài Luận văn Tốt nghiệp:** Nghiên cứu và ứng dụng lớp tích chập lượng tử (Quanvolutional Layer) trong bài toán phân loại ảnh y tế (MedMNIST), so sánh công bằng với các kiến trúc cổ điển (Classical CNN).  
> **Định hướng công bố bài báo:** *"Trainable vs. Fixed Quanvolutional Filters for Medical Image Classification: A Fair, Reproducible Benchmark on MedMNIST"*

---

## 📌 1. Giới thiệu Đề tài & Bối cảnh Nghiên cứu

Dự án tập trung nghiên cứu kiến trúc **Quanvolutional Neural Network (Quanvolution)** dựa trên nền tảng của *Henderson et al. (2019)*. Mạng sử dụng các mạch lượng tử 4-qubit hoạt động như các bộ lọc không gian (Kernel Filter $2 \times 2$) trượt qua ảnh để trích xuất các bản đồ đặc trưng phi tuyến trong không gian trạng thái Hilbert (*Quantum Feature Maps*), kết hợp với mạng phân loại cổ điển (Classifier Head).

### Các nguyên tắc nghiên cứu cốt lõi:
1. **Tính tái lập tuyệt đối (Strict Reproducibility):** Cố định hệ thống hạt giống ngẫu nhiên (10 seeds độc lập), lưu trữ toàn bộ dữ liệu thô định dạng JSON và biểu đồ 300 DPI.
2. **So sánh công bằng (Fair Benchmark):** Kiến trúc cổ điển đối chứng (*Symmetrical Minimum CNN*) được thiết kế đối xứng $1:1$ với tầng trích xuất lượng tử; phần Classifier Head (`Linear 784 → classes`) được giữ **giống nhau 100%** giữa hai phía.
3. **Đánh giá đa chiều & Kiểm định Thống kê Kép:** Đánh giá trên dữ liệu mất cân bằng y tế qua 6 metrics (*Accuracy, Balanced Accuracy, F1-Score, MCC, ROC-AUC, PR-AUC*) kết hợp kiểm định thống kê kép *Paired t-test* và *Wilcoxon signed-rank test*.

---

## 📊 2. Lộ trình Thực hiện & Báo cáo Nghiệm thu (13 Tuần)

Dự án được triển khai và đóng gói độc lập theo từng mốc (milestone) trong kế hoạch 3 tháng của GVHD:

| Mốc | Giai đoạn & Chủ đề | Sản phẩm & Điểm cốt lõi | Báo cáo chi tiết | Trạng thái |
| :---: | :--- | :--- | :---: | :---: |
| **M1** (24/08) | **GĐ0: Nền tảng, Môi trường & Demo** | Tóm tắt lý thuyết, cài đặt PennyLane/PyTorch, Demo MNIST | [`GD0/BAO_CAO_GIAI_DOAN_0.md`](GD0/BAO_CAO_GIAI_DOAN_0.md) | ✅ **100%** |
| **M2** (07/09) | **GĐ1: Pipeline MedMNIST & Baseline** | Pipeline MedMNIST, Symmetrical CNN, 10 seeds benchmark | [`GD1/BAO_CAO_GIAI_DOAN_1.md`](GD1/BAO_CAO_GIAI_DOAN_1.md) | ✅ **100%** |
| **M3** (28/09) | **GĐ2: Khảo sát Mạch Lượng tử Tĩnh** | Circuit Ablation 6 cấu hình, Quán quân `basic_L2` & `random_L1` | [`GD2/BAO_CAO_GIAI_DOAN_2.md`](GD2/BAO_CAO_GIAI_DOAN_2.md) | ✅ **100%** |
| **M4** (19/10) | **GĐ3: Tích chập Lượng tử Tự học** | Ma trận 3 Tầng, QNN khả vi (`backprop`), Đo chi phí phần cứng | [`GD3/BAO_CAO_GIAI_DOAN_3.md`](GD3/BAO_CAO_GIAI_DOAN_3.md) | ✅ **100%** |
| **M5** (09/11) | **GĐ4: Luận văn & Bảo vệ** | Soạn thảo Luận văn, Slide thuyết trình & Nghiệm thu | [`GD4/BAO_CAO_GIAI_DOAN_4.md`](GD4/BAO_CAO_GIAI_DOAN_4.md) | ⏳ Chuẩn bị |

---

## 🔬 3. Bảng Kết quả Thực nghiệm Tổng hợp (10 Seeds Độc lập)

### A. BreastMNIST (780 mẫu, Nhị phân, Lệch lớp nặng, $L=2$)
* **Quán quân ROC-AUC:** `Fixed Basic L2` đạt **$0.8521 \pm 0.0090$** (vượt Classical CNN $0.8336$, $p=0.0309 < 0.05$).
* **Quán quân PR-AUC:** `Fixed Strongly L2` đạt **$0.9182 \pm 0.0067$** (chứng minh *Quantum Inductive Bias* của mạch tĩnh).
* **Độ ổn định:** Độ lệch chuẩn (std) của Lượng tử nhỏ hơn Cổ điển gấp **~2.5x – 3x**.

### B. OCTMNIST (5.000 mẫu, 4 lớp bệnh lý, Dữ liệu lớn, $L=1$)
* **Quán quân:** `Classical CNN` dẫn đầu áp đảo (**ROC-AUC $0.7505 \pm 0.0227$** vs QNN $\sim 0.69$).
* **Ranh giới ứng dụng (Quantum Boundary):** Mạch lượng tử 4-qubit bộc lộ giới hạn dung lượng tham số trên bài toán dữ liệu lớn, khẳng định ưu thế cổ điển ở quy mô lớn.

### C. Hiệu quả Tham số & Độ trễ Suy luận (CPU)
* **Tầng trích xuất đặc trưng (Kernel):** Mạch tĩnh dùng **0 tham số học** (tiết kiệm 20 tham số conv).
* **Độ trễ suy luận:** Classical CNN = $0.310\text{ ms/ảnh}$; Quanvolution = $220.22\text{ ms/ảnh}$ (chậm hơn $\sim 710\text{x}$ trên mô phỏng CPU).

---

## 📂 4. Cấu trúc Thư mục Dự án

```text
├── docs/                        # Tài liệu tổng quan lý thuyết QNN
├── notebooks/                   # Jupyter Notebook demo trực quan hóa
├── src/                         # Mã nguồn mô hình & pipeline
│   ├── data/                    # Nạp MedMNIST & Tiền tính toán đặc trưng
│   ├── models/                  # Quantum Circuits & Classical CNN
│   ├── utils/                   # Hàm seed, 6 metrics y tế, kiểm định thống kê
│   ├── experiments/              # Master runner thực nghiệm 3 tầng
│   └── visual/                  # Trực quan hóa feature maps & loss curves
├── GD0/                         # Gói nghiệm thu độc lập Giai đoạn 0 (Mốc M1)
├── GD1/                         # Gói nghiệm thu độc lập Giai đoạn 1 (Mốc M2)
├── GD2/                         # Gói nghiệm thu độc lập Giai đoạn 2 (Mốc M3)
├── GD3/                         # Gói nghiệm thu độc lập Giai đoạn 3 (Mốc M4)
├── results/                     # Kết quả JSON thô & Biểu đồ 300 DPI
├── measure_params_cost.py       # Script tự đo số tham số & độ trễ suy luận
├── run_all.py                   # Runner tự động Giai đoạn 1
├── run_ablation.py              # Runner tự động Giai đoạn 2
├── run_gd3.py                   # Runner tự động Giai đoạn 3 (Ma trận 3 tầng)
├── requirements.txt             # Danh sách thư viện phụ thuộc
└── README.md                    # Hướng dẫn tổng quan dự án
```

---

## 🚀 5. Hướng dẫn Cài đặt & Tái hiện Thực nghiệm

```bash
# 1. Tạo và kích hoạt môi trường ảo
python -m venv .venv
.\.venv\Scripts\activate   # Trên Windows

# 2. Cài đặt các thư viện phụ thuộc
pip install -r requirements.txt

# 3. Đo đạc tham số & chi phí phần cứng (Script của GVHD)
python measure_params_cost.py

# 4. Tái hiện toàn bộ thực nghiệm Giai đoạn 3 (Ma trận 3 tầng 10 seeds)
python run_gd3.py
```

---

*Tác giả: NamIsStudyingCE | Trường Đại học Công nghệ Thông tin (UIT) — ĐHQG-HCM*
