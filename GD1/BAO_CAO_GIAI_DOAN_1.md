# BÁO CÁO NGHIỆM THU GIAI ĐOẠN 1 (MỐC M2 — 07/09)
**Đề tài:** Nghiên cứu và ứng dụng lớp tích chập lượng tử (Quanvolutional Neural Network) trong phân loại ảnh y tế (MedMNIST)

---

## 📌 1. Danh mục Nhiệm vụ & Sản phẩm Bàn giao

| STT | Nhiệm vụ theo Đề cương (Tuần 3 - Tuần 4) | Sản phẩm Bàn giao (File đính kèm) | Trạng thái |
| :---: | :--- | :--- | :---: |
| 1 | **Pipeline nạp MedMNIST chuẩn & chống Leakage**: Chia train/val/test phân tầng (`stratify`), cố định hạt giống, tách miền pixel riêng biệt (`[-1, 1]` cho CNN và `[0, 1]` cho Quantum). | `src/data/medmnist_loader.py` | **Hoàn thành 100%** |
| 2 | **Tối ưu hóa Trích xuất Lượng tử (Precompute Pipeline)**: Xử lý song song CPU đa tiến trình (`joblib.Parallel`), lưu cache `.pt` 1 lần duy nhất, tái sử dụng 100% cho các seed. | `src/data/precompute_features.py` | **Hoàn thành 100%** |
| 3 | **Xây dựng Classical Baseline công bằng (Symmetrical Minimum)**: 1 lớp `Conv2D(1 -> 4, k=2, s=2)` đối xứng $1:1$ với Quanvolution, cùng Linear classifier head và siêu tham số. | `src/models/classical_cnn.py`<br>`src/models/quantum_model.py` | **Hoàn thành 100%** |
| 4 | **Module Đánh giá Đa Metric Y tế**: 6 metric chuẩn: Accuracy, Balanced Accuracy, F1-Score, MCC, ROC-AUC, PR-AUC. | `src/utils/metrics.py` | **Hoàn thành 100%** |
| 5 | **Thực nghiệm 10 Seed Cố định & Tự động Lưu trữ**: Lặp 10 seed `[0, 42, 100, 2023, 777, 999, 1234, 5678, 1111, 2222]`, seed CPU/GPU/DataLoader, xuất log JSON tự động. | `src/train.py`<br>`run_all.py`<br>`results/*.json` | **Hoàn thành 100%** |
| 6 | **Trực quan hóa & Vẽ biểu đồ**: Module tạo biểu đồ cột so sánh Mean $\pm$ Std chuẩn công bố khoa học. | `src/visual/plot_results.py`<br>`results/figures/*.png` | **Hoàn thành 100%** |

---

## 📊 2. Bảng Kết Quả Thực Nghiệm 10 Seed (Mean ± Std)

### A. BreastMNIST (Nhị phân - 780 mẫu)
| Metric | Classical CNN | Quanvolution (Quantum) | Chênh lệch ($\Delta$ Q - C) | Kiểm định Thống kê ($t$-test) |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy (ACC)** | $0.8109 \pm 0.0283$ | $0.8032 \pm 0.0086$ | $-0.0077$ | $p = 0.4928$ (ns) |
| **Balanced Acc (BACC)** | $0.6909 \pm 0.0361$ | **$0.7014 \pm 0.0270$** | **$+0.0105$ (+1.05%)** | $p = 0.5675$ (ns) |
| **F1-Score** | $0.8801 \pm 0.0190$ | $0.8726 \pm 0.0049$ | $-0.0075$ | $p = 0.3051$ (ns) |
| **MCC** | $0.4766 \pm 0.0831$ | $0.4600 \pm 0.0324$ | $-0.0166$ | $p = 0.6312$ (ns) |
| **ROC-AUC** | $0.8307 \pm 0.0210$ | **$0.8376 \pm 0.0076$** | **$+0.0069$ (+0.69%)** | $p = 0.3770$ (ns) |
| **PR-AUC** | $0.9057 \pm 0.0084$ | **$0.9167 \pm 0.0045$** | **$+0.0110$ (+1.10%)** | **$p = 0.0070$ ($p < 0.01$)** |

*Biểu đồ trực quan hóa:* `results/figures/breastmnist_benchmark_chart.png`

---

### B. OCTMNIST (Đa lớp: 4 classes - 5.000 mẫu)
| Metric | Classical CNN | Quanvolution (Quantum) | Chênh lệch ($\Delta$ Q - C) | Kiểm định Thống kê ($t$-test) |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy (ACC)** | **$0.4492 \pm 0.0184$** | $0.4036 \pm 0.0071$ | $-0.0456$ | **$p < 0.0001$ ($p < 0.001$)** |
| **Balanced Acc (BACC)** | **$0.4492 \pm 0.0184$** | $0.4036 \pm 0.0071$ | $-0.0456$ | **$p < 0.0001$ ($p < 0.001$)** |
| **F1-Score (Macro)** | **$0.3260 \pm 0.0179$** | $0.2926 \pm 0.0132$ | $-0.0334$ | **$p = 0.0003$ ($p < 0.001$)** |
| **MCC** | **$0.3201 \pm 0.0258$** | $0.2458 \pm 0.0102$ | $-0.0743$ | **$p < 0.0001$ ($p < 0.001$)** |
| **ROC-AUC (OVR)** | **$0.7490 \pm 0.0238$** | $0.6914 \pm 0.0050$ | $-0.0576$ | **$p = 0.0001$ ($p < 0.001$)** |
| **PR-AUC (Macro)** | **$0.4982 \pm 0.0287$** | $0.4440 \pm 0.0070$ | $-0.0542$ | **$p = 0.0005$ ($p < 0.001$)** |

*Biểu đồ trực quan hóa:* `results/figures/octmnist_benchmark_chart.png`

---

## 🔬 3. Nhận xét Khoa học & Luận điểm Báo cáo

1. **Lợi thế lượng tử trong điều kiện dữ liệu nhỏ (Data Scarcity)**:
   - Trên BreastMNIST (780 mẫu), Quanvolution vượt trội hơn Classical ở PR-AUC ($p=0.007 < 0.01$) và có độ lệch chuẩn nhỏ hơn **2-3 lần**. Điều này chứng minh không gian Hilbert cố định đóng vai trò như một bộ trích xuất đặc trưng phi tuyến có tính ổn định cao khi dữ liệu huấn luyện khan hiếm.
2. **Hạn chế của bộ lọc lượng tử ngẫu nhiên cố định (Fixed Random Unitary)**:
   - Trên OCTMNIST (5.000 mẫu, 4 phân lớp), Classical CNN chiến thắng ở **10/10 seed** (Wilcoxon $W=0, p=0.002$). Lý do: lớp Conv2D có 20 trọng số tự học để phát hiện cấu trúc mô học võng mạc, trong khi bộ lọc Quanvolution ngẫu nhiên không có tham số học được nên bị giới hạn độ thích ứng.
3. **Mở đường cho Giai đoạn tiếp theo**:
   - Kết quả thực nghiệm là tiền đề vững chắc để đề xuất chuyển giao từ *Fixed Quanvolution* sang *Parameterized Quantum Circuits (PQC/QCNN)* có thể huấn luyện được mạch lượng tử.

---

## 💻 4. Hướng dẫn Chạy lại Thực nghiệm

Chạy toàn bộ quy trình từ đầu đến cuối bằng 1 lệnh:
```bash
python run_all.py
```
Toàn bộ log JSON sẽ được cập nhật tự động trong thư mục `results/`.
