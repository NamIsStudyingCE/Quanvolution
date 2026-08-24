# Quanvolutional Neural Networks for Medical Image Classification

> **Đề tài Luận văn Tốt nghiệp:** Nghiên cứu và ứng dụng lớp tích chập lượng tử (Quanvolutional Layer) trong bài toán phân loại ảnh y tế (MedMNIST), so sánh công bằng với các kiến trúc cổ điển (Classical CNN).

---

## 📌 1. Giới thiệu Đề tài
Dự án tập trung vào việc áp dụng kiến trúc **Quanvolutional Neural Network (Quanvolution)** dựa trên nghiên cứu của *Henderson et al. (2019)*. Mạng sử dụng một mạch lượng tử nhỏ (Quantum Circuit) hoạt động như một bộ lọc không gian (Kernel Filter) trượt qua các vùng ảnh để trích xuất các đặc trưng phi tuyến lượng tử (*Quantum Feature Maps*), sau đó kết hợp với các mạng nơ-ron cổ điển để phân loại.

### Các nguyên tắc nghiên cứu cốt lõi:
1. **Tính tái lập (Reproducibility):** Cố định hạt giống ngẫu nhiên (seed), quy trình thực nghiệm rõ ràng, đảm bảo cùng code cùng seed ra cùng kết quả.
2. **So sánh công bằng (Fair Comparison):** Mô hình cổ điển (Classical Baseline) được tối ưu hóa và huấn luyện nghiêm ngặt tương đương để đối chứng.
3. **Đánh giá đa chiều (Multi-metric Evaluation):** Đánh giá trên dữ liệu mất cân bằng với đầy đủ các chỉ số: *Accuracy, Balanced Accuracy, F1-Score, MCC, ROC-AUC, PR-AUC* và kiểm định thống kê qua đa seed ($\geq 5$ seeds).

---

## Phase 1: Training & Evaluation

Implemented reproducible 10-seed experiment pipeline with perfectly fair classical baseline.

### How to Run

1. Open terminal in `d:\KhoaLuanTotNghiep`.
2. Run all experiments automatically:
   ```bash
   python run_all.py
   ```
3. Or run individually:
   - BreastMNIST (binary, 780 samples):
     ```bash
     python src/train.py --dataset breastmnist --epochs 30
     ```
   - OCTMNIST (multi-class, subset 5000 samples):
     ```bash
     python src/train.py --dataset octmnist --max_samples 5000 --epochs 30
     ```

Script auto-precomputes quantum features first (multiprocessing enabled), saves deterministic splits, then trains 10 fixed seeds. Results save to `results/` folder.

---

## 📂 2. Cấu trúc Thư mục Dự án

```text
├── docs/                        # Tài liệu nghiên cứu, tóm tắt lý thuyết (Tuần 1 - M1)
├── notebooks/                   # Jupyter Notebooks chạy thử nghiệm và trực quan hóa
│   └── 00_quanvolution_demo.ipynb
├── src/                         # Mã nguồn chính (Modules tái sử dụng)
│   ├── data/                    # Pipeline nạp và xử lý MedMNIST
│   ├── models/                  # Định nghĩa Quantum Circuit & Classical CNN
│   ├── utils/                   # Hàm seed, metric y tế, kiểm định thống kê
│   └── visual/                  # Trực quan hóa feature maps và biểu đồ
├── results/                     # Kết quả đầu ra (Figures, CSV tables, Checkpoints)
│   └── figures/
├── quanvolution_demo.py         # Script demo ban đầu (Giai đoạn 0)
├── requirements.txt             # Danh sách thư viện phụ thuộc
├── .gitignore                   # Cấu hình bỏ qua các file tạm / dữ liệu nặng
└── README.md                    # Hướng dẫn dự án
```

---

## 🚀 3. Hướng dẫn Cài đặt & Chạy Thực nghiệm

### Bước 1: Khởi tạo Môi trường Python
Khuyến nghị sử dụng môi trường ảo (`venv` hoặc `conda`):
```bash
python -m venv venv
# Kích hoạt trên Windows:
.\venv\Scripts\activate
```

### Bước 2: Cài đặt Thư viện
```bash
pip install -r requirements.txt
```

### Bước 3: Chạy Demo Giai đoạn 0
Chạy script demo Quanvolution trên tập MNIST:
```bash
python quanvolution_demo.py
```
Sau khi chạy xong, ảnh so sánh đặc trưng lượng tử sẽ được lưu tại: `results/figures/quanvolution_features.png` (hoặc ngay thư mục gốc).

---

## 📊 4. Lộ trình Thực hiện (13 Tuần)
- [x] **GĐ0 (T1-T2):** Nền tảng, môi trường, demo Quanvolution (Mốc M1 - 24/08).
- [x] **GĐ1 (T3-T4):** Pipeline MedMNIST & Baseline CNN công bằng đa seed (Mốc M2 - 07/09).
- [ ] **GĐ2 (T5-T7):** Cài đặt Quanvolution lõi & trích xuất đặc trưng (Mốc M3 - 28/09).
- [ ] **GĐ3 (T8-T10):** Thực nghiệm mở rộng, Ablation study & Kiểm định thống kê (Mốc M4 - 19/10).
- [ ] **GĐ4 (T11-T13):** Hoàn thiện Luận văn, Demo nghiệm thu & Bảo vệ (09/11).
