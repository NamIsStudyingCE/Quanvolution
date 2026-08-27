# GĐ1 — Pipeline Dữ liệu & Classical Baseline (Mốc M2 — 07/09)

> **Đề tài Luận văn Tốt nghiệp:** Nghiên cứu và ứng dụng lớp tích chập lượng tử (Quanvolutional Layer) trong bài toán phân loại ảnh y tế (MedMNIST), so sánh công bằng với các kiến trúc cổ điển (Classical CNN).

**Mục tiêu giai đoạn:** Xây dựng pipeline dữ liệu chuẩn, chống data leakage, và huấn luyện Classical CNN baseline công bằng qua 10 seeds độc lập trên 2 bộ dữ liệu MedMNIST.

---

## 📌 Nhiệm vụ Giai đoạn 1 (Tuần 3 - Tuần 4)

| STT | Nhiệm vụ | Sản phẩm Bàn giao | Trạng thái |
| :---: | :--- | :--- | :---: |
| 1 | Pipeline nạp MedMNIST chuẩn & chống Leakage | `src/data/medmnist_loader.py` | ✅ Hoàn thành |
| 2 | Precompute Quantum Features (đa tiến trình) | `src/data/precompute_features.py` | ✅ Hoàn thành |
| 3 | Classical Baseline công bằng (Symmetrical Minimum CNN) | `src/models/classical_cnn.py` | ✅ Hoàn thành |
| 4 | Module đánh giá 6 Metric Y tế | `src/utils/metrics.py` | ✅ Hoàn thành |
| 5 | Thực nghiệm 10 Seed cố định, lưu JSON tự động | `src/train.py`, `run_all.py` | ✅ Hoàn thành |
| 6 | Biểu đồ trực quan hóa kết quả | `src/visual/plot_results.py` | ✅ Hoàn thành |

---

## 📊 Kết quả Thực nghiệm (Mean ± Std, 10 Seeds)

### BreastMNIST (Nhị phân — 780 mẫu)
| Metric | Classical CNN | Quantum (Fixed) | Δ |
| :--- | :---: | :---: | :---: |
| ROC-AUC | 0.8307 ± 0.0210 | **0.8376 ± 0.0076** | +0.0069 |
| PR-AUC | 0.9057 ± 0.0084 | **0.9167 ± 0.0045** | **+0.0110** *(p=0.007)* |
| Balanced Acc | 0.6909 ± 0.0361 | **0.7014 ± 0.0270** | +0.0105 |

### OCTMNIST (Đa lớp — 5.000 mẫu)
| Metric | Classical CNN | Quantum (Fixed) | Δ |
| :--- | :---: | :---: | :---: |
| ROC-AUC | **0.7490 ± 0.0238** | 0.6914 ± 0.0050 | -0.0576 *(p<0.001)* |
| Accuracy | **0.4492 ± 0.0184** | 0.4036 ± 0.0071 | -0.0456 |

*→ Luận điểm: Fixed Quantum vượt trội trên dữ liệu nhỏ (PR-AUC), thua trên dữ liệu lớn đa lớp → tiền đề cho Trainable Quanvolution ở GĐ3.*

---

## 🚀 Cách Chạy lại Thực nghiệm

```bash
# Từ thư mục GD1/
python run_all.py
```

Kết quả JSON lưu tại `results/`, biểu đồ tại `results/figures/`.

---

## 📂 Cấu trúc Thư mục GD1

```
GD1/
├── BAO_CAO_GIAI_DOAN_1.md      # Báo cáo nghiệm thu chi tiết
├── run_all.py                   # Master runner (1 lệnh chạy toàn bộ)
├── requirements.txt
├── src/
│   ├── train.py                 # Training loop 10 seeds
│   ├── data/
│   │   ├── medmnist_loader.py   # Pipeline nạp & chia tập
│   │   └── precompute_features.py
│   ├── models/
│   │   ├── classical_cnn.py     # Symmetrical Minimum CNN
│   │   └── quantum_model.py     # Fixed Quanvolution Classifier
│   ├── utils/
│   │   └── metrics.py           # 6 Medical Metrics
│   └── visual/
│       └── plot_results.py
└── results/
    ├── breastmnist_classical_latest.json
    ├── breastmnist_quantum_latest.json
    ├── octmnist_classical_latest.json
    ├── octmnist_quantum_latest.json
    └── figures/
        ├── breastmnist_benchmark_chart.png
        └── octmnist_benchmark_chart.png
```

---

*← Quay lại [GĐ0](../GD0/BAO_CAO_GIAI_DOAN_0.md) | Tiếp theo [GĐ2](../GD2/BAO_CAO_GIAI_DOAN_2.md) →*
