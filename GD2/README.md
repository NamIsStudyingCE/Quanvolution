# GĐ2 — Khảo sát Mạch Lượng tử Tĩnh & Xác định Quán quân (Mốc M3 — 28/09)

> **Đề tài Luận văn Tốt nghiệp:** Nghiên cứu và ứng dụng lớp tích chập lượng tử (Quanvolutional Layer) trong bài toán phân loại ảnh y tế (MedMNIST), so sánh công bằng với các kiến trúc cổ điển (Classical CNN).

**Mục tiêu giai đoạn:** Circuit Ablation Study — Khảo sát 6 cấu hình mạch lượng tử tĩnh (Fixed Quanvolution), trực quan hóa Quantum Feature Maps, xác định mạch Quán quân, và Proof-of-Concept Trainable Quanvolution.

---

## 📌 Nhiệm vụ Giai đoạn 2 (Tuần 5 - Tuần 7)

| STT | Nhiệm vụ | Sản phẩm Bàn giao | Trạng thái |
| :---: | :--- | :--- | :---: |
| 1 | Trực quan hóa Feature Maps (Quantum vs Classical Conv) | `src/visual/visualize_features.py` | ✅ Hoàn thành |
| 2 | Khảo sát 6 biến thể mạch lượng tử (Circuit Ablation) | `src/experiments/circuit_ablation.py` | ✅ Hoàn thành |
| 3 | Xác nhận Quán quân & Kiểm định thống kê (10 seeds) | `results/circuit_ablation_champion_10seeds.json` | ✅ Hoàn thành |
| 4 | Trainable Quanvolution POC (Bonus) | `src/models/trainable_quanv.py` | ✅ Hoàn thành |

---

## 🏆 6 Mạch Khảo sát & Kết quả Trên BreastMNIST (5 seeds)

| Mạch | Kiến trúc | ROC-AUC (Mean) | Nhận xét |
| :--- | :--- | :---: | :--- |
| `random_L1` | RandomLayers L=1 | 0.8369 | Mạnh, ổn định |
| `random_L2` | RandomLayers L=2 | 0.8292 | — |
| **`basic_L2`** | BasicEntangler L=2 | **0.8497** | **🏆 Quán quân BreastMNIST** |
| `basic_L1` | BasicEntangler L=1 | 0.8304 | — |
| `strongly_L1` | StronglyEntangling L=1 | 0.8296 | — |
| `strongly_L2` | StronglyEntangling L=2 | 0.8237 | — |

**Quán quân BreastMNIST:** `basic_L2` — ROC-AUC 0.8497 vs Classical CNN 0.8307 (Paired t-test p=0.0309, Wilcoxon p=0.0273 ✅)

**Quán quân OCTMNIST:** `random_L1` — ROC-AUC 0.6922 (mạnh nhất trong tập lớn)

---

## 🚀 Cách Chạy lại Thực nghiệm

```bash
# Từ thư mục GD2/
python run_ablation.py        # Circuit Ablation Study (6 mạch)
python run_trainable.py       # Trainable Quanvolution POC
python run_all.py             # Classical + Fixed Quantum baseline
```

---

## 📂 Cấu trúc Thư mục GD2

```
GD2/
├── BAO_CAO_GIAI_DOAN_2.md                  # Báo cáo nghiệm thu chi tiết
├── run_ablation.py                          # Runner Circuit Ablation
├── run_trainable.py                         # Runner Trainable POC
├── run_all.py
├── requirements.txt
├── src/
│   ├── models/
│   │   ├── circuits.py                      # 6 cấu hình mạch lượng tử
│   │   ├── trainable_quanv.py               # Trainable QNN (POC)
│   │   ├── classical_cnn.py
│   │   └── quantum_model.py
│   ├── data/
│   │   ├── precompute_circuits.py           # Precompute 6 circuit features
│   │   └── medmnist_loader.py
│   ├── experiments/
│   │   ├── circuit_ablation.py              # 2-stage ablation runner
│   │   └── trainable_experiment.py
│   └── visual/
│       ├── visualize_features.py            # Feature map comparison
│       ├── plot_ablation.py
│       └── plot_trainable.py
└── results/
    ├── circuit_ablation_summary.json        # Tổng hợp 6 mạch × 5 seeds
    ├── circuit_ablation_champion_10seeds.json
    ├── trainable_poc_summary.json
    └── figures/
        ├── breastmnist_feature_comparison.png
        ├── octmnist_feature_comparison.png
        ├── circuit_ablation_breastmnist.png
        ├── circuit_ablation_octmnist.png
        ├── trainable_poc_curves.png
        └── trainable_poc_metrics.png
```

---

*← Quay lại [GĐ1](../GD1/BAO_CAO_GIAI_DOAN_1.md) | Tiếp theo [GĐ3](../GD3/BAO_CAO_GIAI_DOAN_3.md) →*
