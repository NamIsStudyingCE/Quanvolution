# GĐ3 — Tích chập Lượng tử Tự học & Ma trận Đối sánh 3 Tầng (Mốc M4 — 19/10)

> **Đề tài Luận văn Tốt nghiệp:** Nghiên cứu và ứng dụng lớp tích chập lượng tử (Quanvolutional Layer) trong bài toán phân loại ảnh y tế (MedMNIST), so sánh công bằng với các kiến trúc cổ điển (Classical CNN).

**Mục tiêu giai đoạn:** Xây dựng kiến trúc Trainable Quanvolution Network (QNN khả vi đầu-cuối), vận hành Ma trận Đối sánh 3 Tầng kiểm soát biến số và xác nhận mạch lượng tử tối ưu đối chứng với Classical CNN.

---

## 📌 Nhiệm vụ Giai đoạn 3 (Tuần 8 - Tuần 10)

| STT | Nhiệm vụ | Sản phẩm Bàn giao | Trạng thái |
| :---: | :--- | :--- | :---: |
| 1 | Kiến trúc QNN khả vi hoàn toàn (`backprop` trên PennyLane + PyTorch Autograd) | `src/models/trainable_quanv.py` | ✅ Hoàn thành |
| 2 | Ma trận Đối sánh 3 Tầng (80 lượt train) | `src/experiments/trainable_experiment.py` | ✅ Hoàn thành |
| 3 | Phân tích Gradient Dynamics & chứng minh ≠ Barren Plateaus | `src/visual/plot_gd3_dynamics.py` | ✅ Hoàn thành |
| 4 | Xác định Winner Tầng 3 & đối sánh với Classical CNN | `run_gd3.py` | ✅ Hoàn thành |

---

## 🏗️ Ma trận 3 Tầng Đối sánh

| Tầng | Mục đích | Cặp đấu |
| :---: | :--- | :--- |
| **Tầng 1** | Cô lập biến "Trainability" (giữ nguyên kiến trúc) | Fixed Basic L2 vs Trainable Basic L2 |
| **Tầng 2** | Stress-test Quán quân GĐ2 | Trainable Basic vs Fixed Champion (`random_L1`) |
| **Tầng 3** | Full-Expressive Showdown → Winner vs Classical CNN | Fixed Strongly L2 vs Trainable Strongly L2 |

---

## 📊 Kết quả Chính (Winner Tầng 3: Trainable Strongly)

### BreastMNIST (10 seeds)
| Mô hình | ROC-AUC | PR-AUC | BAcc |
| :--- | :---: | :---: | :---: |
| Classical CNN | 0.8336 | 0.9041 | 0.6875 |
| **Fixed Basic L2** *(Mạch tĩnh mạnh nhất)* | **0.8521** | 0.9110 | 0.6816 |
| Fixed Strongly L2 | 0.8139 | 0.9182 | 0.6602 |
| **Trainable Strongly L2** *(Thắng mạch cùng họ)* | 0.8306 | **0.9167** | **0.6945** |

*👉 Kết luận trung thực: Khả năng tự học (Trainability) giúp mạch 3 trục đánh bại phiên bản tĩnh của chính nó (Fixed Strongly) và vượt Classical CNN về BAcc/PR-AUC. Tuy nhiên, nếu xét toàn diện, mạch tĩnh `Fixed Basic L2` vẫn đạt ROC-AUC cao nhất (0.8521).*

### OCTMNIST (10 seeds)
| Mô hình | ROC-AUC |
| :--- | :---: |
| **Classical CNN** *(Dẫn đầu)* | **0.7532** |
| Fixed Champion GĐ2 (`random_L1`) | 0.6922 (5-seed cũ) |
| Trainable Strongly L1 | 0.6829 (5-seed cũ) |

*👉 Xác định Boundary Condition: QNN thể hiện ưu thế ở dữ liệu nhỏ (tính ổn định cao, 0 tham số ở tầng đặc trưng), nhưng gặp giới hạn biểu diễn ở dữ liệu lớn/đa lớp so với Classical.*

---

## 🔬 Gradient Dynamics

- Gradient norm: $\|\nabla_\theta \mathcal{L}\|_2 \in [0.05, 0.25]$ — Không có Barren Plateaus.
- $\theta(t)$ hội tụ sau 12–15 epochs trên cả 2 datasets.
- Sai lệch `backprop` vs `parameter-shift`: $|\Delta| < 4.1 \times 10^{-8}$ (đẳng trị về toán học).

---

## 🚀 Cách Chạy lại Thực nghiệm

```bash
# Từ thư mục GD3/
python run_gd3.py
```

Tự động chạy toàn bộ 3-Tier Tournament, lưu JSON thô vào `full_trainable_breastmnist.json` / `full_trainable_octmnist.json`, và xuất 8 biểu đồ 300 DPI vào `figures/`.

---

## 📂 Cấu trúc Thư mục GD3

```
GD3/
├── BAO_CAO_GIAI_DOAN_3.md              # Báo cáo nghiệm thu chi tiết
├── run_gd3.py                           # Master runner 3-Tier Tournament
├── trainable_quanv.py                   # (copy) Kiến trúc QNN khả vi
├── trainable_experiment.py              # (copy) Experiment runner
├── plot_gd3_dynamics.py                 # (copy) Visualization module
├── full_trainable_breastmnist.json      # Raw data: 50 lượt × 6 metrics
├── full_trainable_octmnist.json         # Raw data: 30 lượt × 6 metrics
├── requirements.txt
├── src/
│   ├── models/
│   │   ├── trainable_quanv.py           # QNN: Basic & Strongly (3-axis)
│   │   ├── circuits.py
│   │   ├── classical_cnn.py
│   │   └── quantum_model.py
│   ├── experiments/
│   │   ├── trainable_experiment.py      # run_3tier_dataset_experiment()
│   │   └── circuit_ablation.py
│   └── visual/
│       └── plot_gd3_dynamics.py         # Loss, Theta, Gradient, Benchmark plots
├── figures/                             # 8 figures 300 DPI (gradient, theta, benchmark)
└── results/                             # Symlink/copy kết quả JSON
```

---

*← Quay lại [GĐ2](../GD2/BAO_CAO_GIAI_DOAN_2.md) | Tiếp theo: GĐ4 — Soạn thảo Luận văn & Bảo vệ →*
