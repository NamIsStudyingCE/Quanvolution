> [!NOTE] **Báo cáo lịch sử GĐ3.** Toàn bộ số liệu chính thức cuối cùng (sample std ddof=1, p-value chuẩn) được chuẩn hóa tại `results/reconciliation_canonical.json` từ raw per-seed JSON, và được phản ánh đồng bộ trong Luận văn (GĐ4) và bài báo SOICT 2026. File này đã được chuẩn hóa theo canonical.

# GĐ3 — Tích chập Lượng tử Tự học & Ma trận Đối sánh 3 Tầng (Mốc M4 — 19/10)

> **Đề tài Luận văn Tốt nghiệp:** Nghiên cứu và ứng dụng lớp tích chập lượng tử (Quanvolutional Layer) trong bài toán phân loại ảnh y tế (MedMNIST), so sánh công bằng với các kiến trúc cổ điển (Classical CNN).

**Mục tiêu giai đoạn:** Xây dựng kiến trúc Trainable Quanvolution Network (QNN khả vi đầu-cuối), vận hành Ma trận Đối sánh 3 Tầng kiểm soát biến số và xác nhận mạch lượng tử tối ưu đối chứng với Classical CNN.

---

## 📌 Nhiệm vụ Giai đoạn 3 (Tuần 8 - Tuần 10)

| STT | Nhiệm vụ | Sản phẩm Bàn giao | Trạng thái |
| :---: | :--- | :--- | :---: |
| 1 | Kiến trúc QNN khả vi hoàn toàn (`backprop` trên PennyLane + PyTorch Autograd) | `src/models/trainable_quanv.py` | ✅ Hoàn thành |
| 2 | Ma trận Đối sánh 3 Tầng (110 lượt train trên 10 seeds đồng nhất) | `src/experiments/trainable_experiment.py` | ✅ Hoàn thành |
| 3 | Phân tích Gradient Dynamics & Sanity Check (≠ Barren Plateaus) | `src/visual/plot_gd3_dynamics.py` | ✅ Hoàn thành |
| 4 | Đo lường Tham số & Chi phí Phần cứng (Độ trễ suy luận CPU) | `measure_params_cost.py` | ✅ Hoàn thành |
| 5 | Báo cáo Nghiệm thu chuẩn hóa IMRaD (Giải quyết 5 điểm phản biện) | `GD3/BAO_CAO_GIAI_DOAN_3.md` | ✅ Hoàn thành |

---

## 🏗️ Ma trận 3 Tầng Đối sánh

| Tầng | Mục đích | Cặp đấu |
| :---: | :--- | :--- |
| **Tầng 1** | Cô lập biến "Trainability" (giữ nguyên kiến trúc) | Fixed Basic vs Trainable Basic |
| **Tầng 2** | Stress-test Quán quân GĐ2 | Trainable Basic vs Fixed Champion (`random_L1` / `basic_L2`) |
| **Tầng 3** | Full-Expressive Showdown → Winner vs Classical CNN | Fixed Strongly vs Trainable Strongly (3-Axis) |

---

## 📊 Kết quả Chính (10 Seeds Độc lập, 20 Epochs Đồng nhất)

### BreastMNIST (10 seeds, L=2)
| Mô hình | ROC-AUC | PR-AUC | Balanced Acc |
| :--- | :---: | :---: | :---: |
| Classical CNN | 0.8336 ± 0.0259 | 0.9041 ± 0.0100 | 0.6875 ± 0.0473 |
| **Fixed Basic L2** *(Quán quân ROC-AUC)* | **0.8521 ± 0.0095** | 0.9110 ± 0.0051 | 0.6816 ± 0.0517 |
| Trainable Basic L2 | 0.8406 ± 0.0252 | 0.9173 ± 0.0194 | 0.6732 ± 0.0403 |
| **Fixed Strongly L2** *(Quán quân PR-AUC)* | 0.8139 ± 0.0150 | **0.9182 ± 0.0071** | 0.6602 ± 0.0213 |
| Trainable Strongly L2 *(Thắng mạch cùng họ)* | 0.8306 ± 0.0294 | 0.9167 ± 0.0166 | **0.6945 ± 0.0451** |

*👉 Kết luận trung thực: Khả năng tự học (Trainability) giúp mạch 3 trục đánh bại phiên bản tĩnh của chính nó (Fixed Strongly). Mạch tĩnh `Fixed Basic L2` và `Fixed Strongly L2` đạt hiệu năng hàng đầu về ROC-AUC và PR-AUC với chi phí 0 tham số kernel.*

### OCTMNIST (10 seeds, L=1)
| Mô hình | ROC-AUC | PR-AUC | Balanced Acc |
| :--- | :---: | :---: | :---: |
| **Classical CNN** *(Dẫn đầu áp đảo)* | **0.7505 ± 0.0240** | **0.4991 ± 0.0297** | **0.4433 ± 0.0135** |
| Fixed Basic L1 | 0.6711 ± 0.0042 | 0.4186 ± 0.0074 | 0.4075 ± 0.0042 |
| Trainable Basic L1 | 0.6704 ± 0.0106 | 0.4102 ± 0.0131 | 0.3955 ± 0.0161 |
| Fixed Champion GĐ2 (`random_L1`) | 0.6912 ± 0.0071 | 0.4443 ± 0.0088 | 0.4048 ± 0.0130 |
| Fixed Strongly L1 | 0.6690 ± 0.0055 | 0.4175 ± 0.0047 | 0.4034 ± 0.0046 |
| Trainable Strongly L1 *(Thắng mạch cùng họ)* | 0.6922 ± 0.0199 | 0.4365 ± 0.0289 | 0.4020 ± 0.0148 |

*👉 Xác định Ranh giới Lợi thế Lượng tử (Quantum Boundary): QNN thể hiện ưu thế ở dữ liệu nhỏ/lệch lớp (tính ổn định cao ~3x, 0 tham số ở tầng đặc trưng), nhưng bộc lộ giới hạn dung lượng trước Classical CNN ở dữ liệu lớn/đa lớp.*

---

## 🔬 Gradient Dynamics & Chi Phí Phần Cứng

- Gradient norm: $\|\nabla_\theta \mathcal{L}\|_2 \approx 0.2$--$0.5$ (seed-mean; peak $\approx 1.3$) — Xác nhận không có hiện tượng triệt tiêu gradient.
- $\theta(t)$ hội tụ sau 12–15 epochs trên cả 2 datasets.
- Độ trễ suy luận trên CPU: Classical CNN = $0.310\text{ ms/ảnh}$; Quanvolution = $220.22\text{ ms/ảnh}$ (~710x chậm hơn).
- Hiệu quả tham số: Tầng đặc trưng lượng tử tĩnh dùng đúng **0 tham số học** (tiết kiệm 20 tham số conv).

---

## 🚀 Cách Chạy lại Thực nghiệm

```bash
# Từ thư mục gốc repo:
python run_gd3.py              # Master runner 3-Tier Tournament
python measure_params_cost.py  # Đo tham số và độ trễ suy luận
```

---

## 📂 Cấu trúc Thư mục GD3

```
GD3/
├── BAO_CAO_GIAI_DOAN_3.md              # Báo cáo nghiệm thu chi tiết (đã sửa 5 phản biện)
├── README.md                           # Hướng dẫn chi tiết Giai đoạn 3
├── run_gd3.py                           # Master runner 3-Tier Tournament
├── trainable_quanv.py                   # (copy) Kiến trúc QNN khả vi
├── trainable_experiment.py              # (copy) Experiment runner
├── plot_gd3_dynamics.py                 # (copy) Visualization module
├── full_trainable_breastmnist.json      # Raw data: 50 lượt × 6 metrics (10 seeds)
├── full_trainable_octmnist.json         # Raw data: 60 lượt × 6 metrics (10 seeds)
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
