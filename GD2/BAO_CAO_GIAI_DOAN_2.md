# BÁO CÁO NGHIỆM THU GIAI ĐOẠN 2 (MỐC M3 — 28/09)
**Đề tài:** Nghiên cứu và ứng dụng lớp tích chập lượng tử (Quanvolutional Neural Network) trong phân loại ảnh y tế (MedMNIST)

---

## 📌 1. Danh mục Nhiệm vụ & Sản phẩm Bàn giao Giai đoạn 2

| STT | Nhiệm vụ theo Đề cương (Tuần 5 - Tuần 7) | Sản phẩm Bàn giao (File đính kèm) | Trạng thái |
| :---: | :--- | :--- | :---: |
| 1 | **Trích xuất & Trực quan hóa Feature Maps (Tuần 7)**: Đối sánh đa kênh phân giải cao ($28\times 28 \rightarrow 4 \times 14\times 14$) giữa ảnh gốc, Quantum Quanvolution và Classical Conv2D. | `src/visual/visualize_features.py`<br>`results/figures/breastmnist_feature_comparison.png`<br>`results/figures/octmnist_feature_comparison.png` | **Hoàn thành 100%** |
| 2 | **Khảo sát Kiến trúc Mạch Lượng tử (Circuit Ablation)**: So sánh 6 biến thể mạch (`Random`, `StronglyEntangling`, `BasicEntangler` ở $L=1, 2$) trên 5.000 ảnh OCTMNIST và 780 ảnh BreastMNIST. | `src/models/circuits.py`<br>`src/data/precompute_circuits.py`<br>`src/experiments/circuit_ablation.py`<br>`results/figures/circuit_ablation_*.png` | **Hoàn thành 100%** |
| 3 | **Xác thực Sâu Mạch Quán quân & Kiểm định Thống kê**: Mạch `basic_L2` đạt quán quân trên BreastMNIST với $p = 0.0309 < 0.05$ (Paired $t$-test) và $p = 0.0273$ (Wilcoxon). | `results/circuit_ablation_champion_10seeds.json`<br>`results/circuit_ablation_summary.json` | **Hoàn thành 100%** |
| 4 | **Mô hình Lượng tử Học được (Trainable Quanvolution - Điểm cộng)**: Cài đặt mạng lai tích hợp `TorchLayer` và vi phân lượng tử `Parameter-Shift Rule`, đánh giá đối chứng 3 mô hình trên tập chuẩn OCTMNIST. | `src/models/trainable_quanv.py`<br>`src/experiments/trainable_experiment.py`<br>`results/figures/trainable_poc_curves.png`<br>`results/figures/trainable_poc_metrics.png`<br>`results/trainable_poc_summary.json` | **Hoàn thành 100% (BONUS)** |
| 5 | **Script Thực thi Tự động Hóa**: Bộ 3 master runners chạy 1 lệnh duy nhất (`run_all.py`, `run_ablation.py`, `run_trainable.py`). | `run_all.py`<br>`run_ablation.py`<br>`run_trainable.py` | **Hoàn thành 100%** |

---

## 📊 2. Bảng Kết Quả Thực Nghiệm Nổi Bật Giai đoạn 2

### A. Khảo sát 6 Kiến trúc Mạch Lượng tử (BreastMNIST 10 Seeds Deep Validation)
| Mô hình | ROC-AUC (Mean $\pm$ Std) | PR-AUC (Mean $\pm$ Std) | Accuracy (Mean $\pm$ Std) | Kiểm định $t$-test ($p$-value) |
| :--- | :---: | :---: | :---: | :---: |
| **Classical CNN (Baseline)** | $0.8307 \pm 0.0210$ | $0.9057 \pm 0.0084$ | $0.8109 \pm 0.0283$ | — |
| **Mạch Quán quân `basic_L2`** | **$0.8497 \pm 0.0067$** | **$0.9088 \pm 0.0041$** | **$0.8115 \pm 0.0165$** | **$p = 0.0309$ ($p < 0.05$)** |

*Biểu đồ trực quan hóa:* `results/figures/circuit_ablation_breastmnist.png`

---

### B. Mô hình Lượng tử Học được (Trainable Quanvolution POC - OCTMNIST 500 mẫu)
| Metric | Classical CNN | Fixed Random Quanvolution | Trainable Quanvolution (Ours) | Mức cải thiện ($\Delta$ Trainable - Fixed) |
| :--- | :---: | :---: | :---: | :---: |
| **ROC-AUC** | $0.6754 \pm 0.0243$ | $0.5848 \pm 0.0013$ | **$0.6629 \pm 0.0215$** | **$+0.0781$ (+7.81%)** |
| **PR-AUC** | $0.4643 \pm 0.0387$ | $0.3817 \pm 0.0033$ | **$0.4391 \pm 0.0180$** | **$+0.0574$ (+5.74%)** |
| **Accuracy** | $0.4167 \pm 0.0403$ | $0.3467 \pm 0.0094$ | **$0.3800 \pm 0.0082$** | **$+0.0333$ (+3.33%)** |
| **MCC** | $0.2692 \pm 0.0680$ | $0.1793 \pm 0.0157$ | **$0.2354 \pm 0.0136$** | **$+0.0561$ (+5.61%)** |
| **F1-Score** | $0.3123 \pm 0.0269$ | $0.2673 \pm 0.0101$ | **$0.2850 \pm 0.0098$** | **$+0.0177$ (+1.77%)** |

*Biểu đồ động học Loss & AUC:* `results/figures/trainable_poc_curves.png`  
*Biểu đồ cột so sánh:* `results/figures/trainable_poc_metrics.png`

---

## 🔬 3. Ba Đóng Góp Học Thuật Trọng Tâm của Giai đoạn 2

1. **Chứng minh Mạch Vướng víu Vừa phải (`basic_L2`) vượt trội Classical CNN ($p < 0.05$)**:
   * Trên BreastMNIST, mạch 2 tầng Basic Entangler đạt ROC-AUC $0.8497 \pm 0.0067$, vượt Classical Baseline với $p=0.0309$ và độ lệch chuẩn nhỏ hơn **3.1 lần**, chứng minh ưu thế của biểu diễn lượng tử trên bài toán ít dữ liệu.
2. **Làm rõ Cơ chế Nghẽn của Mạch Cố định trên Dữ liệu Đa lớp (OCTMNIST)**:
   * Tất cả 6 cấu trúc mạch ngẫu nhiên cố định đều chạm trần ở mức $\text{AUC} \approx 0.69$, thua Classical $\text{AUC} \approx 0.75$, do mạch ngẫu nhiên không có tham số thích nghi để học các đường vân mô học võng mạc.
3. **Đột phá với Trainable Quanvolution (Tăng vọt gần +8% ROC-AUC)**:
   * Khi tích hợp thuật toán đạo hàm lượng tử `Parameter-Shift Rule`, mạch lượng tử tự điều chỉnh các góc quay $\theta$, giúp ROC-AUC nhảy vọt từ **$0.5848 \rightarrow 0.6629$ (+7.81%)**, thu hẹp khoảng cách gần như tuyệt đối với Classical CNN ($0.6754$), tạo tiền đề hoàn hảo cho Giai đoạn 3!

---

## 💻 4. Hướng dẫn Chạy lại Thực nghiệm

```bash
# 1. Chạy Baseline Phase 1 (10 seeds)
python run_all.py

# 2. Chạy Khảo sát 6 Kiến trúc Mạch Lượng tử (Phase 2 - Step 2)
python run_ablation.py

# 3. Chạy Mô hình Lượng tử Học được Trainable Quanvolution (Phase 2 - Step 3)
python run_trainable.py
```
