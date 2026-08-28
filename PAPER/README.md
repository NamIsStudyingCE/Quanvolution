# THƯ MỤC VIẾT BÀI BÁO KHOA HỌC (PAPER AUTHOR'S WORKSPACE)

> **Tiêu đề Bài báo:** *Trainable vs. Fixed Quanvolutional Filters for Medical Image Classification: A Fair, Reproducible Benchmark on MedMNIST*  
> **Tạp chí mục tiêu:** IEEE Access / MDPI Electronics (Độ dài chuẩn: ~8–10 trang)  
> **Tác giả:** NamIsStudyingCE (Trường Đại học Bách Khoa — ĐHQG-HCM)

---

## 📌 1. TỔNG QUAN TÀI NGUYÊN ĐÃ SẴN SÀNG

Toàn bộ sơ đồ, bảng biểu, số liệu kiểm định 10 seeds, khoảng tin cậy (CI 95%) và Effect Size (Cohen's $d$) đã được chuẩn bị đầy đủ 100%:

```
PAPER/
├── README.md                            # Bản hướng dẫn & ánh xạ nội dung IMRaD
├── figures/                             # Toàn bộ hình ảnh chất lượng cao 300 DPI
│   ├── Fig1_quanvolution_pipeline.png   # [HÌNH 1] Sơ đồ kiến trúc Pipeline (PNG 300 DPI)
│   ├── Fig1_quanvolution_pipeline.pdf   # [HÌNH 1] Sơ đồ kiến trúc Vector PDF
│   ├── Fig2_feature_comparison.png      # [HÌNH 2] Trực quan hóa Feature Maps (Ảnh gốc vs Quantum vs Conv)
│   ├── Fig3_breastmnist_benchmark.png   # [HÌNH 3a] Biểu đồ Benchmark BreastMNIST
│   ├── Fig3_octmnist_benchmark.png      # [HÌNH 3b] Biểu đồ Benchmark OCTMNIST
│   ├── Fig4a_breastmnist_curves.png     # [HÌNH 4a] Loss & Val AUC curves BreastMNIST
│   ├── Fig4b_octmnist_curves.png        # [HÌNH 4b] Loss & Val AUC curves OCTMNIST
│   ├── Fig4c_theta_trajectories.png     # [HÌNH 4c] Quỹ đạo tham số theta(t) hội tụ 12-15 epochs
│   └── Fig4d_gradient_norms.png         # [HÌNH 4d] Động học Gradient norm (0.05 - 0.25)
├── tables/                              # Toàn bộ bảng số liệu chuẩn hóa
│   ├── Table1_literature_comparison.md  # [BẢNG 1] Đối sánh các công trình liên quan (Related Work)
│   ├── Table2_parameter_split.md        # [BẢNG 2] Bóc tách tham số Feature Extractor vs Head (Fairness)
│   ├── Table3_breastmnist_results.md    # [BẢNG 3] Kết quả BreastMNIST + CI 95% + Cohen's d
│   ├── Table4_octmnist_results.md       # [BẢNG 4] Kết quả OCTMNIST + CI 95% + Cohen's d
│   └── Table5_latency_cost.md           # [BẢNG 5] Độ trễ suy luận CPU & Chi phí phần cứng
└── scripts/                             # Mã nguồn tự động đo đạc & tính toán
    ├── compute_effect_size_ci.py        # Script tính Cohen's d và 95% Confidence Intervals
    └── generate_pipeline_diagram.py     # Script vẽ Figure 1 tự động
```

---

## 🗺️ 2. BẢNG ÁNH XẠ NỘI DUNG IMRaD ↔ HÌNH / BẢNG NGUỒN

| Mục IMRaD | Nội dung trọng tâm cần viết | Hình / Bảng neo trong bài | File nguồn có sẵn |
| :--- | :--- | :---: | :--- |
| **Title & Abstract** | Tóm tắt bối cảnh, phương pháp 1:1, 10 seeds, 3 thông điệp take-home. | — | `README.md` |
| **1. Introduction** | Động lực QML y tế $\to$ Khoảng trống nghiên cứu $\to$ 4 đóng góp C1–C4. | — | `README.md` mục 1 |
| **2. Related Work** | Định vị bài báo với Henderson (2019), Cong (2019), HQCNN, Nature Sci Rep. | **Bảng 1** | `tables/Table1_literature_comparison.md` |
| **3. Method** | Kiến trúc 4-qubit, 3 họ Ansatz, Symmetrical Minimum CNN, 3-tier matrix. | **Hình 1**<br>**Hình 2**<br>**Bảng 2** | `figures/Fig1_quanvolution_pipeline.png`<br>`figures/Fig2_feature_comparison.png`<br>`tables/Table2_parameter_split.md` |
| **4. Experimental Setup** | BreastMNIST (780) + OCTMNIST (5.000 subset), 10 seeds, 20 epochs, Wilcoxon note. | — | `README.md` |
| **5. Results** | Đọc số khách quan: BreastMNIST, OCTMNIST, Trainability, Động học, Chi phí. | **Bảng 3**<br>**Bảng 4**<br>**Bảng 5**<br>**Hình 3**<br>**Hình 4** | `tables/Table3_breastmnist_results.md`<br>`tables/Table4_octmnist_results.md`<br>`tables/Table5_latency_cost.md`<br>`figures/Fig3_*`, `Fig4_*` |
| **6. Discussion** | Lý giải Data Regime, Quantum Inductive Bias, Ý nghĩa lâm sàng, Đánh đổi tốc độ. | — | Luận điểm trong `Table2`, `Table5` |
| **7. Threats & 8. Conclusion** | Hạn chế thực nghiệm, 3 take-home message, hướng mở phần cứng tương lai. | — | `README.md` |

---

## ⛔ 3. BỘ QUY TẮC "GUARDRAILS" BẮT BUỘC TUÂN THỦ

1. **BA ĐIỀU TUYỆT ĐỐI KHÔNG VIẾT:**
   - ❌ KHÔNG viết: *"Mạch Trainable 3-trục là mạch tốt nhất toàn diện"* (OCT hòa random_L1; Breast quán quân là Fixed Basic L2).
   - ❌ KHÔNG viết: *"Quanvolution vượt trội / thắng CNN cổ điển"* (OCT Classical 0.7505 áp đảo).
   - ❌ KHÔNG viết: *"Đã chứng minh không có Barren Plateau"* (4-qubit nông vốn không kỳ vọng có BP; chỉ gọi là Sanity Check).
2. **BA THÔNG ĐIỆP TAKE-HOME PHẢI CÓ (Abstract & Conclusion):**
   - 🎯 Ưu thế lượng tử phụ thuộc chặt chẽ vào chế độ dữ liệu (nhỏ/lệch lớp vs lớn/đa lớp).
   - 🎯 Trên dữ liệu ít mẫu, Quanvolution cho độ ổn định cao hơn ~3x (std nhỏ) với **0 tham số học ở tầng đặc trưng**.
   - 🎯 Trainability chỉ có giá trị khi so sánh trong cùng một họ mạch.
