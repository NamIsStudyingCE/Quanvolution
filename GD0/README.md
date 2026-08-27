# GĐ0 — Nền tảng, Môi trường & Demo Quanvolution (Mốc M1 — 24/08)

> **Đề tài Luận văn Tốt nghiệp:** Nghiên cứu và ứng dụng lớp tích chập lượng tử (Quanvolutional Layer) trong bài toán phân loại ảnh y tế (MedMNIST), so sánh công bằng với các kiến trúc cổ điển (Classical CNN).

**Mục tiêu giai đoạn:** Nắm vững cơ sở lý thuyết QML, thiết lập môi trường thực nghiệm chuẩn hóa, phân biệt Quanvolution (Henderson 2019) vs QCNN (Cong 2019), chạy demo tái hiện trích xuất đặc trưng lượng tử và chốt 2 bộ dữ liệu MedMNIST.

---

## 📌 Nhiệm vụ Giai đoạn 0 (Tuần 1 - Tuần 2)

| STT | Nhiệm vụ | Sản phẩm Bàn giao | Trạng thái |
| :--- :| :--- | :--- | :---: |
| 1 | Tổng quan Lý thuyết & Phân biệt kiến trúc | `docs/theory_summary.md` | ✅ Hoàn thành |
| 2 | Môi trường Python, PyTorch, PennyLane & Git Repo | `requirements.txt`, Repo GitHub | ✅ Hoàn thành |
| 3 | Script Demo Quanvolution & Notebook trực quan hóa | `quanvolution_demo.py`, `notebooks/00_quanvolution_demo.ipynb` | ✅ Hoàn thành |
| 4 | Chốt Dataset chính thức cho Luận văn | BreastMNIST (Nhị phân) & OCTMNIST (Đa lớp) | ✅ Hoàn thành |

---

## 🔬 Tóm tắt Lý thuyết & Nguyên lý Cốt lõi

1. **Quanvolutional Layer (Henderson et al., 2019):**
   - Patch cục bộ $2 \times 2$ (4 pixels) ánh xạ trực tiếp vào mạch lượng tử 4-qubit thông qua Angle Embedding $RY(\pi \cdot x_i)$.
   - Mạch lượng tử thực hiện các phép biến đổi vướng víu (Entanglement) trong không gian trạng thái Hilbert 16 chiều.
   - Phép đo kỳ vọng $\langle Z_i \rangle$ trên 4 qubits trích xuất ra 4 kênh bản đồ đặc trưng lượng tử ($14 \times 14 \times 4$).

2. **Phân biệt Quanvolution vs QCNN (Cong 2019):**
   - *Quanvolution:* Mạng lai (Hybrid QNN) kết hợp bộ lọc lượng tử cục bộ với bộ phân loại cổ điển (Classical Classifier Head).
   - *QCNN:* Mạng lượng tử biến phân toàn phần (Fully Quantum) thiết kế cho bài toán nhận diện pha vật lý.

---

## 🚀 Cách Chạy Demo

```bash
# Từ thư mục gốc:
python quanvolution_demo.py
```
Kết quả trực quan hóa đặc trưng lưu tại `quanvolution_features.png`.

---

## 📂 Cấu trúc Thư mục GD0

```
GD0/
├── BAO_CAO_GIAI_DOAN_0.md              # Báo cáo nghiệm thu chi tiết Mốc M1
├── README.md                           # Hướng dẫn Giai đoạn 0
├── quanvolution_demo.py                 # Script demo trích xuất đặc trưng
├── quanvolution_features.png            # Hình ảnh 4 kênh đặc trưng lượng tử
├── docs/
│   └── theory_summary.md               # Tài liệu tổng quan lý thuyết QNN
└── notebooks/
    └── 00_quanvolution_demo.ipynb       # Jupyter notebook thực hành
```

---

*Tiếp theo: [GĐ1 — Pipeline & Classical Baseline](../GD1/README.md) →*
