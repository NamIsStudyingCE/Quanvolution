# BÁO CÁO NGHIỆM THU GIAI ĐOẠN 0 (MỐC M1 — 24/08)
**Đề tài:** Nghiên cứu và ứng dụng lớp tích chập lượng tử (Quanvolutional Neural Network) trong phân loại ảnh y tế (MedMNIST)

---

## 📌 1. Danh mục Nhiệm vụ & Sản phẩm Bàn giao

| STT | Nhiệm vụ theo Đề cương (Tuần 1 - Tuần 2) | Sản phẩm Bàn giao (File đính kèm) | Trạng thái |
| :---: | :--- | :--- | :---: |
| 1 | **Tóm tắt Lý thuyết & Phân biệt kiến trúc**: Nắm 4 khái niệm (patch quét, mã hóa pixel, đo kỳ vọng, random vs trainable circuit); Phân biệt Quanvolution (Henderson 2019) vs QCNN (Cong 2019). | `docs/theory_summary.md` | **Hoàn thành 100%** |
| 2 | **Khởi tạo Môi trường & Thiết lập Repo**: Python 3.10, PyTorch, PennyLane (`lightning.qubit`), MedMNIST, cấu trúc thư mục chuẩn, `.gitignore`. | `requirements.txt`<br>Repo: `https://github.com/NamIsStudyingCE/Quanvolution` | **Hoàn thành 100%** |
| 3 | **Chạy lại Demo & Tái hiện kết quả**: Tái hiện demo Quanvolution của PennyLane trên MNIST/FashionMNIST, lưu ảnh feature maps. | `quanvolution_demo.py`<br>`notebooks/00_quanvolution_demo.ipynb`<br>`quanvolution_features.png` | **Hoàn thành 100%** |
| 4 | **Chốt Dataset chính cho Giai đoạn 1**: Chọn 2 bộ dữ liệu MedMNIST theo khuyến nghị của GV. | Chốt: **BreastMNIST** (Nhị phân) & **OCTMNIST** (Đa lớp) | **Hoàn thành 100%** |

---

## 🔬 2. Tóm tắt Nội dung Lý thuyết Cốt lõi

1. **Quanvolutional Layer (Henderson et al., 2019)**:
   - Sử dụng một cửa sổ trượt $2 \times 2$ (4 pixel) làm đầu vào cho mạch lượng tử 4-qubit.
   - Mã hóa trạng thái: Sử dụng cổng quay $RY(\pi \cdot \phi)$ để đưa giá trị pixel $\phi \in [0, 1]$ vào trạng thái chồng chập lượng tử $|\psi\rangle$.
   - Tầng biến đổi lượng tử: Sử dụng các khối `RandomLayers` cố định không huấn luyện (fixed random unitary) để ánh xạ đặc trưng phi tuyến vào không gian Hilbert đa chiều.
   - Đo lường: Đo giá trị kỳ vọng Pauli-Z $\langle Z_i \rangle$ trên từng qubit để thu được 4 kênh đặc trưng đầu ra ($28 \times 28 \rightarrow 14 \times 14 \times 4$).

2. **Phân biệt Quanvolution (Henderson 2019) vs QCNN (Cong 2019)**:
   - **Quanvolution**: Mạch lượng tử đóng vai trò là **Bộ trích xuất đặc trưng (Feature Extractor)** cục bộ, không có tham số học được trong mạch, kết hợp với mạng nơ-ron cổ điển phía sau để phân loại.
   - **QCNN**: Toàn bộ mạng từ đầu đến cuối là mạch lượng tử biến phân (Variational Quantum Circuit) bao gồm các lớp tích chập lượng tử và lớp pooling lượng tử có tham số học được (trainable parameters).

---

## 💻 3. Hướng dẫn Tái hiện Demo GĐ0

Chạy script demo độc lập:
```bash
python quanvolution_demo.py
```
Kết quả trực quan hóa 4 kênh đặc trưng lượng tử trích xuất từ ảnh chữ số viết tay sẽ được lưu tại `quanvolution_features.png`.
