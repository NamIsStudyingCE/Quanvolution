# TỔNG QUAN LÝ THUYẾT VỀ MẠNG TÍCH CHẬP LƯỢNG TỬ (QUANVOLUTIONAL NEURAL NETWORKS)
> **Tài liệu nghiên cứu Giai đoạn 0 — Chuẩn bị cho Mốc M1 (24/08)**  
> **Đề tài Luận văn:** Nghiên cứu và ứng dụng lớp tích chập lượng tử (Quanvolution) cho phân loại ảnh y tế (MedMNIST)  
> **Tác giả:** NamIsStudyingCE  

---

## 1. Đặt vấn đề & Động lực nghiên cứu
Trong lĩnh vực thị giác máy tính (Computer Vision), Mạng nơ-ron tích chập cổ điển (**CNNs - Convolutional Neural Networks**) đã trở thành tiêu chuẩn nhờ khả năng trích xuất đặc trưng không gian có tính cục bộ (local spatial patterns) thông qua các phép tích chập dạng trượt (sliding kernels). 

Trong kỷ nguyên máy tính lượng tử kích thước trung bình và có nhiễu (**NISQ - Noisy Intermediate-Scale Quantum**), việc áp dụng Học máy Lượng tử (QML) vào xử lý ảnh thường gặp phải hai rào cản lớn:
1. **Thiếu bộ nhớ lượng tử (QRAM):** Việc mã hóa toàn bộ một bức ảnh lớn (hàng triệu pixel) vào trạng thái lượng tử đòi hỏi số lượng qubit khổng lồ và độ sâu mạch vượt quá khả năng phần cứng hiện tại.
2. **Hiện tượng Barren Plateaus:** Các mạch lượng tử quá sâu hoặc nhận quá nhiều tham số toàn cục thường bị triệt tiêu gradient, khiến mô hình không thể huấn luyện.

Kiến trúc **Quanvolutional Neural Network (Quanvolution)** do *Henderson et al. (2019)* đề xuất ra đời nhằm giải quyết triệt để hai rào cản trên bằng cách kết hợp cơ chế quét cục bộ của CNN với năng lực biểu diễn phi tuyến trong không gian Hilbert của mạch lượng tử.

---

## 2. Bốn Khái niệm Cốt lõi của Lớp Quanvolution

### 2.1. Mạch lượng tử quét trên vùng ảnh cục bộ (Local Patches)
Thay vì nạp toàn bộ ảnh vào máy tính lượng tử, lớp tích chập lượng tử chỉ áp dụng một **mạch lượng tử nhỏ (ví dụ 4 qubits)** quét qua từng cửa sổ không gian nhỏ (ví dụ kích thước $2 \times 2$ pixel) với một bước nhảy cố định (stride).
* **Ưu điểm vượt trội:** Không cần QRAM, số lượng qubit cần dùng rất nhỏ ($N = n \times n$, với $n=2 \to 4\text{ qubits}$), mạch nông (shallow depth), cực kỳ thích hợp cho các thiết bị NISQ.

### 2.2. Mã hóa dữ liệu (Data Encoding / Embedding)
Mỗi giá trị pixel $x_j \in [0, 1]$ trong patch được ánh xạ thành trạng thái lượng tử của qubit thứ $j$. Trong đề tài này, ta sử dụng phương pháp **Mã hóa góc (Angle Encoding)** thông qua cổng xoay Pauli-Y:
$$|\psi_{\text{in}}\rangle = \bigotimes_{j=0}^{3} R_Y(\pi \cdot x_j)|0\rangle$$
*Cổng $R_Y(\theta) = \exp(-i \frac{\theta}{2} Y)$ ánh xạ giá trị thực của pixel sang góc quay trên hình cầu Bloch một cách trực quan và liên tục.*

### 2.3. Lớp biến đổi & Vướng víu lượng tử (Quantum Entanglement)
Sau khi mã hóa, trạng thái lượng tử đi qua một chuỗi các cổng lượng tử $U$ (bao gồm các cổng quay 1-qubit và cổng tương tác 2-qubit như CNOT, CZ):
$$|\psi_{\text{out}}\rangle = U |\psi_{\text{in}}\rangle$$
Lớp này tạo ra hiện tượng **vướng víu lượng tử (entanglement)** và tương quan phi tuyến giữa các pixel lân cận trong patch, ánh xạ dữ liệu vào không gian trạng thái Hilbert đa chiều mà các bộ lọc tuyến tính cổ điển khó mô phỏng hiệu quả.

### 2.4. Phép đo tạo Feature Maps (Measurement & Decoding)
Tại mỗi vị trí kernel, ta thực hiện đo giá trị kỳ vọng của toán tử Pauli-Z trên từng qubit $j$:
$$f_j = \langle \psi_{\text{out}} | Z_j | \psi_{\text{out}} \rangle \in [-1, 1]$$
Kết quả của phép đo trên 4 qubits tạo thành một vector 4 chiều. Khi kernel trượt qua toàn bộ ảnh $28 \times 28$ với stride $= 2$, ta thu được **4 kênh bản đồ đặc trưng lượng tử (Quantum Feature Maps)** kích thước $14 \times 14$.

---

## 3. Sơ đồ Pipeline Tổng thể (End-to-End Architecture)

```text
[ Ảnh Y tế Đầu vào ] (28x28x1)
        │
        ▼ (Trích xuất các Patch 2x2, Stride=2)
┌───────────────────────────────────────────────────────────┐
│              QUANTUM CIRCUIT (4 Qubits)                   │
│                                                           │
│  Pixel 0 ──[ RY(π·x0) ]───[                    ]───[ ⟨Z0⟩ ] ──► Feature Map 0 (14x14)
│  Pixel 1 ──[ RY(π·x1) ]───[   Random Layers    ]───[ ⟨Z1⟩ ] ──► Feature Map 1 (14x14)
│  Pixel 2 ──[ RY(π·x2) ]───[  (Entanglement)    ]───[ ⟨Z2⟩ ] ──► Feature Map 2 (14x14)
│  Pixel 3 ──[ RY(π·x3) ]───[                    ]───[ ⟨Z3⟩ ] ──► Feature Map 3 (14x14)
└───────────────────────────────────────────────────────────┘
        │
        ▼ (Tensor Đặc trưng: 4x14x14 = 784 chiều)
[ Mạng Nơ-ron Cổ điển / Classifier ] (Linear / MLP / Small CNN)
        │
        ▼
[ Dự đoán Phân loại ] (Ví dụ: Lành tính / Ác tính)
```

---

## 4. Phân biệt các Trường phái & Khái niệm Quan trọng

### 4.1. Quanvolutional NN (Henderson et al., 2019) vs. QCNN thật (Cong et al., 2019)
| Tiêu chí | **Quanvolutional NN** (Henderson 2019) | **QCNN thật** (Cong 2019) |
| :--- | :--- | :--- |
| **Kiến trúc** | Mạng lai (Hybrid): Quantum Filter + Classical Head. | Mạng lượng tử thuần / biến phân (Fully Quantum). |
| **Xử lý không gian** | Dùng mạch nhỏ trượt trên ảnh cổ điển 2D. | Mạch phân cấp (MERA/QEC) rút gọn qubit $N \to 1$. |
| **Mục đích sử dụng** | Trích xuất đặc trưng cho dữ liệu ảnh cổ điển (như MedMNIST). | Nhận diện pha lượng tử (QPR) hoặc mã sửa lỗi (QEC). |
| **Ứng dụng đề tài** | **Đây là mô hình chính của Luận văn**. | Dùng để đối chứng lý thuyết, tránh nhầm lẫn thuật ngữ. |

### 4.2. Random Circuit (Mạch Cố định) vs. Trainable Circuit (Mạch Học được)
* **Random Quanvolution (Mạch cố định):** Các góc quay và vị trí cổng lượng tử được khởi tạo ngẫu nhiên nhưng giữ cố định trong toàn bộ quá trình. 
  * *Lợi ích then chốt:* Cho phép **tiền tính toán (Precompute)** toàn bộ Feature Maps cho dataset **đúng 1 lần duy nhất**, giảm thời gian train mạng cổ điển từ nhiều ngày xuống vài giây.
* **Trainable Quanvolution (Mạch biến phân):** Các tham số cổng lượng tử được tối ưu cùng lúc với mạng cổ điển thông qua thuật toán Parameter-shift rule. (Đây là mục tiêu mở rộng / điểm cộng trong Giai đoạn 2 & 3).

---

## 5. Đánh giá Khách quan & Tính Trung thực Học thuật
Dựa trên bài báo gốc của Henderson (2019) và nhận xét của Giảng viên hướng dẫn:
1. **Không ngộ nhận "Quantum luôn thắng":** Việc thêm các phép biến đổi ngẫu nhiên lượng tử giúp tăng tính phi tuyến của đặc trưng, nhưng cần đối chứng công bằng với các mạng Classical CNN được tối ưu nghiêm túc.
2. **Đánh giá trên Dữ liệu Y tế Mất cân bằng:** Với MedMNIST, không dùng đơn độc chỉ số Accuracy mà bắt buộc phải báo cáo: *Balanced Accuracy, F1-Score, MCC, ROC-AUC, PR-AUC* và đo trên đa seed ($\ge 5$ seeds) kèm kiểm định thống kê *Wilcoxon*.

---
*Tài liệu tham khảo chính:*
1. M. Henderson et al., *"Quanvolutional Neural Networks: Powering Image Recognition with Quantum Circuits"*, arXiv:1904.04767 (2019).
2. I. Cong et al., *"Quantum Convolutional Neural Networks"*, Nature Physics / arXiv:1810.03787 (2019).
3. PennyLane Demos, *"Quanvolutional Neural Networks"*, pennylane.ai.
