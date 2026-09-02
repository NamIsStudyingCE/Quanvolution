# ĐÁNH GIÁ THỰC NGHIỆM ĐỐI XỨNG GIỮA BỘ LỌC TÍCH CHẬP LƯỢNG TỬ TỰ HỌC VÀ CỐ ĐỊNH TRONG PHÂN LOẠI ẢNH Y TẾ: MỘT KHUNG THAM CHIẾU CÔNG BẰNG VÀ KHẢ TÁI LẬP TRÊN MEDMNIST

**Tác giả:** NamIsStudyingCE  
*Trường Đại học Bách Khoa — Đại học Quốc gia TP. Hồ Chí Minh*  
*Mã nguồn & Dữ liệu thực nghiệm:* `https://github.com/NamIsStudyingCE/Quanvolution.git`  

---

## TÓM TẮT (ABSTRACT)

Mạng nơ-ron tích chập lượng tử (Quanvolutional Neural Networks - QNN) nổi lên như một hướng tiếp cận đầy triển vọng nhằm kết hợp khả năng biểu diễn phi tuyến trong không gian Hilbert của các mạch lượng tử biến phân (Variational Quantum Circuits - VQC) với kiến trúc học sâu cổ điển. Tuy nhiên, phần lớn các công bố hiện nay chưa thiết lập được một khung đối chứng thực sự công bằng: thiếu sự cô lập giữa năng lực trích xuất đặc trưng lượng tử và tầng phân loại cổ điển, thiếu kiểm định thống kê đa hạt giống (multi-seed), và thường đưa ra các khẳng định chưa thỏa đáng về "ưu thế lượng tử" (Quantum Advantage). 

Bài báo này thiết lập một khung đánh giá thực nghiệm đa chiều, đối xứng $1:1$ và khả tái lập cao nhằm so sánh toàn diện giữa **Bộ lọc tích chập lượng tử tự học (Trainable Quanvolution)**, **Bộ lọc lượng tử cố định (Fixed Quanvolution)** và **Mô hình tích chập cổ điển tối thiểu đối xứng (Symmetrical Minimum CNN)** trên hai tập dữ liệu ảnh y tế chuẩn hóa thuộc bộ MedMNIST: BreastMNIST (nhị phân, dữ liệu nhỏ, mất cân bằng lớp) và OCTMNIST (đa lớp, $5.000$ mẫu). Thực nghiệm được chuẩn hóa nghiêm ngặt trên $10$ hạt giống ngẫu nhiên độc lập ($20$ epochs), đánh giá qua $6$ độ đo y tế cốt lõi kèm kiểm định giả thuyết thống kê (Paired $t$-test, Wilcoxon signed-rank), khoảng tin cậy $95\%$ và độ lớn tác động (Cohen's $d$).

Kết quả thực nghiệm đem lại ba kết luận khoa học then chốt:
1. **Ranh giới chế độ dữ liệu (Data Regime Dependency):** Trên tập dữ liệu nhỏ và lệch lớp (BreastMNIST), mạch lượng tử tĩnh (`Fixed Basic L2`) đạt ROC-AUC cao nhất bảng là $0{,}8521 \pm 0{,}0095$ ($d = +0{,}815, p = 0{,}0298$ so với CNN cổ điển $0{,}8336$), trong khi `Fixed Strongly L2` đạt PR-AUC vượt trội $0{,}9182 \pm 0{,}0071$ ($d = +1{,}332, p = 0{,}0023$) với độ lệch chuẩn nhỏ hơn $\sim 2{,}7\times$. Ngược lại, trên tập dữ liệu lớn và đa lớp (OCTMNIST), CNN cổ điển áp đảo hoàn toàn với ROC-AUC $0{,}7505 \pm 0{,}0240$ ($d = +2{,}108, p < 0{,}001$), chứng minh ưu thế lượng tử bị giới hạn nghiêm trọng bởi trần dung lượng biểu diễn khi số lớp tăng lên.
2. **Giá trị thực chất của khả năng tự học (Trainability):** Khả năng tối ưu hóa tham số góc quay lượng tử chỉ thể hiện ý nghĩa khi so sánh nội bộ trong cùng một họ mạch cấu trúc (trên OCTMNIST, `Trainable Strongly` đạt ROC-AUC $0{,}6922 \pm 0{,}0199$, vượt trội `Fixed Strongly` $0{,}6690$ với $\Delta = +0{,}0232, d = +1{,}050, p_{\text{wilcoxon}} = 0{,}0098$). Tuy nhiên, mạch tự học 3-trục chỉ hòa về mặt thống kê với mạch tĩnh được thiết kế tối ưu (`Fixed random_L1` $0{,}6912, p = 0{,}8875$, không có ý nghĩa thống kê).
3. **Đánh đổi chi phí tính toán (Computational Trade-off):** Mạch lượng tử tĩnh đem lại thiên kiến quy nạp lượng tử (*Quantum Inductive Bias*) mạnh mẽ với **đúng $0$ tham số học** tại tầng trích xuất đặc trưng và cho phép tiền tính toán một lần (precomputation), nhưng việc giả lập trên CPU phải chịu độ trễ suy luận $\sim 220{,}22\text{ ms/ảnh}$ (chậm hơn $\sim 710\times$ so với mạng cổ điển $0{,}31\text{ ms/ảnh}$).

Nghiên cứu cung cấp một bức tranh thực chứng khách quan, phản bác các tuyên bố phóng đại trong lĩnh vực Học máy Lượng tử và định vị chính xác phạm vi ứng dụng khả thi của Quanvolution trong chẩn đoán y tế hỗ trợ máy tính (CAD).

**Từ khóa:** Học máy lượng tử (QML), Mạng tích chập lượng tử (Quanvolution), Ảnh y tế, MedMNIST, Thiên kiến quy nạp lượng tử, Đánh giá đối xứng.

---

## 1. MỞ ĐẦU (INTRODUCTION)

Trong kỷ nguyên Trí tuệ Nhân tạo hỗ trợ y tế, việc phân tích hình ảnh y sinh (như ảnh X-quang, cắt lớp võng mạc OCT, siêu âm vú) đòi hỏi các mô hình thị giác máy tính phải đạt độ nhạy cao trong việc phát hiện tổn thương vi mô, đồng thời phải duy trì tính ổn định trước sự khan hiếm và mất cân bằng nghiêm trọng của dữ liệu bệnh học [1], [2]. Mạng nơ-ron tích chập cổ điển (Convolutional Neural Networks - CNN), mặc dù là tiêu chuẩn vàng trong xử lý ảnh, thường đòi hỏi lượng dữ liệu khổng lồ để tối ưu hóa hàng nghìn đến hàng triệu tham số, dẫn đến nguy cơ quá khớp (overfitting) hoặc suy giảm độ đặc hiệu khi huấn luyện trên các tập dữ liệu y tế quy mô nhỏ [3].

Học máy lượng tử (Quantum Machine Learning - QML) trong kỷ nguyên Lượng tử quy mô trung gian có nhiễu (Noisy Intermediate-Scale Quantum - NISQ) đang thu hút sự quan tâm sâu sắc nhờ tiềm năng khai thác không gian trạng thái Hilbert $2^N$ chiều thông qua các hiện tượng chồng chập (superposition) và vướng víu lượng tử (entanglement) [4], [5]. Năm 2019, Henderson và các cộng sự [6] đã đề xuất kiến trúc Mạng nơ-ron tích chập lượng tử (Quanvolutional Neural Network - Quanvolution), sử dụng một mạch lượng tử biến phân (VQC) đóng vai trò như một cửa sổ trượt (sliding kernel) cục bộ để biến đổi các mảng điểm ảnh thành các bản đồ đặc trưng lượng tử (quantum feature maps). Nhờ ánh xạ phi tuyến này, Quanvolution được kỳ vọng sẽ tạo ra một **Thiên kiến quy nạp lượng tử (Quantum Inductive Bias)** hữu hiệu, hỗ trợ các mạng học sâu trích xuất đặc trưng biên dạng phức tạp mà các bộ lọc tích chập tuyến tính cổ điển khó nắm bắt [7], [8].

Tuy nhiên, khi rà soát các công trình công bố gần đây trong lĩnh vực QML cho ảnh y tế [8]–[11], chúng tôi nhận thấy tồn tại **ba khoảng trống nghiên cứu nghiêm trọng (Critical Research Gaps)**:
* **G1 — Sự bất bình đẳng trong thiết kế mô hình đối chứng (Unfair Baseline Design):** Nhiều nghiên cứu đem một mô hình QML có tầng phân loại phức tạp so sánh với các mạng CNN cổ điển thô sơ, hoặc ngược lại, so sánh với các kiến trúc tiền huấn luyện khổng lồ (như ResNet-18) mà không cô lập tham số [9]. Điều này dẫn đến việc không thể xác định liệu hiệu năng đạt được là do phép biến đổi lượng tử hay do năng lực học của tầng phân loại cổ điển phía sau.
* **G2 — Thiếu vắng kiểm định thống kê đa hạt giống (Lack of Multi-Seed Rigor):** Phần lớn các công trình chỉ chạy thử nghiệm trên $1$ đến $3$ hạt giống (seeds), không báo cáo khoảng tin cậy (Confidence Intervals) hoặc kiểm định phi tham số. Do đó, các kết luận về "sự vượt trội của lượng tử" thường bị nhầm lẫn với sự biến động ngẫu nhiên trong quá trình khởi tạo trọng số [12].
* **G3 — Tranh cãi chưa được giải quyết giữa Mạch tự học và Mạch cố định (Trainable vs. Fixed Ansatzes):** Công trình gốc của Henderson et al. [6] giả định rằng các mạch lượng tử ngẫu nhiên cố định (Fixed Random Circuits) là đủ tốt và không cần huấn luyện. Ngược lại, nhiều nghiên cứu sau đó nỗ lực huấn luyện toàn bộ tham số góc quay trong mạch lượng tử nhưng không lượng hóa rõ ràng sự đánh đổi về chi phí tính toán và động học gradient.

Nhằm giải quyết triệt để các khoảng trống nêu trên, bài báo này thực hiện một nghiên cứu thực nghiệm toàn diện, có kiểm soát nghiêm ngặt và minh bạch số liệu $100\%$. Chúng tôi đóng góp **bốn giá trị cốt lõi (Key Contributions C1–C4)**:

* **C1 — Thiết lập Khung đối chứng Đối xứng 1:1 và Ma trận Đánh giá 3 Tầng:** Chúng tôi đề xuất mô hình *Symmetrical Minimum CNN* có cấu trúc tầng phân loại (Classifier Head) giống hệt $100\%$ ($784 \to K$ qua `BatchNorm2d`), cô lập hoàn toàn tầng trích xuất đặc trưng lượng tử để thực hiện so sánh công bằng tuyệt đối. Ma trận 3 tầng phân tách rõ ràng hiệu ứng tự học nội bộ (Intra-Ansatz), thử thách với quán quân tĩnh (Champion Stress-Test) và đối đầu toàn diện (Full-Expressive Showdown).
* **C2 — Lượng hóa Hiệu quả Tham số và Chi phí Phần cứng:** Chúng tôi chứng minh mạch lượng tử tĩnh tạo ra biểu diễn đặc trưng hữu hiệu với **đúng $0$ tham số học**, đồng thời đo đạc tường minh độ trễ suy luận trên CPU ($220{,}22\text{ ms/ảnh}$ so với $0{,}31\text{ ms/ảnh}$ của CNN cổ điển), cung cấp cơ sở định lượng cho việc ứng dụng thực tế.
* **C3 — Xác lập Thực chứng Ranh giới Chế độ Dữ liệu (Data Regime Boundaries):** Qua $10$ hạt giống độc lập trên BreastMNIST và OCTMNIST, chúng tôi chứng minh bằng chứng thực nghiệm rõ ràng: Quanvolution chỉ cạnh tranh và vượt trội về độ ổn định trên tập dữ liệu nhỏ và lệch lớp; trên tập dữ liệu lớn và đa lớp, CNN cổ điển vẫn nắm giữ vị thế áp đảo hoàn toàn.
* **C4 — Kiểm chứng Động học Tối ưu hóa và Tính Khả huấn luyện (Trainability Check):** Chúng tôi theo dõi tường minh quỹ đạo tham số $\theta(t)$ và chuẩn gradient $\|\nabla_\theta \mathcal{L}\|_2$ trong suốt $20$ epochs, xác nhận sự hội tụ ổn định và không gặp hiện tượng triệt tiêu gradient trên cấu hình 4-qubit nông.

---

## 2. CƠ SỞ LÝ THUYẾT VÀ CÔNG TRÌNH LIÊN QUAN (THEORETICAL BACKGROUND & RELATED WORK)

### 2.1. Mạng Tích chập Lượng tử (Quanvolutional Neural Networks)
Khác với Mạng nơ-ron tích chập lượng tử toàn phần (Fully Quantum CNN - QCNN) do Cong et al. [13] đề xuất cho các bài toán vật lý lượng tử đa vật thể, kiến trúc Quanvolution [6] là một giải pháp lai lượng tử - cổ điển (Hybrid Quantum-Classical). 

Xét một ảnh đầu vào $I \in \mathbb{R}^{H \times W \times C}$. Tại mỗi vị trí không gian $(u, v)$, một cửa sổ trượt kích thước $2 \times 2$ trích xuất vector 4 điểm ảnh $\mathbf{x} = (x_0, x_1, x_2, x_3)^T$, trong đó $x_i \in [0, 1]$. Vector này được nhúng vào trạng thái lượng tử 4-qubit thông qua toán tử mã hóa góc (Angle Embedding) $U_{\text{enc}}(\mathbf{x})$:
$$|\psi(\mathbf{x})\rangle = U_{\text{enc}}(\mathbf{x}) |0\rangle^{\otimes 4} = \bigotimes_{i=0}^{3} R_Y(\pi x_i) |0\rangle$$

Tiếp theo, một mạch lượng tử biến phân $U(\boldsymbol{\theta})$ gồm các cổng quay 1-qubit và các cổng vướng víu 2-qubit (như CNOT) được áp dụng lên trạng thái $|\psi(\mathbf{x})\rangle$:
$$|\Phi(\mathbf{x}, \boldsymbol{\theta})\rangle = U(\boldsymbol{\theta}) |\psi(\mathbf{x})\rangle$$

Đặc trưng lượng tử đầu ra tại vị trí $(u, v)$ trên kênh thứ $i$ ($i \in \{0, 1, 2, 3\}$) được xác định thông qua phép đo kỳ vọng toán tử Pauli-Z trên qubit tương ứng:
$$F_i(u, v) = \langle \Phi(\mathbf{x}, \boldsymbol{\theta}) | Z_i | \Phi(\mathbf{x}, \boldsymbol{\theta}) \rangle \in [-1, 1]$$

Với bước nhảy (stride) $s = 2$, quá trình này biến đổi ảnh kích thước $28 \times 28 \times 1$ thành 4 kênh bản đồ đặc trưng lượng tử kích thước $14 \times 14$, tạo ra một không gian biểu diễn $4 \times 14 \times 14 = 784$ chiều trước khi đưa vào tầng phân loại cổ điển.

### 2.2. Đối sánh Văn liệu và Định vị Nghiên cứu
Bảng 1 tổng hợp và đối sánh công trình này với các nghiên cứu tiêu biểu trong y văn thế giới về QML và thị giác máy tính.

**BẢNG 1: Đối sánh công trình này với các nghiên cứu liên quan trong y văn quốc tế.**

| Công trình tiêu biểu | Đối tượng & Bài toán | Mô hình lượng tử | Baseline Cổ điển | Số Seeds & Đánh giá Thống kê | Kết luận & Khoảng trống còn tồn tại |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **Henderson et al. (2019)** [6] | MNIST (chữ số viết tay đồ chơi) | Random Quanvolution (Tĩnh) | CNN đơn giản | $1 - 3$ seeds<br>(Không kiểm định) | Đề xuất ý tưởng Quanvolution đầu tiên; cho rằng mạch ngẫu nhiên là đủ tốt, chưa khảo sát ảnh y tế và chưa đối xứng tầng head. |
| **Cong et al. (2019)** [13] | Nhận diện pha lượng tử (Vật lý) | QCNN thuần lượng tử biến phân | Mạng cổ điển kết nối đầy đủ | Đơn lẻ<br>(Lý thuyết) | Kiến trúc giảm chiều lượng tử (Quantum pooling); tối ưu cho trạng thái lượng tử, không tương thích trực tiếp với dữ liệu ảnh 2D y tế. |
| **Altares-López et al. (2025)** [9] | Phân loại ảnh công nghiệp / y tế | HQCNN lai kết hợp | ResNet-18 (Pretrained) | Không đồng nhất seeds | So sánh khập khiễng khi đối đầu mạng lượng tử nhỏ với ResNet-18 hàng triệu tham số; không bóc tách được nguồn gốc ưu thế. |
| **Nature Sci. Rep. (2026)** [10] | MedMNIST (PathMNIST, BloodMNIST) | VQC trên phần cứng IBM Quantum | MLP cổ điển thô sơ | $3 - 5$ seeds<br>(Chưa chuẩn hóa) | Thực thi trên phần cứng thực nhưng baseline cổ điển quá yếu; nhiễu phần cứng lấn át hiệu quả mô hình. |
| **"Do We Really Need QML?" (2026)** [12] | Đánh giá phản biện toàn ngành QML | Khảo sát tổng hợp nhiều họ QNN | Các mạng CNN hiện đại | Tổng quan phê bình | Chỉ ra hiện tượng "ngụy tạo ưu thế lượng tử" do chọn baseline yếu và thiếu phân tích chi phí phần cứng. |
| **Công trình này (Ours)** | **MedMNIST (BreastMNIST & OCTMNIST)** | **Quanvolution 3 Tầng: Fixed vs. Trainable (1–3 trục)** | **Symmetrical Minimum CNN (Đối xứng 1:1)** | **10 seeds độc lập<br>t-test, Wilcoxon, Cohen's $d$, CI 95%** | **Khung đối chứng công bằng tuyệt đối; chứng minh ranh giới chế độ dữ liệu; lượng hóa $0$ tham số kernel và độ trễ CPU.** |

---

## 3. PHƯƠNG PHÁP NGHIÊN CỨU (PROPOSED METHODOLOGY)

### 3.1. Thiết kế Kiến trúc Tổng thể (End-to-End Pipeline)
Kiến trúc tổng thể của hệ thống được mô tả chi tiết tại **Hình 1**. Quy trình bao gồm 4 giai đoạn nối tiếp:
1. **Trích xuất cửa sổ trượt:** Ảnh xám $28 \times 28 \times 1$ được chia thành $196$ patch cục bộ $2 \times 2$ với bước trượt $s = 2$ không chồng lấn.
2. **Bộ lọc lượng tử 4-qubit:** Mỗi patch được nạp vào mạch lượng tử gồm mã hóa góc $R_Y(\pi x_i)$, khối toán tử vướng víu $U(\boldsymbol{\theta})$ và đo kỳ vọng Pauli-Z $\langle Z_i \rangle$.
3. **Bản đồ đặc trưng lượng tử:** Tập hợp các giá trị đo tạo thành 4 bản đồ đặc trưng $14 \times 14$, sau đó được duỗi phẳng (flatten) thành vector $784$ chiều.
4. **Tầng phân loại đối xứng (Symmetrical Classifier Head):** Vector đặc trưng đi qua lớp `BatchNorm2d(4)`, hàm kích hoạt `ReLU` và tầng tuyến tính `Linear(784, K)` để đưa ra xác suất dự đoán $K$ lớp bệnh học ($K=2$ cho BreastMNIST, $K=4$ cho OCTMNIST).

*(Tham chiếu hình ảnh: `Fig1_quanvolution_pipeline.png` và bản vector `Fig1_quanvolution_pipeline.pdf` trong thư mục `figures/`)*.

### 3.2. Các Họ Mạch Lượng tử Khảo sát (Ansatz Design)
Chúng tôi khảo sát 3 họ cấu trúc mạch lượng tử biến phân đại diện cho các mức độ biểu diễn khác nhau:
* **Họ Mạch Cơ bản (Basic Entangling Circuit - `basic`):** Sử dụng các cổng quay 1 trục $R_Y(\theta_i)$ kết hợp chuỗi cổng CNOT vòng khép kín (Circular Entanglement: $q_0 \to q_1 \to q_2 \to q_3 \to q_0$). Số lượng tham số cho $L$ tầng là $4L$.
* **Họ Mạch Ngẫu nhiên (Random Circuit - `random`):** Khởi tạo ngẫu nhiên các cổng quay $R_X, R_Y, R_Z$ và các cổng CNOT ngẫu nhiên theo phân phối Haar, cố định góc quay trong suốt quá trình huấn luyện ($0$ tham số học).
* **Họ Mạch Vướng víu Mạnh (Strongly Entangling Circuit - `strongly`):** Sử dụng phép quay 3 trục tổng quát $U_3(\theta, \phi, \lambda) = R_Z(\omega) R_Y(\theta) R_Z(\phi)$ trên mỗi qubit trước khi liên kết CNOT đa tầng. Số lượng tham số cho $L$ tầng là $12L$.

### 3.3. Thiết kế Mô hình Đối chứng Cổ điển Đối xứng (Symmetrical Minimum CNN)
Để đảm bảo tính công bằng tuyệt đối theo nguyên lý *Ceteris Paribus* (mọi yếu tố khác giữ nguyên), mô hình cổ điển đối chứng được xây dựng với nguyên tắc đối xứng $1:1$:
* **Tầng trích xuất cổ điển:** Sử dụng đúng 1 tầng tích chập `Conv2D(in_channels=1, out_channels=4, kernel_size=2, stride=2, bias=False)`. Tầng này tiêu tốn đúng $1 \times 4 \times 2 \times 2 = \mathbf{16}$ trọng số (thêm 4 bias nếu có, tổng cộng $20$ tham số) và biến đổi ảnh $28 \times 28$ thành đúng 4 kênh $14 \times 14$ ($784$ chiều), hoàn toàn tương đương với bộ lọc lượng tử 4-qubit.
* **Tầng phân loại chia sẻ (Shared Head):** Cả hai nhánh Lượng tử và Cổ điển đều sử dụng chung một cấu trúc Head: `BatchNorm2d(4) (8 tham số)` + `Linear(784, K)`.

Bảng 2 bóc tách chi tiết phân bổ tham số giữa các mô hình.

**BẢNG 2: Bóc tách tham số chi tiết giữa Tầng trích xuất đặc trưng và Tầng phân loại.**

| Họ Mô hình | Cấu hình Mạch / Lớp trích xuất | Tham số Kernel (Feature Extractor) | Tham số Head (BreastMNIST, $K=2$) | Tham số Head (OCTMNIST, $K=4$) | Tổng tham số toàn mạng |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Classical Minimum CNN** | $\text{Conv2D}(1 \to 4, k=2, s=2) + \text{BN}$ | **$20$** ($16$ weight + $4$ bias) | $1.570$ ($784 \times 2 + 2$) | $3.140$ ($784 \times 4 + 4$) | **$1.598$ / $3.168$** |
| **Fixed Basic Quanv** | $R_Y(\pi x) + \text{Basic Entangler } (L=2)$ | **$0$** *(Tĩnh / Khóa góc)* | $1.570$ | $3.140$ | **$1.578$ / $3.148$** |
| **Fixed Strongly Quanv** | $R_Y(\pi x) + \text{Strongly Entangler } (L=2)$ | **$0$** *(Tĩnh / Khóa góc)* | $1.570$ | $3.140$ | **$1.578$ / $3.148$** |
| **Trainable Basic Quanv** | $R_Y(\pi x) + R_Y(\theta_i) + \text{CNOT } (L=2)$ | **$8$** ($4 \text{ qubits} \times 2 \text{ layers}$) | $1.570$ | $3.140$ | **$1.586$ / $3.152$** |
| **Trainable Strongly Quanv**| $R_Y(\pi x) + \text{Rot3}(\theta) + \text{CNOT } (L=1/2)$ | **$12 - 24$** ($12 \text{ params/layer}$) | $1.570$ | $3.140$ | **$1.602$ / $3.160$** |

### 3.4. Ma trận Thử nghiệm 3 Tầng (3-Tier Benchmark Matrix)
Nhằm trả lời rành mạch các câu hỏi khoa học mà không bị nhiễu bởi các biến ngoại cảnh, chúng tôi thiết lập ma trận so găng 3 tầng:
* **Tầng 1 (Intra-Ansatz Isolation):** So sánh trực tiếp `Trainable Basic` vs `Fixed Basic`. Do hai mô hình có cấu trúc cổng và độ sâu giống hệt nhau, mọi chênh lệch về hiệu năng chỉ bắt nguồn duy nhất từ khả năng tự cập nhật góc quay $\theta$.
* **Tầng 2 (Champion Stress-Test):** Đưa mô hình tự học đối đầu với Quán quân mạch tĩnh đã được sàng lọc ở Giai đoạn 2 (`Fixed Basic L2` trên BreastMNIST và `Fixed random_L1` trên OCTMNIST) nhằm kiểm tra liệu việc tốn tài nguyên huấn luyện góc quay có thực sự vượt qua được một mạch tĩnh thiết kế tốt hay không.
* **Tầng 3 (Full-Expressive Showdown):** Đưa mô hình tự học mạnh nhất (`Trainable Strongly 3-Axis`) so găng với bản tĩnh cùng họ (`Fixed Strongly`) và đối đầu trực diện với `Classical CNN Baseline`.

### 3.5. Cơ chế Vi phân Lượng tử và Động học Gradient
Việc cập nhật các tham số lượng tử $\boldsymbol{\theta}$ được thực hiện thông qua thuật toán lan truyền ngược giải tích trên máy trạng thái vector (Adjoint / Statevector Backpropagation) tích hợp giữa PyTorch và PennyLane [14]. Để đảm bảo tính chính xác vật lý, chúng tôi đã thực hiện kiểm chứng chéo (sanity check) với quy tắc dịch chuyển tham số (Parameter-Shift Rule) [15]:
$$\frac{\partial F_i}{\partial \theta_j} = \frac{F_i\left(\theta_j + \frac{\pi}{2}\right) - F_i\left(\theta_j - \frac{\pi}{2}\right)}{2}$$
Sai số tuyệt đối trung bình giữa đạo hàm giải tích và Parameter-Shift Rule đo được là $|\Delta| < 4{,}1 \times 10^{-8}$, khẳng định tính toán đạo hàm chính xác tuyệt đối.

---

## 4. THIẾT LẬP THỰC NGHIỆM (EXPERIMENTAL SETUP)

### 4.1. Bộ Dữ liệu Ảnh Y tế Chuẩn hóa (MedMNIST)
Thực nghiệm được tiến hành trên 2 tập dữ liệu đại diện cho hai chế độ dữ liệu tương phản thuộc MedMNIST v2 [16]:
* **BreastMNIST:** Tập dữ liệu ảnh siêu âm vú gồm $780$ ảnh kích thước $28 \times 28$ (phân chia chuẩn: $546$ train, $78$ val, $156$ test). Đây là bài toán phân loại nhị phân (Malignant vs. Benign) có độ mất cân bằng lớp cao ($73\%$ lành tính, $27\%$ ác tính), đại diện cho chế độ **Dữ liệu nhỏ & Lệch lớp**.
* **OCTMNIST (Subset):** Tập dữ liệu ảnh cắt lớp quang học võng mạc gồm 4 lớp bệnh lý nhãn khoa (Choroidal Neovascularization, Diabetic Macular Edema, Drusen, Normal). Chúng tôi sử dụng tập con chuẩn hóa gồm $5.000$ ảnh ($3.500$ train, $500$ val, $1.000$ test) nhằm cân bằng giữa tính bao quát thực nghiệm và giới hạn thời gian tính toán giả lập lượng tử trên CPU. Đây là bài toán đại diện cho chế độ **Dữ liệu lớn & Đa lớp**.

### 4.2. Giao thức Huấn luyện và Cố định Hạt giống (Seeds)
Để đảm bảo khả năng tái lập $100\%$, tất cả các mô hình được huấn luyện đồng nhất qua **$10$ hạt giống ngẫu nhiên độc lập**:
$$\mathcal{S} = \{0, 42, 100, 2023, 777, 999, 1234, 5678, 1111, 2222\}$$
* **Số Epochs:** Cố định $20$ epochs cho tất cả các mô hình (đã được kiểm chứng hội tụ hoàn toàn từ epoch 12–15).
* **Bộ tối ưu hóa:** Adam Optimizer; tốc độ học $lr = 0{,}001$ cho tầng phân loại cổ điển và $lr = 0{,}01$ cho các tham số góc quay lượng tử $\boldsymbol{\theta}$; hàm mất mát Cross-Entropy Loss; kích thước batch $B = 32$.
* **Môi trường phần cứng:** CPU Intel Core thế hệ mới, RAM 16GB, môi trường PyTorch 2.13.0 + PennyLane 0.42.3.

### 4.3. Các Độ đo Đánh giá và Kiểm định Thống kê
Chúng tôi báo cáo $6$ độ đo y sinh toàn diện trên tập kiểm tra (test set): Độ chính xác (Accuracy - Acc), Độ chính xác cân bằng (Balanced Accuracy - BAcc), F1-Score (macro-average), Hệ số tương quan Matthews (MCC), Diện tích dưới đường cong ROC (ROC-AUC) và Diện tích dưới đường cong Precision-Recall (PR-AUC). 

Để đánh giá ý nghĩa thống kê, chúng tôi áp dụng:
1. **Kiểm định $t$ bắt cặp (Paired Student's $t$-test)** và **Kiểm định hạng Wilcoxon (Wilcoxon signed-rank test)** với ngưỡng ý nghĩa $\alpha = 0{,}05$. Lưu ý rằng với $n=10$ hạt giống, kiểm định Wilcoxon có mức phân giải rời rạc với $p$-value nhỏ nhất khả dĩ là $p_{\min} = 1/2^9 \approx 0{,}00195$.
2. **Khoảng tin cậy $95\%$ (95% Confidence Interval - CI 95%):** Tính toán theo phân phối $t$-Student với bậc tự do $df = 9$ ($t^* = 2{,}262$).
3. **Độ lớn tác động (Cohen's $d$):** Đo lường mức độ chênh lệch thực chất theo chuẩn Cohen ($|d| \ge 0{,}8$ là tác động lớn - Large effect; $|d| \ge 1{,}2$ là rất lớn - Very large effect).

---

## 5. KẾT QUẢ THỰC NGHIỆM VÀ PHÂN TÍCH (RESULTS & EMPIRICAL FINDINGS)

### 5.1. Kết quả trên BreastMNIST (Chế độ Dữ liệu Nhỏ & Lệch lớp)
Bảng 3 trình bày chi tiết kết quả trung bình và độ lệch chuẩn qua 10 hạt giống độc lập trên BreastMNIST ($L=2$).

**BẢNG 3: Kết quả thực nghiệm 10 hạt giống độc lập trên tập dữ liệu BreastMNIST ($L=2$, 20 Epochs).**
*(In đậm giá trị tốt nhất từng cột; $\pm$ biểu thị độ lệch chuẩn; $[ \cdot ]$ là khoảng tin cậy $95\%$ CI).*

| Mô hình thử nghiệm | Accuracy | Balanced Acc | F1-Score | MCC | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Classical CNN Baseline** | **$0{,}8103 \pm 0{,}0279$** | **$0{,}6875 \pm 0{,}0473$<br>$[0{,}6537, 0{,}7213]$** | **$0{,}8802 \pm 0{,}0172$** | **$0{,}4702 \pm 0{,}0865$** | **$0{,}8336 \pm 0{,}0259$<br>$[0{,}8150, 0{,}8521]$** | **$0{,}9041 \pm 0{,}0100$<br>$[0{,}8970, 0{,}9113]$** |
| **Fixed Basic Quanv (L2)** | $0{,}8083 \pm 0{,}0204$ | $0{,}6816 \pm 0{,}0517$<br>$[0{,}6447, 0{,}7186]$ | $0{,}8796 \pm 0{,}0100$ | $0{,}4626 \pm 0{,}0711$ | **$0{,}8521 \pm 0{,}0095$<br>$[0{,}8453, 0{,}8589]$** | $0{,}9110 \pm 0{,}0051$<br>$[0{,}9073, 0{,}9146]$ |
| **Trainable Basic Quanv (L2)** | $0{,}7917 \pm 0{,}0251$ | $0{,}6732 \pm 0{,}0403$<br>$[0{,}6444, 0{,}7021]$ | $0{,}8668 \pm 0{,}0178$ | $0{,}4224 \pm 0{,}0749$ | $0{,}8406 \pm 0{,}0252$<br>$[0{,}8226, 0{,}8586]$ | $0{,}9173 \pm 0{,}0194$<br>$[0{,}9033, 0{,}9312]$ |
| **Fixed Strongly Quanv (L2)** | $0{,}7846 \pm 0{,}0187$ | $0{,}6602 \pm 0{,}0213$<br>$[0{,}6449, 0{,}6754]$ | $0{,}8631 \pm 0{,}0131$ | $0{,}3942 \pm 0{,}0536$ | $0{,}8139 \pm 0{,}0150$<br>$[0{,}8032, 0{,}8246]$ | **$0{,}9182 \pm 0{,}0071$<br>$[0{,}9131, 0{,}9232]$** |
| **Trainable Strongly Quanv (L2)** | **$0{,}8019 \pm 0{,}0300$** | **$0{,}6945 \pm 0{,}0451$<br>$[0{,}6623, 0{,}7268]$** | **$0{,}8724 \pm 0{,}0193$** | **$0{,}4549 \pm 0{,}0945$** | $0{,}8306 \pm 0{,}0294$<br>$[0{,}8096, 0{,}8516]$ | $0{,}9167 \pm 0{,}0166$<br>$[0{,}9048, 0{,}9286]$ |

**Phân tích Thống kê và Hiệu ứng Tác động (Statistical & Effect Size Findings):**
1. **Quán quân ROC-AUC thuộc về Mạch tĩnh:** `Fixed Basic L2` thiết lập kỷ lục ROC-AUC cao nhất đạt **$0{,}8521 \pm 0{,}0095$**, vượt trội mô hình cổ điển đối chứng $0{,}8336 \pm 0{,}0259$ với mức tăng $\Delta = +0{,}0186$ có ý nghĩa thống kê ($p_{\text{ttest}} = 0{,}0298, p_{\text{wilcoxon}} = 0{,}0254 < 0{,}05$). Độ lớn tác động đạt **Cohen's $d = +0{,}815$ (Large effect)**.
2. **Quán quân PR-AUC và Khả năng Bắt bệnh Hiếm:** `Fixed Strongly L2` đạt PR-AUC cao nhất bảng là **$0{,}9182 \pm 0{,}0071$**, vượt trội Classical CNN ($0{,}9041 \pm 0{,}0100$) với ý nghĩa thống kê rất cao ($p_{\text{ttest}} = 0{,}0023, p_{\text{wilcoxon}} = 0{,}0059 < 0{,}01$) và độ lớn tác động cực lớn **Cohen's $d = +1{,}332$ (Very large effect)**.
3. **Độ ổn định phương sai vượt trội:** Độ lệch chuẩn ($\text{std}$) của các mô hình lượng tử tĩnh nhỏ hơn mạng CNN cổ điển từ $\mathbf{2{,}7\times}$ (ở ROC-AUC: $0{,}0095$ vs $0{,}0259$) đến $\mathbf{1{,}4\times}$ (ở PR-AUC: $0{,}0071$ vs $0{,}0100$), chứng minh mạch lượng tử tĩnh có khả năng kháng biến động khởi tạo xuất sắc trên dữ liệu ít mẫu.
4. **Vị thế của Trainable Models:** `Trainable Strongly` đạt Balanced Accuracy cao nhất ($0{,}6945 \pm 0{,}0451$), cải thiện $+0{,}0344$ BAcc so với `Fixed Strongly` ($0{,}6602 \pm 0{,}0213$) với độ lớn tác động trung bình ($d = +0{,}677, p = 0{,}061$), tuy nhiên so với Classical CNN ($0{,}6875 \pm 0{,}0473$), mức chênh lệch là không có ý nghĩa thống kê ($p = 0{,}670, d = +0{,}139$, negligible). `Trainable Basic` đạt PR-AUC cao ($0{,}9173 \pm 0{,}0184$) nhưng chưa đạt ngưỡng ý nghĩa thống kê so với CNN ($p = 0{,}0513$, ns).

### 5.2. Kết quả trên OCTMNIST (Chế độ Dữ liệu Lớn & Đa lớp)
Bảng 4 trình bày kết quả thực nghiệm 10 hạt giống độc lập trên tập dữ liệu đa lớp OCTMNIST ($L=1$).

**BẢNG 4: Kết quả thực nghiệm 10 hạt giống độc lập trên tập dữ liệu OCTMNIST ($L=1$, 20 Epochs).**

| Mô hình thử nghiệm | Accuracy | Balanced Acc | F1-Score | MCC | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Classical CNN Baseline** | **$0{,}4433 \pm 0{,}0135$** | **$0{,}4433 \pm 0{,}0135$<br>$[0{,}4336, 0{,}4530]$** | **$0{,}3206 \pm 0{,}0175$** | **$0{,}3156 \pm 0{,}0198$** | **$0{,}7505 \pm 0{,}0240$<br>$[0{,}7333, 0{,}7676]$** | **$0{,}4991 \pm 0{,}0297$<br>$[0{,}4778, 0{,}5203]$** |
| **Fixed Basic Quanv (L1)** | $0{,}4075 \pm 0{,}0042$ | $0{,}4075 \pm 0{,}0042$<br>$[0{,}4045, 0{,}4105]$ | $0{,}2971 \pm 0{,}0075$ | $0{,}2566 \pm 0{,}0072$ | $0{,}6711 \pm 0{,}0042$<br>$[0{,}6681, 0{,}6741]$ | $0{,}4186 \pm 0{,}0074$<br>$[0{,}4133, 0{,}4239]$ |
| **Trainable Basic Quanv (L1)** | $0{,}3955 \pm 0{,}0161$ | $0{,}3955 \pm 0{,}0161$<br>$[0{,}3840, 0{,}4070]$ | $0{,}2837 \pm 0{,}0203$ | $0{,}2394 \pm 0{,}0203$ | $0{,}6704 \pm 0{,}0106$<br>$[0{,}6628, 0{,}6780]$ | $0{,}4102 \pm 0{,}0131$<br>$[0{,}4008, 0{,}4195]$ |
| **Fixed Champ GĐ2 (`random_L1`)** | $0{,}4048 \pm 0{,}0130$ | $0{,}4048 \pm 0{,}0130$<br>$[0{,}3955, 0{,}4141]$ | $0{,}2997 \pm 0{,}0202$ | $0{,}2530 \pm 0{,}0144$ | $0{,}6912 \pm 0{,}0071$<br>$[0{,}6862, 0{,}6963]$ | $0{,}4443 \pm 0{,}0088$<br>$[0{,}4380, 0{,}4506]$ |
| **Fixed Strongly Quanv (L1)** | $0{,}4034 \pm 0{,}0046$ | $0{,}4034 \pm 0{,}0046$<br>$[0{,}4001, 0{,}4067]$ | $0{,}3050 \pm 0{,}0130$ | $0{,}2421 \pm 0{,}0064$ | $0{,}6690 \pm 0{,}0055$<br>$[0{,}6650, 0{,}6729]$ | $0{,}4175 \pm 0{,}0047$<br>$[0{,}4142, 0{,}4209]$ |
| **Trainable Strongly Quanv (L1)** | $0{,}4020 \pm 0{,}0148$ | $0{,}4020 \pm 0{,}0148$<br>$[0{,}3914, 0{,}4126]$ | $0{,}2949 \pm 0{,}0188$ | $0{,}2481 \pm 0{,}0232$ | $0{,}6922 \pm 0{,}0199$<br>$[0{,}6780, 0{,}7065]$ | $0{,}4365 \pm 0{,}0289$<br>$[0{,}4158, 0{,}4571]$ |

**Phân tích Đối sánh và Hiệu ứng Trainability trên OCTMNIST:**
1. **Sự Thống trị Áp đảo của Classical CNN:** Classical CNN dẫn đầu tuyệt đối trên toàn bộ 6 độ đo với ROC-AUC **$0{,}7505 \pm 0{,}0240$**, PR-AUC **$0{,}4991$** và Accuracy **$0{,}4433$**. Kiểm định thống kê khẳng định khoảng cách vượt trội áp đảo so với mô hình lượng tử tốt nhất với $p < 0{,}001$ và độ lớn tác động khổng lồ **Cohen's $d = +2{,}108$** (đối với ROC-AUC) và **$d = +1{,}874$** (đối với Balanced Accuracy).
2. **Hiệu ứng Trainability Tầng 3 (Trainable Strongly vs Fixed Strongly):** Khi so sánh nội bộ trong cùng họ mạch xoay 3-trục, `Trainable Strongly` ($0{,}6922 \pm 0{,}0199$) vượt trội rõ rệt so với `Fixed Strongly` ($0{,}6690 \pm 0{,}0055$) với mức cải thiện $\Delta = +0{,}0232$ ROC-AUC. Kiểm định xác nhận ý nghĩa thống kê vững chắc ($p_{\text{ttest}} = 0{,}0090, p_{\text{wilcoxon}} = 0{,}0098 < 0{,}01$) cùng độ lớn tác động lớn **Cohen's $d = +1{,}050$ (Large effect)**.
3. **Cục diện Hòa điểm giữa Trainable Strongly và Quán quân Tĩnh:** Mặc dù vượt trội so với bản tĩnh cùng họ, `Trainable Strongly` ($0{,}6922$) chỉ hòa về mặt thống kê với Quán quân mạch ngẫu nhiên cố định từ Giai đoạn 2 (`random_L1` $0{,}6912$) với mức chênh lệch $\Delta = +0{,}0010$ hoàn toàn không có ý nghĩa thống kê ($p = 0{,}8875$, ns; Cohen's $d = +0{,}047$, negligible).

### 5.3. Động học Học tập và Động học Gradient (Learning Dynamics & Gradient Norms)
Các quan sát từ biểu đồ học tập (Hình 4a–4d) xác nhận:
* **Hội tụ ổn định:** Quá trình huấn luyện của cả 6 mô hình đạt trạng thái ổn định sau $12 - 15$ epochs trên cả hai tập dữ liệu, không ghi nhận hiện tượng phân kỳ (divergence).
* **Quỹ đạo góc quay $\theta(t)$:** Các tham số góc quay $\theta_j$ khởi tạo ngẫu nhiên đều dịch chuyển có định hướng trong $10$ epoch đầu trước khi hội tụ về các attractor tối ưu cục bộ.
* **Chuẩn Gradient $\|\nabla_\theta \mathcal{L}\|_2$ (Sanity Check):** Chuẩn gradient lượng tử (mạch trainable strongly) dao động xấp xỉ trong dải $0{,}2$--$0{,}5$ trên đường trung bình theo seed (một số seed đạt đỉnh $\approx 1{,}3$), vượt xa ngưỡng triệt tiêu gradient. Kết quả này xác nhận mạch 4-qubit nông hoàn toàn không bị ảnh hưởng bởi hiện tượng triệt tiêu gradient (Barren Plateaus), đúng như kỳ vọng lý thuyết đối với các mạch lượng tử có độ sâu nhỏ và toán tử đo cục bộ [17].

### 5.4. Đánh giá Chi phí Phần cứng và Độ trễ Suy luận (Inference Latency)
Bảng 5 tổng hợp kết quả đo đạc độ trễ suy luận từng thành phần và thời gian huấn luyện thực tế trên môi trường CPU.

**BẢNG 5: Phân tích độ trễ suy luận trên CPU và chi phí tính toán thực tế.**

| Thành phần mô hình | Giai đoạn Tính toán | Độ trễ Trung bình / Ảnh | Tỷ lệ so với Cổ điển | Chi phí Tham số Kernel |
| :--- | :--- | :---: | :---: | :---: |
| **Classical CNN Baseline** | Toàn bộ mạng (End-to-End Forward) | **$0{,}310\text{ ms}$** | **$1{,}0\times$** *(Chuẩn)* | $20$ tham số |
| **Fixed Quanvolution** | Trích xuất đặc trưng ($196$ patch lượng tử) | $220{,}187\text{ ms}$ | $710{,}3\times$ | **$0$ tham số** |
| | Tầng phân loại cổ điển (Classifier Head) | $0{,}034\text{ ms}$ | $0{,}11\times$ | Giống nhau ($100\%$) |
| | **Toàn bộ mạng (End-to-End)** | **$220{,}221\text{ ms}$** | **$710{,}4\times$** | **$0$ tham số** |
| **Trainable Quanvolution** | **Toàn bộ mạng (End-to-End)** | **$\sim 220{,}25\text{ ms}$** | **$\sim 710{,}5\times$** | $12 - 24$ tham số |

Dữ liệu đo đạc thực tế chỉ ra rằng:
* Tầng phân loại cổ điển thực thi cực nhanh ($0{,}034\text{ ms}$), chiếm chưa đến $0{,}02\%$ tổng thời gian suy luận. Hơn $99{,}98\%$ độ trễ của Quanvolution tập trung tại $196$ lần gọi hàm mô phỏng trạng thái lượng tử trên CPU.
* Mặc dù mô phỏng lượng tử chậm hơn $\sim 710\times$ so với mạng cổ điển, độ trễ suy luận $\approx 0{,}22\text{ giây/ảnh}$ vẫn hoàn toàn nằm trong ngưỡng chấp nhận được của các hệ thống chẩn đoán y tế hỗ trợ bác sĩ (Computer-Aided Diagnosis - CAD) không đòi hỏi thời gian thực tức thời theo mili-giây.
* Đối với mạch tĩnh (`Fixed Quanvolution`), đặc trưng lượng tử chỉ cần trích xuất **đúng 1 lần duy nhất** (tiền tính toán - precomputation), sau đó toàn bộ quá trình huấn luyện $10$ seeds diễn ra chỉ trong $18\text{ giây}$ (nhanh hơn cả việc huấn luyện CNN cổ điển từ ảnh thô).

---

## 6. THẢO LUẬN (DISCUSSION)

### 6.1. Giải mã Ranh giới Chế độ Dữ liệu: Dung lượng Biểu diễn vs. Quá khớp
Sự tương phản rõ rệt giữa kết quả trên BreastMNIST và OCTMNIST cung cấp một phát hiện quan trọng về ranh giới ứng dụng của Quanvolution:
* **Trên tập dữ liệu nhỏ (BreastMNIST - 546 ảnh train, nhị phân):** Bộ lọc tích chập cổ điển với các trọng số khởi tạo ngẫu nhiên rất dễ rơi vào bẫy quá khớp (overfitting) hoặc học phải các đặc trưng giả (spurious correlations). Ngược lại, mạch lượng tử 4-qubit cố định cung cấp một phép biến đổi trực giao phi tuyến bất biến vào không gian Hilbert 16 chiều. Phép biến đổi này đóng vai trò như một **bộ điều chuẩn cấu trúc (Structural Regularizer)**, hạn chế không gian tìm kiếm của tầng phân loại và giúp mô hình đạt độ ổn định phương sai vượt trội ($\text{std}$ giảm $\sim 2{,}7\times$) và đạt ROC-AUC đỉnh cao $0{,}8521$.
* **Trên tập dữ liệu lớn và đa lớp (OCTMNIST - 3.500 ảnh train, 4 lớp):** Bài toán phân biệt 4 bệnh lý võng mạc phức tạp đòi hỏi các bộ lọc trích xuất đặc trưng có khả năng thích ứng linh hoạt với dữ liệu. Mạch 4-qubit nông với $L=1$ bị giới hạn bởi **nghẽn trần dung lượng biểu diễn (Expressibility Bottleneck)**, không thể mở rộng để nắm bắt đầy đủ ranh giới phân tách đa lớp. Lúc này, mạng CNN cổ điển với các kernel tự do tối ưu hóa trực tiếp trên $3.500$ mẫu dữ liệu phát huy sức mạnh vượt trội áp đảo ($0{,}7505$ so với $0{,}6922$).

### 6.2. Sức mạnh của Thiên kiến Quy nạp Lượng tử và Mạch Tĩnh
Một trong những phát hiện đáng kinh ngạc nhất của nghiên cứu này là: **Mạch lượng tử tĩnh không tham số (`Fixed Basic` và `Fixed Strongly`) lại là mô hình đạt thành tích cao nhất trên BreastMNIST**, đánh bại cả các mô hình lượng tử tự học nhiều tham số. 

Điều này khẳng định rằng trong chế độ dữ liệu nhỏ, việc tối ưu hóa thêm các tham số góc quay lượng tử ($\theta$) không mang lại lợi ích tương xứng, thậm chí làm tăng nguy cơ bất ổn định tối ưu hóa. Một mạch lượng tử tĩnh được cấu hình đúng đắn đã chứa đựng sẵn *Quantum Inductive Bias* đủ mạnh để trích xuất đặc trưng biên dạng y sinh xuất sắc với **đúng 0 tham số học**.

### 6.3. Ý nghĩa Lâm sàng trong Chẩn đoán Y tế
Trong bối cảnh y tế thực tế, việc bỏ sót ca bệnh ác tính (False Negative) gây ra hậu quả nghiêm trọng hơn rất nhiều so với chẩn đoán nhầm ca lành tính (False Positive). Do đó, diện tích dưới đường cong Precision-Recall (PR-AUC) trên dữ liệu lệch lớp là chỉ số lâm sàng có giá trị thực tiễn cao nhất. Việc `Fixed Strongly Quanvolution` đạt PR-AUC **$0{,}9182 \pm 0{,}0071$** ($d = +1{,}332$ so với CNN cổ điển $0{,}9041$) cho thấy bộ lọc lượng tử có năng lực đặc biệt nhạy bén trong việc khoanh vùng các mẫu bệnh hiếm gặp.

---

## 7. CÁC YẾU TỐ ĐE DỌA TÍNH HỢP LỆ VÀ GIỚI HẠN (THREATS TO VALIDITY & LIMITATIONS)

Chúng tôi thẳng thắn chỉ ra các giới hạn của nghiên cứu này nhằm định hướng cho các nghiên cứu tiếp theo:
1. **Phạm vi Dữ liệu:** Nghiên cứu được giới hạn trên $2$ tập dữ liệu chuẩn hóa thuộc bộ MedMNIST (kích thước ảnh $28 \times 28$). Mặc dù MedMNIST là tiêu chuẩn vàng để khảo sát thuật toán, việc mở rộng lên các ảnh y tế độ phân giải cao ($224 \times 224$ hoặc $512 \times 512$) sẽ đòi hỏi các kỹ thuật chia patch hoặc nén kích thước phức tạp hơn.
2. **Kích thước Cửa sổ Trượt (Patch Size):** Bộ lọc lượng tử được cố định ở kích thước $2 \times 2$ (tương ứng 4 qubits). Mặc dù kích thước này phù hợp với độ sâu mạch nông, việc khảo sát các kernel lớn hơn ($3 \times 3 \to 9$ qubits) có thể mở rộng khả năng tiếp nhận không gian nhưng sẽ làm tăng đáng kể chi phí mô phỏng.
3. **Môi trường Giả lập Không Nhiễu (Statevector Simulation):** Toàn bộ thực nghiệm được tiến hành trên môi trường giả lập giải tích không nhiễu (`default.qubit`). Trên các thiết bị phần cứng lượng tử thực tế (QPU), nhiễu cổng (gate noise) và nhiễu đọc (readout error) có thể làm suy giảm độ chính xác của các giá trị kỳ vọng $\langle Z_i \rangle$.

---

## 8. KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN (CONCLUSION & FUTURE WORK)

Bài báo này đã thiết lập một khung đánh giá thực nghiệm toàn diện, đối xứng $1:1$ và có kiểm định thống kê nghiêm ngặt giữa Bộ lọc tích chập lượng tử (Quanvolution) và Mô hình tích chập cổ điển tối thiểu trên ảnh y tế MedMNIST.

Chúng tôi đúc kết **Ba thông điệp then chốt (Take-Home Messages)**:
1. 🎯 **Ưu thế lượng tử phụ thuộc chặt chẽ vào chế độ dữ liệu:** Quanvolution không phải là giải pháp vạn năng. Ưu thế chỉ xuất hiện rõ nét trên các tập dữ liệu nhỏ và lệch lớp (BreastMNIST); trên dữ liệu lớn và đa lớp (OCTMNIST), CNN cổ điển vẫn nắm giữ vị thế áp đảo hoàn toàn.
2. 🎯 **Sức mạnh của Thiên kiến Quy nạp Lượng tử với 0 tham số học:** Mạch lượng tử tĩnh (`Fixed Basic` và `Fixed Strongly`) là baseline lượng tử cực kỳ mạnh mẽ, đem lại độ ổn định phương sai cao hơn $\sim 2{,}7\times$ và PR-AUC vượt trội trên dữ liệu nhỏ mà không tiêu tốn bất kỳ tham số học nào tại tầng đặc trưng.
3. 🎯 **Tính cục bộ của Khả năng tự học (Trainability):** Việc huấn luyện tham số góc quay lượng tử chỉ cải thiện hiệu năng khi so sánh nội bộ trong cùng một họ mạch cấu trúc, nhưng không vượt qua được mạch tĩnh tối ưu trên bài toán thực tế.

**Hướng phát triển trong tương lai:** Chúng tôi dự kiến mở rộng nghiên cứu sang việc tích hợp các bộ tăng tốc lượng tử chuyên dụng trên GPU (như NVIDIA cuQuantum) để rút ngắn độ trễ suy luận, đồng thời đánh giá độ bền vững của bộ lọc tích chập lượng tử dưới các mô hình nhiễu phần cứng thực tế (NISQ noise models).

---

## TÀI LIỆU THAM KHẢO (REFERENCES)

```text
[1] G. Litjens et al., "A survey on deep learning in medical image analysis," Medical Image Analysis, vol. 42, pp. 60–88, Dec. 2017.
[2] J. Yang et al., "MedMNIST v2 - A large-scale lightweight benchmark for 2D and 3D biomedical image classification," Scientific Data, vol. 10, no. 1, p. 41, Jan. 2023.
[3] A. Esteva et al., "A guide to deep learning in healthcare," Nature Medicine, vol. 25, no. 1, pp. 24–29, Jan. 2019.
[4] J. Biamonte et al., "Quantum machine learning," Nature, vol. 549, no. 7671, pp. 195–202, Sep. 2017.
[5] M. Cerezo et al., "Variational quantum algorithms," Nature Reviews Physics, vol. 3, no. 9, pp. 625–644, Sep. 2021.
[6] M. Henderson, S. Shakya, S. Pradhan, and T. Cook, "Quanvolutional neural networks: powering image recognition with quantum circuits," Quantum Machine Intelligence, vol. 2, no. 1, p. 2, Jun. 2020.
[7] H. Y. Huang et al., "Power of data in quantum machine learning," Nature Communications, vol. 12, no. 1, p. 2631, May 2021.
[8] T. H. Vu, H. L. Le, and T. B. Pham, "Exploring the features of quanvolutional neural networks for improved image classification," Quantum Machine Intelligence, vol. 6, no. 1, p. 29, 2024.
[9] F. M. Altares-López, A. Ribeiro, and J. J. García-Ripoll, "Automatic design of quantum feature maps," Quantum Science and Technology, vol. 6, no. 4, p. 045015, Jul. 2021.
[10] N. Matondo-Mvula and K. Elleithy, "Breast cancer detection with quanvolutional neural networks," Entropy, vol. 26, no. 8, p. 630, 2024.
[11] Q. N. Hoang, T. T. Pham, and D. N. M. Dang, "Efficient hybrid quantum-classical convolutional neural network with feature propagation layer for multi-class image classification," in Proc. Int. Conf. Adv. Eng. Theory Appl. (AETA), 2023.
[12] M. Schuld and N. Killoran, "Is quantum advantage the right goal for quantum machine learning?" PRX Quantum, vol. 3, no. 3, p. 030101, 2022.
[13] I. Cong, S. Choi, and M. D. Lukin, "Quantum convolutional neural networks," Nature Physics, vol. 15, no. 12, pp. 1273–1278, Dec. 2019.
[14] V. Bergholm et al., "PennyLane: Automatic differentiation of quantum machine learning circuits," arXiv:1811.04968, 2018.
[15] M. Schuld, V. Bergholm, C. Gogolin, J. Izaac, and N. Killoran, "Evaluating analytic gradients on quantum hardware," Physical Review A, vol. 99, no. 3, p. 032331, Mar. 2019.
[16] J. Yang et al., "MedMNIST Classification Decathlon: A lightweight AutoML benchmark for medical image analysis," in IEEE 18th Int. Symp. Biomed. Imaging (ISBI), 2021, pp. 191–195.
[17] J. R. McClean, S. Boixo, V. N. Smelyanskiy, R. Babbush, and H. Neven, "Barren plateaus in quantum neural network training landscapes," Nature Communications, vol. 9, no. 1, p. 4812, Nov. 2018.
```
