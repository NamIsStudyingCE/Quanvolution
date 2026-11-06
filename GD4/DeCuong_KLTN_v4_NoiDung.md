# ĐỀ CƯƠNG CHI TIẾT KHÓA LUẬN TỐT NGHIỆP — NỘI DUNG V4
# (Viết lại theo góp ý Hội đồng 22/09/2026, bố cục theo mẫu đề cương ECG Holter đã duyệt)

---

## PHẦN ĐẦU TRANG

- **TÊN ĐỀ TÀI TIẾNG VIỆT:** NGHIÊN CỨU VÀ ĐÁNH GIÁ LỚP TÍCH CHẬP LƯỢNG TỬ (QUANVOLUTION) TRONG PHÂN LOẠI ẢNH Y TẾ MEDMNIST DƯỚI QUY TRÌNH SO SÁNH ĐỐI XỨNG VỚI CNN CỔ ĐIỂN
- **TÊN ĐỀ TÀI TIẾNG ANH:** SYMMETRICAL EMPIRICAL EVALUATION OF QUANVOLUTIONAL LAYERS IN MEDMNIST MEDICAL IMAGE CLASSIFICATION AGAINST CLASSICAL CNN BASELINES
- **Cán bộ hướng dẫn:** ThS. Nguyễn Duy Xuân Bách — Trường Đại học Công nghệ Thông tin, ĐHQG TP. HCM
- **Sinh viên thực hiện:** Nguyễn Hoàng Nam — 22520916
- **Thời gian thực hiện:** Từ ngày 11/08/2026 đến ngày 29/01/2027

---

## NỘI DUNG ĐỀ TÀI

### 1. Tổng quan đề tài

Học sâu đã trở thành công cụ chuẩn trong phân tích ảnh y tế [9], và việc tìm kiếm các kiến trúc thay thế hiệu quả hơn vẫn đang tiếp diễn. Mạng nơ-ron tích chập lượng tử (Quanvolutional Neural Network — QNN) [1] thay thế lớp tích chập cổ điển bằng các mạch lượng tử 4 qubit tác động lên từng vùng 2×2 của ảnh, với kỳ vọng khai thác phép biến đổi phi tuyến trong không gian đặc trưng lượng tử [2] để trích xuất đặc trưng ảnh. Tuy nhiên, các công trình hiện có so sánh QNN với CNN cổ điển [5, 10] chủ yếu theo kiểu "mỗi bên một cấu hình tự chọn": hai phía khác nhau về số tham số, về ngân sách tinh chỉnh siêu tham số lẫn quy trình đánh giá (số lần chạy ít, không có khoảng tin cậy hay kiểm định thống kê) [3, 4], khiến mọi tuyên bố "lượng tử tốt hơn" đều có thể xuất phát từ sự bất đối xứng của thí nghiệm thay vì bản chất lượng tử. Đồng thời, hầu hết đánh giá được thực hiện trên máy mô phỏng statevector lý tưởng [6, 12], trong khi đặc tính nhiễu của phần cứng lượng tử NISQ hiện nay có thể thay đổi hoàn toàn thứ hạng hiệu năng giữa các mô hình — câu hỏi kết quả mô phỏng còn đáng tin ở mức nhiễu nào chưa được trả lời trong các nghiên cứu cùng dạng [10].

Đóng góp chính của đề tài là trả lời hai câu hỏi nghiên cứu dưới một quy trình thí nghiệm đối xứng về ba trục (số tham số, ngân sách tinh chỉnh, chi phí tính toán): **(A) lớp quanvolution có thực sự học được đặc trưng hữu ích, hay lợi ích (nếu có) chỉ đến từ việc mở rộng chiều phi tuyến của phép biến đổi [7]? Câu hỏi A được xác định bằng ablation bốn cấu hình cùng ngân sách (mạch tự học, mạch cố định ngẫu nhiên, mạch gỡ toàn bộ cổng vướng víu, bộ lọc tích chập ngẫu nhiên thuần cổ điển); và (B) ranh giới hiệu năng theo chế độ dữ liệu giữa hai phía lượng tử và cổ điển nằm ở đâu? Câu hỏi B được xác định bằng đường cong học tập trên sáu mức cỡ mẫu từ 100 đến 5.000 mẫu [4].** Toàn bộ kết luận được suy ra từ kiểm định thống kê suy diễn (TOST với biên tương đương chốt trước, hiệu chỉnh đa so sánh Benjamini–Hochberg, bootstrap khoảng tin cậy trên tập kiểm tra) trên hai bộ dữ liệu y tế MedMNIST [8], kèm nghiên cứu độ bền nhiễu depolarizing và nhiễu shot trên máy mô phỏng ma trận mật độ nhằm xác định mức nhiễu tại đó kết luận của đề tài còn đứng vững.

### 2. Mục tiêu của đề tài

**Mục tiêu tổng quát:** Xác định thành phần nào của lớp quanvolution mang lại lợi ích hiệu năng trong phân loại ảnh y tế, và ranh giới chế độ dữ liệu nơi lợi ích đó xuất hiện hoặc biến mất, dưới một quy trình đánh giá đối xứng hoàn toàn và có phân tích độ bền nhiễu trên môi trường mô phỏng.

**Mục tiêu cụ thể:**
- Xây dựng pipeline benchmark đối xứng ba trục giữa phía lượng tử và cổ điển: cùng số tham số tầng trích xuất đặc trưng, cùng quy trình chọn learning rate (cùng lưới giá trị, cùng số lần thử, cùng tập validation, báo cáo ngân sách tinh chỉnh mỗi phía), cùng báo cáo chi phí tính toán (FLOPs và thời gian suy luận mỗi ảnh).
- Thực hiện thí nghiệm câu hỏi A: so sánh bốn cấu hình cùng ngân sách tham số và cùng tầng phân loại (mạch lượng tử tự học, mạch lượng tử cố định ngẫu nhiên, mạch lượng tử gỡ toàn bộ cổng CNOT, và bộ lọc tích chập 2×2 ngẫu nhiên thuần cổ điển) nhằm xác định tính khả học của tham số mạch, đóng góp của entanglement, và tính cần thiết của "tính lượng tử".
- Thực hiện thí nghiệm câu hỏi B: vẽ đường cong hiệu năng hai phía trên sáu mức cỡ mẫu huấn luyện (100 / 250 / 500 / 1.000 / 2.500 / 5.000 mẫu) trên BreastMNIST và OCTMNIST, xác định hai đường có cắt nhau hay không.
- Thiết lập protocol thống kê chốt trước: 10 hạt giống × 20 epoch × 6 chỉ số × 2 bộ dữ liệu; kiểm định TOST với biên tương đương đề xuất ±2% balanced accuracy, được kiểm chứng độ mạnh thống kê trước khi chốt; hiệu chỉnh Benjamini–Hochberg [11] ở α = 0,05; bootstrap 10.000 lần tái lấy mẫu trên tập kiểm tra; phân tích độ mạnh thống kê (mức khác biệt tối thiểu phát hiện được với n = 156 ảnh kiểm tra của BreastMNIST).
- Nghiên cứu độ bền nhiễu trên máy mô phỏng ma trận mật độ (PennyLane default.mixed): nhiễu depolarizing ba mức p ∈ {0,001 / 0,01 / 0,05} và nhiễu shot shots ∈ {100 / 1.000 / 10.000}, đo mức tụt hiệu năng của từng cấu hình theo từng mức nhiễu.
- Xác minh đúng đắn gradient mạch lượng tử bằng quy tắc parameter-shift trên số học double precision (complex128) của backend lightning.qubit; sai số chéo đo được 4,1×10⁻⁸, ngưỡng chấp nhận < 10⁻⁶.

### 3. Phương pháp thực hiện

**Khảo sát và đánh giá cơ sở lý thuyết:**
- Khảo sát nguyên lý quanvolution [1], các ansatz thông dụng (BasicEntanglerLayers, StronglyEntanglingLayers) [5], tối ưu hybrid quantum-classical và quy tắc parameter-shift trên PennyLane [6]; khảo sát PennyLane với các backend statevector và density-matrix cho cả mô phỏng lý tưởng và có nhiễu [6].
- Khảo sát phương pháp luận benchmark QNN (Bowles, Ahmed và Schuld [4]) về cạm bẫy so sánh bất đối xứng và vai trò của baseline cổ điển ngẫu nhiên.
- Khảo sát đặc tính hai bộ dữ liệu MedMNIST (BreastMNIST 780/156 ảnh, mất cân bằng; OCTMNIST đa lớp) và sáu chỉ số đánh giá: accuracy, balanced accuracy, F1, MCC, ROC-AUC và PR-AUC, phù hợp cho dữ liệu mất cân bằng.

**Thiết kế thí nghiệm đối xứng:**
- Kiến trúc đối xứng ceteris paribus: duy nhất tầng trích xuất đặc trưng khác nhau (tích chập 2×2, 4 kênh, stride 2 phía cổ điển; mạch 4 qubit RY(π·x) + ansatz phía lượng tử); mỗi ảnh 28×28 được chia thành 196 vùng 2×2, mỗi vùng được gọi mạch một lần; tầng phân loại thống nhất BN → ReLU → Linear 784→K cho mọi cấu hình.
- Ba trục đối xứng được ghi nhận và báo cáo: (i) số tham số, kèm ghi nhận đây chỉ là một trục (một góc quay tạo phi tuyến sin/cos, một trọng số CNN là tuyến tính, bằng số không có nghĩa là bằng khả năng biểu diễn); (ii) ngân sách tinh chỉnh — cùng lưới learning rate {0,0005 / 0,001 / 0,005 / 0,01}, cùng số lần thử, cùng tập validation cho cả hai phía, cấu hình tốt nhất mỗi phía được đem vào thí nghiệm chính; (iii) chi phí tính toán — báo cáo FLOPs và thời gian suy luận mỗi ảnh đo trên cùng phần cứng.
- Tính tái lập: chốt hạt giống toàn cục, pipeline precompute bản đồ đặc trưng mạch tĩnh, lưu toàn bộ lịch sử huấn luyện và chỉ số từng hạt seed vào tệp kết quả gốc.
- Chính sách huấn luyện thống nhất: cùng số epoch (20), không early stopping; checkpoint báo cáo là epoch đạt balanced accuracy tốt nhất trên tập validation, cùng tiêu chí áp dụng cho mọi cấu hình của cả hai phía.

**Phân tích thống kê:**
- Kiểm định cặp paired t-test và Wilcoxon signed-rank cho khác biệt; TOST cho tương đương: kiểm định chạy trên hiệu số cặp theo 10 hạt giống; biên tương đương đề xuất ±2% BAcc, được xác nhận khả thi bằng phân tích độ mạnh thống kê trước khi chốt — nếu power không đạt tại biên đề xuất, biên được nới theo kết quả power analysis và quyết định được công bố trong báo cáo; hiệu chỉnh đa so sánh Benjamini–Hochberg trên toàn bộ họ phép kiểm định; khoảng tin cậy 95% (t* = 2,262, df = 9) và Cohen's d.
- Bootstrap 10.000 lần tái lấy mẫu trên tập kiểm tra để lượng hóa bất định do tập test nhỏ (BreastMNIST: 156 ảnh, 1 ảnh tương đương 0,64% accuracy); phân tích độ mạnh thống kê để phát biểu mức khác biệt tối thiểu phát hiện được.

**Nghiên cứu độ bền nhiễu:**
- Trên PennyLane default.mixed (mô phỏng ma trận mật độ 16×16) [6], chèn kênh depolarizing ba mức [12] và chuyển đo lường từ giá trị kỳ vọng giải tích sang hữu hạn shots ba mức; đo hiệu năng mỗi cấu hình tại 9 tổ hợp lưới nhiễu (3 × 3).
- Đối chiếu mức nhiễu tại đó thứ hạng cấu hình thay đổi với đặc tính nhiễu công bố của phần cứng NISQ hiện nay, để phát biểu phạm vi áp dụng của kết luận.

**Công cụ và môi trường:** Python, PennyLane, PyTorch, scikit-learn; huấn luyện trên CPU/GPU cá nhân; toàn bộ mã nguồn, dữ liệu kết quả gốc và kịch bản tái tạo lưu trên kho GitHub công khai.

### 4. Các nội dung chính của đề tài

- **Nội dung 1 — Cơ sở lý thuyết và thiết kế benchmark đối xứng:** nghiên cứu lý thuyết quanvolution, QML benchmark; thiết kế kiến trúc đối xứng ba trục; xác định protocol thống kê và KPI (kết quả dự kiến: đặc tả benchmark đầy đủ, định lượng trước mức khác biệt tối thiểu phát hiện được).
- **Nội dung 2 — Hiện thực pipeline và xác minh đúng đắn:** dựng pipeline dữ liệu MedMNIST, precompute đặc trưng lượng tử, xác minh gradient bằng parameter-shift (kết quả dự kiến: sai số chéo < 10⁻⁶; toàn bộ thí nghiệm tái lập được từ tệp kết quả gốc).
- **Nội dung 3 — Câu hỏi nghiên cứu A (ablation bốn cấu hình):** huấn luyện bốn cấu hình cùng ngân sách trên hai bộ dữ liệu, kiểm định thống kê từng cặp và tương đương TOST (kết quả dự kiến: xác định được thành phần chịu trách nhiệm cho lợi ích hiệu năng — nếu cả bốn ngang nhau, kết luận lợi ích đến từ mở rộng phi tuyến, không từ tính lượng tử; đây là kết quả âm có giá trị khoa học).
- **Nội dung 4 — Câu hỏi nghiên cứu B (đường cong chế độ dữ liệu):** lấy mẫu con tập huấn luyện sáu mức, vẽ đường hiệu năng hai phía, kiểm định giao điểm (kết quả dự kiến: xác định được — hoặc loại trừ được — điểm cắt giữa hai đường, thay cho lời hứa "phân định ranh giới" bằng một hình có số liệu).
- **Nội dung 5 — Độ bền nhiễu, tích hợp và viết báo cáo:** chạy lưới nhiễu 9 tổ hợp (3 mức depolarizing × 3 mức shots), đối chiếu với đặc tính NISQ; tổng hợp kết luận, câu chặn phạm vi áp dụng; hoàn thiện mã nguồn công khai, báo cáo và demo.

**Sản phẩm giao nộp:** mã nguồn và kho GitHub công khai; tệp kết quả gốc từng seed; kịch bản tái tạo toàn bộ số liệu; báo cáo khóa luận; demo so sánh bản đồ đặc trưng lượng tử và CNN; bài báo khoa học từ nội dung benchmark của khóa luận (đã nộp hội nghị SOICT 2026, Springer CCIS — đang phản biện).

### 5. Giới hạn của đề tài

- Kết quả chỉ áp dụng cho mạch 4 qubit, ảnh xám 28×28, hai bộ dữ liệu MedMNIST, môi trường mô phỏng statevector và ma trận mật độ lý tưởng; **không được suy rộng thành phát biểu về lợi thế lượng tử trên phần cứng NISQ thật.** Khoảng cách giữa mô phỏng lý tưởng và phần cứng (nhiễu depolarizing, lỗi cổng, lỗi đọc, shot noise) được lượng hóa riêng trong Nội dung 5 và được trích dẫn khi diễn giải mọi kết luận.
- Phân tích độ mạnh thống kê phụ thuộc cỡ mẫu: với tập kiểm tra 156 ảnh của BreastMNIST, khác biệt dưới mức phát hiện được chỉ có nghĩa là "chưa đủ bằng chứng", không phải "bằng chứng tương đương"; phát biểu tương đương chỉ được TOST ở biên ±2% xác nhận.
- Phạm vi kích thước mạch và ansatz khảo sát giới hạn ở hai họ ansatz chuẩn của PennyLane; không khẳng định kết luận cho mọi kiến trúc mạch khác.

### 6. KPI tổng hợp

| Hạng mục | KPI |
|---|---|
| Protocol thí nghiệm | 10 hạt giống × 20 epoch × 6 chỉ số × 2 bộ dữ liệu, tái lập từ tệp gốc |
| Đối xứng tham số | Chênh lệch tham số tầng trích xuất = 0 giữa các cấu hình cùng mức |
| Đối xứng tinh chỉnh | Cùng lưới LR 4 giá trị × cùng số lần thử × cùng tập validation cho hai phía |
| Chi phí tính toán | Báo cáo FLOPs + thời gian suy luận/ảnh, chênh lệch được lượng hóa |
| Kiểm định khác biệt | Paired t-test + Wilcoxon, hiệu chỉnh BH α = 0,05 |
| Kiểm định tương đương | TOST trên hiệu số cặp theo seed; biên đề xuất ±2% BAcc, chốt sau power analysis |
| Bất định tập test | Bootstrap 10.000; phát biểu MDE với n = 156 |
| Xác minh gradient | Parameter-shift chéo trên double precision; đo được 4,1×10⁻⁸, ngưỡng < 10⁻⁶ |
| Lưới nhiễu | 3 mức depolarizing × 3 mức shots × 4 cấu hình |
| Đường cong dữ liệu | 6 mức cỡ mẫu 100–5.000 × 2 phía × 2 bộ dữ liệu |

### 7. Kế hoạch thực hiện

Nội dung 1 và 2 đã hoàn thành tại thời điểm nộp đề cương chỉnh sửa; các nội dung 3 đến 5 là kế hoạch thực hiện.

- 11/08/2026 – 26/09/2026: Nội dung 1 và 2 (cơ sở lý thuyết, thiết kế benchmark đối xứng, hiện thực pipeline, xác minh gradient, chạy protocol gốc 10 hạt giống).
- 27/09/2026 – 26/10/2026: Nội dung 3 (sweep learning rate đối xứng hai phía, ablation bốn cấu hình (tự học / cố định / gỡ CNOT / bộ lọc ngẫu nhiên cổ điển), phân tích thống kê hoàn chỉnh: TOST, BH, bootstrap).
- 27/10/2026 – 26/11/2026: Nội dung 4 (đường cong hiệu năng sáu mức cỡ mẫu hai phía; phân tích tương tác chế độ dữ liệu).
- 27/11/2026 – 26/12/2026: Nội dung 5 — chạy lưới nhiễu 9 tổ hợp (3 mức depolarizing × 3 mức shots) trên default.mixed và đối chiếu với đặc tính phần cứng NISQ. Phần tùy chọn: đối chứng vài chục patch trên phần cứng lượng tử thật qua IBM Quantum để so sánh bản đồ đặc trưng, chỉ thực hiện nếu hạn mức miễn phí còn hiệu lực tại thời điểm thực hiện; kết luận của đề tài không phụ thuộc phần này.
- 27/12/2026 – 29/01/2027: Tổng hợp kết luận, hoàn thiện demo, viết và hoàn thiện báo cáo khóa luận, chuẩn bị phản biện và bảo vệ.

Các mốc hành chính theo lịch Khoa: nộp scan đề cương có chữ ký (trước 10/09/2026); bảo vệ đề cương: 14–18/09/2026; báo cáo tiến độ giữa kỳ (theo lịch Khoa); nộp khóa luận và phản biện: 11–15/01/2027; bảo vệ khóa luận: 25–29/01/2027.

### 8. Ranh giới giữa Đồ án 2 và Khóa luận

- Đồ án 2 đã thực hiện: kiến trúc adaptive gating HQCNN (6 qubit kết hợp ResNet-18) trên OCTMNIST.
- Khóa luận kế thừa: hạ tầng PennyLane/PyTorch và pipeline dữ liệu, được chuẩn hóa lại về chốt hạt giống và protocol đánh giá.
- Khóa luận thực hiện mới: kiến trúc quanvolution, ablation bốn cấu hình, đường cong chế độ dữ liệu, sweep learning rate đối xứng và nghiên cứu độ bền nhiễu. Kết quả Đồ án 2 không được sử dụng lại trong các bảng số liệu của khóa luận.

1. Henderson, M., Shakya, S., Pradhan, S., Cook, T.: Quanvolutional neural networks: powering image recognition with quantum circuits. Quantum Machine Intelligence **2**(1), 2 (2020). doi:10.1007/s42484-020-00012-y
2. Havlíček, V., et al.: Supervised learning with quantum-enhanced feature spaces. Nature **567**(7747), 209–212 (2019). doi:10.1038/s41586-019-0980-2
3. Biamonte, J., et al.: Quantum machine learning. Nature **549**(7671), 195–202 (2017). doi:10.1038/nature23474
4. Schuld, M., Killoran, N.: Is quantum advantage the right goal for quantum machine learning? PRX Quantum **3**(3), 030101 (2022). doi:10.1103/PRXQuantum.3.030101
5. Bowles, J., Ahmed, S., Schuld, M.: Better than classical? The subtle art of benchmarking quantum machine learning models. arXiv:2403.07059 (2024)
6. Hur, T., Kim, L., Park, D.: Quantum convolutional neural network for classical data classification. Quantum Machine Intelligence **4**(1), 3 (2022). doi:10.1007/s42484-021-00061-x
7. Bergholm, V., et al.: PennyLane: Automatic differentiation of hybrid quantum-classical computations. arXiv:1811.04968 (2018)
8. Tacchino, F., et al.: Quantum computers as universal quantum simulators: state-of-the-art and perspectives. Advanced Quantum Technologies **3**(3), 1900052 (2020). doi:10.1002/qute.201900052
9. Jerbi, S., et al.: Quantum machine learning beyond kernel methods. Nature Communications **14**, 517 (2023). doi:10.1038/s41467-023-36159-y
10. Yang, J., et al.: MedMNIST v2 — A large-scale lightweight benchmark for 2D and 3D biomedical image classification. Scientific Data **10**(1), 41 (2023). doi:10.1038/s41597-022-01721-8
11. Litjens, G., et al.: A survey on deep learning in medical image analysis. Medical Image Analysis **42**, 60–88 (2017). doi:10.1016/j.media.2017.07.005
12. Matondo-Mvula, P., Elleithy, K.: Breast cancer detection with quanvolutional neural networks. Entropy **26**(8), 630 (2024). doi:10.3390/e26080630
13. Benjamini, Y., Hochberg, Y.: Controlling the false discovery rate: a practical and powerful approach to multiple testing. Journal of the Royal Statistical Society: Series B **57**(1), 289–300 (1995). doi:10.1111/j.2517-6161.1995.tb02031.x
14. Nielsen, M.A., Chuang, I.L.: Quantum Computation and Quantum Information, 10th Anniversary edn. Cambridge University Press (2010). doi:10.1017/CBO9780511976667

---

### 9. Tài liệu tham khảo

1. Henderson, M., Shakya, S., Pradhan, S., Cook, T.: Quanvolutional neural networks: powering image recognition with quantum circuits. Quantum Machine Intelligence **2**(1), 2 (2020). doi:10.1007/s42484-020-00012-y
2. Havlíček, V., et al.: Supervised learning with quantum-enhanced feature spaces. Nature **567**(7747), 209–212 (2019). doi:10.1038/s41586-019-0980-2
3. Schuld, M., Killoran, N.: Is quantum advantage the right goal for quantum machine learning? PRX Quantum **3**(3), 030101 (2022). doi:10.1103/PRXQuantum.3.030101
4. Bowles, J., Ahmed, S., Schuld, M.: Better than classical? The subtle art of benchmarking quantum machine learning models. arXiv:2403.07059 (2024)
5. Hur, T., Kim, L., Park, D.: Quantum convolutional neural network for classical data classification. Quantum Machine Intelligence **4**(1), 3 (2022). doi:10.1007/s42484-021-00061-x
6. Bergholm, V., et al.: PennyLane: Automatic differentiation of hybrid quantum-classical computations. arXiv:1811.04968 (2018)
7. Jerbi, S., et al.: Quantum machine learning beyond kernel methods. Nature Communications **14**, 517 (2023). doi:10.1038/s41467-023-36159-y
8. Yang, J., et al.: MedMNIST v2 — A large-scale lightweight benchmark for 2D and 3D biomedical image classification. Scientific Data **10**(1), 41 (2023). doi:10.1038/s41597-022-01721-8
9. Litjens, G., et al.: A survey on deep learning in medical image analysis. Medical Image Analysis **42**, 60–88 (2017). doi:10.1016/j.media.2017.07.005
10. Matondo-Mvula, P., Elleithy, K.: Breast cancer detection with quanvolutional neural networks. Entropy **26**(8), 630 (2024). doi:10.3390/e26080630
11. Benjamini, Y., Hochberg, Y.: Controlling the false discovery rate: a practical and powerful approach to multiple testing. Journal of the Royal Statistical Society: Series B **57**(1), 289–300 (1995). doi:10.1111/j.2517-6161.1995.tb02031.x
12. Nielsen, M.A., Chuang, I.L.: Quantum Computation and Quantum Information, 10th Anniversary edn. Cambridge University Press (2010). doi:10.1017/CBO9780511976667
