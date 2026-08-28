# BẢNG 3: HIỆU NĂNG THỰC NGHIỆM & EFFECT SIZE TRÊN BREASTMNIST (10 SEEDS)

> **Cấu hình:** 780 mẫu (Nhị phân, mất cân bằng lớp), 10 seeds độc lập, 20 epochs đồng nhất, Độ sâu mạch $L=2$.  
> **Nguồn chân lý:** `results/full_trainable_breastmnist.json`

---

## 1. BẢNG TỔNG HỢP 6 METRICS (MEAN ± STD & KHOẢNG TIN CẬY 95% CI)

| Mô hình | Feature Params | ROC-AUC [95% CI] | PR-AUC [95% CI] | Balanced Acc [95% CI] | Accuracy | F1-Score | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Classical CNN Baseline** | 20 | $0.8336 \pm 0.0246$<br>$[0.8160, 0.8512]$ | $0.9041 \pm 0.0095$<br>$[0.8973, 0.9109]$ | $0.6875 \pm 0.0448$<br>$[0.6554, 0.7196]$ | $0.8103 \pm 0.0265$ | $0.8802 \pm 0.0145$ | $0.4702 \pm 0.0718$ |
| **Fixed Basic L2** | **0** | **$0.8521 \pm 0.0090$**<br>$[0.8457, 0.8586]$ | $0.9110 \pm 0.0049$<br>$[0.9075, 0.9145]$ | $0.6816 \pm 0.0490$<br>$[0.6465, 0.7167]$ | $0.8083 \pm 0.0193$ | $0.8796 \pm 0.0093$ | $0.4626 \pm 0.0631$ |
| **Trainable Basic L2** | 8 | $0.8406 \pm 0.0239$<br>$[0.8235, 0.8577]$ | $0.9173 \pm 0.0184$<br>$[0.9041, 0.9305]$ | $0.6732 \pm 0.0382$<br>$[0.6459, 0.7005]$ | $0.7917 \pm 0.0239$ | $0.8668 \pm 0.0205$ | $0.4224 \pm 0.0984$ |
| **Fixed Strongly L2** | **0** | $0.8139 \pm 0.0142$<br>$[0.8037, 0.8241]$ | **$0.9182 \pm 0.0067$**<br>$[0.9134, 0.9230]$ | $0.6602 \pm 0.0202$<br>$[0.6457, 0.6747]$ | $0.7846 \pm 0.0177$ | $0.8631 \pm 0.0175$ | $0.3942 \pm 0.0620$ |
| **Trainable Strongly L2** | 24 | $0.8306 \pm 0.0279$<br>$[0.8106, 0.8506]$ | $0.9167 \pm 0.0157$<br>$[0.9055, 0.9279]$ | **$0.6945 \pm 0.0428$**<br>$[0.6639, 0.7251]$ | $0.8019 \pm 0.0284$ | $0.8724 \pm 0.0188$ | $0.4549 \pm 0.0772$ |

*Ghi chú:* Giá trị in đậm là kết quả cao nhất trong cột.

---

## 2. BẢNG EFFECT SIZE (COHEN'S d) & KIỂM ĐỊNH CHO CÁC CẶP ĐẤU CHÍNH

| Cặp đấu đối sánh | Chỉ số (Metric) | Độ lệch ($\Delta$) | Cohen's $d$ | Paired $t$-test | Wilcoxon ($n=10$) | Đánh giá Mức độ Tác động (Effect Size) |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Fixed Basic vs Classical CNN**<br>*(Quán quân ROC-AUC toàn bảng)* | ROC-AUC<br>PR-AUC<br>Balanced Acc | $+0.0186$<br>$+0.0069$<br>$-0.0058$ | **$+0.815$**<br>$+0.694$<br>$-0.160$ | $p=0.0298$ (*)<br>$p=0.0557$ (ns)<br>$p=0.6257$ (ns) | $p=0.0254$<br>$p=0.1309$<br>$p=0.7695$ | **Tác động lớn (Large effect, $p<0.05$)**<br>Tác động trung bình (Medium effect)<br>Tác động không đáng kể (Negligible) |
| **Trainable Strongly vs Fixed Strongly**<br>*(Hiệu ứng Trainability Tầng 3)* | Balanced Acc<br>ROC-AUC<br>PR-AUC | **$+0.0344$**<br>$+0.0167$<br>$-0.0015$ | **$+0.677$**<br>$+0.390$<br>$-0.071$ | $p=0.0611$ (ns)<br>$p=0.2485$ (ns)<br>$p=0.8278$ (ns) | $p=0.0879$<br>$p=0.2324$<br>$p=0.9219$ | **Tác động trung bình (Medium effect)**<br>Tác động nhỏ (Small effect)<br>Tác động không đáng kể (Negligible) |
| **Fixed Strongly vs Classical CNN**<br>*(Quán quân PR-AUC bắt bệnh hiếm)* | PR-AUC<br>ROC-AUC<br>Balanced Acc | **$+0.0140$**<br>$-0.0197$<br>$-0.0273$ | **$+1.332$**<br>$-0.729$<br>$-0.483$ | $p=0.0023$ (**)<br>$p=0.0467$ (*)<br>$p=0.1614$ (ns) | $p=0.0059$<br>$p=0.0488$<br>$p=0.1934$ | **Tác động rất lớn (Very large effect, $p<0.01$)**<br>Tác động trung bình<br>Tác động nhỏ |
| **Trainable Basic vs Classical CNN** | PR-AUC | $+0.0131$ | $+0.711$ | $p=0.0513$ (ns) | $p=0.0645$ | Tác động trung bình (Medium effect, $p \approx 0.05$) |
| **Trainable Strongly vs Classical CNN** | Balanced Acc | $+0.0071$ | $+0.139$ | $p=0.6701$ (ns) | $p=0.7695$ | Tác động không đáng kể (Negligible) |
