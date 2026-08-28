# BẢNG 4: HIỆU NĂNG THỰC NGHIỆM & EFFECT SIZE TRÊN OCTMNIST (10 SEEDS)

> **Cấu hình:** 5.000 mẫu (4 lớp bệnh lý võng mạc, tập dữ liệu lớn), 10 seeds độc lập (60/60 runs), 20 epochs đồng nhất, Độ sâu mạch $L=1$.  
> **Nguồn chân lý:** `results/full_trainable_octmnist.json`

---

## 1. BẢNG TỔNG HỢP 6 METRICS (MEAN ± STD & KHOẢNG TIN CẬY 95% CI)

| Mô hình | Feature Params | ROC-AUC [95% CI] | PR-AUC [95% CI] | Balanced Acc [95% CI] | Accuracy | F1-Score | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Classical CNN Baseline** | 20 | **$0.7505 \pm 0.0240$**<br>$[0.7333, 0.7676]$ | **$0.4991 \pm 0.0297$**<br>$[0.4778, 0.5203]$ | **$0.4433 \pm 0.0135$**<br>$[0.4336, 0.4530]$ | **$0.4433 \pm 0.0135$** | **$0.3206 \pm 0.0175$** | **$0.3156 \pm 0.0198$** |
| **Fixed Basic L1** | **0** | $0.6711 \pm 0.0042$<br>$[0.6681, 0.6741]$ | $0.4186 \pm 0.0074$<br>$[0.4133, 0.4239]$ | $0.4075 \pm 0.0042$<br>$[0.4045, 0.4105]$ | $0.4075 \pm 0.0042$ | $0.2971 \pm 0.0075$ | $0.2566 \pm 0.0072$ |
| **Trainable Basic L1** | 4 | $0.6704 \pm 0.0106$<br>$[0.6628, 0.6780]$ | $0.4102 \pm 0.0131$<br>$[0.4008, 0.4195]$ | $0.3955 \pm 0.0161$<br>$[0.3840, 0.4070]$ | $0.3955 \pm 0.0161$ | $0.2837 \pm 0.0203$ | $0.2394 \pm 0.0202$ |
| **Fixed Champion (`random_L1`)** | **0** | $0.6912 \pm 0.0071$<br>$[0.6862, 0.6963]$ | $0.4443 \pm 0.0088$<br>$[0.4380, 0.4506]$ | $0.4048 \pm 0.0130$<br>$[0.3955, 0.4141]$ | $0.4048 \pm 0.0130$ | $0.2997 \pm 0.0201$ | $0.2530 \pm 0.0143$ |
| **Fixed Strongly L1** | **0** | $0.6690 \pm 0.0055$<br>$[0.6650, 0.6729]$ | $0.4175 \pm 0.0047$<br>$[0.4142, 0.4209]$ | $0.4034 \pm 0.0046$<br>$[0.4001, 0.4067]$ | $0.4034 \pm 0.0046$ | $0.3050 \pm 0.0130$ | $0.2421 \pm 0.0064$ |
| **Trainable Strongly L1** | 12 | $0.6922 \pm 0.0199$<br>$[0.6780, 0.7065]$ | $0.4365 \pm 0.0289$<br>$[0.4158, 0.4571]$ | $0.4020 \pm 0.0148$<br>$[0.3914, 0.4126]$ | $0.4020 \pm 0.0148$ | $0.2949 \pm 0.0188$ | $0.2481 \pm 0.0232$ |

*Ghi chú:* Giá trị in đậm là kết quả cao nhất trong cột.

---

## 2. BẢNG EFFECT SIZE (COHEN'S d) & KIỂM ĐỊNH CHO CÁC CẶP ĐẤU CHÍNH

| Cặp đấu đối sánh | Chỉ số (Metric) | Độ lệch ($\Delta$) | Cohen's $d$ | Paired $t$-test | Wilcoxon ($n=10$) | Đánh giá Mức độ Tác động (Effect Size) |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Trainable Strongly vs Fixed Strongly**<br>*(Hiệu ứng Trainability Tầng 3)* | ROC-AUC<br>PR-AUC<br>Balanced Acc | **$+0.0232$**<br>$+0.0189$<br>$-0.0014$ | **$+1.050$**<br>$+0.621$<br>$-0.112$ | $p=0.0090$ (**)<br>$p=0.0810$ (ns)<br>$p=0.7314$ (ns) | **$p=0.0098$**<br>$p=0.1055$<br>$p=0.6055$ | **Tác động lớn (Large effect, $p<0.01$)**<br>Tác động trung bình<br>Tác động không đáng kể |
| **Trainable Strongly vs Fixed Champ**<br>*(Đối đầu Quán quân Lượng tử)* | ROC-AUC<br>PR-AUC<br>Balanced Acc | $+0.0010$<br>$-0.0078$<br>$-0.0028$ | $+0.046$<br>$-0.255$<br>$-0.139$ | $p=0.8875$ (ns)<br>$p=0.4415$ (ns)<br>$p=0.6702$ (ns) | $p=0.6250$<br>$p=0.4922$<br>$p=0.8652$ | Tác động không đáng kể (Hòa nhau)<br>Tác động nhỏ<br>Tác động không đáng kể |
| **Classical CNN vs Trainable Strongly**<br>*(Ưu thế Cổ điển trên tập lớn)* | ROC-AUC<br>PR-AUC<br>Balanced Acc | **$+0.0583$**<br>**$+0.0626$**<br>**$+0.0413$** | **$+2.108$**<br>**$+1.402$**<br>**$+1.874$** | $p=0.0001$ (***)<br>$p=0.0016$ (**)<br>$p=0.0002$ (***) | **$p=0.0020$**<br>**$p=0.0039$**<br>**$p=0.0020$** | **Tác động khổng lồ (Huge effect, $p<0.001$)**<br>**Tác động rất lớn**<br>**Tác động rất lớn** |
| **Classical CNN vs Fixed Champion** | ROC-AUC | $+0.0592$ | $+2.483$ | $p=0.0000$ (***) | $p=0.0020$ | **Tác động khổng lồ (Huge effect, $p<0.001$)** |
