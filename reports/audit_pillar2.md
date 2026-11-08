# KIỂM TOÁN ĐỘC LẬP CỘT TRỤ 1 & 2 (commit 374136b)

Số liệu thô: `reports/audit_pillar2.json`; mã: `experiments/audit_pillar2.py`; Cột trụ 1 chạy lại bằng `experiments/bench_rust_extreme.py`.

## Cột trụ 1 — tái lập được
| Claim | Đo lại | Kết luận |
|---|---|---|
| INT8 bitshift 0.30 µs/ảnh, 3.33M FPS (B=1) | 0.300 µs, 3,333,333 FPS | Đúng |
| Conv2D 2x2 PyTorch B=1 | 9.9 µs (pre-float32) / 16.2-17.2 µs (từ uint8) | Nhanh 33x / 54-57x |
| B=100, 1 luồng | 0.116 µs vs Conv 1.421 µs | 12.3x (claim cũ 11.4x) |
| Cosine INT8 | 0.990907 (Breast), 0.985600 (OCT) | Đúng |
| Cosine Interp4D | 0.999993 / 0.999992 | Đúng |
| Đa luồng B=1000 | 0.155 µs vs 0.803 µs | Chỉ 5.2x |
| Đa luồng B≈100 | 0.776 µs vs 1.678 µs | Chỉ 2.2x |

Lưu ý: Conv2D PyTorch ở B=1 bị chi phối bởi overhead dispatch của framework, không phải tính toán. Chưa có đối chứng conv 2x2 viết tay bằng Rust/SIMD. Q-LUT dùng mạch cố định (không huấn luyện), lượng tử 4-bit/pixel, đầu ra INT8.

## Cột trụ 2 — kết luận chính bị thay đổi
Đối chứng cổ điển cùng bộ trộn (`ClassicalMultiScaleGatedControl`, `lean_mixer=False`, 11,076 tham số, 0% lượng tử), 5 seeds, chọn epoch theo val:

| Kịch bản | Classical Baseline | Control cổ điển cùng dung lượng | GR-Quanv |
|---|---|---|---|
| OCT 3.5K | 45.72 ± 3.25 | 62.86 ± 3.32 | 62.30 ± 2.73 |
| OCT 100% | 52.94 ± 1.33 | 73.34 ± 1.42 | 73.58 ± 1.36 |
| Few-shot 960 ảnh | 40.38 ± 5.24 | 60.00 ± 4.22 | 59.82 ± 0.77 |
| Few-shot 4800 ảnh | 47.50 ± 3.17 | 65.24 ± 3.83 | 64.68 ± 0.95 |
| Speckle 0.10 | 32.98 ± 3.50 | 51.86 ± 5.42 | 60.74 ± 1.87 |
| Speckle 0.15 | 28.70 ± 3.31 | 43.92 ± 6.22 | 57.56 ± 3.80 |

- Mức tăng so với Classical Baseline đến từ bộ trộn đa tỷ lệ + cổng + dung lượng, không phải từ đặc trưng lượng tử: trên ảnh sạch, control cổ điển ngang GR-Quanv.
- Điểm còn đứng được: nhiễu Speckle (GR-Quanv hơn control +8.9 và +13.6 điểm, độ lệch chuẩn nhỏ hơn ở 0.10) và độ ổn định few-shot (std 0.77 vs 4.22). Chưa kiểm định ghép cặp; nhiễu dùng một realization cố định.
- Các kết quả sau KHÔNG hợp lệ / cần chạy lại: ConvHead 77.40% (chọn epoch theo test, hiệu 0.28 điểm so với Classical 77.12% là không có ý nghĩa); ConvHead 1% (chỉ GR-Quanv được bù prior).
- Chưa kiểm toán: Pure Quanv thắng Classical ở 0.25% dữ liệu; BreastMNIST / PneumoniaMNIST với control cùng dung lượng.
