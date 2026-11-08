# Báo cáo Wide-Fusion v2 (Đòn bẩy 3 Mũi Nhọn): Interp4D + Gate Bias 0.0 + Pre-Concat BN + DropPath

Nguồn: `reports/pillar2_wide_results.json`; mã: `experiments/run_pillar2_wide.py`, `src/models/gr_quanv_wide.py`. Chế độ Q-LUT: **Interp4D SIMD (Cosine 0.999993)**, `gate_bias = 0.0` (alpha_0 = 0.5), `drop_c_prob = 0.15`. 5 seeds [0, 42, 100, 2023, 777], chọn epoch theo val, test đánh giá 1 lần, warmup 3 epochs + cosine.

## Sạch (full train) - breastmnist - WideResidualHead v2
| Mô hình | Tham số | ACC (%) | AUC (%) | F1 (%) |
|---|---|---|---|---|
| Wide Classical Baseline | 116882 | 84.87 ± 0.65 | 89.60 ± 0.80 | 89.81 ± 0.42 |
| Wide Classical Control (0% quantum) | 148406 | 84.10 ± 0.94 | 88.93 ± 0.99 | 89.56 ± 0.51 |
| Proposed Wide-GR-Quanv | 148306 | 84.74 ± 1.79 | 89.14 ± 0.97 | 90.03 ± 0.91 |

**ΔACC** = +0.64 điểm (p = 0.606); **ΔAUC** = +0.21 điểm (p = 0.601) (t-test ghép cặp n=5).

## Sạch (full train) - pneumoniamnist - WideResidualHead v2
| Mô hình | Tham số | ACC (%) | AUC (%) | F1 (%) |
|---|---|---|---|---|
| Wide Classical Baseline | 116882 | 90.90 ± 1.23 | 97.45 ± 0.21 | 93.09 ± 0.80 |
| Wide Classical Control (0% quantum) | 148406 | 90.22 ± 1.65 | 97.69 ± 0.21 | 92.69 ± 1.10 |
| Proposed Wide-GR-Quanv | 148306 | 88.94 ± 1.63 | 97.62 ± 0.25 | 91.84 ± 1.06 |

**ΔACC** = -1.28 điểm (p = 0.235); **ΔAUC** = -0.08 điểm (p = 0.523) (t-test ghép cặp n=5).

## Sạch (full train) - octmnist - WideResidualHead v2
| Mô hình | Tham số | ACC (%) | AUC (%) | F1 (%) |
|---|---|---|---|---|
| Wide Classical Baseline | 117012 | 81.32 ± 1.13 | 97.02 ± 0.15 | 80.24 ± 1.54 |
| Wide Classical Control (0% quantum) | 148536 | 82.52 ± 2.61 | 97.06 ± 0.42 | 81.77 ± 2.95 |
| Proposed Wide-GR-Quanv | 148436 | 84.04 ± 2.45 | 97.32 ± 0.25 | 83.47 ± 2.62 |

**ΔACC** = +1.52 điểm (p = 0.185); **ΔAUC** = +0.26 điểm (p = 0.321) (t-test ghép cặp n=5).

## Kịch bản ít dữ liệu (ảnh sạch; train 5/10/20%)

### pneumoniamnist
| Tỷ lệ train | Wide Control ACC (%) | Wide-GR-Quanv ACC (%) | ΔACC | p |
|---|---|---|---|---|
| 5% | 87.24 ± 1.90 | 84.68 ± 3.68 | -2.56 | 0.340 |
| 10% | 87.37 ± 1.84 | 87.05 ± 2.70 | -0.32 | 0.882 |
| 20% | 88.85 ± 2.12 | 88.75 ± 1.33 | -0.10 | 0.939 |
| 100% (full) | 90.22 ± 1.65 | 88.94 ± 1.63 | -1.28 | - |


### octmnist
| Tỷ lệ train | Wide Control ACC (%) | Wide-GR-Quanv ACC (%) | ΔACC | p |
|---|---|---|---|---|
| 5% | 76.04 ± 2.69 | 76.04 ± 1.57 | -0.00 | 1.000 |
| 10% | 78.82 ± 5.45 | 77.38 ± 3.39 | -1.44 | 0.585 |
| 20% | 80.80 ± 1.31 | 79.38 ± 1.83 | -1.42 | 0.371 |
| 100% (full) | 82.52 ± 2.61 | 84.04 ± 2.45 | +1.52 | - |

## Cổng alpha trung bình sau huấn luyện (trên test, mean 5 seeds)

| Tập | Mô hình | alpha |
|---|---|---|
| breastmnist | control | 0.483 |
| pneumoniamnist | control | 0.440 |
| octmnist | control | 0.422 |
| breastmnist | grquanv | 0.480 |
| pneumoniamnist | grquanv | 0.442 |
| octmnist | grquanv | 0.421 |
