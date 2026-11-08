# Báo cáo Wide-Fusion (Tác vụ 1-3): gỡ nút thắt 4 kênh, warmup+cosine, sample-efficiency

Nguồn: `reports/pillar2_wide_results.json`; mã: `experiments/run_pillar2_wide.py`, `src/models/gr_quanv_wide.py`. 5 seeds [0, 42, 100, 2023, 777], chọn epoch theo val, test đánh giá 1 lần, warmup 3 epochs + cosine.

## Sạch (full train) - breastmnist - WideResidualHead
| Mô hình | Tham số | ACC (%) | AUC (%) | F1 (%) |
|---|---|---|---|---|
| Wide Classical Baseline | 116882 | 84.49 ± 1.31 | 89.70 ± 1.14 | 89.71 ± 0.96 |
| Wide Classical Control (0% quantum) | 148342 | 84.49 ± 2.27 | 88.88 ± 1.59 | 89.98 ± 1.29 |
| Proposed Wide-GR-Quanv | 148242 | 83.72 ± 4.84 | 88.73 ± 2.83 | 89.33 ± 3.03 |

**ΔACC** = -0.77 điểm (p = 0.808); **ΔAUC** = -0.15 điểm (p = 0.925) (t-test ghép cặp n=5).

## Sạch (full train) - pneumoniamnist - WideResidualHead
| Mô hình | Tham số | ACC (%) | AUC (%) | F1 (%) |
|---|---|---|---|---|
| Wide Classical Baseline | 116882 | 89.62 ± 1.23 | 97.58 ± 0.15 | 92.28 ± 0.83 |
| Wide Classical Control (0% quantum) | 148342 | 90.29 ± 2.13 | 97.73 ± 0.27 | 92.76 ± 1.41 |
| Proposed Wide-GR-Quanv | 148242 | 89.04 ± 3.18 | 97.26 ± 0.29 | 91.89 ± 2.05 |

**ΔACC** = -1.25 điểm (p = 0.600); **ΔAUC** = -0.47 điểm (p = 0.117) (t-test ghép cặp n=5).

## Sạch (full train) - octmnist - WideResidualHead
| Mô hình | Tham số | ACC (%) | AUC (%) | F1 (%) |
|---|---|---|---|---|
| Wide Classical Baseline | 117012 | 80.96 ± 0.85 | 96.73 ± 0.23 | 80.07 ± 1.15 |
| Wide Classical Control (0% quantum) | 148472 | 83.22 ± 2.00 | 97.21 ± 0.19 | 82.48 ± 2.29 |
| Proposed Wide-GR-Quanv | 148372 | 83.58 ± 1.98 | 97.24 ± 0.24 | 82.88 ± 2.15 |

**ΔACC** = +0.36 điểm (p = 0.716); **ΔAUC** = +0.03 điểm (p = 0.770) (t-test ghép cặp n=5).

## Kịch bản ít dữ liệu (ảnh sạch; train 5/10/20%)

### pneumoniamnist
| Tỷ lệ train | Wide Control ACC (%) | Wide-GR-Quanv ACC (%) | ΔACC | p |
|---|---|---|---|---|
| 5% | 86.47 ± 1.55 | 82.95 ± 3.04 | -3.53 | 0.105 |
| 10% | 87.66 ± 1.27 | 86.60 ± 1.87 | -1.06 | 0.511 |
| 20% | 89.68 ± 1.84 | 88.14 ± 2.60 | -1.54 | 0.373 |
| 100% (full) | 90.29 ± 2.13 | 89.04 ± 3.18 | -1.25 | - |


### octmnist
| Tỷ lệ train | Wide Control ACC (%) | Wide-GR-Quanv ACC (%) | ΔACC | p |
|---|---|---|---|---|
| 5% | 74.80 ± 1.47 | 75.06 ± 1.95 | +0.26 | 0.795 |
| 10% | 77.12 ± 4.93 | 73.80 ± 5.89 | -3.32 | 0.344 |
| 20% | 81.52 ± 1.80 | 79.84 ± 1.98 | -1.68 | 0.049 |
| 100% (full) | 83.22 ± 2.00 | 83.58 ± 1.98 | +0.36 | - |

## Cổng alpha trung bình sau huấn luyện (trên test, mean 5 seeds)

| Tập | Mô hình | alpha |
|---|---|---|
| breastmnist | control | 0.209 |
| pneumoniamnist | control | 0.190 |
| octmnist | control | 0.242 |
| breastmnist | grquanv | 0.200 |
| pneumoniamnist | grquanv | 0.186 |
| octmnist | grquanv | 0.238 |
