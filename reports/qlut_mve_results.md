# Q-LUT MVE — kết quả đo thật (2026-11-07 14:30:34)

Circuit `strongly_L2`, 100 ảnh test/dataset, seed 42, CPU: Intel64 Family 6 Model 186 Stepping 2, GenuineIntel, torch 2.5.1+cu121 (threads mặc định 8). Ground Truth = PennyLane naive của repo.

## breastmnist — Fidelity (so với Ground Truth)

| Phương pháp | MAE | RMSE | nRMSE | Cosine | Max abs err | P99 abs err |
|---|---|---|---|---|---|---|
| ExactBatched(PennyLane vec) | 0.00000 | 0.00000 | 0.0000 | 1.000000 | 0.0000 | 0.0000 |
| QLUT-4bit-nearest | 0.01975 | 0.02717 | 0.1369 | 0.990519 | 0.1100 | 0.0810 |
| QLUT-4bit-interp | 0.00148 | 0.00202 | 0.0102 | 0.999993 | 0.0088 | 0.0060 |
| QLUT-5bit-nearest | 0.00961 | 0.01317 | 0.0663 | 0.997763 | 0.0586 | 0.0392 |
| QLUT-5bit-interp | 0.00035 | 0.00048 | 0.0024 | 1.000000 | 0.0021 | 0.0014 |

### breastmnist — Latency CPU (threads_default)

| Engine | b=1 ms/ảnh (median) | b=1 p95 | b=1 FPS | b=N ms/ảnh | b=N FPS |
|---|---|---|---|---|---|
| PennyLane naive (repo) | 276.5213 | 589.8253 | 4 | 276.5213 | 4 |
| PennyLane vectorized | 4.2390 | 4.9185 | 236 | 1.5655 | 639 |
| Classical Conv2d 2x2 | 0.0548 | 0.1079 | 18265 | 0.0018 | 566251 |
| QLUT-4bit-nearest (torch) | 0.1112 | 0.2126 | 8993 | 0.0416 | 24061 |
| QLUT-4bit-interp (torch) | 0.3274 | 0.5681 | 3054 | 0.1338 | 7475 |
| QLUT-5bit-nearest (torch) | 0.1495 | 0.2299 | 6689 | 0.0370 | 27060 |
| QLUT-4bit-nearest (numpy) | 0.0149 | 0.0162 | 67114 | 0.0074 | 134626 |

Speedup so với PennyLane naive (b=1, median): PennyLane vectorized: x65, Classical Conv2d 2x2: x5051, QLUT-4bit-nearest (torch): x2487, QLUT-4bit-interp (torch): x845, QLUT-5bit-nearest (torch): x1850, QLUT-4bit-nearest (numpy): x18558

### breastmnist — Latency CPU (threads_1)

| Engine | b=1 ms/ảnh (median) | b=1 p95 | b=1 FPS | b=N ms/ảnh | b=N FPS |
|---|---|---|---|---|---|
| Classical Conv2d 2x2 | 0.0120 | 0.0130 | 83333 | 0.0017 | 578871 |
| QLUT-4bit-nearest (torch) | 0.0566 | 0.1013 | 17652 | 0.0071 | 141523 |
| QLUT-4bit-interp (torch) | 0.2597 | 0.3617 | 3851 | 0.1974 | 5065 |
| QLUT-5bit-nearest (torch) | 0.0598 | 0.0764 | 16722 | 0.0074 | 135731 |
| QLUT-4bit-nearest (numpy) | 0.0149 | 0.0158 | 67114 | 0.0058 | 171939 |

## octmnist — Fidelity (so với Ground Truth)

| Phương pháp | MAE | RMSE | nRMSE | Cosine | Max abs err | P99 abs err |
|---|---|---|---|---|---|---|
| ExactBatched(PennyLane vec) | 0.00000 | 0.00000 | 0.0000 | 1.000000 | 0.0000 | 0.0000 |
| QLUT-4bit-nearest | 0.02156 | 0.03059 | 0.1743 | 0.984833 | 0.1144 | 0.0957 |
| QLUT-4bit-interp | 0.00128 | 0.00178 | 0.0102 | 0.999992 | 0.0083 | 0.0058 |
| QLUT-5bit-nearest | 0.01040 | 0.01452 | 0.0827 | 0.996608 | 0.0561 | 0.0460 |
| QLUT-5bit-interp | 0.00030 | 0.00042 | 0.0024 | 1.000000 | 0.0021 | 0.0014 |

### octmnist — Latency CPU (threads_default)

| Engine | b=1 ms/ảnh (median) | b=1 p95 | b=1 FPS | b=N ms/ảnh | b=N FPS |
|---|---|---|---|---|---|
| PennyLane naive (repo) | 248.8965 | 267.6172 | 4 | 248.8965 | 4 |
| PennyLane vectorized | 3.8783 | 4.4712 | 258 | 1.5820 | 632 |
| Classical Conv2d 2x2 | 0.0524 | 0.0636 | 19102 | 0.0017 | 593296 |
| QLUT-4bit-nearest (torch) | 0.0985 | 0.1666 | 10152 | 0.0333 | 30019 |
| QLUT-4bit-interp (torch) | 0.3056 | 0.5207 | 3272 | 0.1232 | 8116 |
| QLUT-5bit-nearest (torch) | 0.1151 | 0.2150 | 8688 | 0.0397 | 25202 |
| QLUT-4bit-nearest (numpy) | 0.0150 | 0.0252 | 66667 | 0.0069 | 145085 |

Speedup so với PennyLane naive (b=1, median): PennyLane vectorized: x64, Classical Conv2d 2x2: x4754, QLUT-4bit-nearest (torch): x2527, QLUT-4bit-interp (torch): x814, QLUT-5bit-nearest (torch): x2162, QLUT-4bit-nearest (numpy): x16593

### octmnist — Latency CPU (threads_1)

| Engine | b=1 ms/ảnh (median) | b=1 p95 | b=1 FPS | b=N ms/ảnh | b=N FPS |
|---|---|---|---|---|---|
| Classical Conv2d 2x2 | 0.0097 | 0.0109 | 103093 | 0.0014 | 694444 |
| QLUT-4bit-nearest (torch) | 0.0519 | 0.0699 | 19268 | 0.0068 | 147863 |
| QLUT-4bit-interp (torch) | 0.2561 | 0.4152 | 3905 | 0.2026 | 4935 |
| QLUT-5bit-nearest (torch) | 0.0518 | 0.1547 | 19305 | 0.0072 | 138860 |
| QLUT-4bit-nearest (numpy) | 0.0145 | 0.0170 | 68966 | 0.0063 | 159413 |

Kích thước bảng: 4bit=1.00 MiB, 5bit=16.00 MiB