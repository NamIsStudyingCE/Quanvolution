# BÁO CÁO THỰC NGHIỆM TĂNG TỐC CỰC ĐOAN: NATIVE RUST SIMD Q-LUT (2026-11-07 20:28:51)

- **CPU:** Intel64 Family 6 Model 186 Stepping 2, GenuineIntel (Intel Core i5-13420H, 8 Cores / 12 Threads, AVX2 + FMA)
- **Mạch lượng tử:** `strongly_L2` (4-qubit, patch $2 \times 2$, stride 2)
- **Cấu hình đo:** 100 lần warmup, 1,000 lần benchmark với `time.perf_counter_ns()`
- **Dung lượng bảng Q-LUT:** INT8 = `256.0 KiB` (khóa chặt 100% trong L2 Cache), FP32 = `1,024.0 KiB`

## 1. Tập dữ liệu `BREASTMNIST` — Độ bảo toàn đặc trưng (Fidelity vs PennyLane Ground Truth)

| Phương pháp | Bảng nhớ | MAE | RMSE | nRMSE | Cosine Similarity | Max Abs Err | P99 Abs Err |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Rust-INT8-Grid (256KB)** | 256 KiB | 0.02012 | 0.02759 | 0.1401 | **0.990059** | 0.1156 | 0.0835 |
| **Rust-INT8-Bitshift (256KB, p>>4)** | 256 KiB | 0.01921 | 0.02632 | 0.1336 | **0.990907** | 0.1063 | 0.0805 |
| **Rust-FP32-Grid (1MB)** | 1,024 KiB | 0.01998 | 0.02754 | 0.1398 | **0.990084** | 0.1153 | 0.0835 |
| **Rust-FP32-Bitshift (1MB, p>>4)** | 1,024 KiB | 0.01907 | 0.02622 | 0.1331 | **0.990973** | 0.1058 | 0.0806 |
| **Rust-Interp4D-SIMD (1MB)** | 1,024 KiB | 0.00148 | 0.00201 | 0.0102 | **0.999993** | 0.0097 | 0.0060 |

### `BREASTMNIST` — Tốc độ Thực thi Phần cứng (`batch_1_single_thread`)

| Engine | Median (µs/ảnh) | Median (ms/ảnh) | Min (µs/ảnh) | P95 (µs/ảnh) | Throughput (FPS) |
|---|---:|---:|---:|---:|---:|
| **Classical Conv2D 2x2 (from uint8)** | **16.200 µs** | 0.016200 ms | 15.000 µs | 24.225 µs | **61,728 FPS** |
| **Classical Conv2D 2x2 (pre-float32)** | **9.900 µs** | 0.009900 ms | 9.100 µs | 14.805 µs | **101,010 FPS** |
| **PyTorch Q-LUT 4bit Nearest (FP32)** | **50.000 µs** | 0.050000 ms | 46.900 µs | 90.245 µs | **20,000 FPS** |
| **PyTorch Q-LUT 4bit Interp4D (FP32)** | **241.000 µs** | 0.241000 ms | 225.200 µs | 431.350 µs | **4,149 FPS** |
| **Rust Q-LUT INT8 NHWC (256KB, ptr)** | **0.300 µs** | 0.000300 ms | 0.200 µs | 0.300 µs | **3,333,333 FPS** |
| **Rust Q-LUT INT8 NCHW (256KB, SIMD)** | **0.400 µs** | 0.000400 ms | 0.300 µs | 0.400 µs | **2,500,000 FPS** |
| **Rust Q-LUT FP32 NCHW (1MB, SIMD)** | **0.400 µs** | 0.000400 ms | 0.300 µs | 0.400 µs | **2,500,000 FPS** |
| **Rust Q-LUT Interp4D NCHW (SIMD FMA)** | **1.100 µs** | 0.001100 ms | 1.000 µs | 1.200 µs | **909,091 FPS** |

### `BREASTMNIST` — Tốc độ Thực thi Phần cứng (`batch_100_single_thread`)

| Engine | Median (µs/ảnh) | Median (ms/ảnh) | Min (µs/ảnh) | P95 (µs/ảnh) | Throughput (FPS) |
|---|---:|---:|---:|---:|---:|
| **Classical Conv2D 2x2 (from uint8)** | **1.988 µs** | 0.001988 ms | 1.892 µs | 3.109 µs | **503,018 FPS** |
| **Classical Conv2D 2x2 (pre-float32)** | **1.421 µs** | 0.001421 ms | 1.333 µs | 2.069 µs | **703,977 FPS** |
| **PyTorch Q-LUT 4bit Nearest (FP32)** | **7.024 µs** | 0.007024 ms | 6.113 µs | 10.720 µs | **142,379 FPS** |
| **PyTorch Q-LUT 4bit Interp4D (FP32)** | **197.673 µs** | 0.197673 ms | 173.218 µs | 213.348 µs | **5,059 FPS** |
| **Rust Q-LUT INT8 NHWC (256KB, 1-thread)** | **0.116 µs** | 0.000116 ms | 0.113 µs | 0.148 µs | **8,620,690 FPS** |
| **Rust Q-LUT INT8 NCHW (256KB, 1-thread)** | **0.191 µs** | 0.000191 ms | 0.184 µs | 0.263 µs | **5,235,602 FPS** |
| **Rust Q-LUT FP32 NCHW (1MB, 1-thread)** | **0.299 µs** | 0.000299 ms | 0.228 µs | 0.412 µs | **3,344,482 FPS** |
| **Rust Q-LUT Interp4D NCHW (1-thread)** | **0.948 µs** | 0.000947 ms | 0.929 µs | 1.387 µs | **1,055,409 FPS** |

### `BREASTMNIST` — Tốc độ Thực thi Phần cứng (`batch_multithread`)

| Engine | Median (µs/ảnh) | Median (ms/ảnh) | Min (µs/ảnh) | P95 (µs/ảnh) | Throughput (FPS) |
|---|---:|---:|---:|---:|---:|
| **Classical Conv2D 2x2 (from uint8, MT)** | **2.892 µs** | 0.002892 ms | 2.339 µs | 5.700 µs | **345,781 FPS** |
| **Classical Conv2D 2x2 (pre-float32, MT)** | **1.678 µs** | 0.001678 ms | 1.517 µs | 3.703 µs | **595,948 FPS** |
| **PyTorch Q-LUT 4bit Nearest (MT)** | **39.926 µs** | 0.039926 ms | 31.978 µs | 95.306 µs | **25,046 FPS** |
| **Rust Q-LUT INT8 NHWC (256KB, Rayon MT)** | **0.776 µs** | 0.000776 ms | 0.342 µs | 1.150 µs | **1,288,660 FPS** |
| **Rust Q-LUT INT8 NCHW (256KB, Rayon MT)** | **0.833 µs** | 0.000833 ms | 0.297 µs | 1.244 µs | **1,199,760 FPS** |
| **Rust Q-LUT FP32 NCHW (1MB, Rayon MT)** | **0.842 µs** | 0.000842 ms | 0.441 µs | 1.290 µs | **1,188,354 FPS** |
| **Rust Q-LUT Interp4D NCHW (Rayon MT)** | **1.140 µs** | 0.001140 ms | 0.629 µs | 1.587 µs | **877,193 FPS** |
| **Rust Q-LUT INT8 NHWC (B=1000, Rayon MT)** | **0.155 µs** | 0.000155 ms | 0.092 µs | 0.204 µs | **6,453,695 FPS** |
| **Classical Conv2D 2x2 (B=1000, from uint8, MT)** | **1.725 µs** | 0.001725 ms | 0.939 µs | 2.722 µs | **579,828 FPS** |
| **Classical Conv2D 2x2 (B=1000, pre-float32, MT)** | **1.082 µs** | 0.001082 ms | 0.515 µs | 1.628 µs | **923,916 FPS** |

## 1. Tập dữ liệu `OCTMNIST` — Độ bảo toàn đặc trưng (Fidelity vs PennyLane Ground Truth)

| Phương pháp | Bảng nhớ | MAE | RMSE | nRMSE | Cosine Similarity | Max Abs Err | P99 Abs Err |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Rust-INT8-Grid (256KB)** | 256 KiB | 0.02212 | 0.03091 | 0.1758 | **0.984691** | 0.1168 | 0.0960 |
| **Rust-INT8-Bitshift (256KB, p>>4)** | 256 KiB | 0.02196 | 0.03027 | 0.1721 | **0.985600** | 0.1057 | 0.0918 |
| **Rust-FP32-Grid (1MB)** | 1,024 KiB | 0.02195 | 0.03089 | 0.1757 | **0.984670** | 0.1188 | 0.0959 |
| **Rust-FP32-Bitshift (1MB, p>>4)** | 1,024 KiB | 0.02164 | 0.02989 | 0.1700 | **0.985956** | 0.1057 | 0.0906 |
| **Rust-Interp4D-SIMD (1MB)** | 1,024 KiB | 0.00128 | 0.00178 | 0.0101 | **0.999992** | 0.0089 | 0.0057 |

### `OCTMNIST` — Tốc độ Thực thi Phần cứng (`batch_1_single_thread`)

| Engine | Median (µs/ảnh) | Median (ms/ảnh) | Min (µs/ảnh) | P95 (µs/ảnh) | Throughput (FPS) |
|---|---:|---:|---:|---:|---:|
| **Classical Conv2D 2x2 (from uint8)** | **17.200 µs** | 0.017200 ms | 14.900 µs | 27.315 µs | **58,140 FPS** |
| **Classical Conv2D 2x2 (pre-float32)** | **9.900 µs** | 0.009900 ms | 9.100 µs | 14.500 µs | **101,010 FPS** |
| **PyTorch Q-LUT 4bit Nearest (FP32)** | **52.100 µs** | 0.052100 ms | 47.300 µs | 79.700 µs | **19,194 FPS** |
| **PyTorch Q-LUT 4bit Interp4D (FP32)** | **252.450 µs** | 0.252450 ms | 226.500 µs | 448.335 µs | **3,961 FPS** |
| **Rust Q-LUT INT8 NHWC (256KB, ptr)** | **0.300 µs** | 0.000300 ms | 0.200 µs | 0.300 µs | **3,333,333 FPS** |
| **Rust Q-LUT INT8 NCHW (256KB, SIMD)** | **0.300 µs** | 0.000300 ms | 0.300 µs | 0.400 µs | **3,333,333 FPS** |
| **Rust Q-LUT FP32 NCHW (1MB, SIMD)** | **0.400 µs** | 0.000400 ms | 0.300 µs | 0.400 µs | **2,500,000 FPS** |
| **Rust Q-LUT Interp4D NCHW (SIMD FMA)** | **1.100 µs** | 0.001100 ms | 1.000 µs | 1.100 µs | **909,091 FPS** |

### `OCTMNIST` — Tốc độ Thực thi Phần cứng (`batch_100_single_thread`)

| Engine | Median (µs/ảnh) | Median (ms/ảnh) | Min (µs/ảnh) | P95 (µs/ảnh) | Throughput (FPS) |
|---|---:|---:|---:|---:|---:|
| **Classical Conv2D 2x2 (from uint8)** | **2.101 µs** | 0.002101 ms | 1.877 µs | 3.251 µs | **476,077 FPS** |
| **Classical Conv2D 2x2 (pre-float32)** | **1.440 µs** | 0.001440 ms | 1.333 µs | 2.592 µs | **694,444 FPS** |
| **PyTorch Q-LUT 4bit Nearest (FP32)** | **7.198 µs** | 0.007198 ms | 6.173 µs | 12.372 µs | **138,927 FPS** |
| **PyTorch Q-LUT 4bit Interp4D (FP32)** | **204.738 µs** | 0.204738 ms | 174.139 µs | 229.302 µs | **4,884 FPS** |
| **Rust Q-LUT INT8 NHWC (256KB, 1-thread)** | **0.117 µs** | 0.000117 ms | 0.113 µs | 0.133 µs | **8,547,009 FPS** |
| **Rust Q-LUT INT8 NCHW (256KB, 1-thread)** | **0.190 µs** | 0.000190 ms | 0.184 µs | 0.251 µs | **5,263,158 FPS** |
| **Rust Q-LUT FP32 NCHW (1MB, 1-thread)** | **0.234 µs** | 0.000234 ms | 0.217 µs | 0.349 µs | **4,273,504 FPS** |
| **Rust Q-LUT Interp4D NCHW (1-thread)** | **0.910 µs** | 0.000910 ms | 0.870 µs | 1.177 µs | **1,098,901 FPS** |

### `OCTMNIST` — Tốc độ Thực thi Phần cứng (`batch_multithread`)

| Engine | Median (µs/ảnh) | Median (ms/ảnh) | Min (µs/ảnh) | P95 (µs/ảnh) | Throughput (FPS) |
|---|---:|---:|---:|---:|---:|
| **Classical Conv2D 2x2 (from uint8, MT)** | **2.950 µs** | 0.002950 ms | 2.412 µs | 5.353 µs | **338,983 FPS** |
| **Classical Conv2D 2x2 (pre-float32, MT)** | **1.766 µs** | 0.001766 ms | 1.425 µs | 3.645 µs | **566,251 FPS** |
| **PyTorch Q-LUT 4bit Nearest (MT)** | **37.023 µs** | 0.037023 ms | 33.408 µs | 42.751 µs | **27,010 FPS** |
| **Rust Q-LUT INT8 NHWC (256KB, Rayon MT)** | **0.750 µs** | 0.000750 ms | 0.336 µs | 1.138 µs | **1,333,333 FPS** |
| **Rust Q-LUT INT8 NCHW (256KB, Rayon MT)** | **0.809 µs** | 0.000809 ms | 0.311 µs | 1.179 µs | **1,236,094 FPS** |
| **Rust Q-LUT FP32 NCHW (1MB, Rayon MT)** | **0.844 µs** | 0.000844 ms | 0.408 µs | 1.239 µs | **1,184,834 FPS** |
| **Rust Q-LUT Interp4D NCHW (Rayon MT)** | **1.119 µs** | 0.001119 ms | 0.617 µs | 1.601 µs | **894,055 FPS** |
| **Rust Q-LUT INT8 NHWC (B=1000, Rayon MT)** | **0.156 µs** | 0.000156 ms | 0.092 µs | 0.219 µs | **6,424,671 FPS** |
| **Classical Conv2D 2x2 (B=1000, from uint8, MT)** | **1.613 µs** | 0.001613 ms | 0.851 µs | 2.728 µs | **619,982 FPS** |
| **Classical Conv2D 2x2 (B=1000, pre-float32, MT)** | **0.803 µs** | 0.000803 ms | 0.507 µs | 1.410 µs | **1,244,787 FPS** |
