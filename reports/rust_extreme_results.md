# BÁO CÁO THỰC NGHIỆM TĂNG TỐC CỰC ĐOAN: NATIVE RUST SIMD Q-LUT (2026-11-07 17:33:57)

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
| **Classical Conv2D 2x2 (from uint8)** | **16.000 µs** | 0.016000 ms | 15.000 µs | 17.405 µs | **62,500 FPS** |
| **Classical Conv2D 2x2 (pre-float32)** | **9.500 µs** | 0.009500 ms | 9.000 µs | 11.000 µs | **105,263 FPS** |
| **PyTorch Q-LUT 4bit Nearest (FP32)** | **50.700 µs** | 0.050700 ms | 46.900 µs | 87.700 µs | **19,724 FPS** |
| **PyTorch Q-LUT 4bit Interp4D (FP32)** | **250.500 µs** | 0.250500 ms | 230.100 µs | 408.720 µs | **3,992 FPS** |
| **Rust Q-LUT INT8 NHWC (256KB, ptr)** | **0.300 µs** | 0.000300 ms | 0.200 µs | 0.400 µs | **3,333,333 FPS** |
| **Rust Q-LUT INT8 NCHW (256KB, SIMD)** | **0.400 µs** | 0.000400 ms | 0.300 µs | 0.400 µs | **2,500,000 FPS** |
| **Rust Q-LUT FP32 NCHW (1MB, SIMD)** | **0.400 µs** | 0.000400 ms | 0.300 µs | 0.400 µs | **2,500,000 FPS** |
| **Rust Q-LUT Interp4D NCHW (SIMD FMA)** | **1.100 µs** | 0.001100 ms | 1.100 µs | 1.200 µs | **909,091 FPS** |

### `BREASTMNIST` — Tốc độ Thực thi Phần cứng (`batch_100_single_thread`)

| Engine | Median (µs/ảnh) | Median (ms/ảnh) | Min (µs/ảnh) | P95 (µs/ảnh) | Throughput (FPS) |
|---|---:|---:|---:|---:|---:|
| **Classical Conv2D 2x2 (from uint8)** | **1.920 µs** | 0.001920 ms | 1.832 µs | 3.070 µs | **520,833 FPS** |
| **Classical Conv2D 2x2 (pre-float32)** | **1.363 µs** | 0.001362 ms | 1.281 µs | 2.127 µs | **733,945 FPS** |
| **PyTorch Q-LUT 4bit Nearest (FP32)** | **6.585 µs** | 0.006585 ms | 6.052 µs | 9.358 µs | **151,872 FPS** |
| **PyTorch Q-LUT 4bit Interp4D (FP32)** | **192.946 µs** | 0.192946 ms | 176.665 µs | 211.535 µs | **5,183 FPS** |
| **Rust Q-LUT INT8 NHWC (256KB, 1-thread)** | **0.120 µs** | 0.000120 ms | 0.115 µs | 0.134 µs | **8,333,333 FPS** |
| **Rust Q-LUT INT8 NCHW (256KB, 1-thread)** | **0.208 µs** | 0.000208 ms | 0.188 µs | 0.256 µs | **4,807,692 FPS** |
| **Rust Q-LUT FP32 NCHW (1MB, 1-thread)** | **0.303 µs** | 0.000303 ms | 0.230 µs | 0.405 µs | **3,300,330 FPS** |
| **Rust Q-LUT Interp4D NCHW (1-thread)** | **0.948 µs** | 0.000947 ms | 0.922 µs | 1.738 µs | **1,055,409 FPS** |

### `BREASTMNIST` — Tốc độ Thực thi Phần cứng (`batch_multithread`)

| Engine | Median (µs/ảnh) | Median (ms/ảnh) | Min (µs/ảnh) | P95 (µs/ảnh) | Throughput (FPS) |
|---|---:|---:|---:|---:|---:|
| **Classical Conv2D 2x2 (from uint8, MT)** | **2.911 µs** | 0.002911 ms | 2.355 µs | 4.060 µs | **343,584 FPS** |
| **Classical Conv2D 2x2 (pre-float32, MT)** | **1.705 µs** | 0.001705 ms | 1.544 µs | 2.333 µs | **586,510 FPS** |
| **PyTorch Q-LUT 4bit Nearest (MT)** | **36.212 µs** | 0.036213 ms | 32.252 µs | 42.718 µs | **27,615 FPS** |
| **Rust Q-LUT INT8 NHWC (256KB, Rayon MT)** | **0.900 µs** | 0.000900 ms | 0.411 µs | 1.323 µs | **1,111,729 FPS** |
| **Rust Q-LUT INT8 NCHW (256KB, Rayon MT)** | **0.793 µs** | 0.000793 ms | 0.398 µs | 1.200 µs | **1,261,034 FPS** |
| **Rust Q-LUT FP32 NCHW (1MB, Rayon MT)** | **0.807 µs** | 0.000807 ms | 0.335 µs | 1.204 µs | **1,239,926 FPS** |
| **Rust Q-LUT Interp4D NCHW (Rayon MT)** | **1.109 µs** | 0.001109 ms | 0.638 µs | 1.579 µs | **902,120 FPS** |
| **Rust Q-LUT INT8 NHWC (B=1000, Rayon MT)** | **0.164 µs** | 0.000164 ms | 0.083 µs | 0.219 µs | **6,093,845 FPS** |
| **Classical Conv2D 2x2 (B=1000, from uint8, MT)** | **1.829 µs** | 0.001829 ms | 0.842 µs | 4.733 µs | **546,822 FPS** |
| **Classical Conv2D 2x2 (B=1000, pre-float32, MT)** | **0.615 µs** | 0.000615 ms | 0.465 µs | 0.901 µs | **1,626,545 FPS** |

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
| **Classical Conv2D 2x2 (from uint8)** | **16.000 µs** | 0.016000 ms | 14.800 µs | 20.900 µs | **62,500 FPS** |
| **Classical Conv2D 2x2 (pre-float32)** | **10.000 µs** | 0.010000 ms | 9.100 µs | 13.400 µs | **100,000 FPS** |
| **PyTorch Q-LUT 4bit Nearest (FP32)** | **52.800 µs** | 0.052800 ms | 47.400 µs | 77.330 µs | **18,939 FPS** |
| **PyTorch Q-LUT 4bit Interp4D (FP32)** | **264.500 µs** | 0.264500 ms | 227.100 µs | 415.225 µs | **3,781 FPS** |
| **Rust Q-LUT INT8 NHWC (256KB, ptr)** | **0.300 µs** | 0.000300 ms | 0.200 µs | 0.300 µs | **3,333,333 FPS** |
| **Rust Q-LUT INT8 NCHW (256KB, SIMD)** | **0.400 µs** | 0.000400 ms | 0.300 µs | 0.400 µs | **2,500,000 FPS** |
| **Rust Q-LUT FP32 NCHW (1MB, SIMD)** | **0.400 µs** | 0.000400 ms | 0.300 µs | 0.400 µs | **2,500,000 FPS** |
| **Rust Q-LUT Interp4D NCHW (SIMD FMA)** | **1.000 µs** | 0.001000 ms | 1.000 µs | 1.100 µs | **1,000,000 FPS** |

### `OCTMNIST` — Tốc độ Thực thi Phần cứng (`batch_100_single_thread`)

| Engine | Median (µs/ảnh) | Median (ms/ảnh) | Min (µs/ảnh) | P95 (µs/ảnh) | Throughput (FPS) |
|---|---:|---:|---:|---:|---:|
| **Classical Conv2D 2x2 (from uint8)** | **2.071 µs** | 0.002070 ms | 1.903 µs | 3.443 µs | **482,975 FPS** |
| **Classical Conv2D 2x2 (pre-float32)** | **1.524 µs** | 0.001524 ms | 1.340 µs | 2.746 µs | **656,383 FPS** |
| **PyTorch Q-LUT 4bit Nearest (FP32)** | **7.160 µs** | 0.007160 ms | 6.130 µs | 9.583 µs | **139,665 FPS** |
| **PyTorch Q-LUT 4bit Interp4D (FP32)** | **200.319 µs** | 0.200319 ms | 179.319 µs | 219.833 µs | **4,992 FPS** |
| **Rust Q-LUT INT8 NHWC (256KB, 1-thread)** | **0.139 µs** | 0.000139 ms | 0.121 µs | 0.169 µs | **7,194,245 FPS** |
| **Rust Q-LUT INT8 NCHW (256KB, 1-thread)** | **0.209 µs** | 0.000209 ms | 0.202 µs | 0.244 µs | **4,784,689 FPS** |
| **Rust Q-LUT FP32 NCHW (1MB, 1-thread)** | **0.301 µs** | 0.000301 ms | 0.299 µs | 0.356 µs | **3,322,259 FPS** |
| **Rust Q-LUT Interp4D NCHW (1-thread)** | **0.885 µs** | 0.000885 ms | 0.880 µs | 1.123 µs | **1,129,944 FPS** |

### `OCTMNIST` — Tốc độ Thực thi Phần cứng (`batch_multithread`)

| Engine | Median (µs/ảnh) | Median (ms/ảnh) | Min (µs/ảnh) | P95 (µs/ảnh) | Throughput (FPS) |
|---|---:|---:|---:|---:|---:|
| **Classical Conv2D 2x2 (from uint8, MT)** | **2.736 µs** | 0.002736 ms | 2.314 µs | 3.835 µs | **365,564 FPS** |
| **Classical Conv2D 2x2 (pre-float32, MT)** | **1.732 µs** | 0.001732 ms | 1.573 µs | 2.422 µs | **577,367 FPS** |
| **PyTorch Q-LUT 4bit Nearest (MT)** | **37.977 µs** | 0.037977 ms | 32.890 µs | 45.786 µs | **26,332 FPS** |
| **Rust Q-LUT INT8 NHWC (256KB, Rayon MT)** | **0.770 µs** | 0.000770 ms | 0.391 µs | 1.150 µs | **1,297,859 FPS** |
| **Rust Q-LUT INT8 NCHW (256KB, Rayon MT)** | **0.805 µs** | 0.000806 ms | 0.416 µs | 1.180 µs | **1,241,465 FPS** |
| **Rust Q-LUT FP32 NCHW (1MB, Rayon MT)** | **0.817 µs** | 0.000817 ms | 0.458 µs | 1.176 µs | **1,223,990 FPS** |
| **Rust Q-LUT Interp4D NCHW (Rayon MT)** | **1.131 µs** | 0.001131 ms | 0.600 µs | 1.566 µs | **884,173 FPS** |
| **Rust Q-LUT INT8 NHWC (B=1000, Rayon MT)** | **0.157 µs** | 0.000157 ms | 0.099 µs | 0.206 µs | **6,389,776 FPS** |
| **Classical Conv2D 2x2 (B=1000, from uint8, MT)** | **1.377 µs** | 0.001377 ms | 0.797 µs | 2.491 µs | **726,427 FPS** |
| **Classical Conv2D 2x2 (B=1000, pre-float32, MT)** | **0.892 µs** | 0.000892 ms | 0.490 µs | 1.342 µs | **1,121,202 FPS** |
