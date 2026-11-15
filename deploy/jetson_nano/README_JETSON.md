# Hướng dẫn Triển khai & Đo kiểm trên NVIDIA Jetson Nano (5 phút)

Thư mục này chứa gói thực thi độc lập (`aarch64-unknown-linux-gnu`, tối ưu cho **ARM Cortex-A57 @ 1.43 GHz, 2 MB L2 Cache, ARMv8 NEON**), **không cần Python, PyTorch hay CUDA**.

## Bước 1: Copy gói triển khai sang Jetson Nano

Từ máy tính cá nhân, chạy lệnh `scp` (thay `jetson` và `<JETSON_IP>` bằng tài khoản và IP bo mạch):

```bash
scp -r dist/jetson_nano jetson@<JETSON_IP>:~/medquanv_deploy
scp deploy/jetson_nano/run_on_board_bench.sh jetson@<JETSON_IP>:~/medquanv_deploy/
```

## Bước 2: Chạy đo kiểm thực tế trên Jetson Nano

SSH vào Jetson Nano và chạy đúng 1 lệnh:

```bash
ssh jetson@<JETSON_IP>
cd ~/medquanv_deploy
chmod +x medquanv-cli run_on_board_bench.sh
./run_on_board_bench.sh
```

## Bước 3: Lấy kết quả đưa vào Luận văn

Script sẽ tự động tạo 2 tệp báo cáo:
- `jetson_hardware_report.txt`: Thông số `lscpu`, `nvpmodel`, độ trễ `phase_damping_u8`, `rust_qlut` (`INT8 Shift-LUT` 256 KiB và `Interp4D` 1,024 KiB), `End-to-End (ms/img, FPS)`, và `Maximum resident set size` (RAM thực tế).
- `jetson_hardware_report.json`: Số liệu JSON chuẩn hóa.

Copy ngược về máy tính:

```bash
scp jetson@<JETSON_IP>:~/medquanv_deploy/jetson_hardware_report.* ./
```
