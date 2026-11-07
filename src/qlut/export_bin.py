"""
Xuất bảng Q-LUT sang định dạng nhị phân phẳng (raw binary) căn chỉnh 64-byte cache-line
để nhúng trực tiếp (`include_bytes!`) hoặc nạp vào Native Rust SIMD Engine (`rust_qlut`).
"""
import os
import numpy as np
import torch
from src.qlut.compiler import N_QUBITS, WEIGHTS_DIR, build_batched_qnode, load_qlut


def compile_binmid_4bit(circuit: str = "strongly_L2") -> torch.Tensor:
    """
    Biên dịch bảng 4-bit (16^4 = 65,536 trạng thái) tại tâm thực tế của phép dịch bit `p >> 4`:
    Mỗi bin b in 0..15 chứa các mức pixel u8 từ 16*b đến 16*b + 15 -> tâm là (16*b + 7.5) / 255.0.
    """
    qnode = build_batched_qnode(circuit)
    L = 16
    total = L ** 4
    levels = (16.0 * np.arange(L, dtype=np.float64) + 7.5) / 255.0
    idx = np.arange(total)
    p0 = idx >> 12
    p1 = (idx >> 8) & 0xF
    p2 = (idx >> 4) & 0xF
    p3 = idx & 0xF
    phi = np.stack([levels[p0], levels[p1], levels[p2], levels[p3]], axis=1)
    out = np.stack(qnode(phi), axis=-1).astype(np.float32)
    return torch.from_numpy(out).view(L, L, L, L, N_QUBITS).contiguous()


def export_all(circuit: str = "strongly_L2"):
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    table_grid = load_qlut(4, circuit).numpy().reshape(-1, 4).astype(np.float32)
    table_shift = compile_binmid_4bit(circuit).numpy().reshape(-1, 4).astype(np.float32)

    # 1. INT8 (256 KiB) cho lưới chuẩn k/15.0 (dùng với công thức làm tròn u8 -> (p*15+127)>>8)
    int8_grid = np.clip(np.rint(table_grid * 127.0), -127, 127).astype(np.int8)
    p_int8_grid = os.path.join(WEIGHTS_DIR, f"qlut_{circuit}_4bit_int8_grid.bin")
    int8_grid.tofile(p_int8_grid)

    # 2. INT8 (256 KiB) cho phép dịch bit trực tiếp `p >> 4` (tâm bin (16*b+7.5)/255)
    int8_shift = np.clip(np.rint(table_shift * 127.0), -127, 127).astype(np.int8)
    p_int8_shift = os.path.join(WEIGHTS_DIR, f"qlut_{circuit}_4bit_int8_shift.bin")
    int8_shift.tofile(p_int8_shift)

    # 3. FP32 (1 MiB) cho lưới chuẩn k/15.0 (dùng cho FP32 lookup & 4D Multilinear Interpolation)
    p_fp32_grid = os.path.join(WEIGHTS_DIR, f"qlut_{circuit}_4bit_fp32_grid.bin")
    table_grid.tofile(p_fp32_grid)

    # 4. FP32 (1 MiB) cho phép dịch bit `p >> 4`
    p_fp32_shift = os.path.join(WEIGHTS_DIR, f"qlut_{circuit}_4bit_fp32_shift.bin")
    table_shift.tofile(p_fp32_shift)

    for path in (p_int8_grid, p_int8_shift, p_fp32_grid, p_fp32_shift):
        sz = os.path.getsize(path)
        print(f"[export] {os.path.basename(path):<42} {sz:>10,} bytes ({sz / 1024:.1f} KiB)")


if __name__ == "__main__":
    export_all("strongly_L2")
