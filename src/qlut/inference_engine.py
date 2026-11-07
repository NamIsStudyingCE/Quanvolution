"""
Q-LUT Inference Engine  (Step 2 của MVE)
========================================
Các lớp có CÙNG giao diện:  x (B, 1, H, W) trong [0,1]  ->  (B, 4, H/2, W/2)

- QLUT_Quanv2d        : suy luận bằng bảng tra Q-LUT (mode "nearest" hoặc "interp"), KHÔNG vòng lặp for
                        theo patch, KHÔNG mô phỏng lượng tử. Chạy CPU/GPU.
- ExactBatchedQuanv2d : mô phỏng chính xác (liên tục, float64) nhưng vector hoá theo lô patch
                        (PennyLane default.qubit + broadcasting)  -> "Ground Truth" nhanh dùng cho fidelity.
- PennyLaneQuanv2d    : đường chạy NAIVE gốc của repo (apply_quanv_to_image: từng patch, từng ảnh).

Thứ tự pixel trong patch = row-major của F.unfold  ≡ (img[j,k], img[j,k+1], img[j+1,k], img[j+1,k+1]).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .compiler import N_QUBITS, build_batched_qnode, load_qlut


def _extract_patches(x: torch.Tensor) -> torch.Tensor:
    """(B,1,H,W) -> (B, P, 4) với P = (H/2)*(W/2); không vòng lặp."""
    return F.unfold(x, kernel_size=2, stride=2).transpose(1, 2)


class QLUT_Quanv2d(nn.Module):
    """Quanvolution 2x2/stride 2 chạy bằng bảng tra cứu.

    mode="nearest": làm tròn mỗi pixel về lưới n_bit, index bằng phép dịch bit, tra O(1).
    mode="interp" : nội suy đa tuyến 4D giữa 16 đỉnh của siêu lập phương lưới bao quanh điểm thật.
    """

    def __init__(self, table: torch.Tensor = None, n_bits: int = 4, circuit: str = "strongly_L2",
                 mode: str = "nearest"):
        super().__init__()
        if table is None:
            table = load_qlut(n_bits, circuit)
        L = table.shape[0]
        assert table.shape == (L, L, L, L, N_QUBITS) and L == (1 << n_bits), "table không khớp n_bits"
        assert mode in ("nearest", "interp")
        self.n_bits, self.L, self.mode = n_bits, L, mode
        self.register_buffer("table", table.reshape(-1, N_QUBITS).contiguous())          # (L^4, 4)
        shifts = torch.tensor([3 * n_bits, 2 * n_bits, n_bits, 0], dtype=torch.long)
        self.register_buffer("shifts", shifts)
        # 16 đỉnh của hypercube: bits[v] in {0,1}^4 ; strides để đổi toạ độ -> index tuyến tính
        bits = torch.tensor([[(v >> (3 - d)) & 1 for d in range(4)] for v in range(16)], dtype=torch.long)
        self.register_buffer("corner_bits", bits)                                         # (16, 4)
        self.register_buffer("strides", torch.tensor([L ** 3, L ** 2, L, 1], dtype=torch.long))

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        patches = _extract_patches(x).clamp_(0.0, 1.0)                                    # (B, P, 4)
        s = patches * (self.L - 1)
        if self.mode == "nearest":
            p = torch.round(s).long()                                                     # (B, P, 4)
            idx = (p << self.shifts).sum(-1)                                              # ≡ (p0<<3n)|(p1<<2n)|(p2<<n)|p3
            out = self.table[idx]                                                         # (B, P, 4)
        else:
            lo = torch.floor(s).clamp_(0, self.L - 2)                                     # (B, P, 4)
            frac = s - lo
            lo = lo.long()
            corner = lo.unsqueeze(-2) + self.corner_bits                                  # (B, P, 16, 4)
            idx = (corner * self.strides).sum(-1)                                         # (B, P, 16)
            f = frac.unsqueeze(-2)                                                        # (B, P, 1, 4)
            w = torch.where(self.corner_bits.bool(), f, 1.0 - f).prod(-1)                 # (B, P, 16)
            out = (w.unsqueeze(-1) * self.table[idx]).sum(-2)                             # (B, P, 4)
        return out.transpose(1, 2).reshape(B, N_QUBITS, H // 2, W // 2).contiguous()


class ExactBatchedQuanv2d(nn.Module):
    """Mô phỏng lượng tử chính xác, liên tục, vector hoá theo lô patch (không lượng tử hoá đầu vào)."""

    def __init__(self, circuit: str = "strongly_L2"):
        super().__init__()
        self.qnode = build_batched_qnode(circuit)

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        phi = _extract_patches(x).clamp(0, 1).reshape(-1, 4).double().numpy()
        out = np.stack(self.qnode(phi), axis=-1)                                           # (B*P, 4)
        out = torch.from_numpy(out).float()
        return out.reshape(B, -1, N_QUBITS).transpose(1, 2).reshape(B, N_QUBITS, H // 2, W // 2).contiguous()


class PennyLaneQuanv2d(nn.Module):
    """Đường chạy NAIVE của repo: từng ảnh, từng patch 2x2 gọi 1 QNode (apply_quanv_to_image)."""

    def __init__(self, circuit: str = "strongly_L2"):
        super().__init__()
        from src.models.circuits import CIRCUIT_DICT, apply_quanv_to_image
        self.fn, self.apply = CIRCUIT_DICT[circuit], apply_quanv_to_image

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        imgs = x.cpu().numpy()
        out = np.stack([self.apply(img, self.fn) for img in imgs])                         # (B, 4, 14, 14)
        return torch.from_numpy(out).float()
