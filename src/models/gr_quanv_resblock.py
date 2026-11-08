"""
GR-Quanv Residual Head + Medical Data Augmentation (Cột trụ 2 - mở rộng)
=======================================================================
- ResidualQuanvHead: đầu phân loại dạng Residual nhỏ (~55k tham số, ~200 lần nhỏ hơn ResNet-18),
  thay thế đầu `BN -> ReLU -> Linear` của mọi mô hình trong `gr_quanv.py` bằng CÙNG một cấu trúc
  (đối chứng sòng phẳng: Classical Baseline / Classical Control / GR-Quanv đều dùng chung head này).
- GPUMedicalAugment: lật ngang, affine nhẹ (xoay ±10°, dịch ±5%), jitter độ sáng/tương phản.
  Chạy hoàn toàn trên GPU, trả về uint8 để đưa thẳng vào nhân Rust `rust_qlut` (đặc trưng lượng tử
  được tính lại trên ảnh đã augment, nên nhánh lượng tử và nhánh cổ điển luôn nhìn CÙNG một ảnh).
- Không dùng ElasticTransform: ảnh 28x28 quá nhỏ, biến dạng đàn hồi phá hủy cấu trúc lớp võng mạc.
"""
from __future__ import annotations
import math
from typing import Callable, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.gr_quanv import (
    SymmetricalClassicalBaseline,
    ClassicalMultiScaleGatedControl,
    GatedResidualQuanvNet,
    extract_multiscale_qlut,
)


class _ResBlock(nn.Module):
    """Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN  (+ skip; 1x1 chiếu nếu đổi số kênh) -> ReLU."""

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.c1 = nn.Conv2d(c_in, c_out, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(c_out)
        self.c2 = nn.Conv2d(c_out, c_out, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(c_out)
        self.skip = (
            nn.Identity()
            if c_in == c_out
            else nn.Sequential(nn.Conv2d(c_in, c_out, 1, bias=False), nn.BatchNorm2d(c_out))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return F.relu(y + self.skip(x))


class ResidualQuanvHead(nn.Module):
    """Input (B,4,14,14) -> BN -> ReLU -> Conv3x3(32) -> Res(32) -> Pool -> Res(32->48) -> Pool -> GAP -> Dropout -> Linear."""

    def __init__(self, num_classes: int, in_ch: int = 4, width: int = 32, width2: int = 48, dropout: float = 0.2):
        super().__init__()
        self.pre = nn.Sequential(nn.BatchNorm2d(in_ch), nn.ReLU())
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, padding=1, bias=False), nn.BatchNorm2d(width), nn.ReLU()
        )
        self.block1 = _ResBlock(width, width)
        self.pool1 = nn.MaxPool2d(2)  # 14 -> 7
        self.block2 = _ResBlock(width, width2)
        self.pool2 = nn.MaxPool2d(2)  # 7 -> 3
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(width2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(self.pre(x))
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = x.mean(dim=(2, 3))
        return self.fc(self.drop(x))


def build_model(kind: str, num_classes: int, head: str) -> nn.Module:
    """kind in {classical, control, grquanv}; head in {linear, resblock}.
    Các mô hình luôn được khởi tạo với head tuyến tính rồi (nếu cần) thay bằng ResidualQuanvHead giống hệt nhau."""
    if kind == "classical":
        m = SymmetricalClassicalBaseline(num_classes, "linear")
    elif kind == "control":
        m = ClassicalMultiScaleGatedControl(num_classes, 16, "linear", lean_mixer=False)
    elif kind == "grquanv":
        m = GatedResidualQuanvNet(num_classes, 16, "linear", lean_mixer=False)
    else:
        raise ValueError(kind)
    if head == "resblock":
        m.head = ResidualQuanvHead(num_classes)
    elif head != "linear":
        raise ValueError(head)
    return m


MODEL_KINDS = [
    ("Classical Baseline", "classical"),
    ("Classical Control (0% quantum)", "control"),
    ("Proposed GR-Quanv", "grquanv"),
]


class GPUMedicalAugment:
    """Augment trên GPU. Input x_norm (B,1,28,28) trong [-1,1]; Output: uint8 (B,28,28) trên CPU."""

    def __init__(self, deg: float = 10.0, translate: float = 0.05, bright: float = 0.1, contrast: float = 0.1,
                 flip_p: float = 0.5):
        self.deg, self.translate, self.bright, self.contrast, self.flip_p = deg, translate, bright, contrast, flip_p

    def __call__(self, x_norm: torch.Tensor, gen: torch.Generator) -> np.ndarray:
        B, dev = x_norm.shape[0], x_norm.device
        x = x_norm * 0.5 + 0.5  # [0,1]
        ang = (torch.rand(B, device=dev, generator=gen) * 2 - 1) * self.deg * math.pi / 180.0
        tx = (torch.rand(B, device=dev, generator=gen) * 2 - 1) * self.translate * 2
        ty = (torch.rand(B, device=dev, generator=gen) * 2 - 1) * self.translate * 2
        flip = (torch.rand(B, device=dev, generator=gen) < self.flip_p).float() * -2 + 1  # -1 hoặc +1
        cos, sin = torch.cos(ang), torch.sin(ang)
        theta = torch.zeros(B, 2, 3, device=dev)
        theta[:, 0, 0] = cos * flip
        theta[:, 0, 1] = -sin
        theta[:, 0, 2] = tx
        theta[:, 1, 0] = sin * flip
        theta[:, 1, 1] = cos
        theta[:, 1, 2] = ty
        grid = F.affine_grid(theta, x.shape, align_corners=False)
        x = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)
        b = 1 + (torch.rand(B, 1, 1, 1, device=dev, generator=gen) * 2 - 1) * self.bright
        c = 1 + (torch.rand(B, 1, 1, 1, device=dev, generator=gen) * 2 - 1) * self.contrast
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        x = ((x - mean) * c + mean) * b
        u8 = (x.clamp(0, 1) * 255.0).round().to(torch.uint8).squeeze(1)
        return np.ascontiguousarray(u8.cpu().numpy())


def u8_to_inputs(u8: np.ndarray, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """uint8 (N,28,28) -> (x_norm GPU, q_ms GPU) qua Rust Q-LUT. Cả hai nhánh nhìn cùng ảnh đã lượng tử hóa 8-bit."""
    q, _ = extract_multiscale_qlut(u8, include_axial=True)
    x = (torch.from_numpy(u8).float().unsqueeze(1) / 255.0 - 0.5) / 0.5
    return x.to(device), q.to(device)


if __name__ == "__main__":
    for head in ("linear", "resblock"):
        for name, kind in MODEL_KINDS:
            m = build_model(kind, 4, head)
            print(f"{head:<9} {name:<32} params={sum(p.numel() for p in m.parameters() if p.requires_grad)}")
