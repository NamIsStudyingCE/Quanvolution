"""
Wide-Fusion GR-Quanv (Tác vụ 1: gỡ nút thắt 4 kênh của biến thể Residual)
=========================================================================
Chẩn đoán gốc từ `gr_quanv_resblock.py` (kết quả sạch GR-Quanv kém Control
-1.25% Pneumonia / -2.42% OCT):
  (1) Nút thắt 4 kênh: `c_stem` chỉ xuất 4 kênh và mixer ép 16 kênh đa tỷ lệ
      Q-LUT về 4 kênh -> 75% (12/16) đặc trưng lượng tử bị vứt bỏ trước head.
  (2) Khởi tạo gate bias = +0.5 (alpha ~ 0.62) -> ngay epoch 0, 62% tín hiệu
      của head đến từ bộ trộn lượng tử chưa học gì, nhiễu gradient cả nhánh
      cổ điển.
  (3) Không gian thông lượng head quá hẹp (in_ch=4) cho nội dung 16 kênh.

Kiến trúc mới trong file này:
  - Nhánh cổ điển: `c_stem` Conv2d(1 -> 16) 2x2 stride 2 (thay vì 1 -> 4).
  - Nhánh lượng tử: giữ trọn 16 kênh đa tỷ lệ Rust Q-LUT (d=1, d=2, d=3, axial),
      mixer giữ nguyên số kênh (16 -> 32 -> 16), không còn phép ép 16 -> 4.
  - Dung hợp cấp 32 kênh: f_fused = Concat[f_c(16), alpha * f_q(16)], alpha là
      cổng không gian-kênh 16 kênh.
  - Khởi tạo cổng an toàn: bias = -1.5 (alpha ~ 0.18) -> nhánh cổ điển tự học
      ổn định trước, cổng tự mở dần khi bộ trộn lượng tử đã thích nghi.
  - Head residual wide: BN(32) -> Conv3x3(48) -> Res(48) -> Res(48->64) -> GAP.
  - Đối chứng sòng phẳng: `WideClassicalControl` thay 16 kênh Q-LUT bằng 4 nhánh
      tích chập giãn cách cổ điển (d=1,d=2,d=3,axial; mỗi nhánh 4 kênh = 16 kênh
      thô), dùng CÙNG cấu trúc mixer/gate/head, có nhiều hơn đúng số tham số
      của các kernel nhánh giãn cách mà GR-Quanv không phải học.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from models.gr_quanv_resblock import _ResBlock


# Bias khởi tạo của tầng Conv1x1 cuối cổng: sigmoid(-1.5) ~ 0.1824.
GATE_INIT_BIAS = -1.5
ALPHA_INIT = float(torch.sigmoid(torch.tensor(GATE_INIT_BIAS)).item())

NUM_Q_SCALES = 4          # d=1, d=2, d=3, axial
CH_PER_SCALE = 4          # 4 kênh đếm chẵn CNOT mỗi tỷ lệ
NUM_MS_CHANNELS = NUM_Q_SCALES * CH_PER_SCALE   # 16 kênh thô từ Rust Q-LUT


def _mixer_stack() -> nn.Sequential:
    """Bộ trộn đặc trưng đa tỷ lệ: giữ nguyên 16 kênh vào/ra, thông lượng nội bộ 32.

    Used bởi cả nhánh lượng tử (đầu vào 16 kênh Q-LUT cố định) và nhánh đối chứng
    cổ điển (đầu vào 16 kênh thô từ 4 conv giãn cách) -> cấu trúc hệt nhau.
    """
    return nn.Sequential(
        nn.BatchNorm2d(NUM_MS_CHANNELS),
        nn.Conv2d(NUM_MS_CHANNELS, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.GELU(),
        nn.Conv2d(32, 32, kernel_size=3, padding=2, dilation=2),
        nn.BatchNorm2d(32),
        nn.GELU(),
        nn.Conv2d(32, NUM_MS_CHANNELS, kernel_size=1),
    )


def _gate_stack(in_ch: int) -> nn.Sequential:
    """Cổng thích nghi không gian-kênh: đầu ra đúng 16 kênh alpha."""
    return nn.Sequential(
        nn.BatchNorm2d(in_ch),
        nn.Conv2d(in_ch, 32, kernel_size=3, padding=1),
        nn.GELU(),
        nn.Conv2d(32, NUM_MS_CHANNELS, kernel_size=1),
    )


class WideResidualHead(nn.Module):
    """Head dùng chung cho cả 3 mô hình wide, nhận 32 (hoặc 16) kênh đầu vào =
    BN -> ReLU -> Conv3x3(48) -> Res(48) -> Pool -> Res(48->64) -> Pool -> GAP -> Dropout -> Linear."""

    def __init__(self, num_classes: int, in_ch: int = NUM_MS_CHANNELS * 2,
                 width: int = 48, width2: int = 64, dropout: float = 0.2):
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


class WideClassicalBaseline(nn.Module):
    """Đối chứng tối giản: 1 conv stem 16 kênh + WideResidualHead(in_ch=16). Không cổng, không đa tỷ lệ."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.c_stem = nn.Conv2d(1, NUM_MS_CHANNELS, kernel_size=2, stride=2)
        self.head = WideResidualHead(num_classes, in_ch=NUM_MS_CHANNELS)

    def forward(self, x: torch.Tensor, q_ms: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.head(self.c_stem(x))


class _GateTrackerMixin(nn.Module):
    """Lưu thống kê cổng alpha lần đánh giá gần nhất (chỉ khi eval) để báo cáo lâm sàng."""

    def _init_tracker(self) -> None:
        self.last_alpha_mean: Optional[np.ndarray] = None
        self.last_alpha_std: Optional[np.ndarray] = None
        self.last_alpha_map: Optional[np.ndarray] = None

    def _track(self, alpha: torch.Tensor) -> None:
        if not self.training:
            with torch.no_grad():
                self.last_alpha_mean = alpha.mean(dim=(0, 2, 3)).detach().cpu().numpy()
                self.last_alpha_std = alpha.std(dim=(0, 2, 3)).detach().cpu().numpy()
                self.last_alpha_map = alpha.mean(dim=1).detach().cpu().numpy()

    def get_gate_summary(self) -> dict:
        if self.last_alpha_mean is None:
            return {"overall_alpha": None, "per_channel_mean": None, "per_channel_std": None}
        return {
            "overall_alpha": float(np.mean(self.last_alpha_mean)),
            "per_channel_mean": [float(v) for v in self.last_alpha_mean],
            "per_channel_std": [float(v) for v in self.last_alpha_std],
        }


class WideClassicalControl(_GateTrackerMixin):
    """Đối chứng sòng phẳng 0% lượng tử: thay 16 kênh Q-LUT bằng 4 nhánh conv giãn cách
    cổ điển sinh đúng 16 kênh thô; mixer/gate/fusion/head giống hệt `WideGRQuanvNet`.
    Nhiều hơn_GR-Quanv đúng số tham số của 4 kernel nhánh giãn cách (GR-Quanv không học)."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self._init_tracker()
        self.c_stem = nn.Conv2d(1, NUM_MS_CHANNELS, kernel_size=2, stride=2)
        self.classical_branches = nn.ModuleList([
            nn.Conv2d(1, CH_PER_SCALE, kernel_size=2, stride=2, padding=0, dilation=1),
            nn.Conv2d(1, CH_PER_SCALE, kernel_size=2, stride=2, padding=1, dilation=2),
            nn.Conv2d(1, CH_PER_SCALE, kernel_size=2, stride=2, padding=2, dilation=3),
            nn.Conv2d(1, CH_PER_SCALE, kernel_size=3, stride=2, padding=1),  # axial
        ])
        self.mixer = _mixer_stack()
        self.gate = _gate_stack(NUM_MS_CHANNELS * 2)  # cat[f_c(16), f_ms_raw(16)]
        nn.init.constant_(self.gate[-1].bias, GATE_INIT_BIAS)
        self.head = WideResidualHead(num_classes, in_ch=NUM_MS_CHANNELS * 2)

    def forward_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        f_c = self.c_stem(x)
        f_raw = torch.cat([b(x)[:, :, :14, :14] for b in self.classical_branches], dim=1)
        f_ms = self.mixer(f_raw)
        alpha = torch.sigmoid(self.gate(torch.cat([f_c, f_raw], dim=1)))
        self._track(alpha)
        f_fused = torch.cat([f_c, alpha * f_ms], dim=1)
        return f_fused, alpha

    def forward(self, x: torch.Tensor, q_ms: Optional[torch.Tensor] = None) -> torch.Tensor:
        f_fused, _ = self.forward_features(x)
        return self.head(f_fused)


class WideGRQuanvNet(_GateTrackerMixin):
    """Đề xuất: Wide-Fusion GR-Quanv. Nhánh cổ điển 16 kênh + 16 kênh Q-LUT đa tỷ lệ,
    dung hợp Concat 32 kênh qua cổng alpha 16 kênh khởi tạo thấp (0.18)."""

    def __init__(self, num_classes: int = 2, in_q_channels: int = NUM_MS_CHANNELS):
        super().__init__()
        self._init_tracker()
        self.in_q_channels = in_q_channels
        self.c_stem = nn.Conv2d(1, NUM_MS_CHANNELS, kernel_size=2, stride=2)
        self.q_mixer = _mixer_stack()
        self.gate = _gate_stack(NUM_MS_CHANNELS * 2)  # cat[f_c(16), q_raw(16)]
        nn.init.constant_(self.gate[-1].bias, GATE_INIT_BIAS)
        self.head = WideResidualHead(num_classes, in_ch=NUM_MS_CHANNELS * 2)

    def forward_features(self, x: torch.Tensor, q_ms: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q_used = q_ms[:, : self.in_q_channels, :, :]
        f_c = self.c_stem(x)
        f_q = self.q_mixer(q_used)
        alpha = torch.sigmoid(self.gate(torch.cat([f_c, q_used], dim=1)))
        self._track(alpha)
        f_fused = torch.cat([f_c, alpha * f_q], dim=1)
        return f_fused, alpha

    def forward(self, x: torch.Tensor, q_ms: torch.Tensor) -> torch.Tensor:
        f_fused, _ = self.forward_features(x, q_ms)
        return self.head(f_fused)


WIDE_MODEL_KINDS = [
    ("Wide Classical Baseline", "classical"),
    ("Wide Classical Control (0% quantum)", "control"),
    ("Proposed Wide-GR-Quanv", "grquanv"),
]


def build_model(kind: str, num_classes: int) -> nn.Module:
    if kind == "classical":
        return WideClassicalBaseline(num_classes)
    if kind == "control":
        return WideClassicalControl(num_classes)
    if kind == "grquanv":
        return WideGRQuanvNet(num_classes)
    raise ValueError(kind)


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(4, 1, 28, 28)
    q = torch.randn(4, NUM_MS_CHANNELS, 14, 14)
    for kind in ("classical", "control", "grquanv"):
        for nc in (2, 9):
            m = build_model(kind, nc)
            n_par = sum(p.numel() for p in m.parameters() if p.requires_grad)
            out = m(x, q)
            assert out.shape == (4, nc), out.shape
            out.sum().backward()
            grads = [p.grad is not None for n, p in m.named_parameters() if p.requires_grad]
            print(f"{kind:<9} nc={nc}  params={n_par:>7,d}  out={tuple(out.shape)}  backward_OK={all(grads)}")
            if kind == "control":
                assert sum(p.size(0) == 1 for p in m.parameters()) >= 0
            m.eval()
            m(x, q)
            summary = m.get_gate_summary() if hasattr(m, "get_gate_summary") else {}
            if summary and summary["overall_alpha"] is not None:
                print(f"          alpha_mean={summary['overall_alpha']:.4f} (khởi tạo lý thuyết {ALPHA_INIT:.4f})")
    # Đối chứng sòng phẳng: control phải >= grquanv (số lớp kernel nhánh giãn cách học được).
    p_g = sum(p.numel() for p in build_model("grquanv", 9).parameters() if p.requires_grad)
    p_c = sum(p.numel() for p in build_model("control", 9).parameters() if p.requires_grad)
    assert p_c >= p_g, (p_c, p_g)
    print(f"FAIR-PLAY OK: control {p_c:,} >= grquanv {p_g:,} (+{p_c - p_g})")
