"""
Wide-Fusion GR-Quanv v2 (Đòn bẩy 3 Mũi Nhọn)
============================================
Nâng cấp từ chẩn đoán thực nghiệm của `gr_quanv_wide.py` (v1):
  (1) Đòn bẩy 1 - Khắc phục "Đói phương sai & Van lượng tử bị bóp nghẹt 18%":
      - Đổi `GATE_INIT_BIAS = 0.0` (alpha_0 = 0.5 thay vì 0.182).
      - Bổ sung Pre-Concat BatchNorm2d (`norm_c`, `norm_q` / `norm_ms`) cho cả hai
        nhánh trước khi qua cổng và ghép kênh để bảo đảm Var(f_c) == Var(f_q) ~ 1.0.
  (2) Đòn bẩy 2 - Khắc phục "Cắt cứng 4-bit xóa sổ vi tương phản 6.67%":
      - Kết hợp với đặc trưng nội suy đa tuyến tính 4D SIMD (`rust_qlut.forward_interp4d_nchw`,
        Cosine 0.999993) giữ trọn vi tương phản dưới ngưỡng 6.67%.
  (3) Đòn bẩy 3 - Khắc phục "Mạng học đường tắt" bằng Classical Branch DropPath (p=0.15):
      - Khi `self.training == True`, với xác suất p = 0.15 trên từng mẫu trong batch,
        nhân nhánh cổ điển `f_c` với hệ số 0.5 để buộc mạng khai thác đặc trưng đa tỷ lệ.
      - Áp dụng đồng nhất trên cả `WideClassicalControl` và `WideGRQuanvNet` để giữ
        tính đối chứng sòng phẳng tuyệt đối (params(Control) >= params(GR-Quanv)).
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from models.gr_quanv_resblock import _ResBlock


# Đòn bẩy 1: Bias khởi tạo của tầng Conv1x1 cuối cổng: sigmoid(0.0) = 0.5000.
GATE_INIT_BIAS = 0.0
ALPHA_INIT = float(torch.sigmoid(torch.tensor(GATE_INIT_BIAS)).item())
DEFAULT_DROP_C_PROB = 0.15

NUM_Q_SCALES = 4          # d=1, d=2, d=3, axial
CH_PER_SCALE = 4          # 4 kênh đếm chẵn CNOT mỗi tỷ lệ
NUM_MS_CHANNELS = NUM_Q_SCALES * CH_PER_SCALE   # 16 kênh đa tỷ lệ


def _mixer_stack() -> nn.Sequential:
    """Bộ trộn đặc trưng đa tỷ lệ: giữ nguyên 16 kênh vào/ra, thông lượng nội bộ 32.

    Dùng chung cho cả nhánh lượng tử (đầu vào 16 kênh Q-LUT cố định) và nhánh đối chứng
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


def _apply_branch_droppath(f_c: torch.Tensor, drop_prob: float, training: bool) -> torch.Tensor:
    """Đòn bẩy 3: Với xác suất `drop_prob` trên từng mẫu trong batch khi train,
    nhân nhánh cổ điển `f_c` với 0.5 để chống học đường tắt (Shortcut Learning)."""
    if not training or drop_prob <= 0.0:
        return f_c
    B = f_c.shape[0]
    mask = torch.rand(B, 1, 1, 1, device=f_c.device, dtype=f_c.dtype) < drop_prob
    scale = torch.where(mask, torch.full_like(mask, 0.5, dtype=f_c.dtype), torch.ones_like(mask, dtype=f_c.dtype))
    return f_c * scale


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
    cổ điển sinh đúng 16 kênh thô; mixer/norm/gate/fusion/head giống hệt `WideGRQuanvNet`.
    Nhiều hơn GR-Quanv đúng số tham số của 4 kernel nhánh giãn cách (GR-Quanv không học)."""

    def __init__(self, num_classes: int = 2, drop_c_prob: float = DEFAULT_DROP_C_PROB):
        super().__init__()
        self._init_tracker()
        self.drop_c_prob = drop_c_prob
        self.c_stem = nn.Conv2d(1, NUM_MS_CHANNELS, kernel_size=2, stride=2)
        self.classical_branches = nn.ModuleList([
            nn.Conv2d(1, CH_PER_SCALE, kernel_size=2, stride=2, padding=0, dilation=1),
            nn.Conv2d(1, CH_PER_SCALE, kernel_size=2, stride=2, padding=1, dilation=2),
            nn.Conv2d(1, CH_PER_SCALE, kernel_size=2, stride=2, padding=2, dilation=3),
            nn.Conv2d(1, CH_PER_SCALE, kernel_size=3, stride=2, padding=1),  # axial
        ])
        self.mixer = _mixer_stack()
        # Pre-Concat Normalization cân bằng phương sai Var(f_c) == Var(f_ms) ~ 1.0
        self.norm_c = nn.BatchNorm2d(NUM_MS_CHANNELS)
        self.norm_ms = nn.BatchNorm2d(NUM_MS_CHANNELS)
        self.gate = _gate_stack(NUM_MS_CHANNELS * 2)  # cat[f_c_norm(16), f_ms_norm(16)]
        nn.init.constant_(self.gate[-1].bias, GATE_INIT_BIAS)
        self.head = WideResidualHead(num_classes, in_ch=NUM_MS_CHANNELS * 2)

    def forward_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        f_c = self.norm_c(self.c_stem(x))
        f_raw = torch.cat([b(x)[:, :, :14, :14] for b in self.classical_branches], dim=1)
        f_ms = self.norm_ms(self.mixer(f_raw))
        alpha = torch.sigmoid(self.gate(torch.cat([f_c, f_ms], dim=1)))
        self._track(alpha)
        f_c_mod = _apply_branch_droppath(f_c, self.drop_c_prob, self.training)
        f_fused = torch.cat([f_c_mod, alpha * f_ms], dim=1)
        return f_fused, alpha

    def forward(self, x: torch.Tensor, q_ms: Optional[torch.Tensor] = None) -> torch.Tensor:
        f_fused, _ = self.forward_features(x)
        return self.head(f_fused)


class WideGRQuanvNet(_GateTrackerMixin):
    """Đề xuất: Wide-Fusion GR-Quanv v2.
    - Nhánh cổ điển 16 kênh + 16 kênh Q-LUT đa tỷ lệ (Interp4D SIMD).
    - Pre-Concat BatchNorm2d bảo đảm Var(f_c) == Var(f_q) ~ 1.0.
    - Cổng alpha khởi tạo cân bằng tại 0.50 (GATE_INIT_BIAS = 0.0).
    - Classical Branch DropPath (p=0.15) chống học đường tắt trên ảnh sạch."""

    def __init__(
        self,
        num_classes: int = 2,
        in_q_channels: int = NUM_MS_CHANNELS,
        drop_c_prob: float = DEFAULT_DROP_C_PROB,
    ):
        super().__init__()
        self._init_tracker()
        self.in_q_channels = in_q_channels
        self.drop_c_prob = drop_c_prob
        self.c_stem = nn.Conv2d(1, NUM_MS_CHANNELS, kernel_size=2, stride=2)
        self.q_mixer = _mixer_stack()
        # Pre-Concat Normalization cân bằng phương sai Var(f_c) == Var(f_q) ~ 1.0
        self.norm_c = nn.BatchNorm2d(NUM_MS_CHANNELS)
        self.norm_q = nn.BatchNorm2d(NUM_MS_CHANNELS)
        self.gate = _gate_stack(NUM_MS_CHANNELS * 2)  # cat[f_c_norm(16), f_q_norm(16)]
        nn.init.constant_(self.gate[-1].bias, GATE_INIT_BIAS)
        self.head = WideResidualHead(num_classes, in_ch=NUM_MS_CHANNELS * 2)

    def forward_features(self, x: torch.Tensor, q_ms: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q_used = q_ms[:, : self.in_q_channels, :, :]
        f_c = self.norm_c(self.c_stem(x))
        f_q = self.norm_q(self.q_mixer(q_used))
        alpha = torch.sigmoid(self.gate(torch.cat([f_c, f_q], dim=1)))
        self._track(alpha)
        f_c_mod = _apply_branch_droppath(f_c, self.drop_c_prob, self.training)
        f_fused = torch.cat([f_c_mod, alpha * f_q], dim=1)
        return f_fused, alpha

    def forward(self, x: torch.Tensor, q_ms: torch.Tensor) -> torch.Tensor:
        f_fused, _ = self.forward_features(x, q_ms)
        return self.head(f_fused)


WIDE_MODEL_KINDS = [
    ("Wide Classical Baseline", "classical"),
    ("Wide Classical Control (0% quantum)", "control"),
    ("Proposed Wide-GR-Quanv", "grquanv"),
]


def build_model(kind: str, num_classes: int, drop_c_prob: float = DEFAULT_DROP_C_PROB) -> nn.Module:
    if kind == "classical":
        return WideClassicalBaseline(num_classes)
    if kind == "control":
        return WideClassicalControl(num_classes, drop_c_prob=drop_c_prob)
    if kind == "grquanv":
        return WideGRQuanvNet(num_classes, drop_c_prob=drop_c_prob)
    raise ValueError(kind)


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(4, 1, 28, 28)
    q = torch.randn(4, NUM_MS_CHANNELS, 14, 14)
    for kind in ("classical", "control", "grquanv"):
        for nc in (2, 4):
            m = build_model(kind, nc)
            m.train()
            n_par = sum(p.numel() for p in m.parameters() if p.requires_grad)
            out = m(x, q)
            assert out.shape == (4, nc), out.shape
            out.sum().backward()
            grads = [p.grad is not None for n, p in m.named_parameters() if p.requires_grad]
            print(f"{kind:<9} nc={nc}  params={n_par:>7,d}  out={tuple(out.shape)}  backward_OK={all(grads)}")
            m.eval()
            m(x, q)
            summary = m.get_gate_summary() if hasattr(m, "get_gate_summary") else {}
            if summary and summary["overall_alpha"] is not None:
                print(f"          alpha_mean={summary['overall_alpha']:.4f} (theoretical_init={ALPHA_INIT:.4f})")
    p_g = sum(p.numel() for p in build_model("grquanv", 4).parameters() if p.requires_grad)
    p_c = sum(p.numel() for p in build_model("control", 4).parameters() if p.requires_grad)
    assert p_c >= p_g, (p_c, p_g)
    print(f"FAIR-PLAY OK: control {p_c:,} >= grquanv {p_g:,} (+{p_c - p_g})")
