# -*- coding: utf-8 -*-
"""
measure_params_cost.py
----------------------
Tự động đo SỐ THAM SỐ và CHI PHÍ cho bảng so sánh Quantum vs Classical trong bài báo.

Cách dùng:
    Đặt file này ở THƯ MỤC GỐC của repo (cùng cấp với run_gd3.py), rồi chạy:
        python measure_params_cost.py
"""

import sys, os, time, platform
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

try:
    import torch
    import torch.nn as nn
except Exception as e:
    print("[LỖI] Cần cài PyTorch trước:  pip install torch")
    raise

import numpy as np

def count_params(module):
    """Tổng số tham số CÓ THỂ HUẤN LUYỆN của một nn.Module."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

USED_REAL = True
try:
    from models.classical_cnn import SymmetricalMinimumCNN
    from models.quantum_model import QuanvolutionClassifier
except Exception as e:
    USED_REAL = False
    print(f"[CẢNH BÁO] Không import được model từ src/ ({e}). Dùng bản sao inline khớp kiến trúc.\n")

    class SymmetricalMinimumCNN(nn.Module):
        def __init__(self, num_classes=2):
            super().__init__()
            self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)
            self.bn = nn.BatchNorm2d(4); self.relu = nn.ReLU()
            self.flatten = nn.Flatten(); self.fc = nn.Linear(784, num_classes)
        def forward(self, x):
            return self.fc(self.flatten(self.relu(self.bn(self.conv(x)))))

    class QuanvolutionClassifier(nn.Module):
        def __init__(self, num_classes=2):
            super().__init__()
            self.bn = nn.BatchNorm2d(4); self.relu = nn.ReLU()
            self.flatten = nn.Flatten(); self.fc = nn.Linear(784, num_classes)
        def forward(self, x):
            return self.fc(self.flatten(self.relu(self.bn(x))))

N_QUBITS = 4
def quantum_kernel_params(ansatz, n_layers, trainable):
    if not trainable:
        return 0
    if ansatz == "strongly":
        return n_layers * N_QUBITS * 3
    if ansatz == "basic":
        return n_layers * N_QUBITS
    return 0

def head_split(num_classes):
    m = QuanvolutionClassifier(num_classes=num_classes)
    bn = count_params(m.bn)
    fc = count_params(m.fc)
    return bn, fc

def classical_split(num_classes):
    m = SymmetricalMinimumCNN(num_classes=num_classes)
    conv = count_params(m.conv)
    bn = count_params(m.bn)
    fc = count_params(m.fc)
    return conv, bn, fc, count_params(m)

CONFIGS = [
    ("BreastMNIST", 2, 2),
    ("OCTMNIST",    4, 1),
]

print("="*78)
print(f" ĐO THAM SỐ  |  real_modules={USED_REAL}  |  {platform.python_version()}  |  torch {torch.__version__}")
print("="*78)

rows = []
for ds, ncls, nl in CONFIGS:
    conv_c, bn_c, fc_c, tot_c = classical_split(ncls)
    bn_q, fc_q = head_split(ncls)
    head_q = bn_q + fc_q
    rows.append((ds, "Classical CNN", conv_c + bn_c, fc_c, tot_c))
    rows.append((ds, "Fixed Quanv (bất kỳ)", 0 + bn_q, fc_q, 0 + head_q))
    kb = quantum_kernel_params("basic", nl, True)
    rows.append((ds, f"Trainable Basic (L={nl})", kb + bn_q, fc_q, kb + head_q))
    ksg = quantum_kernel_params("strongly", nl, True)
    rows.append((ds, f"Trainable Strongly 3-axis (L={nl})", ksg + bn_q, fc_q, ksg + head_q))

print("\n### BẢNG THAM SỐ (tự động)\n")
print("| Dataset | Mô hình | Feature-extractor params | Classifier-head params | TỔNG (trainable) |")
print("|---|---|---:|---:|---:|")
for ds, name, fe, hd, tot in rows:
    print(f"| {ds} | {name} | {fe} | {hd} | {tot} |")

print("\n" + "="*78)
print(" ĐO ĐỘ TRỄ SUY LUẬN (CPU) — trung bình trên nhiều ảnh")
print("="*78)

N_IMG = 20
dummy_feat = torch.rand(N_IMG, 4, 14, 14)
dummy_img  = torch.rand(N_IMG, 1, 28, 28)

cnn = SymmetricalMinimumCNN(num_classes=2).eval()
with torch.no_grad():
    _ = cnn(dummy_img)
    t0 = time.perf_counter()
    for i in range(N_IMG):
        _ = cnn(dummy_img[i:i+1])
    t_cnn = (time.perf_counter() - t0) / N_IMG * 1000
print(f"Classical CNN (full forward)          : {t_cnn:.3f} ms/ảnh")

t_quanv = None
try:
    from models.circuits import apply_quanv_to_image, CIRCUIT_DICT
    circ = CIRCUIT_DICT["random_L1"]
    img_np = np.random.rand(1, 28, 28)
    _ = apply_quanv_to_image(img_np, circ)
    t0 = time.perf_counter()
    REPS = 5
    for _ in range(REPS):
        _ = apply_quanv_to_image(img_np, circ)
    t_quanv = (time.perf_counter() - t0) / REPS * 1000
    print(f"Quanvolution feature-extract (1 ảnh)  : {t_quanv:.3f} ms/ảnh  (196 patch x mạch 4-qubit)")
except Exception as e:
    print(f"[BỎ QUA] Không đo được quanvolution latency (thiếu PennyLane?): {e}")

head = QuanvolutionClassifier(num_classes=2).eval()
with torch.no_grad():
    _ = head(dummy_feat)
    t0 = time.perf_counter()
    for i in range(N_IMG):
        _ = head(dummy_feat[i:i+1])
    t_head = (time.perf_counter() - t0) / N_IMG * 1000
print(f"Quanvolution head-only (sau feature)  : {t_head:.3f} ms/ảnh")

if t_quanv is not None:
    print(f"\n=> Suy luận quanvolution end-to-end ~ {t_quanv + t_head:.3f} ms/ảnh "
          f"(gấp ~{(t_quanv + t_head)/max(t_cnn,1e-6):.0f}x classical).")
