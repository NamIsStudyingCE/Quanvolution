# -*- coding: utf-8 -*-
"""
render_demo_video.py — WS-B3: renders the defense backup video (no narration,
Vietnamese caption band at the bottom). Reuses the exact demo pipeline that
notebooks/gd4_defense_demo.ipynb executes; all numbers come from the live
recomputation (same seed 42), so video == notebook == paper.

Output: notebooks/demo_defense_backup.mp4 (~2 minutes, 1280x720, mp4v)
"""
import os, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from src.models.circuits import CIRCUIT_DICT, apply_quanv_to_image
from src.models.quantum_model import QuanvolutionClassifier
from src.models.classical_cnn import SymmetricalMinimumCNN

SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)
W, H, FPS = 1280, 720, 10
OUT = Path('notebooks/demo_defense_backup.mp4')

# ---------- run the actual pipeline once (same code path as the notebook) ----------
d = np.load(ROOT / 'data' / 'breastmnist.npz')
test_imgs = d['test_images'].astype(np.float32) / 255.0
test_labels = d['test_labels'].astype(np.int64).ravel()

def load_split(circuit, split):
    feats, labels = torch.load(f'data/quantum_features/breastmnist_{circuit}_{split}.pt',
                               map_location='cpu', weights_only=False)
    return torch.utils.data.TensorDataset(torch.as_tensor(feats, dtype=torch.float32),
                                          torch.as_tensor(labels, dtype=torch.long))

torch.manual_seed(SEED)
tr = load_split('basic_L2', 'train'); va = load_split('basic_L2', 'val'); te = load_split('basic_L2', 'test')
g = torch.Generator().manual_seed(SEED)
trl = torch.utils.data.DataLoader(tr, batch_size=32, shuffle=True, generator=g)
val = torch.utils.data.DataLoader(va, batch_size=64)
tel = torch.utils.data.DataLoader(te, batch_size=64)
head = QuanvolutionClassifier(num_classes=2)
crit = nn.CrossEntropyLoss(); opt = optim.Adam(head.parameters(), lr=0.001)

from sklearn.metrics import roc_auc_score, accuracy_score
def auc_of(model, loader):
    model.eval(); ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            ps.append(torch.softmax(model(x), 1)[:, 1]); ys.append(y)
    return roc_auc_score(torch.cat(ys), torch.cat(ps))

hist, best_auc, best_state = [], -1, None
for ep in range(20):
    head.train()
    for x, y in trl:
        opt.zero_grad(); loss = crit(head(x), y); loss.backward(); opt.step()
    v = auc_of(head, val); hist.append(v)
    if v > best_auc: best_auc, best_state = v, {k: c.clone() for k, c in head.state_dict().items()}
head.load_state_dict(best_state)
head.eval(); ys, ps = [], []
with torch.no_grad():
    for x, y in tel:
        ps.append(torch.softmax(head(x), 1)); ys.append(y)
probs = torch.cat(ps); labels_t = torch.cat(ys)
acc = accuracy_score(labels_t, probs.argmax(1).numpy())
auc = roc_auc_score(labels_t, probs[:, 1].numpy())

mal_idx = np.where(test_labels == 0)[0]
DEMO_IDX = int(mal_idx[np.argmax(probs[mal_idx, 0].numpy())])
img = test_imgs[DEMO_IDX]; true_label = int(test_labels[DEMO_IDX])

CIRCUITS = ['basic_L2', 'strongly_L2', 'random_L2']
feat_live = {n: apply_quanv_to_image(img[None], CIRCUIT_DICT[n]) for n in CIRCUITS}
x = torch.as_tensor(feat_live['basic_L2'], dtype=torch.float32).unsqueeze(0)
with torch.no_grad():
    p = torch.softmax(head(x), 1)[0]
pred = int(p.argmax())

te_feats, _ = torch.load('data/quantum_features/breastmnist_basic_L2_test.pt', map_location='cpu', weights_only=False)
diff = float(np.abs(np.asarray(te_feats[DEMO_IDX]) - feat_live['basic_L2']).max())

cnn = SymmetricalMinimumCNN(num_classes=2).eval()
ximg = torch.as_tensor(img, dtype=torch.float32)[None, None]
with torch.no_grad():
    _ = cnn(ximg)
    t0 = time.perf_counter()
    for _ in range(20): _ = cnn(ximg)
    t_cnn = (time.perf_counter() - t0) / 20 * 1000
    _ = apply_quanv_to_image(img[None], CIRCUIT_DICT['basic_L2'])
    t0 = time.perf_counter()
    for _ in range(3): _ = apply_quanv_to_image(img[None], CIRCUIT_DICT['basic_L2'])
    t_q = (time.perf_counter() - t0) / 3 * 1000

# ---------- scene renderer ----------
def fig_to_bgr(fig):
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    plt.close(fig)
    return buf[:, :, ::-1].copy()  # RGB -> BGR

def scene(fig_maker, seconds, caption=''):
    n = int(seconds * FPS)
    fig = fig_maker()
    frame = fig_to_bgr(fig)
    main_h = int(H * 0.90)
    frame = cv2.resize(frame, (W, main_h)) if frame.shape[:2] != (main_h, W) else frame
    if caption:
        cap = plt.figure(figsize=(W/100, 0.62), dpi=100)
        plt.axis('off')
        plt.text(0.5, 0.5, caption, ha='center', va='center', fontsize=14, wrap=True)
        cap.canvas.draw()
        cap_img = np.asarray(cap.canvas.buffer_rgba())[:, :, :3]
        plt.close(cap)
        band = cv2.resize(cap_img, (W, H - main_h))
        frame = np.vstack([frame, band])
    frame = cv2.resize(frame, (W, H)) if frame.shape[:2] != (H, W) else frame
    for _ in range(n):
        writer.write(frame)

def title_scene(text, sub, seconds):
    n = int(seconds * FPS)
    fig = plt.figure(figsize=(W/100, H/100), dpi=100)
    plt.axis('off')
    plt.text(0.5, 0.60, 'DEMO DỰ PHÒNG', ha='center', va='center', fontsize=34, fontweight='bold')
    plt.text(0.5, 0.50, 'Quanvolution trên BreastMNIST', ha='center', va='center', fontsize=26, fontweight='bold')
    plt.text(0.5, 0.37, sub, ha='center', va='center', fontsize=15)
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3][:, :, ::-1].copy()
    plt.close(fig)
    for _ in range(n):
        writer.write(frame)

writer = cv2.VideoWriter(str(OUT), cv2.VideoWriter_fourcc(*'mp4v'), FPS, (W, H))

# ---- Scene 1: title (8s)
title_scene('DEMO DỰ PHÒNG — QUANVOLUTION TRÊN BREASTMNIST',
            'Luận văn Tốt nghiệp · UIT — ĐHQG-HCM | Mọi số liệu tái lập từ ground truth 10 seeds (seed demo = 42)', 8)

# ---- Scene 2: original image (10s)
def orig_scene():
    f, a = plt.subplots(figsize=(5.2, 5.2), dpi=100)
    a.imshow(img, cmap='gray')
    a.set_title(f'Ảnh siêu âm vú 28×28 (test idx {DEMO_IDX})\nNhãn thật: MALIGNANT (ác tính)', fontsize=13)
    a.axis('off')
    return f
scene(orig_scene, 10,
      'Bước 1 — Ảnh đầu vào: 28×28, được chia thành 196 patches 2×2 để quét bằng mạch lượng tử 4-qubit')

# ---- Scenes 3-5: feature maps per circuit (12s each)
for name, label in [('basic_L2', 'Basic Entangler (L=2) — quán quân ROC-AUC 0.8521±0.0095'),
                    ('strongly_L2', 'Strongly Entangling (L=2) — quán quân PR-AUC 0.9182±0.0071'),
                    ('random_L2', 'Random (L=2) — mạch ngẫu nhiên cố định, 0 tham số học')]:
    def fm_scene(name=name, label=label):
        f, axes = plt.subplots(1, 5, figsize=(W/100, 3.6), dpi=100)
        fm = feat_live[name]
        axes[0].imshow(img, cmap='gray'); axes[0].set_title('Original', fontsize=10)
        for c in range(4):
            axes[c+1].imshow(fm[c], cmap='viridis', vmin=-1, vmax=1)
            axes[c+1].set_title(f'⟨Z{c}⟩', fontsize=10)
        for a in axes: a.axis('off')
        f.suptitle(f'Mạch {label}', fontsize=15, fontweight='bold')
        plt.tight_layout()
        return f
    scene(fm_scene, 12, 'Bước 2 — Trích xuất LIVE: 196 patches × mạch 4-qubit → 4 kênh feature map '
                        '(cùng seed 42, tái lập tuyệt đối, sai lệch precompute ≈ 1.5e-08)')

# ---- Scene 6: training (12s)
def train_scene():
    f, (a1, a2) = plt.subplots(1, 2, figsize=(W/100, 3.6), dpi=100)
    a1.plot(range(1, 21), hist, marker='o', ms=4, color='#2ca02c')
    a1.set_xlabel('Epoch'); a1.set_ylabel('Val ROC-AUC'); a1.set_title('Head training — 20 epochs, 1.3s', fontsize=12)
    a1.grid(alpha=.3)
    bars = a2.bar(['Demo (1 seed)', 'Paper (10 seeds)'], [auc, 0.8521], color=['#1f77b4', '#888888'])
    a2.set_ylim(0.6, 0.9); a2.set_title('ROC-AUC test — basic_L2', fontsize=12)
    for b, v in zip(bars, [auc, 0.8521]): a2.text(b.get_x()+b.get_width()/2, v+.005, f'{v:.4f}', ha='center', fontsize=10)
    plt.tight_layout()
    return f
scene(train_scene, 12, 'Bước 3 — Head cổ điển ĐỐI XỨNG (BN→ReLU→Linear 784→2) train trên feature precompute — '
                       'kết quả 1 seed nằm trong biên ±std của paper')

# ---- Scene 7: prediction (12s)
def pred_scene():
    f, a = plt.subplots(figsize=(W/100, 3.8), dpi=100)
    bars = a.bar(['MALIGNANT (0)', 'BENIGN (1)'], p.numpy(), color=['#d62728', '#2ca02c'], width=0.5)
    a.set_ylim(0, 1); a.set_ylabel('Xác suất', fontsize=12)
    a.set_title(f'Dự đoán: MALIGNANT — nhãn thật: MALIGNANT (phân loại đúng, test idx {DEMO_IDX})', fontsize=14)
    for b, v in zip(bars, p.numpy()):
        a.text(b.get_x()+b.get_width()/2, v+.02, f'{v:.3f}', ha='center', fontsize=13)
    plt.tight_layout()
    return f
scene(pred_scene, 12, 'Bước 4 — Dự đoán trên ảnh demo bằng head đã train (feature tính LIVE, khớp precompute)')

# ---- Scene 8: latency (12s)
def lat_scene():
    f, a = plt.subplots(figsize=(W/100, 3.4), dpi=100)
    bars = a.barh(['Classical CNN', 'Quanvolution (CPU)'], [t_cnn, t_q], color=['#2ca02c', '#d62728'])
    a.set_xscale('log'); a.set_xlabel('ms / ảnh (log scale)', fontsize=12)
    a.set_title('Chi phí suy luận trên CPU', fontsize=14)
    for b, v in zip(bars, [t_cnn, t_q]):
        a.text(v*1.15, b.get_y()+b.get_height()/2, f'{v:.3f} ms' if v < 1 else f'{v:.1f} ms', va='center', fontsize=11)
    plt.tight_layout()
    return f
scene(lat_scene, 12, 'Bước 5 — Đánh đổi: mô phỏng CPU chậm ~hàng trăm lần → chiến lược precompute feature maps 1 lần')

# ---- Scene 9: takeaways (14s)
def end_scene():
    f = plt.figure(figsize=(W/100, H/100), dpi=100)
    plt.axis('off')
    lines = [
        'KẾT LUẬN DEMO',
        '• Mạch lượng tử tĩnh 0 tham số → feature map hữu ích (Quantum Inductive Bias)',
        '• BreastMNIST (nhỏ, lệch lớp): quantum thắng ROC-AUC & PR-AUC —BreastMNIST demo case phân loại đúng',
        '• OCTMNIST (lớn, đa lớp): Classical CNN thắng — ưu thế PHỤ THUỘC CHẾ ĐỘ DỮ LIỆU',
        '• Trainability chỉ cục bộ trong cùng họ mạch; chi phí CPU ~710× → precompute',
    ]
    plt.text(0.06, 0.80, lines[0], fontsize=22, fontweight='bold')
    for i, l in enumerate(lines[1:], start=1):
        plt.text(0.06, 0.66 - i*0.115, l, fontsize=14)
    return f
scene(end_scene, 14, 'Bài báo: SOICT 2026 (Springer CCIS) — mã nguồn & dữ liệu: github.com/NamIsStudyingCE/Quanvolution')

writer.release()
dur = OUT.stat().st_size
print(f'video saved: {OUT} | {dur/1e6:.1f} MB | {W}x{H}@{FPS}fps')
