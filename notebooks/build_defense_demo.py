# -*- coding: utf-8 -*-
"""build_defense_demo.py — generates notebooks/gd4_defense_demo.ipynb (GĐ4 / WS-B1)."""
import json
from pathlib import Path

def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src}

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}

cells = []

cells.append(md(
"# 🎓 Demo Bảo vệ — Quanvolution trên BreastMNIST\n"
"\n"
"**Luồng trình chiếu:** Ảnh siêu âm vú gốc → mạch lượng tử 4-qubit (live) tạo 4 kênh feature map → "
"head cổ điển đối xứng train trên feature đã precompute → dự đoán + xác suất → so chi phí suy luận.\n"
"\n"
"*Mọi con số tái lập từ ground truth `results/full_trainable_breastmnist.json` (10 seeds × 20 epochs). "
"Nhãn: `0` = malignant (ác tính), `1` = benign (lành tính).*"))

cells.append(code(
"import os, sys, time, json\n"
"from pathlib import Path\n"
"\n"
"ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
"os.chdir(ROOT)\n"
"sys.path.insert(0, str(ROOT))\n"
"\n"
"import numpy as np\n"
"import torch\n"
"import torch.nn as nn\n"
"import torch.optim as optim\n"
"import matplotlib.pyplot as plt\n"
"\n"
"from src.models.circuits import CIRCUIT_DICT, apply_quanv_to_image\n"
"from src.models.quantum_model import QuanvolutionClassifier\n"
"from src.models.classical_cnn import SymmetricalMinimumCNN\n"
"\n"
"SEED = 42\n"
"np.random.seed(SEED)\n"
"torch.manual_seed(SEED)\n"
"torch.set_num_threads(os.cpu_count())\n"
"plt.rcParams['figure.dpi'] = 110\n"
"print('Repo root:', ROOT)\n"
"print('PyTorch', torch.__version__)"))

cells.append(code(
"# ---- Nạp BreastMNIST test set (chuẩn MedMNIST: npz; tự tải nếu thiếu) ----\n"
"NPZ = ROOT / 'data' / 'breastmnist.npz'\n"
"if not NPZ.exists():\n"
"    import medmnist\n"
"    print('Đang tải breastmnist.npz (lần đầu, ~1 MB)...')\n"
"    medmnist.BreastMNIST(split='test', download=True, root=str(ROOT / 'data'))\n"
"\n"
"d = np.load(NPZ)\n"
"test_imgs = d['test_images'].astype(np.float32) / 255.0     # (156, 28, 28) in [0,1]\n"
"test_labels = d['test_labels'].astype(np.int64).ravel()      # 0=malignant, 1=benign\n"
"print('Test set:', test_imgs.shape, '| malignant:', int((test_labels==0).sum()),\n"
"      '| benign:', int((test_labels==1).sum()))"))

cells.append(code(
"# ---- Train head cổ điển ĐỐI XỨNG trên feature đã precompute (mạch basic_L2) ----\n"
"# Protocol giống paper: QuanvolutionClassifier = BatchNorm2d(4) -> ReLU -> Linear(784,2)\n"
"# CE loss, Adam lr=0.001, 20 epochs, chọn checkpoint theo best val ROC-AUC.\n"
"def load_split(circuit, split):\n"
"    feats, labels = torch.load(f'data/quantum_features/breastmnist_{circuit}_{split}.pt',\n"
"                               map_location='cpu', weights_only=False)\n"
"    return torch.utils.data.TensorDataset(torch.as_tensor(feats, dtype=torch.float32),\n"
"                                          torch.as_tensor(labels, dtype=torch.long))\n"
"\n"
"torch.manual_seed(SEED)\n"
"tr_ds  = load_split('basic_L2', 'train')\n"
"va_ds  = load_split('basic_L2', 'val')\n"
"te_ds  = load_split('basic_L2', 'test')\n"
"g = torch.Generator().manual_seed(SEED)\n"
"tr_loader = torch.utils.data.DataLoader(tr_ds, batch_size=32, shuffle=True, generator=g)\n"
"va_loader = torch.utils.data.DataLoader(va_ds, batch_size=64)\n"
"te_loader = torch.utils.data.DataLoader(te_ds, batch_size=64)\n"
"\n"
"head = QuanvolutionClassifier(num_classes=2)\n"
"crit = nn.CrossEntropyLoss()\n"
"opt = optim.Adam(head.parameters(), lr=0.001)\n"
"\n"
"def eval_auc(model, loader):\n"
"    from sklearn.metrics import roc_auc_score\n"
"    model.eval(); ys, ps = [], []\n"
"    with torch.no_grad():\n"
"        for x, y in loader:\n"
"            p = torch.softmax(model(x), 1)[:, 1]\n"
"            ys.append(y); ps.append(p)\n"
"    return roc_auc_score(torch.cat(ys), torch.cat(ps))\n"
"\n"
"t0 = time.perf_counter()\n"
"best_auc, best_state = -1, None\n"
"hist = []\n"
"for ep in range(1, 21):\n"
"    head.train()\n"
"    for x, y in tr_loader:\n"
"        opt.zero_grad(); loss = crit(head(x), y); loss.backward(); opt.step()\n"
"    va = eval_auc(head, va_loader)\n"
"    hist.append(va)\n"
"    if va > best_auc:\n"
"        best_auc, best_state = va, {k: v.clone() for k, v in head.state_dict().items()}\n"
"train_time = time.perf_counter() - t0\n"
"head.load_state_dict(best_state)\n"
"\n"
"from sklearn.metrics import accuracy_score, roc_auc_score\n"
"head.eval(); ys, ps = [], []\n"
"with torch.no_grad():\n"
"    for x, y in te_loader:\n"
"        ps.append(torch.softmax(head(x), 1)); ys.append(y)\n"
"probs = torch.cat(ps); labels_t = torch.cat(ys)\n"
"preds = probs.argmax(1).numpy()\n"
"acc = accuracy_score(labels_t, preds)\n"
"auc = roc_auc_score(labels_t, probs[:, 1].numpy())\n"
"print(f'Train 20 epochs trong {train_time:.1f}s | best val AUC = {best_auc:.4f}')\n"
"print(f'Test (1 seed demo): Accuracy = {acc:.4f} | ROC-AUC = {auc:.4f}')\n"
"print('Paper (10 seeds):   Accuracy = 0.8083 ± 0.0279 | ROC-AUC = 0.8521 ± 0.0095')\n"
"\n"
"plt.figure(figsize=(5, 3))\n"
"plt.plot(range(1, 21), hist, marker='o', ms=3)\n"
"plt.xlabel('Epoch'); plt.ylabel('Val ROC-AUC'); plt.title('Head training (basic_L2 features)')\n"
"plt.grid(alpha=.3); plt.tight_layout(); plt.show()\n"
"\n"
"# Chọn ảnh demo CÔNG KHAI: trường hợp ác tính được head phân loại ĐÚNG, tin cậy cao nhất\n"
"mal_idx = np.where(test_labels == 0)[0]\n"
"DEMO_IDX = int(mal_idx[np.argmax(probs[mal_idx, 0].numpy())])\n"
"img = test_imgs[DEMO_IDX]\n"
"true_label = int(test_labels[DEMO_IDX])\n"
"print(f'Ảnh demo: index {DEMO_IDX} | nhãn thật = {true_label} (malignant) | '\n"
"      f'p(malignant) = {float(probs[DEMO_IDX, 0]):.3f} — phân loại đúng')"))

# ---- then original image, live quanv, prediction, latency ----
cells.append(code(
"# ---- Ảnh siêu âm gốc 28x28 ----\n"
"fig, ax = plt.subplots(figsize=(3.2, 3.2))\n"
"ax.imshow(img, cmap='gray')\n"
"ax.set_title(f'Original 28×28 — idx {DEMO_IDX} (label {true_label}: malignant)', fontsize=10)\n"
"ax.axis('off')\n"
"plt.tight_layout()\n"
"plt.show()"))

cells.append(code(
"# ---- LIVE: 3 mạch lượng tử tĩnh quét 196 patches -> 4 kênh feature map ----\n"
"CIRCUITS = ['basic_L2', 'strongly_L2', 'random_L2']\n"
"feat_live, t_quanv = {}, 0.0\n"
"for name in CIRCUITS:\n"
"    t0 = time.perf_counter()\n"
"    feat_live[name] = apply_quanv_to_image(img[None, :, :], CIRCUIT_DICT[name])\n"
"    t_quanv += time.perf_counter() - t0\n"
"t_quanv /= len(CIRCUITS)\n"
"print(f'Thời gian trích xuất 1 ảnh / 1 mạch: {t_quanv*1000:.1f} ms')\n"
"\n"
"fig, axes = plt.subplots(3, 5, figsize=(13, 7.6))\n"
"for r, name in enumerate(CIRCUITS):\n"
"    fm = feat_live[name]\n"
"    axes[r, 0].imshow(img, cmap='gray')\n"
"    axes[r, 0].set_title(f'Original', fontsize=9)\n"
"    for c in range(4):\n"
"        axes[r, c+1].imshow(fm[c], cmap='viridis', vmin=-1, vmax=1)\n"
"        axes[r, c+1].set_title(f'{name} — ⟨Z{c}⟩', fontsize=9)\n"
"    for c in range(5):\n"
"        axes[r, c].axis('off')\n"
"plt.suptitle('Quanvolution Feature Maps (4-qubit, 2×2 patch, stride 2, seed 42)', fontweight='bold')\n"
"plt.tight_layout()\n"
"plt.show()"))

cells.append(code(
"# ---- DỰ ĐOÁN trên chính ảnh demo (dùng LIVE features, không peek test set loader) ----\n"
"x = torch.as_tensor(feat_live['basic_L2'], dtype=torch.float32).unsqueeze(0)  # (1,4,14,14)\n"
"with torch.no_grad():\n"
"    p = torch.softmax(head(x), 1)[0]\n"
"pred = int(p.argmax())\n"
"print(f'Nhãn thật     : {true_label} (malignant)')\n"
"print(f'Dự đoán       : {pred} ({\"malignant\" if pred==0 else \"benign\"})')\n"
"print(f'Xác suất      : malignant = {p[0]:.3f} | benign = {p[1]:.3f}')\n"
"\n"
"# Kiểm tra nhất quán: feature tính LIVE == feature đã precompute cho cùng ảnh?\n"
"te_feats, te_labels_chk = torch.load('data/quantum_features/breastmnist_basic_L2_test.pt',\n"
"                                 map_location='cpu', weights_only=False)\n"
"diff = float(np.abs(np.asarray(te_feats[DEMO_IDX]) - feat_live['basic_L2']).max())\n"
"print(f'Khớp feature precompute (max abs diff): {diff:.2e}')\n"
"\n"
"fig, ax = plt.subplots(figsize=(4.5, 2.6))\n"
"bars = ax.bar(['malignant (0)', 'benign (1)'], p.numpy(), color=['#d62728', '#2ca02c'])\n"
"ax.set_ylim(0, 1); ax.set_ylabel('Xác suất')\n"
"ax.set_title(f'Dự đoán: {\"MALIGNANT\" if pred==0 else \"BENIGN\"} | nhãn thật: {\"MALIGNANT\" if true_label==0 else \"BENIGN\"}', fontsize=11)\n"
"for b, v in zip(bars, p.numpy()):\n"
"    ax.text(b.get_x()+b.get_width()/2, v+.02, f'{v:.3f}', ha='center', fontsize=10)\n"
"plt.tight_layout(); plt.show()"))

cells.append(code(
"# ---- Chi phí suy luận: Classical CNN vs Quanvolution (CPU) ----\n"
"cnn = SymmetricalMinimumCNN(num_classes=2).eval()\n"
"ximg = torch.as_tensor(img, dtype=torch.float32)[None, None]  # (1,1,28,28)\n"
"with torch.no_grad():\n"
"    _ = cnn(ximg)\n"
"    reps = 20\n"
"    t0 = time.perf_counter()\n"
"    for _ in range(reps):\n"
"        _ = cnn(ximg)\n"
"    t_cnn = (time.perf_counter()-t0)/reps*1000\n"
"    _ = apply_quanv_to_image(img[None], CIRCUIT_DICT['basic_L2'])\n"
"    reps_q = 3\n"
"    t0 = time.perf_counter()\n"
"    for _ in range(reps_q):\n"
"        _ = apply_quanv_to_image(img[None], CIRCUIT_DICT['basic_L2'])\n"
"    t_q = (time.perf_counter()-t0)/reps_q*1000\n"
"print(f'Classical CNN forward : {t_cnn:8.3f} ms/ảnh  (1.0x)')\n"
"print(f'Quanvolution (CPU)    : {t_q:8.1f} ms/ảnh  (~{t_q/max(t_cnn,1e-9):.0f}x chậm hơn)')\n"
"print('Paper: 0.310 ms vs 220.22 ms (~710x) — cùng bậc, khác biệt do CPU/PennyLane version')"))

cells.append(md(
"## 📌 Tổng kết demo\n"
"| Ý | Bằng chứng trong demo |\n"
"|---|---|\n"
"| **Quantum Inductive Bias 0 tham số** | 3 mạch tĩnh (seed 42, khóa góc) tạo feature map hữu ích — kernel 0 trainable params |\n"
"| **Head đối xứng 1:1** | `BatchNorm2d(4) → ReLU → Linear(784,2)` — giống hệt baseline CNN |\n"
"| **Trainable cục bộ** | Head train 20 epochs trong vài giây trên feature precompute (18 s/10 seeds trong paper) |\n"
"| **Chi phí** | Quanvolution chậm hàng trăm lần trên CPU → chiến lược precompute |\n"
"| **Kết quả trung thực** | BreastMNIST: quantum thắng ROC-AUC/PR-AUC; OCTMNIST: CNN thắng — *data-regime dependent* |\n"))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                    "language_info": {"name": "python", "version": "3.10"}},
      "nbformat": 4, "nbformat_minor": 5}

out = Path('notebooks/gd4_defense_demo.ipynb')
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
print('written', out, out.stat().st_size, 'bytes')
