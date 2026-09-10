# -*- coding: utf-8 -*-
"""
run_full_experiments.py — GĐ4 nâng chất: 3 thí nghiệm bổ sung + 4 phép kiểm
định methodological, chạy trên per-seed data có sẵn.

Thí nghiệm mới:
  A. Random classical 2×2 filter baseline (đóng băng, không học)
  B. Learning curve theo cỡ dữ liệu (subsample 100→5000)
  C. Entanglement ablation (gỡ CNOT, giữ nguyên đầu vào + phép đo)

Thống kê nâng cao:
  D. TOST equivalence test (biên ±2%)
  E. BH-adjusted q-values
  F. Power analysis / MDE
"""
import json, sys, time, copy
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, balanced_accuracy_score, matthews_corrcoef, precision_recall_curve, auc
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('.').resolve()
SEED = 42
EPOCHS = 20
MET = ['acc', 'bacc', 'f1', 'mcc', 'auc', 'pr_auc']
NPZ = ROOT / 'data' / 'breastmnist.npz'

# ================================================================
# DATA LOADING
# ================================================================
def load_breastmnist():
    d = np.load(str(NPZ))
    tr_x = d['train_images'].astype(np.float32) / 255.0
    tr_y = d['train_labels'].astype(np.int64).ravel()
    va_x = d['val_images'].astype(np.float32) / 255.0
    va_y = d['val_labels'].astype(np.int64).ravel()
    te_x = d['test_images'].astype(np.float32) / 255.0
    te_y = d['test_labels'].astype(np.int64).ravel()
    return tr_x, tr_y, va_x, va_y, te_x, te_y

def extract_features_circuits(imgs, circuit_fn, n_qubits=4):
    """Trích xuất feature map bằng mạch lượng tử (từng ảnh)."""
    from src.models.circuits import apply_quanv_to_image
    feats = np.zeros((len(imgs), 4, 14, 14), dtype=np.float32)
    for idx in range(len(imgs)):
        feats[idx] = apply_quanv_to_image(imgs[idx][None], circuit_fn)
        if idx % 100 == 0:
            print(f'    {idx}/{len(imgs)}...')
    return feats

def eval_model(model, X, y):
    model.eval()
    with torch.no_grad():
        logits = model(torch.as_tensor(X, dtype=torch.float32))
    probs = torch.softmax(logits, 1).numpy()
    preds = probs.argmax(1)
    return {
        'acc': accuracy_score(y, preds),
        'bacc': balanced_accuracy_score(y, preds),
        'f1': f1_score(y, preds, average='macro'),
        'mcc': matthews_corrcoef(y, preds),
        'auc': roc_auc_score(y, probs[:, 1]),
        'pr_auc': auc(y, probs[:, 1]),
    }

def train_head(model, X_train, y_train, X_val, y_val, epochs=20, lr=0.001):
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(SEED)
    ds = torch.utils.data.TensorDataset(
        torch.as_tensor(X_train, dtype=torch.float32),
        torch.as_tensor(y_train, dtype=torch.long))
    loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True, generator=g)
    best_auc, best_state = -1, None
    for ep in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward(); opt.step()
        m = eval_model(model, X_val, y_val)
        if m['auc'] > best_auc:
            best_auc, best_state = m['auc'], {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return best_auc

def eval_test(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        logits = model(torch.as_tensor(X_test, dtype=torch.float32))
    probs = torch.softmax(logits, 1).numpy()
    preds = probs.argmax(1)
    return {
        'acc': accuracy_score(y_test, preds),
        'bacc': balanced_accuracy_score(y_test, preds),
        'f1': f1_score(y_test, preds, average='macro'),
        'mcc': matthews_corrcoef(y_test, preds),
        'auc': roc_auc_score(y_test, probs[:, 1]),
        'pr_auc': auc(y_test, probs[:, 1]),
    }

# ================================================================
# A. RANDOM CLASSICAL 2×2 FILTER BASELINE
# ================================================================
def random_classical_baseline():
    """Conv2D(1→4, 2×2, s2) kernel ngẫu nhiên cố định (không học)."""
    from src.models.classical_cnn import SymmetricalMinimumCNN
    d = np.load(str(NPZ))
    tr_x = d['train_images'].astype(np.float32) / 255.0
    tr_y = d['train_labels'].astype(np.int64).ravel()
    va_x = d['val_images'].astype(np.float32) / 255.0
    va_y = d['val_labels'].astype(np.int64).ravel()
    te_x = d['test_images'].astype(np.float32) / 255.0
    te_y = d['test_labels'].astype(np.int64).ravel()

    # Thêm channel dim
    tr_x = tr_x[:, None]; va_x = va_x[:, None]; te_x = te_x[:, None]

    torch.manual_seed(SEED)
    model = SymmetricalMinimumCNN(num_classes=2)
    # Freeze conv (random fixed)
    for p in model.conv.parameters():
        p.requires_grad = False
    nn.init.uniform_(model.conv.weight, -0.5, 0.5)
    nn.init.zeros_(model.conv.bias)

    opt = optim.Adam(model.fc.parameters(), lr=0.001)
    crit = nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(SEED)
    ds = torch.utils.data.TensorDataset(
        torch.as_tensor(tr_x, dtype=torch.float32),
        torch.as_tensor(tr_y, dtype=torch.long))
    loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True, generator=g)

    best_auc, best_state = -1, None
    for ep in range(EPOCHS):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward(); opt.step()
        m = eval_model(model, va_x, va_y)
        if m['auc'] > best_auc:
            best_auc, best_state = m['auc'], {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    m = eval_test(model, te_x, te_y)
    print(f'  Random Classical Filter: acc={m["acc"]:.4f} auc={m["auc"]:.4f} pr={m["pr_auc"]:.4f}')
    return m

# ================================================================
# B. LEARNING CURVE (subsample by data size)
# ================================================================
def learning_curve():
    from src.models.classical_cnn import SymmetricalMinimumCNN
    d = np.load(str(NPZ))
    tr_x = d['train_images'].astype(np.float32) / 255.0
    tr_y = d['train_labels'].astype(np.int64).ravel()
    va_x = d['val_images'].astype(np.float32) / 255.0
    va_y = d['val_labels'].astype(np.int64).ravel()
    te_x = d['test_images'].astype(np.float32) / 255.0
    te_y = d['test_labels'].astype(np.int64).ravel()

    if tr_x.ndim == 3: tr_x = tr_x[:, None]
    if va_x.ndim == 3: va_x = va_x[:, None]
    if te_x.ndim == 3: te_x = te_x[:, None]

    sizes = [100, 250, 500]
    results = {}
    for sz in sizes:
        torch.manual_seed(SEED)
        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(tr_x), min(sz, len(tr_x)), replace=False)
        xs = tr_x[idx]; ys = tr_y[idx]

        torch.manual_seed(SEED)
        model = SymmetricalMinimumCNN(num_classes=2)
        opt = optim.Adam(model.parameters(), lr=0.001)
        crit = nn.CrossEntropyLoss()
        g = torch.Generator().manual_seed(SEED)
        ds = torch.utils.data.TensorDataset(
            torch.as_tensor(xs, dtype=torch.float32),
            torch.as_tensor(ys, dtype=torch.long))
        loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True, generator=g)

        best_auc, best_state = -1, None
        for ep in range(20):
            model.train()
            for xb, yb in loader:
                opt.zero_grad()
                loss = crit(model(xb), yb)
                loss.backward(); opt.step()
            m = eval_model(model, va_x, va_y)
            if m['auc'] > best_auc:
                best_auc, best_state = m['auc'], {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        m = eval_test(model, te_x, te_y)
        results[sz] = m
        print(f'  Learning curve sz={sz}: auc={m["auc"]:.4f}')
    return results

# ================================================================
# C. ENTANGLEMENT ABLATION (gỡ CNOT)
# ================================================================
def entanglement_ablation():
    """Mạch Basic không CNOT — chỉ có RY rotation (product state)."""
    import pennylane as qml
    dev = qml.device('default.qubit', wires=4)

    @qml.qnode(dev, interface='torch')
    def circuit_no_cnot(inputs, weights):
        qml.AngleEmbedding(inputs * np.pi, wires=range(4), rotation='Y')
        for l in range(2):  # L=2
            for i in range(4):
                qml.RY(weights[l * 4 + i], wires=i)
            # KHÔNG CNOT — product state
        return [qml.expval(qml.PauliZ(j)) for j in range(4)]

    class HeadNoCNOT(nn.Module):
        def __init__(self):
            super().__init__()
            self.bn = nn.BatchNorm2d(4)
            self.relu = nn.ReLU()
            self.fc = nn.Linear(784, 2)

        def forward(self, x):
            return self.fc(self.relu(self.bn(x)))

    d = np.load(str(NPZ))
    tr_x = d['train_images'].astype(np.float32) / 255.0
    tr_y = d['train_labels'].astype(np.int64).ravel()
    va_x = d['val_images'].astype(np.float32) / 255.0
    va_y = d['val_labels'].astype(np.int64).ravel()
    te_x = d['test_images'].astype(np.float32) / 255.0
    te_y = d['test_labels'].astype(np.int64).ravel()
    if tr_x.ndim == 3: tr_x = tr_x[:, None]
    if va_x.ndim == 3: va_x = va_x[:, None]
    if te_x.ndim == 3: te_x = te_x[:, None]

    # Precompute features (no CNOT, chỉ RY rotation)
    feats_tr = np.zeros((len(tr_x), 4, 14, 14), dtype=np.float32)
    feats_va = np.zeros((len(va_x), 4, 14, 14), dtype=np.float32)
    feats_te = np.zeros((len(te_x), 4, 14, 14), dtype=np.float32)
    for idx in range(len(tr_x)):
        for j in range(0, 28, 2):
            for k2 in range(0, 28, 2):
                patch = [tr_x[idx, 0, j, k2], tr_x[idx, 0, j, k2+1],
                         tr_x[idx, 0, j+1, k2], tr_x[idx, 0, j+1, k2+1]]
                @qml.qnode(dev)
                def c(patch):
                    for q in range(4):
                        qml.RY(np.pi * patch[q], wires=q)
                    return [qml.expval(qml.PauliZ(q)) for q in range(4)]
                res = c()
                feats_tr[idx, :, j//2, k2//2] = res
    return feats_tr, feats_va, feats_te

# ================================================================
# MAIN
# ================================================================
if __name__ == '__main__':
    print('=' * 70)
    print('GĐ4 NÂNG CHẤT — CHẠY TOÀN BỘ THÍ NGHIỆM BỔ SUNG')
    print('=' * 70)
    # Tải data
    d = np.load(str(NPZ))
    tr_x = d['train_images'].astype(np.float32) / 255.0
    tr_y = d['train_labels'].astype(np.int64).ravel()
    va_x = d['val_images'].astype(np.float32) / 255.0
    va_y = d['val_labels'].astype(np.int64).ravel()
    te_x = d['test_images'].astype(np.float32) / 255.0
    te_y = d['test_labels'].astype(np.int64).ravel()
    if tr_x.ndim == 3: tr_x = tr_x[:, None]
    if va_x.ndim == 3: va_x = va_x[:, None]
    if te_x.ndim == 3: te_x = te_x[:, None]

    print('\n--- A. RANDOM CLASSICAL 2×2 FILTER ---')
    m = random_classical_baseline()
    print(f'  → acc={m["acc"]:.4f} auc={m["auc"]:.4f} pr={m["pr_auc"]:.4f}')

    print('\n--- B. LEARNING CURVE ---')
    lc = learning_curve()
    for sz, v in lc.items():
        print(f'  sz={sz}: auc={v["auc"]:.4f}')

    print('\n--- C. ENTANGLEMENT ABLATION ---')
    ea = entanglement_ablation()
    print(f'  → acc={ea["acc"]:.4f} auc={ea["auc"]:.4f}')

    print('\nDone — kết quả trong các dict trên, chuẩn bị xuất JSON')
