import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np

# Ensure UTF-8 console output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from models.classical_cnn import SymmetricalMinimumCNN
from models.quantum_model import QuanvolutionClassifier
from models.trainable_quanv import TrainableQuanvolutionalNetwork
from utils.metrics import calculate_metrics

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_epoch_standard(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(loader.dataset)

def evaluate_standard(model, loader, criterion, device, num_classes):
    model.eval()
    running_loss = 0.0
    all_targets, all_probs, all_preds = [], [], []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    loss = running_loss / len(loader.dataset)
    metrics = calculate_metrics(np.array(all_targets), np.array(all_probs), np.array(all_preds), num_classes)
    return loss, metrics

def run_single_model_training(model_type, train_loader, val_loader, test_loader, num_classes, num_epochs, seed, device):
    set_seed(seed)
    
    if model_type == 'classical_cnn':
        model = SymmetricalMinimumCNN(num_classes=num_classes).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    elif model_type == 'fixed_quanv':
        model = QuanvolutionClassifier(num_classes=num_classes).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    elif model_type == 'trainable_quanv':
        model = TrainableQuanvolutionalNetwork(num_classes=num_classes, n_layers=1).to(device)
        param_groups = model.get_parameter_groups(lr_quantum=0.01, lr_classical=0.001)
        optimizer = optim.Adam(param_groups)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    criterion = nn.CrossEntropyLoss()

    history = {
        'train_loss': [],
        'val_loss': [],
        'val_auc': []
    }

    best_val_auc = -1.0
    best_test_metrics = None

    for epoch in range(num_epochs):
        t_loss = train_epoch_standard(model, train_loader, criterion, optimizer, device)
        v_loss, v_metrics = evaluate_standard(model, val_loader, criterion, device, num_classes)

        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_loss)
        history['val_auc'].append(v_metrics['auc'])

        if v_metrics['auc'] > best_val_auc:
            best_val_auc = v_metrics['auc']
            _, best_test_metrics = evaluate_standard(model, test_loader, criterion, device, num_classes)

    return history, best_test_metrics

def run_trainable_poc_experiment(data_splits, num_classes, epochs=20, seeds=[0, 42, 100]):
    """
    Runs the 3-model comparative experiment across fixed seeds:
      1. Classical CNN
      2. Fixed Quanvolution (Precomputed features)
      3. Trainable Quanvolution (End-to-End differentiable)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[DEVICE]: Using {device} for Training & Quantum Backpropagation")

    # 1. Classical DataLoaders (Normalized [-1, 1])
    batch_size = 35
    c_train_ds = data_splits['classical_train']
    c_val_ds   = data_splits['classical_val']
    c_test_ds  = data_splits['classical_test']

    # 2. Quantum Raw DataLoaders (ToTensor only [0, 1]) for Trainable Quanv
    q_train_ds = data_splits['quantum_train']
    q_val_ds   = data_splits['quantum_val']
    q_test_ds  = data_splits['quantum_test']

    # 3. Precomputed Features for Fixed Quanv
    # Load cached fixed features (random_L1)
    q_train_f, q_train_l = torch.load("data/quantum_features/octmnist_random_L1_train.pt", weights_only=True)
    q_val_f,   q_val_l   = torch.load("data/quantum_features/octmnist_random_L1_val.pt",   weights_only=True)
    q_test_f,  q_test_l  = torch.load("data/quantum_features/octmnist_random_L1_test.pt",  weights_only=True)

    # Slice to match 500-sample POC subset (350 train, 50 val, 100 test)
    q_train_f, q_train_l = q_train_f[:350], q_train_l[:350]
    q_val_f,   q_val_l   = q_val_f[:50],   q_val_l[:50]
    q_test_f,  q_test_l  = q_test_f[:100],  q_test_l[:100]

    results = {
        'classical_cnn':   {'histories': [], 'test_metrics': []},
        'fixed_quanv':     {'histories': [], 'test_metrics': []},
        'trainable_quanv': {'histories': [], 'test_metrics': []}
    }

    models_to_run = ['classical_cnn', 'fixed_quanv', 'trainable_quanv']

    for m_type in models_to_run:
        print(f"\n=======================================================")
        print(f" BENCHMARKING MODEL: {m_type.upper()} ({len(seeds)} SEEDS)")
        print(f"=======================================================")

        for seed in seeds:
            print(f"  -> Running Seed {seed} ({epochs} epochs)...")
            g = torch.Generator().manual_seed(seed)

            if m_type == 'classical_cnn':
                tr_loader = DataLoader(c_train_ds, batch_size=batch_size, shuffle=True, generator=g)
                va_loader = DataLoader(c_val_ds,   batch_size=batch_size, shuffle=False)
                te_loader = DataLoader(c_test_ds,  batch_size=batch_size, shuffle=False)
            elif m_type == 'fixed_quanv':
                tr_loader = DataLoader(TensorDataset(q_train_f, q_train_l), batch_size=batch_size, shuffle=True, generator=g)
                va_loader = DataLoader(TensorDataset(q_val_f,   q_val_l),   batch_size=batch_size, shuffle=False)
                te_loader = DataLoader(TensorDataset(q_test_f,  q_test_l),  batch_size=batch_size, shuffle=False)
            else: # trainable_quanv
                tr_loader = DataLoader(q_train_ds, batch_size=batch_size, shuffle=True, generator=g)
                va_loader = DataLoader(q_val_ds,   batch_size=batch_size, shuffle=False)
                te_loader = DataLoader(q_test_ds,  batch_size=batch_size, shuffle=False)

            hist, test_m = run_single_model_training(m_type, tr_loader, va_loader, te_loader, num_classes, epochs, seed, device)
            results[m_type]['histories'].append(hist)
            results[m_type]['test_metrics'].append(test_m)
            print(f"     [Seed {seed} Done] Test AUC: {test_m['auc']:.4f} | PR-AUC: {test_m['pr_auc']:.4f} | Acc: {test_m['acc']:.4f}")

    return results
