import os
import sys
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np
from scipy import stats
from datetime import datetime

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

def train_epoch_trainable(model, loader, criterion, optimizer, device):
    """
    Trains one epoch for TrainableQuanvolutionalNetwork and records quantum gradient norm.
    """
    model.train()
    running_loss = 0.0
    total_grad_norm = 0.0
    num_batches = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        if hasattr(model, 'get_quantum_grad_norm'):
            total_grad_norm += model.get_quantum_grad_norm()

        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        num_batches += 1

    avg_loss = running_loss / len(loader.dataset)
    avg_grad_norm = total_grad_norm / max(num_batches, 1)
    return avg_loss, avg_grad_norm

def train_epoch_standard(model, loader, criterion, optimizer, device):
    """
    Standard training epoch for classical CNN and fixed quanvolution.
    """
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
    return running_loss / len(loader.dataset), 0.0

def evaluate_model(model, loader, criterion, device, num_classes):
    """
    Evaluates model on validation or test dataset.
    """
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

def run_single_model_training(model_type, train_loader, val_loader, test_loader, num_classes, num_epochs, seed, device, n_layers=1):
    set_seed(seed)
    
    if model_type == 'classical_cnn':
        model = SymmetricalMinimumCNN(num_classes=num_classes).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    elif model_type == 'fixed_quanv':
        model = QuanvolutionClassifier(num_classes=num_classes).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    elif model_type == 'trainable_quanv':
        model = TrainableQuanvolutionalNetwork(num_classes=num_classes, n_layers=n_layers, diff_method="backprop").to(device)
        param_groups = model.get_parameter_groups(lr_quantum=0.01, lr_classical=0.001)
        optimizer = optim.Adam(param_groups)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    criterion = nn.CrossEntropyLoss()

    history = {
        'train_loss': [],
        'val_loss': [],
        'val_auc': [],
        'grad_norms': [],
        'theta_trajectories': []
    }

    best_val_auc = -1.0
    best_test_metrics = None

    for epoch in range(1, num_epochs + 1):
        if model_type == 'trainable_quanv':
            theta_curr = model.get_quantum_weights().tolist()
            history['theta_trajectories'].append(theta_curr)
            t_loss, g_norm = train_epoch_trainable(model, train_loader, criterion, optimizer, device)
            history['grad_norms'].append(g_norm)
        else:
            t_loss, g_norm = train_epoch_standard(model, train_loader, criterion, optimizer, device)
            history['grad_norms'].append(0.0)

        v_loss, v_metrics = evaluate_model(model, val_loader, criterion, device, num_classes)

        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_loss)
        history['val_auc'].append(v_metrics['auc'])

        if v_metrics['auc'] > best_val_auc:
            best_val_auc = v_metrics['auc']
            _, best_test_metrics = evaluate_model(model, test_loader, criterion, device, num_classes)

    return history, best_test_metrics

def compute_trainable_statistical_tests(results):
    """
    Computes pairwise statistical significance (Paired t-test and Wilcoxon signed-rank test)
    between all 3 model pairs across all 6 classification metrics.
    """
    metrics = ['acc', 'bacc', 'f1', 'mcc', 'auc', 'pr_auc']
    pairs = [
        ('trainable_quanv', 'fixed_quanv', 'trainable_vs_fixed'),
        ('trainable_quanv', 'classical_cnn', 'trainable_vs_classical'),
        ('classical_cnn', 'fixed_quanv', 'classical_vs_fixed')
    ]
    
    stat_report = {}
    
    for m1, m2, key in pairs:
        stat_report[key] = {}
        for m in metrics:
            v1 = [r[m] for r in results[m1]['test_metrics']]
            v2 = [r[m] for r in results[m2]['test_metrics']]
            delta = float(np.mean(v1) - np.mean(v2))
            t_stat, p_val = stats.ttest_rel(v1, v2)
            try:
                w_stat, w_pval = stats.wilcoxon([a - b for a, b in zip(v1, v2)])
            except Exception:
                w_stat, w_pval = 0.0, 1.0
                
            stat_report[key][m] = {
                'delta': delta,
                'mean_' + m1: float(np.mean(v1)),
                'mean_' + m2: float(np.mean(v2)),
                't_stat': float(t_stat),
                'p_value_ttest': float(p_val),
                'wilcoxon_p_value': float(w_pval)
            }
            
    return stat_report

def run_trainable_poc_experiment(data_splits, num_classes, epochs=20, seeds=[0, 42, 100]):
    """
    Runs the legacy 3-model comparative POC experiment.
    """
    return run_full_dataset_experiment(
        dataset_name="octmnist_poc_500",
        data_splits=data_splits,
        num_classes=num_classes,
        num_epochs=epochs,
        batch_size=35,
        seeds=seeds,
        n_quantum_layers=1
    )[0]

def run_full_dataset_experiment(dataset_name, data_splits, num_classes, num_epochs=20, batch_size=32, seeds=[0, 42, 100, 2023, 777], n_quantum_layers=1, save_json_path=None):
    """
    Master runner executing the 3-model comparative experiment across all specified seeds.
    Includes real-time ETA, trajectory logging, and atomic checkpointing.
    """
    import time
    from datetime import datetime

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "="*85)
    print(f" EXPERIMENT: FULL-SCALE BENCHMARK ON {dataset_name.upper()}")
    print(f" Device: {device} | Seeds: {len(seeds)} {seeds} | Epochs: {num_epochs} | Batch Size: {batch_size}")
    print("="*85)

    c_train_ds = data_splits['classical_train']
    c_val_ds   = data_splits['classical_val']
    c_test_ds  = data_splits['classical_test']

    q_train_ds = data_splits['quantum_train']
    q_val_ds   = data_splits['quantum_val']
    q_test_ds  = data_splits['quantum_test']

    feature_tag = "basic_L2" if "breast" in dataset_name.lower() else "random_L1"
    q_feat_dir = "data/quantum_features"
    q_train_f, q_train_l = torch.load(f"{q_feat_dir}/{dataset_name}_{feature_tag}_train.pt", weights_only=True)
    q_val_f,   q_val_l   = torch.load(f"{q_feat_dir}/{dataset_name}_{feature_tag}_val.pt",   weights_only=True)
    q_test_f,  q_test_l  = torch.load(f"{q_feat_dir}/{dataset_name}_{feature_tag}_test.pt",  weights_only=True)

    results = {
        'classical_cnn':   {'histories': [], 'test_metrics': []},
        'fixed_quanv':     {'histories': [], 'test_metrics': []},
        'trainable_quanv': {'histories': [], 'test_metrics': []}
    }

    models_to_run = [
        ('classical_cnn',   'Classical CNN Baseline'),
        ('fixed_quanv',     f'Fixed Quanvolution ({feature_tag})'),
        ('trainable_quanv', f'Trainable Quanvolution (L={n_quantum_layers})')
    ]

    total_models = len(models_to_run)
    total_seeds = len(seeds)
    total_runs = total_models * total_seeds
    run_idx = 0
    start_all_time = time.time()

    for m_type, m_desc in models_to_run:
        print("\n" + "#"*85)
        print(f" [MODEL BENCHMARK]: {m_desc.upper()}")
        print("#"*85)

        for s_idx, seed in enumerate(seeds):
            run_idx += 1
            run_start_time = time.time()
            set_seed(seed)
            g = torch.Generator().manual_seed(seed)

            if m_type == 'classical_cnn':
                tr_loader = DataLoader(c_train_ds, batch_size=batch_size, shuffle=True, generator=g)
                va_loader = DataLoader(c_val_ds,   batch_size=batch_size, shuffle=False)
                te_loader = DataLoader(c_test_ds,  batch_size=batch_size, shuffle=False)
                model = SymmetricalMinimumCNN(num_classes=num_classes).to(device)
                optimizer = optim.Adam(model.parameters(), lr=0.001)
            elif m_type == 'fixed_quanv':
                tr_loader = DataLoader(TensorDataset(q_train_f, q_train_l), batch_size=batch_size, shuffle=True, generator=g)
                va_loader = DataLoader(TensorDataset(q_val_f,   q_val_l),   batch_size=batch_size, shuffle=False)
                te_loader = DataLoader(TensorDataset(q_test_f,  q_test_l),  batch_size=batch_size, shuffle=False)
                model = QuanvolutionClassifier(num_classes=num_classes).to(device)
                optimizer = optim.Adam(model.parameters(), lr=0.001)
            else:
                tr_loader = DataLoader(q_train_ds, batch_size=batch_size, shuffle=True, generator=g)
                va_loader = DataLoader(q_val_ds,   batch_size=batch_size, shuffle=False)
                te_loader = DataLoader(q_test_ds,  batch_size=batch_size, shuffle=False)
                model = TrainableQuanvolutionalNetwork(num_classes=num_classes, n_layers=n_quantum_layers, diff_method="backprop").to(device)
                param_groups = model.get_parameter_groups(lr_quantum=0.01, lr_classical=0.001)
                optimizer = optim.Adam(param_groups)

            criterion = nn.CrossEntropyLoss()

            history = {
                'train_loss': [],
                'val_loss': [],
                'val_auc': [],
                'grad_norms': [],
                'theta_trajectories': []
            }

            best_val_auc = -1.0
            best_test_metrics = None

            print(f"\n>>> [{run_idx}/{total_runs}] Starting {m_desc} | Seed {seed} ({s_idx+1}/{total_seeds})")

            for epoch in range(1, num_epochs + 1):
                ep_start = time.time()

                if m_type == 'trainable_quanv':
                    theta_curr = model.get_quantum_weights().tolist()
                    history['theta_trajectories'].append(theta_curr)
                    t_loss, g_norm = train_epoch_trainable(model, tr_loader, criterion, optimizer, device)
                    history['grad_norms'].append(g_norm)
                else:
                    t_loss, g_norm = train_epoch_standard(model, tr_loader, criterion, optimizer, device)
                    history['grad_norms'].append(0.0)

                v_loss, v_metrics = evaluate_model(model, va_loader, criterion, device, num_classes)
                history['train_loss'].append(t_loss)
                history['val_loss'].append(v_loss)
                history['val_auc'].append(v_metrics['auc'])

                is_best = False
                if v_metrics['auc'] > best_val_auc:
                    best_val_auc = v_metrics['auc']
                    _, best_test_metrics = evaluate_model(model, te_loader, criterion, device, num_classes)
                    is_best = True

                ep_time = time.time() - ep_start
                best_flag = " [BEST]" if is_best else ""

                if epoch % 5 == 0 or epoch == num_epochs or is_best:
                    print(f"    Epoch {epoch:02d}/{num_epochs:02d} ({ep_time:.2f}s) | Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f} | Val AUC: {v_metrics['auc']:.4f} | Val ACC: {v_metrics['acc']:.4f}{best_flag}")

            run_elapsed = time.time() - run_start_time
            results[m_type]['histories'].append(history)
            results[m_type]['test_metrics'].append(best_test_metrics)

            elapsed_all = time.time() - start_all_time
            avg_per_run = elapsed_all / run_idx
            remain_runs = total_runs - run_idx
            eta_seconds = avg_per_run * remain_runs

            print(f"  --> Seed {seed} Completed in {run_elapsed:.2f}s | Test ROC-AUC: {best_test_metrics['auc']:.4f} | PR-AUC: {best_test_metrics['pr_auc']:.4f} | ACC: {best_test_metrics['acc']:.4f}")
            print(f"      [Progress: {run_idx}/{total_runs} runs] | Elapsed: {elapsed_all/60:.1f}m | ETA: {eta_seconds/60:.1f}m")

            if save_json_path:
                os.makedirs(os.path.dirname(save_json_path), exist_ok=True)
                stat_report = compute_trainable_statistical_tests(results) if run_idx == total_runs else {}
                export_data = {
                    "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                    "dataset": dataset_name,
                    "num_epochs": num_epochs,
                    "seeds": seeds,
                    "progress": f"{run_idx}/{total_runs} completed",
                    "statistical_tests": stat_report,
                    "raw_results": results
                }
                with open(save_json_path, "w", encoding="utf-8") as f:
                    json.dump(export_data, f, indent=4)

    final_stats = compute_trainable_statistical_tests(results)
    return results, final_stats


