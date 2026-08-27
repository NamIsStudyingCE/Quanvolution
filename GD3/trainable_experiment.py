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

def compute_comprehensive_statistical_tests(results):
    """
    Computes pairwise statistical significance (Paired t-test and Wilcoxon signed-rank test)
    between all available model pairs across all 6 classification metrics.
    """
    metrics = ['acc', 'bacc', 'f1', 'mcc', 'auc', 'pr_auc']
    available_models = list(results.keys())
    
    stat_report = {}
    
    for i in range(len(available_models)):
        for j in range(i + 1, len(available_models)):
            m1 = available_models[i]
            m2 = available_models[j]
            pair_key = f"{m1}_vs_{m2}"
            stat_report[pair_key] = {}
            
            for m in metrics:
                v1 = [r[m] for r in results[m1]['test_metrics']]
                v2 = [r[m] for r in results[m2]['test_metrics']]
                delta = float(np.mean(v1) - np.mean(v2))
                t_stat, p_val = stats.ttest_rel(v1, v2)
                try:
                    w_stat, w_pval = stats.wilcoxon([a - b for a, b in zip(v1, v2)])
                except Exception:
                    w_stat, w_pval = 0.0, 1.0
                    
                stat_report[pair_key][m] = {
                    'delta': delta,
                    'mean_' + m1: float(np.mean(v1)),
                    'mean_' + m2: float(np.mean(v2)),
                    't_stat': float(t_stat),
                    'p_value_ttest': float(p_val),
                    'wilcoxon_p_value': float(w_pval)
                }
                
    return stat_report

def run_3tier_dataset_experiment(dataset_name, data_splits, num_classes, num_epochs=20, batch_size=32, seeds=[0, 42, 100, 2023, 777], n_quantum_layers=1, save_json_path=None):
    """
    Master runner executing the comprehensive 3-Tier Multi-Model benchmark:
      Tier 1: Fixed Basic vs Trainable Basic
      Tier 2: Trainable Basic vs Fixed Champion GĐ2 (random_L1 / basic_L2)
      Tier 3: Fixed Strongly (Full-Exp) vs Trainable Strongly (Full-Exp) vs Classical CNN
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "="*90)
    print(f" EXPERIMENT: 3-TIER MULTI-MODEL BENCHMARK ON {dataset_name.upper()}")
    print(f" Device: {device} | Seeds: {len(seeds)} {seeds} | Epochs: {num_epochs} | Batch Size: {batch_size}")
    print("="*90)

    c_train_ds = data_splits['classical_train']
    c_val_ds   = data_splits['classical_val']
    c_test_ds  = data_splits['classical_test']

    q_train_ds = data_splits['quantum_train']
    q_val_ds   = data_splits['quantum_val']
    q_test_ds  = data_splits['quantum_test']

    q_feat_dir = "data/quantum_features"
    is_breast = "breast" in dataset_name.lower()
    basic_tag = "basic_L2" if is_breast else "basic_L1"
    strongly_tag = "strongly_L2" if is_breast else "strongly_L1"
    champion_tag = "basic_L2" if is_breast else "random_L1"

    # Precomputed features loader helper
    def load_cached_feats(tag):
        tr_f, tr_l = torch.load(f"{q_feat_dir}/{dataset_name}_{tag}_train.pt", weights_only=True)
        va_f, va_l = torch.load(f"{q_feat_dir}/{dataset_name}_{tag}_val.pt",   weights_only=True)
        te_f, te_l = torch.load(f"{q_feat_dir}/{dataset_name}_{tag}_test.pt",  weights_only=True)
        return tr_f, tr_l, va_f, va_l, te_f, te_l

    basic_tr_f, basic_tr_l, basic_va_f, basic_va_l, basic_te_f, basic_te_l = load_cached_feats(basic_tag)
    strongly_tr_f, strongly_tr_l, strongly_va_f, strongly_va_l, strongly_te_f, strongly_te_l = load_cached_feats(strongly_tag)
    champ_tr_f, champ_tr_l, champ_va_f, champ_va_l, champ_te_f, champ_te_l = load_cached_feats(champion_tag)

    models_to_run = [
        ('classical_cnn',      'Classical CNN Baseline',                             'classical', None),
        ('fixed_basic',        f'Fixed BasicEntangler ({basic_tag})',                'precomputed', (basic_tr_f, basic_tr_l, basic_va_f, basic_va_l, basic_te_f, basic_te_l)),
        ('trainable_basic',    f'Trainable BasicEntangler (L={n_quantum_layers})',    'trainable_basic', None),
    ]

    if not is_breast:
        models_to_run.append(('fixed_champion_gd2', f'Fixed Champion GĐ2 ({champion_tag})', 'precomputed', (champ_tr_f, champ_tr_l, champ_va_f, champ_va_l, champ_te_f, champ_te_l)))

    models_to_run.extend([
        ('fixed_strongly',     f'Fixed Full-Expressive ({strongly_tag})',             'precomputed', (strongly_tr_f, strongly_tr_l, strongly_va_f, strongly_va_l, strongly_te_f, strongly_te_l)),
        ('trainable_strongly', f'Trainable Full-Expressive (L={n_quantum_layers}, 3-Axis)', 'trainable_strongly', None)
    ])

    results = {m_id: {'histories': [], 'test_metrics': []} for m_id, _, _, _ in models_to_run}

    total_models = len(models_to_run)
    total_seeds = len(seeds)
    total_runs = total_models * total_seeds
    run_idx = 0
    start_all_time = time.time()

    for m_id, m_desc, m_category, m_data in models_to_run:
        print("\n" + "#"*90)
        print(f" [MODEL BENCHMARK]: {m_desc.upper()}")
        print("#"*90)

        for s_idx, seed in enumerate(seeds):
            run_idx += 1
            run_start_time = time.time()
            set_seed(seed)
            g = torch.Generator().manual_seed(seed)

            if m_category == 'classical':
                tr_loader = DataLoader(c_train_ds, batch_size=batch_size, shuffle=True, generator=g)
                va_loader = DataLoader(c_val_ds,   batch_size=batch_size, shuffle=False)
                te_loader = DataLoader(c_test_ds,  batch_size=batch_size, shuffle=False)
                model = SymmetricalMinimumCNN(num_classes=num_classes).to(device)
                optimizer = optim.Adam(model.parameters(), lr=0.001)
            elif m_category == 'precomputed':
                tr_f, tr_l, va_f, va_l, te_f, te_l = m_data
                tr_loader = DataLoader(TensorDataset(tr_f, tr_l), batch_size=batch_size, shuffle=True, generator=g)
                va_loader = DataLoader(TensorDataset(va_f, va_l), batch_size=batch_size, shuffle=False)
                te_loader = DataLoader(TensorDataset(te_f, te_l), batch_size=batch_size, shuffle=False)
                model = QuanvolutionClassifier(num_classes=num_classes).to(device)
                optimizer = optim.Adam(model.parameters(), lr=0.001)
            elif m_category == 'trainable_basic':
                tr_loader = DataLoader(q_train_ds, batch_size=batch_size, shuffle=True, generator=g)
                va_loader = DataLoader(q_val_ds,   batch_size=batch_size, shuffle=False)
                te_loader = DataLoader(q_test_ds,  batch_size=batch_size, shuffle=False)
                model = TrainableQuanvolutionalNetwork(num_classes=num_classes, n_layers=n_quantum_layers, ansatz="basic", diff_method="backprop").to(device)
                param_groups = model.get_parameter_groups(lr_quantum=0.01, lr_classical=0.001)
                optimizer = optim.Adam(param_groups)
            elif m_category == 'trainable_strongly':
                tr_loader = DataLoader(q_train_ds, batch_size=batch_size, shuffle=True, generator=g)
                va_loader = DataLoader(q_val_ds,   batch_size=batch_size, shuffle=False)
                te_loader = DataLoader(q_test_ds,  batch_size=batch_size, shuffle=False)
                model = TrainableQuanvolutionalNetwork(num_classes=num_classes, n_layers=n_quantum_layers, ansatz="strongly", diff_method="backprop").to(device)
                param_groups = model.get_parameter_groups(lr_quantum=0.01, lr_classical=0.001)
                optimizer = optim.Adam(param_groups)
            else:
                raise ValueError(f"Unknown category: {m_category}")

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

                if 'trainable' in m_category:
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
            results[m_id]['histories'].append(history)
            results[m_id]['test_metrics'].append(best_test_metrics)

            elapsed_all = time.time() - start_all_time
            avg_per_run = elapsed_all / run_idx
            remain_runs = total_runs - run_idx
            eta_seconds = avg_per_run * remain_runs

            print(f"  --> Seed {seed} Completed in {run_elapsed:.2f}s | Test ROC-AUC: {best_test_metrics['auc']:.4f} | PR-AUC: {best_test_metrics['pr_auc']:.4f} | ACC: {best_test_metrics['acc']:.4f}")
            print(f"      [Progress: {run_idx}/{total_runs} runs] | Elapsed: {elapsed_all/60:.1f}m | ETA: {eta_seconds/60:.1f}m")

            if save_json_path:
                os.makedirs(os.path.dirname(save_json_path), exist_ok=True)
                stat_report = compute_comprehensive_statistical_tests(results) if run_idx == total_runs else {}
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

    final_stats = compute_comprehensive_statistical_tests(results)
    return results, final_stats

# Maintain backward compatibility alias
def run_full_dataset_experiment(dataset_name, data_splits, num_classes, num_epochs=20, batch_size=32, seeds=[0, 42, 100, 2023, 777], n_quantum_layers=1, save_json_path=None):
    return run_3tier_dataset_experiment(dataset_name, data_splits, num_classes, num_epochs, batch_size, seeds, n_quantum_layers, save_json_path)



