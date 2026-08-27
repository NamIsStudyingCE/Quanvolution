import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import json
from scipy import stats

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from models.quantum_model import QuanvolutionClassifier
from models.circuits import CIRCUIT_DICT
from utils.metrics import calculate_metrics

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def make_quantum_loaders(q_train_f, q_train_l, q_val_f, q_val_l, q_test_f, q_test_l, batch_size, seed):
    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(TensorDataset(q_train_f, q_train_l), batch_size=batch_size, shuffle=True, generator=g, num_workers=0)
    val_loader   = DataLoader(TensorDataset(q_val_f,   q_val_l),   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(TensorDataset(q_test_f,  q_test_l),  batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader

def train_epoch(model, loader, criterion, optimizer, device):
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

def evaluate(model, loader, criterion, device, num_classes):
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

def train_quantum_model(train_loader, val_loader, test_loader, num_classes, num_epochs, seed):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QuanvolutionClassifier(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_val_auc = -1.0
    best_test_metrics = None

    for epoch in range(num_epochs):
        train_epoch(model, train_loader, criterion, optimizer, device)
        _, val_metrics = evaluate(model, val_loader, criterion, device, num_classes)

        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            _, best_test_metrics = evaluate(model, test_loader, criterion, device, num_classes)

    return best_test_metrics

def run_stage1_screening(dataset_name, num_classes, feature_dir="data/quantum_features", epochs=30, batch_size=64):
    """
    Stage 1: Screen all 6 circuit variants using 5 fixed seeds.
    """
    screening_seeds = [0, 42, 100, 2023, 777]
    circuits = list(CIRCUIT_DICT.keys())
    stage1_results = {}

    print(f"\n=======================================================")
    print(f" [STAGE 1] SCREENING 6 CIRCUITS (5 SEEDS) - {dataset_name.upper()}")
    print(f"=======================================================")

    for circuit_name in circuits:
        print(f"\n>>> Evaluating Circuit: {circuit_name} across 5 seeds...")
        q_train_f, q_train_l = torch.load(f"{feature_dir}/{dataset_name}_{circuit_name}_train.pt", weights_only=True)
        q_val_f,   q_val_l   = torch.load(f"{feature_dir}/{dataset_name}_{circuit_name}_val.pt",   weights_only=True)
        q_test_f,  q_test_l  = torch.load(f"{feature_dir}/{dataset_name}_{circuit_name}_test.pt",  weights_only=True)

        circuit_seed_metrics = []
        for seed in screening_seeds:
            train_l, val_l, test_l = make_quantum_loaders(q_train_f, q_train_l, q_val_f, q_val_l, q_test_f, q_test_l, batch_size, seed)
            metrics = train_quantum_model(train_l, val_l, test_l, num_classes, epochs, seed)
            circuit_seed_metrics.append(metrics)

        # Compute summary
        metric_keys = circuit_seed_metrics[0].keys()
        summary = {}
        for m in metric_keys:
            vals = [r[m] for r in circuit_seed_metrics]
            summary[m] = {
                'mean': float(np.mean(vals)),
                'std': float(np.std(vals)),
                'formatted': f"{np.mean(vals):.4f} +- {np.std(vals):.4f}"
            }

        stage1_results[circuit_name] = {
            'summary': summary,
            'raw_seed_metrics': circuit_seed_metrics
        }
        print(f"  Result: AUC = {summary['auc']['formatted']} | PR-AUC = {summary['pr_auc']['formatted']} | ACC = {summary['acc']['formatted']}")

    # Select champion based on highest mean ROC-AUC
    champion = max(stage1_results.keys(), key=lambda c: stage1_results[c]['summary']['auc']['mean'])
    print(f"\n>>> [CHAMPION CIRCUIT] for {dataset_name.upper()}: [{champion}] with AUC = {stage1_results[champion]['summary']['auc']['formatted']}")
    return stage1_results, champion

def run_stage2_deep_validation(dataset_name, champion_circuit, num_classes, feature_dir="data/quantum_features", epochs=30, batch_size=64):
    """
    Stage 2: Deep 10-seed evaluation on the Champion Circuit + Statistical Significance.
    """
    deep_seeds = [0, 42, 100, 2023, 777, 999, 1234, 5678, 1111, 2222]
    print(f"\n=======================================================")
    print(f" [STAGE 2] DEEP VALIDATION (10 SEEDS) ON CHAMPION: {champion_circuit} ({dataset_name.upper()})")
    print(f"=======================================================")

    q_train_f, q_train_l = torch.load(f"{feature_dir}/{dataset_name}_{champion_circuit}_train.pt", weights_only=True)
    q_val_f,   q_val_l   = torch.load(f"{feature_dir}/{dataset_name}_{champion_circuit}_val.pt",   weights_only=True)
    q_test_f,  q_test_l  = torch.load(f"{feature_dir}/{dataset_name}_{champion_circuit}_test.pt",  weights_only=True)

    champion_seed_metrics = []
    for seed in deep_seeds:
        train_l, val_l, test_l = make_quantum_loaders(q_train_f, q_train_l, q_val_f, q_val_l, q_test_f, q_test_l, batch_size, seed)
        metrics = train_quantum_model(train_l, val_l, test_l, num_classes, epochs, seed)
        champion_seed_metrics.append(metrics)

    metric_keys = champion_seed_metrics[0].keys()
    summary = {}
    for m in metric_keys:
        vals = [r[m] for r in champion_seed_metrics]
        summary[m] = {
            'mean': float(np.mean(vals)),
            'std': float(np.std(vals)),
            'formatted': f"{np.mean(vals):.4f} +- {np.std(vals):.4f}"
        }

    # Compare with Classical Baseline if available
    stat_report = {}
    classical_file = f"results/{dataset_name}_classical_latest.json"
    if os.path.exists(classical_file):
        with open(classical_file, "r") as f:
            c_data = json.load(f)
        c_raw = c_data['raw_seed_metrics']
        print(f"\n--- Statistical Tests: Champion ({champion_circuit}) vs Classical Baseline CNN ---")
        for m in metric_keys:
            c_vals = [r[m] for r in c_raw]
            q_vals = [r[m] for r in champion_seed_metrics]
            t_stat, p_val = stats.ttest_rel(c_vals, q_vals)
            try:
                w_stat, w_pval = stats.wilcoxon([qi - ci for ci, qi in zip(c_vals, q_vals)])
            except Exception:
                w_stat, w_pval = 0.0, 1.0
            
            delta = np.mean(q_vals) - np.mean(c_vals)
            stat_report[m] = {
                'delta': float(delta),
                't_stat': float(t_stat),
                'p_value_ttest': float(p_val),
                'wilcoxon_p_value': float(w_pval)
            }
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            print(f"  {m.upper():<8}: Delta={delta:+.4f} | t-test p={p_val:.4f} ({sig}) | Wilcoxon p={w_pval:.4f}")

    stage2_result = {
        'dataset': dataset_name,
        'champion_circuit': champion_circuit,
        'num_seeds': 10,
        'epochs': epochs,
        'summary': summary,
        'statistical_comparison_vs_classical': stat_report,
        'raw_seed_metrics': champion_seed_metrics
    }
    return stage2_result
