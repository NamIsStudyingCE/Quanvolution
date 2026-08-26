import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np
import json
from datetime import datetime

from data.medmnist_loader import prepare_data
from data.precompute_features import run_precompute
from models.classical_cnn import SymmetricalMinimumCNN
from models.quantum_model import QuanvolutionClassifier
from utils.metrics import calculate_metrics


def set_seed(seed: int):
    """
    Fix all random sources for full reproducibility on GPU + CPU.
    Must be called before every model initialization and DataLoader creation.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)      # Fix Bug #1: GPU RNG
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True   # Fix Bug #1: deterministic CUDA ops
    torch.backends.cudnn.benchmark = False      # Fix Bug #1: disable auto-tuning


def make_loaders(classical_train, classical_val, classical_test,
                 q_train_features, q_train_labels,
                 q_val_features, q_val_labels,
                 q_test_features, q_test_labels,
                 batch_size, seed):
    """
    Fix Bug #2: Create fresh DataLoaders with per-seed generator for each seed.
    This ensures each seed gets its own independent shuffle sequence,
    making results independent of training order and fully reproducible.
    """
    g = torch.Generator()
    g.manual_seed(seed)

    c_train_loader = DataLoader(
        classical_train, batch_size=batch_size, shuffle=True,
        generator=g, num_workers=0
    )
    c_val_loader  = DataLoader(classical_val,  batch_size=batch_size, shuffle=False, num_workers=0)
    c_test_loader = DataLoader(classical_test, batch_size=batch_size, shuffle=False, num_workers=0)

    # Quantum loaders use a separate generator with the same seed for symmetry
    qg = torch.Generator()
    qg.manual_seed(seed)

    q_train_loader = DataLoader(
        TensorDataset(q_train_features, q_train_labels),
        batch_size=batch_size, shuffle=True,
        generator=qg, num_workers=0
    )
    q_val_loader  = DataLoader(TensorDataset(q_val_features,  q_val_labels),  batch_size=batch_size, shuffle=False, num_workers=0)
    q_test_loader = DataLoader(TensorDataset(q_test_features, q_test_labels), batch_size=batch_size, shuffle=False, num_workers=0)

    return c_train_loader, c_val_loader, c_test_loader, q_train_loader, q_val_loader, q_test_loader


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
    all_targets = []
    all_probs = []
    all_preds = []

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
    metrics = calculate_metrics(
        np.array(all_targets),
        np.array(all_probs),
        np.array(all_preds),
        num_classes
    )
    return loss, metrics


def train_model(model_name, train_loader, val_loader, test_loader, num_classes, num_epochs, seed):
    """
    Initialize model with fixed seed, train for num_epochs, return best test metrics
    (selected by best Val ROC-AUC across epochs).
    """
    set_seed(seed)  # Fix Bug #1 + #2: seed before model init and loader use

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_name == "classical":
        model = SymmetricalMinimumCNN(num_classes=num_classes).to(device)
    else:
        model = QuanvolutionClassifier(num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_val_auc = -1
    best_test_metrics = None

    for epoch in range(num_epochs):
        train_epoch(model, train_loader, criterion, optimizer, device)
        _, val_metrics = evaluate(model, val_loader, criterion, device, num_classes)

        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            _, best_test_metrics = evaluate(model, test_loader, criterion, device, num_classes)

    return best_test_metrics


def print_comparison_table(results, dataset_name):
    metrics = list(results['classical'][0].keys())

    print("\n" + "="*68)
    print(f" FINAL BENCHMARK SUMMARY: {dataset_name.upper()} (10-SEED EVALUATION)")
    print("="*68)
    print(f"{'Metric':<12} | {'Classical (Mean +- Std)':<24} | {'Quantum (Mean +- Std)':<24}")
    print("-" * 68)

    summary_data = {'classical': {}, 'quantum': {}}
    for m in metrics:
        c_vals = [r[m] for r in results['classical']]
        q_vals = [r[m] for r in results['quantum']]

        c_str = f"{np.mean(c_vals):.4f} +- {np.std(c_vals):.4f}"
        q_str = f"{np.mean(q_vals):.4f} +- {np.std(q_vals):.4f}"

        summary_data['classical'][m] = c_str
        summary_data['quantum'][m] = q_str

        print(f"{m.upper():<12} | {c_str:<24} | {q_str:<24}")
    print("="*68 + "\n")
    return summary_data


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',     type=str, default='breastmnist', choices=['breastmnist', 'octmnist'])
    parser.add_argument('--max_samples', type=int, default=5000, help="Max samples for OCTMNIST")
    parser.add_argument('--epochs',      type=int, default=30)
    parser.add_argument('--smoke_test',  action='store_true', help="Run 1 seed, 1 epoch for quick validation")
    args = parser.parse_args()

    dataset_name = args.dataset

    # ── Step 1: Deterministic data splits ─────────────────────────────────────
    print("\n>>> [Step 1/3] Preparing Deterministic Data Splits...")
    max_train, max_val, max_test = None, None, None
    if dataset_name == 'octmnist':
        max_train = int(args.max_samples * 0.7)
        max_val   = int(args.max_samples * 0.1)
        max_test  = int(args.max_samples * 0.2)

    splits = prepare_data(
        dataset_name=dataset_name,
        max_train=max_train,
        max_val=max_val,
        max_test=max_test,
        seed=42
    )
    num_classes = splits['num_classes']

    # ── Step 2: Precompute quantum features (once, cached) ───────────────────
    print("\n>>> [Step 2/3] Precomputing Quantum Features...")
    run_precompute(dataset_name=dataset_name, data_splits=splits)

    # ── Load precomputed quantum feature tensors ──────────────────────────────
    q_train_features, q_train_labels = torch.load(f"data/quantum_features/{dataset_name}_train.pt", weights_only=True)
    q_val_features,   q_val_labels   = torch.load(f"data/quantum_features/{dataset_name}_val.pt",   weights_only=True)
    q_test_features,  q_test_labels  = torch.load(f"data/quantum_features/{dataset_name}_test.pt",  weights_only=True)

    # ── Step 3: Train 10 seeds ────────────────────────────────────────────────
    print("\n>>> [Step 3/3] Training 10 Fixed Seeds...")
    seeds  = [0, 42, 100, 2023, 777, 999, 1234, 5678, 1111, 2222]
    epochs = args.epochs

    if args.smoke_test:
        print("!!! SMOKE TEST MODE: 1 seed, 1 epoch !!!")
        seeds  = [42]
        epochs = 1

    batch_size = 64
    results = {'classical': [], 'quantum': []}

    pbar = tqdm(seeds, desc=f"10-Seed Progress ({dataset_name})", unit="seed")
    for seed in pbar:
        # Fix Bug #2: fresh DataLoaders with per-seed generator on every iteration
        (c_train_loader, c_val_loader, c_test_loader,
         q_train_loader, q_val_loader, q_test_loader) = make_loaders(
            splits['classical_train'], splits['classical_val'], splits['classical_test'],
            q_train_features, q_train_labels,
            q_val_features,   q_val_labels,
            q_test_features,  q_test_labels,
            batch_size, seed
        )

        c_metrics = train_model("classical", c_train_loader, c_val_loader, c_test_loader, num_classes, epochs, seed)
        results['classical'].append(c_metrics)

        q_metrics = train_model("quantum",   q_train_loader, q_val_loader, q_test_loader, num_classes, epochs, seed)
        results['quantum'].append(q_metrics)

        pbar.set_postfix({
            'Seed':  seed,
            'C-AUC': f"{c_metrics['auc']:.3f}",
            'Q-AUC': f"{q_metrics['auc']:.3f}"
        })

    # ── Summarize & save ──────────────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_data = print_comparison_table(results, dataset_name)

    for model_type in ['classical', 'quantum']:
        summary = {
            "dataset":         dataset_name,
            "model":           model_type,
            "timestamp":       timestamp,
            "num_seeds":       len(seeds),
            "epochs":          epochs,          # Fix Bug #4: use runtime epochs, not args.epochs
            "metrics_summary": summary_data[model_type],
            "raw_seed_metrics": results[model_type]
        }
        with open(f"results/{dataset_name}_{model_type}_latest.json", "w") as f:
            json.dump(summary, f, indent=4)
        with open(f"results/{dataset_name}_{model_type}_{timestamp}.json", "w") as f:
            json.dump(summary, f, indent=4)

    print(f"Logged results to 'results/' ({dataset_name}_latest.json & timestamped)")


if __name__ == "__main__":
    main()
