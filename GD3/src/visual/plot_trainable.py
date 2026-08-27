import os
import sys
import matplotlib.pyplot as plt
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

MODEL_LABELS = {
    'classical_cnn':   'Classical CNN',
    'fixed_quanv':     'Fixed Random Quanv',
    'trainable_quanv': 'Trainable Quanv (Ours)'
}

MODEL_COLORS = {
    'classical_cnn':   '#4C72B0', # Blue
    'fixed_quanv':     '#C44E52', # Red
    'trainable_quanv': '#55A868'  # Green
}

def plot_training_curves(results, save_dir="results/figures"):
    os.makedirs(save_dir, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = len(results['classical_cnn']['histories'][0]['train_loss'])
    epoch_axis = np.arange(1, epochs + 1)
    
    # 1. Plot Loss curves (mean across seeds)
    for m_key, label in MODEL_LABELS.items():
        color = MODEL_COLORS[m_key]
        train_losses = [h['train_loss'] for h in results[m_key]['histories']]
        val_losses   = [h['val_loss']   for h in results[m_key]['histories']]
        
        mean_tr_loss = np.mean(train_losses, axis=0)
        mean_va_loss = np.mean(val_losses, axis=0)
        
        ax1.plot(epoch_axis, mean_tr_loss, label=f"{label} (Train)", color=color, linestyle='--', alpha=0.7)
        ax1.plot(epoch_axis, mean_va_loss, label=f"{label} (Val)",   color=color, linewidth=2.0)
        
    ax1.set_title("Training & Validation Loss Dynamics", fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlabel("Epochs", fontsize=10, fontweight='semibold')
    ax1.set_ylabel("CrossEntropy Loss", fontsize=10, fontweight='semibold')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(fontsize=9)
    
    # 2. Plot Val ROC-AUC curves
    for m_key, label in MODEL_LABELS.items():
        color = MODEL_COLORS[m_key]
        val_aucs = [h['val_auc'] for h in results[m_key]['histories']]
        mean_va_auc = np.mean(val_aucs, axis=0)
        std_va_auc  = np.std(val_aucs, axis=0)
        
        ax2.plot(epoch_axis, mean_va_auc, label=label, color=color, linewidth=2.2)
        ax2.fill_between(epoch_axis, mean_va_auc - std_va_auc, mean_va_auc + std_va_auc, color=color, alpha=0.15)
        
    ax2.set_title("Validation ROC-AUC Dynamics", fontsize=12, fontweight='bold', pad=10)
    ax2.set_xlabel("Epochs", fontsize=10, fontweight='semibold')
    ax2.set_ylabel("ROC-AUC Score", fontsize=10, fontweight='semibold')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(fontsize=10, loc='lower right')
    
    plt.suptitle("Proof-of-Concept: Trainable Quanvolution vs Fixed Quanvolution vs Classical CNN\n(OCTMNIST 500-Sample Subset Dynamics)", 
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    out_curves = os.path.join(save_dir, "trainable_poc_curves.png")
    plt.savefig(out_curves, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] Training curves figure -> {out_curves}")

def plot_metrics_bar_comparison(results, save_dir="results/figures"):
    os.makedirs(save_dir, exist_ok=True)
    
    metrics = ['acc', 'bacc', 'f1', 'mcc', 'auc', 'pr_auc']
    metric_labels = ['Accuracy', 'Balanced Acc', 'F1-Score', 'MCC', 'ROC-AUC', 'PR-AUC']
    
    models = list(MODEL_LABELS.keys())
    n_models = len(models)
    
    x = np.arange(len(metrics))
    width = 0.25
    
    plt.figure(figsize=(12, 6))
    
    for i, m_key in enumerate(models):
        label = MODEL_LABELS[m_key]
        color = MODEL_COLORS[m_key]
        raw_m = results[m_key]['test_metrics']
        
        means = [np.mean([r[m] for r in raw_m]) for m in metrics]
        stds  = [np.std([r[m]  for r in raw_m]) for m in metrics]
        
        offset = (i - 1) * width
        plt.bar(x + offset, means, width, yerr=stds, capsize=4, label=label, color=color, alpha=0.9, edgecolor='black', linewidth=0.8)
        
    plt.ylabel('Score (0 - 1.0)', fontsize=11, fontweight='bold')
    plt.title('Final Test Performance Comparison: Trainable vs Fixed vs Classical\n(OCTMNIST 500-Sample Subset)', fontsize=13, fontweight='bold', pad=12)
    plt.xticks(x, metric_labels, fontsize=10, fontweight='semibold')
    plt.ylim(0, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.legend(fontsize=10, loc='upper right')
    plt.tight_layout()
    
    out_bars = os.path.join(save_dir, "trainable_poc_metrics.png")
    plt.savefig(out_bars, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] Final metrics comparison figure -> {out_bars}")
