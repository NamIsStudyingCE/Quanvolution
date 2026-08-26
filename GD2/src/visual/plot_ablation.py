import os
import sys
import json
import matplotlib.pyplot as plt
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

CIRCUIT_DISPLAY_NAMES = {
    'random_L1': 'Random (L=1)',
    'random_L2': 'Random (L=2)',
    'strongly_L1': 'Strongly Ent. (L=1)',
    'strongly_L2': 'Strongly Ent. (L=2)',
    'basic_L1': 'Basic Ent. (L=1)',
    'basic_L2': 'Basic Ent. (L=2)'
}

COLORS = ['#4C72B0', '#55A868', '#C44E52', '#8172B3', '#CCB974', '#64B5CD']

def plot_circuit_ablation_chart(dataset_name, stage1_results, save_dir="results/figures"):
    os.makedirs(save_dir, exist_ok=True)
    
    circuits = list(stage1_results.keys())
    labels = [CIRCUIT_DISPLAY_NAMES.get(c, c) for c in circuits]
    
    metrics = ['acc', 'bacc', 'f1', 'mcc', 'auc', 'pr_auc']
    metric_titles = ['Accuracy', 'Balanced Acc', 'F1-Score', 'MCC', 'ROC-AUC', 'PR-AUC']
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()
    
    x = np.arange(len(circuits))
    
    for idx, (m, title) in enumerate(zip(metrics, metric_titles)):
        ax = axes[idx]
        means = [stage1_results[c]['summary'][m]['mean'] for c in circuits]
        stds  = [stage1_results[c]['summary'][m]['std'] for c in circuits]
        
        bars = ax.bar(x, means, yerr=stds, capsize=5, color=COLORS, alpha=0.85, edgecolor='black', linewidth=0.8)
        
        # Highlight best bar with a star
        best_idx = int(np.argmax(means))
        bars[best_idx].set_edgecolor('red')
        bars[best_idx].set_linewidth(2.0)
        ax.text(best_idx, means[best_idx] + stds[best_idx] + 0.02, 'Best', ha='center', va='bottom', color='red', fontweight='bold', fontsize=9)
        
        ax.set_title(title, fontsize=12, fontweight='bold', pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=9)
        ax.set_ylim(0, max(1.05, max(means) + 0.15))
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        
    plt.suptitle(f"Quantum Circuit Architecture Ablation Study: {dataset_name.upper()}\n(Ansatz Types & Depths Comparison across 5 Seeds)",
                 fontsize=15, fontweight='bold', y=0.99)
    plt.tight_layout()
    
    out_file = os.path.join(save_dir, f"circuit_ablation_{dataset_name}.png")
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] Circuit ablation figure -> {out_file}")
