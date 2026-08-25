import os
import json
import matplotlib.pyplot as plt
import numpy as np

def plot_metric_comparison(dataset_name, save_dir="results/figures"):
    os.makedirs(save_dir, exist_ok=True)
    
    with open(f"results/{dataset_name}_classical_latest.json", "r") as f:
        c_data = json.load(f)
    with open(f"results/{dataset_name}_quantum_latest.json", "r") as f:
        q_data = json.load(f)
        
    metrics = ['acc', 'bacc', 'f1', 'mcc', 'auc', 'pr_auc']
    metric_labels = ['Accuracy', 'Balanced Acc', 'F1-Score', 'MCC', 'ROC-AUC', 'PR-AUC']
    
    c_means, c_stds = [], []
    q_means, q_stds = [], []
    
    for m in metrics:
        c_vals = [r[m] for r in c_data['raw_seed_metrics']]
        q_vals = [r[m] for r in q_data['raw_seed_metrics']]
        c_means.append(np.mean(c_vals))
        c_stds.append(np.std(c_vals))
        q_means.append(np.mean(q_vals))
        q_stds.append(np.std(q_vals))
        
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, c_means, width, yerr=c_stds, capsize=5, label='Classical CNN', color='#4C72B0', alpha=0.9)
    plt.bar(x + width/2, q_means, width, yerr=q_stds, capsize=5, label='Quanvolution (Quantum)', color='#55A868', alpha=0.9)
    
    plt.ylabel('Score (0 - 1.0)', fontsize=12, fontweight='bold')
    plt.title(f'10-Seed Benchmark Comparison: {dataset_name.upper()} (Mean +- Std)', fontsize=14, fontweight='bold', pad=15)
    plt.xticks(x, metric_labels, fontsize=11, fontweight='semibold')
    plt.ylim(0, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.legend(fontsize=11, loc='lower right' if dataset_name == 'breastmnist' else 'upper right')
    plt.tight_layout()
    
    out_path = os.path.join(save_dir, f"{dataset_name}_benchmark_chart.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved benchmark figure to {out_path}")

def main():
    plot_metric_comparison("breastmnist")
    plot_metric_comparison("octmnist")

if __name__ == "__main__":
    main()
