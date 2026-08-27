import os
import matplotlib.pyplot as plt
import numpy as np

# Use high-quality styling
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8

def plot_gd3_training_curves(results, dataset_name, save_path='results/figures/gd3_curves.png'):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    models = ['classical_cnn', 'fixed_quanv', 'trainable_quanv']
    labels = ['Classical CNN', 'Fixed Quanvolution', 'Trainable Quanv (Ours)']
    colors = ['#1f77b4', '#d62728', '#2ca02c']

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    # 1. Loss Dynamics
    ax1 = axes[0]
    for m, label, col in zip(models, labels, colors):
        hists = results[m]['histories']
        train_losses = np.array([h['train_loss'] for h in hists])
        val_losses = np.array([h['val_loss'] for h in hists])
        epochs = np.arange(1, train_losses.shape[1] + 1)

        # Plot Train (dashed) and Val (solid)
        ax1.plot(epochs, np.mean(train_losses, axis=0), linestyle='--', color=col, alpha=0.5, label=f'{label} (Train)')
        ax1.plot(epochs, np.mean(val_losses, axis=0), linestyle='-', color=col, linewidth=2.2, label=f'{label} (Val)')
        ax1.fill_between(epochs,
                         np.mean(val_losses, axis=0) - np.std(val_losses, axis=0),
                         np.mean(val_losses, axis=0) + np.std(val_losses, axis=0),
                         color=col, alpha=0.15)

    ax1.set_title(f'Training & Validation Loss Dynamics ({dataset_name})', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Cross-Entropy Loss', fontsize=11)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper right', framealpha=0.9, fontsize=9)

    # 2. Validation AUC Dynamics
    ax2 = axes[1]
    for m, label, col in zip(models, labels, colors):
        hists = results[m]['histories']
        val_aucs = np.array([h['val_auc'] for h in hists])
        epochs = np.arange(1, val_aucs.shape[1] + 1)

        mean_auc = np.mean(val_aucs, axis=0)
        std_auc = np.std(val_aucs, axis=0)

        ax2.plot(epochs, mean_auc, color=col, linewidth=2.2, label=f'{label} (Final: {mean_auc[-1]:.4f})')
        ax2.fill_between(epochs, mean_auc - std_auc, mean_auc + std_auc, color=col, alpha=0.15)

    ax2.set_title(f'Validation ROC-AUC Dynamics ({dataset_name})', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('ROC-AUC', fontsize=11)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='lower right', framealpha=0.9, fontsize=9)

    plt.suptitle(f'Quantum vs Classical Training Dynamics Benchmark: {dataset_name}', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f'Saved training curves to: {save_path}')

def plot_gd3_theta_trajectories(results, save_path='results/figures/gd3_theta_trajectories.png'):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    hists = results['trainable_quanv']['histories']
    
    fig, axes = plt.subplots(1, min(len(hists), 3), figsize=(5 * min(len(hists), 3), 4.5), dpi=300, squeeze=False)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

    for s_idx in range(min(len(hists), 3)):
        ax = axes[0, s_idx]
        thetas = np.array(hists[s_idx]['theta_trajectories']) # Shape: (epochs, n_layers, n_qubits) or (epochs, n_params)
        thetas = thetas.reshape(thetas.shape[0], -1)
        epochs = np.arange(1, thetas.shape[0] + 1)

        for p_idx in range(thetas.shape[1]):
            col = colors[p_idx % len(colors)]
            ax.plot(epochs, thetas[:, p_idx], label=f'$\theta_{{{p_idx+1}}}$', color=col, linewidth=1.8)

        ax.set_title(f'Seed {s_idx+1} Trajectory', fontsize=11, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel('Angle $\theta$ (radians)', fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='best', fontsize=8)

    plt.suptitle('Quantum Variational Angles ($\theta$) Trajectory Convergence over Epochs', fontsize=13, fontweight='bold', y=1.03)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f'Saved theta trajectories to: {save_path}')

def plot_gd3_gradient_norms(results, save_path='results/figures/gd3_gradient_norms.png'):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    hists = results['trainable_quanv']['histories']
    grad_norms = np.array([h['grad_norms'] for h in hists])
    epochs = np.arange(1, grad_norms.shape[1] + 1)

    mean_gn = np.mean(grad_norms, axis=0)
    std_gn = np.std(grad_norms, axis=0)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    ax.plot(epochs, mean_gn, color='#2ca02c', linewidth=2.2, label=r'Mean $||\nabla_\theta \mathcal{L}||_2$')
    ax.fill_between(epochs, np.maximum(0, mean_gn - std_gn), mean_gn + std_gn, color='#2ca02c', alpha=0.2, label=r'$\pm 1$ Std Dev')

    for s_idx, gn in enumerate(grad_norms):
        ax.plot(epochs, gn, linestyle=':', alpha=0.4, label=f'Seed {s_idx+1}' if s_idx < 3 else None)

    ax.set_title('Quantum Parameter Gradient Norm Dynamics (Proof of Trainability)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel(r'Gradient L2 Norm $||\nabla_\theta \mathcal{L}||_2$', fontsize=11)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper right', framealpha=0.9, fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f'Saved gradient norm dynamics to: {save_path}')

def plot_gd3_benchmark_summary(results, dataset_name, save_path='results/figures/gd3_benchmark.png'):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    metrics = ['acc', 'bacc', 'f1', 'mcc', 'auc', 'pr_auc']
    metric_titles = ['Accuracy', 'Balanced Acc', 'F1-Score', 'MCC', 'ROC-AUC', 'PR-AUC']
    models = ['classical_cnn', 'fixed_quanv', 'trainable_quanv']
    labels = ['Classical CNN', 'Fixed Quanv', 'Trainable Quanv']
    colors = ['#4A90E2', '#E74C3C', '#2ECC71']

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), dpi=300)
    axes = axes.flatten()

    for idx, (m, title) in enumerate(zip(metrics, metric_titles)):
        ax = axes[idx]
        means = [np.mean([r[m] for r in results[model]['test_metrics']]) for model in models]
        stds = [np.std([r[m] for r in results[model]['test_metrics']]) for model in models]

        bars = ax.bar(labels, means, yerr=stds, capsize=5, color=colors, edgecolor='black', alpha=0.85)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_ylim(0, max(1.0, max(means) + 0.1))
        ax.grid(axis='y', linestyle=':', alpha=0.6)

        best_idx = np.argmax(means)
        bars[best_idx].set_edgecolor('red')
        bars[best_idx].set_linewidth(2)
        ax.text(best_idx, means[best_idx] + stds[best_idx] + 0.02, 'Best', ha='center', color='red', fontweight='bold', fontsize=9)

    plt.suptitle(f'Comprehensive 6-Metric Benchmark: {dataset_name}', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f'Saved benchmark summary to: {save_path}')
