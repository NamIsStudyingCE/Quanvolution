# -*- coding: utf-8 -*-
"""
plot_gd3_dynamics.py
--------------------
Visualization module for Phase 3 training dynamics.
Updated with legends positioned cleanly OUTSIDE plot areas to prevent any overlapping.
"""

import os
import matplotlib.pyplot as plt
import numpy as np

# Use high-quality styling
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8

def plot_gd3_training_curves(results, dataset_name, save_path='results/figures/gd3_curves.png'):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    models = list(results.keys())
    palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    
    name_map = {
        'classical_cnn': 'Classical CNN',
        'fixed_basic': 'Fixed Basic',
        'trainable_basic': 'Trainable Basic',
        'fixed_champion_gd2': 'Fixed Champion (GĐ2)',
        'fixed_strongly': 'Fixed Strongly (Full-Exp)',
        'trainable_strongly': 'Trainable Strongly (Full-Exp)'
    }

    # Increased height to accommodate legends placed outside
    fig, axes = plt.subplots(1, 2, figsize=(17, 6.2), dpi=300)

    # 1. Loss Dynamics
    ax1 = axes[0]
    for idx, m in enumerate(models):
        label = name_map.get(m, m)
        col = palette[idx % len(palette)]
        hists = results[m]['histories']
        train_losses = np.array([h['train_loss'] for h in hists])
        val_losses = np.array([h['val_loss'] for h in hists])
        epochs = np.arange(1, train_losses.shape[1] + 1)

        ax1.plot(epochs, np.mean(train_losses, axis=0), linestyle='--', color=col, alpha=0.35)
        ax1.plot(epochs, np.mean(val_losses, axis=0), linestyle='-', color=col, linewidth=2.0, label=f'{label}')
        ax1.fill_between(epochs,
                         np.mean(val_losses, axis=0) - np.std(val_losses, axis=0),
                         np.mean(val_losses, axis=0) + np.std(val_losses, axis=0),
                         color=col, alpha=0.08)

    ax1.set_title(f'Training & Validation Loss ({dataset_name})', fontsize=12, fontweight='bold', pad=12)
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Cross-Entropy Loss', fontsize=11)
    ax1.grid(True, linestyle=':', alpha=0.6)
    # Legend placed below the plot area
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, framealpha=0.95, fontsize=8.5)

    # 2. Validation AUC Dynamics
    ax2 = axes[1]
    for idx, m in enumerate(models):
        label = name_map.get(m, m)
        col = palette[idx % len(palette)]
        hists = results[m]['histories']
        val_aucs = np.array([h['val_auc'] for h in hists])
        epochs = np.arange(1, val_aucs.shape[1] + 1)

        mean_auc = np.mean(val_aucs, axis=0)
        std_auc = np.std(val_aucs, axis=0)

        ax2.plot(epochs, mean_auc, color=col, linewidth=2.0, label=f'{label} ({mean_auc[-1]:.4f})')
        ax2.fill_between(epochs, mean_auc - std_auc, mean_auc + std_auc, color=col, alpha=0.08)

    ax2.set_title(f'Validation ROC-AUC ({dataset_name})', fontsize=12, fontweight='bold', pad=12)
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('ROC-AUC', fontsize=11)
    ax2.grid(True, linestyle=':', alpha=0.6)
    # Legend placed below the plot area
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, framealpha=0.95, fontsize=8.5)

    plt.suptitle(f'3-Tier Benchmark Learning Dynamics: {dataset_name}', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f'Saved training curves (external legends) to: {save_path}')

def plot_gd3_theta_trajectories(results, target_model='trainable_strongly', save_path='results/figures/gd3_theta_trajectories.png'):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if target_model not in results:
        target_model = [m for m in results if 'trainable' in m][0]
    hists = results[target_model]['histories']
    
    n_seeds = min(len(hists), 3)
    fig, axes = plt.subplots(1, n_seeds, figsize=(5.2 * n_seeds, 4.8), dpi=300, squeeze=False)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8', '#ffbb78']

    lines_for_legend = []
    labels_for_legend = []

    for s_idx in range(n_seeds):
        ax = axes[0, s_idx]
        thetas = np.array(hists[s_idx]['theta_trajectories'])
        thetas = thetas.reshape(thetas.shape[0], -1)
        epochs = np.arange(1, thetas.shape[0] + 1)

        for p_idx in range(min(thetas.shape[1], 12)):
            col = colors[p_idx % len(colors)]
            line, = ax.plot(epochs, thetas[:, p_idx], color=col, linewidth=1.6)
            if s_idx == 0:
                lines_for_legend.append(line)
                labels_for_legend.append(f'$\\theta_{{{p_idx+1}}}$')

        ax.set_title(f'Seed {s_idx+1} Parameter Trajectories', fontsize=11, fontweight='bold', pad=10)
        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel('Angle $\\theta$ (rad)', fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.6)

    # Place a single shared legend at the bottom outside all subplots
    fig.legend(lines_for_legend, labels_for_legend, loc='lower center', 
               bbox_to_anchor=(0.5, -0.06), ncol=min(12, len(labels_for_legend)), 
               fontsize=8.5, framealpha=0.95)

    plt.suptitle(f'Quantum Angles ($\\theta$) Trajectories ({target_model})', fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f'Saved theta trajectories (external legend) to: {save_path}')

def plot_gd3_gradient_norms(results, target_model='trainable_strongly', save_path='results/figures/gd3_gradient_norms.png'):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if target_model not in results:
        target_model = [m for m in results if 'trainable' in m][0]
    hists = results[target_model]['histories']
    grad_norms = np.array([h['grad_norms'] for h in hists])
    epochs = np.arange(1, grad_norms.shape[1] + 1)

    mean_gn = np.mean(grad_norms, axis=0)
    std_gn = np.std(grad_norms, axis=0)

    # Increased width to put legend on the right outside
    fig, ax = plt.subplots(figsize=(9.2, 5.0), dpi=300)
    ax.plot(epochs, mean_gn, color='#2ca02c', linewidth=2.2, label=r'Mean $||\nabla_\theta \mathcal{L}||_2$')
    ax.fill_between(epochs, np.maximum(0, mean_gn - std_gn), mean_gn + std_gn, color='#2ca02c', alpha=0.18, label=r'$\pm 1$ Std Dev')

    for s_idx, gn in enumerate(grad_norms):
        ax.plot(epochs, gn, linestyle=':', alpha=0.35, label=f'Seed {s_idx+1}' if s_idx < 4 else None)

    ax.set_title(f'Quantum Gradient Norm Dynamics ({target_model})', fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel(r'Gradient L2 Norm $||\nabla_\theta \mathcal{L}||_2$', fontsize=11)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # Legend placed outside to the right of the plot area
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', framealpha=0.95, fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f'Saved gradient norm dynamics (external legend) to: {save_path}')

def plot_gd3_benchmark_summary(results, dataset_name, save_path='results/figures/gd3_benchmark.png'):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    metrics = ['acc', 'bacc', 'f1', 'mcc', 'auc', 'pr_auc']
    metric_titles = ['Accuracy', 'Balanced Acc', 'F1-Score', 'MCC', 'ROC-AUC', 'PR-AUC']
    models = list(results.keys())
    
    name_map = {
        'classical_cnn': 'Classical CNN',
        'fixed_basic': 'Fixed Basic',
        'trainable_basic': 'Train Basic',
        'fixed_champion_gd2': 'Fixed Champ',
        'fixed_strongly': 'Fixed Strong',
        'trainable_strongly': 'Train Strong'
    }
    labels = [name_map.get(m, m) for m in models]
    palette = ['#4A90E2', '#E67E22', '#2ECC71', '#E74C3C', '#9B59B6', '#1ABC9C']

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), dpi=300)
    axes = axes.flatten()

    for idx, (m, title) in enumerate(zip(metrics, metric_titles)):
        ax = axes[idx]
        means = [np.mean([r[m] for r in results[model]['test_metrics']]) for model in models]
        stds = [np.std([r[m] for r in results[model]['test_metrics']], ddof=1) for model in models]

        bars = ax.bar(labels, means, yerr=stds, capsize=4, color=palette[:len(models)], edgecolor='black', alpha=0.85)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_ylim(0, max(1.0, max(means) + 0.12))
        ax.tick_params(axis='x', rotation=20, labelsize=8)
        ax.grid(axis='y', linestyle=':', alpha=0.6)

        best_idx = np.argmax(means)
        bars[best_idx].set_edgecolor('red')
        bars[best_idx].set_linewidth(2)
        ax.text(best_idx, means[best_idx] + stds[best_idx] + 0.02, 'Best', ha='center', color='red', fontweight='bold', fontsize=8)

    plt.suptitle(f'3-Tier Multi-Model Benchmark: {dataset_name}', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f'Saved benchmark summary to: {save_path}')
