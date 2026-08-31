# -*- coding: utf-8 -*-
"""
regenerate_figs_bigfont.py
--------------------------
Regenerates Fig3 (x2) and Fig4a-d with print-legible typography:
canvas size reduced to near-final display size and fonts scaled up,
so effective text size in the compiled PDF is ~2-3x larger than before.
Data source: the two JSON ground-truth files (unchanged, 10-seed data).
Output: D:\\KLTN_Paper\\figures\\ (repo originals untouched).
"""
import os
import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8

REPO = r'D:\KLTN_Paper\..\KhoaLuanTotNghiep'
REPO = os.path.abspath(REPO)
OUT = r'D:\KLTN_Paper\figures'
os.makedirs(OUT, exist_ok=True)

breast = json.load(open(os.path.join(REPO, 'results', 'full_trainable_breastmnist.json'), encoding='utf-8'))['raw_results']
octj = json.load(open(os.path.join(REPO, 'results', 'full_trainable_octmnist.json'), encoding='utf-8'))['raw_results']

NAME_MAP_BARS = {
    'classical_cnn': 'Classical CNN',
    'fixed_basic': 'Fixed Basic',
    'trainable_basic': 'Train Basic',
    'fixed_champion_gd2': 'Fixed Champ',
    'fixed_strongly': 'Fixed Strong',
    'trainable_strongly': 'Train Strong',
}
NAME_MAP_CURVES = {
    'classical_cnn': 'Classical CNN',
    'fixed_basic': 'Fixed Basic',
    'trainable_basic': 'Trainable Basic',
    'fixed_champion_gd2': 'Fixed Champion (GD2)',
    'fixed_strongly': 'Fixed Strongly (Full-Exp)',
    'trainable_strongly': 'Trainable Strongly (Full-Exp)',
}
PALETTE_BARS = ['#4A90E2', '#E67E22', '#2ECC71', '#E74C3C', '#9B59B6', '#1ABC9C']
PALETTE_LINES = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']


def benchmark_summary(results, dataset_name, save_path):
    metrics = ['acc', 'bacc', 'f1', 'mcc', 'auc', 'pr_auc']
    titles = ['Accuracy', 'Balanced Acc', 'F1-Score', 'MCC', 'ROC-AUC', 'PR-AUC']
    models = list(results.keys())
    labels = [NAME_MAP_BARS.get(m, m) for m in models]

    fig, axes = plt.subplots(2, 3, figsize=(9.0, 4.6), dpi=300)
    axes = axes.flatten()

    for idx, (m, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx]
        means = [np.mean([r[m] for r in results[model]['test_metrics']]) for model in models]
        stds = [np.std([r[m] for r in results[model]['test_metrics']], ddof=1) for model in models]
        bars = ax.bar(range(len(models)), means, yerr=stds, capsize=3,
                      color=PALETTE_BARS[:len(models)], edgecolor='black', alpha=0.85)
        ax.set_title(title, fontsize=15, fontweight='bold')
        ax.set_ylim(0, max(1.0, max(means) + 0.14))
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([])
        ax.tick_params(axis='y', labelsize=11)
        ax.grid(axis='y', linestyle=':', alpha=0.6)
        best_idx = int(np.argmax(means))
        bars[best_idx].set_edgecolor('red')
        bars[best_idx].set_linewidth(2.2)
        ax.text(best_idx, means[best_idx] + stds[best_idx] + 0.03, 'Best', ha='center',
                color='red', fontweight='bold', fontsize=12)

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=PALETTE_BARS[i], edgecolor='black')
               for i in range(len(models))]
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.005),
               ncol=min(len(models), 3), fontsize=12, framealpha=0.95)
    plt.suptitle(f'3-Tier Multi-Model Benchmark: {dataset_name}', fontsize=17, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0.09, 1, 0.96])
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f'saved {save_path}')


def training_curves(results, dataset_name, save_path):
    models = list(results.keys())
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2), dpi=300)

    ax1 = axes[0]
    for idx, m in enumerate(models):
        label = NAME_MAP_CURVES.get(m, m)
        col = PALETTE_LINES[idx % len(PALETTE_LINES)]
        hists = results[m]['histories']
        tr = np.array([h['train_loss'] for h in hists])
        vl = np.array([h['val_loss'] for h in hists])
        ep = np.arange(1, tr.shape[1] + 1)
        ax1.plot(ep, np.mean(tr, axis=0), linestyle='--', color=col, alpha=0.35)
        ax1.plot(ep, np.mean(vl, axis=0), linestyle='-', color=col, linewidth=1.8, label=label)
        ax1.fill_between(ep, np.mean(vl, axis=0) - np.std(vl, axis=0),
                         np.mean(vl, axis=0) + np.std(vl, axis=0), color=col, alpha=0.08)
    ax1.set_title(f'Training & Validation Loss ({dataset_name})', fontsize=11.5, fontweight='bold', pad=6)
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Cross-Entropy Loss', fontsize=11)
    ax1.tick_params(labelsize=9.5)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper right', framealpha=0.9, fontsize=7.8, borderaxespad=0.4)

    ax2 = axes[1]
    for idx, m in enumerate(models):
        label = NAME_MAP_CURVES.get(m, m)
        col = PALETTE_LINES[idx % len(PALETTE_LINES)]
        hists = results[m]['histories']
        va = np.array([h['val_auc'] for h in hists])
        ep = np.arange(1, va.shape[1] + 1)
        mean_auc, std_auc = np.mean(va, axis=0), np.std(va, axis=0)
        ax2.plot(ep, mean_auc, color=col, linewidth=1.8, label=f'{label} ({mean_auc[-1]:.4f})')
        ax2.fill_between(ep, mean_auc - std_auc, mean_auc + std_auc, color=col, alpha=0.08)
    ax2.set_title(f'Validation ROC-AUC ({dataset_name})', fontsize=11.5, fontweight='bold', pad=6)
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('ROC-AUC', fontsize=11)
    ax2.tick_params(labelsize=9.5)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='lower right', framealpha=0.9, fontsize=7.2, borderaxespad=0.4)

    plt.suptitle(f'3-Tier Benchmark Learning Dynamics: {dataset_name}', fontsize=13.5, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f'saved {save_path}')


def theta_trajectories(results, target_model, save_path):
    hists = results[target_model]['histories']
    n_seeds = min(len(hists), 3)
    fig, axes = plt.subplots(1, n_seeds, figsize=(7.2, 2.7), dpi=300, squeeze=False)
    colors = PALETTE_LINES + ['#bcbd22', '#17becf', '#aec7e8', '#ffbb78']

    lines_for_legend, labels_for_legend = [], []
    for s_idx in range(n_seeds):
        ax = axes[0, s_idx]
        thetas = np.array(hists[s_idx]['theta_trajectories']).reshape(
            np.array(hists[s_idx]['theta_trajectories']).shape[0], -1)
        ep = np.arange(1, thetas.shape[0] + 1)
        for p_idx in range(min(thetas.shape[1], 12)):
            line, = ax.plot(ep, thetas[:, p_idx], color=colors[p_idx % len(colors)], linewidth=1.4)
            if s_idx == 0:
                lines_for_legend.append(line)
                labels_for_legend.append(f'$\\theta_{{{p_idx+1}}}$')
        ax.set_title(f'Seed {s_idx+1}', fontsize=12.5, fontweight='bold', pad=7)
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel(r'Angle $\theta$ (rad)', fontsize=11)
        ax.tick_params(labelsize=9.5)
        ax.grid(True, linestyle=':', alpha=0.6)

    fig.legend(lines_for_legend, labels_for_legend, loc='lower center',
               bbox_to_anchor=(0.5, -0.05), ncol=6, fontsize=10, framealpha=0.95)
    plt.suptitle(rf'Quantum Angles ($\theta$) Trajectories ({target_model})', fontsize=13.5, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0.10, 1, 0.94])
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f'saved {save_path}')


def gradient_norms(results, target_model, save_path):
    hists = results[target_model]['histories']
    grad_norms = np.array([h['grad_norms'] for h in hists])
    ep = np.arange(1, grad_norms.shape[1] + 1)
    mean_gn, std_gn = np.mean(grad_norms, axis=0), np.std(grad_norms, axis=0)

    fig, ax = plt.subplots(figsize=(5.2, 3.1), dpi=300)
    ax.plot(ep, mean_gn, color='#2ca02c', linewidth=2.0, label=r'Mean $||\nabla_\theta \mathcal{L}||_2$')
    ax.fill_between(ep, np.maximum(0, mean_gn - std_gn), mean_gn + std_gn, color='#2ca02c',
                    alpha=0.18, label=r'$\pm 1$ Std Dev')
    for s_idx, gn in enumerate(grad_norms):
        ax.plot(ep, gn, linestyle=':', alpha=0.35, label=f'Seed {s_idx+1}' if s_idx < 4 else None)
    ax.set_title(f'Quantum Gradient Norm Dynamics ({target_model})', fontsize=12, fontweight='bold', pad=8)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel(r'Gradient L2 Norm $||\nabla_\theta \mathcal{L}||_2$', fontsize=11)
    ax.tick_params(labelsize=10)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', framealpha=0.95, fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f'saved {save_path}')


if __name__ == '__main__':
    benchmark_summary(breast, 'BreastMNIST', os.path.join(OUT, 'Fig3_breastmnist_benchmark.png'))
    benchmark_summary(octj, 'OCTMNIST', os.path.join(OUT, 'Fig3_octmnist_benchmark.png'))
    training_curves(breast, 'BreastMNIST', os.path.join(OUT, 'Fig4a_breastmnist_curves.png'))
    training_curves(octj, 'OCTMNIST', os.path.join(OUT, 'Fig4b_octmnist_curves.png'))
    theta_trajectories(breast, 'trainable_strongly', os.path.join(OUT, 'Fig4c_theta_trajectories.png'))
    gradient_norms(breast, 'trainable_strongly', os.path.join(OUT, 'Fig4d_gradient_norms.png'))
    print('done')
