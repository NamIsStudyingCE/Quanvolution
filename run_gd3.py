import os
import sys
import time
import json
import numpy as np
from datetime import datetime

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Ensure src is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from data.medmnist_loader import prepare_data
from experiments.trainable_experiment import run_full_dataset_experiment
from visual.plot_gd3_dynamics import (
    plot_gd3_training_curves,
    plot_gd3_theta_trajectories,
    plot_gd3_gradient_norms,
    plot_gd3_benchmark_summary
)

def print_dataset_benchmark_table(dataset_title, results, stat_report):
    metrics = ['acc', 'bacc', 'f1', 'mcc', 'auc', 'pr_auc']
    metric_names = ['Accuracy', 'Balanced Acc', 'F1-Score', 'MCC', 'ROC-AUC', 'PR-AUC']
    
    print('\n' + '='*95)
    print(f' BENCHMARK RESULTS: {dataset_title.upper()}')
    print('='*95)
    print(f"{'Metric':<14} | {'Classical CNN':<20} | {'Fixed Quanvolution':<20} | {'Trainable Quanv':<20} | {'p-value (Trainable vs Fixed)':<20}")
    print('-' * 95)
    
    for m, m_name in zip(metrics, metric_names):
        c_vals = [r[m] for r in results['classical_cnn']['test_metrics']]
        f_vals = [r[m] for r in results['fixed_quanv']['test_metrics']]
        t_vals = [r[m] for r in results['trainable_quanv']['test_metrics']]
        
        c_str = f"{np.mean(c_vals):.4f} +- {np.std(c_vals):.4f}"
        f_str = f"{np.mean(f_vals):.4f} +- {np.std(f_vals):.4f}"
        t_str = f"{np.mean(t_vals):.4f} +- {np.std(t_vals):.4f}"
        
        s = stat_report['trainable_vs_fixed'][m]
        p_val = s['p_value_ttest']
        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
        p_str = f"p={p_val:.4f} ({sig})"
        
        print(f"{m_name:<14} | {c_str:<20} | {f_str:<20} | {t_str:<20} | {p_str:<20}")
    print('='*95)

def main():
    print('================================================================================')
    print('          PHASE 3 MASTER EXECUTION: FULL-SCALE TRAINABLE QUANVOLUTION           ')
    print(f'          Start Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}          ')
    print('          Backend: Differentiable Analytical Backprop (default.qubit)           ')
    print('================================================================================')
    
    total_start = time.time()
    
    # ---------------------------------------------------------
    # PART 1: BREASTMNIST (780 samples, 10 seeds, L=2 Champion)
    # ---------------------------------------------------------
    print('\n' + '#' * 80)
    print(' [PART 1/2]: FULL BREASTMNIST EXPERIMENT (780 SAMPLES, 10 SEEDS, L=2)')
    print('#' * 80)
    
    breast_splits = prepare_data(
        dataset_name="breastmnist",
        seed=42
    )
    
    breast_seeds = [0, 42, 100, 2023, 777, 999, 1234, 5678, 1111, 2222]
    breast_results, breast_stats = run_full_dataset_experiment(
        dataset_name="breastmnist",
        data_splits=breast_splits,
        num_classes=2,
        num_epochs=20,
        batch_size=32,
        seeds=breast_seeds,
        n_quantum_layers=2,
        save_json_path="results/full_trainable_breastmnist.json"
    )
    
    # Generate Visualizations for BreastMNIST
    print('\n>>> Generating Visualizations for BreastMNIST...')
    plot_gd3_training_curves(breast_results, "BreastMNIST", save_path="results/figures/gd3_breastmnist_curves.png")
    plot_gd3_theta_trajectories(breast_results, save_path="results/figures/gd3_breastmnist_theta_trajectories.png")
    plot_gd3_gradient_norms(breast_results, save_path="results/figures/gd3_breastmnist_gradient_norms.png")
    plot_gd3_benchmark_summary(breast_results, "BreastMNIST", save_path="results/figures/gd3_breastmnist_benchmark.png")
    
    print_dataset_benchmark_table("BreastMNIST (Full 780 Samples - 10 Seeds)", breast_results, breast_stats)
    
    # ---------------------------------------------------------
    # PART 2: OCTMNIST (5,000 samples, 5 seeds, L=1)
    # ---------------------------------------------------------
    print('\n' + '#' * 80)
    print(' [PART 2/2]: FULL OCTMNIST EXPERIMENT (5,000 SAMPLES, 5 SEEDS, L=1)')
    print('#' * 80)
    
    oct_splits = prepare_data(
        dataset_name="octmnist",
        seed=42
    )
    
    oct_seeds = [0, 42, 100, 2023, 777]
    oct_results, oct_stats = run_full_dataset_experiment(
        dataset_name="octmnist",
        data_splits=oct_splits,
        num_classes=4,
        num_epochs=20,
        batch_size=50,
        seeds=oct_seeds,
        n_quantum_layers=1,
        save_json_path="results/full_trainable_octmnist.json"
    )
    
    # Generate Visualizations for OCTMNIST
    print('\n>>> Generating Visualizations for OCTMNIST...')
    plot_gd3_training_curves(oct_results, "OCTMNIST", save_path="results/figures/gd3_octmnist_curves.png")
    plot_gd3_theta_trajectories(oct_results, save_path="results/figures/gd3_octmnist_theta_trajectories.png")
    plot_gd3_gradient_norms(oct_results, save_path="results/figures/gd3_octmnist_gradient_norms.png")
    plot_gd3_benchmark_summary(oct_results, "OCTMNIST", save_path="results/figures/gd3_octmnist_benchmark.png")
    
    print_dataset_benchmark_table("OCTMNIST (Full 5,000 Samples - 5 Seeds)", oct_results, oct_stats)
    
    total_elapsed = time.time() - total_start
    print('\n' + '#' * 80)
    print(f' PHASE 3 FULL EXECUTION COMPLETED SUCCESSFULLY IN {total_elapsed/60:.2f} MINUTES!')
    print(' Results JSON saved in: results/full_trainable_*.json')
    print(' Figures saved in: results/figures/gd3_*.png')
    print('#' * 80 + '\n')

if __name__ == '__main__':
    main()
