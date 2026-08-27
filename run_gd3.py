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

def print_3tier_benchmark_report(dataset_title, results, stat_report):
    metrics = ['acc', 'bacc', 'f1', 'mcc', 'auc', 'pr_auc']
    metric_names = ['Accuracy', 'Balanced Acc', 'F1-Score', 'MCC', 'ROC-AUC', 'PR-AUC']
    models = list(results.keys())
    
    print('\n' + '='*100)
    print(f' COMPREHENSIVE 3-TIER BENCHMARK REPORT: {dataset_title.upper()}')
    print('='*100)

    # 1. Master Overview Table
    print(f"\n--- [MASTER TABLE: ALL MODELS PERFORMANCE (MEAN +- STD)] ---")
    header_cols = [f"{'Metric':<14}"] + [f"{m:<18}" for m in models]
    print(" | ".join(header_cols))
    print("-" * (17 * len(models) + 16))
    
    for m, m_name in zip(metrics, metric_names):
        row = [f"{m_name:<14}"]
        for model in models:
            vals = [r[m] for r in results[model]['test_metrics']]
            row.append(f"{np.mean(vals):.4f} +- {np.std(vals):.4f}")
        print(" | ".join(row))
    print("-" * (17 * len(models) + 16))

    # 2. Tier 1: Controlled Intra-Ansatz (Trainable Basic vs Fixed Basic)
    if 'trainable_basic' in results and 'fixed_basic' in results:
        key = 'fixed_basic_vs_trainable_basic' if 'fixed_basic_vs_trainable_basic' in stat_report else 'trainable_basic_vs_fixed_basic'
        print(f"\n>>> [TIER 1: CONTROLLED INTRA-ANSATZ] Trainable Basic vs Fixed Basic")
        print(f"{'Metric':<14} | {'Fixed Basic':<18} | {'Trainable Basic':<18} | {'Delta':<10} | {'p-value (t-test)':<18} | {'p-value (Wilcoxon)':<18}")
        print("-" * 102)
        for m, m_name in zip(metrics, metric_names):
            f_vals = [r[m] for r in results['fixed_basic']['test_metrics']]
            t_vals = [r[m] for r in results['trainable_basic']['test_metrics']]
            delta = np.mean(t_vals) - np.mean(f_vals)
            s = stat_report.get(key, {}).get(m, {})
            p_t = s.get('p_value_ttest', 1.0)
            p_w = s.get('wilcoxon_p_value', 1.0)
            sig_t = '***' if p_t < 0.001 else '**' if p_t < 0.01 else '*' if p_t < 0.05 else 'ns'
            print(f"{m_name:<14} | {np.mean(f_vals):.4f} +- {np.std(f_vals):.4f} | {np.mean(t_vals):.4f} +- {np.std(t_vals):.4f} | {delta:+.4f}    | p={p_t:.4f} ({sig_t:<3})    | p={p_w:.4f}")
        print("-" * 102)

    # 3. Tier 2: Champion Stress-Test
    if 'fixed_champion_gd2' in results and 'trainable_basic' in results:
        key = 'trainable_basic_vs_fixed_champion_gd2' if 'trainable_basic_vs_fixed_champion_gd2' in stat_report else 'fixed_champion_gd2_vs_trainable_basic'
        print(f"\n>>> [TIER 2: FIXED CHAMPION STRESS-TEST] Trainable Basic vs Fixed Champion GĐ2 (random_L1)")
        print(f"{'Metric':<14} | {'Fixed Champ GĐ2':<18} | {'Trainable Basic':<18} | {'Delta':<10} | {'p-value (t-test)':<18} | {'p-value (Wilcoxon)':<18}")
        print("-" * 102)
        for m, m_name in zip(metrics, metric_names):
            f_vals = [r[m] for r in results['fixed_champion_gd2']['test_metrics']]
            t_vals = [r[m] for r in results['trainable_basic']['test_metrics']]
            delta = np.mean(t_vals) - np.mean(f_vals)
            s = stat_report.get(key, {}).get(m, {})
            p_t = s.get('p_value_ttest', 1.0)
            p_w = s.get('wilcoxon_p_value', 1.0)
            sig_t = '***' if p_t < 0.001 else '**' if p_t < 0.01 else '*' if p_t < 0.05 else 'ns'
            print(f"{m_name:<14} | {np.mean(f_vals):.4f} +- {np.std(f_vals):.4f} | {np.mean(t_vals):.4f} +- {np.std(t_vals):.4f} | {delta:+.4f}    | p={p_t:.4f} ({sig_t:<3})    | p={p_w:.4f}")
        print("-" * 102)

    # 4. Tier 3: Full-Expressive Showdown (Fixed Strongly vs Trainable Strongly)
    if 'fixed_strongly' in results and 'trainable_strongly' in results:
        key = 'fixed_strongly_vs_trainable_strongly' if 'fixed_strongly_vs_trainable_strongly' in stat_report else 'trainable_strongly_vs_fixed_strongly'
        print(f"\n>>> [TIER 3: FULL-EXPRESSIVE SHOWDOWN] Trainable Strongly (3-Axis) vs Fixed Strongly")
        print(f"{'Metric':<14} | {'Fixed Strongly':<18} | {'Trainable Strongly':<18} | {'Delta':<10} | {'p-value (t-test)':<18} | {'p-value (Wilcoxon)':<18}")
        print("-" * 102)
        for m, m_name in zip(metrics, metric_names):
            f_vals = [r[m] for r in results['fixed_strongly']['test_metrics']]
            t_vals = [r[m] for r in results['trainable_strongly']['test_metrics']]
            delta = np.mean(t_vals) - np.mean(f_vals)
            s = stat_report.get(key, {}).get(m, {})
            p_t = s.get('p_value_ttest', 1.0)
            p_w = s.get('wilcoxon_p_value', 1.0)
            sig_t = '***' if p_t < 0.001 else '**' if p_t < 0.01 else '*' if p_t < 0.05 else 'ns'
            print(f"{m_name:<14} | {np.mean(f_vals):.4f} +- {np.std(f_vals):.4f} | {np.mean(t_vals):.4f} +- {np.std(t_vals):.4f} | {delta:+.4f}    | p={p_t:.4f} ({sig_t:<3})    | p={p_w:.4f}")
        print("-" * 102)

        # Determine Winner between Fixed Strongly vs Trainable Strongly
        mean_auc_f = np.mean([r['auc'] for r in results['fixed_strongly']['test_metrics']])
        mean_auc_t = np.mean([r['auc'] for r in results['trainable_strongly']['test_metrics']])
        winner_key = 'trainable_strongly' if mean_auc_t >= mean_auc_f else 'fixed_strongly'
        winner_name = "Trainable Strongly (Full-Exp)" if winner_key == 'trainable_strongly' else "Fixed Strongly (Full-Exp)"
        print(f"\n🏆 [TIER 3 QUANTUM WINNER]: {winner_name} (ROC-AUC: {max(mean_auc_f, mean_auc_t):.4f})")

        # Compare Winner vs Classical CNN
        if 'classical_cnn' in results:
            pair_k = f"classical_cnn_vs_{winner_key}" if f"classical_cnn_vs_{winner_key}" in stat_report else f"{winner_key}_vs_classical_cnn"
            print(f">>> [TIER 3 ULTIMATE MATCH]: {winner_name} vs Classical CNN Baseline")
            print(f"{'Metric':<14} | {'Classical CNN':<18} | {winner_name:<26} | {'Delta':<10} | {'p-value (t-test)':<18}")
            print("-" * 92)
            for m, m_name in zip(metrics, metric_names):
                c_vals = [r[m] for r in results['classical_cnn']['test_metrics']]
                w_vals = [r[m] for r in results[winner_key]['test_metrics']]
                delta = np.mean(w_vals) - np.mean(c_vals)
                s = stat_report.get(pair_k, {}).get(m, {})
                p_t = s.get('p_value_ttest', 1.0)
                sig_t = '***' if p_t < 0.001 else '**' if p_t < 0.01 else '*' if p_t < 0.05 else 'ns'
                print(f"{m_name:<14} | {np.mean(c_vals):.4f} +- {np.std(c_vals):.4f} | {np.mean(w_vals):.4f} +- {np.std(w_vals):.4f} | {delta:+.4f}    | p={p_t:.4f} ({sig_t:<3})")
            print("-" * 92)
    print('='*100 + '\n')

def main():
    print('================================================================================')
    print('          PHASE 3 MASTER EXECUTION: 3-TIER MULTI-MODEL QUANTUM TOURNAMENT       ')
    print(f'          Start Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}          ')
    print('          Backend: Differentiable Analytical Backprop (default.qubit)           ')
    print('================================================================================')
    
    total_start = time.time()
    
    # ---------------------------------------------------------
    # PART 1: BREASTMNIST (780 samples, 10 seeds, L=2)
    # ---------------------------------------------------------
    print('\n' + '#' * 90)
    print(' [PART 1/2]: FULL BREASTMNIST 3-TIER EXPERIMENT (780 SAMPLES, 10 SEEDS, L=2)')
    print('#' * 90)
    
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
    plot_gd3_theta_trajectories(breast_results, target_model="trainable_strongly", save_path="results/figures/gd3_breastmnist_theta_trajectories.png")
    plot_gd3_gradient_norms(breast_results, target_model="trainable_strongly", save_path="results/figures/gd3_breastmnist_gradient_norms.png")
    plot_gd3_benchmark_summary(breast_results, "BreastMNIST", save_path="results/figures/gd3_breastmnist_benchmark.png")
    
    print_3tier_benchmark_report("BreastMNIST (Full 780 Samples - 10 Seeds)", breast_results, breast_stats)
    
    # ---------------------------------------------------------
    # PART 2: OCTMNIST (5,000 samples, 5 seeds, L=1)
    # ---------------------------------------------------------
    print('\n' + '#' * 90)
    print(' [PART 2/2]: FULL OCTMNIST 3-TIER EXPERIMENT (5,000 SAMPLES, 5 SEEDS, L=1)')
    print('#' * 90)
    
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
    plot_gd3_theta_trajectories(oct_results, target_model="trainable_strongly", save_path="results/figures/gd3_octmnist_theta_trajectories.png")
    plot_gd3_gradient_norms(oct_results, target_model="trainable_strongly", save_path="results/figures/gd3_octmnist_gradient_norms.png")
    plot_gd3_benchmark_summary(oct_results, "OCTMNIST", save_path="results/figures/gd3_octmnist_benchmark.png")
    
    print_3tier_benchmark_report("OCTMNIST (Full 5,000 Samples - 5 Seeds)", oct_results, oct_stats)
    
    total_elapsed = time.time() - total_start
    print('\n' + '#' * 90)
    print(f' PHASE 3 3-TIER FULL EXECUTION COMPLETED SUCCESSFULLY IN {total_elapsed/60:.2f} MINUTES!')
    print(' Results JSON saved in: results/full_trainable_*.json')
    print(' Figures saved in: results/figures/gd3_*.png')
    print('#' * 90 + '\n')

if __name__ == '__main__':
    main()
