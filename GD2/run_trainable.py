import os
import sys
import time
import json
from datetime import datetime
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Ensure src is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from data.medmnist_loader import prepare_data
from experiments.trainable_experiment import run_trainable_poc_experiment
from visual.plot_trainable import plot_training_curves, plot_metrics_bar_comparison

def print_trainable_summary_table(results):
    metrics = ['acc', 'bacc', 'f1', 'mcc', 'auc', 'pr_auc']
    metric_names = ['Accuracy', 'Balanced Acc', 'F1-Score', 'MCC', 'ROC-AUC', 'PR-AUC']
    
    print("\n" + "="*85)
    print(" TRAINABLE QUANVOLUTION PROOF-OF-CONCEPT BENCHMARK (OCTMNIST 500 SAMPLES)")
    print("="*85)
    print(f"{'Metric':<14} | {'Classical CNN':<20} | {'Fixed Quanvolution':<20} | {'Trainable Quanv':<20}")
    print("-" * 85)
    
    summary_export = {}
    
    for m, m_name in zip(metrics, metric_names):
        c_vals = [r[m] for r in results['classical_cnn']['test_metrics']]
        f_vals = [r[m] for r in results['fixed_quanv']['test_metrics']]
        t_vals = [r[m] for r in results['trainable_quanv']['test_metrics']]
        
        c_str = f"{np.mean(c_vals):.4f} +- {np.std(c_vals):.4f}"
        f_str = f"{np.mean(f_vals):.4f} +- {np.std(f_vals):.4f}"
        t_str = f"{np.mean(t_vals):.4f} +- {np.std(t_vals):.4f}"
        
        summary_export[m] = {
            'classical_cnn':   c_str,
            'fixed_quanv':     f_str,
            'trainable_quanv': t_str
        }
        print(f"{m_name:<14} | {c_str:<20} | {f_str:<20} | {t_str:<20}")
    print("="*85 + "\n")
    return summary_export

def main():
    print("=======================================================")
    print(" STARTING TRAINABLE QUANVOLUTION EXPERIMENT (STEP 3)")
    print(f" Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(" Architecture: 4-Qubit Variational Quanvolution with TorchLayer")
    print(" Gradient Method: Parameter-Shift Rule")
    print(" Dataset: OCTMNIST 500-Sample Stratified POC Subset")
    print(" Seeds: [0, 42, 100] | Epochs: 20 | Batch Size: 35")
    print("=======================================================\n")
    
    start_time = time.time()
    
    # 1. Prepare POC data splits (500 samples: 350 train, 50 val, 100 test)
    print(">>> [1/3] Preparing POC Data Splits (OCTMNIST 500 Samples)...")
    splits = prepare_data(
        dataset_name="octmnist_poc_500",
        max_train=350,
        max_val=50,
        max_test=100,
        seed=42
    )
    num_classes = 4
    
    # 2. Run Comparative Training Experiment
    print("\n>>> [2/3] Running Comparative Training (Classical vs Fixed vs Trainable)...")
    results = run_trainable_poc_experiment(
        data_splits=splits,
        num_classes=num_classes,
        epochs=20,
        seeds=[0, 42, 100]
    )
    
    # 3. Plot Learning Dynamics and Bar Comparisons
    print("\n>>> [3/3] Generating Visualizations and Exporting Results...")
    plot_training_curves(results)
    plot_metrics_bar_comparison(results)
    
    # Print formatted table
    summary_export = print_trainable_summary_table(results)
    
    # Save results to JSON
    os.makedirs("results", exist_ok=True)
    export_payload = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "experiment": "Trainable Quanvolution Proof-of-Concept",
        "dataset": "OCTMNIST (500 Samples Subset)",
        "num_epochs": 20,
        "seeds": [0, 42, 100],
        "summary_table": summary_export,
        "raw_results": {
            k: {
                "test_metrics": results[k]["test_metrics"],
                "histories": results[k]["histories"]
            }
            for k in results.keys()
        }
    }
    
    json_path = "results/trainable_poc_summary.json"
    with open(json_path, "w") as f:
        json.dump(export_payload, f, indent=4)
        
    elapsed = time.time() - start_time
    print(f"\n" + "#"*85)
    print(f" TRAINABLE QUANVOLUTION EXPERIMENT COMPLETED IN {elapsed/60:.2f} MINUTES!")
    print(f" Log saved to: {json_path}")
    print(f" Figures saved to: results/figures/trainable_poc_*.png")
    print("#"*85)

if __name__ == "__main__":
    main()
