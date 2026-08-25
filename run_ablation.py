import os
import sys
import time
import json
from datetime import datetime

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Ensure src is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from data.medmnist_loader import prepare_data
from data.precompute_circuits import precompute_all_circuits_for_dataset
from experiments.circuit_ablation import run_stage1_screening, run_stage2_deep_validation
from visual.plot_ablation import plot_circuit_ablation_chart

def print_final_ablation_table(dataset_name, stage1_results, champion_name, stage2_result):
    print("\n" + "="*80)
    print(f" CIRCUIT ARCHITECTURE ABLATION SUMMARY: {dataset_name.upper()}")
    print("="*80)
    print(f"{'Circuit Architecture':<22} | {'ROC-AUC (5 Seeds)':<22} | {'PR-AUC (5 Seeds)':<22}")
    print("-" * 80)
    for c_name, data in stage1_results.items():
        auc_str = data['summary']['auc']['formatted']
        pr_str  = data['summary']['pr_auc']['formatted']
        prefix = "[CHAMPION] " if c_name == champion_name else "  "
        print(f"{prefix + c_name:<25} | {auc_str:<22} | {pr_str:<22}")
    print("-" * 80)
    print(f" Deep 10-Seed Validation for [{champion_name}]:")
    print(f"   ROC-AUC (10 Seeds): {stage2_result['summary']['auc']['formatted']}")
    print(f"   PR-AUC  (10 Seeds): {stage2_result['summary']['pr_auc']['formatted']}")
    print(f"   ACC     (10 Seeds): {stage2_result['summary']['acc']['formatted']}")
    print("="*80 + "\n")

def main():
    print(f"=======================================================")
    print(f" STARTING QUANTUM CIRCUIT ABLATION PIPELINE (STEP 2)")
    print(f" Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Strategy 1B: Full BreastMNIST (780) + Full OCTMNIST (5000)")
    print(f" Strategy 2A: 5-Seed Screening -> 10-Seed Champion Validation")
    print(f"=======================================================\n")
    
    total_start_time = time.time()
    
    overall_ablation_summary = {}
    overall_champion_summary = {}
    
    datasets_config = [
        {'name': 'breastmnist', 'max_train': None, 'max_val': None, 'max_test': None},
        {'name': 'octmnist',    'max_train': 3500, 'max_val': 500,  'max_test': 1000}
    ]
    
    for cfg in datasets_config:
        d_name = cfg['name']
        print(f"\n#######################################################")
        print(f" PROCESSING DATASET: {d_name.upper()}")
        print(f"#######################################################")
        
        # 1. Prepare deterministic data splits
        print(f"\n>>> [1/4] Checking Data Splits for {d_name.upper()}...")
        splits = prepare_data(
            dataset_name=d_name,
            max_train=cfg['max_train'],
            max_val=cfg['max_val'],
            max_test=cfg['max_test'],
            seed=42
        )
        num_classes = splits['num_classes']
        
        # 2. Precompute all 6 circuits
        print(f"\n>>> [2/4] Precomputing 6 Quantum Circuits for {d_name.upper()}...")
        precompute_all_circuits_for_dataset(dataset_name=d_name, data_splits=splits)
        
        # 3. Stage 1: 5-Seed Screening
        print(f"\n>>> [3/4] Running Stage 1 Screening (5 Seeds across 6 Circuits)...")
        stage1_results, champion_circuit = run_stage1_screening(
            dataset_name=d_name,
            num_classes=num_classes,
            epochs=30
        )
        overall_ablation_summary[d_name] = stage1_results
        
        # 4. Stage 2: Deep 10-Seed Validation on Champion
        print(f"\n>>> [4/4] Running Stage 2 Deep Validation (10 Seeds on Champion)...")
        stage2_result = run_stage2_deep_validation(
            dataset_name=d_name,
            champion_circuit=champion_circuit,
            num_classes=num_classes,
            epochs=30
        )
        overall_champion_summary[d_name] = stage2_result
        
        # Plot ablation chart
        plot_circuit_ablation_chart(dataset_name=d_name, stage1_results=stage1_results)
        
        # Print summary table
        print_final_ablation_table(d_name, stage1_results, champion_circuit, stage2_result)
        
    # Save overall JSON summaries
    os.makedirs("results", exist_ok=True)
    with open("results/circuit_ablation_summary.json", "w") as f:
        json.dump(overall_ablation_summary, f, indent=4)
    with open("results/circuit_ablation_champion_10seeds.json", "w") as f:
        json.dump(overall_champion_summary, f, indent=4)
        
    total_elapsed = time.time() - total_start_time
    print("\n" + "#"*80)
    print(f" ALL CIRCUIT ABLATION EXPERIMENTS COMPLETED IN {total_elapsed/60:.2f} MINUTES!")
    print(f" Results & figures saved to 'results/' and 'results/figures/'")
    print("#"*80)

if __name__ == "__main__":
    main()
