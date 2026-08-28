# -*- coding: utf-8 -*-
"""
regenerate_all_figures.py
-------------------------
Re-generates all figures using updated plot_gd3_dynamics.py and copies them into
results/figures/, GD3/figures/, and PAPER/figures/.
"""

import os, json, sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, root)

from src.visual.plot_gd3_dynamics import (
    plot_gd3_training_curves,
    plot_gd3_theta_trajectories,
    plot_gd3_gradient_norms,
    plot_gd3_benchmark_summary
)

def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    breast_json = os.path.join(root, "results", "full_trainable_breastmnist.json")
    oct_json = os.path.join(root, "results", "full_trainable_octmnist.json")

    print("=== Re-generating BreastMNIST Figures ===")
    with open(breast_json, 'r', encoding='utf-8') as f:
        breast_data = json.load(f)['raw_results']

    plot_gd3_training_curves(breast_data, "BreastMNIST", save_path=os.path.join(root, "PAPER", "figures", "Fig4a_breastmnist_curves.png"))
    plot_gd3_theta_trajectories(breast_data, target_model="trainable_strongly", save_path=os.path.join(root, "PAPER", "figures", "Fig4c_theta_trajectories.png"))
    plot_gd3_gradient_norms(breast_data, target_model="trainable_strongly", save_path=os.path.join(root, "PAPER", "figures", "Fig4d_gradient_norms.png"))
    plot_gd3_benchmark_summary(breast_data, "BreastMNIST", save_path=os.path.join(root, "PAPER", "figures", "Fig3_breastmnist_benchmark.png"))

    print("\n=== Re-generating OCTMNIST Figures ===")
    with open(oct_json, 'r', encoding='utf-8') as f:
        oct_data = json.load(f)['raw_results']

    plot_gd3_training_curves(oct_data, "OCTMNIST", save_path=os.path.join(root, "PAPER", "figures", "Fig4b_octmnist_curves.png"))
    plot_gd3_benchmark_summary(oct_data, "OCTMNIST", save_path=os.path.join(root, "PAPER", "figures", "Fig3_octmnist_benchmark.png"))

    # Also copy to results/figures/ and GD3/figures/
    for fn in ["Fig3_breastmnist_benchmark.png", "Fig3_octmnist_benchmark.png", "Fig4a_breastmnist_curves.png", "Fig4b_octmnist_curves.png", "Fig4c_theta_trajectories.png", "Fig4d_gradient_norms.png"]:
        src_f = os.path.join(root, "PAPER", "figures", fn)
        # map name to standard gd3 name
        gd3_name = fn.replace("Fig3_", "gd3_").replace("Fig4a_", "gd3_").replace("Fig4b_", "gd3_").replace("Fig4c_", "gd3_").replace("Fig4d_", "gd3_")
        
        dst_res = os.path.join(root, "results", "figures", gd3_name)
        dst_gd3 = os.path.join(root, "GD3", "figures", gd3_name)
        
        with open(src_f, 'rb') as f_in:
            content = f_in.read()
        with open(dst_res, 'wb') as f_out:
            f_out.write(content)
        with open(dst_gd3, 'wb') as f_out:
            f_out.write(content)

    print("\n[THÀNH CÔNG] Toàn bộ biểu đồ Fig 3 & Fig 4 đã được vẽ lại với legend nằm ngoài hoàn toàn!")

if __name__ == '__main__':
    main()
