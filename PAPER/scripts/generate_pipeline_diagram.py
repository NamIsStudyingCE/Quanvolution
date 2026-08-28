# -*- coding: utf-8 -*-
"""
generate_pipeline_diagram.py
----------------------------
Vẽ Sơ đồ Kiến trúc Pipeline Quanvolution (Figure 1) chuẩn công bố khoa học quốc tế (300 DPI).
Bao gồm: Luồng ảnh đầu vào -> Patch 2x2 -> 4-Qubit Circuit (Angle Embedding, Entanglement, Measurement)
-> Feature Maps 4x14x14 -> Symmetrical Head -> Dự đoán phân loại.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

# Set style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['mathtext.fontset'] = 'cm'

def draw_pipeline():
    fig, ax = plt.subplots(figsize=(16, 7.5), dpi=300)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 7.5)
    ax.axis('off')

    # Color palette (IEEE/Nature inspired)
    c_blue = '#1f77b4'
    c_light_blue = '#e8f1f8'
    c_purple = '#7b1fa2'
    c_light_purple = '#f3e5f5'
    c_orange = '#ff7f0e'
    c_light_orange = '#fff3e0'
    c_green = '#2ca02c'
    c_light_green = '#e8f5e9'
    c_gray = '#424242'
    c_box_gray = '#fafafa'

    # Title
    ax.text(8.0, 7.1, "Figure 1: End-to-End Architecture of the Quanvolutional Neural Network (Quanvolution)",
            ha='center', va='center', fontsize=13, fontweight='bold', color='#111111')
    ax.text(8.0, 6.75, "Comparison of Symmetrical 4-Qubit Quantum Kernel vs. Minimum Classical Convolutional Filter",
            ha='center', va='center', fontsize=10, style='italic', color='#555555')

    # -------------------------------------------------------------
    # 1. INPUT IMAGE BLOCK
    # -------------------------------------------------------------
    # Image box
    rect_img = patches.FancyBboxPatch((0.5, 2.5), 1.8, 2.2, boxstyle="round,pad=0.08,rounding_size=0.1",
                                      facecolor='#f0f0f0', edgecolor=c_gray, linewidth=1.5)
    ax.add_patch(rect_img)
    ax.text(1.4, 4.4, "Medical Image", ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(1.4, 4.15, r"$28 \times 28 \times 1$", ha='center', va='center', fontsize=8.5, color='#333333')
    
    # Mock ultrasound breast pattern
    x_grid = np.linspace(0.7, 2.1, 14)
    y_grid = np.linspace(2.7, 3.9, 12)
    for x in x_grid:
        for y in y_grid:
            val = np.sin((x-0.7)*3)*np.cos((y-2.7)*4)
            alpha = 0.2 + 0.5 * abs(val)
            ax.plot(x, y, 's', color='black', markersize=3, alpha=alpha)

    # 2x2 Patch highlighted
    patch_box = patches.Rectangle((0.9, 3.2), 0.4, 0.4, linewidth=1.8, edgecolor='red', facecolor='red', alpha=0.3)
    ax.add_patch(patch_box)
    ax.text(1.1, 3.0, "Patch $2 \\times 2$\n(Stride = 2)", ha='center', va='top', fontsize=7.5, color='darkred', fontweight='bold')

    # Arrow to extraction
    ax.annotate('', xy=(3.0, 3.6), xytext=(2.4, 3.6),
                arrowprops=dict(facecolor=c_gray, edgecolor=c_gray, width=1.5, headwidth=6, shrink=0.05))
    ax.text(2.7, 3.9, "196 Patches\n$(x_0, x_1, x_2, x_3)$", ha='center', va='bottom', fontsize=7.5, color='#222222')

    # -------------------------------------------------------------
    # 2. QUANTUM CIRCUIT CORE (4 QUBITS)
    # -------------------------------------------------------------
    q_bg = patches.FancyBboxPatch((3.1, 1.4), 4.8, 4.6, boxstyle="round,pad=0.1,rounding_size=0.15",
                                 facecolor=c_light_purple, edgecolor=c_purple, linewidth=2.0)
    ax.add_patch(q_bg)
    ax.text(5.5, 5.75, "4-Qubit Quantum Kernel Filter", ha='center', va='center', fontsize=11, fontweight='bold', color=c_purple)

    # Sub-boxes for circuit stages
    # A) Embedding
    emb_box = patches.Rectangle((3.3, 1.7), 1.2, 3.7, facecolor='#ffffff', edgecolor='#9c27b0', linestyle='--', linewidth=1)
    ax.add_patch(emb_box)
    ax.text(3.9, 5.2, "Angle Embedding\n" + r"$|\psi_{\mathrm{in}}\rangle = \bigotimes R_Y(\pi x_i)|0\rangle$", 
            ha='center', va='center', fontsize=7.5, fontweight='bold', color='#4a148c')

    # B) Unitary
    uni_box = patches.Rectangle((4.7, 1.7), 1.8, 3.7, facecolor='#ffffff', edgecolor=c_purple, linewidth=1.5)
    ax.add_patch(uni_box)
    ax.text(5.6, 5.2, "Entangling Unitary $U(\\theta)$\n(Fixed or Trainable)", 
            ha='center', va='center', fontsize=8, fontweight='bold', color=c_purple)

    # C) Measurement
    meas_box = patches.Rectangle((6.7, 1.7), 1.0, 3.7, facecolor='#ffffff', edgecolor='#d81b60', linestyle='--', linewidth=1)
    ax.add_patch(meas_box)
    ax.text(7.2, 5.2, "Pauli-Z Meas.\n" + r"$f_i = \langle Z_i \rangle$", ha='center', va='center', fontsize=7.5, fontweight='bold', color='#880e4f')

    # Qubit lines & gates
    q_ys = [4.5, 3.7, 2.9, 2.1]
    for i, qy in enumerate(q_ys):
        # Qubit line
        ax.plot([3.4, 7.6], [qy, qy], color=c_gray, linewidth=1.2, zorder=1)
        ax.text(3.2, qy, f"$|0\\rangle_{i}$", ha='right', va='center', fontsize=8.5, fontweight='bold')
        
        # RY Gate
        ry_gate = patches.Rectangle((3.5, qy-0.25), 0.7, 0.5, facecolor='#ce93d8', edgecolor='#4a148c', linewidth=1, zorder=2)
        ax.add_patch(ry_gate)
        ax.text(3.85, qy, f"$R_Y(\\pi x_{i})$", ha='center', va='center', fontsize=7, color='black', zorder=3)
        
        # Entanglement / Rotations
        u_gate = patches.Rectangle((4.9, qy-0.28), 1.4, 0.56, facecolor='#ba68c8', edgecolor='#4a148c', linewidth=1, zorder=2)
        ax.add_patch(u_gate)
        ax.text(5.6, qy, "Rot + CNOT" if i%2==0 else "3-Axis Rot $\\theta$", ha='center', va='center', fontsize=7, color='white', fontweight='bold', zorder=3)
        
        # Measurement box
        m_gate = patches.Rectangle((6.8, qy-0.25), 0.7, 0.5, facecolor='#f48fb1', edgecolor='#880e4f', linewidth=1, zorder=2)
        ax.add_patch(m_gate)
        ax.text(7.15, qy, r"$\langle Z \rangle$", ha='center', va='center', fontsize=7.5, fontweight='bold', zorder=3)
        
        # Output arrow
        ax.annotate('', xy=(8.2, qy), xytext=(7.6, qy),
                    arrowprops=dict(facecolor=c_purple, edgecolor=c_purple, width=1.0, headwidth=4, shrink=0.05))

    # CNOT vertical lines in Unitary
    ax.plot([5.3, 5.3], [4.5, 3.7], color='black', linewidth=1.5, zorder=4)
    ax.plot(5.3, 4.5, 'o', color='black', markersize=4, zorder=5)
    ax.plot(5.3, 3.7, '+', color='black', markersize=7, markeredgewidth=1.5, zorder=5)

    ax.plot([6.0, 6.0], [2.9, 2.1], color='black', linewidth=1.5, zorder=4)
    ax.plot(6.0, 2.9, 'o', color='black', markersize=4, zorder=5)
    ax.plot(6.0, 2.1, '+', color='black', markersize=7, markeredgewidth=1.5, zorder=5)

    # -------------------------------------------------------------
    # 3. FEATURE MAPS BLOCK
    # -------------------------------------------------------------
    f_bg = patches.FancyBboxPatch((8.3, 2.2), 2.2, 2.8, boxstyle="round,pad=0.08,rounding_size=0.1",
                                 facecolor=c_light_blue, edgecolor=c_blue, linewidth=1.5)
    ax.add_patch(f_bg)
    ax.text(9.4, 4.75, "Quantum Feature Maps", ha='center', va='center', fontsize=9.5, fontweight='bold', color=c_blue)
    ax.text(9.4, 4.5, r"$4 \times 14 \times 14$ Channels", ha='center', va='center', fontsize=8, color='#333333')

    # 4 stacked mini-maps
    map_colors = ['#bbdefb', '#90caf9', '#64b5f6', '#42a5f5']
    for idx, mc in enumerate(map_colors):
        ox = 8.6 + idx * 0.18
        oy = 2.5 + idx * 0.22
        map_rect = patches.Rectangle((ox, oy), 0.9, 0.9, facecolor=mc, edgecolor=c_blue, linewidth=1.0, alpha=0.85)
        ax.add_patch(map_rect)
        ax.text(ox+0.45, oy+0.45, f"$F_{idx}$", ha='center', va='center', fontsize=7.5, fontweight='bold', color='#0d47a1')

    # Arrow to classifier
    ax.annotate('', xy=(11.3, 3.6), xytext=(10.6, 3.6),
                arrowprops=dict(facecolor=c_gray, edgecolor=c_gray, width=1.5, headwidth=6, shrink=0.05))
    ax.text(10.95, 3.9, "Flatten\n(784)", ha='center', va='bottom', fontsize=7.5, color='#222222')

    # -------------------------------------------------------------
    # 4. SYMMETRICAL CLASSIFIER HEAD BLOCK
    # -------------------------------------------------------------
    c_bg = patches.FancyBboxPatch((11.4, 2.0), 4.1, 3.2, boxstyle="round,pad=0.1,rounding_size=0.15",
                                 facecolor=c_light_green, edgecolor=c_green, linewidth=2.0)
    ax.add_patch(c_bg)
    ax.text(13.45, 4.95, "Symmetrical Classifier Head", ha='center', va='center', fontsize=10.5, fontweight='bold', color=c_green)
    ax.text(13.45, 4.65, "(100% Identical between Quantum & Classical)", ha='center', va='center', fontsize=7.5, color='#2e7d32')

    # BatchNorm + Linear
    bn_box = patches.Rectangle((11.6, 2.5), 1.5, 1.8, facecolor='#ffffff', edgecolor=c_green, linewidth=1.2)
    ax.add_patch(bn_box)
    ax.text(12.35, 3.6, "BatchNorm2d", ha='center', va='center', fontsize=8, fontweight='bold')
    ax.text(12.35, 3.1, "+ ReLU", ha='center', va='center', fontsize=8, color='#555555')
    ax.text(12.35, 2.7, "8 params", ha='center', va='center', fontsize=7.5, color='#2e7d32')

    ax.annotate('', xy=(13.3, 3.4), xytext=(13.1, 3.4),
                arrowprops=dict(facecolor=c_green, edgecolor=c_green, width=1.2, headwidth=4))

    fc_box = patches.Rectangle((13.4, 2.5), 1.9, 1.8, facecolor='#ffffff', edgecolor=c_green, linewidth=1.2)
    ax.add_patch(fc_box)
    ax.text(14.35, 3.8, "Linear Layer", ha='center', va='center', fontsize=8.5, fontweight='bold')
    ax.text(14.35, 3.4, r"$784 \to K$ Classes", ha='center', va='center', fontsize=8, color='#333333')
    ax.text(14.35, 2.95, "Breast: 1,570 p\nOCT: 3,140 p", ha='center', va='center', fontsize=7, color='#1b5e20')

    # Final Output Arrow
    ax.annotate('', xy=(15.8, 3.4), xytext=(15.3, 3.4),
                arrowprops=dict(facecolor='darkred', edgecolor='darkred', width=1.5, headwidth=6, shrink=0.05))
    
    # Class predictions
    ax.text(15.9, 3.8, "Diagnosis", ha='left', va='center', fontsize=8.5, fontweight='bold', color='darkred')
    ax.text(15.9, 3.4, "• Malignant / Benign\n• 4 OCT Pathology Classes", ha='left', va='center', fontsize=7, color='#333333')

    # -------------------------------------------------------------
    # 5. BOTTOM BOX: SYMMETRICAL CLASSICAL BASELINE EQUIVALENCE
    # -------------------------------------------------------------
    cl_bg = patches.FancyBboxPatch((0.5, 0.2), 15.0, 1.0, boxstyle="round,pad=0.06,rounding_size=0.1",
                                  facecolor=c_light_orange, edgecolor=c_orange, linewidth=1.5)
    ax.add_patch(cl_bg)
    ax.text(0.7, 0.7, "Classical Baseline (Symmetrical Minimum CNN):", fontsize=8.5, fontweight='bold', color='#d84315')
    ax.text(5.5, 0.7, r"$\mathrm{Conv2D}(c_{\mathrm{in}}=1, c_{\mathrm{out}}=4, k=2, s=2)$" + " (20 params)", fontsize=8.5, color='#bf360c')
    ax.text(10.2, 0.7, r"$\to 4 \times 14 \times 14 \text{ Feature Maps}$", fontsize=8.5, color=c_blue)
    ax.text(13.2, 0.7, r"$\to \text{Same Classifier Head}$", fontsize=8.5, color=c_green)
    ax.text(0.7, 0.35, "★ Key Fairness Guarantee: Same parameter count in head, same feature dimensions (784), strictly isolated feature extractor.", 
            fontsize=7.5, style='italic', color='#444444')

    plt.tight_layout()
    
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, "Fig1_quanvolution_pipeline.png")
    out_pdf = os.path.join(out_dir, "Fig1_quanvolution_pipeline.pdf")
    
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.close()
    print(f"Saved Figure 1 (PNG 300 DPI): {out_png}")
    print(f"Saved Figure 1 (Vector PDF) : {out_pdf}")

if __name__ == '__main__':
    draw_pipeline()
