# -*- coding: utf-8 -*-
"""
generate_pipeline_diagram.py
----------------------------
Vẽ Sơ đồ Kiến trúc Pipeline Quanvolution (Figure 1) chuẩn công bố khoa học quốc tế (300 DPI).
Đã sửa toàn bộ các lỗi hiển thị:
1. Làm rõ ô patch 2x2 với viền và nhãn phóng to trực quan.
2. Dịch chữ "Patch 2x2 (Stride=2)" xuống dưới hẳn khối ảnh.
3. Tăng khoảng cách giữa Khối 1 và Khối 2, kéo dài mũi tên để chữ 196 patches không dính viền.
4. Đưa các nhãn |0> vào sâu bên trong Khối 2 với khoảng đệm an toàn.
5. Thiết kế lại mạch lượng tử: Tách riêng cổng quay tham số R(theta) và các cổng CNOT ở khoảng trống riêng,
   hoàn toàn không để đường dây CNOT đè lên chữ.
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
    # Canvas width expanded to 18 for ultra-clean spacing
    fig, ax = plt.subplots(figsize=(18, 7.8), dpi=300)
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 7.8)
    ax.axis('off')

    # Color palette
    c_blue = '#1f77b4'
    c_light_blue = '#e8f1f8'
    c_purple = '#7b1fa2'
    c_light_purple = '#f6eef9'
    c_orange = '#ff7f0e'
    c_light_orange = '#fff3e0'
    c_green = '#2ca02c'
    c_light_green = '#e8f5e9'
    c_gray = '#424242'

    # Title
    ax.text(9.0, 7.45, "Figure 1: End-to-End Architecture of the Quanvolutional Neural Network (Quanvolution)",
            ha='center', va='center', fontsize=13, fontweight='bold', color='#111111')
    ax.text(9.0, 7.12, "Symmetrical 4-Qubit Quantum Kernel vs. Minimum Classical Convolutional Filter Baseline",
            ha='center', va='center', fontsize=10, style='italic', color='#555555')

    # =============================================================
    # 1. INPUT MEDICAL IMAGE (x: 0.4 -> 2.4, y: 2.2 -> 6.2)
    # =============================================================
    rect_img = patches.FancyBboxPatch((0.4, 2.3), 2.1, 4.0, boxstyle="round,pad=0.08,rounding_size=0.12",
                                      facecolor='#f8f9fa', edgecolor=c_gray, linewidth=1.5)
    ax.add_patch(rect_img)
    ax.text(1.45, 6.0, "Medical Image", ha='center', va='center', fontsize=10.5, fontweight='bold')
    ax.text(1.45, 5.68, r"$28 \times 28 \times 1$ (Grayscale)", ha='center', va='center', fontsize=8.5, color='#444444')
    
    # 2D Grid pixels representation
    x_grid = np.linspace(0.65, 2.25, 12)
    y_grid = np.linspace(2.9, 5.2, 12)
    for x in x_grid:
        for y in y_grid:
            val = np.sin((x-0.65)*4)*np.cos((y-2.9)*4)
            alpha = 0.15 + 0.45 * abs(val)
            ax.plot(x, y, 's', color='#263238', markersize=3.2, alpha=alpha)

    # Clearly highlighted 2x2 Local Sliding Patch
    # Positioned nicely in the upper-left of grid
    patch_rect = patches.Rectangle((0.77, 4.3), 0.35, 0.45, linewidth=2.0, edgecolor='#d32f2f', facecolor='#ffcdd2', alpha=0.75, zorder=5)
    ax.add_patch(patch_rect)
    
    # Visual bracket connecting patch to description below
    ax.annotate("Sliding $2 \\times 2$ Patch\n(Kernel Window)", xy=(0.95, 4.3), xytext=(1.45, 2.55),
                ha='center', va='center', fontsize=7.5, fontweight='bold', color='#b71c1c',
                arrowprops=dict(arrowstyle="->", color='#d32f2f', lw=1.2, connectionstyle="arc3,rad=-0.2"))

    # Text below the image box (clean, outside the box)
    ax.text(1.45, 2.05, "Step Size: Stride = 2\nTotal: 196 Local Patches", ha='center', va='top', fontsize=8, color='#333333')

    # =============================================================
    # TRANSITION ARROW: BLOCK 1 -> BLOCK 2 (Widened spacing)
    # =============================================================
    ax.annotate('', xy=(3.8, 4.2), xytext=(2.6, 4.2),
                arrowprops=dict(facecolor=c_gray, edgecolor=c_gray, width=2.0, headwidth=7, shrink=0.05))
    ax.text(3.2, 4.55, "196 Patches\n" + r"$\mathbf{x} = (x_0, x_1, x_2, x_3)$", ha='center', va='bottom', fontsize=8, fontweight='bold', color='#111111')

    # =============================================================
    # 2. 4-QUBIT QUANTUM KERNEL FILTER (x: 3.9 -> 9.7, y: 1.4 -> 6.5)
    # =============================================================
    q_bg = patches.FancyBboxPatch((3.9, 1.4), 5.8, 5.1, boxstyle="round,pad=0.1,rounding_size=0.15",
                                 facecolor=c_light_purple, edgecolor=c_purple, linewidth=2.0)
    ax.add_patch(q_bg)
    ax.text(6.8, 6.2, "4-Qubit Quantum Kernel Filter", ha='center', va='center', fontsize=11.5, fontweight='bold', color=c_purple)

    # Sub-stages inside Quantum Kernel:
    # A) Angle Embedding (x: 4.8 -> 5.9)
    emb_box = patches.Rectangle((4.8, 1.65), 1.1, 4.2, facecolor='#ffffff', edgecolor='#ab47bc', linestyle='--', linewidth=1.2)
    ax.add_patch(emb_box)
    ax.text(5.35, 5.6, "Angle Embedding\n" + r"$R_Y(\pi x_i)|0\rangle$", 
            ha='center', va='center', fontsize=7.5, fontweight='bold', color='#6a1b9a')

    # B) Entangling Unitary (x: 6.1 -> 8.2)
    uni_box = patches.Rectangle((6.1, 1.65), 2.2, 4.2, facecolor='#ffffff', edgecolor=c_purple, linewidth=1.5)
    ax.add_patch(uni_box)
    ax.text(7.2, 5.6, "Entangling Unitary $U(\\theta)$\n(Fixed / Trainable)", 
            ha='center', va='center', fontsize=8, fontweight='bold', color=c_purple)

    # C) Pauli-Z Measurement (x: 8.5 -> 9.5)
    meas_box = patches.Rectangle((8.5, 1.65), 1.0, 4.2, facecolor='#ffffff', edgecolor='#c2185b', linestyle='--', linewidth=1.2)
    ax.add_patch(meas_box)
    ax.text(9.0, 5.6, "Pauli-Z\nMeas. $\\langle Z_i \\rangle$", ha='center', va='center', fontsize=7.5, fontweight='bold', color='#880e4f')

    # Qubit wires and gates
    q_ys = [4.9, 4.0, 3.1, 2.2]
    
    for i, qy in enumerate(q_ys):
        # Qubit line from x=4.1 to x=9.6
        ax.plot([4.1, 9.5], [qy, qy], color='#37474f', linewidth=1.3, zorder=1)
        
        # State label |0> comfortably inside the purple box (x=4.35)
        ax.text(4.45, qy, f"$|0\\rangle_{i}$", ha='center', va='center', fontsize=9, fontweight='bold', color='#1a237e', zorder=3)
        
        # RY Embedding Gate (x: 4.95 -> 5.75)
        ry_gate = patches.Rectangle((4.95, qy-0.26), 0.8, 0.52, facecolor='#e1bee7', edgecolor='#6a1b9a', linewidth=1.1, zorder=2)
        ax.add_patch(ry_gate)
        ax.text(5.35, qy, f"$R_Y(\\pi x_{i})$", ha='center', va='center', fontsize=7.5, color='#111111', zorder=4)

        # Unitary: Rotation Gate block (x: 6.25 -> 7.05)
        rot_gate = patches.Rectangle((6.25, qy-0.26), 0.82, 0.52, facecolor='#ba68c8', edgecolor='#4a148c', linewidth=1.1, zorder=2)
        ax.add_patch(rot_gate)
        ax.text(6.66, qy, f"$R(\\theta_{i})$", ha='center', va='center', fontsize=7.5, color='white', fontweight='bold', zorder=4)

        # Measurement gate (x: 8.65 -> 9.35)
        m_gate = patches.Rectangle((8.65, qy-0.26), 0.7, 0.52, facecolor='#f8bbd0', edgecolor='#880e4f', linewidth=1.1, zorder=2)
        ax.add_patch(m_gate)
        ax.text(9.0, qy, r"$\langle Z \rangle$", ha='center', va='center', fontsize=8, fontweight='bold', color='#880e4f', zorder=4)

    # Entangling CNOT gates placed cleanly in the gap (x: 7.3 -> 8.0)
    # CNOT 0 -> 1
    ax.plot([7.45, 7.45], [4.9, 4.0], color='#212121', linewidth=1.6, zorder=3)
    ax.plot(7.45, 4.9, 'o', color='#212121', markersize=5, zorder=5)
    ax.plot(7.45, 4.0, 'o', markerfacecolor='white', markeredgecolor='#212121', markersize=8, markeredgewidth=1.6, zorder=5)
    ax.text(7.45, 4.0, '+', ha='center', va='center', fontsize=9, fontweight='bold', color='#212121', zorder=6)

    # CNOT 2 -> 3
    ax.plot([7.45, 7.45], [3.1, 2.2], color='#212121', linewidth=1.6, zorder=3)
    ax.plot(7.45, 3.1, 'o', color='#212121', markersize=5, zorder=5)
    ax.plot(7.45, 2.2, 'o', markerfacecolor='white', markeredgecolor='#212121', markersize=8, markeredgewidth=1.6, zorder=5)
    ax.text(7.45, 2.2, '+', ha='center', va='center', fontsize=9, fontweight='bold', color='#212121', zorder=6)

    # CNOT 1 -> 2 (Ring Entanglement)
    ax.plot([7.9, 7.9], [4.0, 3.1], color='#212121', linewidth=1.6, zorder=3)
    ax.plot(7.9, 4.0, 'o', color='#212121', markersize=5, zorder=5)
    ax.plot(7.9, 3.1, 'o', markerfacecolor='white', markeredgecolor='#212121', markersize=8, markeredgewidth=1.6, zorder=5)
    ax.text(7.9, 3.1, '+', ha='center', va='center', fontsize=9, fontweight='bold', color='#212121', zorder=6)

    # =============================================================
    # 3. FEATURE MAPS BLOCK (x: 10.2 -> 12.4, y: 2.2 -> 5.8)
    # =============================================================
    # Connecting 4 arrows from Qubit outputs to Feature Maps
    for qy in q_ys:
        ax.annotate('', xy=(10.2, qy), xytext=(9.7, qy),
                    arrowprops=dict(facecolor=c_purple, edgecolor=c_purple, width=1.2, headwidth=4.5, shrink=0.05))

    f_bg = patches.FancyBboxPatch((10.2, 2.1), 2.2, 3.9, boxstyle="round,pad=0.08,rounding_size=0.1",
                                 facecolor=c_light_blue, edgecolor=c_blue, linewidth=1.6)
    ax.add_patch(f_bg)
    ax.text(11.3, 5.65, "Quantum Feature Maps", ha='center', va='center', fontsize=9.5, fontweight='bold', color=c_blue)
    ax.text(11.3, 5.35, r"$4 \times 14 \times 14$ Channels", ha='center', va='center', fontsize=8, color='#333333')

    # 4 stacked visual feature channels
    map_colors = ['#bbdefb', '#90caf9', '#64b5f6', '#42a5f5']
    for idx, mc in enumerate(map_colors):
        ox = 10.55 + idx * 0.18
        oy = 2.45 + idx * 0.26
        map_rect = patches.Rectangle((ox, oy), 1.0, 1.0, facecolor=mc, edgecolor=c_blue, linewidth=1.1, alpha=0.9, zorder=3+idx)
        ax.add_patch(map_rect)
        ax.text(ox+0.5, oy+0.5, f"$F_{idx}$", ha='center', va='center', fontsize=8, fontweight='bold', color='#0d47a1', zorder=10)

    ax.text(11.3, 2.3, "Flatten (784 Dim)", ha='center', va='top', fontsize=8, color='#1565c0', fontweight='bold')

    # Arrow to classifier
    ax.annotate('', xy=(13.0, 3.8), xytext=(12.5, 3.8),
                arrowprops=dict(facecolor=c_gray, edgecolor=c_gray, width=2.0, headwidth=7, shrink=0.05))

    # =============================================================
    # 4. SYMMETRICAL CLASSIFIER HEAD (x: 13.1 -> 16.4, y: 1.8 -> 6.2)
    # =============================================================
    c_bg = patches.FancyBboxPatch((13.1, 1.8), 3.3, 4.4, boxstyle="round,pad=0.1,rounding_size=0.15",
                                 facecolor=c_light_green, edgecolor=c_green, linewidth=2.0)
    ax.add_patch(c_bg)
    ax.text(14.75, 5.85, "Symmetrical Classifier Head", ha='center', va='center', fontsize=10.5, fontweight='bold', color=c_green)
    ax.text(14.75, 5.5, "(100% Identical Architecture)", ha='center', va='center', fontsize=7.5, color='#2e7d32')

    # BatchNorm box
    bn_box = patches.Rectangle((13.35, 2.4), 1.2, 2.6, facecolor='#ffffff', edgecolor=c_green, linewidth=1.2)
    ax.add_patch(bn_box)
    ax.text(13.95, 4.3, "BatchNorm", ha='center', va='center', fontsize=8, fontweight='bold')
    ax.text(13.95, 3.7, "+ ReLU", ha='center', va='center', fontsize=8, color='#555555')
    ax.text(13.95, 2.9, "8 params", ha='center', va='center', fontsize=7.5, color='#2e7d32', fontweight='bold')

    ax.annotate('', xy=(14.9, 3.7), xytext=(14.6, 3.7),
                arrowprops=dict(facecolor=c_green, edgecolor=c_green, width=1.2, headwidth=4))

    # Linear box
    fc_box = patches.Rectangle((14.95, 2.4), 1.3, 2.6, facecolor='#ffffff', edgecolor=c_green, linewidth=1.2)
    ax.add_patch(fc_box)
    ax.text(15.6, 4.4, "Linear FC", ha='center', va='center', fontsize=8.5, fontweight='bold')
    ax.text(15.6, 3.9, r"$784 \to K$", ha='center', va='center', fontsize=8, color='#333333')
    ax.text(15.6, 3.1, "Breast: 1,570 p\nOCT: 3,140 p", ha='center', va='center', fontsize=7, color='#1b5e20')

    # Output prediction arrow
    ax.annotate('', xy=(17.0, 3.7), xytext=(16.5, 3.7),
                arrowprops=dict(facecolor='darkred', edgecolor='darkred', width=2.0, headwidth=7, shrink=0.05))
    
    # Diagnosis outcome box
    ax.text(17.1, 4.4, "Clinical Output", ha='left', va='center', fontsize=9, fontweight='bold', color='darkred')
    ax.text(17.1, 3.8, "• BreastMNIST:\n  Malignant vs. Benign\n• OCTMNIST:\n  4 Retinal Classes", ha='left', va='center', fontsize=7.5, color='#212121')

    # =============================================================
    # 5. BOTTOM BAR: SYMMETRICAL CLASSICAL BASELINE COMPARISON
    # =============================================================
    cl_bg = patches.FancyBboxPatch((0.4, 0.25), 17.2, 0.95, boxstyle="round,pad=0.06,rounding_size=0.1",
                                  facecolor=c_light_orange, edgecolor=c_orange, linewidth=1.5)
    ax.add_patch(cl_bg)
    ax.text(0.65, 0.75, "Classical Baseline (Symmetrical Minimum CNN):", fontsize=8.5, fontweight='bold', color='#d84315')
    ax.text(6.0, 0.75, r"$\mathrm{Conv2D}(c_{\mathrm{in}}=1, c_{\mathrm{out}}=4, k=2, s=2)$" + " (20 params)", fontsize=8.5, color='#bf360c')
    ax.text(11.2, 0.75, r"$\to 4 \times 14 \times 14 \text{ Feature Maps}$", fontsize=8.5, color=c_blue)
    ax.text(14.5, 0.75, r"$\to \text{Identical Classifier Head}$", fontsize=8.5, color=c_green)
    ax.text(0.65, 0.45, "★ Key Fairness Guarantee: Same parameter count in head, same feature dimension (784), isolated feature extractor comparison.", 
            fontsize=7.5, style='italic', color='#424242')

    plt.tight_layout()
    
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, "Fig1_quanvolution_pipeline.png")
    out_pdf = os.path.join(out_dir, "Fig1_quanvolution_pipeline.pdf")
    
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.close()
    print(f"Successfully generated clean Figure 1 (PNG 300 DPI): {out_png}")
    print(f"Successfully generated clean Figure 1 (Vector PDF) : {out_pdf}")

if __name__ == '__main__':
    draw_pipeline()
