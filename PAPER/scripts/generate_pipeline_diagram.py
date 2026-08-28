# -*- coding: utf-8 -*-
"""
generate_pipeline_diagram.py
----------------------------
Vẽ Sơ đồ Kiến trúc Pipeline Quanvolution (Figure 1) chuẩn công bố khoa học quốc tế (300 DPI).

Cải tiến toàn diện:
1. Ô Patch 2x2: Có khung viền đỏ nổi bật và mũi tên chú thích rõ ràng.
2. Chữ "Step Size: Stride=2": Nằm hoàn toàn bên dưới khối ảnh.
3. Khoảng cách giữa Khối 1 và Khối 2: Rộng rãi, chữ "196 Patches" nằm giữa thoáng đãng.
4. Nhãn |0>: Thụt vào trong Khối 2, đường dây qubit bắt đầu sau nhãn |0> (không đè lên chữ).
5. Mạch lượng tử: Cổng 1-qubit và CNOT được xếp riêng, đường CNOT không đè lên chữ.
6. 4 Mũi tên đầu ra từ Khối 2 sang Khối 3: Nằm hoàn toàn trong khoảng trống giữa 2 khối, không đè lên viền.
7. Khối Classical Baseline bên dưới: Được chia thành các khối mini-pipeline trực quan (Conv2D -> Feature Maps -> Classifier Head),
   thể hiện rõ ràng đối xứng 1:1 với nhánh Lượng tử phía trên.
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
    # Canvas dimensions
    fig, ax = plt.subplots(figsize=(18.2, 8.2), dpi=300)
    ax.set_xlim(0, 18.2)
    ax.set_ylim(0, 8.2)
    ax.axis('off')

    # Color palette (IEEE/Nature publication style)
    c_blue = '#1976d2'
    c_light_blue = '#e3f2fd'
    c_purple = '#7b1fa2'
    c_light_purple = '#f3e5f5'
    c_orange = '#e65100'
    c_light_orange = '#fff3e0'
    c_green = '#2e7d32'
    c_light_green = '#e8f5e9'
    c_gray = '#37474f'

    # Title
    ax.text(9.1, 7.85, "Figure 1: End-to-End Architecture of the Quanvolutional Neural Network (Quanvolution)",
            ha='center', va='center', fontsize=13, fontweight='bold', color='#111111')
    ax.text(9.1, 7.55, "Symmetrical 4-Qubit Quantum Kernel vs. Minimum Classical Convolutional Filter Baseline",
            ha='center', va='center', fontsize=10, style='italic', color='#555555')

    # =============================================================
    # 1. INPUT MEDICAL IMAGE (x: 0.4 -> 2.4, y: 2.5 -> 6.5)
    # =============================================================
    rect_img = patches.FancyBboxPatch((0.4, 2.5), 2.0, 4.2, boxstyle="round,pad=0.08,rounding_size=0.12",
                                      facecolor='#f8f9fa', edgecolor=c_gray, linewidth=1.5)
    ax.add_patch(rect_img)
    ax.text(1.4, 6.35, "Medical Image", ha='center', va='center', fontsize=10.5, fontweight='bold')
    ax.text(1.4, 6.05, r"$28 \times 28 \times 1$ (Grayscale)", ha='center', va='center', fontsize=8, color='#555555')
    
    # 2D Grid pixels representation
    x_grid = np.linspace(0.62, 2.18, 11)
    y_grid = np.linspace(3.2, 5.5, 11)
    for x in x_grid:
        for y in y_grid:
            val = np.sin((x-0.62)*4)*np.cos((y-3.2)*4)
            alpha = 0.15 + 0.45 * abs(val)
            ax.plot(x, y, 's', color='#263238', markersize=3.2, alpha=alpha)

    # Clearly highlighted 2x2 Local Sliding Patch
    patch_rect = patches.Rectangle((0.74, 4.6), 0.38, 0.48, linewidth=2.0, edgecolor='#d32f2f', facecolor='#ffcdd2', alpha=0.8, zorder=5)
    ax.add_patch(patch_rect)
    
    # Visual bracket connecting patch to description below
    ax.annotate("Sliding $2 \\times 2$ Patch\n(Kernel Window)", xy=(0.93, 4.6), xytext=(1.4, 2.85),
                ha='center', va='center', fontsize=7.5, fontweight='bold', color='#b71c1c',
                arrowprops=dict(arrowstyle="->", color='#d32f2f', lw=1.2, connectionstyle="arc3,rad=-0.2"))

    # Text below the image box (clean, outside the box)
    ax.text(1.4, 2.3, "Step Size: Stride = 2\nTotal: 196 Local Patches", ha='center', va='top', fontsize=7.5, color='#333333')

    # =============================================================
    # TRANSITION ARROW: BLOCK 1 -> BLOCK 2 (Widened spacing)
    # =============================================================
    ax.annotate('', xy=(3.8, 4.5), xytext=(2.6, 4.5),
                arrowprops=dict(facecolor=c_gray, edgecolor=c_gray, width=2.0, headwidth=7, shrink=0.05))
    ax.text(3.2, 4.85, "196 Patches\n" + r"$\mathbf{x} = (x_0, x_1, x_2, x_3)$", ha='center', va='bottom', fontsize=8, fontweight='bold', color='#111111')

    # =============================================================
    # 2. 4-QUBIT QUANTUM KERNEL FILTER (x: 3.9 -> 9.4, y: 1.8 -> 6.8)
    # =============================================================
    q_bg = patches.FancyBboxPatch((3.9, 1.8), 5.5, 5.0, boxstyle="round,pad=0.1,rounding_size=0.15",
                                 facecolor=c_light_purple, edgecolor=c_purple, linewidth=2.0)
    ax.add_patch(q_bg)
    ax.text(6.65, 6.55, "4-Qubit Quantum Kernel Filter", ha='center', va='center', fontsize=11.5, fontweight='bold', color=c_purple)

    # Sub-stages inside Quantum Kernel:
    # A) Angle Embedding (x: 4.8 -> 5.8)
    emb_box = patches.Rectangle((4.8, 2.0), 1.0, 4.15, facecolor='#ffffff', edgecolor='#ab47bc', linestyle='--', linewidth=1.2)
    ax.add_patch(emb_box)
    ax.text(5.3, 5.9, "Angle Embedding\n" + r"$R_Y(\pi x_i)|0\rangle$", 
            ha='center', va='center', fontsize=7.5, fontweight='bold', color='#6a1b9a')

    # B) Entangling Unitary (x: 6.0 -> 8.0)
    uni_box = patches.Rectangle((6.0, 2.0), 2.0, 4.15, facecolor='#ffffff', edgecolor=c_purple, linewidth=1.5)
    ax.add_patch(uni_box)
    ax.text(7.0, 5.9, "Entangling Unitary $U(\\theta)$\n(Fixed / Trainable)", 
            ha='center', va='center', fontsize=8, fontweight='bold', color=c_purple)

    # C) Pauli-Z Measurement (x: 8.2 -> 9.1)
    meas_box = patches.Rectangle((8.2, 2.0), 0.9, 4.15, facecolor='#ffffff', edgecolor='#c2185b', linestyle='--', linewidth=1.2)
    ax.add_patch(meas_box)
    ax.text(8.65, 5.9, "Pauli-Z\nMeas. $\\langle Z_i \\rangle$", ha='center', va='center', fontsize=7.5, fontweight='bold', color='#880e4f')

    # Qubit wires and gates
    q_ys = [5.2, 4.3, 3.4, 2.5]
    
    for i, qy in enumerate(q_ys):
        # State label |0> placed cleanly inside box with no overlapping wire
        ax.text(4.35, qy, f"$|0\\rangle_{i}$", ha='center', va='center', fontsize=9.5, fontweight='bold', color='#1a237e', zorder=3)

        # Qubit line starts AFTER |0> and ends right at measurement gate (x: 4.65 -> 9.0)
        ax.plot([4.65, 9.0], [qy, qy], color='#37474f', linewidth=1.3, zorder=1)
        
        # RY Embedding Gate (x: 4.9 -> 5.7)
        ry_gate = patches.Rectangle((4.9, qy-0.25), 0.8, 0.5, facecolor='#e1bee7', edgecolor='#6a1b9a', linewidth=1.1, zorder=2)
        ax.add_patch(ry_gate)
        ax.text(5.3, qy, f"$R_Y(\\pi x_{i})$", ha='center', va='center', fontsize=7.5, color='#111111', zorder=4)

        # Unitary: Rotation Gate block (x: 6.15 -> 6.95)
        rot_gate = patches.Rectangle((6.15, qy-0.25), 0.8, 0.5, facecolor='#ba68c8', edgecolor='#4a148c', linewidth=1.1, zorder=2)
        ax.add_patch(rot_gate)
        ax.text(6.55, qy, f"$R(\\theta_{i})$", ha='center', va='center', fontsize=7.5, color='white', fontweight='bold', zorder=4)

        # Measurement gate (x: 8.3 -> 9.0)
        m_gate = patches.Rectangle((8.3, qy-0.25), 0.7, 0.5, facecolor='#f8bbd0', edgecolor='#880e4f', linewidth=1.1, zorder=2)
        ax.add_patch(m_gate)
        ax.text(8.65, qy, r"$\langle Z \rangle$", ha='center', va='center', fontsize=8, fontweight='bold', color='#880e4f', zorder=4)

    # Entangling CNOT gates placed cleanly in the gap between Rotation and Measurement (x: 7.2 -> 7.8)
    # CNOT 0 -> 1
    ax.plot([7.25, 7.25], [5.2, 4.3], color='#212121', linewidth=1.6, zorder=3)
    ax.plot(7.25, 5.2, 'o', color='#212121', markersize=5, zorder=5)
    ax.plot(7.25, 4.3, 'o', markerfacecolor='white', markeredgecolor='#212121', markersize=8, markeredgewidth=1.6, zorder=5)
    ax.text(7.25, 4.3, '+', ha='center', va='center', fontsize=9, fontweight='bold', color='#212121', zorder=6)

    # CNOT 2 -> 3
    ax.plot([7.25, 7.25], [3.4, 2.5], color='#212121', linewidth=1.6, zorder=3)
    ax.plot(7.25, 3.4, 'o', color='#212121', markersize=5, zorder=5)
    ax.plot(7.25, 2.5, 'o', markerfacecolor='white', markeredgecolor='#212121', markersize=8, markeredgewidth=1.6, zorder=5)
    ax.text(7.25, 2.5, '+', ha='center', va='center', fontsize=9, fontweight='bold', color='#212121', zorder=6)

    # CNOT 1 -> 2 (Ring Entanglement)
    ax.plot([7.7, 7.7], [4.3, 3.4], color='#212121', linewidth=1.6, zorder=3)
    ax.plot(7.7, 4.3, 'o', color='#212121', markersize=5, zorder=5)
    ax.plot(7.7, 3.4, 'o', markerfacecolor='white', markeredgecolor='#212121', markersize=8, markeredgewidth=1.6, zorder=5)
    ax.text(7.7, 3.4, '+', ha='center', va='center', fontsize=9, fontweight='bold', color='#212121', zorder=6)

    # =============================================================
    # 3. FEATURE MAPS BLOCK (x: 10.3 -> 12.5, y: 2.3 -> 6.3)
    # =============================================================
    # 4 Output arrows connecting cleanly in the gap (from x=9.5 to x=10.2, perfectly spaced)
    for qy in q_ys:
        ax.annotate('', xy=(10.25, qy), xytext=(9.55, qy),
                    arrowprops=dict(facecolor=c_purple, edgecolor=c_purple, width=1.3, headwidth=5, shrink=0.05))

    f_bg = patches.FancyBboxPatch((10.3, 2.3), 2.2, 4.0, boxstyle="round,pad=0.08,rounding_size=0.1",
                                 facecolor=c_light_blue, edgecolor=c_blue, linewidth=1.6)
    ax.add_patch(f_bg)
    ax.text(11.4, 6.0, "Quantum Feature Maps", ha='center', va='center', fontsize=9.5, fontweight='bold', color=c_blue)
    ax.text(11.4, 5.7, r"$4 \times 14 \times 14$ Channels", ha='center', va='center', fontsize=8, color='#333333')

    # 4 stacked visual feature channels
    map_colors = ['#bbdefb', '#90caf9', '#64b5f6', '#42a5f5']
    for idx, mc in enumerate(map_colors):
        ox = 10.65 + idx * 0.18
        oy = 2.8 + idx * 0.26
        map_rect = patches.Rectangle((ox, oy), 1.0, 1.0, facecolor=mc, edgecolor=c_blue, linewidth=1.1, alpha=0.9, zorder=3+idx)
        ax.add_patch(map_rect)
        ax.text(ox+0.5, oy+0.5, f"$F_{idx}$", ha='center', va='center', fontsize=8, fontweight='bold', color='#0d47a1', zorder=10)

    ax.text(11.4, 2.55, "Flatten (784 Dim)", ha='center', va='top', fontsize=8, color='#1565c0', fontweight='bold')

    # Arrow to classifier (x: 12.6 -> 13.2)
    ax.annotate('', xy=(13.2, 4.1), xytext=(12.6, 4.1),
                arrowprops=dict(facecolor=c_gray, edgecolor=c_gray, width=2.0, headwidth=7, shrink=0.05))

    # =============================================================
    # 4. SYMMETRICAL CLASSIFIER HEAD (x: 13.3 -> 16.5, y: 2.0 -> 6.6)
    # =============================================================
    c_bg = patches.FancyBboxPatch((13.3, 2.0), 3.2, 4.6, boxstyle="round,pad=0.1,rounding_size=0.15",
                                 facecolor=c_light_green, edgecolor=c_green, linewidth=2.0)
    ax.add_patch(c_bg)
    ax.text(14.9, 6.3, "Symmetrical Classifier Head", ha='center', va='center', fontsize=10.5, fontweight='bold', color=c_green)
    ax.text(14.9, 5.95, "(100% Identical Architecture)", ha='center', va='center', fontsize=7.5, color='#2e7d32')

    # BatchNorm box
    bn_box = patches.Rectangle((13.55, 2.7), 1.1, 2.8, facecolor='#ffffff', edgecolor=c_green, linewidth=1.2)
    ax.add_patch(bn_box)
    ax.text(14.1, 4.8, "BatchNorm", ha='center', va='center', fontsize=8, fontweight='bold')
    ax.text(14.1, 4.1, "+ ReLU", ha='center', va='center', fontsize=8, color='#555555')
    ax.text(14.1, 3.2, "8 params", ha='center', va='center', fontsize=7.5, color='#2e7d32', fontweight='bold')

    ax.annotate('', xy=(15.0, 4.1), xytext=(14.7, 4.1),
                arrowprops=dict(facecolor=c_green, edgecolor=c_green, width=1.2, headwidth=4))

    # Linear box
    fc_box = patches.Rectangle((15.05, 2.7), 1.25, 2.8, facecolor='#ffffff', edgecolor=c_green, linewidth=1.2)
    ax.add_patch(fc_box)
    ax.text(15.68, 4.9, "Linear FC", ha='center', va='center', fontsize=8.5, fontweight='bold')
    ax.text(15.68, 4.3, r"$784 \to K$", ha='center', va='center', fontsize=8, color='#333333')
    ax.text(15.68, 3.4, "Breast: 1,570 p\nOCT: 3,140 p", ha='center', va='center', fontsize=7, color='#1b5e20')

    # Output prediction arrow
    ax.annotate('', xy=(17.0, 4.1), xytext=(16.55, 4.1),
                arrowprops=dict(facecolor='darkred', edgecolor='darkred', width=2.0, headwidth=7, shrink=0.05))
    
    # Diagnosis outcome box
    ax.text(17.1, 4.7, "Clinical Output", ha='left', va='center', fontsize=9, fontweight='bold', color='darkred')
    ax.text(17.1, 4.1, "• BreastMNIST:\n  Malignant vs. Benign\n• OCTMNIST:\n  4 Retinal Classes", ha='left', va='center', fontsize=7.5, color='#212121')

    # =============================================================
    # 5. BOTTOM SECTION: REDESIGNED STRUCTURED CLASSICAL BASELINE
    # =============================================================
    # Outer frame for classical baseline
    cl_bg = patches.FancyBboxPatch((0.4, 0.2), 17.4, 1.5, boxstyle="round,pad=0.08,rounding_size=0.12",
                                  facecolor=c_light_orange, edgecolor=c_orange, linewidth=1.8)
    ax.add_patch(cl_bg)
    
    # Header tag for Classical Baseline
    ax.text(0.65, 1.45, "Classical Baseline Workflow (Symmetrical Minimum CNN - 1:1 Comparative Equivalence):", 
            fontsize=9, fontweight='bold', color='#bf360c')

    # Mini-Pipeline Block 1: Conv2D Feature Extractor
    cl_box1 = patches.Rectangle((0.65, 0.4), 4.2, 0.85, facecolor='#ffffff', edgecolor=c_orange, linewidth=1.2)
    ax.add_patch(cl_box1)
    ax.text(2.75, 0.95, "Classical Feature Extractor", ha='center', va='center', fontsize=8, fontweight='bold', color='#d84315')
    ax.text(2.75, 0.65, r"$\mathrm{Conv2D}(c_{\mathrm{in}}=1, c_{\mathrm{out}}=4, k=2, s=2)$" + " (20 params)", 
            ha='center', va='center', fontsize=7.5, color='#333333')

    # Arrow 1 -> 2
    ax.annotate('', xy=(5.3, 0.82), xytext=(4.9, 0.82),
                arrowprops=dict(facecolor=c_orange, edgecolor=c_orange, width=1.5, headwidth=5))

    # Mini-Pipeline Block 2: Classical Feature Maps
    cl_box2 = patches.Rectangle((5.35, 0.4), 4.6, 0.85, facecolor='#ffffff', edgecolor=c_blue, linewidth=1.2)
    ax.add_patch(cl_box2)
    ax.text(7.65, 0.95, "Classical Feature Maps", ha='center', va='center', fontsize=8, fontweight='bold', color=c_blue)
    ax.text(7.65, 0.65, r"$4 \times 14 \times 14 \text{ Maps (784 Flattened Dimensions)}$", 
            ha='center', va='center', fontsize=7.5, color='#333333')

    # Arrow 2 -> 3
    ax.annotate('', xy=(10.4, 0.82), xytext=(10.0, 0.82),
                arrowprops=dict(facecolor=c_orange, edgecolor=c_orange, width=1.5, headwidth=5))

    # Mini-Pipeline Block 3: Shared Classifier Head
    cl_box3 = patches.Rectangle((10.45, 0.4), 5.0, 0.85, facecolor='#ffffff', edgecolor=c_green, linewidth=1.2)
    ax.add_patch(cl_box3)
    ax.text(12.95, 0.95, "Identical Classifier Head", ha='center', va='center', fontsize=8, fontweight='bold', color=c_green)
    ax.text(12.95, 0.65, "BatchNorm (8 p) + Linear (1,570 / 3,140 p)", 
            ha='center', va='center', fontsize=7.5, color='#333333')

    # Arrow 3 -> Outcome
    ax.annotate('', xy=(15.9, 0.82), xytext=(15.5, 0.82),
                arrowprops=dict(facecolor='darkred', edgecolor='darkred', width=1.5, headwidth=5))
    ax.text(16.0, 0.82, "Same Diagnosis\nOutput Classes", ha='left', va='center', fontsize=7.5, fontweight='bold', color='darkred')

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
