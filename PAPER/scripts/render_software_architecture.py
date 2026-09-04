# -*- coding: utf-8 -*-
"""render_software_architecture.py — Hình 3.3: Kiến trúc phần mềm đề tài (4 tầng)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(14, 8.6), dpi=150)
ax.set_xlim(0, 100); ax.set_ylim(0, 62); ax.axis('off')

def box(x, y, w, h, title, lines, fc, ec='#2F4F4F', ts=11.5, ls=9.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.5',
                                linewidth=1.4, edgecolor=ec, facecolor=fc))
    ax.text(x + w/2, y + h - 2.6, title, ha='center', va='center',
            fontsize=ts, fontweight='bold', color='#1a1a1a')
    for i, line in enumerate(lines):
        ax.text(x + w/2, y + h - 6.4 - i*3.1, line, ha='center', va='center',
                fontsize=ls, color='#262626')

def arrow(x1, y1, x2, y2, label='', lx=0, ly=0, style='-|>'):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=16, linewidth=1.5, color='#4a4a4a'))
    if label:
        ax.text((x1+x2)/2 + lx, (y1+y2)/2 + ly, label, ha='center', va='center',
                fontsize=9, color='#333333', style='italic')

# Tầng 1 — Dữ liệu
box(4, 50, 26, 10, 'TẦNG DỮ LIỆU — src/data',
    ['medmnist_loader.py: BreastMNIST / OCTMNIST',
     'split cố định 546/78/156 · 3500/500/1000',
     'chuẩn hóa [0,1] · seed cố định'], '#DCEBFA')
box(36, 50, 27, 10, 'PRECOMPUTE — mạch tĩnh',
    ['circuits.py: 6 mạch (basic/strongly/random × L1,L2)',
     'precompute_features.py → quantum_features/*.pt',
     '196 patches/ảnh × 4 qubit → 4 kênh ⟨Z⟩'], '#DCEBFA')

# Tầng 2 — Mô hình
box(4, 36, 43, 10, 'TẦNG MÔ HÌNH CỔ ĐIỂN — src/models',
    ['classical_cnn.py · SymmetricalMinimumCNN',
     'Conv2D(1→4, 2×2, s2) → BN(4) → ReLU → Linear(784→K)',
     '20 tham số kernel — baseline đối xứng 1:1'], '#DDF3DD')
box(53, 36, 43, 10, 'TẦNG MÔ HÌNH LƯỢNG TỬ — src/models',
    ['quantum_model.py · QuanvolutionClassifier (fixed, head như CNN)',
     'trainable_quanv.py · TorchLayer differentiable (backprop)',
     'kernel: 0 (fixed) / 4–24 (trainable) tham số'], '#DDF3DD')

# Tầng 3 — Thí nghiệm
box(4, 21, 62, 10, 'TẦNG THÍ NGHIỆM — src/experiments',
    ['run_gd3.py / trainable_experiment.py: 10 seeds × 20 epochs × 6 metrics',
     'Adam lr kép (0.001 cổ điển / 0.01 lượng tử) · CE loss · batch 32',
     'best-val checkpoint · paired t-test + Wilcoxon + CI95 + Cohen\u2019s d'], '#FDEBD0')
box(72, 21, 24, 10, 'QA — PAPER/scripts',
    ['reconcile_verify.py (canonical ddof=1)',
     'final_gate_audit.py (đối chiếu PDF)',
     'normalize_gd3.py'], '#FDEBD0')

# Tầng 4 — Đầu ra
box(4, 6, 43, 10, 'ĐẦU RA — results/',
    ['full_trainable_breastmnist.json (50 runs)',
     'full_trainable_octmnist.json (60 runs)',
     'reconciliation_canonical.json (nguồn số chuẩn)'], '#E8DFF5')
box(53, 6, 43, 10, 'ĐẦU RA — tài liệu & demo',
    ['regenerate_figs_bigfont.py → biểu đồ 300 DPI',
     'KLTN_draft_full.docx · bài báo SOICT (tag soict-submission-v4)',
     'gd4_defense_demo.ipynb · demo_defense_backup.mp4'], '#E8DFF5')

# Arrows
arrow(30, 55, 36, 55, 'ảnh [0,1]')
arrow(17, 50, 17, 46, 'feature 4×14×14')
arrow(74.5, 50, 74.5, 46, 'gradient end-to-end')
arrow(25.5, 36, 25.5, 31, 'predictions')
arrow(74.5, 36, 74.5, 31)
arrow(66, 21, 66, 16, 'JSON')
arrow(47, 11, 53, 11)
arrow(25.5, 16, 25.5, 21, 'audit ngược', style='-|>')

plt.tight_layout()
fig.savefig('GD4/fig_software_architecture.png', bbox_inches='tight', facecolor='white')
print('saved GD4/fig_software_architecture.png')
