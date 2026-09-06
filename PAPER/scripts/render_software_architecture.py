# -*- coding: utf-8 -*-
"""render_software_architecture.py — Hình 3.3: Kiến trúc phần mềm đề tài (4 tầng).
Thiết kế tối ưu theo yêu cầu:
1. Nội dung trong hộp căn giữa (center) hoàn hảo theo cả trục ngang (X) lẫn dọc (Y).
2. Viết hoa chữ cái đầu câu cho tất cả các dòng nội dung.
3. Tăng nhẹ font-size trong hộp và trên mũi tên để dễ đọc rõ ràng.
4. Đưa chữ ở các mũi tên vào khung chữ nhật bo góc nhẹ (badge/pill), có viền và nền sáng để nổi bật,
   tránh cảm giác chữ bị văng/lệch khỏi bảng.
5. Riêng nhãn "Tích hợp số liệu & đồ thị": chỉ viết hoa đầu câu và tăng font-size, không đóng khung viền.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(16, 10.2), dpi=300)
ax.set_xlim(0, 100); ax.set_ylim(0, 74); ax.axis('off')

def box(x, y, w, h, title, lines, fc, ec='#3A506B', ts=12.2, ls=10.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.6',
                                linewidth=1.6, edgecolor=ec, facecolor=fc))
    
    # Tiêu đề hộp
    ax.text(x + w/2, y + h - 2.6, title, ha='center', va='center',
            fontsize=ts, fontweight='bold', color='#0F172A')
    
    # Căn giữa hoàn hảo vùng nội dung
    n = len(lines)
    content_area_top = y + h - 4.8
    content_area_bottom = y + 1.2
    content_center_y = (content_area_top + content_area_bottom) / 2
    
    line_gap = 2.75
    total_text_h = (n - 1) * line_gap
    start_y = content_center_y + total_text_h / 2
    
    for i, line in enumerate(lines):
        curr_y = start_y - i * line_gap
        ax.text(x + w/2, curr_y, line, ha='center', va='center',
                fontsize=ls, color='#1E293B')

def arrow(x1, y1, x2, y2, label='', lx=0, ly=0, style='-|>', color='#475569', lw=1.8, badge=True):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=16, linewidth=lw, color=color))
    if label:
        pos_x = (x1 + x2) / 2 + lx
        pos_y = (y1 + y2) / 2 + ly
        if badge:
            bbox_props = dict(boxstyle='round,pad=0.35,rounding_size=0.25',
                              fc='#FFFFFF', ec='#94A3B8', lw=1.1, alpha=0.98)
        else:
            bbox_props = dict(boxstyle='square,pad=0.2', fc='#FFFFFF', ec='none', alpha=0.9)
            
        ax.text(pos_x, pos_y, label, ha='center', va='center',
                fontsize=9.8, color='#0F172A', fontweight='semibold',
                bbox=bbox_props)

# ---------------- TẦNG 1: DỮ LIỆU & TIỀN XỬ LÝ ----------------
box(3, 58, 43, 12, 'TẦNG DỮ LIỆU — src/data',
    ['Medmnist_loader.py: BreastMNIST & OCTMNIST',
     'Phân chia tập: Train / Val / Test (Cố định theo chuẩn MedMNIST)',
     'Chuẩn hóa pixel [0, 1] · Quản lý seed độc lập toàn phần'], '#E0EDFD')

box(54, 58, 43, 12, 'PRECOMPUTE ĐẶC TRƯNG TĨNH — src/data',
    ['Circuits.py: Định nghĩa 6 mạch (Basic / Strongly / Random × L1, L2)',
     'Precompute_features.py → data/quantum_features/*.pt',
     '196 patches / ảnh × mạch 4-qubit → 4 kênh kỳ vọng ⟨Z_i⟩ ∈ [−1, 1]'], '#E0EDFD')

# ---------------- TẦNG 2: MÔ HÌNH (ĐỐI XỨNG 1:1) ----------------
box(3, 39.5, 43, 12, 'TẦNG MÔ HÌNH CỔ ĐIỂN — src/models',
    ['Classical_cnn.py · SymmetricalMinimumCNN',
     'Conv2D(1→4, kernel 2×2, s2) → BatchNorm2d(4) → ReLU → Linear(784→K)',
     '20 tham số kernel (có bias) — Baseline đối xứng nghiêm ngặt 1:1'], '#DCFCE7')

box(54, 39.5, 43, 12, 'TẦNG MÔ HÌNH LƯỢNG TỬ — src/models',
    ['Quantum_model.py · QuanvolutionClassifier (Head đối xứng cổ điển)',
     'Trainable_quanv.py · TorchLayer khả vi (Analytic statevector backprop)',
     'Tham số kernel: 0 (Mạch tĩnh) / 4–24 (Mạch tự học)'], '#DCFCE7')

# ---------------- TẦNG 3: THÍ NGHIỆM & KIỂM THỬ ----------------
box(3, 21, 62, 12, 'TẦNG THÍ NGHIỆM — src/experiments',
    ['Run_gd3.py / Trainable_experiment.py: Ma trận 3 tầng (10 seeds × 20 epochs × 6 metrics)',
     'Optimizer Adam kép: lr = 0.001 (Head cổ điển) / lr = 0.01 (Góc quay θ lượng tử)',
     'Checkpointing theo best val ROC-AUC · Đánh giá test trên checkpoint tối ưu'], '#FEF3C7')

box(70, 21, 27, 12, 'KIỂM TOÁN DỮ LIỆU — PAPER/scripts',
    ['Reconcile_verify.py (Canonical ddof=1)',
     'Final_gate_audit.py (Đối soát PDF)',
     'Kiểm định: t-test, Wilcoxon, Cohen’s d'], '#FEF3C7')

# ---------------- TẦNG 4: ĐẦU RA & TÀI LIỆU ----------------
box(3, 2.5, 43, 12, 'KẾT QUẢ THỰC NGHIỆM — results/',
    ['Full_trainable_breastmnist.json (50 runs = 5 mô hình × 10 seeds)',
     'Full_trainable_octmnist.json (60 runs = 6 mô hình × 10 seeds)',
     'Reconciliation_canonical.json (Nguồn số liệu chân lý ddof=1)'], '#F3E8FF')

box(54, 2.5, 43, 12, 'XUẤT BẢN, BẢO VỆ & HỆ THỐNG DEMO',
    ['Regenerate_figs_bigfont.py → Trực quan hóa 9 hình độ phân giải 300 DPI',
     'KLTN_draft_full.docx · Bài báo SOICT 2026 (Tag: soict-submission-v4)',
     'Gd4_defense_demo.ipynb · Video backup thuyết minh (104 giây)'], '#F3E8FF')

# ==========================================
# CÁC MŨI TÊN LIÊN KẾT (ARROWS & BADGES)
# ==========================================
# 1. Tầng 1 ngang: Dữ liệu -> Precompute
arrow(46, 64, 54, 64, label='Ảnh [0, 1]', ly=2.2, badge=True)

# 2. Tầng 1 -> Tầng 2 (Ảnh nạp vào CNN)
arrow(24.5, 58, 24.5, 51.5, label='Ảnh 28×28 [0, 1]', lx=-6.8, badge=True)

# 3. Tầng 1 -> Tầng 2 (Đặc trưng vào mô hình lượng tử)
arrow(75.5, 58, 75.5, 51.5, label='Tensor đặc trưng 4×14×14', lx=8.6, badge=True)

# 4. Tầng 2 -> Tầng 3 (Mô hình cổ điển -> Thí nghiệm)
arrow(24.5, 39.5, 24.5, 33, label='Forward / Loss cổ điển', lx=-7.2, badge=True)

# 5. Tầng 2 -> Tầng 3 (Mô hình lượng tử -> Thí nghiệm: badge đặt ở chính giữa đường xiên, nâng ly lên một chút để thoáng viền)
arrow(62, 39.5, 45, 33, label='Backward vi phân statevector', lx=4.5, ly=2.2, badge=True)

# 6. Tầng 3 -> Tầng 4 (Thí nghiệm -> JSON kết quả)
arrow(24.5, 21, 24.5, 14.5, label='Xuất kết quả raw per-seed', lx=-8.0, badge=True)

# 7. Tầng 3 -> Tầng 3 (Thí nghiệm -> Kiểm toán QA)
arrow(65, 27, 70, 27, label='File JSON', ly=2.2, badge=True)

# 8. Tầng 4 ngang: Kết quả JSON -> Xuất bản / Demo (Riêng chữ này KHÔNG đóng khung badge)
arrow(46, 8.5, 54, 8.5, label='Tích hợp số liệu & đồ thị', ly=2.4, badge=False)

# 9. Tầng 4 lên Tầng 3 (Kiểm toán ngược từ kết quả chuẩn)
arrow(83.5, 14.5, 83.5, 21, label='Đối soát tính toàn vẹn', lx=8.2, badge=True)

plt.tight_layout()
output_path = 'D:/KhoaLuanTotNghiep/GD4/fig_software_architecture.png'
fig.savefig(output_path, bbox_inches='tight', facecolor='white')
fig.savefig('D:/KhoaLuanTotNghiep/PAPER/scripts/fig_software_architecture.png', bbox_inches='tight', facecolor='white')
print(f'Rendered cleanly to {output_path}')
