# -*- coding: utf-8 -*-
import os, shutil, sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

src_paper_dir = r"D:\KhoaLuanTotNghiep\PAPER"
dst_dir = r"D:\Overleaf_Quanvolution_Paper"

# Tạo thư mục đích
os.makedirs(dst_dir, exist_ok=True)
dst_fig_dir = os.path.join(dst_dir, "figures")
os.makedirs(dst_fig_dir, exist_ok=True)

# 1. Copy file LaTeX (cả main.tex và manuscript_ieee.tex để tiện dụng cho Overleaf)
tex_src = os.path.join(src_paper_dir, "manuscript_ieee.tex")
shutil.copy2(tex_src, os.path.join(dst_dir, "main.tex"))
shutil.copy2(tex_src, os.path.join(dst_dir, "manuscript_ieee.tex"))
print("Copied LaTeX source -> main.tex & manuscript_ieee.tex")

# 2. Copy toàn bộ hình ảnh trong PAPER/figures
src_fig_dir = os.path.join(src_paper_dir, "figures")
if os.path.exists(src_fig_dir):
    for f in os.listdir(src_fig_dir):
        s_file = os.path.join(src_fig_dir, f)
        d_file = os.path.join(dst_fig_dir, f)
        if os.path.isfile(s_file):
            shutil.copy2(s_file, d_file)
            print(f"  Copied figure: {f}")

# 3. Copy bản Markdown hoàn chỉnh
shutil.copy2(os.path.join(src_paper_dir, "MANUSCRIPT_FINAL_EN.md"), os.path.join(dst_dir, "MANUSCRIPT_FINAL_EN.md"))
shutil.copy2(os.path.join(src_paper_dir, "MANUSCRIPT_DRAFT_VI.md"), os.path.join(dst_dir, "MANUSCRIPT_DRAFT_VI.md"))
print("Copied Markdown manuscript references.")

# 4. Tạo file README hướng dẫn đóng gói
readme_content = """# OVERLEAF STANDALONE PAPER PACKAGE (IEEE TRANSACTIONS FORMAT)

Thư mục này là gói xuất bản độc lập (Standalone Export Package) dành riêng cho việc biên dịch bài báo trên Overleaf.

## Danh mục tệp tin:
1. `main.tex` & `manuscript_ieee.tex`: Mã nguồn LaTeX toàn văn chuẩn IEEEtran (Đã tối ưu hóa `tabularx`, không tràn lề, không đè chữ).
2. `figures/`: Chứa toàn bộ 9 tệp hình ảnh đồ họa 300 DPI và vector PDF:
   - `Fig1_quanvolution_pipeline.png` & `.pdf`: Sơ đồ kiến trúc đối xứng.
   - `Fig2_feature_comparison.png`: Bản đồ đặc trưng trích xuất.
   - `Fig3_breastmnist_benchmark.png` & `Fig3_octmnist_benchmark.png`: Cặp biểu đồ Bar charts so sánh hiệu năng 10 seeds.
   - `Fig4a_breastmnist_curves.png`, `Fig4b_octmnist_curves.png`: Đồ thị Loss & ROC AUC curves.
   - `Fig4c_theta_trajectories.png`, `Fig4d_gradient_norms.png`: Quỹ đạo góc quay và chuẩn gradient.
3. `MANUSCRIPT_FINAL_EN.md`: Bản thảo Tiếng Anh toàn văn chuẩn Markdown.
4. `MANUSCRIPT_DRAFT_VI.md`: Bản thảo Tiếng Việt toàn văn chuẩn Markdown.

## Cách sử dụng trên Overleaf:
- **Cách 1 (Nhanh nhất):** Chọn toàn bộ các file trong thư mục này -> Nén thành file `.zip` -> Vào Overleaf -> **New Project** -> **Upload Project** -> Chọn file `.zip` -> Bấm **Recompile**.
- **Cách 2:** Tạo dự án trống trên Overleaf -> Tạo thư mục `figures/` trên Overleaf -> Tải các file tương ứng lên đúng vị trí.
"""

with open(os.path.join(dst_dir, "README.md"), "w", encoding="utf-8") as f:
    f.write(readme_content)

print("\nSUCCESS: Overleaf standalone package is fully prepared at D:\\Overleaf_Quanvolution_Paper")
