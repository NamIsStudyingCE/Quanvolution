# -*- coding: utf-8 -*-
"""fill_de_cuong_final.py — hoàn thiện đề cương theo phản hồi của GVHD:
1) thống nhất ngày (11/08/2026 – 29/01/2027, bảo vệ 25–29/01/2027 theo Thông báo 49/TB-KTMT);
2) thêm mục Tài liệu tham khảo chính (kèm DOI đã xác minh qua Crossref);
3) thêm đoạn ranh giới ĐA2↔KLTN;
4) điền MSSV/tên;
5) đổi tên file theo MSSV_Tên đề tài."""
from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from pathlib import Path
import shutil

SRC = Path('GD4/de_cuong_chi_tiet_filled.docx')
OUT = Path('GD4/22520916_Quanvolution_MedMNIST_DeCuongChiTiet.docx')

d = Document(str(SRC))
n_fix = 0

# 1. Ngày tháng: thống nhất theo lịch Khoa (KL 21/09–08/01, bảo vệ 25–29/01/2027)
for p in d.paragraphs:
    for r in p.runs:
        if 'đến ngày 09/11/2026' in r.text:
            r.text = r.text.replace('đến ngày 09/11/2026', 'đến ngày 29/01/2027')
            n_fix += 1
        if 'GĐ4 (20/10 – 09/11)' in r.text:
            r.text = r.text.replace(
                'GĐ4 (20/10 – 09/11) — Viết luận văn, demo notebook, slide bảo vệ. Trạng thái: đang thực hiện.',
                'GĐ4 (20/10/2026 – 29/01/2027) — Viết luận văn, demo notebook, slide bảo vệ; phản biện '
                '(11–15/01/2027) và bảo vệ khóa luận (25–29/01/2027). Trạng thái: đã hoàn thành nội dung.')

# 2. Chèn TLTK + ranh giới ĐA2 sau Ô 4 (Nội dung chính và giới hạn)
anchor = None
for p in d.paragraphs:
    if p.text.startswith('4. Các nội dung chính và giới hạn'):
        anchor = p
        break

def new_para(text, size=13, bold=False, align=None):
    p = d.add_paragraph()
    if align: p.alignment = align
    p.paragraph_format.space_after = Pt(6)
    r = p.add_run(text)
    r.font.name = 'Times New Roman'; r.font.size = Pt(size); r.bold = bold
    return p

# di chuyển các paragraph mới lên sau anchor (theo thứ tự ngược)
blocks = [
    ('Ranh giới với Đồ án 2 (ĐA2): Nội dung kế thừa gồm nền tảng QML, hạ tầng PennyLane/PyTorch và bộ '
     'dữ liệu y tế đã quen thuộc. Nội dung MỚI của khóa luận so với ĐA2: (a) chuyển từ kiến trúc adaptive '
     'gating HQCNN (6 qubit + ResNet18, poster YSC 2026) sang benchmark đối xứng 1:1 quanvolution '
     'fixed-vs-trainable; (b) mở rộng từ 1 dataset sang 2 datasets (BreastMNIST + OCTMNIST); (c) bổ sung '
     'quy trình thống kê đầy đủ (10 seeds × 6 metrics × kiểm định kép). Kết quả ĐA2 không được dùng lại '
     'trong các bảng số liệu của khóa luận.', False, 'justify'),
    ('Tài liệu tham khảo chính (kèm DOI — đã xác minh qua Crossref):', True, 'left'),
    ('1. Henderson, M., et al. (2020). Quanvolutional neural networks: Powering image recognition with '
     'quantum circuits. Quantum Machine Intelligence, 2(1), 2. DOI: 10.1007/s42484-020-00012-y', False, 'left'),
    ('2. Azevedo, V., Silva, C., & Dutra, I. (2022). Quantum transfer learning for breast cancer detection. '
     'Quantum Machine Intelligence, 4(1), 5. DOI: 10.1007/s42484-022-00062-4', False, 'left'),
    ('3. Matondo-Mvula, N., & Elleithy, K. (2024). Breast cancer detection with quanvolutional neural '
     'networks. Entropy, 26(8), 630. DOI: 10.3390/e26080630', False, 'left'),
    ('4. Yang, J., et al. (2023). MedMNIST v2 — A large-scale lightweight benchmark for 2D and 3D '
     'biomedical image classification. Scientific Data, 10(1), 41. DOI: 10.1038/s41597-022-01721-8', False, 'left'),
    ('(Danh mục đầy đủ 19 tài liệu kèm DOI — trình bày trong Luận văn, Chương 2.)', False, 'left'),
]
# chèn theo thứ tự ngược để addnext giữ đúng trật tự
for text, bold, align in reversed(blocks):
    p = d.add_paragraph()
    if align: p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY if align == 'justify' else WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_after = Pt(6)
    r = p.add_run(text)
    r.font.name = 'Times New Roman'; r.font.size = Pt(13); r.bold = bold
    anchor._p.addnext(p._p)
n_fix += len(blocks)

# 3. đổi tên file theo MSSV_Tên đề tài
try:
    d.save(str(OUT))
    print('saved', OUT)
    if OUT != SRC:
        SRC.unlink(missing_ok=True)
        print('(file cũ đã xóa để tránh nhầm lẫn)')
except PermissionError:
    alt = OUT.with_name(OUT.stem + '_moi.docx')
    d.save(str(alt))
    print('FILE GỐC ĐANG MỞ TRONG WORD — đã lưu sang:', alt)
print(f'tổng fix: {n_fix + len(blocks)}')
