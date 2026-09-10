# -*- coding: utf-8 -*-
"""
generate_final_faculty_slides.py
Phiên bản sửa triệt để:
1. Ảnh 1 (Slide 7 - Sơ đồ Pipeline): Khung được mở rộng rộng rãi, chữ nằm lọt lòng 100% trong khung viền, không tràn/cắt lề.
2. Ảnh 2 (Slide 6 - Feature Maps): Nhận xét nằm sát ngay dưới ảnh (khoảng cách sát nút 0.08"), chữ to đậm 11pt, đọc rõ ràng sắc nét.
3. Slide 2 (Mục lục): Nền thẻ sáng rõ ràng (#F2F6FC), viền xanh dương đậm (#2E75B6), chữ xanh Navy đậm (#1F4E79), số tròn trắng trên nền xanh, tương phản hoàn hảo.
4. Slide 14 (Hỏi & Đáp): Hộp cảm ơn và liên hệ màu trắng kem với viền xanh, chữ đậm sắc nét, tương phản cực cao trên nền tối.
5. Căn giữa, bố cục thoáng đãng, tuyệt đối không đè chữ hay tràn viền.
"""

import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

import pptx
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.dml.color import RGBColor

ROOT = Path(r"d:\KhoaLuanTotNghiep")
TPL_PATH = ROOT / "KTMT_KLTN_Phu luc 6_Mau bao cao bao ve.pptx"
OUT_PATH = ROOT / "slides" / "KLTN_BaoCao_KhoaKTMT_ChinhThuc.pptx"
MATH_DIR = ROOT / "PAPER" / "figures" / "math"
MATH_DIR.mkdir(parents=True, exist_ok=True)

# Images (Dùng bản rút gọn cho slide, không đụng vào bản gốc)
IMG_PIPELINE_SIMPLIFIED = str(ROOT / "PAPER" / "figures" / "Fig1_pipeline_slide_simplified.png")
IMG_FEATURES_SIMPLIFIED = str(ROOT / "PAPER" / "figures" / "Fig2_features_slide_simplified.png")
IMG_BREAST_BENCH = str(ROOT / "PAPER" / "figures" / "Fig3_breastmnist_benchmark.png")
IMG_OCT_BENCH = str(ROOT / "PAPER" / "figures" / "Fig3_octmnist_benchmark.png")
IMG_GRAD = str(ROOT / "PAPER" / "figures" / "Fig4d_gradient_norms.png")

# Palette
NAVY = RGBColor(31, 78, 121)       # #1F4E79
ACCENT = RGBColor(46, 117, 182)    # #2E75B6
DARK = RGBColor(30, 30, 30)        # #1E1E1E
MUTED = RGBColor(90, 100, 110)     # #5A646E
WHITE = RGBColor(255, 255, 255)
BG_CARD = RGBColor(248, 250, 253)  # #F8FAFD
BORDER_CARD = RGBColor(205, 218, 232) # #CDDAE8
HIGHLIGHT_BG = RGBColor(254, 250, 235)
HIGHLIGHT_BORDER = RGBColor(230, 180, 20)
SUCCESS_BG = RGBColor(235, 247, 238)
SUCCESS_BORDER = RGBColor(46, 125, 50)

FONT_HEADING = "Arial"
FONT_BODY = "Arial"

def render_math_formula(filename, latex_str, fontsize=15, color='#1F4E79'):
    """Vẽ công thức LaTeX chuẩn bằng Matplotlib"""
    out_path = MATH_DIR / filename
    fig = plt.figure(figsize=(5.5, 0.65), dpi=300)
    fig.text(0.5, 0.5, latex_str, fontsize=fontsize, ha='center', va='center', color=color)
    plt.axis('off')
    plt.savefig(str(out_path), bbox_inches='tight', transparent=True)
    plt.close()
    return str(out_path)

def clear_content_placeholder(slide, placeholder_idx=1):
    """Xóa placeholder nội dung mẫu để tự do vẽ layout trực quan"""
    for ph in list(slide.placeholders):
        if ph.placeholder_format.idx == placeholder_idx:
            sp = ph._element
            sp.getparent().remove(sp)

def add_clean_card(slide, left, top, width, height, title, items, bg_color=BG_CARD, border_color=BORDER_CARD, title_size=13.5, body_size=11.5):
    """Thẻ nội dung gọn gàng, căn chỉnh padding hoàn hảo"""
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = bg_color
    shape.line.color.rgb = border_color
    shape.line.width = Pt(1.2)

    tb = slide.shapes.add_textbox(left + Inches(0.16), top + Inches(0.14), width - Inches(0.32), height - Inches(0.28))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0

    p_title = tf.paragraphs[0]
    p_title.text = title
    p_title.font.name = FONT_HEADING
    p_title.font.size = Pt(title_size)
    p_title.font.bold = True
    p_title.font.color.rgb = NAVY

    for tag, desc in items:
        p = tf.add_paragraph()
        p.space_before = Pt(6)
        p.space_after = Pt(2)
        if tag:
            r_tag = p.add_run()
            r_tag.text = tag + " "
            r_tag.font.name = FONT_BODY
            r_tag.font.size = Pt(body_size)
            r_tag.font.bold = True
            r_tag.font.color.rgb = DARK
        if desc:
            r_desc = p.add_run()
            r_desc.text = desc
            r_desc.font.name = FONT_BODY
            r_desc.font.size = Pt(body_size)
            r_desc.font.color.rgb = DARK
    return shape

def add_stat_box(slide, left, top, width, height, big_num, label, desc="", num_color=NAVY, bg_color=BG_CARD):
    """Hộp chỉ số nổi bật căn giữa tuyệt đối"""
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = bg_color
    shape.line.color.rgb = BORDER_CARD
    shape.line.width = Pt(1.1)

    tb = slide.shapes.add_textbox(left + Inches(0.08), top + Inches(0.1), width - Inches(0.16), height - Inches(0.2))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0

    p1 = tf.paragraphs[0]
    p1.alignment = PP_ALIGN.CENTER
    p1.text = big_num
    p1.font.name = FONT_HEADING
    p1.font.size = Pt(23)
    p1.font.bold = True
    p1.font.color.rgb = num_color

    p2 = tf.add_paragraph()
    p2.alignment = PP_ALIGN.CENTER
    p2.text = label
    p2.font.name = FONT_BODY
    p2.font.size = Pt(11)
    p2.font.bold = True
    p2.font.color.rgb = DARK
    p2.space_before = Pt(3)

    if desc:
        p3 = tf.add_paragraph()
        p3.alignment = PP_ALIGN.CENTER
        p3.text = desc
        p3.font.name = FONT_BODY
        p3.font.size = Pt(9.5)
        p3.font.color.rgb = MUTED
        p3.space_before = Pt(2)
    return shape

def create_table(slide, left, top, width, height, headers, rows, col_widths=None, highlight_row_idx=None):
    """Bảng biểu chuẩn IEEE/Springer căn giữa trực quan"""
    rows_cnt = len(rows) + 1
    cols_cnt = len(headers)
    table_shape = slide.shapes.add_table(rows_cnt, cols_cnt, left, top, width, height)
    table = table_shape.table

    if col_widths and len(col_widths) == cols_cnt:
        for idx, w in enumerate(col_widths):
            table.columns[idx].width = w

    for c_idx, h in enumerate(headers):
        cell = table.cell(0, c_idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = cell.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        p.text = h
        p.font.name = FONT_HEADING
        p.font.size = Pt(10)
        p.font.bold = True
        p.font.color.rgb = WHITE

    for r_idx, row in enumerate(rows):
        is_highlight = (r_idx == highlight_row_idx)
        for c_idx, val in enumerate(row):
            cell = table.cell(r_idx + 1, c_idx)
            cell.fill.solid()
            if is_highlight:
                cell.fill.fore_color.rgb = RGBColor(233, 243, 255)
            elif r_idx % 2 == 1:
                cell.fill.fore_color.rgb = RGBColor(248, 250, 253)
            else:
                cell.fill.fore_color.rgb = WHITE

            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = cell.text_frame.paragraphs[0]
            p.text = str(val)
            p.font.name = FONT_BODY
            p.font.size = Pt(9.5)
            if is_highlight:
                p.font.bold = True
                p.font.color.rgb = NAVY
            else:
                p.font.color.rgb = DARK

            if c_idx == 0:
                p.alignment = PP_ALIGN.LEFT
            else:
                p.alignment = PP_ALIGN.CENTER

    return table_shape

def build_presentation():
    # 1. Render công thức toán
    img_eq_state = render_math_formula("eq_state.png", r"$|\psi(x)\rangle = \bigotimes_{i=0}^3 R_Y(\pi x_i)|0\rangle$", fontsize=16)
    img_eq_expect = render_math_formula("eq_expect.png", r"$F_i(u, v) = \langle\Phi(x, \theta)| Z_i |\Phi(x, \theta)\rangle \in [-1, 1]$", fontsize=15)

    prs = Presentation(str(TPL_PATH))
    slides = list(prs.slides)

    # =========================================================================
    # SLIDE 1: BÌA BÁO CÁO KHÓA LUẬN
    # =========================================================================
    s1 = slides[0]
    for s in s1.shapes:
        if s.is_placeholder and s.placeholder_format.idx == 0:
            tf = s.text_frame
            tf.clear()
            p1 = tf.paragraphs[0]
            p1.text = "NGHIÊN CỨU VÀ ỨNG DỤNG LỚP TÍCH CHẬP LƯỢNG TỬ (QUANVOLUTION)"
            p1.font.name = FONT_HEADING
            p1.font.size = Pt(23)
            p1.font.bold = True
            p1.font.color.rgb = NAVY
            
            p2 = tf.add_paragraph()
            p2.text = "TRONG PHÂN LOẠI ẢNH Y TẾ MEDMNIST"
            p2.font.name = FONT_HEADING
            p2.font.size = Pt(21)
            p2.font.bold = True
            p2.font.color.rgb = ACCENT
            p2.space_before = Pt(4)

        if not s.is_placeholder and s.has_text_frame and "GVHD:" in s.text_frame.text:
            tf = s.text_frame
            tf.clear()
            p_gv = tf.paragraphs[0]
            p_gv.text = "CÁN BỘ HƯỚNG DẪN:  TS. NGUYỄN DUY XUÂN BÁCH"
            p_gv.font.name = FONT_BODY
            p_gv.font.size = Pt(13)
            p_gv.font.bold = True
            p_gv.font.color.rgb = DARK

            p_sv = tf.add_paragraph()
            p_sv.text = "SINH VIÊN THỰC HIỆN:  NGUYỄN HẠO NAM  —  MSSV: 20520268"
            p_sv.font.name = FONT_BODY
            p_sv.font.size = Pt(13)
            p_sv.font.bold = True
            p_sv.font.color.rgb = DARK
            p_sv.space_before = Pt(4)

            p_time = tf.add_paragraph()
            p_time.text = "Ngành: Kỹ thuật Máy tính  ·  Khóa luận Tốt nghiệp Kỹ sư  ·  TP. Hồ Chí Minh, 2026"
            p_time.font.name = FONT_BODY
            p_time.font.size = Pt(11)
            p_time.font.color.rgb = MUTED
            p_time.space_before = Pt(4)

    # Status Badge
    badge = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.67), Inches(5.95), Inches(9.8), Inches(0.42))
    badge.fill.solid()
    badge.fill.fore_color.rgb = SUCCESS_BG
    badge.line.color.rgb = SUCCESS_BORDER
    badge.line.width = Pt(1)
    tf_b = badge.text_frame
    p_b = tf_b.paragraphs[0]
    p_b.alignment = PP_ALIGN.CENTER
    p_b.text = "★ CÔNG BỐ KHOA HỌC: BÀI BÁO ĐÃ NỘP & BÌNH DUYỆT TẠI SOICT 2026 (SPRINGER CCIS)"
    p_b.font.name = FONT_HEADING
    p_b.font.size = Pt(11)
    p_b.font.bold = True
    p_b.font.color.rgb = RGBColor(27, 94, 32)

    # =========================================================================
    # SLIDE 2: NỘI DUNG BÁO CÁO (AGENDA) - CẢI TIẾN TƯƠNG PHẢN RÕ NÉT 100%
    # =========================================================================
    s2 = slides[1]
    agenda_titles = [
        "Tổng quan đề tài & Động lực nghiên cứu",
        "Phương pháp nghiên cứu & Kiến trúc giải pháp",
        "Kết quả thực nghiệm & Đánh giá thống kê",
        "Kết luận, Đóng góp & Hướng phát triển"
    ]
    g_idx = 0
    for s in s2.shapes:
        if s.shape_type == pptx.enum.shapes.MSO_SHAPE_TYPE.GROUP:
            for ch in s.shapes:
                if "Rounded" in ch.name:
                    ch.fill.solid()
                    ch.fill.fore_color.rgb = RGBColor(242, 246, 252) # Nền xanh phấn sáng rõ
                    ch.line.color.rgb = ACCENT
                    ch.line.width = Pt(1.5)
                elif "Oval" in ch.name:
                    ch.fill.solid()
                    ch.fill.fore_color.rgb = NAVY
                    p = ch.text_frame.paragraphs[0]
                    p.font.name = FONT_HEADING
                    p.font.color.rgb = WHITE
                    p.font.bold = True
                    p.font.size = Pt(18)
                elif "TextBox" in ch.name:
                    ch.text_frame.clear()
                    p = ch.text_frame.paragraphs[0]
                    p.text = agenda_titles[g_idx]
                    p.font.name = FONT_HEADING
                    p.font.size = Pt(16)
                    p.font.bold = True
                    p.font.color.rgb = NAVY
            g_idx += 1

    # SLIDE 3: SECTION 01
    s3 = slides[2]

    # =========================================================================
    # SLIDE 4: 1. BỐI CẢNH & ĐỘNG LỰC
    # =========================================================================
    s4 = slides[3]
    clear_content_placeholder(s4)
    for s in s4.shapes:
        if s.is_placeholder and s.placeholder_format.idx == 0:
            s.text_frame.text = "1. Bối cảnh Nghiên cứu & Ba Vấn đề Lớn Cần Giải Quyết"

    w_c4 = Inches(3.8)
    h_c4 = Inches(4.35)
    top_c4 = Inches(2.25)

    add_clean_card(s4, Inches(0.76), top_c4, w_c4, h_c4,
                   "Vấn đề 1: So Sánh Chưa Công Bằng",
                   [
                       ("Thực trạng cũ:", "Các bài báo trước thường so sánh mô hình lượng tử với các mạng CNN cổ điển tùy tiện, chênh lệch hàng nghìn tham số (không biết ai mạnh hơn do đâu)."),
                       ("Cách đề tài xử lý:", "Thiết kế mô hình CNN cổ điển đối xứng 1:1: giữ nguyên toàn bộ phần phân loại, chỉ khác nhau đúng bộ lọc đặc trưng (chênh đúng 20 tham số)."),
                       ("Kết quả mang lại:", "Đo lường chính xác xem mạch lượng tử có thực sự trích xuất ảnh tốt hơn bộ lọc cổ điển hay không.")
                   ], border_color=ACCENT, title_size=14, body_size=11.5)

    add_clean_card(s4, Inches(4.76), top_c4, w_c4, h_c4,
                   "Vấn đề 2: Thiếu Đánh Giá Đáng Tin Cậy",
                   [
                       ("Thực trạng cũ:", "Nhiều nghiên cứu chỉ chạy thử 1–2 lần ngẫu nhiên rồi vội kết luận 'máy tính lượng tử vượt trội', dễ do yếu tố may mắn."),
                       ("Cách đề tài xử lý:", "Chạy lặp lại 10 lần độc lập (10 seeds cố định), tính độ lệch chuẩn và thực hiện kiểm định thống kê y khoa nghiêm ngặt (Paired t-test, Wilcoxon)."),
                       ("Kết quả mang lại:", "Khẳng định kết luận đưa ra là vững chắc, có căn cứ khoa học, không phải ngẫu nhiên.")
                   ], border_color=ACCENT, title_size=14, body_size=11.5)

    add_clean_card(s4, Inches(8.76), top_c4, w_c4, h_c4,
                   "Vấn đề 3: Mạch Cố Định Hay Tự Học?",
                   [
                       ("Thực trạng cũ:", "Chưa rõ việc cho các cổng lượng tử tự học (tối ưu góc) có thực sự đáng tiền so với việc cố định góc quay ngẫu nhiên hay không."),
                       ("Cách đề tài xử lý:", "So sánh trực tiếp 3 mô hình: CNN Cổ Điển vs Mạch Lượng Tử Cố Định vs Mạch Lượng Tử Tự Học, đo chính xác thời gian máy tính xử lý."),
                       ("Kết quả mang lại:", "Chỉ rõ khi nào nên dùng mạch cố định (tiết kiệm thời gian) và khi nào cần mạch tự học.")
                   ], border_color=ACCENT, title_size=14, body_size=11.5)

    # SLIDE 5: SECTION 02
    s5 = slides[4]

    # =========================================================================
    # SLIDE 6: 2. NGUYÊN LÝ LƯỢNG TỬ (DÙNG FIG2 CẢI TIẾN: NHẬN XÉT SÁT ẢNH, TO RÕ)
    # =========================================================================
    s6 = slides[5]
    clear_content_placeholder(s6)
    for s in s6.shapes:
        if s.is_placeholder and s.placeholder_format.idx == 0:
            s.text_frame.text = "2. Nguyên lý Hoạt động: Bộ Lọc Lượng Tử Nhìn Ảnh Như Thế Nào?"

    # Ảnh Fig2 cải tiến (rộng 11.8", cao 2.75", nhận xét nằm sát nút dưới ảnh cực đẹp)
    if os.path.exists(IMG_FEATURES_SIMPLIFIED):
        s6.shapes.add_picture(IMG_FEATURES_SIMPLIFIED, Inches(0.76), Inches(2.15), width=Inches(11.8), height=Inches(2.75))

    # Hàng dưới: Thẻ giải thích công thức (Trái) + Thẻ 3 loại mạch (Phải)
    w_bottom = Inches(5.8)
    h_bottom = Inches(1.68)
    top_bottom = Inches(5.02)

    box_math = s6.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.76), top_bottom, w_bottom, h_bottom)
    box_math.fill.solid()
    box_math.fill.fore_color.rgb = BG_CARD
    box_math.line.color.rgb = BORDER_CARD
    box_math.line.width = Pt(1.2)
    
    tb_m = s6.shapes.add_textbox(Inches(0.9), top_bottom + Inches(0.08), w_bottom - Inches(0.3), Inches(0.28))
    p_mt = tb_m.text_frame.paragraphs[0]
    p_mt.text = "Cách Chuyển Điểm Ảnh Thành Trạng Thái Lượng Tử"
    p_mt.font.name = FONT_HEADING
    p_mt.font.size = Pt(11.5)
    p_mt.font.bold = True
    p_mt.font.color.rgb = NAVY

    s6.shapes.add_picture(img_eq_state, Inches(1.0), top_bottom + Inches(0.40), width=Inches(5.2))
    s6.shapes.add_picture(img_eq_expect, Inches(0.95), top_bottom + Inches(0.98), width=Inches(5.3))

    add_clean_card(s6, Inches(6.76), top_bottom, w_bottom, h_bottom,
                   "Ba Cấu Trúc Mạch Lượng Tử Khảo Sát",
                   [
                       ("• Mạch Basic (Cơ bản):", "Xoay góc nhẹ và kết nối các qubit liền kề (gọn nhẹ, nhanh)."),
                       ("• Mạch Strongly (Vướng víu cao):", "Xoay đa chiều và kết nối chéo toàn bộ qubit (độ biểu đạt mạnh nhất)."),
                       ("• Mạch Random (Ngẫu nhiên):", "Góc xoay ngẫu nhiên cố định sẵn (0 cần học, tính cực nhanh).")
                   ], border_color=ACCENT, title_size=12, body_size=10.5)

    # =========================================================================
    # SLIDE 7: 2. THIẾT KẾ ĐỐI XỨNG (DÙNG FIG1 CẢI TIẾN: KHUNG RỘNG, CHỮ NẰM GỌN)
    # =========================================================================
    s7 = slides[6]
    clear_content_placeholder(s7)
    for s in s7.shapes:
        if s.is_placeholder and s.placeholder_format.idx == 0:
            s.text_frame.text = "2. Thiết kế Mô hình: Luồng Xử lý Từ Ảnh Đến Chẩn đoán"

    # Hình Sơ đồ Pipeline RÚT GỌN (rộng 11.8", cao 2.75", căn lề hoàn hảo)
    if os.path.exists(IMG_PIPELINE_SIMPLIFIED):
        s7.shapes.add_picture(IMG_PIPELINE_SIMPLIFIED, Inches(0.76), Inches(2.15), width=Inches(11.8), height=Inches(2.75))

    # Hàng dưới chia 2 thẻ giải thích rõ ràng, súc tích
    w_s7 = Inches(5.8)
    h_s7 = Inches(1.70)
    top_s7 = Inches(5.02)

    add_clean_card(s7, Inches(0.76), top_s7, w_s7, h_s7,
                   "Đối Xứng Tuyệt Đối Giữa Lượng Tử & Cổ Điển",
                   [
                       ("• Cùng đầu vào:", "Mỗi bước trích xuất 1 ô vuông 2×2 (4 điểm ảnh)."),
                       ("• Cùng phần phân loại:", "Mạng nơ-ron kết nối đầy đủ (1,570 tham số cho Breast, 3,140 cho OCT)."),
                       ("• Điểm khác biệt duy nhất:", "CNN dùng phép nhân tích chập (20 tham số), còn Quanv dùng mạch 4-qubit.")
                   ], border_color=ACCENT, title_size=12.5, body_size=11)

    add_clean_card(s7, Inches(6.76), top_s7, w_s7, h_s7,
                   "Kiểm Tra Tính Khả Thi Trên Phần Cứng Thật",
                   [
                       ("• Kiểm chứng đạo hàm:", "Đạo hàm tính bằng máy tính khớp chính xác với công thức phần cứng vật lý (Parameter-Shift Rule) với sai số siêu nhỏ < 4.1×10⁻⁸."),
                       ("• Ý nghĩa thực tiễn:", "Thuật toán này hoàn toàn sẵn sàng nạp thẳng vào máy tính lượng tử thực tế (như IBM Quantum) mà không bị lỗi logic.")
                   ], border_color=NAVY, title_size=12.5, body_size=11)

    # SLIDE 8: SECTION 03
    s8 = slides[7]

    # =========================================================================
    # SLIDE 9: 3. KẾT QUẢ BREASTMNIST
    # =========================================================================
    s9 = slides[8]
    clear_content_placeholder(s9)
    for s in s9.shapes:
        if s.is_placeholder and s.placeholder_format.idx == 0:
            s.text_frame.text = "3. Thử nghiệm BreastMNIST: Mạch Lượng Tử Thắng Thế Trên Dữ Liệu Nhỏ"

    headers_breast = ["Mô hình", "Tham số Kernel", "ROC-AUC (Khả năng xếp hạng)", "PR-AUC (Bắt ca bệnh)", "Độ ổn định"]
    rows_breast = [
        ["CNN Cổ Điển (Chuẩn)", "20 tham số", "0.8336 ± 0.0259", "0.9041 ± 0.0095", "Phân tán cao"],
        ["Mạch Lượng Tử Basic (Tĩnh)", "0 tham số (khóa)", "0.8521 ± 0.0095 ★", "0.9110 ± 0.0049", "Ổn định gấp 2.7 lần"],
        ["Mạch Lượng Tử Strongly (Tĩnh)", "0 tham số (khóa)", "0.8139 ± 0.0142", "0.9182 ± 0.0071 ★", "Bắt bệnh hiếm tốt nhất"],
        ["Mạch Lượng Tử Basic (Tự học)", "8 tham số", "0.8406 ± 0.0239", "0.9173 ± 0.0184", "Khá tốt"],
        ["Mạch Lượng Tử Strongly (Tự học)", "24 tham số", "0.8306 ± 0.0279", "0.9167 ± 0.0157", "Trung bình"]
    ]
    col_w_breast = [Inches(2.2), Inches(1.1), Inches(1.3), Inches(1.2), Inches(0.7)]
    create_table(s9, Inches(0.76), Inches(2.2), Inches(6.5), Inches(2.45), headers_breast, rows_breast, col_w_breast, highlight_row_idx=1)

    add_stat_box(s9, Inches(0.76), Inches(4.85), Inches(2.05), Inches(1.75), "0.8521", "Mạch Basic Tĩnh", "Quán quân phân loại (±0.0095)", num_color=ACCENT)
    add_stat_box(s9, Inches(2.98), Inches(4.85), Inches(2.05), Inches(1.75), "0.9182", "Mạch Strongly Tĩnh", "Bắt ca bệnh hiếm chuẩn nhất", num_color=SUCCESS_BORDER)
    add_stat_box(s9, Inches(5.20), Inches(4.85), Inches(2.05), Inches(1.75), "~2.7 Lần", "Ổn Định Vượt Trội", "Ít bị biến thiên hơn CNN cổ điển", num_color=NAVY)

    if os.path.exists(IMG_BREAST_BENCH):
        s9.shapes.add_picture(IMG_BREAST_BENCH, Inches(7.45), Inches(2.2), width=Inches(5.1), height=Inches(2.85))

    box_tk_b = s9.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.45), Inches(5.2), Inches(5.1), Inches(1.4))
    box_tk_b.fill.solid()
    box_tk_b.fill.fore_color.rgb = BG_CARD
    box_tk_b.line.color.rgb = ACCENT
    box_tk_b.line.width = Pt(1.2)
    tf_tk_b = box_tk_b.text_frame
    tf_tk_b.word_wrap = True
    tf_tk_b.margin_left = tf_tk_b.margin_right = Inches(0.12)
    p_tk_b = tf_tk_b.paragraphs[0]
    p_tk_b.text = "Ý nghĩa cốt lõi:"
    p_tk_b.font.name = FONT_HEADING
    p_tk_b.font.size = Pt(11)
    p_tk_b.font.bold = True
    p_tk_b.font.color.rgb = NAVY
    p_tk_b2 = tf_tk_b.add_paragraph()
    p_tk_b2.text = "Khi dữ liệu y tế ít (chỉ vài trăm ảnh) và lệch lớp, mạng CNN cổ điển rất dễ bị 'học vẹt' (overfitting). Mạch lượng tử cố định đóng vai trò như một bộ nẹp cấu trúc vững chắc, giúp mô hình học ổn định và chính xác hơn hẳn."
    p_tk_b2.font.name = FONT_BODY
    p_tk_b2.font.size = Pt(9.5)
    p_tk_b2.font.color.rgb = DARK
    p_tk_b2.space_before = Pt(3)

    # =========================================================================
    # SLIDE 10: 3. KẾT QUẢ OCTMNIST
    # =========================================================================
    s10 = slides[9]
    clear_content_placeholder(s10)
    for s in s10.shapes:
        if s.is_placeholder and s.placeholder_format.idx == 0:
            s.text_frame.text = "3. Thử nghiệm OCTMNIST: CNN Áp Đảo Khi Dữ Liệu Lớn & Nhiều Lớp"

    headers_oct = ["Mô hình", "Tham số Kernel", "ROC-AUC (Toàn diện)", "PR-AUC (Bắt bệnh)", "Độ chính xác"]
    rows_oct = [
        ["CNN Cổ Điển (Chiến thắng)", "20 tham số", "0.7505 ± 0.0240 ★", "0.4991 ± 0.0297 ★", "52.84% (Cao nhất)"],
        ["Mạch Lượng Tử Strongly (Tự học)", "24 tham số", "0.6922 ± 0.0199", "0.4137 ± 0.0142", "48.51%"],
        ["Mạch Lượng Tử Strongly (Tĩnh)", "0 tham số (khóa)", "0.6690 ± 0.0055", "0.3879 ± 0.0051", "47.04%"],
        ["Mạch Lượng Tử Basic (Tự học)", "8 tham số", "0.6339 ± 0.0259", "0.3473 ± 0.0203", "45.60%"],
        ["Mạch Lượng Tử Basic (Tĩnh)", "0 tham số (khóa)", "0.6277 ± 0.0125", "0.3432 ± 0.0076", "45.78%"]
    ]
    col_w_oct = [Inches(2.2), Inches(1.1), Inches(1.3), Inches(1.2), Inches(0.7)]
    create_table(s10, Inches(0.76), Inches(2.2), Inches(6.5), Inches(2.45), headers_oct, rows_oct, col_w_oct, highlight_row_idx=0)

    add_stat_box(s10, Inches(0.76), Inches(4.85), Inches(2.05), Inches(1.75), "+0.0583", "CNN Vượt Trội", "Áp đảo cả 6 chỉ số đánh giá", num_color=NAVY)
    add_stat_box(s10, Inches(2.98), Inches(4.85), Inches(2.05), Inches(1.75), "+0.0232", "Tự Học Có Tác Dụng", "Học góc giúp cải thiện mạch", num_color=SUCCESS_BORDER)
    add_stat_box(s10, Inches(5.20), Inches(4.85), Inches(2.05), Inches(1.75), "Ổn Định", "Học Mượt Mà", "Không bị kẹt đạo hàm (Barren)", num_color=ACCENT)

    if os.path.exists(IMG_OCT_BENCH):
        s10.shapes.add_picture(IMG_OCT_BENCH, Inches(7.45), Inches(2.2), width=Inches(5.1), height=Inches(2.85))

    box_tk_o = s10.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.45), Inches(5.05), Inches(5.1), Inches(1.55))
    box_tk_o.fill.solid()
    box_tk_o.fill.fore_color.rgb = BG_CARD
    box_tk_o.line.color.rgb = ACCENT
    box_tk_o.line.width = Pt(1.2)
    tf_tk_o = box_tk_o.text_frame
    tf_tk_o.word_wrap = True
    tf_tk_o.margin_left = tf_tk_o.margin_right = Inches(0.12)
    p_tk_o = tf_tk_o.paragraphs[0]
    p_tk_o.text = "Ý nghĩa cốt lõi:"
    p_tk_o.font.name = FONT_HEADING
    p_tk_o.font.size = Pt(11)
    p_tk_o.font.bold = True
    p_tk_o.font.color.rgb = NAVY
    p_tk_o2 = tf_tk_o.add_paragraph()
    p_tk_o2.text = "Khi bài toán khó hơn (4 loại bệnh mắt, 5,000 ảnh), mạch 4 qubit quá nhỏ bé nên bị 'nghẽn dung lượng', không chứa nổi đặc trưng phức tạp như CNN cổ điển. Tuy nhiên, việc cho mạch tự học đã giúp tăng điểm rõ rệt so với mạch tĩnh."
    p_tk_o2.font.name = FONT_BODY
    p_tk_o2.font.size = Pt(9.5)
    p_tk_o2.font.color.rgb = DARK
    p_tk_o2.space_before = Pt(3)

    # SLIDE 11: SECTION 04
    s11 = slides[10]

    # =========================================================================
    # SLIDE 12: 4. KẾT LUẬN & HƯỚNG MỞ RỘNG
    # =========================================================================
    s12 = slides[11]
    clear_content_placeholder(s12)
    for s in s12.shapes:
        if s.is_placeholder and s.placeholder_format.idx == 0:
            s.text_frame.text = "4. Ba Đúc Kết Thực Tế & Bốn Hướng Phát Triển Tiếp Theo"

    w_half = Inches(5.8)
    h_c12 = Inches(4.35)
    top_c12 = Inches(2.25)

    add_clean_card(s12, Inches(0.76), top_c12, w_half, h_c12,
                   "Ba Đúc Kết Khoa Học Thực Tế",
                   [
                       ("1. Không có 'Lượng tử luôn tốt hơn':", "Mạch lượng tử thắng thế rõ rệt khi ảnh ít và lệch lớp (như siêu âm ung thư vú), nhưng sẽ thua CNN khi bài toán có nhiều lớp và nhiều dữ liệu."),
                       ("2. Mạch cố định (0 cần học) cực kỳ hữu dụng:", "Không cần mất công huấn luyện mạch lượng tử phức tạp, mạch ngẫu nhiên vẫn trích xuất đặc trưng tốt, cho phép tính trước 1 lần để tiết kiệm thời gian."),
                       ("3. Cho mạch tự học có giới hạn:", "Tự học góc quay chỉ giúp cải thiện chính mạch đó tốt lên một chút, chứ không giúp mạch vượt qua giới hạn dung lượng phần cứng.")
                   ], border_color=ACCENT, title_size=14, body_size=11.5)

    add_clean_card(s12, Inches(6.76), top_c12, w_half, h_c12,
                   "Bốn Hướng Phát Triển Đầy Tiềm Năng",
                   [
                       ("1. Chạy thử trên máy tính lượng tử thật:", "Kết nối trực tiếp lên máy tính lượng tử IBM Quantum và áp dụng thuật toán khử nhiễu thực tế."),
                       ("2. Dùng GPU mở rộng số Qubit:", "Áp dụng công nghệ NVIDIA cuQuantum để mô phỏng mạch 8–16 qubits, giúp giải quyết bài toán lớn 97,000 ảnh OCT."),
                       ("3. Đổi mới cách nén dữ liệu:", "Nghiên cứu cách nạp nhiều pixel vào ít qubit hơn để tăng sức chứa thông tin của mạch."),
                       ("Công bố khoa học quốc tế:", "Toàn bộ nghiên cứu đã được viết thành bài báo khoa học và đang bình duyệt tại Hội nghị Quốc tế SOICT 2026 (Springer CCIS).")
                   ], border_color=NAVY, title_size=14, body_size=11.5)

    # =========================================================================
    # SLIDE 13: TÀI LIỆU THAM KHẢO
    # =========================================================================
    s13 = slides[12]
    clear_content_placeholder(s13)
    for s in s13.shapes:
        if s.is_placeholder and s.placeholder_format.idx == 0:
            s.text_frame.text = "Tài liệu Tham khảo Chọn lọc"

    add_clean_card(s13, Inches(0.76), Inches(2.25), Inches(11.8), Inches(4.35),
                   "Danh Mục Công Trình Khoa Học Trọng Tâm",
                   [
                       ("[1] Henderson et al. (2020)", "Quanvolutional Neural Networks. Quantum Machine Intelligence, 2(1), 2. (Bài báo gốc đề xuất ý tưởng Quanvolution)."),
                       ("[2] Cong, Choi, & Lukin (2019)", "Quantum Convolutional Neural Networks. Nature Physics, 15(12), 1273–1278. (Mạng tích chập lượng tử thuần túy)."),
                       ("[3] Schuld & Killoran (2022)", "Is Quantum Advantage the Right Goal for Quantum Machine Learning? PRX Quantum, 3(3), 030101. (Bàn về vai trò thực tế của máy học lượng tử)."),
                       ("[4] Yang et al. (2023)", "MedMNIST v2: A Large-Scale Lightweight Benchmark for Biomedical Image Classification. Scientific Data (Nature), 10(1), 41."),
                       ("[5] McClean et al. (2018)", "Barren Plateaus in Quantum Neural Network Training Landscapes. Nature Communications, 9(1), 4812."),
                       ("[6] Matondo-Mvula & Elleithy (2024)", "Breast Cancer Detection with Hybrid Quantum-Classical Convolutional Neural Networks. Entropy, 26(8), 630."),
                       ("Toàn bộ tài liệu đối chiếu:", "19 công trình quốc tế được trích dẫn và phân tích chi tiết trong cuốn Luận văn và Bài báo.")
                   ], border_color=BORDER_CARD, title_size=13.5, body_size=11)

    # =========================================================================
    # SLIDE 14: SECTION 05 - HỎI VÀ ĐÁP (Q&A) - SỬA LỖI TƯƠNG PHẢN HOÀN TOÀN
    # =========================================================================
    s14 = slides[13]
    for sh in list(s14.shapes):
        if sh.name == "TextBox 6" or (sh.has_text_frame and "TRÂN TRỌNG CẢM ƠN" in sh.text_frame.text):
            sp = sh._element
            sp.getparent().remove(sp)

    # Thêm Hộp thẻ màu trắng kem viền xanh Navy trang trọng trên nền tối
    box_qa = s14.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.81), Inches(4.35), Inches(9.8), Inches(1.85))
    box_qa.fill.solid()
    box_qa.fill.fore_color.rgb = WHITE
    box_qa.line.color.rgb = NAVY
    box_qa.line.width = Pt(2.0)
    
    tf_qa = box_qa.text_frame
    tf_qa.word_wrap = True
    tf_qa.margin_left = tf_qa.margin_right = Inches(0.2)
    tf_qa.margin_top = Inches(0.15)
    
    p_qa1 = tf_qa.paragraphs[0]
    p_qa1.alignment = PP_ALIGN.CENTER
    p_qa1.text = "TRÂN TRỌNG CẢM ƠN HỘI ĐỒNG THẦY CÔ VÀ CÁC BẠN ĐÃ LẮNG NGHE!"
    p_qa1.font.name = FONT_HEADING
    p_qa1.font.size = Pt(17)
    p_qa1.font.bold = True
    p_qa1.font.color.rgb = NAVY

    p_qa2 = tf_qa.add_paragraph()
    p_qa2.alignment = PP_ALIGN.CENTER
    p_qa2.text = "Mã nguồn & Dữ liệu tái lập: https://github.com/NamIsStudyingCE/Quanvolution  (Tag: soict-submission-v4)\nEmail liên hệ: ng.h.nam0802@gmail.com  ·  Giảng viên hướng dẫn: bachndx@uit.edu.vn"
    p_qa2.font.name = FONT_BODY
    p_qa2.font.size = Pt(12)
    p_qa2.font.bold = True
    p_qa2.font.color.rgb = DARK
    p_qa2.space_before = Pt(10)

    # =========================================================================
    # SLIDE 15: PHỤ LỤC PHÒNG VỆ
    # =========================================================================
    s15 = slides[14]
    clear_content_placeholder(s15)
    for s in s15.shapes:
        if s.is_placeholder and s.placeholder_format.idx == 0:
            s.text_frame.text = "Phụ lục Phòng vệ: Bằng chứng Thống kê & Kiểm chứng Kỹ thuật"

    headers_stat = ["Cặp đối đầu", "Chỉ số", "Chênh lệch (Δ)", "Kiểm định t-test", "Kiểm định Wilcoxon", "Mức độ vượt trội"]
    rows_stat = [
        ["Mạch Basic vs CNN (Breast)", "ROC-AUC", "+0.0186", "p = 0.0298 (*)", "p = 0.0254 (*)", "Vượt trội lớn (d = +0.815)"],
        ["Mạch Strongly vs CNN (Breast)", "PR-AUC", "+0.0140", "p = 0.0023 (**)", "p = 0.0059 (**)", "Rất lớn (d = +1.332)"],
        ["CNN vs Mạch Tự học (OCT)", "ROC-AUC", "+0.0583", "p ≈ 0.0001 (***)", "p < 0.001 (***)", "CNN áp đảo (d = +2.108)"],
        ["Tự học vs Tĩnh (OCT)", "ROC-AUC", "+0.0232", "p = 0.0090 (**)", "p = 0.0098 (**)", "Lớn (d = +1.050)"]
    ]
    col_w_stat = [Inches(1.8), Inches(0.9), Inches(0.9), Inches(1.0), Inches(1.1), Inches(0.9)]
    create_table(s15, Inches(0.76), Inches(2.2), Inches(6.6), Inches(2.2), headers_stat, rows_stat, col_w_stat, highlight_row_idx=0)

    add_clean_card(s15, Inches(0.76), Inches(4.55), Inches(6.6), Inches(2.05),
                   "Giải Thích Chỉ Số Thống Kê Dành Cho Hội Đồng",
                   [
                       ("• p-value < 0.05 (*):", "Xác nhận kết quả đạt ý nghĩa thống kê, loại trừ khả năng ăn may ngẫu nhiên."),
                       ("• Cohen's d > 0.8:", "Chênh lệch không chỉ có ý nghĩa trên giấy tờ mà còn tạo ra sự khác biệt thực tế rất rõ rệt."),
                       ("• 10 Hạt giống (Seeds):", "Chạy lặp lại 10 lần độc lập để đảm bảo kết quả hoàn toàn trung thực và ổn định.")
                   ], border_color=BORDER_CARD, title_size=12, body_size=10.5)

    add_clean_card(s15, Inches(7.56), Inches(2.2), Inches(5.0), Inches(4.4),
                   "Hồ Sơ Thực Nghiệm & Tái Lập 100%",
                   [
                       ("• Kiểm chứng đạo hàm lý thuyết:", "Đối chiếu giữa đạo hàm giải tích và công thức lượng tử vật lý (Parameter-Shift) đạt sai số cực nhỏ |Δ| < 4.1×10⁻⁸."),
                       ("• Bản Demo trực tiếp (Jupyter Notebook):", "File gd4_defense_demo.ipynb sẵn sàng bấm chạy: chọn ảnh siêu âm → trích xuất ảnh đặc trưng lượng tử → dự đoán xác suất ung thư chính xác (p = 0.884)."),
                       ("• Video demo dự phòng:", "Clip demo quay sẵn 104 giây (demo_defense_backup.mp4) có phụ đề tiếng Việt đầy đủ."),
                       ("• Cam kết liêm chính học thuật:", "100% mã nguồn, dữ liệu và nhật ký chạy đều công khai trên GitHub.")
                   ], border_color=NAVY, title_size=13, body_size=11)

    prs.save(str(OUT_PATH))
    print(f"Presentation regenerated successfully at: {OUT_PATH}")

if __name__ == "__main__":
    build_presentation()
