# BÁO CÁO GIAI ĐOẠN 4 — VIẾT LUẬN VĂN, DEMO & CHUẨN BỊ BẢO VỆ

**Đề tài:** Quanvolution — Lớp tích chập lượng tử cho ảnh y tế (MedMNIST)
**Sinh viên thực hiện:** Nguyễn Hoàng Nam — MSSV 22520916
**GVHD:** ThS. Nguyễn Duy Xuân Bách — Khoa Kỹ thuật Máy tính, UIT
**Trạng thái:** 🟢 Hoàn thành nội dung chính · ⏳ Chờ: kết quả SOICT (12/10) · duyệt slide trực quan · kiểm tra đạo văn (chờ khoa)

---

## 1. Những việc ĐÃ HOÀN THÀNH

### 1.1. Bài báo khoa học quốc tế (điểm cộng vượt kế hoạch)
- Bản thảo hoàn thiện, đã nộp SOICT 2026 (Springer CCIS) trên EasyChair — **13 trang, SHA-256 `c8d5d093…`** (đóng băng, xem tag `soict-submission-v4`).
- Abstract + Keywords trên EasyChair đồng bộ ddof=1. Trạng thái: under review.
- Mọi số liệu paper đối chiếu 100% với `results/reconciliation_canonical.json` (117/117 cặp ±, 33/33 CI).

### 1.2. Luận văn (`GD4/KLTN_draft_full.docx`)
- **~14,600 từ ≈ 65 trang** (vượt chuẩn tối thiểu 50 trang — Phụ lục 2 UIT).
- Đủ 6 chương IMRaD + công thức 6 metrics + bảng canonical + 9 hình 300 DPI + 19 tài liệu tham khảo (đã verify từng bài) + Phụ lục A/B.2/C/D.
- 2 sections: front matter không số trang; số trang Ả-rập từ Tóm tắt.
- ThS. đồng bộ trên bìa phụ; tên sinh viên + MSSV đã điền.

### 1.3. Slide bảo vệ (`slides/KLTN_slides_faculty_inplace.pptx`)
- **Dựng in-place trên template Khoa KTMT (Phụ lục 6)** — giữ nguyên 100% thiết kế (theme CE-UIT, section header 01–05, footer Copyrights, badge logo).
- Nội dung cô đọng theo chuẩn academic-pptx: ≤ 3 thẻ/slide, mỗi thẻ ≤ 35 từ, stat callout nổi bật.
- Số liệu slide = canonical (đối chiếu máy 100%).
- Kèm 4 slide Phụ lục phòng vệ: kiểm định thống kê, phân rã tham số, Parameter-Shift, hồ sơ demo.

### 1.4. Demo bảo vệ (tiêu chí #9)
- `notebooks/gd4_defense_demo.ipynb` — ảnh → 3 mạch live → head train 1.3s → dự đoán malignant p=0.884 → latency so sánh.
- `notebooks/demo_defense_backup.mp4` — video dự phòng 104 giây, phụ đề tiếng Việt.

### 1.5. Repo & kiểm định
- Mọi thay đổi số liệu đi qua `reconcile_verify.py` → `reconciliation_canonical.json` (ddof=1).
- Stale scan CLEAN toàn bộ tài liệu; `final_ground_truth.json` cũ đã cách ly vào `results/archive/`.
- 3 nguồn (paper ↔ luận văn ↔ slide) đồng bộ tuyệt đối — đã đối chiếu máy.

---

## 2. Những việc CHƯA HOÀN THÀNH

| # | Việc | Chờ vào | Ghi chú |
|---|---|---|---|
| 1 | Kết quả SOICT | 12/10/2026 | Nhánh A: camera-ready 23/10 · Nhánh B: nộp venue khác sau bảo vệ |
| 2 | Điền MSSV vào đề cương `.doc` + in + chữ ký CBHD | ⏳ | Nội dung đã điền sẵn trong `GD4/de_cuong_chi_tiet_filled.docx` |
| 3 | Kiểm tra đạo văn | Chờ khoa công bố tool/ngưỡng | Đã viết chuẩn paraphrase + trích dẫn ngay từ đầu |
| 4 | TOC tự động trong luận văn (Word → Update Field) | Sau khi chốt nội dung | |
| 5 | In thử + đóng quyển | Trước 09/11 | |
| 6 | Slide: duyệt trực quan lần cuối trên máy chiếu thật | Trước bảo vệ | 4 slide trống chrome cuối deck để bạn tự chèn bảng tay |

---

## 3. Quy ước bất biến (nhắc lại cho ai tiếp quản dự án)

1. Nguồn số duy nhất: `results/full_trainable_*.json` → `results/reconciliation_canonical.json` (ddof=1). Cấm gõ tay số liệu.
2. `results/archive/final_ground_truth_SUPERSEDED.json` là file cũ — KHÔNG dùng.
3. 19 references đã verify — cấm thêm/bớt nếu chưa qua đối chiếu web.
4. Paper trên EasyChair = hash `c8d5d093…` — chỉ thay khi có quyết định từ hội nghị.
5. Slide bảo vệ: giữ template Khoa, KHÔNG vẽ textbox tự do ngoài placeholder.
