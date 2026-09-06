# -*- coding: utf-8 -*-
"""
apply_pass2.py
Run this script to copy KLTN_draft_full_pass2.docx over KLTN_draft_full.docx
after closing WPS Office / Microsoft Word.
"""
import os
import shutil
import sys

src = r'D:\KhoaLuanTotNghiep\GD4\KLTN_draft_full_pass2.docx'
dst = r'D:\KhoaLuanTotNghiep\GD4\KLTN_draft_full.docx'

if not os.path.exists(src):
    print("Error: Source file does not exist:", src)
    sys.exit(1)

try:
    shutil.copyfile(src, dst)
    print("SUCCESS: KLTN_draft_full.docx has been updated with Pass 2 content!")
except Exception as e:
    print("LOCKED: Word/WPS is still holding the file lock. Error:", e)
    print("Please close WPS Office / Microsoft Word and run this script again.")
