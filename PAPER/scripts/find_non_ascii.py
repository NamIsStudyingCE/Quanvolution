# -*- coding: utf-8 -*-
import os, sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
tex_path = os.path.join(root, "PAPER", "manuscript_ieee.tex")

with open(tex_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

found = []
for i, l in enumerate(lines):
    for ch in l:
        if ord(ch) > 127:
            found.append((i+1, ch, ord(ch), l.strip()))

if found:
    print(f"Found {len(found)} non-ASCII characters in manuscript_ieee.tex:")
    for line_no, ch, code, line_str in found:
        print(f"  Line {line_no}: character '{ch}' (U+{code:04X}) in: {line_str}")
else:
    print("No non-ASCII characters found! File is 100% clean ASCII / LaTeX compatible.")
