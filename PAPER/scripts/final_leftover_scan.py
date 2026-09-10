# -*- coding: utf-8 -*-
"""final_leftover_scan: hunt any remaining stale ddof=0 stds / fake refs / old claims
across ALL manuscripts. Zero tolerance: every hit must be reviewed."""
import re, glob

STALE_SUBSTRINGS = [
    '2.73', '2{,}73', '[0.05, 0.25]', '[0{,}05; 0{,}25]', '0.05 - 0.25',
    'Sannakki', 'sannakki', 'extra steps', 'Quantum Information Processing',
    '0.4792', '0{,}4792', '0.5298', '0{,}5298', '0.8115', '0{,}8115',
    '0.8192', '0{,}8192', '0.4492', '0{,}4492', '0.8307', '0.8497',
    'A. S. C. et al', 'p.~15, Feb', 'p. 15, Feb', '15 (2024)',
]
# ddof=0 std cells that must no longer appear in tables/prose of tex/md
STALE_TEX_CELLS = [
    r'0\.8521 \\pm 0\.0090\b', r'0\.8336 \\pm 0\.0246\b', r'0\.9182 \\pm 0\.0067\b',
    r'0\.7505 \\pm 0\.0227\b', r'0\.6922 \\pm 0\.0189\b', r'0\.6690 \\pm 0\.0052\b',
    r'0\.6912 \\pm 0\.0067\b', r'0\.6945 \\pm 0\.0428\b', r'0\.4991 \\pm 0\.0282\b',
]

FILES = [
    'PAPER/manuscript_springer_ccis.tex', 'PAPER/manuscript_ieee.tex',
    r'D:\KLTN_Paper\main.tex',
    'PAPER/MANUSCRIPT_FINAL_EN.md', 'PAPER/MANUSCRIPT_DRAFT_VI.md',
    'GD3/README.md', 'GD3/BAO_CAO_GIAI_DOAN_3.md', 'README.md',
]
total = 0
for f in FILES:
    try:
        t = open(f, encoding='utf-8').read()
    except FileNotFoundError:
        print(f'{f}: MISSING'); continue
    hits = []
    for s in STALE_SUBSTRINGS:
        if s in t:
            hits.append(('substr', s, t.count(s)))
    for pat in STALE_TEX_CELLS:
        for m in re.finditer(pat, t):
            hits.append(('regex', pat, 1))
    if hits:
        print(f'{f}:')
        for kind, s, n in hits:
            print(f'   [{kind}] {s!r} x{n}')
        total += len(hits)
    else:
        print(f'{f}: CLEAN')
print(f'\nTOTAL leftover issues: {total}')
