"""Static verification of D:\\KLTN_Paper\\main.tex against guardrails."""
import re
import os

os.chdir(r'D:\KLTN_Paper')
tex = open('main.tex', encoding='utf-8').read()

print('=== 1. GUARDRAIL NUMBERS ===')
checks = [
    (r'0\.8521 \\pm 0\.0090', 'Breast Fixed Basic ROC-AUC'),
    (r'0\.8336 \\pm 0\.0246', 'Breast CNN ROC-AUC'),
    (r'0\.9182 \\pm 0\.0067', 'Breast Fixed Strongly PR-AUC'),
    (r'0\.9041 \\pm 0\.0095', 'Breast CNN PR-AUC'),
    (r'0\.6945 \\pm 0\.0428', 'Breast Trainable Strongly BAcc'),
    (r'0\.7505 \\pm 0\.0227', 'OCT CNN ROC-AUC'),
    (r'0\.4991 \\pm 0\.0282', 'OCT CNN PR-AUC'),
    (r'0\.6922 \\pm 0\.0189', 'OCT Trainable Strongly ROC'),
    (r'0\.6690 \\pm 0\.0052', 'OCT Fixed Strongly ROC'),
    (r'0\.6912 \\pm 0\.0067', 'OCT Fixed Champion ROC'),
    (r'0\.0232', 'Delta OCT trainable vs fixed'),
    (r'\+0\.815', 'Cohen d 0.815'),
    (r'\+1\.332', 'Cohen d 1.332'),
    (r'\+2\.108', 'Cohen d 2.108'),
    (r'\+1\.050', 'Cohen d 1.050'),
    (r'\+1\.874', 'Cohen d 1.874'),
    (r'220\.22', 'Latency quanv 220.22'),
    (r'0\.31\\text\{ ms\}', 'Latency CNN 0.31 ms'),
    (r'1\{,\}570', 'Head K=2 1570'),
    (r'1\{,\}598', 'Total classical breast 1598'),
    (r'1\{,\}578', 'Total fixed 1578'),
    (r'3\{,\}140', 'Head K=4 3140'),
    (r'\\sim 2\.7\\times', 'Std ratio 2.7x'),
    (r'0\.0298', 'p ttest 0.0298'),
    (r'0\.0254', 'p wilcoxon 0.0254'),
    (r'0\.0023', 'p ttest 0.0023'),
    (r'0\.0098', 'p wilcoxon 0.0098'),
    (r'0\.8875', 'p tie 0.8875'),
    (r'4\.1 \\times 10\^\{-8\}', 'Parameter-shift 4.1e-8'),
    (r'1\{,\}602|1\{,\}586|1\{,\}578|1\{,\}598', 'Table II totals present'),
]
fails = 0
for pat, name in checks:
    if not re.search(pat, tex):
        print(f'  [FAIL] {name}: pattern {pat}')
        fails += 1
print(f'  => {len(checks) - fails}/{len(checks)} number checks passed')

print('=== 2. WRONG CLAIMS REMOVED ===')
for bad, name in [
    (r'\[0\.05, 0\.25\]', 'gradient interval [0.05, 0.25]'),
    (r'2\.73', '2.73x'),
    (r'prove that quantum advantages', '"we prove" overclaim'),
]:
    found = re.search(bad, tex)
    print(f'  [{"STILL PRESENT - FAIL" if found else "removed - OK"}] {name}')

print('=== 3. STRUCTURE ===')
print('  pagestyle empty:', r'\pagestyle{empty}' in tex)
print('  thispagestyle empty:', r'\thispagestyle{empty}' in tex)
bibitems = re.findall(r'\\bibitem\{([^}]+)\}', tex)
print(f'  bibitems: {len(bibitems)} (expect 19)')
cites = set()
for grp in re.findall(r'\\cite\{([^}]+)\}', tex):
    cites.update(k.strip() for k in grp.split(','))
undefined = cites - set(bibitems)
uncited = set(bibitems) - cites
print(f'  cited keys: {len(cites)}; undefined: {sorted(undefined) if undefined else "NONE"}; uncited: {sorted(uncited) if uncited else "NONE"}')

print('=== 4. FIGURES ===')
figs = re.findall(r'\\includegraphics\[[^\]]*\]\{([^}]+)\}', tex)
for f in figs:
    print(f'  {f}: {"OK" if os.path.exists(f) else "MISSING!"}')

print('=== 5. ENVIRONMENTS / BRACES ===')
ok = True
for env in ['abstract', 'itemize', 'enumerate', 'table', 'figure', 'equation',
            'tabularx', 'tabular', 'thebibliography', 'document', 'keywords']:
    b = len(re.findall(r'\\begin\{' + env + r'\}', tex))
    e = len(re.findall(r'\\end\{' + env + r'\}', tex))
    if env == 'keywords':
        b = len(re.findall(r'\\keywords\{', tex)); e = b  # keyword is a macro, not env
    if b != e:
        ok = False
        print(f'  MISMATCH {env}: begin={b} end={e}')
print('  environments balanced:', ok)
print('  braces balanced:', tex.count('{') == tex.count('}'),
      f"(open={tex.count('{')} close={tex.count('}')})")
print('  author block (single-blind):', 'Hoang-Nam Nguyen' in tex and 'bachndx@uit.edu.vn' in tex)
print('  AI-slop scan:', re.findall(r'\b(delve|tapestry|testament|beacon|plethora|groundbreaking|revolutioniz\w*)', tex, re.I) or 'clean')
