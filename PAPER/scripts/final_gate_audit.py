# -*- coding: utf-8 -*-
"""FINAL GATE AUDIT for the resubmission PDF (v4).
Every claim is checked against results/full_trainable_*.json (raw per-seed) —
no hardcoded trust: canonical stats are RECOMPUTED here, then compared with
every mean/std/CI pair extracted from the PDF text layer."""
import json, re, sys, io
import numpy as np
from scipy import stats as st
from pypdf import PdfReader

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
PDF = (r'D:\KhoaLuanTotNghiep\Symmetrical_Empirical_Evaluation_of_Trainable_versus_'
       r'Fixed_Quanvolutional_Filters_in_Medical_Image_Classification__A_Rigorous__'
       r'Reproducible_Benchmark_on_MedMNIST.pdf')
MET = ['acc','bacc','f1','mcc','auc','pr_auc']
T = 2.262

# ---- recompute canonical (ddof=1) and legacy (ddof=0) stats from RAW json ----
def load(fn):
    d = json.load(open(fn, encoding='utf-8'))
    out = {}
    for model, rr in d['raw_results'].items():
        out[model] = {}
        for k in MET:
            x = np.array([m[k] for m in rr['test_metrics']], float)
            s1, s0 = np.std(x, ddof=1), np.std(x, ddof=0)
            out[model][k] = {'mean': round(float(np.mean(x)),4),
                             'std1': round(float(s1),4), 'std0': round(float(s0),4),
                             'ci1': (round(float(np.mean(x)-T*s1/np.sqrt(10)),4), round(float(np.mean(x)+T*s1/np.sqrt(10)),4)),
                             'ci0': (round(float(np.mean(x)-T*s0/np.sqrt(10)),4), round(float(np.mean(x)+T*s0/np.sqrt(10)),4))}
    return out
B = load('results/full_trainable_breastmnist.json')
O = load('results/full_trainable_octmnist.json')
canon_cells = {(B[m][k]['mean'], B[m][k]['std1']) for m in B for k in MET} | \
              {(O[m][k]['mean'], O[m][k]['std1']) for m in O for k in MET}
legacy_cells = {(B[m][k]['mean'], B[m][k]['std0']) for m in B for k in MET} | \
               {(O[m][k]['mean'], O[m][k]['std0']) for m in O for k in MET}
canon_ci = {B[m][k]['ci1'] for m in B for k in MET if k in ('bacc','auc','pr_auc')} | \
           {O[m][k]['ci1'] for m in O for k in MET if k in ('bacc','auc','pr_auc')}
legacy_ci = {B[m][k]['ci0'] for m in B for k in MET if k in ('bacc','auc','pr_auc')} | \
            {O[m][k]['ci0'] for m in O for k in MET if k in ('bacc','auc','pr_auc')}
canon_ci.discard(legacy_ci & canon_ci)  # keep only distinguishable

r = PdfReader(PDF)
texts = [p.extract_text() or '' for p in r.pages]
whole = '\n'.join(texts)
flat = re.sub(r'\s+', '', whole)  # whitespace-free copy for robust contains

fails, warns = [], []

# ---- 1. mean±std pairs ----
pairs = re.findall(r'(\d\.\d{4})\s*±\s*(\d\.\d{4})', whole)
pairs = [(round(float(a),4), round(float(b),4)) for a,b in pairs]
from collections import Counter
pc = Counter(pairs)
missing = [c for c in canon_cells if pc[c] == 0]
legacy_found = [c for c in legacy_cells if pc[c] > 0 and c not in canon_cells]
print(f'[1] mean±std pairs extracted: {len(pairs)} | canonical cells present: {len(canon_cells)-len(missing)}/{len(canon_cells)}')
if missing: fails.append(f'missing canonical cells: {missing}')
if legacy_found: fails.append(f'LEGACY ddof=0 cells still present: {legacy_found}')

# ---- 2. CI pairs ----
cis = set((round(float(a),4), round(float(b),4)) for a,b in re.findall(r'\[(\d\.\d{4}),\s*(\d\.\d{4})\]', whole))
miss_ci = [c for c in canon_ci if c not in cis]
legacy_ci_found = [c for c in legacy_ci if c in cis and c not in canon_ci]
print(f'[2] CI pairs: {len(cis)} extracted | canonical present: {len(canon_ci)-len(miss_ci)}/{len(canon_ci)}')
if miss_ci: fails.append(f'missing CI: {miss_ci}')
if legacy_ci_found: fails.append(f'LEGACY CI present: {legacy_ci_found}')

# ---- 3. scalars: must-present ----
must = {'p/d & deltas': ['0.815','1.332','2.108','1.050','1.874','0.0298','0.0254','0.0023','0.0059','0.0098','0.8875','0.0232','0.046'],
        'latency': ['220.22','0.31','710'],
        'params': ['1,570','3,140','1,598','3,168','1,578','3,148','1,586','3,152','1,602','3,160'],
        'misc': ['2.262','4.1×10−8','2.7×','784','220.187','0.034'],
        'refs-fixed': ['Azevedo, V.','Matondo-Mvula','Entropy 26(8), 630','PRX Quantum 3(3), 030101','Is quantum advantage the right goal'],
        'table1': ['Matondo-Mvula and Elleithy (2024)','Breast (Mammography)','Schuld & Killoran (2022)'],
        'gradient': ['seed-averaged','peaking near 1.3','0.2–0.5'],
        'identity': ['Hoang-Nam Nguyen','Duy-Xuan-Bach Nguyen','bachndx@uit.edu.vn','NamIsStudyingCE/Quanvolution']}
for grp, items in must.items():
    miss = [x for x in items if x.replace(' ', '') not in flat and x not in whole]
    print(f'[3:{grp}] {"OK all" if not miss else "MISSING: "+str(miss)}')
    if miss: fails.append(f'{grp}: missing {miss}')

# ---- 4. must-ABSENT (stale/fake/artifacts) ----
absent = {'stale-claims': ['2.73','[0.05, 0.25]','2,73'],
          'fake-refs': ['Sannakki','extra steps','Quantum Information Processing','A. S. C.'],
          'old-params': ['1,590','3,172','3,156'],
          'stale-p': ['0.0309','0.0018'],
          'compile-artifacts': ['??','[?]','{,}','\\pm','\\textbf','TODO','FIXME'],
          'ai-slop': ['delve','tapestry','testament to','plethora','game changer','beacon','groundbreaking','revolutioniz']}
for grp, items in absent.items():
    found = [x for x in items if x in whole or x.replace(' ', '') in flat]
    print(f'[4:{grp}] {"clean" if not found else "FOUND(!): "+str(found)}')
    if found: fails.append(f'{grp}: found {found}')

# ---- 5. refs section: count + content ----
ref_start = next(i for i,t in enumerate(texts) if re.search(r'^References', t, re.M))
refpage = ref_start + 1
refs_txt = '\n'.join(texts[ref_start:])
nums = sorted(set(int(n) for n in re.findall(r'^(\d{1,2})\.\s+[A-Z]', refs_txt, re.M)))
print(f'[5] References heading on page {refpage}; entries numbered: {nums}')
if nums != list(range(1,20)): fails.append(f'ref numbering: {nums}')

# ---- 6. body within 12 pages + no page numbers ----
fw = next((i+1 for i,t in enumerate(texts) if 'Future Work' in t), None)
print(f'[6] Future Work (body end) page: {fw}; References page: {refpage}')
if fw and fw > 12: fails.append(f'body extends to page {fw} > 12')
if refpage > 12: fails.append(f'References start on page {refpage} > 12')
for i,t in enumerate(texts):
    lines = [l.strip() for l in t.splitlines() if l.strip()]
    solo = [l for l in lines if re.fullmatch(r'\d{1,2}', l)]
    if solo: warns.append(f'page {i+1}: standalone digits {solo} (check not page numbers)')

# ---- 7. figure/table mentions ----
for x in ['Fig. 1','Fig. 2','Fig. 3','Fig. 4','Fig. 5','Fig. 6','Table 1','Table 2','Table 3','Table 4','Table 5']:
    if x.replace(' ','') not in flat: fails.append(f'{x} never referenced')
print('[7] figure/table mentions: OK' if not any('never referenced' in f for f in fails) else '[7] see fails')

# ---- 8. Prism rephrase canaries (meaning preserved) ----
canaries = [('Fig2 ref', ['juxtaposes these quantum feature maps','Fig. 2 compares the resulting quantum feature maps']),
            ('trainability claim', ['beneficial only within the same ansatz family','only within the same ansatz family']),
            ('ceteris paribus', ['Ceteris Paribus']),
            ('honest framing', ['data-regime dependent','data regime dependent'])]
for name, variants in canaries:
    ok = any(v.replace(' ','') in flat for v in variants)
    print(f'[8:{name}] {"OK" if ok else "MISSING all variants"}')
    if not ok: fails.append(f'canary {name} missing')

print('\n' + '='*70)
print(f'FAILS: {len(fails)}')
for f_ in fails: print('  ✗', f_)
print(f'WARNINGS (manual eyeball): {len(warns)}')
for w in warns: print('  ⚠', w)
print('VERDICT:', 'CLEAN — ready for resubmission' if not fails else 'ISSUES FOUND — do NOT submit yet')
