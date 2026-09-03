# -*- coding: utf-8 -*-
"""normalize_gd3.py — chuẩn hóa toàn bộ số liệu GD3 về canonical ddof=1.
Mọi cặp 'mean ± std' được tra theo mean trong reconciliation_canonical.json.
Thêm NOTE header đánh dấu báo cáo lịch sử."""
import json, re

CANON = json.load(open('results/reconciliation_canonical.json', encoding='utf-8'))
MET = ['acc', 'bacc', 'f1', 'mcc', 'auc', 'pr_auc']

# mean -> canonical std (ddof=1); phát hiện collision
mean2std = {}
collisions = []
for ds in ('breastmnist', 'octmnist'):
    for model, mm in CANON[ds]['models'].items():
        for k in MET:
            v = mm[k]
            key = round(v['mean'], 4)
            if key in mean2std and abs(mean2std[key] - v['std']) > 1e-9:
                collisions.append((key, mean2std[key], v['std'], ds, model, k))
            mean2std.setdefault(key, v['std'])
print('collisions:', collisions if collisions else 'NONE')

# p-value fixes (cũ/sai -> canonical)
P_FIX = {'0.0309': '0.0298', '0.0018': '0.0023', '0.0323': '0.0090'}

NOTE = ('> [!NOTE] **Báo cáo lịch sử GĐ3.** Toàn bộ số liệu chính thức cuối cùng (sample std ddof=1, '
        'p-value chuẩn) được chuẩn hóa tại `results/reconciliation_canonical.json` từ raw per-seed JSON, '
        'và được phản ánh đồng bộ trong Luận văn (GĐ4) và bài báo SOICT 2026. File này đã được chuẩn hóa '
        'theo canonical.\n\n')

for path in ['GD3/BAO_CAO_GIAI_DOAN_3.md', 'GD3/README.md']:
    t = open(path, encoding='utf-8').read()
    n_std = n_p = 0
    def sub_std(m):
        global n_std
        mean = round(float(m.group(1)), 4)
        if mean in mean2std:
            if abs(float(m.group(2)) - mean2std[mean]) > 5e-5:
                n_std += 1
            return f"{m.group(1)} ± {mean2std[mean]:.4f}"
        return m.group(0)
    t = re.sub(r'(\d\.\d{4})\s*±\s*(\d\.\d{4})', sub_std, t)
    for old, new in P_FIX.items():
        cnt = t.count(old)
        if cnt:
            t = t.replace(old, new)
            n_p += cnt
    if not t.startswith('> [!NOTE]'):
        t = NOTE + t
    open(path, 'w', encoding='utf-8', newline='\n').write(t)
    print(f'{path}: std fixed={n_std}, p fixed={n_p}')

# verify: quét lại
for path in ['GD3/BAO_CAO_GIAI_DOAN_3.md', 'GD3/README.md']:
    t = open(path, encoding='utf-8').read()
    bad = []
    for m in re.finditer(r'(\d\.\d{4})\s*±\s*(\d\.\d{4})', t):
        mean, std = round(float(m.group(1)), 4), round(float(m.group(2)), 4)
        if mean in mean2std and abs(std - mean2std[mean]) > 5e-5:
            bad.append((mean, std))
    stale = [s for s in ['0.0309', '2.73', '[0.05, 0.25]', 'Sannakki'] if s in t]
    print(f'{path}: non-canonical pairs={bad[:5]} stale={stale}')
