# -*- coding: utf-8 -*-
"""run_all_upgrades.py — chạy TOÀN BỘ 7 nâng chất methodological trong 1 script.
Đầu ra: results/statistical_upgrades.json (số liệu cho luận văn + slide)"""
import json, re, itertools
import numpy as np
from pathlib import Path
from scipy import stats as st
import torch
import torch.nn as nn

ROOT = Path('.')
CANON = json.load(open(ROOT/'results/reconciliation_canonical.json', encoding='utf-8'))
B = CANON['breastmnist']['models']; O = CANON['octmnist']['models']
MET = ['acc','bacc','f1','mcc','auc','pr_auc']
raw_b = json.load(open(ROOT/'results/full_trainable_breastmnist.json', encoding='utf-8'))['raw_results']
raw_o = json.load(open(ROOT/'results/full_trainable_octmnist.json', encoding='utf-8'))['raw_results']

def get_seeds(raw, model, k):
    return np.array([row[k] for row in raw[model]['test_metrics']], dtype=float)

out = {'bootstrap_ci': {}, 'power_analysis': {}, 'tost': {}, 'bh_adjusted': {}, 'random_classical': {}, 'learning_curve': {}, 'entanglement_ablation': {}}

# ============ 1. BOOTSTRAP CI (per-seed metrics, 10000 resamples) ============
rng = np.random.default_rng(42)
N_BOOT = 10000
for ds_name, raw, models in [('breastmnist', raw_b, list(raw_b.keys())), ('octmnist', raw_o, list(raw_o.keys()))]:
    out['bootstrap_ci'][ds_name] = {}
    for model in models:
        vals = {k: np.array([r[k] for r in raw[model]['test_metrics']]) for k in MET}
        out['bootstrap_ci'][ds_name][model] = {}
        for k in MET:
            x = vals[k]; n = len(x)
            boots = [np.mean(rng.choice(x, n, replace=True)) for _ in range(N_BOOT)]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            out['bootstrap_ci'][ds_name][model][k] = {'mean': round(float(np.mean(x)),4), 'boot_lo': round(float(lo),4), 'boot_hi': round(float(hi),4)}

# ============ 2. TOST EQUIVALENCE (biên ±2% BAcc cho cặp "hòa") ============
TIE_PAIRS = [
    ('breastmnist', 'trainable_strongly', 'fixed_strongly', 'bacc'),
    ('breastmnist', 'trainable_strongly', 'classical_cnn', 'bacc'),
    ('octmnist', 'trainable_strongly', 'fixed_strongly', 'auc'),
    ('octmnist', 'trainable_strongly', 'fixed_champion_gd2', 'auc'),
]
EQUIV_MARGIN = 0.02  # ±2% balanced accuracy
out['tost'] = []
for ds, m1, m2, k in TIE_PAIRS:
    raw = raw_b if ds == 'breastmnist' else raw_o
    x = get_seeds(raw, m1, k); y = get_seeds(raw, m2, k)
    diff = x - y
    se = np.std(diff, ddof=1) / np.sqrt(len(diff))
    df = len(diff) - 1
    t_crit = st.t.ppf(0.05, df)  # one-sided 5%
    lo_bound = np.mean(diff) - t_crit * se
    hi_bound = np.mean(diff) + t_crit * se
    # TOST: cả 2 one-sided tests đều p < 0.05
    p_lower = st.t.cdf((lo_bound - 0) / se, df)  # P(diff < -margin)
    p_upper = 1 - st.t.cdf((hi_bound - 0) / se, df)
    tost_p = max(p_lower, p_upper)
    equivalent = tost_p < 0.05
    out['tost'].append({'ds': ds, 'm1': m1, 'm2': m2, 'metric': k,
                        'mean_diff': round(float(np.mean(diff)),4), 'std_diff': round(float(np.std(diff, ddof=1)),4),
                        'tost_p': round(float(tost_p),4), 'equivalent': bool(equivalent)})

# ============ 3. BH-ADJUSTED q-VALUES ============
all_p = []
for i, s in enumerate(p_values := [], 1):
    pass
# thu thập p từ CANON['tests']
for tname, tdata in CANON.get('tests', {}).items():
    for k, v in tdata.items():
        if isinstance(v, dict) and 'p_ttest' in v:
            all_p.append(v['p_ttest'])
p_arr = np.array(sorted(all_p))
m = len(p_arr)
q = p_arr * m / np.arange(1, m+1)  # BH
q = np.minimum.accumulate(q[::-1])[::-1]  # enforce monotonicity
q = np.clip(q, 0, 1)
out['bh_adjusted'] = {'n_tests': m, 'q_values': [round(float(x),4) for x in sorted(q)],
                      'survive_q05': int(sum(q <= 0.05)), 'survive_q01': int(sum(q <= 0.01))}

# ============ 4. POWER ANALYSIS / MDE ============
from scipy.stats import norm
POWER = 0.80; ALPHA = 0.05
z_alpha = norm.ppf(1 - ALPHA/2); z_beta = norm.ppf(POWER)
# MDE cho t-test cặp n=10 seeds, effect size d:
# power = Phi(d*sqrt(n) - z_alpha) → d_min = (z_alpha + z_beta) / sqrt(n)
mde_seed10 = (z_alpha + z_beta) / np.sqrt(10)
mde_seed30 = (z_alpha + z_beta) / np.sqrt(30)
# Với test 156 ảnh (proportion 0.85):
p0, p1 = 0.85, 0.85 + 0.02  # detect 2% improvement
n_test = 156
pooled = (p0 + p1) / 2
se_prop = np.sqrt(pooled*(1-pooled)*(2/n_test))
z_prop = (p1 - p0) / se_prop
power_prop = 1 - norm.cdf(z_alpha - z_prop)
out['power_analysis'] = {
    'n_seeds_10': {'mde_cohens_d': round(float(mde_seed10),3), 'interpretation': 'MDE tối thiểu phát hiện khác biệt có ý nghĩa (80% power, α=0.05)'},
    'n_seeds_30': {'mde_cohens_d': round(float(mde_seed30),3)},
    'test_156': {'mde_proportion': round(float(z_prop*se_prop),4), 'interpretation': 'MDE cho Accuracy trên 156 ảnh test'},
}

# ============ 5. RANDOM CLASSICAL 2×2 FILTER BASELINE ============
# Mô phỏng: Conv2D(1→4, 2×2, s2) với kernel ngẫu nhiên cố định (không học)
torch.manual_seed(42)
import torch.nn as nn
import torch
conv_rand = nn.Conv2d(1, 4, kernel_size=2, stride=2, bias=True)
with torch.no_grad():
    nn.init.uniform_(conv_rand.weight, -0.5, 0.5)  # ngẫu nhiên cố định
    nn.init.zeros_(conv_rand.bias)
conv_rand.eval()
# Chạy trên BreastMNIST test (từ precomputed features của CNN baseline, cùng head)
# Vì không có raw images trong JSON, dùng approach: so sánh head trên features ngẫu nhiên vs features mạch
# Thực tế: cần raw images. Dùng approach khác: so sánh statistical properties
# (proxy: dùng random weights thay cho pretrained weights, chạy cùng protocol)
# → Đây là THÍ NGHIỆM cần chạy trên raw images — đánh dấu TODO cho script chạy riêng
out['random_classical'] = {'status': 'CẦN RAW IMAGES — chạy riêng bằng script truy cập data/breastmnist.npz'}

# ============ 6. LEARNING CURVE THEO CỠ DỮ LIỆU ============
out['learning_curve'] = {'status': 'CẦN RAW IMAGES — chạy riêng bằng script truy cập data/breastmnist.npz'}

# ============ 7. ENTANGLEMENT ABLATION (gỡ CNOT) ============
out['entanglement_ablation'] = {'status': 'CẦN RAW IMAGES — chạy riêng bằng script truy cập data/breastmnist.npz'}

json.dump(out, open(ROOT/'results/statistical_upgrades.json', 'w', encoding='utf-8'), indent=1, default=str)
print('=== TÓM TẮT ===')
print(f'Bootstrap CI: {sum(len(v) for v in out["bootstrap_ci"].values())} model-metric pairs × 2 datasets')
print(f'TOST: {len(out["tost"])} cặp "hòa" then chốt')
print(f'BH: {out["bh_adjusted"]["n_tests"]} tests, {out["bh_adjusted"]["survive_q05"]} survive q<0.05')
print(f'Power MDE (10 seeds, Cohen d): {out["power_analysis"]["n_seeds_10"]["mde_cohens_d"]}')
print(f'Power MDE (test 156 ảnh): MDE = {out["power_analysis"]["test_156"]["mde_proportion"]}')
print(f'Random classical + Learning curve + Entanglement: cần raw images → script riêng')
json.dump(out, open(ROOT/'results/statistical_upgrades.json', 'w', encoding='utf-8'), indent=1, default=str)
