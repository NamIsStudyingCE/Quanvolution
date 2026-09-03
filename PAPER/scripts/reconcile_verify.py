# -*- coding: utf-8 -*-
"""
reconcile_verify.py — STEP 1 of the zero-tolerance reconciliation.
Recomputes ALL manuscript statistics directly from raw per-seed JSON:
  mean, SAMPLE std (ddof=1), 95% CI (t*=2.262, n=10), paired t-test,
  Wilcoxon signed-rank, Cohen's d (paired, ddof=1).
Cross-checks the GD2 champion (circuit_ablation_champion_10seeds.json)
against the fixed_champion_gd2 entry of the final OCT run.
Saves canonical numbers to results/reconciliation_canonical.json.
"""
import json
import numpy as np
from scipy import stats

TSTAR = 2.262  # t* df=9, 95%
METRICS = ['acc', 'bacc', 'f1', 'mcc', 'auc', 'pr_auc']

def canon(fname):
    d = json.load(open(fname, encoding='utf-8'))
    out = {'meta': {'dataset': d['dataset'], 'num_epochs': d['num_epochs'],
                    'seeds': d['seeds'], 'progress': d['progress']},
           'models': {}}
    for model, rr in d['raw_results'].items():
        out['models'][model] = {}
        for k in METRICS:
            x = np.array([m[k] for m in rr['test_metrics']], dtype=float)
            n = len(x)
            mean = float(np.mean(x))
            std1 = float(np.std(x, ddof=1))
            hw = TSTAR * std1 / np.sqrt(n)
            out['models'][model][k] = {
                'mean': round(mean, 4), 'std': round(std1, 4),
                'ci_lo': round(mean - hw, 4), 'ci_hi': round(mean + hw, 4)}
    return out, d

def paired_tests(d, a, b):
    res = {}
    for k in METRICS:
        x = np.array([m[k] for m in d['raw_results'][a]['test_metrics']])
        y = np.array([m[k] for m in d['raw_results'][b]['test_metrics']])
        t, p_t = stats.ttest_rel(x, y)
        _, p_w = stats.wilcoxon(x, y)
        dz = (np.mean(x - y)) / np.std(x - y, ddof=1)
        res[k] = {'delta': round(float(np.mean(x - y)), 4),
                  'p_ttest': round(float(p_t), 4),
                  'p_wilcoxon': round(float(p_w), 4),
                  'cohens_d': round(float(dz), 3)}
    return res

breast_c, breast_d = canon('results/full_trainable_breastmnist.json')
oct_c, oct_d = canon('results/full_trainable_octmnist.json')

print('=' * 110)
print('CANONICAL TABLE (mean +/- sample std ddof=1, CI95 [lo, hi]) — source: raw per-seed JSON, 10 seeds x 20 epochs')
print('=' * 110)
for tag, c in [('BREASTMNIST', breast_c), ('OCTMNIST', oct_c)]:
    print(f'\n### {tag}  ({c["meta"]["progress"]})')
    hdr = f'{"model":22s}' + ''.join(f'{k:>26s}' for k in METRICS)
    print(hdr)
    for model, mm in c['models'].items():
        row = f'{model:22s}'
        for k in METRICS:
            v = mm[k]
            row += f'{v["mean"]:.4f}±{v["std"]:.4f} [{v["ci_lo"]:.4f},{v["ci_hi"]:.4f}] '.rjust(27)
        print(row)

print('\n### KEY PAIRED TESTS (canonical)')
tests = {
    'BREAST CNN vs FixedBasic': paired_tests(breast_d, 'classical_cnn', 'fixed_basic'),
    'BREAST CNN vs FixedStrongly': paired_tests(breast_d, 'classical_cnn', 'fixed_strongly'),
    'BREAST CNN vs TrainStrongly': paired_tests(breast_d, 'classical_cnn', 'trainable_strongly'),
    'BREAST FixedStrongly vs TrainStrongly': paired_tests(breast_d, 'fixed_strongly', 'trainable_strongly'),
    'OCT CNN vs TrainStrongly': paired_tests(oct_d, 'classical_cnn', 'trainable_strongly'),
    'OCT TrainStrongly vs FixedStrongly': paired_tests(oct_d, 'trainable_strongly', 'fixed_strongly'),
    'OCT TrainStrongly vs FixedChamp': paired_tests(oct_d, 'trainable_strongly', 'fixed_champion_gd2'),
}
for name, tt in tests.items():
    for k in ['auc', 'pr_auc', 'bacc']:
        v = tt[k]
        print(f'  {name:38s} [{k:7s}] Δ={v["delta"]:+.4f} p_t={v["p_ttest"]:.4f} p_w={v["p_wilcoxon"]:.4f} d={v["cohens_d"]:+.3f}')

# Variance-ratio claims
s = breast_c['models']
r_auc = s['classical_cnn']['auc']['std'] / s['fixed_basic']['auc']['std']
r_pr = s['classical_cnn']['pr_auc']['std'] / s['fixed_strongly']['pr_auc']['std']
print(f'\n### VARIANCE RATIOS (ddof=1): ROC-AUC CNN/FixedBasic = {r_auc:.4f}x ; PR-AUC CNN/FixedStrongly = {r_pr:.4f}x')

# Champion cross-check vs GD2 file
champ = json.load(open('results/circuit_ablation_champion_10seeds.json', encoding='utf-8'))
print('\n### CHAMPION CROSS-CHECK (GD2 file vs fixed_champion_gd2 in final OCT run)')
def walk(o, path=''):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from walk(v, f'{path}.{k}' if path else k)
    else:
        yield path, o
final_auc = oct_c['models']['fixed_champion_gd2']['auc']
hits = [(p, v) for p, v in walk(champ) if isinstance(v, (int, float)) and abs(v - 0.6912) < 0.01]
print(f'  final run fixed_champion_gd2 auc = {final_auc}')
print(f'  GD2 champion file values near 0.6912: {hits[:6]}')

json.dump({'breastmnist': breast_c, 'octmnist': oct_c,
           'tests': {k: {m: v for m, v in t.items()} for k, t in tests.items()},
           'variance_ratios': {'roc_auc_ddof1': round(r_auc, 4), 'pr_auc_ddof1': round(r_pr, 4)}},
          open('results/reconciliation_canonical.json', 'w', encoding='utf-8'), indent=1)
print('\nSaved canonical -> results/reconciliation_canonical.json')
