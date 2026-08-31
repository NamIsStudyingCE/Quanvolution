"""FORENSIC AUDIT: independently recompute all statistics from raw_results
and compare with paper claims. No reliance on hardcoded verify script values."""
import json
import numpy as np
from scipy import stats

def load(name):
    with open(f'results/{name}') as f:
        return json.load(f)

def summarize(tm_list):
    arr = {k: np.array([m[k] for m in tm_list], dtype=float) for k in
           ['acc', 'bacc', 'f1', 'mcc', 'auc', 'pr_auc']}
    return arr

def ci95(x):
    n = len(x)
    sem = stats.sem(x)
    return stats.t.interval(0.95, n - 1, loc=np.mean(x), scale=sem)

def paired(x, y, label=''):
    t, p_t = stats.ttest_rel(x, y)
    try:
        w, p_w = stats.wilcoxon(x, y)
    except Exception as e:
        w, p_w = np.nan, np.nan
    d = (np.mean(x) - np.mean(y)) / np.std(x - y, ddof=1)
    return {'delta': np.mean(x) - np.mean(y), 't_p': p_t, 'w_p': p_w, 'cohen_d_paired': d}

def fmt(m, s):
    return f"{m:.4f} +/- {s:.4f}"

print("=" * 100)
print("INDEPENDENT FORENSIC RECOMPUTATION FROM RAW_RESULTS (per-seed)")
print("=" * 100)

for fname, ds in [('full_trainable_breastmnist.json', 'BREASTMNIST'),
                  ('full_trainable_octmnist.json', 'OCTMNIST')]:
    d = load(fname)
    print(f"\n### {ds} (epochs={d['num_epochs']}, seeds={len(d['seeds'])}, progress={d['progress']})")
    S = {}
    for model, rr in d['raw_results'].items():
        arr = summarize(rr['test_metrics'])
        S[model] = arr
        n = len(arr['auc'])
        # std with ddof=1 (sample) and ddof=0 (population) to detect convention
        line = [f"  {model:22s} n={n}"]
        for k in ['acc', 'bacc', 'f1', 'mcc', 'auc', 'pr_auc']:
            m = np.mean(arr[k]); s1 = np.std(arr[k], ddof=1); s0 = np.std(arr[k], ddof=0)
            lo, hi = ci95(arr[k])
            line.append(f"    {k:7s} mean={m:.4f} std(ddof=1)={s1:.4f} std(ddof=0)={s0:.4f} CI95=[{lo:.4f},{hi:.4f}]")
        print("\n".join(line))

    print(f"\n  --- Paired tests ({ds}) ---")
    pairs = []
    if ds == 'BREASTMNIST':
        pairs = [('classical_cnn', 'fixed_basic'), ('classical_cnn', 'fixed_strongly'),
                 ('classical_cnn', 'trainable_strongly'), ('fixed_strongly', 'trainable_strongly'),
                 ('fixed_basic', 'trainable_basic')]
    else:
        pairs = [('classical_cnn', 'trainable_strongly'), ('classical_cnn', 'fixed_strongly'),
                 ('classical_cnn', 'fixed_champion_gd2'), ('trainable_strongly', 'fixed_strongly'),
                 ('trainable_strongly', 'fixed_champion_gd2')]
    for a, b in pairs:
        for k in ['auc', 'pr_auc', 'bacc']:
            r = paired(S[a][k], S[b][k])
            print(f"    {a} vs {b} [{k:7s}] delta={r['delta']:+.4f} t_p={r['t_p']:.4g} wilcoxon_p={r['w_p']:.4g} d={r['cohen_d_paired']:+.3f}")

    # key claims
    print(f"\n  --- KEY CLAIM CHECKS ({ds}) ---")
    if ds == 'BREASTMNIST':
        s_fb_auc = np.std(S['fixed_basic']['auc'], ddof=1)
        s_cnn_auc = np.std(S['classical_cnn']['auc'], ddof=1)
        print(f"    Std reduction ratio (CNN/fixed_basic AUC): {s_cnn_auc:.4f}/{s_fb_auc:.4f} = {s_cnn_auc/s_fb_auc:.4f}")
        s_cnn_auc0 = np.std(S['classical_cnn']['auc'], ddof=0)
        s_fb_auc0 = np.std(S['fixed_basic']['auc'], ddof=0)
        print(f"    (ddof=0): {s_cnn_auc0:.4f}/{s_fb_auc0:.4f} = {s_cnn_auc0/s_fb_auc0:.4f}")
        r = paired(S['classical_cnn']['auc'], S['fixed_basic']['auc'])
        print(f"    CNN vs FixedBasic AUC: delta={r['delta']:+.4f}, t_p={r['t_p']:.4f}, w_p={r['w_p']:.4f}")
        r = paired(S['classical_cnn']['pr_auc'], S['fixed_strongly']['pr_auc'])
        print(f"    CNN vs FixedStrongly PR: delta={r['delta']:+.4f}, t_p={r['t_p']:.4f}, w_p={r['w_p']:.4f}")
