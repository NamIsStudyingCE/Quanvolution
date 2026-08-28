# -*- coding: utf-8 -*-
"""
compute_effect_size_ci.py
-------------------------
Tính toán Effect Size (Cohen's d) và Khoảng tin cậy (95% Confidence Interval)
cho toàn bộ các mô hình và cặp đấu trong Giai đoạn 3 (10 seeds độc lập).

Xuất bảng Markdown chi tiết phục vụ viết phần Results & Discussion cho Bài báo.
"""

import sys, os, json
import numpy as np
import scipy.stats as stats

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def compute_ci(data, confidence=0.95):
    """Tính Mean và Khoảng tin cậy CI 95% bằng phân phối Student-t (ddof=1)."""
    a = np.array(data, dtype=float)
    n = len(a)
    m = np.mean(a)
    se = stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, m - h, m + h, h

def paired_cohens_d(x, y):
    """Tính Cohen's d cho mẫu bắt cặp (Paired samples).
    d = mean(diff) / std(diff, ddof=1)
    Quy ước Cohen: |d| < 0.2: negligible; 0.2-0.5: small; 0.5-0.8: medium; > 0.8: large.
    """
    diff = np.array(x, dtype=float) - np.array(y, dtype=float)
    sd = np.std(diff, ddof=1)
    if sd < 1e-12:
        return 0.0
    return np.mean(diff) / sd

def interpret_d(d):
    ad = abs(d)
    if ad < 0.2:
        tag = "Tác động không đáng kể (Negligible)"
    elif ad < 0.5:
        tag = "Tác động nhỏ (Small effect)"
    elif ad < 0.8:
        tag = "Tác động trung bình (Medium effect)"
    else:
        tag = "Tác động lớn (Large effect)"
    return tag

def process_dataset(json_path, dataset_name):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    raw = data['raw_results']
    metrics = ['auc', 'pr_auc', 'bacc', 'acc', 'f1', 'mcc']
    metric_names = ['ROC-AUC', 'PR-AUC', 'Balanced Acc', 'Accuracy', 'F1-Score', 'MCC']
    
    print(f"\n{'='*95}")
    print(f" DATASET: {dataset_name.upper()} (10 SEEDS ĐỘC LẬP, 20 EPOCHS)")
    print(f"{'='*95}")
    
    # 1. Bảng Mean ± Std kèm 95% CI
    print(f"\n### 1. BẢNG HIỆU NĂNG CHI TIẾT (MEAN ± STD & KHOẢNG TIN CẬY 95% CI)\n")
    header = f"| {'Mô hình':<28} | {'ROC-AUC [95% CI]':<26} | {'PR-AUC [95% CI]':<26} | {'Balanced Acc [95% CI]':<26} |"
    print(header)
    print(f"|{'-'*30}|{'-'*28}|{'-'*28}|{'-'*28}|")
    
    summary_dict = {}
    for model_key, model_data in raw.items():
        test_m = model_data['test_metrics']
        summary_dict[model_key] = {}
        row_str = f"| {model_key:<28} |"
        for m in ['auc', 'pr_auc', 'bacc']:
            vals = [s[m] for s in test_m]
            mean_val, ci_low, ci_high, h = compute_ci(vals)
            std_val = np.std(vals, ddof=1)
            summary_dict[model_key][m] = {
                'vals': vals, 'mean': mean_val, 'std': std_val,
                'ci_low': ci_low, 'ci_high': ci_high, 'margin': h
            }
            cell = f"{mean_val:.4f} ± {std_val:.4f} [{ci_low:.4f}, {ci_high:.4f}]"
            row_str += f" {cell:<26} |"
        print(row_str)
    
    # 2. Bảng Effect Size (Cohen's d) cho các cặp đối sánh chính
    print(f"\n### 2. BẢNG EFFECT SIZE (COHEN'S d) & KIỂM ĐỊNH CHO CÁC CẶP ĐẤU CHÍNH\n")
    print(f"| {'Cặp đấu':<45} | {'Metric':<14} | {'Delta':<10} | {'Cohen d':<10} | {'p (t-test)':<12} | {'p (Wilcoxon)':<12} | {'Đánh giá Effect Size':<32} |")
    print(f"|{'-'*47}|{'-'*16}|{'-'*12}|{'-'*12}|{'-'*14}|{'-'*14}|{'-'*34}|")
    
    pairs_to_test = []
    if dataset_name.lower() == 'breastmnist':
        pairs_to_test = [
            ("fixed_basic", "classical_cnn", "Fixed Basic vs Classical CNN (Quán quân ROC)"),
            ("trainable_basic", "classical_cnn", "Trainable Basic vs Classical CNN (PR-AUC)"),
            ("trainable_strongly", "fixed_strongly", "Trainable Strongly vs Fixed Strongly (Tier 3)"),
            ("trainable_strongly", "classical_cnn", "Trainable Strongly vs Classical CNN (Balanced Acc)"),
            ("fixed_strongly", "classical_cnn", "Fixed Strongly vs Classical CNN (PR-AUC)")
        ]
    else:  # octmnist
        pairs_to_test = [
            ("trainable_strongly", "fixed_strongly", "Trainable Strongly vs Fixed Strongly (Tier 3)"),
            ("trainable_strongly", "fixed_champion_gd2", "Trainable Strongly vs Fixed Champ (random_L1)"),
            ("classical_cnn", "trainable_strongly", "Classical CNN vs Trainable Strongly (Classical Win)"),
            ("classical_cnn", "fixed_champion_gd2", "Classical CNN vs Fixed Champion (Classical Win)")
        ]
    
    stat_tests = data.get('statistical_tests', {})
    
    for m1, m2, label in pairs_to_test:
        if m1 in raw and m2 in raw:
            pair_key = f"{m1}_vs_{m2}"
            pair_stat = stat_tests.get(pair_key, stat_tests.get(f"{m2}_vs_{m1}", {}))
            
            for m, m_name in [('auc', 'ROC-AUC'), ('pr_auc', 'PR-AUC'), ('bacc', 'Balanced Acc')]:
                v1 = [s[m] for s in raw[m1]['test_metrics']]
                v2 = [s[m] for s in raw[m2]['test_metrics']]
                d = paired_cohens_d(v1, v2)
                delta = np.mean(v1) - np.mean(v2)
                
                m_stat = pair_stat.get(m, {})
                pt = m_stat.get('p_value_ttest', m_stat.get('p_ttest', float('nan')))
                pw = m_stat.get('wilcoxon_p_value', m_stat.get('p_wilcoxon', float('nan')))
                
                tag = interpret_d(d)
                sig_t = '***' if pt < 0.001 else '**' if pt < 0.01 else '*' if pt < 0.05 else 'ns'
                
                print(f"| {label:<45} | {m_name:<14} | {delta:+.4f}    | {d:+.3f}     | p={pt:.4f} ({sig_t}) | p={pw:.4f}    | {tag:<32} |")
            print(f"|{'-'*47}|{'-'*16}|{'-'*12}|{'-'*12}|{'-'*14}|{'-'*14}|{'-'*34}|")

    return summary_dict

def main():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    breast_json = os.path.join(root_dir, "results", "full_trainable_breastmnist.json")
    oct_json = os.path.join(root_dir, "results", "full_trainable_octmnist.json")
    
    print("="*95)
    print(" TÍNH TOÁN STATISTICAL EFFECT SIZE (COHEN'S d) & 95% CONFIDENCE INTERVALS")
    print("="*95)
    
    if os.path.exists(breast_json):
        process_dataset(breast_json, "BreastMNIST")
    else:
        print(f"[CẢNH BÁO] Không tìm thấy {breast_json}")
        
    if os.path.exists(oct_json):
        process_dataset(oct_json, "OCTMNIST")
    else:
        print(f"[CẢNH BÁO] Không tìm thấy {oct_json}")

if __name__ == '__main__':
    main()
