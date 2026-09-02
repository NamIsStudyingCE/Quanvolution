import re
import sys
import io

# Force UTF-8 stdout
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def verify():
    with open(r'd:\KhoaLuanTotNghiep\PAPER\manuscript_ieee.tex', 'r', encoding='utf-8') as f:
        tex = f.read()

    print("=" * 80)
    print("1. KIEM DINH TINH CHINH XAC CUA SO LIEU (VS CANONICAL ddof=1, 10 SEEDS x 20 EPOCHS)")
    print("=" * 80)

    checks = [
        ("Breast Fixed Basic L2 ROC-AUC", r"0\.8521 \\pm 0\.0095", "0.8521±0.0095"),
        ("Breast Fixed Strongly L2 PR-AUC", r"0\.9182 \\pm 0\.0071", "0.9182±0.0071"),
        ("Breast Classical CNN ROC-AUC", r"0\.8336 \\pm 0\.0259", "0.8336±0.0259"),
        ("Breast Classical CNN PR-AUC", r"0\.9041 \\pm 0\.0100", "0.9041±0.0100"),
        ("Breast Trainable Strongly L2 BAcc", r"0\.6945 \\pm 0\.0451", "0.6945±0.0451"),
        ("OCT Classical CNN ROC-AUC", r"0\.7505 \\pm 0\.0240", "0.7505±0.0240"),
        ("OCT Classical CNN PR-AUC", r"0\.4991 \\pm 0\.0297", "0.4991±0.0297"),
        ("OCT Trainable Strongly L1 ROC-AUC", r"0\.6922 \\pm 0\.0199", "0.6922±0.0199"),
        ("OCT Fixed Strongly L1 ROC-AUC", r"0\.6690 \\pm 0\.0055", "0.6690±0.0055"),
        ("OCT Fixed Champion (random_L1) ROC-AUC", r"0\.6912 \\pm 0\.0071", "0.6912±0.0071"),
        ("Delta Trainable vs Fixed Strongly (OCT)", r"0\.0232", "0.0232"),
        ("Cohen's d Fixed Basic vs CNN (Breast ROC)", r"0\.815", "0.815"),
        ("Cohen's d Fixed Strongly vs CNN (Breast PR)", r"1\.332", "1.332"),
        ("Cohen's d CNN vs QNN (OCT ROC)", r"2\.108", "2.108"),
        ("Cohen's d Trainable vs Fixed Strongly (OCT)", r"1\.050", "1.050"),
        ("Do tre suy luan Quanvolution CPU (ms)", r"220\.22", "220.22"),
        ("Do tre suy luan Classical CNN CPU (ms)", r"0\.31", "0.31"),
        ("Tham so Classifier Head (Breast)", r"1\{,\}570", "1,570"),
        ("Tham so Classifier Head (OCT)", r"3\{,\}140", "3,140"),
        ("Ty le giam phuong sai (Std reduction)", r"2\.7\\times", "2.7x"),
    ]

    passed = 0
    for name, pattern, expected in checks:
        if re.search(pattern, tex):
            print(f"  [PASS] {name:45s}: {expected}")
            passed += 1
        else:
            print(f"  [FAIL] {name:45s}: Khong tim thay '{expected}'")

    print(f"\n=> Ket qua kiem tra so lieu: {passed}/{len(checks)} ({passed/len(checks)*100:.1f}%)")

    print("\n" + "=" * 80)
    print("2. KIEM DINH CITATION (SAU VONG THAM DINH ZERO-TOLERANCE)")
    print("=" * 80)

    ref_checks = [
        ("Azevedo 2022 dung tac gia", r"V\.~?Azevedo|Azevedo, V\.", "Azevedo, V."),
        ("Azevedo 2022 QMI 4(1) p.5", r"vol\.~4, no\.~1, p\.~5|\\textbf\{4\}\(1\), 5", "4(1),5"),
        ("Matondo-Mvula 2024 (thay the Sannakki bia)", r"Matondo-Mvula", "Matondo-Mvula"),
        ("Entropy 26(8) 630", r"Entropy", "Entropy"),
        ("Schuld & Killoran 2022 PRX Quantum 030101", r"PRX Quantum", "PRX Quantum"),
        ("Vu 2024 article 29", r"p\.~?29|\\textbf\{6\}\(1\), 29", "29"),
        ("Khong con Sannakki", r"Sannakki", None),
        ("Khong con ref gia 'extra steps'", r"extra steps", None),
        ("Khong con Quantum Information Processing (journal cu sai)", r"Quantum Information Processing", None),
    ]

    for name, pattern, expected in ref_checks:
        found = re.search(pattern, tex)
        if expected is None:
            print(f"  [{'PASS' if not found else 'FAIL'}] {name}: {'da sach' if not found else 'VAN CON!'}")
        else:
            print(f"  [{'PASS' if found else 'FAIL'}] {name}: {expected}")

    print("\n" + "=" * 80)
    print("3. KIEM DINH ANTI-OVERCLAIM + VAN PHONG")
    print("=" * 80)

    for bad, desc in [(r"\[0\.05, 0\.25\]", "gradient interval cu"),
                      (r"2\.73", "ty so 2.73 cu"),
                      (r"prove that quantum advantages", "'we prove' overclaim")]:
        print(f"  [{'PASS - removed' if not re.search(bad, tex) else 'FAIL - con ton tai'}] {desc}")

    ai_slop = ["delve", "testament", "tapestry", "worth noting", "game changer",
               "beacon", "plethora", "groundbreaking", "revolutioniz"]
    slop = [w for w in ai_slop if re.search(r'\b' + w, tex, re.I)]
    print(f"  [{'PASS' if not slop else 'FAIL ' + str(slop)}] AI-slop scan")


if __name__ == "__main__":
    verify()
