import re
import sys
import io

# Force UTF-8 stdout
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def verify():
    with open(r'd:\KhoaLuanTotNghiep\PAPER\manuscript_ieee.tex', 'r', encoding='utf-8') as f:
        tex = f.read()

    print("=" * 80)
    print("1. KIEM DINH TINH CHINH XAC CUA SO LIEU THUC NGHIEM (VS JSON GD3 10-SEEDS)")
    print("=" * 80)
    
    checks = [
        ("Breast Fixed Basic L2 ROC-AUC (Quan quan)", r"0\.8521 \\pm 0\.0090", "0.8521"),
        ("Breast Fixed Basic L2 ROC-AUC Std", r"0\.0090", "0.0090"),
        ("Breast Fixed Strongly L2 PR-AUC (Quan quan)", r"0\.9182 \\pm 0\.0067", "0.9182"),
        ("Breast Fixed Strongly L2 PR-AUC Std", r"0\.0067", "0.0067"),
        ("Breast Classical CNN ROC-AUC", r"0\.8336 \\pm 0\.0246", "0.8336"),
        ("Breast Classical CNN PR-AUC", r"0\.9041 \\pm 0\.0095", "0.9041"),
        ("Breast Trainable Strongly L2 BAcc", r"0\.6945 \\pm 0\.0428", "0.6945"),
        ("OCT Classical CNN ROC-AUC (Ap dao)", r"0\.7505 \\pm 0\.0227", "0.7505"),
        ("OCT Classical CNN ROC-AUC Std", r"0\.0227", "0.0227"),
        ("OCT Classical CNN PR-AUC", r"0\.4991 \\pm 0\.0282", "0.4991"),
        ("OCT Trainable Strongly L1 ROC-AUC", r"0\.6922 \\pm 0\.0189", "0.6922"),
        ("OCT Trainable Strongly L1 ROC-AUC Std", r"0\.0189", "0.0189"),
        ("OCT Fixed Strongly L1 ROC-AUC", r"0\.6690 \\pm 0\.0052", "0.6690"),
        ("OCT Fixed Champion (random_L1) ROC-AUC", r"0\.6912 \\pm 0\.0067", "0.6912"),
        ("Delta Trainable vs Fixed Strongly (OCT)", r"0\.0232", "0.0232"),
        ("Cohen's d Fixed Basic vs CNN (Breast ROC)", r"0\.815", "0.815"),
        ("Cohen's d Fixed Strongly vs CNN (Breast PR)", r"1\.332", "1.332"),
        ("Cohen's d CNN vs QNN (OCT ROC)", r"2\.108", "2.108"),
        ("Cohen's d Trainable vs Fixed Strongly (OCT)", r"1\.050", "1.050"),
        ("Do tre suy luan Quanvolution CPU (ms)", r"220\.22", "220.22"),
        ("Do tre suy luan Classical CNN CPU (ms)", r"0\.31", "0.31"),
        ("Tham so Classifier Head (Breast)", r"1{,}570", "1,570"),
        ("Tham so Classifier Head (OCT)", r"3{,}140", "3,140"),
        ("Ty le giam phuong sai (Std reduction)", r"2\.73", "2.73"),
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
    print("2. KIEM DINH CAC QUY TAC PHONG CHONG OVERCLAIM CUA GIAO VIEN")
    print("=" * 80)
    
    overclaims = [
        ("Trainable 3-axis is the best overall", r"trainable strongly.*is the best overall", "Mach Trainable 3-truc la tot nhat toan dien"),
        ("Quantum dominates classical CNN overall", r"quanvolution.*dominates classical cnn on oct", "Quanvolution thang CNN co dien toan dien"),
        ("Barren plateau is completely absent in QML", r"proves barren plateaus do not exist", "Da chung minh khong co Barren Plateau"),
    ]

    oc_passed = True
    for name, pattern, desc in overclaims:
        if re.search(pattern, tex, re.IGNORECASE):
            print(f"  [FAIL] Phat hien phong dai: '{desc}'")
            oc_passed = False
        else:
            print(f"  [PASS] Khong vi pham loi phong dai: \"{desc}\"")

    core_claims = [
        ("Data regime dependency", r"quantum advantage is strictly data-regime dependent", "Uu the phu thuoc che do du lieu"),
        ("0-parameter inductive bias", r"strictly.*0.*trainable.*parameters", "Suc manh mach tinh 0 tham so"),
        ("Intra-family trainability", r"trainability is localized", "Tinh cuc bo cua trainability trong ho"),
    ]

    for name, pattern, desc in core_claims:
        if re.search(pattern, tex, re.IGNORECASE):
            print(f"  [PASS] Day du thong diep cot loi: \"{desc}\"")
        else:
            print(f"  [FAIL] Thieu thong diep cot loi: \"{desc}\"")
            oc_passed = False

    print("\n" + "=" * 80)
    print("3. KIEM DINH VAN PHONG HOC THUAT & KHU TU SAO RONG AI (PHRASEBANK AUDIT)")
    print("=" * 80)

    ai_slop = [
        "delve", "testament", "tapestry", "worth noting", "game changer",
        "beacon", "plethora", "groundbreaking", "revolutioniz"
    ]
    
    slop_count = 0
    for word in ai_slop:
        if re.search(r'\b' + word, tex, re.IGNORECASE):
            print(f"  [FAIL] Phat hien tu sao rong AI: '{word}'")
            slop_count += 1
            
    if slop_count == 0:
        print("  [PASS] 100% sach tu ngu sao rong may moc AI.")

    phrasebank = [
        ("Introducing paradigm", r"classical deep learning architectures"),
        ("Methodological rigor", r"adhere strictly to the principle of"),
        ("Objective reporting", r"conclusive dominance of classical cnn"),
        ("Deep theoretical discussion", r"structural regularizer"),
        ("Scientific explanation", r"expressibility bottleneck"),
        ("Clinical relevance", r"demonstrates the unique sensitivity"),
        ("Careful hedging", r"ruling out barren plateaus in shallow"),
    ]

    pb_passed = 0
    for label, pattern in phrasebank:
        if re.search(pattern, tex, re.IGNORECASE):
            print(f"  [PASS] {label:28s}: \"{pattern}\"")
            pb_passed += 1
        else:
            print(f"  [FAIL] {label:28s}: Thieu cum mau cau \"{pattern}\"")

    print(f"\n=> Ket qua kiem tra mau cau hoc thuat: {pb_passed}/{len(phrasebank)} ({pb_passed/len(phrasebank)*100:.1f}%)")

if __name__ == "__main__":
    verify()
