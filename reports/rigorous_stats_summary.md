# Rigorous Statistical Analysis: Paired Tests, TOST (±2%), and Benjamini–Hochberg FDR

- **Sample size:** $n = 5$ paired seeds (`[0, 42, 100, 2023, 777]`), degrees of freedom $df = 4$.
- **Equivalence test (TOST):** Two one-sided paired $t$-tests with equivalence bounds $\Delta = \pm 2.0\%$ at $\alpha = 0.05$ (equivalent iff the 90% paired CI lies strictly inside $[-2.0\%, +2.0\%]$, i.e. $p_{\text{TOST}} < 0.05$).
- **Multiple-testing correction:** Benjamini–Hochberg (BH) FDR reported at **family level** (`p_adj_fam`, 5 noise levels per comparison) and **global level** (`p_adj_glob`, across all 345 tests).
- **Power caveat ($n=5$):** With $df=4$ ($t_{0.95,4} = 2.1318$), declaring statistical equivalence within $\pm 2.0\%$ requires paired sample SD $s_d \lesssim 2.1\%$ when $\bar{d} \approx 0$. Inconclusive TOST under high variance is reported transparently as `INCONCLUSIVE`, never as equivalence.

## 1. Extended 7-Branch Ablation Table (Accuracy %, Mean ± Std over 5 Seeds, `Interp4D` for LUTs)

### PNEUMONIAMNIST — Raw Stem (No Phase Damping, `stabilize_tau=999`)

| Model / Branch | $\sigma=0.00$ | $\sigma=0.05$ | $\sigma=0.10$ | $\sigma=0.15$ | $\sigma=0.20$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Ctrl (Conv2d, 148.4K)** | 90.22 ± 1.65 | 88.95 ± 1.70 | 60.60 ± 16.28 | 45.16 ± 6.94 | 40.20 ± 2.68 |
| **Conv2d + tanh [-1,1] (148.4K)** | 90.71 ± 1.64 | 89.62 ± 1.82 | 63.39 ± 11.31 | 44.67 ± 5.02 | 39.69 ± 2.08 |
| **Conv2d + clip [-1,1] (148.4K)** | 90.00 ± 1.28 | 89.28 ± 1.92 | 64.63 ± 15.57 | 46.25 ± 8.91 | 40.37 ± 3.50 |
| **Conv2d + NoiseAug (148.4K)** | 90.77 ± 1.20 | 90.34 ± 1.46 | 89.54 ± 1.42 | 88.77 ± 1.39 | 87.67 ± 1.35 |
| **CLUT-SepTrig (148.3K)** | 90.10 ± 0.79 | 90.01 ± 0.89 | 72.29 ± 14.17 | 50.94 ± 8.24 | 41.46 ± 1.79 |
| **CLUT-RandMLP (148.3K)** | 90.48 ± 1.85 | 88.72 ± 2.53 | 71.53 ± 16.01 | 55.87 ± 13.45 | 46.69 ± 8.92 |
| **Q-LUT SU(16) L=2 (148.3K)** | 88.94 ± 1.63 | 90.37 ± 1.08 | 71.29 ± 8.07 | 46.12 ± 3.50 | 39.49 ± 0.68 |

### PNEUMONIAMNIST — Stabilized Stem Symmetric (`phase_damping_u8` + `stabilize_tau=0.024` on ALL branches)

| Model / Branch | $\sigma=0.00$ | $\sigma=0.05$ | $\sigma=0.10$ | $\sigma=0.15$ | $\sigma=0.20$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Ctrl (Conv2d, 148.4K)** | 90.22 ± 1.65 | 89.48 ± 1.53 | 87.79 ± 2.06 | 86.65 ± 1.87 | 82.68 ± 1.83 |
| **Conv2d + tanh [-1,1] (148.4K)** | 90.71 ± 1.64 | 89.52 ± 1.13 | 88.16 ± 1.59 | 87.50 ± 1.25 | 85.02 ± 1.30 |
| **Conv2d + clip [-1,1] (148.4K)** | 90.00 ± 1.28 | 89.06 ± 1.02 | 87.45 ± 1.61 | 86.29 ± 1.03 | 83.21 ± 2.22 |
| **Conv2d + NoiseAug (148.4K)** | 90.77 ± 1.20 | 89.34 ± 1.70 | 87.40 ± 1.97 | 86.59 ± 2.08 | 85.76 ± 1.89 |
| **CLUT-SepTrig (148.3K)** | 90.10 ± 0.79 | 88.82 ± 1.04 | 86.88 ± 1.68 | 85.67 ± 0.92 | 82.74 ± 1.15 |
| **CLUT-RandMLP (148.3K)** | 90.48 ± 1.85 | 88.78 ± 2.42 | 87.71 ± 2.54 | 86.32 ± 2.04 | 82.52 ± 0.86 |
| **Q-LUT SU(16) L=2 (148.3K)** | 88.94 ± 1.63 | 88.02 ± 1.34 | 85.80 ± 1.28 | 84.78 ± 1.49 | 83.39 ± 0.90 |

### BREASTMNIST — Raw Stem (No Phase Damping, `stabilize_tau=999`)

| Model / Branch | $\sigma=0.00$ | $\sigma=0.05$ | $\sigma=0.10$ | $\sigma=0.15$ | $\sigma=0.20$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Ctrl (Conv2d, 148.4K)** | 84.10 ± 0.94 | 83.76 ± 1.38 | 81.37 ± 2.32 | 78.68 ± 3.12 | 76.07 ± 1.89 |
| **Conv2d + tanh [-1,1] (148.4K)** | 84.23 ± 1.26 | 83.80 ± 0.86 | 81.50 ± 2.00 | 77.91 ± 2.60 | 75.43 ± 2.04 |
| **Conv2d + clip [-1,1] (148.4K)** | 84.87 ± 0.51 | 83.80 ± 0.78 | 81.24 ± 1.26 | 77.69 ± 1.10 | 75.47 ± 0.91 |
| **Conv2d + NoiseAug (148.4K)** | 85.51 ± 1.04 | 84.87 ± 1.86 | 84.96 ± 1.63 | 83.85 ± 1.13 | 82.31 ± 1.37 |
| **CLUT-SepTrig (148.3K)** | 84.87 ± 0.77 | 84.87 ± 0.49 | 82.74 ± 1.79 | 80.13 ± 1.88 | 77.44 ± 1.85 |
| **CLUT-RandMLP (148.3K)** | 85.77 ± 1.43 | 85.56 ± 1.30 | 83.25 ± 1.56 | 80.17 ± 2.36 | 77.69 ± 2.54 |
| **Q-LUT SU(16) L=2 (148.3K)** | 84.74 ± 1.79 | 84.02 ± 1.63 | 81.88 ± 2.78 | 79.53 ± 1.70 | 77.05 ± 1.46 |

### BREASTMNIST — Stabilized Stem Symmetric (`phase_damping_u8` + `stabilize_tau=0.024` on ALL branches)

| Model / Branch | $\sigma=0.00$ | $\sigma=0.05$ | $\sigma=0.10$ | $\sigma=0.15$ | $\sigma=0.20$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Ctrl (Conv2d, 148.4K)** | 84.10 ± 0.94 | 83.80 ± 1.72 | 83.21 ± 1.41 | 81.97 ± 2.85 | 81.58 ± 4.29 |
| **Conv2d + tanh [-1,1] (148.4K)** | 84.23 ± 1.26 | 83.72 ± 0.79 | 84.19 ± 0.70 | 82.78 ± 0.94 | 82.26 ± 1.73 |
| **Conv2d + clip [-1,1] (148.4K)** | 84.87 ± 0.51 | 83.97 ± 0.78 | 85.73 ± 1.49 | 85.13 ± 1.54 | 84.74 ± 1.61 |
| **Conv2d + NoiseAug (148.4K)** | 85.51 ± 1.04 | 84.74 ± 1.90 | 83.46 ± 2.19 | 82.18 ± 2.49 | 81.50 ± 2.76 |
| **CLUT-SepTrig (148.3K)** | 84.87 ± 0.77 | 84.66 ± 0.93 | 83.16 ± 1.92 | 81.45 ± 2.74 | 80.73 ± 2.95 |
| **CLUT-RandMLP (148.3K)** | 85.77 ± 1.43 | 85.94 ± 1.26 | 85.73 ± 2.01 | 84.23 ± 2.05 | 83.21 ± 2.25 |
| **Q-LUT SU(16) L=2 (148.3K)** | 84.74 ± 1.79 | 84.53 ± 1.73 | 85.00 ± 2.80 | 82.95 ± 1.65 | 82.14 ± 1.98 |

## 2. Corrected Paired Analysis of Table 4.2 (Original Asymmetric Isolation, PneumoniaMNIST)

Recomputed read-only from the locked `weights/pillar2_wide/` checkpoints to obtain exact 5-seed pairs for the advisor's Section 3 critique (`(4) vs (2)` fair baseline and `(3) vs (1)` raw-stem claim).

### (4) GR-Quanv + Pre-LUT Damping (Asymmetric) vs (2) Classical Control + Damping

| $\sigma$ | Comparison (A vs B) | Mean A | Mean B | $\Delta$ (A−B) | 95% CI | 90% CI (TOST) | $p_{\text{raw}}$ | $p_{\text{adj,fam}}$ | $p_{\text{adj,glob}}$ | $p_{\text{TOST}}$ | Verdict |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `0.00` | (4) Quanv+PreLUT vs (2) Ctrl+Damp | 88.94% | 90.22% | -1.28% | [-3.83, +1.27] | [-3.24, +0.67] | 0.2350 (ns) | 0.2937 (ns) | 0.5629 (ns) | 0.2389 | INCONCLUSIVE (p_tost=0.239) |
| `0.05` | (4) Quanv+PreLUT vs (2) Ctrl+Damp | 89.96% | 89.48% | +0.48% | [-1.90, +2.87] | [-1.35, +2.31] | 0.6055 (ns) | 0.6055 (ns) | 0.8570 (ns) | 0.0758 | INCONCLUSIVE (p_tost=0.076) |
| `0.10` | (4) Quanv+PreLUT vs (2) Ctrl+Damp | 83.72% | 87.79% | -4.07% | [-11.27, +3.13] | [-9.60, +1.46] | 0.1918 (ns) | 0.2937 (ns) | 0.5089 (ns) | 0.7652 | INCONCLUSIVE (p_tost=0.765) |
| `0.15` | (4) Quanv+PreLUT vs (2) Ctrl+Damp | 77.30% | 86.65% | -9.35% | [-19.39, +0.69] | [-17.06, -1.64] | 0.0610 (ns) | 0.2390 (ns) | 0.3461 (ns) | 0.9441 | INCONCLUSIVE (p_tost=0.944) |
| `0.20` | (4) Quanv+PreLUT vs (2) Ctrl+Damp | 76.62% | 82.68% | -6.06% | [-13.80, +1.69] | [-12.00, -0.11] | 0.0956 (ns) | 0.2390 (ns) | 0.4333 (ns) | 0.8903 | INCONCLUSIVE (p_tost=0.890) |

### (3) GR-Quanv Raw vs (1) Classical Control Raw

| $\sigma$ | Comparison (A vs B) | Mean A | Mean B | $\Delta$ (A−B) | 95% CI | 90% CI (TOST) | $p_{\text{raw}}$ | $p_{\text{adj,fam}}$ | $p_{\text{adj,glob}}$ | $p_{\text{TOST}}$ | Verdict |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `0.00` | (3) Quanv Raw vs (1) Ctrl Raw | 88.94% | 90.22% | -1.28% | [-3.83, +1.27] | [-3.24, +0.67] | 0.2350 (ns) | 0.3916 (ns) | 0.5629 (ns) | 0.2389 | INCONCLUSIVE (p_tost=0.239) |
| `0.05` | (3) Quanv Raw vs (1) Ctrl Raw | 90.37% | 88.95% | +1.42% | [-1.24, +4.08] | [-0.62, +3.46] | 0.2117 (ns) | 0.3916 (ns) | 0.5339 (ns) | 0.2888 | INCONCLUSIVE (p_tost=0.289) |
| `0.10` | (3) Quanv Raw vs (1) Ctrl Raw | 71.29% | 60.60% | +10.69% | [-6.54, +27.93] | [-2.54, +23.93] | 0.1600 (ns) | 0.3916 (ns) | 0.4654 (ns) | 0.8831 | INCONCLUSIVE (p_tost=0.883) |
| `0.15` | (3) Quanv Raw vs (1) Ctrl Raw | 46.12% | 45.16% | +0.96% | [-7.85, +9.77] | [-5.80, +7.73] | 0.7769 (ns) | 0.7769 (ns) | 0.9232 (ns) | 0.3799 | INCONCLUSIVE (p_tost=0.380) |
| `0.20` | (3) Quanv Raw vs (1) Ctrl Raw | 39.49% | 40.20% | -0.72% | [-4.34, +2.91] | [-3.50, +2.07] | 0.6128 (ns) | 0.7660 (ns) | 0.8570 (ns) | 0.1906 | INCONCLUSIVE (p_tost=0.191) |

### (2) Classical Control + Damping vs (1) Classical Control Raw (Isolated Damping Gain)

| $\sigma$ | Comparison (A vs B) | Mean A | Mean B | $\Delta$ (A−B) | 95% CI | 90% CI (TOST) | $p_{\text{raw}}$ | $p_{\text{adj,fam}}$ | $p_{\text{adj,glob}}$ | $p_{\text{TOST}}$ | Verdict |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `0.00` | (2) Ctrl+Damp vs (1) Ctrl Raw | 90.22% | 90.22% | +0.00% | [+0.00, +0.00] | [+0.00, +0.00] | 1.0000 (ns) | 1.0000 (ns) | 1.0000 (ns) | 0.0000 | EQUIV (p=0.0000) |
| `0.05` | (2) Ctrl+Damp vs (1) Ctrl Raw | 89.48% | 88.95% | +0.52% | [-2.77, +3.82] | [-2.00, +3.05] | 0.6817 (ns) | 0.8521 (ns) | 0.8963 (ns) | 0.1405 | INCONCLUSIVE (p_tost=0.141) |
| `0.10` | (2) Ctrl+Damp vs (1) Ctrl Raw | 87.79% | 60.60% | +27.19% | [+3.01, +51.37] | [+8.62, +45.76] | 0.0355 (*) | 0.0591 (ns) | 0.2662 (ns) | 0.9778 | DIFF A>B (p_tost=0.978) |
| `0.15` | (2) Ctrl+Damp vs (1) Ctrl Raw | 86.65% | 45.16% | +41.49% | [+30.94, +52.03] | [+33.39, +49.58] | 0.0004 (***) | 0.0010 (***) | 0.0131 (*) | 0.9998 | DIFF A>B (p_tost=1.000) |
| `0.20` | (2) Ctrl+Damp vs (1) Ctrl Raw | 82.68% | 40.20% | +42.48% | [+40.16, +44.80] | [+40.70, +44.26] | 0.0000 (***) | 0.0000 (***) | 0.0001 (***) | 1.0000 | DIFF A>B (p_tost=1.000) |

## 3. Key Paired Comparisons & TOST (±2%) on Extended 7-Branch Symmetric Ablation (`Interp4D`)

### Dataset: PNEUMONIAMNIST

### PNEUMONIAMNIST | Stabilized Stem Symmetric | Q-LUT vs Ctrl (Conv2d)

| $\sigma$ | Comparison (A vs B) | Mean A | Mean B | $\Delta$ (A−B) | 95% CI | 90% CI (TOST) | $p_{\text{raw}}$ | $p_{\text{adj,fam}}$ | $p_{\text{adj,glob}}$ | $p_{\text{TOST}}$ | Verdict |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `0.00` | Q-LUT vs Ctrl (Conv2d) | 88.94% | 90.22% | -1.28% | [-3.83, +1.27] | [-3.24, +0.67] | 0.2350 (ns) | 0.2937 (ns) | 0.5629 (ns) | 0.2389 | INCONCLUSIVE (p_tost=0.239) |
| `0.05` | Q-LUT vs Ctrl (Conv2d) | 88.02% | 89.48% | -1.45% | [-3.66, +0.75] | [-3.15, +0.24] | 0.1416 (ns) | 0.2361 (ns) | 0.4654 (ns) | 0.2646 | INCONCLUSIVE (p_tost=0.265) |
| `0.10` | Q-LUT vs Ctrl (Conv2d) | 85.80% | 87.79% | -1.99% | [-4.78, +0.81] | [-4.13, +0.16] | 0.1198 (ns) | 0.2361 (ns) | 0.4541 (ns) | 0.4952 | INCONCLUSIVE (p_tost=0.495) |
| `0.15` | Q-LUT vs Ctrl (Conv2d) | 84.78% | 86.65% | -1.87% | [-4.45, +0.71] | [-3.85, +0.11] | 0.1149 (ns) | 0.2361 (ns) | 0.4512 (ns) | 0.4477 | INCONCLUSIVE (p_tost=0.448) |
| `0.20` | Q-LUT vs Ctrl (Conv2d) | 83.39% | 82.68% | +0.71% | [-2.32, +3.73] | [-1.62, +3.03] | 0.5524 (ns) | 0.5524 (ns) | 0.8570 (ns) | 0.1500 | INCONCLUSIVE (p_tost=0.150) |

### PNEUMONIAMNIST | Stabilized Stem Symmetric | Q-LUT vs CLUT-RandMLP (Random Nonlinear LUT)

| $\sigma$ | Comparison (A vs B) | Mean A | Mean B | $\Delta$ (A−B) | 95% CI | 90% CI (TOST) | $p_{\text{raw}}$ | $p_{\text{adj,fam}}$ | $p_{\text{adj,glob}}$ | $p_{\text{TOST}}$ | Verdict |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `0.00` | Q-LUT vs CLUT-RandMLP | 88.94% | 90.48% | -1.54% | [-5.38, +2.31] | [-4.49, +1.41] | 0.3288 (ns) | 0.4366 (ns) | 0.6596 (ns) | 0.3778 | INCONCLUSIVE (p_tost=0.378) |
| `0.05` | Q-LUT vs CLUT-RandMLP | 88.02% | 88.78% | -0.76% | [-4.22, +2.70] | [-3.41, +1.90] | 0.5755 (ns) | 0.5755 (ns) | 0.8570 (ns) | 0.1877 | INCONCLUSIVE (p_tost=0.188) |
| `0.10` | Q-LUT vs CLUT-RandMLP | 85.80% | 87.71% | -1.91% | [-6.01, +2.18] | [-5.06, +1.23] | 0.2647 (ns) | 0.4366 (ns) | 0.5799 (ns) | 0.4778 | INCONCLUSIVE (p_tost=0.478) |
| `0.15` | Q-LUT vs CLUT-RandMLP | 84.78% | 86.32% | -1.55% | [-5.61, +2.51] | [-4.67, +1.57] | 0.3493 (ns) | 0.4366 (ns) | 0.6886 (ns) | 0.3867 | INCONCLUSIVE (p_tost=0.387) |
| `0.20` | Q-LUT vs CLUT-RandMLP | 83.39% | 82.52% | +0.87% | [-0.40, +2.13] | [-0.11, +1.84] | 0.1305 (ns) | 0.4366 (ns) | 0.4654 (ns) | 0.0338 | EQUIV (p=0.0338) |

### PNEUMONIAMNIST | Stabilized Stem Symmetric | Q-LUT vs CLUT-SepTrig (Separable Trig LUT)

| $\sigma$ | Comparison (A vs B) | Mean A | Mean B | $\Delta$ (A−B) | 95% CI | 90% CI (TOST) | $p_{\text{raw}}$ | $p_{\text{adj,fam}}$ | $p_{\text{adj,glob}}$ | $p_{\text{TOST}}$ | Verdict |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `0.00` | Q-LUT vs CLUT-SepTrig | 88.94% | 90.10% | -1.15% | [-3.51, +1.21] | [-2.97, +0.66] | 0.2462 (ns) | 0.3551 (ns) | 0.5637 (ns) | 0.1880 | INCONCLUSIVE (p_tost=0.188) |
| `0.05` | Q-LUT vs CLUT-SepTrig | 88.02% | 88.82% | -0.80% | [-2.33, +0.73] | [-1.97, +0.37] | 0.2190 (ns) | 0.3551 (ns) | 0.5437 (ns) | 0.0474 | EQUIV (p=0.0474) |
| `0.10` | Q-LUT vs CLUT-SepTrig | 85.80% | 86.88% | -1.08% | [-3.41, +1.25] | [-2.87, +0.71] | 0.2673 (ns) | 0.3551 (ns) | 0.5799 (ns) | 0.1667 | INCONCLUSIVE (p_tost=0.167) |
| `0.15` | Q-LUT vs CLUT-SepTrig | 84.78% | 85.67% | -0.90% | [-2.91, +1.12] | [-2.45, +0.65] | 0.2841 (ns) | 0.3551 (ns) | 0.6125 (ns) | 0.1017 | INCONCLUSIVE (p_tost=0.102) |
| `0.20` | Q-LUT vs CLUT-SepTrig | 83.39% | 82.74% | +0.65% | [-1.31, +2.61] | [-0.85, +2.16] | 0.4080 (ns) | 0.4080 (ns) | 0.7488 (ns) | 0.0643 | INCONCLUSIVE (p_tost=0.064) |

### PNEUMONIAMNIST | Stabilized Stem Symmetric | Q-LUT vs Conv2d + tanh [-1, 1]

| $\sigma$ | Comparison (A vs B) | Mean A | Mean B | $\Delta$ (A−B) | 95% CI | 90% CI (TOST) | $p_{\text{raw}}$ | $p_{\text{adj,fam}}$ | $p_{\text{adj,glob}}$ | $p_{\text{TOST}}$ | Verdict |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `0.00` | Q-LUT vs Conv2d+tanh | 88.94% | 90.71% | -1.76% | [-2.89, -0.64] | [-2.63, -0.90] | 0.0122 (*) | 0.0152 (*) | 0.1430 (ns) | 0.2950 | DIFF A<B (p_tost=0.295) |
| `0.05` | Q-LUT vs Conv2d+tanh | 88.02% | 89.52% | -1.50% | [-2.36, -0.63] | [-2.16, -0.83] | 0.0087 (**) | 0.0145 (*) | 0.1152 (ns) | 0.0906 | DIFF A<B (p_tost=0.091) |
| `0.10` | Q-LUT vs Conv2d+tanh | 85.80% | 88.16% | -2.36% | [-3.53, -1.19] | [-3.26, -1.46] | 0.0050 (**) | 0.0125 (*) | 0.0819 (ns) | 0.7801 | DIFF A<B (p_tost=0.780) |
| `0.15` | Q-LUT vs Conv2d+tanh | 84.78% | 87.50% | -2.72% | [-3.96, -1.49] | [-3.67, -1.78] | 0.0036 (**) | 0.0125 (*) | 0.0731 (ns) | 0.9107 | DIFF A<B (p_tost=0.911) |
| `0.20` | Q-LUT vs Conv2d+tanh | 83.39% | 85.02% | -1.63% | [-3.34, +0.07] | [-2.94, -0.33] | 0.0561 (ns) | 0.0561 (ns) | 0.3418 (ns) | 0.2917 | INCONCLUSIVE (p_tost=0.292) |

### PNEUMONIAMNIST | Stabilized Stem Symmetric | Q-LUT vs Conv2d + hard-clip [-1, 1]

| $\sigma$ | Comparison (A vs B) | Mean A | Mean B | $\Delta$ (A−B) | 95% CI | 90% CI (TOST) | $p_{\text{raw}}$ | $p_{\text{adj,fam}}$ | $p_{\text{adj,glob}}$ | $p_{\text{TOST}}$ | Verdict |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `0.00` | Q-LUT vs Conv2d+hardclip | 88.94% | 90.00% | -1.06% | [-2.76, +0.65] | [-2.37, +0.25] | 0.1598 (ns) | 0.2671 (ns) | 0.4654 (ns) | 0.0997 | INCONCLUSIVE (p_tost=0.100) |
| `0.05` | Q-LUT vs Conv2d+hardclip | 88.02% | 89.06% | -1.04% | [-3.03, +0.96] | [-2.57, +0.50] | 0.2228 (ns) | 0.2785 (ns) | 0.5491 (ns) | 0.1256 | INCONCLUSIVE (p_tost=0.126) |
| `0.10` | Q-LUT vs Conv2d+hardclip | 85.80% | 87.45% | -1.65% | [-4.30, +1.01] | [-3.68, +0.39] | 0.1603 (ns) | 0.2671 (ns) | 0.4654 (ns) | 0.3647 | INCONCLUSIVE (p_tost=0.365) |
| `0.15` | Q-LUT vs Conv2d+hardclip | 84.78% | 86.29% | -1.52% | [-3.95, +0.91] | [-3.38, +0.35] | 0.1582 (ns) | 0.2671 (ns) | 0.4654 (ns) | 0.3053 | INCONCLUSIVE (p_tost=0.305) |
| `0.20` | Q-LUT vs Conv2d+hardclip | 83.39% | 83.21% | +0.18% | [-3.23, +3.59] | [-2.43, +2.80] | 0.8895 (ns) | 0.8895 (ns) | 0.9650 (ns) | 0.1063 | INCONCLUSIVE (p_tost=0.106) |

### PNEUMONIAMNIST | Stabilized Stem Symmetric | Q-LUT vs Conv2d + Speckle Noise Augmentation

| $\sigma$ | Comparison (A vs B) | Mean A | Mean B | $\Delta$ (A−B) | 95% CI | 90% CI (TOST) | $p_{\text{raw}}$ | $p_{\text{adj,fam}}$ | $p_{\text{adj,glob}}$ | $p_{\text{TOST}}$ | Verdict |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `0.00` | Q-LUT vs Conv2d+NoiseAug | 88.94% | 90.77% | -1.83% | [-5.14, +1.48] | [-4.37, +0.72] | 0.2002 (ns) | 0.2977 (ns) | 0.5234 (ns) | 0.4458 | INCONCLUSIVE (p_tost=0.446) |
| `0.05` | Q-LUT vs Conv2d+NoiseAug | 88.02% | 89.34% | -1.31% | [-4.36, +1.74] | [-3.66, +1.03] | 0.2977 (ns) | 0.2977 (ns) | 0.6264 (ns) | 0.2831 | INCONCLUSIVE (p_tost=0.283) |
| `0.10` | Q-LUT vs Conv2d+NoiseAug | 85.80% | 87.40% | -1.60% | [-5.25, +2.04] | [-4.40, +1.20] | 0.2894 (ns) | 0.2977 (ns) | 0.6163 (ns) | 0.3886 | INCONCLUSIVE (p_tost=0.389) |
| `0.15` | Q-LUT vs Conv2d+NoiseAug | 84.78% | 86.59% | -1.82% | [-5.95, +2.31] | [-4.99, +1.36] | 0.2892 (ns) | 0.2977 (ns) | 0.6163 (ns) | 0.4538 | INCONCLUSIVE (p_tost=0.454) |
| `0.20` | Q-LUT vs Conv2d+NoiseAug | 83.39% | 85.76% | -2.37% | [-6.07, +1.32] | [-5.21, +0.46] | 0.1493 (ns) | 0.2977 (ns) | 0.4654 (ns) | 0.6031 | INCONCLUSIVE (p_tost=0.603) |

### PNEUMONIAMNIST | Stabilized Stem Symmetric | Conv2d + tanh [-1, 1] vs Unbounded Ctrl (Conv2d)

| $\sigma$ | Comparison (A vs B) | Mean A | Mean B | $\Delta$ (A−B) | 95% CI | 90% CI (TOST) | $p_{\text{raw}}$ | $p_{\text{adj,fam}}$ | $p_{\text{adj,glob}}$ | $p_{\text{TOST}}$ | Verdict |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `0.00` | Conv2d+tanh vs Ctrl (Conv2d) | 90.71% | 90.22% | +0.48% | [-1.72, +2.68] | [-1.21, +2.17] | 0.5764 (ns) | 0.8485 (ns) | 0.8570 (ns) | 0.0637 | INCONCLUSIVE (p_tost=0.064) |
| `0.05` | Conv2d+tanh vs Ctrl (Conv2d) | 89.52% | 89.48% | +0.04% | [-1.94, +2.03] | [-1.48, +1.57] | 0.9553 (ns) | 0.9553 (ns) | 0.9838 (ns) | 0.0261 | EQUIV (p=0.0261) |
| `0.10` | Conv2d+tanh vs Ctrl (Conv2d) | 88.16% | 87.79% | +0.37% | [-1.96, +2.70] | [-1.41, +2.16] | 0.6788 (ns) | 0.8485 (ns) | 0.8963 (ns) | 0.0623 | INCONCLUSIVE (p_tost=0.062) |
| `0.15` | Conv2d+tanh vs Ctrl (Conv2d) | 87.50% | 86.65% | +0.85% | [-0.90, +2.61] | [-0.49, +2.20] | 0.2467 (ns) | 0.6168 (ns) | 0.5637 (ns) | 0.0717 | INCONCLUSIVE (p_tost=0.072) |
| `0.20` | Conv2d+tanh vs Ctrl (Conv2d) | 85.02% | 82.68% | +2.34% | [+0.46, +4.22] | [+0.89, +3.79] | 0.0260 (*) | 0.1302 (ns) | 0.2246 (ns) | 0.6787 | DIFF A>B (p_tost=0.679) |

### PNEUMONIAMNIST | Raw Stem (No Damping) | Q-LUT vs Ctrl (Conv2d)

| $\sigma$ | Comparison (A vs B) | Mean A | Mean B | $\Delta$ (A−B) | 95% CI | 90% CI (TOST) | $p_{\text{raw}}$ | $p_{\text{adj,fam}}$ | $p_{\text{adj,glob}}$ | $p_{\text{TOST}}$ | Verdict |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `0.00` | Q-LUT vs Ctrl (Conv2d) | 88.94% | 90.22% | -1.28% | [-3.83, +1.27] | [-3.24, +0.67] | 0.2350 (ns) | 0.3916 (ns) | 0.5629 (ns) | 0.2389 | INCONCLUSIVE (p_tost=0.239) |
| `0.05` | Q-LUT vs Ctrl (Conv2d) | 90.37% | 88.95% | +1.42% | [-1.24, +4.08] | [-0.62, +3.46] | 0.2117 (ns) | 0.3916 (ns) | 0.5339 (ns) | 0.2888 | INCONCLUSIVE (p_tost=0.289) |
| `0.10` | Q-LUT vs Ctrl (Conv2d) | 71.29% | 60.60% | +10.69% | [-6.54, +27.93] | [-2.54, +23.93] | 0.1600 (ns) | 0.3916 (ns) | 0.4654 (ns) | 0.8831 | INCONCLUSIVE (p_tost=0.883) |
| `0.15` | Q-LUT vs Ctrl (Conv2d) | 46.12% | 45.16% | +0.96% | [-7.85, +9.77] | [-5.80, +7.73] | 0.7769 (ns) | 0.7769 (ns) | 0.9232 (ns) | 0.3799 | INCONCLUSIVE (p_tost=0.380) |
| `0.20` | Q-LUT vs Ctrl (Conv2d) | 39.49% | 40.20% | -0.72% | [-4.34, +2.91] | [-3.50, +2.07] | 0.6128 (ns) | 0.7660 (ns) | 0.8570 (ns) | 0.1906 | INCONCLUSIVE (p_tost=0.191) |

### PNEUMONIAMNIST | Raw Stem (No Damping) | Q-LUT vs CLUT-RandMLP (Random Nonlinear LUT)

| $\sigma$ | Comparison (A vs B) | Mean A | Mean B | $\Delta$ (A−B) | 95% CI | 90% CI (TOST) | $p_{\text{raw}}$ | $p_{\text{adj,fam}}$ | $p_{\text{adj,glob}}$ | $p_{\text{TOST}}$ | Verdict |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `0.00` | Q-LUT vs CLUT-RandMLP | 88.94% | 90.48% | -1.54% | [-5.38, +2.31] | [-4.49, +1.41] | 0.3288 (ns) | 0.4110 (ns) | 0.6596 (ns) | 0.3778 | INCONCLUSIVE (p_tost=0.378) |
| `0.05` | Q-LUT vs CLUT-RandMLP | 90.37% | 88.72% | +1.66% | [-0.71, +4.02] | [-0.16, +3.47] | 0.1242 (ns) | 0.2681 (ns) | 0.4607 (ns) | 0.3537 | INCONCLUSIVE (p_tost=0.354) |
| `0.10` | Q-LUT vs CLUT-RandMLP | 71.29% | 71.53% | -0.24% | [-19.41, +18.94] | [-14.96, +14.49] | 0.9745 (ns) | 0.9745 (ns) | 0.9859 (ns) | 0.4055 | INCONCLUSIVE (p_tost=0.405) |
| `0.15` | Q-LUT vs CLUT-RandMLP | 46.12% | 55.87% | -9.74% | [-24.62, +5.13] | [-21.17, +1.68] | 0.1432 (ns) | 0.2681 (ns) | 0.4654 (ns) | 0.8890 | INCONCLUSIVE (p_tost=0.889) |
| `0.20` | Q-LUT vs CLUT-RandMLP | 39.49% | 46.69% | -7.20% | [-18.84, +4.43] | [-16.13, +1.73] | 0.1609 (ns) | 0.2681 (ns) | 0.4654 (ns) | 0.8588 | INCONCLUSIVE (p_tost=0.859) |

### PNEUMONIAMNIST | Raw Stem (No Damping) | Q-LUT vs CLUT-SepTrig (Separable Trig LUT)

| $\sigma$ | Comparison (A vs B) | Mean A | Mean B | $\Delta$ (A−B) | 95% CI | 90% CI (TOST) | $p_{\text{raw}}$ | $p_{\text{adj,fam}}$ | $p_{\text{adj,glob}}$ | $p_{\text{TOST}}$ | Verdict |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `0.00` | Q-LUT vs CLUT-SepTrig | 88.94% | 90.10% | -1.15% | [-3.51, +1.21] | [-2.97, +0.66] | 0.2462 (ns) | 0.4104 (ns) | 0.5637 (ns) | 0.1880 | INCONCLUSIVE (p_tost=0.188) |
| `0.05` | Q-LUT vs CLUT-SepTrig | 90.37% | 90.01% | +0.36% | [-1.81, +2.54] | [-1.31, +2.03] | 0.6672 (ns) | 0.8148 (ns) | 0.8927 (ns) | 0.0526 | INCONCLUSIVE (p_tost=0.053) |
| `0.10` | Q-LUT vs CLUT-SepTrig | 71.29% | 72.29% | -0.99% | [-12.02, +10.03] | [-9.46, +7.47] | 0.8148 (ns) | 0.8148 (ns) | 0.9236 (ns) | 0.4062 | INCONCLUSIVE (p_tost=0.406) |
| `0.15` | Q-LUT vs CLUT-SepTrig | 46.12% | 50.94% | -4.82% | [-12.03, +2.39] | [-10.35, +0.72] | 0.1371 (ns) | 0.3428 (ns) | 0.4654 (ns) | 0.8306 | INCONCLUSIVE (p_tost=0.831) |
| `0.20` | Q-LUT vs CLUT-SepTrig | 39.49% | 41.46% | -1.98% | [-3.69, -0.26] | [-3.29, -0.66] | 0.0327 (*) | 0.1636 (ns) | 0.2614 (ns) | 0.4857 | DIFF A<B (p_tost=0.486) |

### PNEUMONIAMNIST | Raw Stem (No Damping) | Q-LUT vs Conv2d + tanh [-1, 1]

| $\sigma$ | Comparison (A vs B) | Mean A | Mean B | $\Delta$ (A−B) | 95% CI | 90% CI (TOST) | $p_{\text{raw}}$ | $p_{\text{adj,fam}}$ | $p_{\text{adj,glob}}$ | $p_{\text{TOST}}$ | Verdict |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `0.00` | Q-LUT vs Conv2d+tanh | 88.94% | 90.71% | -1.76% | [-2.89, -0.64] | [-2.63, -0.90] | 0.0122 (*) | 0.0402 (*) | 0.1430 (ns) | 0.2950 | DIFF A<B (p_tost=0.295) |
| `0.05` | Q-LUT vs Conv2d+tanh | 90.37% | 89.62% | +0.76% | [-2.25, +3.76] | [-1.55, +3.07] | 0.5221 (ns) | 0.6526 (ns) | 0.8504 (ns) | 0.1577 | INCONCLUSIVE (p_tost=0.158) |
| `0.10` | Q-LUT vs Conv2d+tanh | 71.29% | 63.39% | +7.91% | [+2.42, +13.39] | [+3.70, +12.12] | 0.0161 (*) | 0.0402 (*) | 0.1546 (ns) | 0.9798 | DIFF A>B (p_tost=0.980) |
| `0.15` | Q-LUT vs Conv2d+tanh | 46.12% | 44.67% | +1.45% | [-1.12, +4.02] | [-0.52, +3.43] | 0.1917 (ns) | 0.3196 (ns) | 0.5089 (ns) | 0.2932 | INCONCLUSIVE (p_tost=0.293) |
| `0.20` | Q-LUT vs Conv2d+tanh | 39.49% | 39.69% | -0.20% | [-2.38, +1.97] | [-1.87, +1.47] | 0.8081 (ns) | 0.8081 (ns) | 0.9232 (ns) | 0.0416 | EQUIV (p=0.0416) |

### PNEUMONIAMNIST | Raw Stem (No Damping) | Q-LUT vs Conv2d + hard-clip [-1, 1]

| $\sigma$ | Comparison (A vs B) | Mean A | Mean B | $\Delta$ (A−B) | 95% CI | 90% CI (TOST) | $p_{\text{raw}}$ | $p_{\text{adj,fam}}$ | $p_{\text{adj,glob}}$ | $p_{\text{TOST}}$ | Verdict |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `0.00` | Q-LUT vs Conv2d+hardclip | 88.94% | 90.00% | -1.06% | [-2.76, +0.65] | [-2.37, +0.25] | 0.1598 (ns) | 0.3996 (ns) | 0.4654 (ns) | 0.0997 | INCONCLUSIVE (p_tost=0.100) |
| `0.05` | Q-LUT vs Conv2d+hardclip | 90.37% | 89.28% | +1.09% | [-2.13, +4.31] | [-1.38, +3.56] | 0.4003 (ns) | 0.6672 (ns) | 0.7472 (ns) | 0.2381 | INCONCLUSIVE (p_tost=0.238) |
| `0.10` | Q-LUT vs Conv2d+hardclip | 71.29% | 64.63% | +6.67% | [-3.94, +17.27] | [-1.48, +14.81] | 0.1559 (ns) | 0.3996 (ns) | 0.4654 (ns) | 0.8555 | INCONCLUSIVE (p_tost=0.856) |
| `0.15` | Q-LUT vs Conv2d+hardclip | 46.12% | 46.25% | -0.13% | [-8.54, +8.28] | [-6.59, +6.33] | 0.9683 (ns) | 0.9683 (ns) | 0.9859 (ns) | 0.2850 | INCONCLUSIVE (p_tost=0.285) |
| `0.20` | Q-LUT vs Conv2d+hardclip | 39.49% | 40.37% | -0.89% | [-5.03, +3.26] | [-4.07, +2.30] | 0.5844 (ns) | 0.7305 (ns) | 0.8570 (ns) | 0.2486 | INCONCLUSIVE (p_tost=0.249) |

### PNEUMONIAMNIST | Raw Stem (No Damping) | Q-LUT vs Conv2d + Speckle Noise Augmentation

| $\sigma$ | Comparison (A vs B) | Mean A | Mean B | $\Delta$ (A−B) | 95% CI | 90% CI (TOST) | $p_{\text{raw}}$ | $p_{\text{adj,fam}}$ | $p_{\text{adj,glob}}$ | $p_{\text{TOST}}$ | Verdict |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `0.00` | Q-LUT vs Conv2d+NoiseAug | 88.94% | 90.77% | -1.83% | [-5.14, +1.48] | [-4.37, +0.72] | 0.2002 (ns) | 0.2503 (ns) | 0.5234 (ns) | 0.4458 | INCONCLUSIVE (p_tost=0.446) |
| `0.05` | Q-LUT vs Conv2d+NoiseAug | 90.37% | 90.34% | +0.03% | [-2.94, +3.00] | [-2.25, +2.31] | 0.9775 (ns) | 0.9775 (ns) | 0.9861 (ns) | 0.0698 | INCONCLUSIVE (p_tost=0.070) |
| `0.10` | Q-LUT vs Conv2d+NoiseAug | 71.29% | 89.54% | -18.25% | [-30.20, -6.30] | [-27.42, -9.07] | 0.0133 (*) | 0.0221 (*) | 0.1430 (ns) | 0.9902 | DIFF A<B (p_tost=0.990) |
| `0.15` | Q-LUT vs Conv2d+NoiseAug | 46.12% | 88.77% | -42.65% | [-48.21, -37.09] | [-46.92, -38.38] | 0.0000 (***) | 0.0001 (***) | 0.0017 (**) | 1.0000 | DIFF A<B (p_tost=1.000) |
| `0.20` | Q-LUT vs Conv2d+NoiseAug | 39.49% | 87.67% | -48.18% | [-50.41, -45.95] | [-49.90, -46.47] | 0.0000 (***) | 0.0000 (***) | 0.0001 (***) | 1.0000 | DIFF A<B (p_tost=1.000) |

### PNEUMONIAMNIST | Raw Stem (No Damping) | Conv2d + tanh [-1, 1] vs Unbounded Ctrl (Conv2d)

| $\sigma$ | Comparison (A vs B) | Mean A | Mean B | $\Delta$ (A−B) | 95% CI | 90% CI (TOST) | $p_{\text{raw}}$ | $p_{\text{adj,fam}}$ | $p_{\text{adj,glob}}$ | $p_{\text{TOST}}$ | Verdict |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `0.00` | Conv2d+tanh vs Ctrl (Conv2d) | 90.71% | 90.22% | +0.48% | [-1.72, +2.68] | [-1.21, +2.17] | 0.5764 (ns) | 0.8945 (ns) | 0.8570 (ns) | 0.0637 | INCONCLUSIVE (p_tost=0.064) |
| `0.05` | Conv2d+tanh vs Ctrl (Conv2d) | 89.62% | 88.95% | +0.66% | [-0.12, +1.44] | [+0.06, +1.26] | 0.0779 (ns) | 0.3893 (ns) | 0.4121 (ns) | 0.0044 | EQUIV (p=0.0044) |
| `0.10` | Conv2d+tanh vs Ctrl (Conv2d) | 63.39% | 60.60% | +2.79% | [-11.66, +17.24] | [-8.31, +13.89] | 0.6206 (ns) | 0.8945 (ns) | 0.8583 (ns) | 0.5565 | INCONCLUSIVE (p_tost=0.557) |
| `0.15` | Conv2d+tanh vs Ctrl (Conv2d) | 44.67% | 45.16% | -0.49% | [-10.16, +9.17] | [-7.91, +6.93] | 0.8945 (ns) | 0.8945 (ns) | 0.9650 (ns) | 0.3435 | INCONCLUSIVE (p_tost=0.344) |
| `0.20` | Conv2d+tanh vs Ctrl (Conv2d) | 39.69% | 40.20% | -0.51% | [-5.02, +3.99] | [-3.97, +2.95] | 0.7678 (ns) | 0.8945 (ns) | 0.9232 (ns) | 0.2057 | INCONCLUSIVE (p_tost=0.206) |

### Dataset: BREASTMNIST

### BREASTMNIST | Stabilized Stem Symmetric | Q-LUT vs Ctrl (Conv2d)

| $\sigma$ | Comparison (A vs B) | Mean A | Mean B | $\Delta$ (A−B) | 95% CI | 90% CI (TOST) | $p_{\text{raw}}$ | $p_{\text{adj,fam}}$ | $p_{\text{adj,glob}}$ | $p_{\text{TOST}}$ | Verdict |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `0.00` | Q-LUT vs Ctrl (Conv2d) | 84.74% | 84.10% | +0.64% | [-2.54, +3.82] | [-1.80, +3.09] | 0.6060 (ns) | 0.7806 (ns) | 0.8570 (ns) | 0.1508 | INCONCLUSIVE (p_tost=0.151) |
| `0.05` | Q-LUT vs Ctrl (Conv2d) | 84.53% | 83.80% | +0.73% | [-2.61, +4.06] | [-1.83, +3.29] | 0.5777 (ns) | 0.7806 (ns) | 0.8570 (ns) | 0.1743 | INCONCLUSIVE (p_tost=0.174) |
| `0.10` | Q-LUT vs Ctrl (Conv2d) | 85.00% | 83.21% | +1.79% | [-3.22, +6.81] | [-2.06, +5.65] | 0.3769 (ns) | 0.7806 (ns) | 0.7291 (ns) | 0.4576 | INCONCLUSIVE (p_tost=0.458) |
| `0.15` | Q-LUT vs Ctrl (Conv2d) | 82.95% | 81.97% | +0.98% | [-4.17, +6.14] | [-2.97, +4.94] | 0.6245 (ns) | 0.7806 (ns) | 0.8583 (ns) | 0.3064 | INCONCLUSIVE (p_tost=0.306) |
| `0.20` | Q-LUT vs Ctrl (Conv2d) | 82.14% | 81.58% | +0.56% | [-6.10, +7.21] | [-4.55, +5.66] | 0.8280 (ns) | 0.8280 (ns) | 0.9305 (ns) | 0.2896 | INCONCLUSIVE (p_tost=0.290) |

### BREASTMNIST | Stabilized Stem Symmetric | Q-LUT vs CLUT-RandMLP (Random Nonlinear LUT)

| $\sigma$ | Comparison (A vs B) | Mean A | Mean B | $\Delta$ (A−B) | 95% CI | 90% CI (TOST) | $p_{\text{raw}}$ | $p_{\text{adj,fam}}$ | $p_{\text{adj,glob}}$ | $p_{\text{TOST}}$ | Verdict |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `0.00` | Q-LUT vs CLUT-RandMLP | 84.74% | 85.77% | -1.03% | [-5.15, +3.09] | [-4.19, +2.14] | 0.5275 (ns) | 0.6594 (ns) | 0.8504 (ns) | 0.2737 | INCONCLUSIVE (p_tost=0.274) |
| `0.05` | Q-LUT vs CLUT-RandMLP | 84.53% | 85.94% | -1.41% | [-5.38, +2.56] | [-4.46, +1.64] | 0.3802 (ns) | 0.6594 (ns) | 0.7291 (ns) | 0.3507 | INCONCLUSIVE (p_tost=0.351) |
| `0.10` | Q-LUT vs CLUT-RandMLP | 85.00% | 85.73% | -0.73% | [-6.51, +5.05] | [-5.17, +3.71] | 0.7448 (ns) | 0.7448 (ns) | 0.9232 (ns) | 0.2869 | INCONCLUSIVE (p_tost=0.287) |
| `0.15` | Q-LUT vs CLUT-RandMLP | 82.95% | 84.23% | -1.28% | [-4.93, +2.37] | [-4.09, +1.52] | 0.3849 (ns) | 0.6594 (ns) | 0.7297 (ns) | 0.3071 | INCONCLUSIVE (p_tost=0.307) |
| `0.20` | Q-LUT vs CLUT-RandMLP | 82.14% | 83.21% | -1.07% | [-5.06, +2.92] | [-4.13, +2.00] | 0.4988 (ns) | 0.6594 (ns) | 0.8395 (ns) | 0.2762 | INCONCLUSIVE (p_tost=0.276) |

### BREASTMNIST | Stabilized Stem Symmetric | Q-LUT vs CLUT-SepTrig (Separable Trig LUT)

| $\sigma$ | Comparison (A vs B) | Mean A | Mean B | $\Delta$ (A−B) | 95% CI | 90% CI (TOST) | $p_{\text{raw}}$ | $p_{\text{adj,fam}}$ | $p_{\text{adj,glob}}$ | $p_{\text{TOST}}$ | Verdict |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `0.00` | Q-LUT vs CLUT-SepTrig | 84.74% | 84.87% | -0.13% | [-3.23, +2.97] | [-2.51, +2.25] | 0.9142 (ns) | 0.9170 (ns) | 0.9675 (ns) | 0.0846 | INCONCLUSIVE (p_tost=0.085) |
| `0.05` | Q-LUT vs CLUT-SepTrig | 84.53% | 84.66% | -0.13% | [-3.34, +3.08] | [-2.59, +2.34] | 0.9170 (ns) | 0.9170 (ns) | 0.9675 (ns) | 0.0903 | INCONCLUSIVE (p_tost=0.090) |
| `0.10` | Q-LUT vs CLUT-SepTrig | 85.00% | 83.16% | +1.84% | [-4.36, +8.03] | [-2.92, +6.59] | 0.4563 (ns) | 0.9170 (ns) | 0.8040 (ns) | 0.4727 | INCONCLUSIVE (p_tost=0.473) |
| `0.15` | Q-LUT vs CLUT-SepTrig | 82.95% | 81.45% | +1.50% | [-4.01, +7.00] | [-2.73, +5.72] | 0.4924 (ns) | 0.9170 (ns) | 0.8327 (ns) | 0.4058 | INCONCLUSIVE (p_tost=0.406) |
| `0.20` | Q-LUT vs CLUT-SepTrig | 82.14% | 80.73% | +1.41% | [-5.05, +7.87] | [-3.55, +6.37] | 0.5770 (ns) | 0.9170 (ns) | 0.8570 (ns) | 0.4062 | INCONCLUSIVE (p_tost=0.406) |

### BREASTMNIST | Stabilized Stem Symmetric | Q-LUT vs Conv2d + tanh [-1, 1]

| $\sigma$ | Comparison (A vs B) | Mean A | Mean B | $\Delta$ (A−B) | 95% CI | 90% CI (TOST) | $p_{\text{raw}}$ | $p_{\text{adj,fam}}$ | $p_{\text{adj,glob}}$ | $p_{\text{TOST}}$ | Verdict |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `0.00` | Q-LUT vs Conv2d+tanh | 84.74% | 84.23% | +0.51% | [-1.02, +2.04] | [-0.66, +1.69] | 0.4050 (ns) | 0.9265 (ns) | 0.7472 (ns) | 0.0271 | EQUIV (p=0.0271) |
| `0.05` | Q-LUT vs Conv2d+tanh | 84.53% | 83.72% | +0.81% | [-0.83, +2.45] | [-0.45, +2.07] | 0.2412 (ns) | 0.9265 (ns) | 0.5637 (ns) | 0.0573 | INCONCLUSIVE (p_tost=0.057) |
| `0.10` | Q-LUT vs Conv2d+tanh | 85.00% | 84.19% | +0.81% | [-3.18, +4.81] | [-2.26, +3.88] | 0.6027 (ns) | 0.9265 (ns) | 0.8570 (ns) | 0.2277 | INCONCLUSIVE (p_tost=0.228) |
| `0.15` | Q-LUT vs Conv2d+tanh | 82.95% | 82.78% | +0.17% | [-2.27, +2.61] | [-1.70, +2.05] | 0.8554 (ns) | 0.9265 (ns) | 0.9458 (ns) | 0.0530 | INCONCLUSIVE (p_tost=0.053) |
| `0.20` | Q-LUT vs Conv2d+tanh | 82.14% | 82.26% | -0.13% | [-3.75, +3.50] | [-2.91, +2.66] | 0.9265 (ns) | 0.9265 (ns) | 0.9745 (ns) | 0.1125 | INCONCLUSIVE (p_tost=0.113) |

### BREASTMNIST | Stabilized Stem Symmetric | Q-LUT vs Conv2d + hard-clip [-1, 1]

| $\sigma$ | Comparison (A vs B) | Mean A | Mean B | $\Delta$ (A−B) | 95% CI | 90% CI (TOST) | $p_{\text{raw}}$ | $p_{\text{adj,fam}}$ | $p_{\text{adj,glob}}$ | $p_{\text{TOST}}$ | Verdict |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `0.00` | Q-LUT vs Conv2d+hardclip | 84.74% | 84.87% | -0.13% | [-3.28, +3.03] | [-2.55, +2.29] | 0.9156 (ns) | 0.9156 (ns) | 0.9675 (ns) | 0.0874 | INCONCLUSIVE (p_tost=0.087) |
| `0.05` | Q-LUT vs Conv2d+hardclip | 84.53% | 83.97% | +0.56% | [-2.38, +3.50] | [-1.70, +2.81] | 0.6276 (ns) | 0.7845 (ns) | 0.8592 (ns) | 0.1221 | INCONCLUSIVE (p_tost=0.122) |
| `0.10` | Q-LUT vs Conv2d+hardclip | 85.00% | 85.73% | -0.73% | [-3.19, +1.74] | [-2.62, +1.17] | 0.4591 (ns) | 0.7651 (ns) | 0.8040 (ns) | 0.1124 | INCONCLUSIVE (p_tost=0.112) |
| `0.15` | Q-LUT vs Conv2d+hardclip | 82.95% | 85.13% | -2.18% | [-4.58, +0.23] | [-4.03, -0.33] | 0.0657 (ns) | 0.2490 (ns) | 0.3568 (ns) | 0.5770 | INCONCLUSIVE (p_tost=0.577) |
| `0.20` | Q-LUT vs Conv2d+hardclip | 82.14% | 84.74% | -2.61% | [-6.00, +0.78] | [-5.21, -0.00] | 0.0996 (ns) | 0.2490 (ns) | 0.4350 (ns) | 0.6774 | INCONCLUSIVE (p_tost=0.677) |

### BREASTMNIST | Stabilized Stem Symmetric | Q-LUT vs Conv2d + Speckle Noise Augmentation

| $\sigma$ | Comparison (A vs B) | Mean A | Mean B | $\Delta$ (A−B) | 95% CI | 90% CI (TOST) | $p_{\text{raw}}$ | $p_{\text{adj,fam}}$ | $p_{\text{adj,glob}}$ | $p_{\text{TOST}}$ | Verdict |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `0.00` | Q-LUT vs Conv2d+NoiseAug | 84.74% | 85.51% | -0.77% | [-4.48, +2.94] | [-3.62, +2.08] | 0.5955 (ns) | 0.9044 (ns) | 0.8570 (ns) | 0.2044 | INCONCLUSIVE (p_tost=0.204) |
| `0.05` | Q-LUT vs Conv2d+NoiseAug | 84.53% | 84.74% | -0.21% | [-4.85, +4.42] | [-3.77, +3.35] | 0.9044 (ns) | 0.9044 (ns) | 0.9675 (ns) | 0.1725 | INCONCLUSIVE (p_tost=0.173) |
| `0.10` | Q-LUT vs Conv2d+NoiseAug | 85.00% | 83.46% | +1.54% | [-4.86, +7.94] | [-3.38, +6.45] | 0.5411 (ns) | 0.9044 (ns) | 0.8570 (ns) | 0.4256 | INCONCLUSIVE (p_tost=0.426) |
| `0.15` | Q-LUT vs Conv2d+NoiseAug | 82.95% | 82.18% | +0.77% | [-4.93, +6.47] | [-3.61, +5.15] | 0.7270 (ns) | 0.9044 (ns) | 0.9187 (ns) | 0.2906 | INCONCLUSIVE (p_tost=0.291) |
| `0.20` | Q-LUT vs Conv2d+NoiseAug | 82.14% | 81.50% | +0.64% | [-5.77, +7.05] | [-4.28, +5.56] | 0.7950 (ns) | 0.9044 (ns) | 0.9232 (ns) | 0.2938 | INCONCLUSIVE (p_tost=0.294) |

### BREASTMNIST | Stabilized Stem Symmetric | Conv2d + tanh [-1, 1] vs Unbounded Ctrl (Conv2d)

| $\sigma$ | Comparison (A vs B) | Mean A | Mean B | $\Delta$ (A−B) | 95% CI | 90% CI (TOST) | $p_{\text{raw}}$ | $p_{\text{adj,fam}}$ | $p_{\text{adj,glob}}$ | $p_{\text{TOST}}$ | Verdict |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `0.00` | Conv2d+tanh vs Ctrl (Conv2d) | 84.23% | 84.10% | +0.13% | [-2.48, +2.73] | [-1.87, +2.13] | 0.8979 (ns) | 0.9400 (ns) | 0.9650 (ns) | 0.0583 | INCONCLUSIVE (p_tost=0.058) |
| `0.05` | Conv2d+tanh vs Ctrl (Conv2d) | 83.72% | 83.80% | -0.09% | [-3.05, +2.88] | [-2.36, +2.19] | 0.9400 (ns) | 0.9400 (ns) | 0.9838 (ns) | 0.0737 | INCONCLUSIVE (p_tost=0.074) |
| `0.10` | Conv2d+tanh vs Ctrl (Conv2d) | 84.19% | 83.21% | +0.98% | [-0.34, +2.30] | [-0.03, +2.00] | 0.1078 (ns) | 0.5389 (ns) | 0.4375 (ns) | 0.0497 | EQUIV (p=0.0497) |
| `0.15` | Conv2d+tanh vs Ctrl (Conv2d) | 82.78% | 81.97% | +0.81% | [-4.02, +5.65] | [-2.90, +4.52] | 0.6653 (ns) | 0.9400 (ns) | 0.8927 (ns) | 0.2663 | INCONCLUSIVE (p_tost=0.266) |
| `0.20` | Conv2d+tanh vs Ctrl (Conv2d) | 82.26% | 81.58% | +0.68% | [-6.56, +7.92] | [-4.88, +6.24] | 0.8061 (ns) | 0.9400 (ns) | 0.9232 (ns) | 0.3201 | INCONCLUSIVE (p_tost=0.320) |

### BREASTMNIST | Raw Stem (No Damping) | Q-LUT vs Ctrl (Conv2d)

| $\sigma$ | Comparison (A vs B) | Mean A | Mean B | $\Delta$ (A−B) | 95% CI | 90% CI (TOST) | $p_{\text{raw}}$ | $p_{\text{adj,fam}}$ | $p_{\text{adj,glob}}$ | $p_{\text{TOST}}$ | Verdict |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `0.00` | Q-LUT vs Ctrl (Conv2d) | 84.74% | 84.10% | +0.64% | [-2.54, +3.82] | [-1.80, +3.09] | 0.6060 (ns) | 0.8222 (ns) | 0.8570 (ns) | 0.1508 | INCONCLUSIVE (p_tost=0.151) |
| `0.05` | Q-LUT vs Ctrl (Conv2d) | 84.02% | 83.76% | +0.26% | [-2.60, +3.12] | [-1.94, +2.45] | 0.8157 (ns) | 0.8222 (ns) | 0.9236 (ns) | 0.0829 | INCONCLUSIVE (p_tost=0.083) |
| `0.10` | Q-LUT vs Ctrl (Conv2d) | 81.88% | 81.37% | +0.51% | [-5.42, +6.45] | [-4.04, +5.07] | 0.8222 (ns) | 0.8222 (ns) | 0.9270 (ns) | 0.2624 | INCONCLUSIVE (p_tost=0.262) |
| `0.15` | Q-LUT vs Ctrl (Conv2d) | 79.53% | 78.68% | +0.85% | [-4.90, +6.61] | [-3.56, +5.27] | 0.7012 (ns) | 0.8222 (ns) | 0.8997 (ns) | 0.3050 | INCONCLUSIVE (p_tost=0.305) |
| `0.20` | Q-LUT vs Ctrl (Conv2d) | 77.05% | 76.07% | +0.98% | [-2.96, +4.93] | [-2.04, +4.01] | 0.5269 (ns) | 0.8222 (ns) | 0.8504 (ns) | 0.2567 | INCONCLUSIVE (p_tost=0.257) |

### BREASTMNIST | Raw Stem (No Damping) | Q-LUT vs CLUT-RandMLP (Random Nonlinear LUT)

| $\sigma$ | Comparison (A vs B) | Mean A | Mean B | $\Delta$ (A−B) | 95% CI | 90% CI (TOST) | $p_{\text{raw}}$ | $p_{\text{adj,fam}}$ | $p_{\text{adj,glob}}$ | $p_{\text{TOST}}$ | Verdict |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `0.00` | Q-LUT vs CLUT-RandMLP | 84.74% | 85.77% | -1.03% | [-5.15, +3.09] | [-4.19, +2.14] | 0.5275 (ns) | 0.5275 (ns) | 0.8504 (ns) | 0.2737 | INCONCLUSIVE (p_tost=0.274) |
| `0.05` | Q-LUT vs CLUT-RandMLP | 84.02% | 85.56% | -1.54% | [-5.21, +2.13] | [-4.36, +1.28] | 0.3096 (ns) | 0.5275 (ns) | 0.6395 (ns) | 0.3724 | INCONCLUSIVE (p_tost=0.372) |
| `0.10` | Q-LUT vs CLUT-RandMLP | 81.88% | 83.25% | -1.37% | [-5.05, +2.31] | [-4.19, +1.46] | 0.3604 (ns) | 0.5275 (ns) | 0.7064 (ns) | 0.3290 | INCONCLUSIVE (p_tost=0.329) |
| `0.15` | Q-LUT vs CLUT-RandMLP | 79.53% | 80.17% | -0.64% | [-2.79, +1.51] | [-2.29, +1.01] | 0.4538 (ns) | 0.5275 (ns) | 0.8040 (ns) | 0.0769 | INCONCLUSIVE (p_tost=0.077) |
| `0.20` | Q-LUT vs CLUT-RandMLP | 77.05% | 77.69% | -0.64% | [-2.76, +1.48] | [-2.27, +0.99] | 0.4489 (ns) | 0.5275 (ns) | 0.8025 (ns) | 0.0750 | INCONCLUSIVE (p_tost=0.075) |

### BREASTMNIST | Raw Stem (No Damping) | Q-LUT vs CLUT-SepTrig (Separable Trig LUT)

| $\sigma$ | Comparison (A vs B) | Mean A | Mean B | $\Delta$ (A−B) | 95% CI | 90% CI (TOST) | $p_{\text{raw}}$ | $p_{\text{adj,fam}}$ | $p_{\text{adj,glob}}$ | $p_{\text{TOST}}$ | Verdict |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `0.00` | Q-LUT vs CLUT-SepTrig | 84.74% | 84.87% | -0.13% | [-3.23, +2.97] | [-2.51, +2.25] | 0.9142 (ns) | 0.9142 (ns) | 0.9675 (ns) | 0.0846 | INCONCLUSIVE (p_tost=0.085) |
| `0.05` | Q-LUT vs CLUT-SepTrig | 84.02% | 84.87% | -0.85% | [-3.39, +1.68] | [-2.80, +1.09] | 0.4027 (ns) | 0.6883 (ns) | 0.7472 (ns) | 0.1392 | INCONCLUSIVE (p_tost=0.139) |
| `0.10` | Q-LUT vs CLUT-SepTrig | 81.88% | 82.74% | -0.85% | [-3.45, +1.74] | [-2.85, +1.14] | 0.4130 (ns) | 0.6883 (ns) | 0.7512 (ns) | 0.1442 | INCONCLUSIVE (p_tost=0.144) |
| `0.15` | Q-LUT vs CLUT-SepTrig | 79.53% | 80.13% | -0.60% | [-2.12, +0.92] | [-1.76, +0.57] | 0.3351 (ns) | 0.6883 (ns) | 0.6682 (ns) | 0.0311 | EQUIV (p=0.0311) |
| `0.20` | Q-LUT vs CLUT-SepTrig | 77.05% | 77.44% | -0.38% | [-2.83, +2.06] | [-2.26, +1.50] | 0.6853 (ns) | 0.8566 (ns) | 0.8963 (ns) | 0.0705 | INCONCLUSIVE (p_tost=0.070) |

### BREASTMNIST | Raw Stem (No Damping) | Q-LUT vs Conv2d + tanh [-1, 1]

| $\sigma$ | Comparison (A vs B) | Mean A | Mean B | $\Delta$ (A−B) | 95% CI | 90% CI (TOST) | $p_{\text{raw}}$ | $p_{\text{adj,fam}}$ | $p_{\text{adj,glob}}$ | $p_{\text{TOST}}$ | Verdict |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `0.00` | Q-LUT vs Conv2d+tanh | 84.74% | 84.23% | +0.51% | [-1.02, +2.04] | [-0.66, +1.69] | 0.4050 (ns) | 0.6750 (ns) | 0.7472 (ns) | 0.0271 | EQUIV (p=0.0271) |
| `0.05` | Q-LUT vs Conv2d+tanh | 84.02% | 83.80% | +0.21% | [-1.11, +1.54] | [-0.80, +1.23] | 0.6779 (ns) | 0.7368 (ns) | 0.8963 (ns) | 0.0101 | EQUIV (p=0.0101) |
| `0.10` | Q-LUT vs Conv2d+tanh | 81.88% | 81.50% | +0.38% | [-2.58, +3.35] | [-1.89, +2.66] | 0.7368 (ns) | 0.7368 (ns) | 0.9232 (ns) | 0.1023 | INCONCLUSIVE (p_tost=0.102) |
| `0.15` | Q-LUT vs Conv2d+tanh | 79.53% | 77.91% | +1.62% | [-0.92, +4.17] | [-0.33, +3.58] | 0.1508 (ns) | 0.3770 (ns) | 0.4654 (ns) | 0.3512 | INCONCLUSIVE (p_tost=0.351) |
| `0.20` | Q-LUT vs Conv2d+tanh | 77.05% | 75.43% | +1.62% | [-0.50, +3.75] | [-0.01, +3.26] | 0.1014 (ns) | 0.3770 (ns) | 0.4374 (ns) | 0.3246 | INCONCLUSIVE (p_tost=0.325) |

### BREASTMNIST | Raw Stem (No Damping) | Q-LUT vs Conv2d + hard-clip [-1, 1]

| $\sigma$ | Comparison (A vs B) | Mean A | Mean B | $\Delta$ (A−B) | 95% CI | 90% CI (TOST) | $p_{\text{raw}}$ | $p_{\text{adj,fam}}$ | $p_{\text{adj,glob}}$ | $p_{\text{TOST}}$ | Verdict |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `0.00` | Q-LUT vs Conv2d+hardclip | 84.74% | 84.87% | -0.13% | [-3.28, +3.03] | [-2.55, +2.29] | 0.9156 (ns) | 0.9156 (ns) | 0.9675 (ns) | 0.0874 | INCONCLUSIVE (p_tost=0.087) |
| `0.05` | Q-LUT vs Conv2d+hardclip | 84.02% | 83.80% | +0.21% | [-2.41, +2.83] | [-1.80, +2.23] | 0.8319 (ns) | 0.9156 (ns) | 0.9319 (ns) | 0.0656 | INCONCLUSIVE (p_tost=0.066) |
| `0.10` | Q-LUT vs Conv2d+hardclip | 81.88% | 81.24% | +0.64% | [-3.82, +5.10] | [-2.78, +4.07] | 0.7102 (ns) | 0.9156 (ns) | 0.9041 (ns) | 0.2226 | INCONCLUSIVE (p_tost=0.223) |
| `0.15` | Q-LUT vs Conv2d+hardclip | 79.53% | 77.69% | +1.84% | [+0.12, +3.55] | [+0.52, +3.15] | 0.0410 (*) | 0.1592 (ns) | 0.2747 (ns) | 0.4028 | DIFF A>B (p_tost=0.403) |
| `0.20` | Q-LUT vs Conv2d+hardclip | 77.05% | 75.47% | +1.58% | [-0.14, +3.31] | [+0.26, +2.91] | 0.0637 (ns) | 0.1592 (ns) | 0.3544 (ns) | 0.2687 | INCONCLUSIVE (p_tost=0.269) |

### BREASTMNIST | Raw Stem (No Damping) | Q-LUT vs Conv2d + Speckle Noise Augmentation

| $\sigma$ | Comparison (A vs B) | Mean A | Mean B | $\Delta$ (A−B) | 95% CI | 90% CI (TOST) | $p_{\text{raw}}$ | $p_{\text{adj,fam}}$ | $p_{\text{adj,glob}}$ | $p_{\text{TOST}}$ | Verdict |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `0.00` | Q-LUT vs Conv2d+NoiseAug | 84.74% | 85.51% | -0.77% | [-4.48, +2.94] | [-3.62, +2.08] | 0.5955 (ns) | 0.6126 (ns) | 0.8570 (ns) | 0.2044 | INCONCLUSIVE (p_tost=0.204) |
| `0.05` | Q-LUT vs Conv2d+NoiseAug | 84.02% | 84.87% | -0.85% | [-5.18, +3.47] | [-4.18, +2.47] | 0.6126 (ns) | 0.6126 (ns) | 0.8570 (ns) | 0.2516 | INCONCLUSIVE (p_tost=0.252) |
| `0.10` | Q-LUT vs Conv2d+NoiseAug | 81.88% | 84.96% | -3.08% | [-8.12, +1.97] | [-6.95, +0.80] | 0.1655 (ns) | 0.2759 (ns) | 0.4682 (ns) | 0.7074 | INCONCLUSIVE (p_tost=0.707) |
| `0.15` | Q-LUT vs Conv2d+NoiseAug | 79.53% | 83.85% | -4.32% | [-7.47, -1.16] | [-6.74, -1.89] | 0.0192 (*) | 0.0480 (*) | 0.1743 (ns) | 0.9443 | DIFF A<B (p_tost=0.944) |
| `0.20` | Q-LUT vs Conv2d+NoiseAug | 77.05% | 82.31% | -5.26% | [-7.80, -2.71] | [-7.21, -3.30] | 0.0046 (**) | 0.0228 (*) | 0.0787 (ns) | 0.9882 | DIFF A<B (p_tost=0.988) |

### BREASTMNIST | Raw Stem (No Damping) | Conv2d + tanh [-1, 1] vs Unbounded Ctrl (Conv2d)

| $\sigma$ | Comparison (A vs B) | Mean A | Mean B | $\Delta$ (A−B) | 95% CI | 90% CI (TOST) | $p_{\text{raw}}$ | $p_{\text{adj,fam}}$ | $p_{\text{adj,glob}}$ | $p_{\text{TOST}}$ | Verdict |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `0.00` | Conv2d+tanh vs Ctrl (Conv2d) | 84.23% | 84.10% | +0.13% | [-2.48, +2.73] | [-1.87, +2.13] | 0.8979 (ns) | 0.9655 (ns) | 0.9650 (ns) | 0.0583 | INCONCLUSIVE (p_tost=0.058) |
| `0.05` | Conv2d+tanh vs Ctrl (Conv2d) | 83.80% | 83.76% | +0.04% | [-2.54, +2.62] | [-1.94, +2.03] | 0.9655 (ns) | 0.9655 (ns) | 0.9859 (ns) | 0.0515 | INCONCLUSIVE (p_tost=0.052) |
| `0.10` | Conv2d+tanh vs Ctrl (Conv2d) | 81.50% | 81.37% | +0.13% | [-5.54, +5.80] | [-4.22, +4.48] | 0.9529 (ns) | 0.9655 (ns) | 0.9838 (ns) | 0.2055 | INCONCLUSIVE (p_tost=0.206) |
| `0.15` | Conv2d+tanh vs Ctrl (Conv2d) | 77.91% | 78.68% | -0.77% | [-8.32, +6.78] | [-6.57, +5.03] | 0.7914 (ns) | 0.9655 (ns) | 0.9232 (ns) | 0.3372 | INCONCLUSIVE (p_tost=0.337) |
| `0.20` | Conv2d+tanh vs Ctrl (Conv2d) | 75.43% | 76.07% | -0.64% | [-5.90, +4.62] | [-4.68, +3.40] | 0.7522 (ns) | 0.9655 (ns) | 0.9232 (ns) | 0.2565 | INCONCLUSIVE (p_tost=0.257) |

## 4. Paired Statistical Cost of 4-bit Input Quantization (`INT8` 256 KiB vs `Interp4D` 1,024 KiB)

### PNEUMONIAMNIST (Pareto Pre-LUT Damped) | INT8 Shift (256 KiB) vs Interp4D (1,024 KiB)

| $\sigma$ | Comparison (A vs B) | Mean A | Mean B | $\Delta$ (A−B) | 95% CI | 90% CI (TOST) | $p_{\text{raw}}$ | $p_{\text{adj,fam}}$ | $p_{\text{adj,glob}}$ | $p_{\text{TOST}}$ | Verdict |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `0.00` | INT8 Shift vs Interp4D | 88.49% | 88.94% | -0.45% | [-0.86, -0.04] | [-0.76, -0.14] | 0.0378 (*) | 0.0465 (*) | 0.2664 (ns) | 0.0002 | EQUIV (p=0.0002) |
| `0.05` | INT8 Shift vs Interp4D | 89.59% | 89.96% | -0.36% | [-0.46, -0.26] | [-0.44, -0.29] | 0.0005 (***) | 0.0026 (**) | 0.0147 (*) | 0.0000 | EQUIV (p=0.0000) |
| `0.10` | INT8 Shift vs Interp4D | 82.31% | 83.72% | -1.41% | [-2.64, -0.18] | [-2.35, -0.47] | 0.0333 (*) | 0.0465 (*) | 0.2614 (ns) | 0.1268 | DIFF A<B (p_tost=0.127) |
| `0.15` | INT8 Shift vs Interp4D | 75.83% | 77.30% | -1.46% | [-2.29, -0.64] | [-2.10, -0.83] | 0.0080 (**) | 0.0200 (*) | 0.1149 (ns) | 0.0732 | DIFF A<B (p_tost=0.073) |
| `0.20` | INT8 Shift vs Interp4D | 74.88% | 76.62% | -1.74% | [-3.44, -0.04] | [-3.05, -0.44] | 0.0465 (*) | 0.0465 (*) | 0.2971 (ns) | 0.3471 | DIFF A<B (p_tost=0.347) |

### BREASTMNIST (Pareto Pre-LUT Damped) | INT8 Shift (256 KiB) vs Interp4D (1,024 KiB)

| $\sigma$ | Comparison (A vs B) | Mean A | Mean B | $\Delta$ (A−B) | 95% CI | 90% CI (TOST) | $p_{\text{raw}}$ | $p_{\text{adj,fam}}$ | $p_{\text{adj,glob}}$ | $p_{\text{TOST}}$ | Verdict |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `0.00` | INT8 Shift vs Interp4D | 84.49% | 84.74% | -0.26% | [-1.81, +1.30] | [-1.45, +0.93] | 0.6702 (ns) | 0.6702 (ns) | 0.8927 (ns) | 0.0178 | EQUIV (p=0.0178) |
| `0.05` | INT8 Shift vs Interp4D | 84.06% | 84.49% | -0.43% | [-1.35, +0.49] | [-1.13, +0.28] | 0.2663 (ns) | 0.5958 (ns) | 0.5799 (ns) | 0.0045 | EQUIV (p=0.0045) |
| `0.10` | INT8 Shift vs Interp4D | 83.42% | 83.50% | -0.09% | [-0.39, +0.22] | [-0.32, +0.15] | 0.4766 (ns) | 0.5958 (ns) | 0.8140 (ns) | 0.0000 | EQUIV (p=0.0000) |
| `0.15` | INT8 Shift vs Interp4D | 82.74% | 82.61% | +0.13% | [-0.27, +0.53] | [-0.18, +0.44] | 0.4263 (ns) | 0.5958 (ns) | 0.7660 (ns) | 0.0001 | EQUIV (p=0.0001) |
| `0.20` | INT8 Shift vs Interp4D | 80.34% | 80.73% | -0.38% | [-0.86, +0.09] | [-0.75, -0.02] | 0.0876 (ns) | 0.4382 (ns) | 0.4320 (ns) | 0.0003 | EQUIV (p=0.0003) |

### OCTMNIST (Pareto Pre-LUT Damped) | INT8 Shift (256 KiB) vs Interp4D (1,024 KiB)

| $\sigma$ | Comparison (A vs B) | Mean A | Mean B | $\Delta$ (A−B) | 95% CI | 90% CI (TOST) | $p_{\text{raw}}$ | $p_{\text{adj,fam}}$ | $p_{\text{adj,glob}}$ | $p_{\text{TOST}}$ | Verdict |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `0.00` | INT8 Shift vs Interp4D | 80.64% | 84.04% | -3.40% | [-3.93, -2.87] | [-3.81, -2.99] | 0.0001 (***) | 0.0003 (***) | 0.0030 (**) | 0.9991 | DIFF A<B (p_tost=0.999) |
| `0.05` | INT8 Shift vs Interp4D | 72.73% | 77.42% | -4.69% | [-5.74, -3.65] | [-5.50, -3.89] | 0.0002 (***) | 0.0006 (***) | 0.0091 (**) | 0.9990 | DIFF A<B (p_tost=0.999) |
| `0.10` | INT8 Shift vs Interp4D | 55.40% | 57.93% | -2.53% | [-3.99, -1.07] | [-3.65, -1.41] | 0.0086 (**) | 0.0143 (*) | 0.1152 (ns) | 0.8135 | DIFF A<B (p_tost=0.814) |
| `0.15` | INT8 Shift vs Interp4D | 44.83% | 45.90% | -1.07% | [-2.36, +0.21] | [-2.06, -0.09] | 0.0810 (ns) | 0.1012 (ns) | 0.4121 (ns) | 0.0578 | INCONCLUSIVE (p_tost=0.058) |
| `0.20` | INT8 Shift vs Interp4D | 40.19% | 40.59% | -0.40% | [-1.06, +0.26] | [-0.90, +0.10] | 0.1656 (ns) | 0.1656 (ns) | 0.4682 (ns) | 0.0012 | EQUIV (p=0.0012) |

### PNEUMONIAMNIST (stabilized_stem) | Q-LUT INT8 Shift vs Interp4D

| $\sigma$ | Comparison (A vs B) | Mean A | Mean B | $\Delta$ (A−B) | 95% CI | 90% CI (TOST) | $p_{\text{raw}}$ | $p_{\text{adj,fam}}$ | $p_{\text{adj,glob}}$ | $p_{\text{TOST}}$ | Verdict |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `0.00` | Q-LUT INT8 Shift vs Interp4D | 88.49% | 88.94% | -0.45% | [-0.86, -0.04] | [-0.76, -0.14] | 0.0378 (*) | 0.1035 (ns) | 0.2664 (ns) | 0.0002 | EQUIV (p=0.0002) |
| `0.05` | Q-LUT INT8 Shift vs Interp4D | 87.72% | 88.02% | -0.30% | [-0.76, +0.16] | [-0.65, +0.05] | 0.1444 (ns) | 0.2406 (ns) | 0.4654 (ns) | 0.0003 | EQUIV (p=0.0003) |
| `0.10` | Q-LUT INT8 Shift vs Interp4D | 85.52% | 85.80% | -0.28% | [-0.81, +0.25] | [-0.68, +0.13] | 0.2189 (ns) | 0.2736 (ns) | 0.5437 (ns) | 0.0004 | EQUIV (p=0.0004) |
| `0.15` | Q-LUT INT8 Shift vs Interp4D | 84.68% | 84.78% | -0.10% | [-0.37, +0.17] | [-0.30, +0.11] | 0.3804 (ns) | 0.3804 (ns) | 0.7291 (ns) | 0.0000 | EQUIV (p=0.0000) |
| `0.20` | Q-LUT INT8 Shift vs Interp4D | 83.03% | 83.39% | -0.35% | [-0.68, -0.02] | [-0.61, -0.10] | 0.0414 (*) | 0.1035 (ns) | 0.2747 (ns) | 0.0001 | EQUIV (p=0.0001) |

### PNEUMONIAMNIST (raw_stem) | Q-LUT INT8 Shift vs Interp4D

| $\sigma$ | Comparison (A vs B) | Mean A | Mean B | $\Delta$ (A−B) | 95% CI | 90% CI (TOST) | $p_{\text{raw}}$ | $p_{\text{adj,fam}}$ | $p_{\text{adj,glob}}$ | $p_{\text{TOST}}$ | Verdict |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `0.00` | Q-LUT INT8 Shift vs Interp4D | 88.49% | 88.94% | -0.45% | [-0.86, -0.04] | [-0.76, -0.14] | 0.0378 (*) | 0.0946 (ns) | 0.2664 (ns) | 0.0002 | EQUIV (p=0.0002) |
| `0.05` | Q-LUT INT8 Shift vs Interp4D | 90.03% | 90.37% | -0.34% | [-0.58, -0.10] | [-0.52, -0.16] | 0.0161 (*) | 0.0807 (ns) | 0.1546 (ns) | 0.0000 | EQUIV (p=0.0000) |
| `0.10` | Q-LUT INT8 Shift vs Interp4D | 70.71% | 71.29% | -0.59% | [-1.29, +0.12] | [-1.13, -0.05] | 0.0812 (ns) | 0.1354 (ns) | 0.4121 (ns) | 0.0025 | EQUIV (p=0.0025) |
| `0.15` | Q-LUT INT8 Shift vs Interp4D | 46.21% | 46.12% | +0.09% | [-0.32, +0.49] | [-0.22, +0.39] | 0.5870 (ns) | 0.6885 (ns) | 0.8570 (ns) | 0.0001 | EQUIV (p=0.0001) |
| `0.20` | Q-LUT INT8 Shift vs Interp4D | 39.51% | 39.49% | +0.02% | [-0.12, +0.16] | [-0.08, +0.13] | 0.6885 (ns) | 0.6885 (ns) | 0.8963 (ns) | 0.0000 | EQUIV (p=0.0000) |

### BREASTMNIST (stabilized_stem) | Q-LUT INT8 Shift vs Interp4D

| $\sigma$ | Comparison (A vs B) | Mean A | Mean B | $\Delta$ (A−B) | 95% CI | 90% CI (TOST) | $p_{\text{raw}}$ | $p_{\text{adj,fam}}$ | $p_{\text{adj,glob}}$ | $p_{\text{TOST}}$ | Verdict |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `0.00` | Q-LUT INT8 Shift vs Interp4D | 84.49% | 84.74% | -0.26% | [-1.81, +1.30] | [-1.45, +0.93] | 0.6702 (ns) | 0.6702 (ns) | 0.8927 (ns) | 0.0178 | EQUIV (p=0.0178) |
| `0.05` | Q-LUT INT8 Shift vs Interp4D | 84.19% | 84.53% | -0.34% | [-1.29, +0.61] | [-1.07, +0.39] | 0.3739 (ns) | 0.6232 (ns) | 0.7288 (ns) | 0.0042 | EQUIV (p=0.0042) |
| `0.10` | Q-LUT INT8 Shift vs Interp4D | 84.23% | 85.00% | -0.77% | [-1.62, +0.08] | [-1.42, -0.12] | 0.0662 (ns) | 0.3309 (ns) | 0.3568 (ns) | 0.0080 | EQUIV (p=0.0080) |
| `0.15` | Q-LUT INT8 Shift vs Interp4D | 83.08% | 82.95% | +0.13% | [-0.42, +0.68] | [-0.29, +0.55] | 0.5529 (ns) | 0.6702 (ns) | 0.8570 (ns) | 0.0004 | EQUIV (p=0.0004) |
| `0.20` | Q-LUT INT8 Shift vs Interp4D | 81.97% | 82.14% | -0.17% | [-0.52, +0.17] | [-0.44, +0.09] | 0.2420 (ns) | 0.6050 (ns) | 0.5637 (ns) | 0.0001 | EQUIV (p=0.0001) |

### BREASTMNIST (raw_stem) | Q-LUT INT8 Shift vs Interp4D

| $\sigma$ | Comparison (A vs B) | Mean A | Mean B | $\Delta$ (A−B) | 95% CI | 90% CI (TOST) | $p_{\text{raw}}$ | $p_{\text{adj,fam}}$ | $p_{\text{adj,glob}}$ | $p_{\text{TOST}}$ | Verdict |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `0.00` | Q-LUT INT8 Shift vs Interp4D | 84.49% | 84.74% | -0.26% | [-1.81, +1.30] | [-1.45, +0.93] | 0.6702 (ns) | 0.8377 (ns) | 0.8927 (ns) | 0.0178 | EQUIV (p=0.0178) |
| `0.05` | Q-LUT INT8 Shift vs Interp4D | 83.85% | 84.02% | -0.17% | [-1.04, +0.70] | [-0.84, +0.50] | 0.6135 (ns) | 0.8377 (ns) | 0.8570 (ns) | 0.0021 | EQUIV (p=0.0021) |
| `0.10` | Q-LUT INT8 Shift vs Interp4D | 81.92% | 81.88% | +0.04% | [-0.59, +0.68] | [-0.44, +0.53] | 0.8605 (ns) | 0.8605 (ns) | 0.9485 (ns) | 0.0005 | EQUIV (p=0.0005) |
| `0.15` | Q-LUT INT8 Shift vs Interp4D | 79.40% | 79.53% | -0.13% | [-0.37, +0.11] | [-0.31, +0.05] | 0.2080 (ns) | 0.8154 (ns) | 0.5339 (ns) | 0.0000 | EQUIV (p=0.0000) |
| `0.20` | Q-LUT INT8 Shift vs Interp4D | 76.84% | 77.05% | -0.21% | [-0.74, +0.32] | [-0.62, +0.19] | 0.3262 (ns) | 0.8154 (ns) | 0.6596 (ns) | 0.0004 | EQUIV (p=0.0004) |

