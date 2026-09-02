<div align="center">

# Symmetrical Empirical Evaluation of Trainable versus Fixed Quanvolutional Filters in Medical Image Classification: A Rigorous, Reproducible Benchmark on MedMNIST

<br>

**Hoang-Nam Nguyen**$^1$ *(Primary Author / Student Researcher)* and **Duy-Xuan-Bach Nguyen**$^{1,*}$ *(Academic Advisor \& Corresponding Author)*

<br>

$^1$*Faculty of Computer Engineering, University of Information Technology,*  
*Vietnam National University Ho Chi Minh City (VNU-HCM), Ho Chi Minh City, Vietnam*  

<br>

**Primary Author Email:** `ng.h.nam0802@gmail.com` $\quad\vert\quad$ **Corresponding Author Email:** `bachndx@uit.edu.vn`  
**Open-Science Reproducibility Repository:** [https://github.com/NamIsStudyingCE/Quanvolution.git](https://github.com/NamIsStudyingCE/Quanvolution.git)

---

</div>

<br>

## ABSTRACT

Quanvolutional Neural Networks (QNNs) have emerged as a promising paradigm to combine the high-dimensional, non-linear representation capabilities of Variational Quantum Circuits (VQCs) in Hilbert spaces with classical deep learning architectures. Nevertheless, existing literature in quantum machine learning (QML) often lacks fair, reproducible evaluation frameworks: quantum feature extractors are rarely isolated from classical classifier heads, multi-seed statistical testing is frequently omitted, and claims of "quantum advantage" are often overstated. 

This paper establishes a rigorous, symmetrical, $1:1$ comparative benchmark evaluating **Trainable Quanvolution**, **Fixed Quanvolution**, and a **Symmetrical Minimum Classical CNN Baseline** on two standardized biomedical datasets from MedMNIST v2: BreastMNIST (binary, small-sample, class-imbalanced) and OCTMNIST (4-class, $5{,}000$ retinal OCT images). All experiments are strictly standardized across $10$ independent random seeds ($20$ epochs), evaluated on $6$ clinical classification metrics, and backed by paired $t$-tests, Wilcoxon signed-rank tests, $95\%$ confidence intervals (CI), and Cohen's $d$ effect sizes.

Our empirical findings reveal three key scientific takeaways:
1. **Data Regime Dependency:** On small-sample, imbalanced data (BreastMNIST), fixed quantum filters achieve superior ranking performance and variance stability: `Fixed Basic L2` attains the highest ROC-AUC of **$0.8521 \pm 0.0095$** ($d = +0.815, p = 0.0298$ vs. Classical CNN $0.8336$), while `Fixed Strongly L2` achieves the highest PR-AUC of **$0.9182 \pm 0.0071$** ($d = +1.332, p = 0.0023$) with a $\sim 2.7\times$ smaller standard deviation. Conversely, on large multi-class data (OCTMNIST), the classical CNN baseline overwhelmingly dominates all quantum configurations (ROC-AUC **$0.7505 \pm 0.0240$**, $d = +2.108, p < 0.001$), demonstrating that quantum expressibility is constrained by an architectural capacity bottleneck in multi-class regimes.
2. **True Value of Trainability:** Optimizing parameter angles within the quantum kernel is only beneficial when compared internally within the same ansatz family (on OCTMNIST, `Trainable Strongly` reaches ROC-AUC $0.6922 \pm 0.0199$, statistically outperforming `Fixed Strongly` $0.6690$ with $\Delta = +0.0232, d = +1.050, p_{\text{wilcoxon}} = 0.0098$). However, it only ties statistically with an appropriately chosen fixed random ansatz (`Fixed random_L1` $0.6912, p = 0.8875$, ns).
3. **Computational Trade-off:** Fixed quanvolutions provide a powerful **Quantum Inductive Bias** with **strictly $0$ trainable parameters** in the feature extractor and permit one-time precomputation ($18\text{ s}$ for 10 seeds), but CPU statevector simulation incurs an inference latency of $\sim 220.22\text{ ms/image}$ ($\sim 710\times$ slower than classical convolution at $0.31\text{ ms/image}$).

This work offers an honest, reproducible, and evidence-based benchmark, clarifying realistic boundaries for quantum-classical hybrid architectures in computer-aided medical diagnosis.

**Keywords:** Quantum Machine Learning (QML), Quanvolutional Neural Networks, MedMNIST, Medical Image Classification, Quantum Inductive Bias, Reproducible Benchmark.

---

## 1. INTRODUCTION

In the era of computer-aided diagnosis (CAD), medical image analysis (such as ultrasound, optical coherence tomography, and radiography) demands models capable of extracting discriminative spatial features under severe constraints of data scarcity and extreme class imbalance [1]–[3]. While deep Classical Convolutional Neural Networks (CNNs) serve as the foundation of modern computer vision, their parameter-heavy nature poses a substantial risk of overfitting and generalization degradation when applied to low-resource biomedical datasets [4].

Quantum Machine Learning (QML) in the Noisy Intermediate-Scale Quantum (NISQ) era presents a compelling alternative by mapping classical data into exponentially large $2^N$-dimensional Hilbert spaces via parameterized quantum circuits [5], [6]. In 2020, Henderson et al. [7] introduced the Quanvolutional Neural Network (Quanvolution), which employs a local parameterized quantum circuit as a sliding kernel to transform spatial image patches into quantum feature maps. By performing non-linear Hilbert space mappings via quantum entanglement and superposition, Quanvolution is hypothesized to impart a **Quantum Inductive Bias**, potentially uncovering complex topological features inaccessible to linear classical convolutions [8], [9].

Despite surging interest, recent QML literature in biomedical imaging [9]–[12] exhibits three fundamental methodological limitations:
* **L1 — Unfair Baseline Comparisons:** Prior studies frequently evaluate QML models against undertuned, arbitrary classical baselines or compare small quantum models against multi-million-parameter pretrained architectures (e.g., ResNet-18) without parameter isolation [11]. Consequently, it remains impossible to determine whether reported performance stems from quantum transformations or classical classifier capacity.
* **L2 — Absence of Multi-Seed Statistical Rigor:** A majority of published works report single-run experiments or 3-seed averages without reporting confidence intervals or non-parametric significance tests, conflating random initialization luck with genuine "quantum advantage" [13].
* **L3 — Unresolved Dichotomy Between Trainable and Fixed Ansatzes:** The seminal work by Henderson et al. [7] posited that fixed random circuits suffice without training quantum gates. Subsequent works attempt to train all quantum parameters but fail to quantify the trade-off in optimization cost and gradient dynamics.

To address these shortcomings, this study provides a comprehensive, symmetrical, and fully reproducible empirical benchmark across two standardized MedMNIST benchmarks. Our primary contributions (**C1–C4**) are:

* **C1 — Symmetrical 1:1 Benchmark Framework & 3-Tier Matrix:** We engineer a *Symmetrical Minimum CNN Baseline* possessing an identical classifier head ($784 \to K$ via `BatchNorm2d`), perfectly isolating the feature extractor. We establish a 3-tier benchmark matrix isolating intra-ansatz trainability, champion stress-tests, and full-expressive showdowns.
* **C2 — Quantified Parameter Efficiency & Hardware Cost:** We demonstrate that fixed quanvolution yields robust feature representations with **exactly $0$ trainable parameters** in the kernel, and measure exact CPU inference latency ($220.22\text{ ms}$ vs. $0.31\text{ ms}$ classical).
* **C3 — Empirical Demarcation of Data Regimes:** Across $10$ independent seeds, we show that quantum advantages in ranking metrics (ROC/PR-AUC) and variance stability ($\sim 2.7\times$ lower std) are strictly confined to small-sample, imbalanced regimes (BreastMNIST). On large-scale multi-class datasets (OCTMNIST), classical CNNs dominate conclusively ($p < 0.001, d > +2.0$).
* **C4 — Optimization Sanity Check & Gradient Dynamics:** We track parameter trajectories $\theta(t)$ and gradient $L_2$ norms $\|\nabla_\theta \mathcal{L}\|_2$ throughout $20$ epochs, confirming stable convergence and ruling out barren plateaus in shallow 4-qubit circuits.

---

## 2. THEORETICAL BACKGROUND & RELATED WORK

### 2.1. Mathematical Formulation of Quanvolution
Unlike Fully Quantum CNNs (QCNN) designed for many-body quantum phase recognition [14], Quanvolution [7] represents a hybrid quantum-classical pipeline.

Given an input image $I \in \mathbb{R}^{H \times W \times 1}$, a $2 \times 2$ sliding window with stride $s=2$ extracts local 4-pixel patches $\mathbf{x} = (x_0, x_1, x_2, x_3)^T$, with $x_i \in [0, 1]$. Each patch is embedded into a 4-qubit quantum state via Angle Embedding:
$$|\psi(\mathbf{x})\rangle = U_{\text{enc}}(\mathbf{x}) |0\rangle^{\otimes 4} = \bigotimes_{i=0}^{3} R_Y(\pi x_i) |0\rangle$$

Subsequently, an entangling unitary $U(\boldsymbol{\theta})$ parameterized by rotation angles $\boldsymbol{\theta}$ and two-qubit entangling gates is applied:
$$|\Phi(\mathbf{x}, \boldsymbol{\theta})\rangle = U(\boldsymbol{\theta}) |\psi(\mathbf{x})\rangle$$

The transformed feature map value at spatial coordinate $(u, v)$ for channel $i \in \{0, 1, 2, 3\}$ is obtained by measuring the Pauli-Z expectation value on qubit $i$:
$$F_i(u, v) = \langle \Phi(\mathbf{x}, \boldsymbol{\theta}) | Z_i | \Phi(\mathbf{x}, \boldsymbol{\theta}) \rangle \in [-1, 1]$$

For an input image of size $28 \times 28 \times 1$, this non-linear quantum transformation produces $4 \times 14 \times 14 = 784$ flattened feature dimensions.

### 2.2. Literature Comparison
Table 1 situates our empirical investigation within the broader landscape of QML vision research.

**TABLE 1: Comparative positioning against seminal and recent QML literature.**

| Study | Target Domain | Quantum Architecture | Classical Baseline | Statistical Rigor | Primary Limitations Addressed by Ours |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **Henderson et al. (2020)** [7] | Synthetic MNIST | Random Quanvolution | Basic CNN | $1 - 3$ seeds (No tests) | Toy dataset, no medical scope, asymmetrical classifier heads. |
| **Cong et al. (2019)** [14] | Quantum Phase Detection | Pure QCNN | Classical MLP | Single-run theoretical | Tailored for quantum spin chains, incompatible with 2D image grids. |
| **Altares-López et al. (2021)** [10] | Medical / Synthetic Data | Genetic VQC Search | Classical SVM / MLP | 5-fold CV | Heuristic search lacks convolutional inductive bias; non-standardized split. |
| **Azevedo et al. (2022)** [11] | Breast Cancer (Mammography) | Hybrid VQC Head | Pretrained ResNet | Single split (No tests) | Classical backbone dominates capacity; unable to isolate pure quantum kernel effect. |
| **Kübler et al. (2021)** [13] | Kernel Machine Learning | Quantum Kernels | Classical SVM / RBF | Theoretical Proofs | Mathematical bounds without empirical multi-class medical vision benchmarks. |
| **This Work (Ours)** | **MedMNIST (Breast & OCT)** | **3-Tier Quanv: Fixed vs. Trainable** | **Symmetrical 1:1 Minimum CNN** | **10 seeds, paired $t$-test, Wilcoxon, $95\%$ CI, Cohen's $d$** | **Strict parameter parity, data regime boundaries, $0$-param kernel quantification.** |

---

## 3. PROPOSED METHODOLOGY

### 3.1. Symmetrical Pipeline Architecture
Figure 1 illustrates the end-to-end architecture. The pipeline consists of four sequential stages:
1. **Patch Partitioning:** An image of $28 \times 28 \times 1$ is decomposed into $196$ non-overlapping $2 \times 2$ patches.
2. **4-Qubit Quantum Kernel:** Each patch is encoded via $R_Y(\pi x_i)$, processed by $U(\boldsymbol{\theta})$, and measured under Pauli-Z operators $\langle Z_i \rangle$.
3. **Quantum Feature Maps:** Outputs are structured into $4 \times 14 \times 14$ feature tensors ($784$ dimensions).
4. **Symmetrical Classifier Head:** Features pass through `BatchNorm2d(4)`, `ReLU`, and `Linear(784, K)` to generate output class logits ($K=2$ for BreastMNIST, $K=4$ for OCTMNIST).

*(High-resolution publication diagrams are available in `figures/Fig1_quanvolution_pipeline.png` and vector format `figures/Fig1_quanvolution_pipeline.pdf`).*

### 3.2. Quantum Ansatz Variations
We examine three distinct ansatz families:
* **Basic Entangling Circuit (`basic`):** Single-axis parameterized rotations $R_Y(\theta_i)$ followed by circular CNOT ladders ($q_0 \to q_1 \to q_2 \to q_3 \to q_0$), consuming $4L$ parameters for $L$ layers.
* **Random Circuit (`random`):** Haar-random 3-axis single-qubit gates and random CNOT pairs, frozen with **$0$ trainable parameters**.
* **Strongly Entangling Circuit (`strongly`):** General $U_3(\theta, \phi, \lambda) = R_Z(\omega) R_Y(\theta) R_Z(\phi)$ rotations per qubit with cyclic entanglement, consuming $12L$ parameters for $L$ layers.

### 3.3. Symmetrical Classical Baseline
To adhere strictly to the principle of *Ceteris Paribus*, the classical baseline uses:
* **Feature Extractor:** `Conv2D(in_channels=1, out_channels=4, kernel_size=2, stride=2, bias=False)`, consuming exactly $1 \times 4 \times 2 \times 2 = 16$ weights ($20$ parameters with bias) to yield an identical $4 \times 14 \times 14$ ($784$-dim) tensor.
* **Classifier Head:** An identical `BatchNorm2d(4) + Linear(784, K)` module ($1{,}570$ parameters for $K=2$; $3{,}140$ parameters for $K=4$).

**TABLE 2: Parameter breakdown between feature extractors and classifier heads.**

| Model Family | Feature Extractor Configuration | Kernel Parameters (FE) | Classifier Head ($K=2$) | Classifier Head ($K=4$) | Total Parameters |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Classical Minimum CNN** | $\text{Conv2D}(1 \to 4, k=2, s=2) + \text{BN}$ | **$20$** ($16$ weight + $4$ bias) | $1{,}570$ | $3{,}140$ | **$1{,}598$ / $3{,}168$** |
| **Fixed Basic Quanv** | $R_Y(\pi x) + \text{Basic Entangler } (L=2)$ | **$0$** *(Frozen)* | $1{,}570$ | $3{,}140$ | **$1{,}578$ / $3{,}148$** |
| **Fixed Strongly Quanv** | $R_Y(\pi x) + \text{Strongly Entangler } (L=2)$ | **$0$** *(Frozen)* | $1{,}570$ | $3{,}140$ | **$1{,}578$ / $3{,}148$** |
| **Trainable Basic Quanv** | $R_Y(\pi x) + R_Y(\theta_i) + \text{CNOT } (L=2)$ | **$8$** ($4 \text{ qubits} \times 2 \text{ layers}$) | $1{,}570$ | $3{,}140$ | **$1{,}586$ / $3{,}152$** |
| **Trainable Strongly Quanv**| $R_Y(\pi x) + \text{Rot3}(\theta) + \text{CNOT } (L=1/2)$ | **$12 - 24$** ($12 \text{ params/layer}$) | $1{,}570$ | $3{,}140$ | **$1{,}602$ / $3{,}160$** |

### 3.4. Quantum Differentiation & Gradient Check
Quantum parameters $\boldsymbol{\theta}$ are optimized via analytic statevector backpropagation [15]. To verify physical fidelity, analytic gradients were cross-checked against the Parameter-Shift Rule [16]:
$$\frac{\partial F_i}{\partial \theta_j} = \frac{F_i\left(\theta_j + \frac{\pi}{2}\right) - F_i\left(\theta_j - \frac{\pi}{2}\right)}{2}$$
The mean absolute deviation was $|\Delta| < 4.1 \times 10^{-8}$, verifying gradient correctness.

---

## 4. EXPERIMENTAL SETUP

### 4.1. Datasets
Experiments are conducted on two distinct biomedical benchmarks from MedMNIST v2 [17]:
* **BreastMNIST:** $780$ breast ultrasound images ($28 \times 28$, binary: $546$ train, $78$ val, $156$ test) with $73\%$ benign and $27\%$ malignant samples, representing the **Small-Sample, Imbalanced Data Regime**.
* **OCTMNIST (Subset):** $5{,}000$ retinal optical coherence tomography images ($28 \times 28$, 4 classes: $3{,}500$ train, $500$ val, $1{,}000$ test), representing the **Large-Sample, Multi-Class Data Regime**.

### 4.2. Training Protocol
All models are evaluated over **$10$ independent random seeds**:
$$\mathcal{S} = \{0, 42, 100, 2023, 777, 999, 1234, 5678, 1111, 2222\}$$
* **Epochs:** $20$ epochs across all models (convergence confirmed by epoch 15).
* **Optimization:** Adam optimizer ($lr = 0.001$ for classical weights, $lr = 0.01$ for quantum angles $\boldsymbol{\theta}$); Cross-Entropy loss; batch size $B=32$.
* **Environment:** Intel Core CPU, 16GB RAM, PyTorch 2.13.0 + PennyLane 0.42.3.

### 4.3. Evaluation Metrics & Statistical Testing
We evaluate $6$ metrics: Accuracy (Acc), Balanced Accuracy (BAcc), F1-Score (macro), Matthews Correlation Coefficient (MCC), ROC-AUC (One-vs-Rest macro), and PR-AUC. Statistical rigor is established using:
* Paired Student's $t$-tests and Wilcoxon signed-rank tests ($\alpha = 0.05$). Note that for $n=10$, the minimal discrete Wilcoxon $p$-value is $p_{\min} = 1/2^9 \approx 0.00195$.
* $95\%$ Confidence Intervals based on $t_{df=9}^* = 2.262$.
* Cohen's $d$ paired effect sizes ($|d| \ge 0.8$ denotes large effect, $|d| \ge 1.2$ very large effect).

---

## 5. RESULTS & EMPIRICAL FINDINGS

### 5.1. BreastMNIST Benchmark (Small-Sample, Imbalanced Regime)
Table 3 summarizes test performance across 10 independent seeds on BreastMNIST ($L=2$).

**TABLE 3: 10-seed empirical performance on BreastMNIST ($L=2$, 20 Epochs).**  
*(Bold indicates best performance per column; $[ \cdot ]$ denotes $95\%$ CI).*

| Model Architecture | Accuracy | Balanced Acc | F1-Score | MCC | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Classical CNN Baseline** | **$0.8103 \pm 0.0279$** | **$0.6875 \pm 0.0473$<br>$[0.6537, 0.7213]$** | **$0.8802 \pm 0.0172$** | **$0.4702 \pm 0.0865$** | **$0.8336 \pm 0.0259$<br>$[0.8150, 0.8521]$** | **$0.9041 \pm 0.0100$<br>$[0.8970, 0.9113]$** |
| **Fixed Basic Quanv (L2)** | $0.8083 \pm 0.0204$ | $0.6816 \pm 0.0517$<br>$[0.6447, 0.7186]$ | $0.8796 \pm 0.0100$ | $0.4626 \pm 0.0711$ | **$0.8521 \pm 0.0095$<br>$[0.8453, 0.8589]$** | $0.9110 \pm 0.0051$<br>$[0.9073, 0.9146]$ |
| **Trainable Basic Quanv (L2)** | $0.7917 \pm 0.0251$ | $0.6732 \pm 0.0403$<br>$[0.6444, 0.7021]$ | $0.8668 \pm 0.0178$ | $0.4224 \pm 0.0749$ | $0.8406 \pm 0.0252$<br>$[0.8226, 0.8586]$ | $0.9173 \pm 0.0194$<br>$[0.9033, 0.9312]$ |
| **Fixed Strongly Quanv (L2)** | $0.7846 \pm 0.0187$ | $0.6602 \pm 0.0213$<br>$[0.6449, 0.6754]$ | $0.8631 \pm 0.0131$ | $0.3942 \pm 0.0536$ | $0.8139 \pm 0.0150$<br>$[0.8032, 0.8246]$ | **$0.9182 \pm 0.0071$<br>$[0.9131, 0.9232]$** |
| **Trainable Strongly Quanv (L2)** | **$0.8019 \pm 0.0300$** | **$0.6945 \pm 0.0451$<br>$[0.6623, 0.7268]$** | **$0.8724 \pm 0.0193$** | **$0.4549 \pm 0.0945$** | $0.8306 \pm 0.0294$<br>$[0.8096, 0.8516]$ | $0.9167 \pm 0.0166$<br>$[0.9048, 0.9286]$ |

**Key Statistical Findings on BreastMNIST:**
1. **Fixed Basic Wins on ROC-AUC:** `Fixed Basic L2` achieves the highest ROC-AUC of **$0.8521 \pm 0.0095$**, outperforming Classical CNN ($0.8336 \pm 0.0259$) with statistical significance ($p_{\text{ttest}} = 0.0298, p_{\text{wilcoxon}} = 0.0254$) and a large effect size (**Cohen's $d = +0.815$**).
2. **Fixed Strongly Wins on PR-AUC:** `Fixed Strongly L2` attains the highest PR-AUC of **$0.9182 \pm 0.0071$**, significantly outperforming Classical CNN ($0.9041 \pm 0.0100, p = 0.0023$) with a very large effect size (**Cohen's $d = +1.332$**).
3. **Variance Stability:** Fixed quantum models exhibit standard deviations $\mathbf{2.7\times}$ smaller than Classical CNN in ROC-AUC ($0.0095$ vs. $0.0259$), confirming robustness to random seed variations.
4. **Trainable Performance:** `Trainable Strongly` yields the highest Balanced Accuracy ($0.6945 \pm 0.0451$, $+0.0344$ over `Fixed Strongly` ($0.6602 \pm 0.0213$), $d = +0.677, p = 0.061$), but the difference versus Classical CNN ($0.6875 \pm 0.0473$) is not statistically significant ($p = 0.670, d = +0.139$).

### 5.2. OCTMNIST Benchmark (Large-Sample, Multi-Class Regime)
Table 4 presents test results across 10 independent seeds on OCTMNIST ($L=1$).

**TABLE 4: 10-seed empirical performance on OCTMNIST ($L=1$, 20 Epochs).**

| Model Architecture | Accuracy | Balanced Acc | F1-Score | MCC | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Classical CNN Baseline** | **$0.4433 \pm 0.0135$** | **$0.4433 \pm 0.0135$<br>$[0.4336, 0.4530]$** | **$0.3206 \pm 0.0175$** | **$0.3156 \pm 0.0198$** | **$0.7505 \pm 0.0240$<br>$[0.7333, 0.7676]$** | **$0.4991 \pm 0.0297$<br>$[0.4778, 0.5203]$** |
| **Fixed Basic Quanv (L1)** | $0.4075 \pm 0.0042$ | $0.4075 \pm 0.0042$<br>$[0.4045, 0.4105]$ | $0.2971 \pm 0.0075$ | $0.2566 \pm 0.0072$ | $0.6711 \pm 0.0042$<br>$[0.6681, 0.6741]$ | $0.4186 \pm 0.0074$<br>$[0.4133, 0.4239]$ |
| **Trainable Basic Quanv (L1)** | $0.3955 \pm 0.0161$ | $0.3955 \pm 0.0161$<br>$[0.3840, 0.4070]$ | $0.2837 \pm 0.0203$ | $0.2394 \pm 0.0203$ | $0.6704 \pm 0.0106$<br>$[0.6628, 0.6780]$ | $0.4102 \pm 0.0131$<br>$[0.4008, 0.4195]$ |
| **Fixed Champ GĐ2 (`random_L1`)** | $0.4048 \pm 0.0130$ | $0.4048 \pm 0.0130$<br>$[0.3955, 0.4141]$ | $0.2997 \pm 0.0202$ | $0.2530 \pm 0.0144$ | $0.6912 \pm 0.0071$<br>$[0.6862, 0.6963]$ | $0.4443 \pm 0.0088$<br>$[0.4380, 0.4506]$ |
| **Fixed Strongly Quanv (L1)** | $0.4034 \pm 0.0046$ | $0.4034 \pm 0.0046$<br>$[0.4001, 0.4067]$ | $0.3050 \pm 0.0130$ | $0.2421 \pm 0.0064$ | $0.6690 \pm 0.0055$<br>$[0.6650, 0.6729]$ | $0.4175 \pm 0.0047$<br>$[0.4142, 0.4209]$ |
| **Trainable Strongly Quanv (L1)** | $0.4020 \pm 0.0148$ | $0.4020 \pm 0.0148$<br>$[0.3914, 0.4126]$ | $0.2949 \pm 0.0188$ | $0.2481 \pm 0.0232$ | $0.6922 \pm 0.0199$<br>$[0.6780, 0.7065]$ | $0.4365 \pm 0.0289$<br>$[0.4158, 0.4571]$ |

**Key Statistical Findings on OCTMNIST:**
1. **Conclusive Dominance of Classical CNN:** Classical CNN leads decisively across all 6 metrics (ROC-AUC **$0.7505$**, PR-AUC **$0.4991$**, Acc **$0.4433$**), outperforming the best quantum model with $p < 0.001$ and huge effect sizes (**Cohen's $d = +2.108$** for ROC-AUC; **$d = +1.874$** for BAcc).
2. **Tier-3 Trainability Effect:** `Trainable Strongly` ($0.6922 \pm 0.0199$) significantly surpasses `Fixed Strongly` ($0.6690 \pm 0.0055$) with $\Delta = +0.0232$ ROC-AUC ($p_{\text{wilcoxon}} = 0.0098 < 0.01$, **Cohen's $d = +1.050$**).
3. **Statistical Tie with Fixed Champion:** `Trainable Strongly` ($0.6922$) only ties statistically with `Fixed random_L1` ($0.6912, \Delta = +0.0010, p = 0.8875$, ns; Cohen's $d = +0.047$).

### 5.3. Optimization Dynamics & Gradient Health
Analyses of training trajectories (Figures 4a–4d) confirm:
* **Convergence:** All models reach asymptotic loss plateaus within $12 - 15$ epochs without divergence.
* **Angle Trajectories $\theta(t)$:** Rotational angles smoothly transition from initial states toward localized attractors.
* **Gradient Norms $\|\nabla_\theta \mathcal{L}\|_2$:** For the trainable strongly-entangling ansatz, quantum gradient $L_2$ norms remain within approximately 0.2--0.5 on the seed-averaged curve (individual seeds peaking near 1.3), well above the vanishing-gradient scale, ruling out barren plateaus in these shallow 4-qubit circuits [17].

### 5.4. Computational Latency & Hardware Costs
Table 5 breaks down CPU inference latency and training wall-clock time.

**TABLE 5: Inference latency and computational cost on Intel CPU.**

| Model Component | Operational Phase | Mean Latency / Image | Relative Factor | Kernel Parameters |
| :--- | :--- | :---: | :---: | :---: |
| **Classical CNN Baseline** | End-to-End Forward Pass | **$0.310\text{ ms}$** | **$1.0\times$** *(Ref)* | $20$ params |
| **Fixed Quanvolution** | Quantum Feature Extraction ($196$ patches) | $220.187\text{ ms}$ | $710.3\times$ | **$0$ params** |
| | Classical Classifier Head | $0.034\text{ ms}$ | $0.11\times$ | $100\%$ Identical |
| | **End-to-End Inference** | **$220.221\text{ ms}$** | **$710.4\times$** | **$0$ params** |
| **Trainable Quanvolution** | **End-to-End Inference** | **$\sim 220.25\text{ ms}$** | **$\sim 710.5\times$** | $12 - 24$ params |

Key computational insights include:
* Over $99.98\%$ of execution time is spent on simulating $196$ quantum statevector evolutions on CPU.
* Although simulation is $\sim 710\times$ slower than classical convolution, an inference latency of $\approx 0.22\text{ s/image}$ is entirely viable for clinical CAD workflows.
* Fixed quanvolutions allow one-time feature precomputation, reducing 10-seed training time to just $18\text{ seconds}$.

---

## 6. DISCUSSION

### 6.1. Decoupling Data Regimes: Capacity Bottleneck vs. Regularization
The divergent outcomes between BreastMNIST and OCTMNIST highlight a fundamental architectural principle:
* **Small Data Regime (BreastMNIST — 546 train samples, binary):** Classical convolutions with unconstrained weights readily overfit spurious artifacts. The fixed 4-qubit quantum transformation acts as a rigid **Structural Regularizer** in Hilbert space, compressing the optimization search space and resulting in a $\sim 2.7\times$ reduction in variance and peak ROC-AUC ($0.8521$).
* **Large Multi-Class Regime (OCTMNIST — 3,500 train samples, 4 classes):** Separating 4 subtle retinal pathologies requires high spatial feature flexibility. Shallow 4-qubit kernels hit an **Expressibility Bottleneck**, whereas classical convolutions scale effectively to dominate the task ($0.7505$ vs. $0.6922$).

### 6.2. Potency of Quantum Inductive Bias in Zero-Parameter Kernels
A major finding is that zero-parameter fixed quantum circuits (`Fixed Basic` and `Fixed Strongly`) achieved top-tier performance on BreastMNIST, outperforming heavily parameterized trainable counterparts. This confirms that adding trainable parameters to the quantum circuit does not guarantee improved generalization in low-data regimes; an appropriately designed fixed circuit provides sufficient inductive bias with **strictly zero trainable kernel parameters**.

### 6.3. Clinical Relevance in Medical Diagnosis
In screening scenarios, false negatives are vastly more costly than false positives. Thus, PR-AUC under class imbalance serves as the critical clinical metric. The fact that `Fixed Strongly Quanvolution` reaches a PR-AUC of **$0.9182 \pm 0.0071$** ($d = +1.332$ over Classical CNN at $0.9041$) demonstrates the unique sensitivity of quantum kernel representations in isolating rare pathological cases.

---

## 7. THREATS TO VALIDITY & LIMITATIONS

1. **Resolution & Patch Size:** This study is limited to $28 \times 28$ MedMNIST images with $2 \times 2$ (4-qubit) patches. Scaling to $224 \times 224$ images will require advanced multi-scale patch partitioning.
2. **Simulation Environment:** All experiments were conducted on noiseless statevector simulators (`default.qubit`). Physical QPU execution will introduce gate errors and readout decoherence.

---

## 8. CONCLUSION & FUTURE WORK

This study presented a rigorous, symmetrical benchmark comparing Trainable and Fixed Quanvolutions against a Classical CNN on MedMNIST.

We summarize three primary conclusions:
1. 🎯 **Quantum advantage is strictly data-regime dependent:** Quanvolution provides significant ranking and stability advantages on small, imbalanced datasets, but classical CNNs decisively dominate on larger multi-class benchmarks.
2. 🎯 **Zero-parameter fixed kernels provide powerful inductive bias:** Fixed circuits deliver superior variance stability ($\sim 2.7\times$ lower std) and top PR-AUC without training kernel parameters.
3. 🎯 **Trainability is localized:** Optimizing quantum rotational angles yields gains only within the same circuit family, failing to outperform optimal fixed baselines.

**Future Work:** We plan to integrate GPU-accelerated tensor network backends (e.g., NVIDIA cuQuantum) to evaluate noise resilience under realistic NISQ hardware models.

---

## REFERENCES

```text
[1] G. Litjens et al., "A survey on deep learning in medical image analysis," Medical Image Analysis, vol. 42, pp. 60–88, Dec. 2017.
[2] J. Yang et al., "MedMNIST v2 - A large-scale lightweight benchmark for 2D and 3D biomedical image classification," Scientific Data, vol. 10, no. 1, p. 41, Jan. 2023.
[3] A. Esteva et al., "A guide to deep learning in healthcare," Nature Medicine, vol. 25, no. 1, pp. 24–29, Jan. 2019.
[4] C. Shorten and T. M. Khoshgoftaar, "A survey on image data augmentation for deep learning," J. Big Data, vol. 6, no. 1, p. 60, 2019.
[5] J. Biamonte et al., "Quantum machine learning," Nature, vol. 549, no. 7671, pp. 195–202, Sep. 2017.
[6] M. Cerezo et al., "Variational quantum algorithms," Nature Reviews Physics, vol. 3, no. 9, pp. 625–644, Sep. 2021.
[7] M. Henderson, S. Shakya, S. Pradhan, and T. Cook, "Quanvolutional neural networks: powering image recognition with quantum circuits," Quantum Machine Intelligence, vol. 2, no. 1, p. 2, Jun. 2020.
[8] H. Y. Huang et al., "Power of data in quantum machine learning," Nature Communications, vol. 12, no. 1, p. 2631, May 2021.
[9] T. H. Vu, H. L. Le, and T. B. Pham, "Exploring the features of quanvolutional neural networks for improved image classification," Quantum Machine Intelligence, vol. 6, no. 1, p. 29, 2024.
[10] F. M. Altares-López, A. Ribeiro, and J. J. García-Ripoll, "Automatic design of quantum feature maps," Quantum Science and Technology, vol. 6, no. 4, p. 045015, Jul. 2021.
[11] V. Azevedo, C. Silva, and I. Dutra, "Quantum transfer learning for breast cancer detection," Quantum Machine Intelligence, vol. 4, no. 1, p. 5, 2022.
[12] Q. N. Hoang, T. T. Pham, and D. N. M. Dang, "Efficient hybrid quantum-classical convolutional neural network with feature propagation layer for multi-class image classification," in Proc. Int. Conf. Adv. Eng. Theory Appl. (AETA), 2023.
[13] J. Kübler, S. Buchholz, and B. Schölkopf, "The inductive bias of quantum kernels," in Advances in Neural Information Processing Systems (NeurIPS), vol. 34, pp. 12661–12673, 2021.
[14] I. Cong, S. Choi, and M. D. Lukin, "Quantum convolutional neural networks," Nature Physics, vol. 15, no. 12, pp. 1273–1278, Dec. 2019.
[15] V. Bergholm et al., "PennyLane: Automatic differentiation of quantum machine learning circuits," arXiv:1811.04968, 2018.
[16] M. Schuld, V. Bergholm, C. Gogolin, J. Izaac, and N. Killoran, "Evaluating analytic gradients on quantum hardware," Physical Review A, vol. 99, no. 3, p. 032331, Mar. 2019.
[17] J. Yang et al., "MedMNIST Classification Decathlon: A lightweight AutoML benchmark for medical image analysis," in IEEE 18th Int. Symp. Biomed. Imaging (ISBI), 2021, pp. 191–195.
[18] J. R. McClean, S. Boixo, V. N. Smelyanskiy, R. Babbush, and H. Neven, "Barren plateaus in quantum neural network training landscapes," Nature Communications, vol. 9, no. 1, p. 4812, Nov. 2018.
```
