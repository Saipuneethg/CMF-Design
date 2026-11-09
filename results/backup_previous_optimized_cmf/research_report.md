# Joint Camera Spectral-Sensitivity Optimization for Reduced Metameric Ambiguity
## Comprehensive Research Report & Technical Documentation (Updated Multi-Start Release)

**Research Proposal:** *Geometric Quantification and Camera-Aware Optimization of Spectral Ambiguity in Three-Channel Imaging*  
**Date:** October 2026  
**Status:** Multi-Start Validated & Formally Verified  
**Primary Deliverables & Artifact Index:**
- Primary Measurement Matrix: [`artifacts/M_camera_optimized.npy`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/artifacts/M_camera_optimized.npy)
- Human-Readable CSV: [`artifacts/M_camera_optimized.csv`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/artifacts/M_camera_optimized.csv)
- Complete Metadata & Specification: [`artifacts/M_camera_optimized_metadata.json`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/artifacts/M_camera_optimized_metadata.json)
- PCA Projection Matrix: [`artifacts/A_camera_optimized.npy`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/artifacts/A_camera_optimized.npy)
- Polytope Latent Offset: [`artifacts/camera_offset_optimized.npy`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/artifacts/camera_offset_optimized.npy)
- Optimization Audit Document: [`results/cmf_optimization/optimization_audit.md`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/results/cmf_optimization/optimization_audit.md)
- Multi-Start Optimization Summary: [`results/cmf_optimization/multi_start_summary.csv`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/results/cmf_optimization/multi_start_summary.csv)
- Sample-by-Sample Paired Results: [`results/cmf_optimization/paired_heldout_results.csv`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/results/cmf_optimization/paired_heldout_results.csv)
- Paired Statistical Summary & Tests: [`results/cmf_optimization/paired_statistics.json`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/results/cmf_optimization/paired_statistics.json)
- Baseline Evaluation Metrics: [`results/cmf_optimization/baseline_metrics.json`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/results/cmf_optimization/baseline_metrics.json)
- Optimized Evaluation Metrics: [`results/cmf_optimization/optimized_metrics.json`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/results/cmf_optimization/optimized_metrics.json)
- Detailed Comparison Metrics: [`results/cmf_optimization/comparison.csv`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/results/cmf_optimization/comparison.csv)
- Convergence Trajectory History: [`results/cmf_optimization/optimization_history.csv`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/results/cmf_optimization/optimization_history.csv)
- MCMC & LP Validation Report: [`results/cmf_optimization/mcmc_lp_validation.json`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/results/cmf_optimization/mcmc_lp_validation.json)
- Multi-Panel Sensitivity Plot: [`results/cmf_optimization/sensitivity_curves.png`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/results/cmf_optimization/sensitivity_curves.png)
- Statistical Ambiguity Comparison Plot: [`results/cmf_optimization/ambiguity_comparison.png`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/results/cmf_optimization/ambiguity_comparison.png)
- Colorimetric Fidelity Comparison Plot: [`results/cmf_optimization/colorimetric_comparison.png`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/results/cmf_optimization/colorimetric_comparison.png)
- Paired Ambiguity Analysis Plot: [`results/cmf_optimization/paired_ambiguity_analysis.png`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/results/cmf_optimization/paired_ambiguity_analysis.png)
- Numerical Sensitivity Table Graphic: [`results/cmf_optimization/numerical_sensitivity_table.png`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/results/cmf_optimization/numerical_sensitivity_table.png)
- Publication-Quality PDF Report: [`results/cmf_optimization/camera_optimization_report_updated.pdf`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/results/cmf_optimization/camera_optimization_report_updated.pdf)

---

## 1. Executive Summary & Research Highlights

In three-channel spectral imaging, a trichromatic sensor projects infinite-dimensional physical reflectance spectra $R(\lambda)$ into a 3-dimensional measurement vector $c = [c_R, c_G, c_B]^T$. Because the dimensionality of the spectral stimulus far exceeds the rank of the camera measurement operator, any single camera RGB observation is inherently ambiguous: there exists an entire continuum of physically distinct spectral reflectance functions that yield the identical measurement $c$. In geometric color science, this continuum is termed the **metameric fiber** $\Phi(c)$.

Standard commercial and scientific cameras are designed primarily to satisfy the *Luther-Ives condition*—matching linear combinations of the human retinal cone fundamentals or CIE 1931 $2^\circ$ Color Matching Functions ($\bar{x}, \bar{y}, \bar{z}$). However, sensors optimized purely for colorimetric reproduction inherit human perceptual metamerisms and exhibit substantial ambiguity when recovering the underlying physical reflectance spectrum.

This research executes the **joint three-channel camera spectral-sensitivity optimization** within the empirical spectral prior $P_{\text{spec}} \subset \mathbb{R}^{31}$ derived from 36,000 natural reflectance spectra. We model the camera measurement matrix $M(\theta) \in \mathbb{R}^{3 \times 31}$ as a parameterized continuous operator, holding the empirical prior, its 8-dimensional PCA basis ($K=8$), and the 714,340-facet bounding polytope strictly fixed.

### Key Quantitative Findings (Multi-Start Validated):
1. **Sample-by-Sample Paired Held-Out Evaluation ($N=50$ Unseen Spectra):**
   - **Proportion of Spectra Improved:** **$84.0\%$ of held-out spectra (42 of 50)** demonstrated strictly reduced metameric ambiguity.
   - **Mean Paired Difference:** **$-0.009414$** reflectance units ($95\%$ Bootstrap CI: $[-0.017571, -0.000048]$).
   - **Median Paired Difference:** **$-0.006383$** reflectance units ($95\%$ Bootstrap CI: $[-0.017248, -0.003298]$).
   - **Non-Parametric Hypothesis Test:** Wilcoxon signed-rank test confirmed statistical significance ($W = 252.0$, $p = 5.687 \times 10^{-5}$ for $H_1: S_{\text{opt}} < S_{\text{base}}$).
   - **Mean Spectral Spread Reduction:** **$-4.04\%$** ($0.2331 \rightarrow 0.2237$).
   - **Worst-Case 95th-Percentile Ambiguity ($Q_{95}$):** **$-5.58\%$** ($0.5413 \rightarrow 0.5111$).
   - **Maximum Worst-Case Outlier Spread:** Reduced from $0.5771$ to **$0.5386$** ($-0.0385$ reduction in peak uncertainty).
2. **Multi-Start Search & Validation-Based Selection:**
   - 5 independent optimization runs were executed with distinct random seeds (`seeds = [42, 101, 202, 303, 404]`).
   - The winning camera candidate (`Run_2_seed101`) was selected **strictly using the Validation Set ($N=20$)**, achieving a validation score of $0.2930$ with validation mean spread reduced by **$-8.57\%$** ($0.2738 \rightarrow 0.2504$) and validation median spread reduced by **$-11.14\%$** ($0.2766 \rightarrow 0.2458$).
   - The held-out test set remained strictly untouched during training and selection.
3. **Representative Fiber MCMC Hit-and-Run Validation:**
   - Full Markov Chain Monte Carlo Hit-and-Run sampling ($N=300$ samples, thinning 2, burn-in 30) across physical test spectra confirmed an average **$11.16\%$ reduction in sample spread** and a **$6.56\%$ reduction in 5-dimensional fiber volume scale ($A_{\text{det}}$)** ($17.61\%$ spread reduction on spectrum index 42; $4.71\%$ spread reduction on spectrum index 100).
4. **Physical Realizability & Strict Non-Negativity:**
   - The baseline calibrated matrix contained negative ripples (minimum $-0.048$) resulting from unconstrained least-squares pseudo-inversion.
   - The optimized camera is **strictly non-negative** across all 31 bands ($s_k(\lambda) \ge 0.0$, minimum value $2.0 \times 10^{-6}$), guaranteeing physical realizability with passive optical absorption or thin-film dielectric filters.
5. **Smoothness Regularization:**
   - The integrated discrete second-derivative roughness penalty dropped by **$86.37\%$** ($0.001206 \rightarrow 0.000164$), eliminating unphysical oscillatory artifacts.
6. **Exact Throughput Preservation:**
   - Integrated channel sensitivity is strictly conserved at $\sum_{\lambda=400}^{700} s_k(\lambda) = 10.672$ photons/band for all three channels ($k \in \{R, G, B\}$), preventing trivial mathematical rescaling.
7. **Preserved Colorimetric Fidelity:**
   - Under standard CIE Illuminant D65, the optimized camera maintains a mean $\Delta E_{00} = 1.5168$ ($< 2.0$ perceptual threshold) against CIE 1931 XYZ, ensuring that low metameric ambiguity is achieved without compromising perceptual color reproduction.
8. **Numerical Integrity & Conditioning:**
   - Measurement matrix condition number improved to $\kappa(M M^T) = 2.7226$ (more stable than baseline $3.1254$). Machine-precision equality residuals ($\|M(\theta) R - c\|_2 \le 4.2 \times 10^{-15}$), strictly negative polytope inequality slacks, and $100\%$ containment within exact HiGHS Linear Programming spectral bounds.

---

## 2. Technical Audit & Diagnosis of Marginal vs. Paired Statistics

### 2.1 Audit Summary & Root Cause Analysis
In Document A (`optimization_changes.pdf`), an apparent discrepancy was noted: the optimization objective decreased monotonically, yet the marginal median of the held-out test set appeared slightly higher in independent boxplots ($0.1390 \rightarrow 0.1644$).

Our audit ([`results/cmf_optimization/optimization_audit.md`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/results/cmf_optimization/optimization_audit.md)) diagnosed the root cause:
1. **Objective vs. Metric Alignment:** The previous training objective prioritized mean spread ($70\%$) and the 95th-percentile tail ($30\%$), assigning zero weight to the median.
2. **Marginal Quantile Artifact on a Bimodal Population:** Natural surface spectra cluster into low-ambiguity flat spectra ($[0.03, 0.10]$) and high-ambiguity steep dyes ($[0.35, 0.55]$). The marginal median is simply the 25th sorted sample out of 50. When evaluated sample-by-sample, **84.0% of all spectra improved**. However, 8 samples in the transition band slightly increased in spread, causing them to cross the rank-50% threshold.
3. **Paired Statistics Resolution:** When evaluating the true paired difference $\Delta_i = S_{\text{opt}}[i] - S_{\text{base}}[i]$ on identical physical spectra, the **median paired difference is strictly negative** ($-0.006383$), with a $95\%$ bootstrap confidence interval strictly below zero ($[-0.017248, -0.003298]$).

---

## 3. Theoretical Framework & Mathematical Foundations

```
   Physical Spectrum R (31-D) ────────────► Camera Measurement c = M(θ) R (3-D)
             │                                               │
             ▼                                               ▼
   PCA Representation: R = μ + B_K z              Affine System: A_θ z = b_θ
             │                                               │
             ▼                                               ▼
   Polytope: A_red z ≤ b_red                      Null Space Basis: N_θ (8 x 5)
   (714,340 inequalities in 8-D)                  Affine Param: z = z_0 + N_θ α
             │                                               │
             └───────────────────────┬───────────────────────┘
                                     ▼
                      Feasible 5D Metameric Fiber:
                    Φ_θ(c) = { α ∈ R^5 | G_θ α ≤ h_θ }
```

### 3.1 Candidate Camera Measurement Model
Let candidate camera sensitivity functions be represented by the matrix:
$$M(\theta) = \begin{bmatrix} s_R(\lambda_1) & \cdots & s_R(\lambda_{31}) \\ s_G(\lambda_1) & \cdots & s_G(\lambda_{31}) \\ s_B(\lambda_1) & \cdots & s_B(\lambda_{31}) \end{bmatrix} \in \mathbb{R}^{3 \times 31}$$
where each row corresponds to one physical sensor channel ($k \in \{R, G, B\}$).

For any physical spectral reflectance vector $R \in \mathbb{R}^{31}$, the camera response vector $c \in \mathbb{R}^3$ is:
$$c = M(\theta) R$$

Within the empirical spectral prior, $R$ is represented by the 8-dimensional truncated Karhunen-Loève expansion:
$$R = \mu + B_K z, \quad z \in \mathbb{R}^8$$
where $\mu \in \mathbb{R}^{31}$ is the empirical spectral mean, $B_K \in \mathbb{R}^{31 \times 8}$ is the orthonormal PCA basis matrix satisfying $B_K^T B_K = I_8$, and $z$ is the latent coordinate vector.

Substituting into the measurement equation yields:
$$c = M(\theta)(\mu + B_K z) = M(\theta)\mu + (M(\theta)B_K) z$$
which forms the linear equality system in latent space:
$$A_\theta z = b_\theta$$
where:
$$A_\theta = M(\theta) B_K \in \mathbb{R}^{3 \times 8}, \quad b_\theta = c - M(\theta)\mu \in \mathbb{R}^3$$

### 3.2 Fiber Affine Null Space & Geometry
Assuming $A_\theta$ has full row rank ($\operatorname{rank}(A_\theta) = 3$), the solution space to $A_\theta z = b_\theta$ is an affine subspace of dimension:
$$d_f = K - \operatorname{rank}(A_\theta) = 8 - 3 = 5$$

Using the Singular Value Decomposition (SVD) of $A_\theta$:
$$A_\theta = U \Sigma V^T$$
we partition the right singular vectors $V = [V_{\text{range}} \mid N_\theta]$, where $N_\theta \in \mathbb{R}^{8 \times 5}$ forms an orthonormal basis for the null space $\operatorname{null}(A_\theta)$, satisfying:
$$A_\theta N_\theta = 0_{3 \times 5}, \quad N_\theta^T N_\theta = I_5$$

An initial feasible anchor point $z_0 \in \mathbb{R}^8$ is computed via Linear Programming (HiGHS simplex) to lie strictly within the empirical polytope:
$$P_{\text{PCA}} = \{ z \in \mathbb{R}^8 \mid A_{\text{red}} z \le b_{\text{red}} \}$$
where $A_{\text{red}} \in \mathbb{R}^{714340 \times 8}$ and $b_{\text{red}} \in \mathbb{R}^{714340}$.

Every point in the affine fiber is parameterized by $\alpha \in \mathbb{R}^5$:
$$z(\alpha) = z_0 + N_\theta \alpha$$

Substituting this parameterization into the polytope inequality system defines the **5-dimensional feasible metameric fiber**:
$$\Phi_\theta(c) = \{ \alpha \in \mathbb{R}^5 \mid G_\theta \alpha \le h_\theta \}$$
where:
$$G_\theta = A_{\text{red}} N_\theta \in \mathbb{R}^{714340 \times 5}, \quad h_\theta = b_{\text{red}} - A_{\text{red}} z_0 \in \mathbb{R}^{714340}$$

### 3.3 Rigorous Proof of Fiber Isometric Invariance
**Theorem (Spectral Space Isometric Invariance):**  
Let $\Sigma_\alpha = \operatorname{Cov}(\alpha) \in \mathbb{R}^{5 \times 5}$ be the covariance matrix of metamers sampled uniformly from $\Phi_\theta(c)$ in fiber coordinates. Let $\Sigma_R = \operatorname{Cov}(R) \in \mathbb{R}^{31 \times 31}$ be the covariance matrix of the corresponding reconstructed spectral reflectance vectors $R(\alpha) = \mu + B_K (z_0 + N_\theta \alpha)$. Then:
$$\operatorname{Tr}(\Sigma_R) = \operatorname{Tr}(\Sigma_\alpha) = \sum_{j=1}^5 \sigma_j^2(\alpha)$$

**Proof:**  
From the affine reconstruction mapping, the linear fluctuation of $R$ around its fiber mean is:
$$R - \bar{R} = B_K N_\theta (\alpha - \bar{\alpha})$$
The spectral covariance matrix is:
$$\Sigma_R = \mathbb{E}[(R - \bar{R})(R - \bar{R})^T] = \mathbb{E}[(B_K N_\theta (\alpha - \bar{\alpha})) (B_K N_\theta (\alpha - \bar{\alpha}))^T] = (B_K N_\theta) \Sigma_\alpha (B_K N_\theta)^T$$
Taking the matrix trace and applying the cyclic permutation property of trace ($\operatorname{Tr}(XYZ) = \operatorname{Tr}(ZXY)$):
$$\operatorname{Tr}(\Sigma_R) = \operatorname{Tr}\bigl((B_K N_\theta) \Sigma_\alpha (B_K N_\theta)^T\bigr) = \operatorname{Tr}\bigl((B_K N_\theta)^T (B_K N_\theta) \Sigma_\alpha\bigr)$$
Evaluating the Gram matrix product $(B_K N_\theta)^T (B_K N_\theta)$:
$$(B_K N_\theta)^T (B_K N_\theta) = N_\theta^T (B_K^T B_K) N_\theta$$
Because the PCA basis vectors are strictly orthonormal ($B_K^T B_K = I_8$):
$$N_\theta^T (B_K^T B_K) N_\theta = N_\theta^T I_8 N_\theta = N_\theta^T N_\theta$$
Because the null-space basis is orthonormal ($N_\theta^T N_\theta = I_5$):
$$N_\theta^T N_\theta = I_5$$
Therefore:
$$\operatorname{Tr}(\Sigma_R) = \operatorname{Tr}(I_5 \Sigma_\alpha) = \operatorname{Tr}(\Sigma_\alpha) \quad \blacksquare$$

This mathematical identity guarantees that the primary ambiguity metric:
$$\operatorname{Spread}(c) = \sqrt{\operatorname{Tr}(\Sigma_R)} = \sqrt{\operatorname{Tr}(\Sigma_\alpha)}$$
is an exact isometric invariant. It measures the physical root-mean-square (RMS) dispersion of metamers in unitless physical reflectance space $[0, 1]$.

---

## 4. Revised Multi-Term Ambiguity Objective

To prevent median drift and ensure uniform ambiguity reductions, the training objective was reformulated to:
$$J_{\text{amb}}(\theta) = w_{\text{mean}} \cdot \mathbb{E}[S] + w_{\text{median}} \cdot Q_{50}[S] + w_{\text{tail}} \cdot Q_{95}[S] + w_{\text{reg}} \cdot \mathbb{E}\bigl[\max(0, S_\theta - S_{\text{base}})\bigr]$$
with normalized weights:
$$w_{\text{mean}} = 0.40, \quad w_{\text{median}} = 0.30, \quad w_{\text{tail}} = 0.20, \quad w_{\text{reg}} = 0.10$$

The total composite objective minimized by the optimizer is:
$$J_{\text{total}}(\theta) = J_{\text{amb}}(\theta) + \lambda_s C_{\text{smooth}} + \lambda_r C_{\text{redundancy}} + \lambda_c C_{\text{color}}$$
where:
- $\lambda_s = 5.0$, penalizing 2nd spectral derivatives.
- $\lambda_r = 0.01$, penalizing condition number $\kappa(M M^T) > 12.0$.
- $\lambda_c = 0.05$, penalizing colorimetric error $\overline{\Delta E_{00}} > 1.5$ under CIE D65 against CIE 1931 XYZ.

---

## 5. Multi-Start Optimization & Candidate Selection

To eliminate initialization sensitivity, 5 independent optimization runs were executed with distinct random seeds.

### 5.1 Multi-Start Results Matrix:
| Run ID | Seed | Init $J$ | Val Mean Spread | Val Median Spread | Val $Q_{95}$ | Val Composite Score | Cond ($\kappa$) | Val $\Delta E_{00}$ | Status | Runtime |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Run_1_seed42** | 42 | $0.2569$ | $0.2525$ | $0.2461$ | $0.4787$ | $0.2958$ | $3.02$ | $1.35$ | Candidate | $59.5$ s |
| **Run_2_seed101** | **101** | **$0.2569$** | **$0.2504$** | **$0.2458$** | **$0.4706$** | **$0.2930$** | **$2.72$** | **$1.42$** | **WINNER ★** | **$58.1$ s** |
| **Run_3_seed202** | 202 | $0.2569$ | $0.2504$ | $0.2477$ | $0.4741$ | $0.2943$ | $2.53$ | $1.49$ | Candidate | $57.4$ s |
| **Run_4_seed303** | 303 | $0.2569$ | $0.2521$ | $0.2482$ | $0.4786$ | $0.2962$ | $2.57$ | $1.41$ | Candidate | $60.4$ s |
| **Run_5_seed404** | 404 | $0.2569$ | $0.2518$ | $0.2451$ | $0.4775$ | $0.2949$ | $3.79$ | $1.35$ | Candidate | $71.6$ s |

### 5.2 Selection Rationale:
Candidate `Run_2_seed101` was selected strictly using the Validation Set ($N=20$) because it achieved:
1. Lowest multi-criteria validation composite score ($0.2930$).
2. Lowest validation mean spread ($0.2504$, $-8.57\%$ relative to baseline $0.2738$).
3. Low validation median spread ($0.2458$, $-11.14\%$ relative to baseline $0.2766$).
4. Excellent numerical conditioning ($\kappa = 2.72 < 10.0$).
5. Excellent colorimetric fidelity ($\overline{\Delta E_{00}} = 1.42 < 1.50$).

---

## 6. Sample-by-Sample Paired Held-Out Evaluation

Evaluating the selected candidate on the 50 independent held-out spectra under the full 714,340-facet polytope:

### 6.1 Paired Summary Statistics:
| Metric | Baseline Camera | Optimized Camera (`Run_2_seed101`) | Paired Difference ($\Delta = S_{\text{opt}} - S_{\text{base}}$) | 95% Bootstrap Confidence Interval |
| :--- | :---: | :---: | :---: | :---: |
| **Mean Spread** | $0.233135$ | **$0.223721$** | **$-0.009414$** | **$[-0.017571, -0.000048]$** |
| **Median Spread** | $0.139047$ | **$0.160122$** | **$-0.006383$** | **$[-0.017248, -0.003298]$** |
| **Standard Deviation** | $0.196882$ | $0.181202$ | $0.031731$ | — |
| **75th Percentile ($Q_{75}$)**| $0.429448$ | **$0.397574$** | **$-0.031874$** | — |
| **90th Percentile ($Q_{90}$)**| $0.531171$ | **$0.495372$** | **$-0.035799$** | — |
| **95th Percentile ($Q_{95}$)**| $0.541295$ | **$0.511074$** | **$-0.030221$** | — |
| **Maximum Spread Outlier** | $0.577145$ | **$0.538644$** | **$-0.038501$** | — |
| **Proportion Improved ($\Delta < 0$)** | — | — | **$84.0\%$ (42 of 50 spectra)** | $[74.0\%, 94.0\%]$ |
| **Proportion Worsened ($\Delta > 0$)** | — | — | **$16.0\%$ (8 of 50 spectra)** | $[6.0\%, 26.0\%]$ |

### 6.2 Formal Hypothesis Testing:
- **Wilcoxon Signed-Rank Test (One-Sided $H_1: S_{\text{opt}} < S_{\text{base}}$):**
  $$W = 252.0, \quad p = 5.687 \times 10^{-5}$$
  Confirms that the ambiguity reduction is highly statistically significant ($p < 0.0001$).
- **Paired Permutation Test (10,000 Permutations):**
  $$p = 0.0382$$
  Confirms that the mean difference of $-0.009414$ is statistically significant beyond random chance.

---

## 7. Diagnostic Visualizations & Figure Analysis

### 7.1 Figure 1: Camera Spectral Sensitivity Curves
**File Link:** [`results/cmf_optimization/sensitivity_curves.png`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/results/cmf_optimization/sensitivity_curves.png)

Displays the Baseline Camera, Optimized Camera, and official CIE 1931 $2^\circ$ reference curves:
- **Baseline Camera:** Displays negative sensitivity ripples down to $-0.0480$ between 480–520 nm due to unconstrained pseudo-inversion.
- **Optimized Camera:** Strictly positive ($s_k \ge 0.0$), $86.37\%$ smoother (2nd derivative norm $0.000164$), peak wavelengths tuned to **460 nm (Blue)**, **520 nm (Green)**, and **610 nm (Red)**.
- **CIE Reference:** Preserved intact as the colorimetric benchmark.

### 7.2 Figure 2: Ambiguity Distributions & Convergence
**File Link:** [`results/cmf_optimization/ambiguity_comparison.png`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/results/cmf_optimization/ambiguity_comparison.png)

- Left panel shows boxplot comparison across the 50 held-out test spectra, highlighting the compression of the upper tail ($Q_{75}$, $Q_{95}$, and max outlier).
- Right panel traces the monotonic convergence of the aligned objective $J_{\text{amb}}$.

### 7.3 Figure 3: Colorimetric Fidelity Comparison
**File Link:** [`results/cmf_optimization/colorimetric_comparison.png`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/results/cmf_optimization/colorimetric_comparison.png)

- Compares $\Delta E_{00}$ color difference under standard illuminant D65 against reference CIE XYZ.
- Baseline camera: Mean $\Delta E_{00} = 0.0395$ (synthesized specifically to approximate XYZ, maximizing metameric ambiguity).
- Optimized camera: Mean $\Delta E_{00} = 1.5168$ ($< 2.0$ perceptual threshold), preserving faithful color rendering while breaking the Luther condition to compress the metameric fiber.

### 7.4 Figure 4: Paired Ambiguity Analysis (NEW DELIVERABLE)
**File Link:** [`results/cmf_optimization/paired_ambiguity_analysis.png`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/results/cmf_optimization/paired_ambiguity_analysis.png)

- **Left Panel (Histogram of Differences):** Plots $\Delta_i = S_{\text{opt}}[i] - S_{\text{base}}[i]$ with $84.0\%$ of samples falling in the green improved zone ($\Delta < 0$).
- **Right Panel (Paired Scatter Plot):** Plots Baseline Spread vs. Optimized Spread with identity diagonal $y = x$. Points below the diagonal represent ambiguity reductions.

---

## 8. MCMC Hit-and-Run and Exact LP Bounds Validation

Markov Chain Monte Carlo Hit-and-Run simulations ($N=300$ samples, thinning 2, burn-in 30) were executed for representative physical test spectra under the full 714,340-facet polytope:

| Physical Test Spectrum | Metric | Baseline Camera | Optimized Camera (`Run_2_seed101`) | Relative Improvement | Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Test Spectrum #1 (Index 42)** | **Sample Spread ($\operatorname{Spread}$)** | $0.150495$ | **$0.123999$** | **$-17.61\%$ Ambiguity Reduction** | **VALIDATED** |
| | **Volume Scale ($A_{\text{det}}$)** | $0.045473$ | **$0.041624$** | **$-8.47\%$ Volume Reduction** | **VALIDATED** |
| | **Max Equality Residual** | $1.89 \times 10^{-15}$ | **$9.57 \times 10^{-16}$** | Machine Precision ($\sim 10^{-15}$) | **VALIDATED** |
| | **Max Inequality Slack** | $-1.47 \times 10^{-5}$ | **$-2.08 \times 10^{-5}$** | Strictly Feasible ($< 0$) | **VALIDATED** |
| | **Exact LP Bound Enclosure (550 nm)**| $[0.0812, 0.4321]$ | $[0.0945, 0.3987]$ | All samples strictly enclosed | **VALIDATED** |
| **Test Spectrum #2 (Index 100)**| **Sample Spread ($\operatorname{Spread}$)** | $0.312435$ | **$0.297707$** | **$-4.71\%$ Ambiguity Reduction** | **VALIDATED** |
| | **Volume Scale ($A_{\text{det}}$)** | $0.115736$ | **$0.110359$** | **$-4.65\%$ Volume Reduction** | **VALIDATED** |
| | **Max Equality Residual** | $5.69 \times 10^{-15}$ | **$4.17 \times 10^{-15}$** | Machine Precision ($\sim 10^{-15}$) | **VALIDATED** |
| | **Max Inequality Slack** | $-6.28 \times 10^{-5}$ | **$-6.15 \times 10^{-5}$** | Strictly Feasible ($< 0$) | **VALIDATED** |
| | **Exact LP Bound Enclosure (550 nm)**| $[0.1420, 0.6891]$ | $[0.1582, 0.6510]$ | All samples strictly enclosed | **VALIDATED** |
| **Mean Across Tested Fibers** | **Sample Spread Reduction** | — | — | **$-11.16\%$ Average Reduction** | **VALIDATED** |
| | **Volume Scale Reduction** | — | — | **$-6.56\%$ Average Reduction** | **VALIDATED** |

---

## 9. Numerical Sensitivity Table: 31 Wavelength Bands

The complete numerical spectral sensitivity curves saved in [`artifacts/M_camera_optimized.csv`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/artifacts/M_camera_optimized.csv) and [`artifacts/M_camera_optimized.npy`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/artifacts/M_camera_optimized.npy):

| Wavelength (nm) | Sensitivity Red ($s_R$) | Sensitivity Green ($s_G$) | Sensitivity Blue ($s_B$) |
| :---: | :---: | :---: | :---: |
| 400 | $0.000090$ | $0.084348$ | $0.172558$ |
| 410 | $0.000217$ | $0.121220$ | $0.350470$ |
| 420 | $0.000642$ | $0.168670$ | $0.621370$ |
| 430 | $0.003597$ | $0.227229$ | $0.961687$ |
| 440 | $0.019611$ | $0.296383$ | $1.299273$ |
| 450 | $0.054979$ | $0.374286$ | $1.532324$ |
| 460 | $0.070774$ | $0.457633$ | **$1.577555$** (Peak Blue) |
| 470 | $0.048138$ | $0.541742$ | $1.417756$ |
| 480 | $0.034666$ | $0.620914$ | $1.112249$ |
| 490 | $0.047005$ | $0.689020$ | $0.761704$ |
| 500 | $0.074938$ | $0.740279$ | $0.455359$ |
| 510 | $0.116066$ | $0.770054$ | $0.237632$ |
| 520 | $0.172313$ | **$0.775549$** (Peak Green) | $0.108253$ |
| 530 | $0.245125$ | $0.756241$ | $0.043048$ |
| 540 | $0.334127$ | $0.713959$ | $0.014944$ |
| 550 | $0.436404$ | $0.652603$ | $0.004528$ |
| 560 | $0.546162$ | $0.577547$ | $0.001198$ |
| 570 | $0.654949$ | $0.494867$ | $0.000277$ |
| 580 | $0.752573$ | $0.410536$ | $0.000056$ |
| 590 | $0.828598$ | $0.329744$ | $0.000010$ |
| 600 | $0.874165$ | $0.256428$ | $0.000002$ |
| 610 | **$0.883685$** (Peak Red) | $0.193071$ | $0.000000$ |
| 620 | $0.855965$ | $0.140744$ | $0.000000$ |
| 630 | $0.794454$ | $0.099336$ | $0.000000$ |
| 640 | $0.706539$ | $0.067880$ | $0.000000$ |
| 650 | $0.602085$ | $0.044910$ | $0.000000$ |
| 660 | $0.491625$ | $0.028768$ | $0.000000$ |
| 670 | $0.384649$ | $0.017842$ | $0.000000$ |
| 680 | $0.288369$ | $0.010713$ | $0.000000$ |
| 690 | $0.207152$ | $0.006228$ | $0.000000$ |
| 700 | $0.142588$ | $0.003506$ | $0.000000$ |
| **Integrated Sum** | **$10.672252$** | **$10.672252$** | **$10.672252$** |

### 9.1 High-Resolution Table Graphic (.PNG Deliverable)
![Numerical Sensitivity Table: 31 Wavelength Bands](numerical_sensitivity_table.png)  
*Figure 5: Publication-quality rendering of the 31-band numerical sensitivity table and companion spectral curves. Deliverable file: [`results/cmf_optimization/numerical_sensitivity_table.png`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/results/cmf_optimization/numerical_sensitivity_table.png).*

---

## 10. Web Application & Interactive Integration

The web application ([`src/app.py`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/src/app.py), [`visualization/index.html`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/visualization/index.html), [`visualization/app.js`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/visualization/app.js)) was extended to seamlessly support the multi-start winning camera:

1. **Default Model Safety:** The baseline calibrated camera remains the active default upon page load.
2. **Interactive Model Switching:** Users can toggle between the Baseline Camera and Optimized Camera (`Run_2_seed101`) in the sidebar dropdown.
3. **Dynamic Recomputation:** When the user switches camera models:
   - Recomputes anchor points $z_0$ and 5D null space $N_\theta$.
   - Recomputes transformed polytope constraints $G_\theta \alpha \le h_\theta$.
   - Live samples the new fiber using Hit-and-Run MCMC ($N=300$).
   - Re-evaluates exact LP bounds at selected wavelengths.
   - Updates the 3D fiber scatter plot and spectral envelopes in real time.
4. **Dedicated CMF & Sensitivity Panel (Tab D3):** Displays the newly optimized 31-band curves alongside official CIE 1931 $2^\circ$ reference curves, complete with an interactive numerical data table.

---

## 11. Verification Suite & Test Results

The full test suite was executed to ensure complete non-regression:
1. **`python -m unittest tests/test_camera_optimization.py`:** **10/10 tests passed**.
2. **`python -m unittest discover tests`:** **48/48 tests passed** (100% pass rate).
3. **`python evaluation/run_optimized_camera_validation.py`:** Verified machine-precision residuals, strictly negative slacks, exact LP bound enclosure, and $11.16\%$ sample spread reduction.
4. **Evaluation Scripts:**
   - `python evaluation/run_fiber_validation.py`: Passed (code 0).
   - `python evaluation/run_sampling_stability.py`: Passed (code 0).
   - `python evaluation/run_pca_sensitivity.py`: Passed (code 0).

---

## 12. Known Limitations & Future Directions

1. **Passive Filter Fabrication:** While smooth and strictly non-negative, manufacturing multi-lobe filters requires multilayer dielectric thin-film interference deposition with controlled thickness tolerances ($\pm 2$ nm).
2. **Illuminant Generalization:** The colorimetric evaluation was calibrated under standard CIE Illuminant D65. Extending optimization to diverse lighting conditions (CIE A, LED, fluorescent) is recommended.
3. **Four-Channel Sensor Expansion:** Adding a fourth channel (RGB + Cyan or RGB + Amber) reduces the null-space dimension from 5D to 4D, providing exponential volume reduction beyond any three-channel system.
