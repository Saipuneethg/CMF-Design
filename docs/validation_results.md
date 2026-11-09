# Spectral Metameric Fiber Analyzer — Numerical Validation Results

This document records the empirical results, numerical benchmarks, and convergence diagnostics obtained from the validated research pipeline.

---

## 1. Reproducible Four-Target Evaluation Benchmark

Run script: `evaluation/run_fiber_validation.py`  
Artifacts generated: `artifacts/fiber_validation_results.json`, `artifacts/fiber_validation_results.csv`

### Evaluation Summary Table

| Target Name | Target RGB | Gamut Feasible | Fiber Dim $d_f$ | Camera Rank | Anchor Residual $\|A z_0 - b\|_2$ | Max Polytope Violation | Valid Samples (%) | Reconstructed RGB RMS Error | Alpha Spread Score $\sigma_{\text{spread}}$ | Volume Scale $\mathcal{A}_{\det}$ | Total Runtime |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Known Feasible Neutral** | `[0.5, 0.5, 0.5]` | **True** | 5 | 3 | $1.90 \times 10^{-15}$ | $-1.32 \times 10^{-5}$ | 1000 / 1000 (100.0%) | $2.70 \times 10^{-15}$ | 0.0742 | 0.0487 | 139.97 s |
| **Interior Feasible Chromatic** | `[0.3, 0.35, 0.4]` | **True** | 5 | 3 | $1.57 \times 10^{-15}$ | $-1.44 \times 10^{-5}$ | 1000 / 1000 (100.0%) | $2.28 \times 10^{-15}$ | 0.0518 | 0.0330 | 157.09 s |
| **Near-Boundary Low Luminance** | `[0.18, 0.18, 0.18]` | **True** | 5 | 3 | $1.35 \times 10^{-15}$ | $-3.97 \times 10^{-6}$ | 1000 / 1000 (100.0%) | $1.79 \times 10^{-15}$ | 0.0287 | 0.0184 | 143.76 s |
| **Outside-Gamut Saturated Green** | `[0.05, 0.95, 0.05]` | **False** | — | — | — | — | 0 (Rejected) | — | — | — | 7.28 s |

### Key Observations:
1. **Machine-Precision RGB Equality:** For all feasible targets, reconstructed RGB error $\|M R - c\|_2$ and anchor residuals are on the order of $10^{-15}$, confirming exact algebraic consistency.
2. **Strict Polytope Interiority:** Maximum constraint violations are strictly negative (slack $> 0$), demonstrating that zero invalid points were admitted into the representation.
3. **Boundary Contraction of Ambiguity:** As target luminance approaches the boundary (`[0.18, 0.18, 0.18]`), the volume scale $\mathcal{A}_{\det}$ drops from 0.0487 down to 0.0184, quantitatively confirming that near-boundary colors suffer less spectral ambiguity than interior neutral colors.
4. **Clean Out-of-Gamut Rejection:** The outside-gamut target is decisively rejected in 7.28 seconds with zero stale points rendered.

---

## 2. Multi-Chain Agreement Diagnostics

Two independent Markov chains ($N=500$ each, thinning=2, burn-in=50) with separate seeds:

| Target | Centroid Distance $\|\bar{z}_1 - \bar{z}_2\|_2$ | Relative Covariance Difference $\|\Sigma_1 - \Sigma_2\|_F / \|\Sigma_1\|_F$ | Subspace Projection Distance $\|P_1 - P_2\|_F / \sqrt{2d}$ |
| :--- | :---: | :---: | :---: |
| **Known Feasible Neutral** | 0.0173 | 0.231 | 0.3104 |
| **Interior Feasible Chromatic** | 0.0129 | 0.208 | 0.2770 |
| **Near-Boundary Low Luminance** | 0.0031 | 0.056 | 0.0613 |

*Note on Subspace Distance:* The subspace projection metric measures the difference between orthogonal projectors $P = U U^T$ onto the top principal ambiguity directions, guaranteeing strict invariance to eigenvector sign ambiguities and rotational degeneracies of near-repeated eigenvalues.

---

## 3. Hit-and-Run Sample-Count Stability & Convergence

Run script: `evaluation/run_sampling_stability.py`  
Target: RGB `[0.5, 0.5, 0.5]`, Ground-Truth LP Bounds at 550 nm: `[0.0263, 0.0664]`

| Sample Count $N$ | Centroid Drift vs Reference | Covariance Relative Error | Spread Score $\sigma_{\text{spread}}$ | Volume Scale $\mathcal{A}_{\det}$ | Empirical 550 nm Extrema | Gap to Exact Bounds (Min / Max) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **100** | $3.56 \times 10^{-2}$ | 0.3603 | 0.0641 | 0.0472 | [0.0291, 0.0578] | 0.0028 / 0.0086 |
| **250** | $3.32 \times 10^{-2}$ | 0.3998 | 0.0703 | 0.0482 | [0.0287, 0.0587] | 0.0024 / 0.0077 |
| **500** | $2.20 \times 10^{-2}$ | 0.2235 | 0.0666 | 0.0480 | [0.0286, 0.0599] | 0.0022 / 0.0065 |
| **1,000** | $6.30 \times 10^{-3}$ | 0.1886 | 0.0661 | 0.0475 | [0.0286, 0.0610] | 0.0022 / 0.0054 |
| **2,000 (Ref)** | $0.00 \times 10^{0}$ | 0.0000 | 0.0624 | 0.0470 | [0.0286, 0.0639] | 0.0022 / 0.0025 |

### Convergence Findings:
- Centroid drift decreases monotonically from $3.56 \times 10^{-2}$ at $N=100$ to $6.30 \times 10^{-3}$ at $N=1000$.
- Volume scale estimate $\mathcal{A}_{\det}$ stabilizes rapidly around $0.0470 \pm 0.001$.
- As sample count grows, the empirical sample maximum at 550 nm expands outward toward the exact physical bound (gap drops from 0.0086 to 0.0025).

---

## 4. Exact LP Bounds vs Empirical Sample Extrema

Evaluated on Known Feasible Neutral (`RGB = [0.5, 0.5, 0.5]`):

| Wavelength | Exact LP Min | Exact LP Max | LP Span | Sample Min ($N=1000$) | Sample Max ($N=1000$) | Sample Median ($p_{50}$) | 5th Percentile | 95th Percentile |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **400 nm** | 0.0061 | 0.0476 | 0.0415 | 0.0094 | 0.0435 | 0.0249 | 0.0142 | 0.0381 |
| **450 nm** | 0.0083 | 0.0634 | 0.0551 | 0.0118 | 0.0589 | 0.0336 | 0.0189 | 0.0514 |
| **500 nm** | 0.0175 | 0.0577 | 0.0402 | 0.0201 | 0.0543 | 0.0357 | 0.0247 | 0.0482 |
| **550 nm** | 0.0263 | 0.0664 | 0.0401 | 0.0286 | 0.0628 | 0.0441 | 0.0331 | 0.0567 |
| **600 nm** | 0.0279 | 0.0638 | 0.0359 | 0.0304 | 0.0607 | 0.0442 | 0.0346 | 0.0552 |
| **650 nm** | 0.0253 | 0.0818 | 0.0565 | 0.0298 | 0.0766 | 0.0514 | 0.0369 | 0.0691 |
| **700 nm** | 0.0224 | 0.0934 | 0.0710 | 0.0284 | 0.0872 | 0.0560 | 0.0378 | 0.0784 |

*Important Geometric Distinction:* The exact LP bounds enclose all empirical sample extrema ($R_{\min} \le R_{\text{sample}} \le R_{\max}$ everywhere). The percentile ribbons (e.g. 5th–95th) represent high-probability empirical density, not the absolute boundary of the continuous fiber.

---

## 5. PCA Dimensionality Sensitivity Analysis

Run script: `evaluation/run_pca_sensitivity.py`  
Tested on 6,208 surface spectra sampled from the 36K image dataset:

| PCA Dimension $K$ | Variance Explained | Spectral Recon RMSE | Camera Rank | Fiber Dim $d_f(K)$ | Facet Complexity Scaling | Polytope Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **5** | 98.98% | $2.16 \times 10^{-2}$ | 3 | **2** | $\mathcal{O}(V^2)$ (Tractable) | Requires Offline Rebuild |
| **8 (Current)** | 99.71% | $1.15 \times 10^{-2}$ | 3 | **5** | 714,340 facets (Computed) | **Active ($K=8$ Artifacts)** |
| **10** | 99.84% | $8.53 \times 10^{-3}$ | 3 | **7** | $\mathcal{O}(V^5)$ (Explosive) | Combinatorially Heavy |
| **12** | 99.90% | $6.79 \times 10^{-3}$ | 3 | **9** | $\mathcal{O}(V^6)$ (Explosive) | Combinatorially Heavy |
| **15** | 99.95% | $4.68 \times 10^{-3}$ | 3 | **12** | $\mathcal{O}(V^7)$ (Explosive) | Combinatorially Heavy |

### Research Conclusions on $K$:
- **$K=5$** under-represents metameric degrees of freedom ($d_f = 2$ instead of 5), losing critical spectral variation.
- **$K=8$** represents the computational "sweet spot": capturing $>99.7\%$ spectral energy with a mathematically tractable 8D convex hull (714,340 facets).
- **$K \ge 10$** increases fiber dimensionality to $7 \dots 12$, but exact facet enumeration grows exponentially as $\mathcal{O}(V^{\lfloor K/2 \rfloor})$, requiring millions of facets.

---

## 6. Synthetic Geometry Benchmark Validation

Tested on canonical 5D hypercube $[-1, 1]^5$ ($N=10,000$ points, seed 42):

| Metric | Analytical Ground Truth | Empirical Estimate | Relative Error | Validation Status |
| :--- | :---: | :---: | :---: | :---: |
| **Mean Coordinate Norm** | 0.0000 | 0.0094 | — | **Passed** |
| **Variance per Coordinate** | 0.3333 ($1/3$) | 0.3338 | **0.14%** | **Passed** |
| **Trace Spread Score $\sigma_{\text{spread}}$** | 0.57735 ($1/\sqrt{3}$) | 0.57777 | **0.07%** | **Passed** |
| **Volume Scale $\mathcal{A}_{\det}$** | 0.57735 ($1/\sqrt{3}$) | 0.57742 | **0.01%** | **Passed** |
| **Bounding Box Volume** | 32.0000 ($2^5$) | 31.8415 | **0.50%** | **Passed** |

All synthetic benchmark relative errors are $< 1\%$, validating the statistical spread score, generalized variance volume scale, and bounding geometry estimators.
