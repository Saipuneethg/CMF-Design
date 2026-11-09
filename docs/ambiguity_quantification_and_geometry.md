# Technical Documentation: Spectral Metameric Fiber Ambiguity Quantification, Geometry, and Validation

## 1. Mathematical Spectral Model

The continuous spectral reflectance $R \in \mathbb{R}^{31}$ represents surface reflectance sampled at 31 discrete wavelengths covering $\lambda \in [400, 700]\text{ nm}$ at $10\text{ nm}$ intervals.
The prior distribution of physical reflectances is learned from approximately 36,000 spectral scenes and represented via Principal Component Analysis (PCA) with $K = 8$ components:

$$R = \mu + B_K z$$

where:
- $\mu \in \mathbb{R}^{31}$ is the empirical spectral prior mean (`reflectance (0–1)`).
- $B_K \in \mathbb{R}^{31 \times K}$ has orthonormal columns ($B_K^T B_K = I_K$), representing the dominant spectral variation modes.
- $z \in \mathbb{R}^K$ are the latent PCA coordinates.

The feasible spectral prior polytope $P_{\mathrm{spec}} \subset \mathbb{R}^{31}$ is represented in PCA coordinates as the bounded system:

$$P_{\mathrm{spec}} = \{z \in \mathbb{R}^K : A_{\mathrm{red}} z \le b_{\mathrm{red}}\}$$

with $A_{\mathrm{red}} \in \mathbb{R}^{714340 \times 8}$ and $b_{\mathrm{red}} \in \mathbb{R}^{714340}$.

---

## 2. RGB-to-Measurement Conversion

Given a target camera observation $c = (R_{\mathrm{cam}}, G_{\mathrm{cam}}, B_{\mathrm{cam}})^T \in [0, 1]^3$ and spectral camera sensitivity matrix $M \in \mathbb{R}^{3 \times 31}$:

$$c = M R = M (\mu + B_K z) = M\mu + (M B_K) z$$

Defining the linear equality system:

$$A z = b$$

where:
- $A = M B_K \in \mathbb{R}^{3 \times K}$
- $b = c - M\mu \in \mathbb{R}^3$

---

## 3. Metameric Fiber Formulation

The metameric fiber $\Phi(c)$ is the set of all spectra $R \in P_{\mathrm{spec}}$ satisfying $M R = c$.
By singular value decomposition of $A$, the null-space basis matrix $N \in \mathbb{R}^{K \times d_f}$ satisfies $A N = 0$ with orthonormal columns ($N^T N = I_{d_f}$).
For any particular feasible anchor point $z_0 \in \mathbb{R}^K$ satisfying $A z_0 = b$, all solutions to the equality constraint can be parameterized by reduced coordinates $\alpha \in \mathbb{R}^{d_f}$:

$$z(\alpha) = z_0 + N \alpha$$

Substituting into the prior inequalities $A_{\mathrm{red}} z \le b_{\mathrm{red}}$ yields the exact reduced-dimensional fiber polytope:

$$P_\alpha = \{\alpha \in \mathbb{R}^{d_f} : G \alpha \le h\}$$

where:
- $G = A_{\mathrm{red}} N \in \mathbb{R}^{714340 \times d_f}$
- $h = b_{\mathrm{red}} - A_{\mathrm{red}} z_0 \in \mathbb{R}^{714340}$

Crucially, because both $B_K$ and $N$ have orthonormal columns:

$$(B_K N)^T (B_K N) = N^T (B_K^T B_K) N = N^T I_K N = I_{d_f}$$

Therefore, **distances and volumes in $\alpha$-space equal Euclidean distances and volumes in 31-band spectral reflectance space** ($(\text{reflectance})^{d_f}$).

---

## 4. Distinction Between Affine and Feasible Fiber Dimensions

1. **PCA Component Count ($K$):** Number of retained latent basis components ($K = 8$ `components`).
2. **Constraint Rank ($\operatorname{rank}(A)$):** Numerical rank of the camera measurement matrix ($\operatorname{rank}(A) = 3$ `rank`).
3. **Affine Fiber Dimension ($d_{\mathrm{affine}}$):** Dimension of the affine solution space of equality constraints:
   $$d_{\mathrm{affine}} = K - \operatorname{rank}(A) = 8 - 3 = 5\text{ dimensions}$$
4. **Feasible Fiber Dimension ($d_{\mathrm{feasible}}$):** Topological dimension of the compact intersection $P_\alpha = \{\alpha : G\alpha \le h\}$. When the Chebyshev inradius $r_{\mathrm{in}} > 10^{-9}$, $P_\alpha$ contains an open ball in $\mathbb{R}^{d_{\mathrm{affine}}}$, proving $d_{\mathrm{feasible}} = d_{\mathrm{affine}} = 5\text{ dimensions}$.
5. **Sampling Dimension:** Dimension of the reduced parameter space used by Hit-and-Run ($5\text{ dimensions}$).

---

## 5. Ambiguity Metrics and Mathematical Formulas

### A. Spectral RMS Spread Score
Measures the root-mean-square distance of sampled metamers from the sample mean:

$$S_{\mathrm{spread}} = \sqrt{\frac{1}{N}\sum_{i=1}^N \|R_i - \bar{R}\|_2^2} = \sqrt{\frac{N-1}{N}\operatorname{Tr}(\Sigma_R)}$$

- **Unit:** `reflectance (dimensionless)`
- **Kind:** Sample estimate

### B. Alpha-Space Coordinate Spread
Coordinate-normalized standard deviation:

$$\sigma_{\mathrm{spread}} = \sqrt{\frac{\operatorname{Tr}(\Sigma_\alpha)}{d_f}}$$

- **Unit:** `reflectance (dimensionless)`
- **Kind:** Sample estimate

### C. Pairwise Spectral Distance Distribution
For distinct pairs of sampled metamers, Euclidean distance:

$$D_{ij} = \|R_i - R_j\|_2 = \|\alpha_i - \alpha_j\|_2$$

Reported metrics:
- Mean pairwise distance (`reflectance-vector distance`)
- Median pairwise distance (`reflectance-vector distance`)
- 95th percentile distance (`reflectance-vector distance`)
- Maximum observed pairwise distance (empirical lower bound for the fiber diameter)
- **Kind:** Sample estimate (evaluated on up to 50,000 distinct pairs)

### D. Full Spectral Covariance and Ambiguity Modes
Sample covariance matrix in 31-dimensional spectral space:

$$\Sigma_R = \frac{1}{N-1}\sum_{i=1}^N (R_i - \bar{R})(R_i - \bar{R})^T \in \mathbb{R}^{31 \times 31}$$

- Non-zero eigenvalues $\lambda_1 \ge \dots \ge \lambda_{d_f} > 0$ (`reflectance^2`).
- Eigenvectors $v_j \in \mathbb{R}^{31}$ (`dimensionless`) represent principal spectral ambiguity modes.
- Explained variance percentage: $\frac{\lambda_j}{\sum \lambda_k} \times 100\%$ (`%`).

---

## 6. Determinant-Based Scores and Volume Scaling

The generalized variance in reduced coordinates is:

$$\operatorname{det}(\Sigma_\alpha) = \prod_{j=1}^{d_f} \lambda_j(\Sigma_\alpha)$$

- **Unit:** $\text{reflectance}^{2 d_f}$ ($\text{reflectance}^{10}$)

The characteristic ellipsoidal volume scale $\mathcal{A}_{\mathrm{det}}$ is:

$$\mathcal{A}_{\mathrm{det}} = (\operatorname{det}(\Sigma_\alpha))^{\frac{1}{2 d_f}}$$

- **Unit:** `reflectance (dimensionless)`
- **Geometric Interpretation:** Linear radius of the equivalent spherical volume proxy whose moment matches the generalized variance.

---

## 7. Wavelength-Wise Spectral Bounds (Linear Programming)

For any wavelength $\lambda_j$ ($j = 0, \dots, 30$):

$$R_{\min}(\lambda_j) = \min_{\alpha} \left( \mu_j + B_j z_0 + (B_j N) \alpha \right) \quad \text{s.t.} \quad G \alpha \le h$$

$$R_{\max}(\lambda_j) = \max_{\alpha} \left( \mu_j + B_j z_0 + (B_j N) \alpha \right) \quad \text{s.t.} \quad G \alpha \le h$$

- Solved using the HiGHS dual simplex optimizer.
- Spectral span: $\Delta R(\lambda_j) = R_{\max}(\lambda_j) - R_{\min}(\lambda_j)$ (`reflectance (0–1)`).
- Compared with empirical sample extrema and percentile ribbons to detect undercoverage.

---

## 8. Constructed Composite Ambiguity Index

An optional, transparent composite index combining three normalized, orthogonal ambiguity dimensions:

$$I_{\mathrm{composite}} = \frac{1}{3} \left( \frac{S_{\mathrm{spread}}}{0.20} + \frac{\overline{\Delta R}}{0.30} + \frac{\mathcal{A}_{\mathrm{det}}}{0.10} \right)$$

- **Unit:** `constructed index (dimensionless)`
- **Kind:** Constructed index
- Normalization benchmarks reflect typical dispersion scales of the 36K-image prior.

---

## 9. Sampling Methodology and Statistical Reliability

Hit-and-Run Markov Chain Monte Carlo generates uniform points within the bounded polytope $G\alpha \le h$:
- Configurable budget $N \in [100, 5000]$.
- Default burn-in of 50 steps, thinning interval of 2 steps.
- Fixed random seed (default 42) ensures exact reproducibility.
- Effective Sample Size (ESS) computed using Geyer's initial positive sequence estimator on each coordinate.
- Lag-1 autocorrelation monitored to ensure adequate chain mixing.

---

## 10. Numerical Validation & Integrity Checks

For every target observation, the validator verifies:
1. **Measurement Reconstruction:** $\|\hat{c}_i - c\|_2 < 10^{-4}$ where $\hat{c}_i = M R_i$ (`RGB units`).
2. **Equality Residual:** $\|A z_i - b\|_2 < 10^{-5}$ (`RGB units`).
3. **Polytope Feasibility:** $\max_j (A_{\mathrm{red}, j} z_i - b_{\mathrm{red}, j}) \le 10^{-5}$ (`constraint units`).
4. **Feasibility Status:** Reports PASS, WARN, or FAIL based on rigorous thresholds.

---

## 11. Units Summary Table

| Metric | Symbol | Unit | Status |
|---|---|---|---|
| Affine Fiber Dimension | $d_{\mathrm{affine}}$ | dimensions | Exact |
| Feasible Fiber Dimension | $d_{\mathrm{feasible}}$ | dimensions | Exact / Verified |
| Component Count | $K$ | components | Exact |
| Constraint Rank | $\operatorname{rank}(A)$ | rank | Exact |
| Spectral RMS Spread | $S_{\mathrm{spread}}$ | reflectance (0–1) | Sample Estimate |
| Alpha Coordinate Spread | $\sigma_{\mathrm{spread}}$ | reflectance (0–1) | Sample Estimate |
| Pairwise Spectral Distance | $D_{ij}$ | reflectance-vector dist | Sample Estimate |
| Volume Scale | $\mathcal{A}_{\mathrm{det}}$ | reflectance (0–1) | Sample Estimate |
| Generalized Variance | $\operatorname{det}(\Sigma_\alpha)$ | reflectance$^{10}$ | Sample Estimate |
| Bounding Box Volume | $V_{\mathrm{box}}$ | reflectance$^5$ | Sample Estimate |
| Wavelength Bounds | $R_{\min}, R_{\max}$ | reflectance (0–1) | Exact (LP) |
| Spectral Span | $\Delta R$ | reflectance (0–1) | Exact (LP) |
| Reconstruction Error | $\|\hat{c} - c\|_2$ | RGB units | Measured |
| Equality Residual | $\|Az - b\|_2$ | RGB units | Measured |
| Polytope Slack Violation | $\max(G\alpha - h)$ | constraint units | Measured |
| Effective Sample Size | ESS | effective samples | Diagnostic |
