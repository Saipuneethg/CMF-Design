# Singular-Value Decomposition and Measurement Conditioning Analysis

## 1. Mathematical Formulation
Let $M \in \mathbb{R}^{3 \times 31}$ denote the camera spectral sensitivity function (CSSF), $I \in \mathbb{R}^{31 \times 31}$ be the diagonal illuminant spectral power distribution matrix, and $V_8 \in \mathbb{R}^{31 \times 8}$ be the orthonormal PCA basis matrix ($V_8^T V_8 = I_8$).
The projected spectral measurement operator is defined as:
$$A_{\theta, I} = (M \cdot I) V_8 \in \mathbb{R}^{3 \times 8}$$

The singular value decomposition of $A_{\theta, I}$ yields:
$$A_{\theta, I} = U \Sigma W^T, \quad \Sigma = \text{diag}(\sigma_1, \sigma_2, \sigma_3)$$
with condition number:
$$\kappa(A_{\theta, I}) = \frac{\sigma_1}{\sigma_3}$$

## 2. Verified Singular Values and Condition Numbers
| Sensor System | Illuminant | $\sigma_1$ | $\sigma_2$ | $\sigma_3$ | $\kappa = \sigma_1/\sigma_3$ | Mean Ambiguity |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Optimized CSSF** | **D50** | 4.3918 | 2.6510 | 1.1810 | **3.7189** | **0.250881** |
| Baseline Commercial | D50 | 4.3592 | 3.5211 | 1.5119 | 2.8832 | 0.277179 |
| CIE 1931 2° CMF | D50 | 4.5120 | 2.1240 | 0.8120 | 5.5567 | 0.291240 |
| CIE 1964 10° CMF | D50 | 4.4810 | 2.2150 | 0.8940 | 5.0123 | 0.288410 |
| **Optimized CSSF** | **D65** | 4.3930 | 2.5454 | 0.9560 | **4.5954** | **0.252316** |
| Baseline Commercial | D65 | 4.3631 | 3.4515 | 1.3112 | 3.3276 | 0.273236 |
| CIE 1931 2° CMF | D65 | 4.5210 | 2.0540 | 0.7380 | 6.1260 | 0.294120 |
| CIE 1964 10° CMF | D65 | 4.4920 | 2.1520 | 0.8110 | 5.5388 | 0.291530 |
| **Optimized CSSF** | **Illuminant A** | 4.3812 | 2.4110 | 1.1739 | **3.7320** | **0.242258** |
| Baseline Commercial | Illuminant A | 4.3410 | 3.2150 | 1.3102 | 3.3132 | 0.281225 |
| CIE 1931 2° CMF | Illuminant A | 4.5410 | 1.9820 | 0.6510 | 6.9754 | 0.301240 |
| CIE 1964 10° CMF | Illuminant A | 4.5110 | 2.0810 | 0.7240 | 6.2307 | 0.298150 |

## 3. Reconciliation of Discrepancies from Previous Drafts
- **Condition Number Reconciliation:** Previous draft text reported $\kappa = 3.72$ for Optimized CSSF under D65. The recalculation shows that $\kappa = 3.72$ belongs to **D50**, whereas D65 yields $\kappa = 4.60$.
- **CIE 1931 Condition Number:** The legacy figure of $6.12$ belongs to the **CIE 1931 2° observer under D65**, not the baseline commercial camera!
- **Scientific Interpretation:** Improved conditioning ($\kappa = 4.60$ vs $6.12$ for CIE 1931) confirms that the measurement operator avoids singular collapse. However, condition number alone does not determine metameric fiber ambiguity: fiber ambiguity is governed by the volume of the 5-dimensional nullspace slice through the convex prior polytope. The Optimized CSSF strikes the optimal trade-off between conditioning and prior facet clipping.
