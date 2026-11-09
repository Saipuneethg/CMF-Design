# Sensor Noise Robustness and Covariance Constraint Analysis

## 1. Physics-Based Noise Model
The measurement model incorporates both signal-dependent Poisson photon shot noise and signal-independent Gaussian readout noise:
$$y_c = \alpha \cdot \mathcal{P}\left(\frac{(A \rho)_c}{\alpha}\right) + \mathcal{N}(0, \sigma_{\text{read}}^2), \quad c \in \{R, G, B\}$$
where:
- Full-well capacity: $FWC = 20,000 \, e^-$
- Read noise: $\sigma_{\text{read}} = 3.0 \, e^-$
- Signal-to-Noise Ratio (SNR) benchmarks:
  - Noise-Free: Theoretical limit ($\infty$ dB)
  - Low Noise: Target $\sim 40$ dB (ISO 100 equivalent)
  - Moderate Noise: Target $\sim 32$ dB (ISO 800 equivalent)
  - High Noise: Target $\sim 24$ dB (ISO 3200 equivalent)

## 2. Comparison: Scalar vs. Covariance-Aware Mahalanobis Constraints
1. **Scalar Tolerance:**
   $$\lVert A \rho - y_{\text{noisy}} \rVert_2 \le \epsilon, \quad \epsilon = 2 \bar{\sigma}_{RGB}$$
2. **Covariance-Aware (Mahalanobis) Constraint:**
   $$(r - y_{\text{noisy}})^T \Sigma^{-1} (r - y_{\text{noisy}}) \le \tau, \quad \tau = \chi_3^2(0.95) = 7.815$$
   where $\Sigma = \text{diag}(\sigma_R^2, \sigma_G^2, \sigma_B^2)$ accounts for channel-specific variance due to differing spectral photon yields.

## 3. Robustness Results Across Noise Regimes (D65)
| Noise Regime | Target SNR | Baseline Ambiguity | Optimized Ambiguity | Ambiguity Reduction | Covariance vs Scalar Difference |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Noise-Free** | $\infty$ | 0.273236 | 0.252316 | **+7.66\%** | N/A |
| **Low Noise** | 40 dB | 0.273296 | 0.252459 | **+7.62\%** | < 0.05\% |
| **Moderate Noise** | 32 dB | 0.273419 | 0.252724 | **+7.57\%** | < 0.12\% |
| **High Noise** | 24 dB | 0.273957 | 0.253795 | **+7.36\%** | < 0.38\% |

## 4. Key Scientific Conclusion
Even under severe 24 dB high noise, the Optimized CSSF preserves an ambiguity reduction advantage of **+7.36%** over baseline. The covariance-aware constraint modifies the fiber cross-section by less than 0.4%, demonstrating that the 5-dimensional nullspace boundaries imposed by the empirical spectral prior polytope dominate over measurement uncertainty ellipsoids.
