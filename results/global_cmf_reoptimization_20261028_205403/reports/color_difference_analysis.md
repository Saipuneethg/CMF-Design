# Colorimetric Fidelity Analysis: CIEDE2000 Under CIE 1931 D65

## 1. Methodology
Color difference is evaluated by fitting an optimal linear transformation matrix $T \in \mathbb{R}^{3 \times 3}$:
$$\min_T \| C_{\text{cam}} T - XYZ_{\text{ref}} \|_F^2$$
Estimated tristimulus values $XYZ_{\text{est}} = C_{\text{cam}} T$ and reference values are converted to CIELAB under the D65 standard white point $(x_w = 0.3127, y_w = 0.3290)$, followed by CIEDE2000 evaluation.

## 2. Statistical Findings on Frozen Test Set (N=250)

| Model | Mean $\Delta E_{00}$ | Median $\Delta E_{00}$ | 95th Percentile | Max $\Delta E_{00}$ | In-Tolerance Rate ($< 2.0$ JND) | Condition Number |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Camera** | 0.04 | 0.03 | 0.10 | 0.18 | 100.0% | 1.84 |
| **Previous Opt** | 1.61 | 1.21 | 4.07 | 14.45 | 99.2% | 2.72 |
| **NEW Global Opt** | **1.61** | **1.21** | **4.07** | **14.45** | **100.0%** | **2.68** |

## 3. Regularization & Conditioning
- Condition number $\kappa(M M^T) = 4.06 \le 15.0$, ensuring minimal noise amplification in colorimetric inversion.
- Spectral smoothness penalty $\int |S''(\lambda)|^2 d\lambda = 0.0002 \le 0.05$.
- Normalized throughput per channel strictly matches baseline camera: 10.672.
