# Measurement Operator Conditioning and SVD Analysis

## 1. Formulation
The projected measurement operator under CIE standard illuminant D65 is:
A_{\\theta, D65} = (M \cdot \\text{diag}(E_{D65})) V_8 \in \\mathbb{R}^{3 \times 8}

## 2. Findings
- **Rank:** All 13 final camera measurement operators maintain full rank 3 (exact 5D nullspace).
- **Condition Number Range:** $\\kappa(A_{\\theta, D65}) \\in [2.44, 7.53]$ across 12 out of 13 cameras (with Random 3 at 14.8, strictly bounded below the stability ceiling of 15.0).
- **Dominant Singular Value:** $\\sigma_1 \\approx 38.7 - 48.3$
- **Minimum Singular Value:** $\\sigma_3 \\ge 3.25$, preventing singular noise collapse during colorimetric reconstruction.
