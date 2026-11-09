# Pairwise Sensitivity Curve Distance and Permutation Alignment

## 1. Alignment Protocol
Channel identities were matched by spectral centroid prior to distance calculation:
\\bar{\\lambda}_c = \\frac{\\sum \\lambda s_c(\\lambda)}{\\sum s_c(\\lambda)}
Sorted into [Blue ~450 nm, Green ~540 nm, Red ~610 nm].

## 2. Pairwise Normalized Frobenius Distance
D_{ij} = \\frac{\\|M_i^* - M_j^*\\|_F}{\\|M_i^*\\|_F}
- **Mean Pairwise Distance:** 0.4600
- **Median Pairwise Distance:** 0.4485
- **Maximum Pairwise Distance:** 0.8230
- **Interpretation:** Curves differ in fine spectral bandwidths and secondary lobes while achieving identical functional ambiguity reduction (Outcome B).
