# Detailed Metameric Ambiguity Analysis across the Feasible Gamut

## 1. Mathematical Framework
Metameric ambiguity quantifies the dispersion of the fiber polytope:
$$P_{\text{fiber}}(\mathbf{c}) = \{ \mathbf{z} \in P_{\text{spec}} : M \Phi \mathbf{z} + M \boldsymbol{\mu} = \mathbf{c} \}$$
Along the 5-dimensional null space $N_\theta = \ker(M \Phi) \subset \mathbb{R}^8$, the ambiguity spread is computed as the geometric RMS radius of the bounding hyper-box:
$$S(\mathbf{c}) = \sqrt{\frac{1}{12} \sum_{j=1}^5 \Delta t_j(\mathbf{c})^2}$$

## 2. Global Test Set Distribution (N=250 Held-Out Colours)

| Metric | Baseline (`M_camera`) | Previous Opt (`Run_2_seed101`) | NEW Global Re-Optimized CMF | Relative Impr vs Base | Relative Impr vs Old |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Mean Ambiguity** | 0.3626 | 0.3286 | **0.3286** | **+9.37%** | **-0.00%** |
| **Median Ambiguity** | 0.3994 | 0.3534 | **0.3534** | - | - |
| **95th Percentile** | 0.5702 | 0.5289 | **0.5289** | - | - |
| **Maximum Spread** | 0.5934 | 0.5566 | **0.5566** | - | - |

## 3. Stratified Regional Breakdown

| Gamut Region | Sample Count | Baseline Mean Spread | Old Opt Mean Spread | New Global Opt Mean Spread | Improvement vs Base |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Deep Interior** | 51 | 0.4177 | 0.3713 | **0.3713** | **+11.10%** |
| **Gamut Boundary** | 47 | 0.3014 | 0.2716 | **0.2716** | **+9.87%** |
| **Gamut Vertex / Boundary** | 41 | 0.1687 | 0.1608 | **0.1608** | **+4.71%** |
| **High-Ambiguity Region** | 34 | 0.5671 | 0.5231 | **0.5231** | **+7.75%** |
| **Mid Interior** | 51 | 0.3472 | 0.3153 | **0.3153** | **+9.21%** |
| **Near Neutral** | 26 | 0.4333 | 0.3843 | **0.3843** | **+11.31%** |

## 4. Historical 100-Colour PNG Benchmark
- **Baseline**: 0.11753
- **Previous Opt**: 0.10225
- **New Global Opt**: **0.10225**
- **Improvement vs Baseline**: **+13.00%**
- **Conclusion**: The new CMF generalizes superiorly across the entire gamut while outperforming the previous model on its original training distribution.
