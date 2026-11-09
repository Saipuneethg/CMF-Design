# Average Metameric Ambiguity Comparison Report: 100 PNG-Derived Colours

**Research Project**: Spectral Metameric Fiber Analyzer & Joint Sensitivity Optimization  
**Date**: 2026-10-25 15:05:53  
**Evaluation Scope**: 100 Unique In-Gamut Colours Sampled from `Dataset/image.png`  

---

## A. Experimental Setup

- **Source Image**: `Dataset/image.png` (RGB, 186,123 unique colors).
- **Colour Selection Protocol**: Exactly identical 100 in-gamut colours extracted during `run_20261019_211619` (`seed=42`).
- **Measurement Models Compared**:
  1. **Original Model**: `artifacts/M_camera.npy` (calibrated 3-channel camera matrix, $\mathbb{R}^{3 \times 31}$).
  2. **Optimized Model**: `artifacts/M_camera_optimized.npy` (winning `Run_2_seed101` camera sensitivity matrix).
- **Mathematical Comparison Protocol**: **Physical Surface Reflectance Equivalence**.
  Each colour represents the physical surface reflectance $\bar{R}^{(i)} \in \mathbb{R}^{31}$ identified in the original experiment. The original camera measured $c_{\text{orig}}^{(i)} = \mathbf{M}_{\text{base}} \bar{R}^{(i)}$, and the optimized camera measured $c_{\text{opt}}^{(i)} = \mathbf{M}_{\text{opt}} \bar{R}^{(i)}$. This guarantees that both models evaluate the exact same physical stimulus.
- **Ambiguity Metric**: Spectral RMS Spread $S_{\text{spread}} = \sqrt{\frac{1}{N} \sum_{j=1}^N \|R_j - \bar{R}\|_2^2}$ (reflectance units, $[0, 1]$).
- **Sampling Budget**: $N = 500$ Hit-and-Run samples per colour, burn-in $= 50$, thinning $= 2$, reproducible seed $= 42 + \text{idx}$.
- **Polytope Constraints**: $A_{\text{red}} z \le b_{\text{red}}$ ($714,340$ linear inequalities).

---

## B. Summary Comparison

| Metric | Original Model (`M_camera`) | Optimized Model (`M_camera_optimized`) |
| :--- | :---: | :---: |
| **Number of colours evaluated** | 100 | 100 |
| **Valid fibre count** | 100 | 100 |
| **Infeasible colour count** | 0 | 0 |
| **Failed evaluation count** | 0 | 0 |
| **Average ambiguity** | **0.111872** | **0.099748** |
| **Median ambiguity** | **0.101666** | **0.090710** |
| **Standard deviation** | 0.063165 | 0.056440 |
| **Minimum ambiguity** | 0.007286 | 0.005137 |
| **Maximum ambiguity** | 0.251864 | 0.222747 |

### Core Ambiguity Contraction:
- **Absolute Difference**: `-0.012124` reflectance units
- **Percentage Reduction**: **`+10.84%`**
- **Colours with Reduced Ambiguity**: **86 / 100 (86.0%)**
- **Colours with Increased Ambiguity**: 14 / 100
- **Colours with Equal Ambiguity**: 0 / 100
- **Mean Paired Difference**: `-0.012124` reflectance units
- **Median Paired Difference**: `-0.012809` reflectance units

---

## C. Individual Colour Analysis

### 1. Colours with Largest Ambiguity Reduction
| Colour ID | RGB (uint8) | Original Ambiguity | Optimized Ambiguity | Reduction (%) |
| :--- | :---: | :---: | :---: | :---: |
| `colour_002` | (101, 207, 215) | 0.012524 | 0.005137 | **+58.98%** |
| `colour_060` | (51, 62, 19) | 0.037174 | 0.026117 | **+29.74%** |
| `colour_065` | (68, 69, 51) | 0.094976 | 0.067353 | **+29.08%** |
| `colour_079` | (79, 111, 46) | 0.101065 | 0.072659 | **+28.11%** |
| `colour_032` | (48, 41, 49) | 0.083536 | 0.061749 | **+26.08%** |

### 2. Colours with Largest Ambiguity Increase
| Colour ID | RGB (uint8) | Original Ambiguity | Optimized Ambiguity | Change (%) |
| :--- | :---: | :---: | :---: | :---: |
| `colour_025` | (43, 94, 64) | 0.009326 | 0.018577 | **-99.20%** |
| `colour_037` | (87, 176, 182) | 0.021404 | 0.035986 | **-68.13%** |
| `colour_012` | (81, 174, 119) | 0.037463 | 0.052812 | **-40.97%** |
| `colour_069` | (84, 173, 155) | 0.029616 | 0.038299 | **-29.32%** |
| `colour_029` | (41, 63, 16) | 0.007286 | 0.008736 | **-19.91%** |

### 3. Lowest Ambiguity Colours
| Rank | Original Model Colour | Ambiguity | Optimized Model Colour | Ambiguity |
| :---: | :---: | :---: | :---: | :---: |
| 1 | `colour_029` ([41, 63, 16]) | 0.007286 | `colour_002` ([101, 207, 215]) | 0.005137 |
| 2 | `colour_025` ([43, 94, 64]) | 0.009326 | `colour_029` ([41, 63, 16]) | 0.008736 |
| 3 | `colour_027` ([23, 36, 16]) | 0.011461 | `colour_027` ([23, 36, 16]) | 0.011286 |
| 4 | `colour_002` ([101, 207, 215]) | 0.012524 | `colour_025` ([43, 94, 64]) | 0.018577 |
| 5 | `colour_037` ([87, 176, 182]) | 0.021404 | `colour_056` ([51, 60, 15]) | 0.018781 |

---

## D. Scientific Interpretation

The empirical experiment demonstrates that the **optimized camera spectral sensitivity model successfully reduced average metameric ambiguity across this 100-colour dataset by 10.84%** (from 0.11187 to 0.09975 reflectance units).
Furthermore, **86% of all 100 individual colours** experienced strict ambiguity reduction.

Key findings:
1. **Global Contraction**: The average fiber spread contracted from 0.11187 to 0.09975, representing a net collapse of metameric uncertainty.
2. **Distributional Shift**: Both the median (0.10167 $\to$ 0.09071) and maximum observed ambiguity (0.25186 $\to$ 0.22275) shifted downward, demonstrating that the camera optimization compresses worst-case spectral ambiguity.
3. **Physical Feasibility**: 100% of the physical spectra evaluated produced strictly valid metameric fibers under both cameras with zero constraint violations ($\|M R - c\| < 10^{-14}$ and $A_{\text{red}} z \le b_{\text{red}}$).

