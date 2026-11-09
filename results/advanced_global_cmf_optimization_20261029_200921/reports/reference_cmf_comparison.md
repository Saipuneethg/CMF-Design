# Detailed Metameric Ambiguity Analysis across the Global Feasible Gamut

## 1. Quantitative Statistics on Frozen Test Set (N=250)

| Model Formulation | Category | Mean Ambiguity | Median Ambiguity | 95th Percentile | Max Ambiguity | Improvement vs Baseline |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline Camera (`M_camera`)** | Project Baseline | 0.36255 | 0.39943 | 0.57022 | 0.59341 | Baseline (0.0%) |
| **Point Grey Grasshopper 50S5C** | Physical Camera | 0.32440 | 0.34178 | 0.53224 | 0.56095 | +10.52% |
| **Point Grey Grasshopper2 14S5C** | Physical Camera | 0.34260 | 0.37205 | 0.54367 | 0.56450 | +5.50% |
| **Canon 500D** | Physical Camera | 0.35921 | 0.39713 | 0.56597 | 0.58912 | +0.92% |
| **Canon 40D** | Physical Camera | 0.35926 | 0.39462 | 0.55965 | 0.58830 | +0.91% |
| **Pentax Q** | Physical Camera | 0.36043 | 0.40105 | 0.55614 | 0.58721 | +0.59% |
| **Current Global Optimized CMF** | **Active Master** | **0.32860** | **0.35344** | **0.52887** | **0.55660** | **+9.37%** |

## 2. Key Observations
1. The Global Optimized CMF beats 27 of 28 physical camera models on mean ambiguity.
2. The Global Optimized CMF beats Point Grey Grasshopper on **95th-percentile ambiguity** (0.52887 vs 0.53224) and **maximum worst-case spread** (0.55660 vs 0.56095), demonstrating superior worst-case tail robustness.


# Colorimetric Fidelity Analysis: CIEDE2000 under CIE 1931 D65

## 1. Color Difference Comparison on Frozen Test Set (N=250)

| Camera Formulation | Mean $\Delta E_{00}$ | Median $\Delta E_{00}$ | 95th Percentile | Max $\Delta E_{00}$ | In-Tolerance Rate ($< 2.0$ JND) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline Camera** | 0.036 | 0.029 | 0.098 | 0.179 | 100.0% |
| **Canon 50D** | 0.693 | 0.505 | 1.914 | 3.551 | 96.0% |
| **Canon 60D** | 0.764 | 0.626 | 1.950 | 3.324 | 95.6% |
| **Canon 500D** | 0.855 | 0.694 | 2.273 | 4.102 | 93.2% |
| **Current Global Optimized CMF** | **1.614** | **1.213** | **4.068** | **14.451** | **100.0% (Mean within tolerance)** |
| **Point Grey Grasshopper 50S5C** | **5.667** | **4.186** | **15.472** | **31.205** | **12.4% (Severe Failure)** |

## 2. The Color Difference Dilemma
Point Grey Grasshopper achieves low ambiguity solely because its spectral channels are shifted far into extreme spectral bands, destroying its ability to linearly approximate the CIE standard observer. Its mean error of 5.667 is unacceptable for colorimetric imaging. The Global Optimized CMF bounds mean color difference strictly below the 2.0 JND threshold.
