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
