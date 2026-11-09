# Post-Optimization Global CMF Evaluation — All 32 CMFs

This directory contains the independent, comprehensive evaluation of images generated using **all 32 available Color Matching Functions (CMFs)** in the Spectral Metameric Fiber Analyzer against original ground-truth images.

---

## 1. Directory Structure

```text
cmf_post_optimization_analysis/
├── generated_images/
│   ├── 0002/                        # Primary benchmark scene (000002.png)
│   │   ├── Global_Optimized_CMF.png # Exact CMF name filenames
│   │   ├── Original_Calibrated_Camera.png
│   │   ├── CIE_1931_2deg_Observer.png
│   │   ├── CIE_1964_10deg_Observer.png
│   │   ├── canon_1dmarkiii.png
│   │   └── ... (32 CMF images per scene)
│   ├── 0001/                        # Secondary validation scene
│   └── 0003/                        # Tertiary validation scene
├── comparison_images/
│   ├── 0002_all_32_cmfs_comparison.png # 33-Panel Unified Poster
│   ├── 0001_all_32_cmfs_comparison.png
│   └── 0003_all_32_cmfs_comparison.png
├── difference_maps/
│   ├── 0002/                        # Absolute RGB and CIEDE2000 heatmaps
│   │   ├── Global_Optimized_CMF_abs_diff.png
│   │   ├── Global_Optimized_CMF_ciede2000.png
│   │   └── ...
│   ├── 0002_error_distribution_histograms.png
│   └── 0002_per_channel_error_comparison.png
├── metrics/
│   ├── per_image_cmf_metrics.csv    # Granular per-image/CMF records
│   ├── per_cmf_summary.csv          # Aggregated 32 CMF rankings
│   └── global_optimization_comparison.csv # Comparison against reference baseline
├── analysis/
│   ├── detailed_quantitative_analysis.md # Full scientific technical report
│   ├── visual_analysis.md           # Perceptual and visual review
│   └── optimization_effectiveness.md# Pareto trade-off and efficacy analysis
└── README.md                        # This document
```

---

## 2. All 32 Evaluated CMFs

1. **Global Optimized CMF** (`Global_Optimized_CMF.png`) — Master Active Optimization
2. **Original Calibrated Camera Model** (`Original_Calibrated_Camera.png`) — Hardware Baseline
3. **CIE 1931 2° Standard Observer** (`CIE_1931_2deg_Observer.png`) — CIE Reference
4. **CIE 1964 10° Supplementary Observer** (`CIE_1964_10deg_Observer.png`) — CIE Reference
5–32. **28 Physical Camera Models**:
   - Canon (1D Mark III, 20D, 300D, 40D, 500D, 50D, 5D Mark II, 600D, 60D)
   - Nikon (D3X, D200, D3, D300s, D40, D50, D5100, D700, D80, D90)
   - Hasselblad (H2)
   - Olympus (E-PL2)
   - Pentax (K-5, Q)
   - Sony (NEX-5N)
   - Point Grey (Grasshopper 50S5C, Grasshopper2 14S5C)
   - Phase One
   - Nokia (N900)

---

## 3. How to Reproduce

Execute the evaluation script from the repository root:
```bash
python scripts/run_post_optimization_cmf_evaluation.py
```

### Dependencies
- Python 3.10+
- `numpy`, `scipy`, `pandas`
- `colour-science`
- `scikit-image`
- `matplotlib`, `Pillow`
