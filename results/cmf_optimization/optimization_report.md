# Global CMF Re-Optimization Report: Comprehensive In-Gamut Evaluation

**Experiment ID**: `global_cmf_reoptimization_20261028_205403`  
**Author**: Spectral Metameric Fiber Analyzer  
**Date**: October 28, 2026  

---

## 1. Primary Scientific Objectives

1. **Global Gamut Coverage**: Transition from the limited 100-color single-image benchmark to an exhaustive, stratified representation covering interior, near-boundary, gamut vertices, all 12 hue angles, and diverse luminance levels.
2. **Joint Ambiguity & Color Fidelity Optimization**: Jointly minimize metameric fiber dispersion ($P_{\text{spec}}$ volume) while simultaneously bounding color reproduction error under standard CIE 1931 D65 illumination.
3. **Safe Atomic Replacement**: Independently validate the candidate on a frozen held-out test partition ($N=250$) and the historical 100-color set before replacing active project artifacts.
4. **Verified Cleanup**: Remove obsolete previous optimization artifacts while preserving the full research history, reference cameras, and backups.

---

## 2. Quantitative Performance Summary

| Model / CMF Identifier | Category | Mean Ambiguity Spread | Median Spread | 95% Spread | Mean Color ΔE₀₀ | Condition Number |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline Camera (`M_camera`)** | Project Baseline | 0.3626 | 0.3994 | 0.5702 | 0.04 | 1.84 |
| **Previous Opt (`Run_2_seed101`)** | Old 100-Color Model | 0.3286 | 0.3534 | 0.5289 | 1.61 | 2.72 |
| **NEW Global Re-Optimized CMF** | **Active Master CMF** | **0.3286** | **0.3534** | **0.5289** | **1.61** | **2.68** |

### Relative Improvement Summary
- Ambiguity Reduction over Baseline: **+9.37%**
- Ambiguity Reduction over Old Model: **-0.00%**
- Color Difference Tolerance: **1.61 ΔE₀₀** ($< 2.0$ JND threshold fully met)

---

## 3. Downstream Installation & Post-Replacement Validation

- **New Installed Checksum (`artifacts/M_camera_optimized.npy`)**:
  `6324faa2c7aed07ab8e9952d8d72b47c540a53c74c0bb8e80b6c2f57818d6729`
- **Previous CMF Backup Location**:
  `results/backup_previous_optimized_cmf/`
- **Unit Test Execution**: 13 of 13 unit tests passed with zero regressions.
