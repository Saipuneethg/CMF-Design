# Seven Camera Comparisons Comprehensive Report: Metameric Ambiguity, Sensitivity & Robustness

**Research Project**: Spectral Metameric Fiber Analyzer & Joint Sensitivity Optimization  
**Date**: 2026-10-26 16:08:14  
**Evaluation Scope**: 100 Natural PNG Colours Evaluated across 30 Camera Models under Comparisons 1–7  
**Visual Deliverables Directory**: `camera comparisions/ambiguity_results/` (All 27 comparison plots preserved)  

---

## Executive Summary

This document integrates **seven distinct comparative evaluations** of camera spectral sensitivity functions and metameric ambiguity across the benchmark of 100 natural colours from `Dataset/image.png`. The evaluation compares the **Optimized Sensitivity Model (`Run_2_seed101`)**, the **Original Calibrated Camera Model (`artifacts/M_camera.npy`)**, and a diverse database of **28 Measured Physical Cameras**.

Across all seven comparative dimensions:
1. **Comparison 1 (Per-Colour Improvement)**: The Optimized Model achieves an average ambiguity reduction of **+3.19%**, improving **64% of all colours** relative to the original baseline.
2. **Comparison 2 (Statistical Rigor)**: The ambiguity reduction is statistically significant under both paired t-test (p = 1.71e-06) and Wilcoxon signed-rank test (p = 6.51e-06), with a 95% bootstrap confidence interval of `[+0.00370, +0.00837]`.
3. **Comparison 3 (Spectral Sensitivity Geometry)**: The optimized model achieves spectral decoupling by shifting the green channel peak from 560 nm to **520 nm**, reducing linear redundancy against the red channel (610 nm peak).
4. **Comparison 4 (Pareto Trade-Off)**: The optimized camera establishes an empirical Pareto frontier between metameric fibre suppression and colorimetric reconstruction error ($\Delta E_{00}$).
5. **Comparison 5 (30x100 Ambiguity Matrix)**: Full cross-camera matrix analysis demonstrates consistent lower spread across natural colour clusters.
6. **Comparison 6 (Feasibility & Polytope Slack)**: 100% of the 3,000 evaluations across all 30 cameras achieved strictly valid polytope feasibility ($A_{\text{red}} z \le b_{\text{red}}$) and machine-precision camera measurement residuals.
7. **Comparison 7 (Sampling Robustness)**: Camera ambiguity rankings remain invariant across MCMC sampling budgets ($N=25$ to $N=200$, Spearman $\rho > 0.98$) and independent random seeds.

---

## Comparison 1: Per-Colour Ambiguity & Improvement Analysis

- **Numerical Artifact**: [`camera comparisions/ambiguity_results/comparison_1_per_colour_improvement.csv`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/camera%20comparisions/ambiguity_results/comparison_1_per_colour_improvement.csv)
- **Generated Plots**:
  - [`comparison_1_paired_per_colour_ambiguity.png`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/camera%20comparisions/ambiguity_results/comparison_1_paired_per_colour_ambiguity.png)
  - [`comparison_1_percentage_improvement.png`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/camera%20comparisions/ambiguity_results/comparison_1_percentage_improvement.png)
  - [`comparison_1_improvement_distribution.png`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/camera%20comparisions/ambiguity_results/comparison_1_improvement_distribution.png)
  - [`comparison_1_best_worst_colours.png`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/camera%20comparisions/ambiguity_results/comparison_1_best_worst_colours.png)

### Top 5 Improved Colours (Optimized vs Original)
| Colour ID | RGB (uint8) | Percentage Improvement |
| :--- | :---: | :---: |
| `colour_002` | `[101, 207, 215]` | **+57.27%** |
| `colour_053` | `[193, 217, 204]` | **+31.69%** |
| `colour_044` | `[72, 78, 68]` | **+31.27%** |
| `colour_041` | `[87, 86, 29]` | **+28.94%** |
| `colour_081` | `[104, 124, 131]` | **+26.76%** |

### Top 5 Regressed Colours
| Colour ID | RGB (uint8) | Percentage Change |
| :--- | :---: | :---: |
| `colour_069` | `[84, 173, 155]` | **-74.41%** |
| `colour_025` | `[43, 94, 64]` | **-34.39%** |
| `colour_027` | `[23, 36, 16]` | **-29.07%** |
| `colour_012` | `[81, 174, 119]` | **-28.22%** |
| `colour_072` | `[36, 61, 23]` | **-27.82%** |

---

## Comparison 2: Statistical Rigor & Hypothesis Testing

- **Numerical Artifact**: [`camera comparisions/ambiguity_results/comparison_2_statistical_results.json`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/camera%20comparisions/ambiguity_results/comparison_2_statistical_results.json)
- **Generated Plots**:
  - [`comparison_2_paired_improvement_confidence_interval.png`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/camera%20comparisions/ambiguity_results/comparison_2_paired_improvement_confidence_interval.png)
  - [`comparison_2_paired_difference_distribution.png`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/camera%20comparisions/ambiguity_results/comparison_2_paired_difference_distribution.png)
  - [`comparison_2_cumulative_improvement.png`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/camera%20comparisions/ambiguity_results/comparison_2_cumulative_improvement.png)

| Statistical Metric | Value | Interpretation |
| :--- | :---: | :--- |
| Mean Paired Reduction | `+0.00600` | Net contraction across natural colours |
| Median Paired Reduction | `+0.00250` | Positive central tendency shift |
| Bootstrap 95% Confidence Interval | `[+0.00370, +0.00837]` | Strictly excludes zero |
| Paired t-test t-statistic | `5.0897` | $p = 1.71e-06$ (significant) |
| Wilcoxon Signed-Rank W | `3793.0` | $p = 6.51e-06$ (significant) |
| Cohen's d Effect Size | `0.5090` | Moderate-to-large effect size |

---

## Comparison 3: Spectral Sensitivity Curves & Similarity Analysis

- **Numerical Artifact**: [`camera comparisions/ambiguity_results/comparison_3_sensitivity_similarity.csv`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/camera%20comparisions/ambiguity_results/comparison_3_sensitivity_similarity.csv)
- **Generated Plots**:
  - [`comparison_3_sensitivity_red.png`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/camera%20comparisions/ambiguity_results/comparison_3_sensitivity_red.png)
  - [`comparison_3_sensitivity_green.png`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/camera%20comparisions/ambiguity_results/comparison_3_sensitivity_green.png)
  - [`comparison_3_sensitivity_blue.png`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/camera%20comparisions/ambiguity_results/comparison_3_sensitivity_blue.png)
  - [`comparison_3_sensitivity_all_channels.png`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/camera%20comparisions/ambiguity_results/comparison_3_sensitivity_all_channels.png)
  - [`comparison_3_sensitivity_difference.png`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/camera%20comparisions/ambiguity_results/comparison_3_sensitivity_difference.png)
  - [`comparison_3_sensitivity_similarity_heatmap.png`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/camera%20comparisions/ambiguity_results/comparison_3_sensitivity_similarity_heatmap.png)

---

## Comparison 4: Ambiguity vs Colorimetric Reconstruction Error

- **Numerical Artifact**: [`camera comparisions/ambiguity_results/comparison_4_reconstruction_results.csv`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/camera%20comparisions/ambiguity_results/comparison_4_reconstruction_results.csv)
- **Generated Plots**:
  - [`comparison_4_ambiguity_vs_reconstruction_error.png`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/camera%20comparisions/ambiguity_results/comparison_4_ambiguity_vs_reconstruction_error.png)
  - [`comparison_4_camera_tradeoff.png`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/camera%20comparisions/ambiguity_results/comparison_4_camera_tradeoff.png)
  - [`comparison_4_pareto_comparison.png`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/camera%20comparisions/ambiguity_results/comparison_4_pareto_comparison.png)

---

## Comparison 5: Full 30-Camera × 100-Colour Ambiguity Matrix

- **Numerical Artifact**: [`camera comparisions/ambiguity_results/comparison_5_ambiguity_matrix.csv`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/camera%20comparisions/ambiguity_results/comparison_5_ambiguity_matrix.csv)
- **Generated Plots**:
  - [`comparison_5_ambiguity_heatmap.png`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/camera%20comparisions/ambiguity_results/comparison_5_ambiguity_heatmap.png)
  - [`comparison_5_standardized_heatmap.png`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/camera%20comparisions/ambiguity_results/comparison_5_standardized_heatmap.png)
  - [`comparison_5_improvement_heatmap.png`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/camera%20comparisions/ambiguity_results/comparison_5_improvement_heatmap.png)

---

## Comparison 6: Feasibility, Polytope Slack & Coverage Analysis

- **Numerical Artifact**: [`camera comparisions/ambiguity_results/comparison_6_feasibility_results.csv`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/camera%20comparisions/ambiguity_results/comparison_6_feasibility_results.csv)
- **Generated Plots**:
  - [`comparison_6_feasibility_coverage.png`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/camera%20comparisions/ambiguity_results/comparison_6_feasibility_coverage.png)
  - [`comparison_6_feasibility_status_counts.png`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/camera%20comparisions/ambiguity_results/comparison_6_feasibility_status_counts.png)
  - [`comparison_6_ambiguity_vs_coverage.png`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/camera%20comparisions/ambiguity_results/comparison_6_ambiguity_vs_coverage.png)
  - [`comparison_6_common_subset_comparison.png`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/camera%20comparisions/ambiguity_results/comparison_6_common_subset_comparison.png)

---

## Comparison 7: Robustness, Sample Budget & Seed Variability

- **Numerical Artifact**: [`camera comparisions/ambiguity_results/comparison_7_robustness_runs.csv`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/camera%20comparisions/ambiguity_results/comparison_7_robustness_runs.csv)
- **Generated Plots**:
  - [`comparison_7_ambiguity_vs_budget.png`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/camera%20comparisions/ambiguity_results/comparison_7_ambiguity_vs_budget.png)
  - [`comparison_7_seed_variability.png`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/camera%20comparisions/ambiguity_results/comparison_7_seed_variability.png)
  - [`comparison_7_ranking_stability.png`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/camera%20comparisions/ambiguity_results/comparison_7_ranking_stability.png)
  - [`comparison_7_rank_correlation_heatmap.png`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/camera%20comparisions/ambiguity_results/comparison_7_rank_correlation_heatmap.png)

---

## Complete Deliverables Inventory in `camera comparisions/ambiguity_results/`

All 27 visualization plots are saved directly in `camera comparisions/ambiguity_results/`:
- [`comparison_1_paired_per_colour_ambiguity.png`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/camera%20comparisions/ambiguity_results/comparison_1_paired_per_colour_ambiguity.png)
- [`comparison_1_percentage_improvement.png`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/camera%20comparisions/ambiguity_results/comparison_1_percentage_improvement.png)
- [`comparison_1_improvement_distribution.png`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/camera%20comparisions/ambiguity_results/comparison_1_improvement_distribution.png)
- [`comparison_1_best_worst_colours.png`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/camera%20comparisions/ambiguity_results/comparison_1_best_worst_colours.png)
- [`comparison_2_paired_improvement_confidence_interval.png`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/camera%20comparisions/ambiguity_results/comparison_2_paired_improvement_confidence_interval.png)
- [`comparison_2_paired_difference_distribution.png`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/camera%20comparisions/ambiguity_results/comparison_2_paired_difference_distribution.png)
- [`comparison_2_cumulative_improvement.png`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/camera%20comparisions/ambiguity_results/comparison_2_cumulative_improvement.png)
- [`comparison_3_sensitivity_red.png`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/camera%20comparisions/ambiguity_results/comparison_3_sensitivity_red.png)
- [`comparison_3_sensitivity_green.png`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/camera%20comparisions/ambiguity_results/comparison_3_sensitivity_green.png)
- [`comparison_3_sensitivity_blue.png`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/camera%20comparisions/ambiguity_results/comparison_3_sensitivity_blue.png)
- [`comparison_3_sensitivity_all_channels.png`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/camera%20comparisions/ambiguity_results/comparison_3_sensitivity_all_channels.png)
- [`comparison_3_sensitivity_difference.png`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/camera%20comparisions/ambiguity_results/comparison_3_sensitivity_difference.png)
- [`comparison_3_sensitivity_similarity_heatmap.png`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/camera%20comparisions/ambiguity_results/comparison_3_sensitivity_similarity_heatmap.png)
- [`comparison_4_ambiguity_vs_reconstruction_error.png`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/camera%20comparisions/ambiguity_results/comparison_4_ambiguity_vs_reconstruction_error.png)
- [`comparison_4_camera_tradeoff.png`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/camera%20comparisions/ambiguity_results/comparison_4_camera_tradeoff.png)
- [`comparison_4_pareto_comparison.png`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/camera%20comparisions/ambiguity_results/comparison_4_pareto_comparison.png)
- [`comparison_5_ambiguity_heatmap.png`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/camera%20comparisions/ambiguity_results/comparison_5_ambiguity_heatmap.png)
- [`comparison_5_standardized_heatmap.png`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/camera%20comparisions/ambiguity_results/comparison_5_standardized_heatmap.png)
- [`comparison_5_improvement_heatmap.png`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/camera%20comparisions/ambiguity_results/comparison_5_improvement_heatmap.png)
- [`comparison_6_feasibility_coverage.png`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/camera%20comparisions/ambiguity_results/comparison_6_feasibility_coverage.png)
- [`comparison_6_feasibility_status_counts.png`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/camera%20comparisions/ambiguity_results/comparison_6_feasibility_status_counts.png)
- [`comparison_6_ambiguity_vs_coverage.png`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/camera%20comparisions/ambiguity_results/comparison_6_ambiguity_vs_coverage.png)
- [`comparison_6_common_subset_comparison.png`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/camera%20comparisions/ambiguity_results/comparison_6_common_subset_comparison.png)
- [`comparison_7_ambiguity_vs_budget.png`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/camera%20comparisions/ambiguity_results/comparison_7_ambiguity_vs_budget.png)
- [`comparison_7_seed_variability.png`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/camera%20comparisions/ambiguity_results/comparison_7_seed_variability.png)
- [`comparison_7_ranking_stability.png`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/camera%20comparisions/ambiguity_results/comparison_7_ranking_stability.png)
- [`comparison_7_rank_correlation_heatmap.png`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/camera%20comparisions/ambiguity_results/comparison_7_rank_correlation_heatmap.png)

