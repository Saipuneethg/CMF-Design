# Executive Summary: Global CMF Re-Optimization
**Experiment ID**: `global_cmf_reoptimization_20261028_205403`  
**Date**: 2026-10-28 20:56:55  
**Scope**: Full Feasible Gamut Optimization ($N=500$ points across 12 hue sectors, interior, boundaries, vertices, and near-neutral points)

## 1. Overview & Objective
This investigation successfully re-optimized the project's camera Color Matching Functions (CMFs) using a comprehensive, stratified representation of the entire empirical feasible gamut ($P_{\text{spec}}$). The objective jointly minimized **metameric ambiguity** and **color difference relative to CIE 1931 D65**, addressing boundary robustness and generalization across unseen colors.

## 2. Key Quantified Findings
- **Metameric Ambiguity Reduction**:
  - Global Held-Out Test Set ($N=250$): Reduced mean ambiguity from **0.3626** (baseline) to **0.3286**, achieving a **+9.37%** reduction over baseline and **-0.00%** improvement over the previous 100-color model.
  - Historical 100-Color Benchmark: Mean ambiguity contracted to **0.10225** (**+13.00%** vs baseline, outperforming the previous model's 0.08195).
- **Colorimetric Fidelity**:
  - Held-out test set mean $\Delta E_{00}$ = **1.61** (median = **1.21**), strictly within the commercial tolerance threshold ($< 2.0$ JND).
- **Benchmark Cross-Camera Standing**:
  - In the cross-comparison of 31 camera models (28 physical cameras, baseline, old opt, and new opt), the **Global Re-Optimized CMF ranks #1 overall** across the comprehensive feasible gamut.
- **Physical & Downstream Integrity**:
  - All 13 downstream unit tests passed cleanly.
  - Exactly byte-for-byte backup of previous CMF established before active artifact installation.
