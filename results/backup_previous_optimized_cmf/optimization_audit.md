# Optimization Audit & Methodological Diagnosis
## Joint Camera Spectral-Sensitivity Optimization for Reduced Metameric Ambiguity

**Date:** October 2026  
**Auditor:** Computational Color Science & Optimization Engineering Team  
**Scope:** Investigation of the optimization objective, evaluation pipeline, and held-out distribution behavior identified in `optimization_changes.pdf` and `camera_optimization_report.pdf`.

---

## 1. Executive Summary of Audit Findings

A rigorous technical audit of the optimization pipeline (`src/camera_optimizer.py`), mathematical formulation, and evaluation data revealed the following root causes for the phenomena described in Document A (`optimization_changes.pdf`):

1. **Root Cause of the Held-Out Median Discrepancy:**
   - **Objective vs. Metric Divergence:** The previous optimization objective in `src/camera_optimizer.py` minimized:
     $$J_{\text{amb}} = 0.70 \cdot \mathbb{E}[\operatorname{Spread}] + 0.30 \cdot Q_{95}[\operatorname{Spread}]$$
     This objective explicitly assigned **zero weight to the median** ($Q_{50}$) and placed heavy weight ($30\%$) on the extreme upper tail ($Q_{95}$).
   - **Marginal vs. Paired Statistics:** In the held-out evaluation, the marginal median shifted from $0.1390 \rightarrow 0.1644$, creating the visual impression in independent boxplots that the camera worsened around the median. However, per-sample paired analysis reveals that **84.0% of held-out spectra (42 of 50) actually improved**, and the **median paired difference is strictly negative** ($\operatorname{median}(\Delta) = -0.006393$).
   - **Ranking Artifact on a Bimodal Population:** Natural reflectance spectra cluster into low-ambiguity ($[0.03, 0.10]$) and high-ambiguity ($[0.35, 0.55]$) modes. Because the optimizer did not penalize individual sample regressions ($\max(0, S_{\text{opt}}[i] - S_{\text{base}}[i])$), a small handful of samples (specifically indices 8, 18, 36) in the transition band increased in ambiguity and crossed the 50th percentile rank boundary, shifting the marginal 50th percentile upward despite the widespread reduction in ambiguity across 84% of the population.

2. **Methodological Deficiencies Identified:**
   - **Single Optimization Run:** The previous camera was obtained from a single optimization run (`seed=42`) without multi-start exploration across diverse parameter initializations.
   - **Model Selection Protocol:** The winning candidate was not selected using an explicit validation-set criterion comparing multiple candidate initializations.
   - **Absence of Paired Statistics:** Held-out results were reported primarily via independent aggregate statistics rather than paired differences ($\Delta_i = S_{\text{opt}}[i] - S_{\text{base}}[i]$), bootstrap confidence intervals, or formal hypothesis tests (Wilcoxon signed-rank / permutation tests).

---

## 2. Source Code & Function Audit Matrix

| System Component | Source File & Location | Equation / Implementation | Identified Limitation | Required Correction |
| :--- | :--- | :--- | :--- | :--- |
| **Ambiguity Objective** | `src/camera_optimizer.py` (L347) | $J_{\text{amb}} = 0.70 \bar{S} + 0.30 Q_{95}$ | Ignores median; allows individual sample regressions. | Add median term and paired regression penalty against baseline. |
| **Model Selection** | `src/camera_optimizer.py` (L387-470) | Single DE run with `seed=42`. | Susceptible to local minima; no validation selection. | Implement multi-start ($\ge 5$ runs); select winning candidate strictly on validation data. |
| **Paired Evaluation** | `src/camera_optimizer.py` (L460-476) | Independent aggregate means. | Conceals paired sample shifts; lacks statistical testing. | Implement sample-by-sample paired analysis, bootstrap CIs, and Wilcoxon signed-rank test. |
| **Fast Bounding** | `src/camera_optimizer.py` (L305-325) | Coordinate spans along 5 axes: $\sqrt{\sum \operatorname{span}_j^2 / 12}$ | Approximates uniform bounding box variance. | Preserve for screening; validate with full MCMC Hit-and-Run. |
| **Throughput Normalization** | `src/camera_optimizer.py` (L166-174) | $\sum_{j=1}^{31} s_k(\lambda_j) = 10.672$ | Verified correct. | Retain identical normalization across all candidate cameras. |
| **Non-Negativity** | `src/camera_optimizer.py` (L161-164) | Gaussian mixtures $a_{kj} e^{-(\cdot)^2}$ | Guaranteed strictly positive ($s_k \ge 0$). | Retain; verify numerical tolerance ($\ge 0$). |
| **Smoothness Regularization** | `src/camera_optimizer.py` (L210-219) | $\sum_k \sum_j (d^2 s_k / d\lambda^2)^2 \Delta \lambda$ | Discrete curvature penalty with $\Delta \lambda = 10$ nm. | Retain; weight properly balanced against ambiguity terms. |
| **Colorimetric Fidelity** | `src/camera_optimizer.py` (L242-267) | $\Delta E_{00}$ under D65 against CIE XYZ | Verified standard CIE conversion and matrix fit. | Retain acceptance threshold ($\Delta E_{00} \le 2.0$). |

---

## 3. Detailed Diagnosis: Marginal vs. Paired Statistics

### 3.1 Mathematical Analysis
Let $\mathcal{R}_{\text{test}} = \{R_1, \dots, R_N\}$ be the fixed held-out evaluation set ($N=50$). For each spectrum $R_i$, let:
$$S_{\text{base}}[i] = \operatorname{Spread}\bigl(\Phi_{\text{base}}(M_{\text{base}} R_i)\bigr)$$
$$S_{\text{opt}}[i] = \operatorname{Spread}\bigl(\Phi_{\text{opt}}(M_{\text{opt}} R_i)\bigr)$$
$$\Delta[i] = S_{\text{opt}}[i] - S_{\text{base}}[i]$$

The baseline and candidate distributions on the held-out set evaluate to:
- **Baseline:** Mean = $0.233135$, Median = $0.139047$, $Q_{95} = 0.541295$.
- **Optimized:** Mean = $0.223529$, Median = $0.164411$, $Q_{95} = 0.507696$.

### 3.2 The Paired Breakdown
Computing the paired differences $\Delta[i]$ reveals:
- **Mean Paired Difference:** $\mathbb{E}[\Delta] = -0.009605$ ($-4.12\%$ on average).
- **Median Paired Difference:** $\operatorname{median}(\Delta) = -0.006393$ (Ambiguity decreased for the median sample!).
- **Improved Samples ($\Delta[i] < 0$):** **42 out of 50 samples (84.0%)**.
- **Worsened Samples ($\Delta[i] > 0$):** **8 out of 50 samples (16.0%)**.
- **Unchanged Samples ($\Delta[i] = 0$):** **0 samples (0.0%)**.

The 8 worsened samples were:
- Index 3: $0.4616 \rightarrow 0.4806$ ($+4.1\%$)
- Index 8: $0.0639 \rightarrow 0.1153$ ($+80.5\%$)
- Index 15: $0.1070 \rightarrow 0.1085$ ($+1.4\%$)
- Index 17: $0.0536 \rightarrow 0.0583$ ($+8.6\%$)
- Index 18: $0.1101 \rightarrow 0.2252$ ($+104.6\%$)
- Index 31: $0.2690 \rightarrow 0.3049$ ($+13.4\%$)
- Index 36: $0.1007 \rightarrow 0.1665$ ($+65.3\%$)
- Index 47: $0.1462 \rightarrow 0.1739$ ($+19.0\%$)

Notice that in the baseline camera, samples 8, 15, 17, 18, 36 were all ranked *below* the median ($< 0.1390$). When samples 8, 18, and 36 experienced regressions, they crossed the median ranking threshold, moving from the lower 50% into the upper 50%. This shifted the 25th sorted element upward to $0.1644$, despite 84% of the population experiencing reductions in ambiguity!

---

## 4. Revised Formulation Plan

To directly address this diagnostic and fulfill all project requirements:

1. **Revised Optimization Objective:**
   Augment the training ambiguity loss to include explicit median control and a regression penalty against the baseline camera on training spectra:
   $$J_{\text{amb}}(\theta) = w_{\text{mean}} \cdot \mathbb{E}[S] + w_{\text{median}} \cdot Q_{50}[S] + w_{\text{tail}} \cdot Q_{95}[S] + w_{\text{reg}} \cdot \mathbb{E}\bigl[\max(0, S_\theta - S_{\text{base}})\bigr]$$
   Recommended normalized weights:
   $$w_{\text{mean}} = 0.40, \quad w_{\text{median}} = 0.30, \quad w_{\text{tail}} = 0.20, \quad w_{\text{reg}} = 0.10$$
   This penalizes any upward shift in the median and penalizes individual sample regressions, ensuring that improvements are distributed uniformly across both typical and extreme spectra.

2. **Multi-Start Optimization Pipeline:**
   Execute 5 independent optimization runs with distinct random seeds (`seeds = [42, 101, 202, 303, 404]`). For each run, track:
   - Initial & final objectives
   - Training & validation mean, median, Q95 spread
   - Condition number $\kappa(M M^T)$
   - Smoothness penalty $C_{\text{smooth}}$
   - Colorimetric error $\Delta E_{00}$ under D65
   - Convergence status & runtime

3. **Strict Validation-Based Candidate Selection:**
   Rank candidates on the **Validation Set** ($N=20$) using a multi-criteria scoring function:
   $$\text{Score}_{\text{val}} = J_{\text{val}} + \lambda_{\text{color}} \max(0, \overline{\Delta E_{00}} - 1.5)^2 + \lambda_{\text{cond}} \max(0, \kappa - 10)^2$$
   The held-out test set ($N=50$) remains strictly untouched during optimization and model selection.

4. **Rigorous Paired Held-Out Evaluation:**
   Evaluate the winning candidate on the held-out test set and compute:
   - Per-sample differences $\Delta_i$
   - Percentage of improved vs. worsened samples
   - Bootstrap 95% confidence intervals (10,000 resamples) for paired mean and median differences
   - Wilcoxon signed-rank test and paired permutation test p-values

5. **Deliverables to Regenerate:**
   - `results/cmf_optimization/multi_start_summary.csv`
   - `results/cmf_optimization/paired_heldout_results.csv`
   - `results/cmf_optimization/paired_statistics.json`
   - `results/cmf_optimization/paired_ambiguity_analysis.png`
   - Updated `artifacts/M_camera_optimized.*`, `results/cmf_optimization/baseline_metrics.json`, `optimized_metrics.json`, `comparison.csv`, etc.
   - Updated comprehensive research report: `results/cmf_optimization/research_report.md` and `camera_optimization_report_updated.pdf`.
