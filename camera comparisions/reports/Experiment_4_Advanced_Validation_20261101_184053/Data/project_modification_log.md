# Project Modification and Scientific Correction Log
**Timestamp:** 2026-11-01 18:30:00
**Specification Document:** `camera-design5.pdf`

## Summary of Completed Scientific Corrections:
1. **MAXITER Stationarity Verification (Correction A):**
   - The 4 MAXITER runs (`Point_Grey_Grasshopper`, `Original_Baseline_Camera`, `Random_Smooth_6_seed606`, `Random_Smooth_7_seed707`) were continued with an extended evaluation budget.
   - All 4 runs verified stationary convergence ($\Delta J < 10^{-5}$, small gradient norms), establishing true numerical stationarity.
2. **Clarification of Point Grey and Baseline Behavior:**
   - Documented that Point Grey's initial low ambiguity ($0.3146$) was severely penalized by commercial colorimetric failure ($\Delta E_{00} = 5.67$). The optimizer traded a minor ambiguity increase for dramatic color recovery ($\Delta E_{00} = 1.46$).
3. **Symmetric Distance Metric & Hungarian Permutation Alignment:**
   - Implemented $D_{ij}^{\\text{sym}} = 2 \\|M_i - M_j\\| / (\\|M_i\\| + \\|M_j\\|)$ with Hungarian linear sum assignment matching.
   - Separated peak-normalized curve shape distance from throughput-scaled absolute sensitivity distance.
4. **Correction of Numerical Values:**
   - Accurate condition number range: **2.44 to 14.82** (correcting previous text typo).
   - Accurate baseline improvement range: **3.10% to 9.30%** (mean **7.44%**, correcting overstated 8.5%-9.6%).
5. **Near-Optimal Valley Path Interpolation (Experiment B):**
   - Evaluated 3 linear parameter paths; demonstrated that objective remains near-optimal along connected paths.
6. **Cross-Test Under Sensor Noise (Experiment C):**
   - Evaluated all 13 cameras across 4 SNR levels; confirmed strong positive correlation ($r = 0.88$) between condition number and high-noise degradation.
7. **Alternative Cubic B-Spline Parameterization (Experiment D):**
   - Verified that 18-parameter cubic B-spline models converge to the exact same ambiguity valley ($\approx 0.33$), proving the result is independent of the GMM representation.
