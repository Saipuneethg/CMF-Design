# Replacement & Integrity Validation Report

## 1. Acceptance Criteria Evaluation
- **Ambiguity Reduced vs Baseline**: PASS (+9.37%)
- **Color Difference Tolerance**: PASS (Mean $\Delta E_{00} = 1.614 \le 2.0$ JND)
- **Reference Dominance**: Evaluated against all 28 references. Outperforms 27 of 28 on mean ambiguity, and outperforms all 28 on tail robustness (Q95) and color fidelity.
- **Physical Non-Negativity**: PASS ($M \ge 0$)
- **Full Measurement Rank**: PASS (Rank = 3, Fiber Dimension = 5)
- **Condition Number**: PASS ($\kappa = 2.68 \le 15.0$)
- **Downstream Unit Tests**: 13/13 PASS.

## 2. Integrity Status
The current active CMF at `artifacts/M_camera_optimized.npy` remains valid, tested, and protected by backup in `C:\Users\saipu\OneDrive\Desktop\fiber polytopes-2\results\advanced_global_cmf_optimization_20261029_200921\backup_current_cmf`.
