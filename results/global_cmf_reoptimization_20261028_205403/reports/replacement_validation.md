# Replacement & Integrity Validation Report

## 1. Acceptance Criteria Evaluation
All required acceptance criteria were evaluated on the frozen held-out test partition ($N=250$) prior to replacement:
- **Ambiguity Reduced vs Baseline**: PASS (+9.37%)
- **Ambiguity Maintained/Reduced vs Old Model**: PASS (-0.00%)
- **Color Difference Tolerance**: PASS (Mean $\Delta E_{00} = 1.61 \le 2.0$ JND)
- **Physical Non-Negativity**: PASS ($M \ge 0$)
- **Full Measurement Rank**: PASS (Rank = 3, Fiber Dimension = 5)
- **Condition Number**: PASS ($\kappa \le 15.0$)
- **Spectral Smoothness**: PASS ($S \le 0.05$)

## 2. Checksums & Artifact Verification
- **New Active CMF Checksum (`artifacts/M_camera_optimized.npy`)**:
  `6324faa2c7aed07ab8e9952d8d72b47c540a53c74c0bb8e80b6c2f57818d6729`
- **Previous CMF Backup Checksum**:
  `0bf3a13c719346da19b48632f4a7fe95ccdca5247f8b7f15b22f2451f6d00127`
- **Previous CMF Backup Directory**:
  `results/backup_previous_optimized_cmf/`

## 3. Downstream Compatibility
- **Unit Tests**: 13 of 13 unit tests passed successfully.
- **Array Dimensions**: Camera matrix $(3, 31)$, Polytope measurement matrix $(3, 8)$, Offset vector $(3,)$.
