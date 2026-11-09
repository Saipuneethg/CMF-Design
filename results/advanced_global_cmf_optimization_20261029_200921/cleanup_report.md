# Final CMF Optimization Artifact Cleanup Report

**Timestamp**: 2026-10-29 20:46:00  
**Scope**: Permanent deletion of obsolete previous optimization images (`.png`), markdown reports (`.md`), legacy PDF report, and legacy checksum manifest from `results/cmf_optimization/`.

---

## 1. Summary of Actions
- **Total Obsolete Files Deleted**: 13 files
- **Total Disk Space Recovered**: 3,620,103 bytes (~3.45 MB)
- **Rollback Backup Integrity**: 100% verified in `results/backup_previous_optimized_cmf/` (29 files intact).
- **Active Global CMF Integrity**: Verified in `artifacts/M_camera_optimized.npy` (SHA-256: `6324faa2c7aed07ab8e9952d8d72b47c540a53c74c0bb8e80b6c2f57818d6729`).
- **Downstream Unit Tests**: 13/13 passed (`tests/test_camera_optimization.py`).

---

## 2. Deleted Obsolete Files

| Filename | Type | Size | Rollback Backup Status | Rationale |
| :--- | :---: | :---: | :---: | :--- |
| `FINAL_RESULTS_MANIFEST.md` | MARKDOWN | 9,955 B | Verified Intact | Superseded final results manifest from 100-sample run (Run_2_seed101, Oct 25 2026); backed up in results/backup_previous_optimized_cmf/ |
| `optimization_audit.md` | MARKDOWN | 9,556 B | Verified Intact | Superseded diagnostic audit document for 100-sample run; backed up in results/backup_previous_optimized_cmf/ |
| `research_report.md` | MARKDOWN | 31,024 B | Verified Intact | Superseded comprehensive research report for 100-sample run; backed up in results/backup_previous_optimized_cmf/ |
| `all_camera_models_ambiguity_comparison.png` | PNG | 528,072 B | Verified Intact | Obsolete 100-sample 28-camera comparison plot; superseded by all_cameras_global_ranking.png; backed up in backup directory |
| `ambiguity_comparison.png` | PNG | 117,394 B | Verified Intact | Obsolete 100-sample ambiguity boxplot; superseded by paired_heldout_reduction.png; backed up in backup directory |
| `camera_models_ambiguity_reduction_analysis.png` | PNG | 222,007 B | Verified Intact | Obsolete 100-sample reduction plot; superseded by paired_heldout_reduction.png; backed up in backup directory |
| `colorimetric_comparison.png` | PNG | 72,820 B | Verified Intact | Obsolete 100-sample delta E plot; superseded by color_difference_distribution.png; backed up in backup directory |
| `numerical_sensitivity_table.png` | PNG | 941,414 B | Verified Intact | Obsolete table graphic for old 100-sample CMF; backed up in backup directory |
| `paired_ambiguity_analysis.png` | PNG | 208,149 B | Verified Intact | Obsolete 100-sample paired scatter plot; superseded by per_color_joint_scatter.png; backed up in backup directory |
| `sensitivity_curves.png` | PNG | 311,587 B | Verified Intact | Obsolete 100-sample sensitivity curves; superseded by sensitivity_curves_global_opt.png; backed up in backup directory |
| `sensitivity_curves_comparison.png` | PNG | 163,362 B | Verified Intact | Combined comparison plot superseded by separate sensitivity_curves_old_opt.png and sensitivity_curves_global_opt.png |
| `camera_optimization_report_updated.pdf` | PDF | 1,002,643 B | Verified Intact | Superseded PDF report for 100-sample run; superseded by optimization_report.pdf and CMF_Optimization_Algorithms_and_Process_Report.pdf; backed up in backup directory |
| `checksums.sha256` | SHA256 | 2,120 B | Verified Intact | Legacy checksum manifest for the 23 files of the 100-sample release; backed up in backup directory |

---

## 3. Preserved Active Deliverables in `results/cmf_optimization/`

The canonical results directory now cleanly holds only the active Global Optimized CMF artifacts, new global diagnostic plots, active scientific reports, and current PDF documentation:

### 3.1 Active Installed CMF Deliverables
- [`artifacts/M_camera_optimized.npy`](file:///C:/Users/saipu/OneDrive/Desktop/fiber polytopes-2/artifacts/M_camera_optimized.npy) & [`results/cmf_optimization/M_camera_optimized.npy`](file:///C:/Users/saipu/OneDrive/Desktop/fiber polytopes-2/results/cmf_optimization/M_camera_optimized.npy)
- [`results/cmf_optimization/M_camera_optimized.csv`](file:///C:/Users/saipu/OneDrive/Desktop/fiber polytopes-2/results/cmf_optimization/M_camera_optimized.csv)
- [`results/cmf_optimization/M_camera_optimized_metadata.json`](file:///C:/Users/saipu/OneDrive/Desktop/fiber polytopes-2/results/cmf_optimization/M_camera_optimized_metadata.json)
- [`results/cmf_optimization/A_camera_optimized.npy`](file:///C:/Users/saipu/OneDrive/Desktop/fiber polytopes-2/results/cmf_optimization/A_camera_optimized.npy)
- [`results/cmf_optimization/camera_offset_optimized.npy`](file:///C:/Users/saipu/OneDrive/Desktop/fiber polytopes-2/results/cmf_optimization/camera_offset_optimized.npy)

### 3.2 Active Global Visualization Plots (PNG)
- [`all_cameras_global_ranking.png`](file:///C:/Users/saipu/OneDrive/Desktop/fiber polytopes-2/results/cmf_optimization/all_cameras_global_ranking.png): 31-camera ranking across global test set
- [`color_difference_distribution.png`](file:///C:/Users/saipu/OneDrive/Desktop/fiber polytopes-2/results/cmf_optimization/color_difference_distribution.png): Global CIEDE2000 distribution vs reference
- [`gamut_sampling_distribution.png`](file:///C:/Users/saipu/OneDrive/Desktop/fiber polytopes-2/results/cmf_optimization/gamut_sampling_distribution.png): Stratified gamut sampling distribution
- [`paired_heldout_reduction.png`](file:///C:/Users/saipu/OneDrive/Desktop/fiber polytopes-2/results/cmf_optimization/paired_heldout_reduction.png): Paired held-out ambiguity contraction
- [`pareto_ambiguity_vs_color_tradeoff.png`](file:///C:/Users/saipu/OneDrive/Desktop/fiber polytopes-2/results/cmf_optimization/pareto_ambiguity_vs_color_tradeoff.png): Pareto ambiguity vs color trade-off frontier
- [`per_color_joint_scatter.png`](file:///C:/Users/saipu/OneDrive/Desktop/fiber polytopes-2/results/cmf_optimization/per_color_joint_scatter.png): Per-color joint ambiguity and delta E scatter
- [`reference_camera_detailed_ranking.png`](file:///C:/Users/saipu/OneDrive/Desktop/fiber polytopes-2/results/cmf_optimization/reference_camera_detailed_ranking.png): Reference cameras ranking breakdown
- [`sensitivity_curves_global_opt.png`](file:///C:/Users/saipu/OneDrive/Desktop/fiber polytopes-2/results/cmf_optimization/sensitivity_curves_global_opt.png): Global Optimized CMF spectral sensitivity curves
- [`sensitivity_curves_old_opt.png`](file:///C:/Users/saipu/OneDrive/Desktop/fiber polytopes-2/results/cmf_optimization/sensitivity_curves_old_opt.png): Previous Optimized CMF spectral sensitivity curves

### 3.3 Active Global Scientific Reports & PDFs
- [`optimization_report.pdf`](file:///C:/Users/saipu/OneDrive/Desktop/fiber polytopes-2/results/cmf_optimization/optimization_report.pdf): Global CMF optimization report
- [`CMF_Optimization_Algorithms_and_Process_Report.pdf`](file:///C:/Users/saipu/OneDrive/Desktop/fiber polytopes-2/results/cmf_optimization/CMF_Optimization_Algorithms_and_Process_Report.pdf): Complete working algorithms and process documentation
- [`ambiguity_analysis.md`](file:///C:/Users/saipu/OneDrive/Desktop/fiber polytopes-2/results/cmf_optimization/ambiguity_analysis.md): Feasible gamut ambiguity analysis
- [`color_difference_analysis.md`](file:///C:/Users/saipu/OneDrive/Desktop/fiber polytopes-2/results/cmf_optimization/color_difference_analysis.md): CIEDE2000 color fidelity analysis
- [`gamut_coverage_analysis.md`](file:///C:/Users/saipu/OneDrive/Desktop/fiber polytopes-2/results/cmf_optimization/gamut_coverage_analysis.md): Regional gamut coverage and boundary performance
- [`executive_summary.md`](file:///C:/Users/saipu/OneDrive/Desktop/fiber polytopes-2/results/cmf_optimization/executive_summary.md): Executive summary of global optimization
- [`optimization_report.md`](file:///C:/Users/saipu/OneDrive/Desktop/fiber polytopes-2/results/cmf_optimization/optimization_report.md): Technical summary report
- [`reference_cmf_comparison.md`](file:///C:/Users/saipu/OneDrive/Desktop/fiber polytopes-2/results/cmf_optimization/reference_cmf_comparison.md): Cross-database 28-camera evaluation
- [`replacement_validation.md`](file:///C:/Users/saipu/OneDrive/Desktop/fiber polytopes-2/results/cmf_optimization/replacement_validation.md): Replacement validation gate checks
- [`cleanup_manifest.json`](file:///C:/Users/saipu/OneDrive/Desktop/fiber polytopes-2/results/cmf_optimization/cleanup_manifest.json): Machine-readable cleanup inventory
- [`cleanup_report.md`](file:///C:/Users/saipu/OneDrive/Desktop/fiber polytopes-2/results/cmf_optimization/cleanup_report.md): This report

---

## 4. Verification and Validation Gate Results
- `tests/test_camera_optimization.py`: **13/13 tests PASSED**.
- Active CMF matrix dimensions, normalization, non-negativity, smoothness, rank, fiber dimension, conditioning, and solver compatibility remain 100% verified.
