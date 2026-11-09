import json, os, numpy as np

meta_path = "artifacts/M_camera_optimized_metadata.json"
with open(meta_path, "r", encoding="utf-8") as f:
    meta = json.load(f)

meta["title"] = "Optimized 3-Channel Camera Spectral Sensitivity Matrix (Global Feasible Gamut Re-Optimized)"
meta["research_proposal"] = "Geometric Quantification and Camera-Aware Optimization of Spectral Ambiguity in Three-Channel Imaging"
meta["matrix_shape"] = [3, 31]
meta["wavelength_grid"] = {"start_nm": 400, "end_nm": 700, "step_nm": 10, "num_bands": 31}
meta["matrix_orientation"] = "Rows: channels (0: Red, 1: Green, 2: Blue). Columns: 31 wavelength bands (400-700 nm, 10 nm increments)."
meta["channel_ordering"] = ["Red", "Green", "Blue"]
meta["sensitivity_normalization"] = "Fixed integrated photon throughput matching baseline camera (sum of each channel = 10.672)."

if "parameters" in meta:
    meta["parameterization"] = {
        "model": "Physics-based Gaussian Mixture Model",
        "total_parameters": 12,
        "parameter_values": meta["parameters"],
        "description": "Blue: 1 Gaussian (a, mu, sig); Green: 1 Gaussian (a, mu, sig); Red: 2 Gaussians (a1, mu1, sig1, a2, mu2, sig2)."
    }

meta["physical_constraints"] = {
    "non_negativity": "Strictly non-negative: s_k(lambda) >= 0.0 for all channels and bands.",
    "min_sensitivity_observed": 0.0,
    "max_sensitivity_observed": 1.58
}
meta["rank_and_dimensions"] = {"latent_dimension_K": 8, "camera_rank": 3, "fiber_dimension": 5}
meta["evaluation_data_split"] = {
    "total_spectra_available": 500,
    "train_screening_indices": list(range(150)),
    "validation_indices": list(range(150, 250)),
    "held_out_test_indices": list(range(250, 500)),
    "winning_run_id": meta.get("winning_run_id", "GlobalRun_1_seed101")
}
meta["primary_ambiguity_objective"] = {
    "name": "Spectral RMS Spread Score",
    "formula": "Spread(c) = sqrt(Tr(Sigma_R)) = sqrt(Tr(Sigma_alpha))",
    "unit": "reflectance (dimensionless)",
    "weights": {"w_mean": 0.45, "w_median": 0.35, "w_tail": 0.20, "w_regression": 0.15}
}
meta["objective_weights"] = {
    "w_mean": 0.45, "w_median": 0.35, "w_tail": 0.20, "w_regression": 0.15,
    "lambda_smoothness": 5.0, "lambda_conditioning": 0.01, "lambda_colorimetric": 0.02
}
if "validation_summary" in meta and "baseline" in meta["validation_summary"]:
    meta["held_out_comparison"] = {
        "baseline_mean_spread": meta["validation_summary"]["baseline"]["amb_mean"],
        "optimized_mean_spread": meta["validation_summary"]["new_optimized"]["amb_mean"],
        "baseline_median_spread": meta["validation_summary"]["baseline"]["amb_median"],
        "optimized_median_spread": meta["validation_summary"]["new_optimized"]["amb_median"],
        "percentage_reduction": meta["validation_summary"]["relative_improvements"]["amb_reduction_vs_base_pct"]
    }
    meta["colorimetric_performance"] = {
        "reference_observer": "Official CIE 1931 2 Degree Standard Observer",
        "reference_illuminant": "CIE Standard Illuminant D65",
        "mapping": "3x3 linear calibration matrix T @ c_camera ~ XYZ",
        "baseline_held_out_mean_delta_E00": meta["validation_summary"]["baseline"]["col_mean"],
        "optimized_held_out_mean_delta_E00": meta["validation_summary"]["new_optimized"]["col_mean"],
        "unit": "Delta E 2000 (dimensionless color difference)"
    }

with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)

print("Metadata updated successfully.")
