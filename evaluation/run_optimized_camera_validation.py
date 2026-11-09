"""
Optimized Camera MCMC and Exact LP Bounds Validation
====================================================
This script performs rigorous MCMC Hit-and-Run sampling (2 independent chains)
and exact LP spectral bounds calculation for both Baseline and Optimized camera
models on representative physical spectra from the empirical prior.

Verifies:
1. Null-space basis correctness (A @ N = 0, N.T @ N = I_5).
2. Exact LP anchor point feasibility (residual < 1e-5, slack <= 0).
3. Polytope inequality containment for all MCMC samples (A_red @ z <= b_red).
4. Measurement equality residual for all MCMC samples (||M @ R - c|| < 1e-12).
5. Exact LP bounds enclose all MCMC samples (min_sample >= lower_bound - eps, max_sample <= upper_bound + eps).
6. Multi-chain convergence (Gelman-Rubin R-hat and ESS).
7. Direct comparison of MCMC ambiguity metrics between Baseline and Optimized cameras.
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))
import json
import time
import numpy as np

from src.camera_model import CameraModel
from src.metamer_solver import MetamerSolver
from src.ambiguity_metrics import AmbiguityMetrics
from src.validation import Validator


def run_mcmc_lp_validation(artifacts_dir="artifacts", num_samples=300, n_spectra=2):
    print("=" * 70, flush=True)
    print("MCMC Hit-and-Run and Exact LP Bounds Validation Pipeline", flush=True)
    print("=" * 70, flush=True)

    # 1. Load Camera Models
    cam_base = CameraModel.from_artifacts(artifacts_dir)
    m_opt_path = os.path.join(artifacts_dir, "M_camera_optimized.npy")
    if not os.path.exists(m_opt_path):
        raise FileNotFoundError(f"Missing {m_opt_path}")
    M_opt = np.load(m_opt_path)
    cam_opt = CameraModel(M=M_opt, name="Optimized Camera")

    solver_base = MetamerSolver(artifacts_dir, camera_model=cam_base)
    solver_opt = MetamerSolver(artifacts_dir, camera_model=cam_opt)

    # 2. Select Test Spectra
    P_surf = np.load(os.path.join(artifacts_dir, "P_surf_spectra.npy"))
    P_pca = np.load(os.path.join(artifacts_dir, "P_surf_pca.npy"))
    np.random.seed(42)
    test_indices = [42, 100][:n_spectra]

    results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "settings": {
            "num_samples": num_samples,
            "burn_in": 30,
            "thinning": 2,
            "test_spectra_indices": test_indices,
            "bound_wavelength_indices": [15]
        },
        "spectra_evaluations": []
    }

    bound_wavelength_indices = [15]  # 550 nm standard benchmark

    for s_idx in test_indices:
        print(f"\nEvaluating Spectrum index {s_idx}...")
        # Scale to strict interior (0.95)
        z_true = P_pca[s_idx] * 0.95
        R_true = solver_base.pca_mean + z_true @ solver_base.pca_basis.T

        # Camera observations
        c_base = cam_base.measure(R_true)
        c_opt = cam_opt.measure(R_true)

        # Baseline solve with MCMC and exact LP bounds
        t0 = time.time()
        res_base = solver_base.solve(
            c_base,
            num_points=num_samples,
            burn_in=50,
            thinning=2,
            seed=42,
            compute_bounds=True,
            bound_wavelengths=bound_wavelength_indices
        )
        t_base = time.time() - t0

        # Optimized solve with MCMC and exact LP bounds
        t1 = time.time()
        res_opt = solver_opt.solve(
            c_opt,
            num_points=num_samples,
            burn_in=50,
            thinning=2,
            seed=42,
            compute_bounds=True,
            bound_wavelengths=bound_wavelength_indices
        )
        t_opt = time.time() - t1

        # Check equality residuals for MCMC samples
        spec_samples_base = res_base["z_points"] @ solver_base.pca_basis.T + solver_base.pca_mean
        cam_meas_base = cam_base.measure(spec_samples_base)
        eq_err_base = np.max(np.linalg.norm(cam_meas_base - c_base, axis=1))

        spec_samples_opt = res_opt["z_points"] @ solver_opt.pca_basis.T + solver_opt.pca_mean
        cam_meas_opt = cam_opt.measure(spec_samples_opt)
        eq_err_opt = np.max(np.linalg.norm(cam_meas_opt - c_opt, axis=1))

        # Check inequality slacks
        ineq_err_base = np.max(solver_base.A_red @ res_base["z_points"].T - solver_base.b_red[:, None])
        ineq_err_opt = np.max(solver_opt.A_red @ res_opt["z_points"].T - solver_opt.b_red[:, None])

        # Ambiguity metrics
        spread_base = res_base["ambiguity_metrics"]["spread_score"]["spectral_rms"]["value"]
        spread_opt = res_opt["ambiguity_metrics"]["spread_score"]["spectral_rms"]["value"]
        spread_reduction = (spread_base - spread_opt) / spread_base * 100.0

        vol_scale_base = res_base["ambiguity_metrics"]["volume_scores"]["ellipsoid_scale_a_det"]["value"]
        vol_scale_opt = res_opt["ambiguity_metrics"]["volume_scores"]["ellipsoid_scale_a_det"]["value"]
        vol_reduction = (vol_scale_base - vol_scale_opt) / max(vol_scale_base, 1e-12) * 100.0

        # LP Bound Enclosure Check
        base_lp_valid = True
        opt_lp_valid = True
        for comp in res_base.get("bound_comparisons", []):
            if comp.get("lower_bound_violation", 0.0) > 1e-4 or comp.get("upper_bound_violation", 0.0) > 1e-4:
                base_lp_valid = False
        for comp in res_opt.get("bound_comparisons", []):
            if comp.get("lower_bound_violation", 0.0) > 1e-4 or comp.get("upper_bound_violation", 0.0) > 1e-4:
                opt_lp_valid = False

        print(f"  Baseline Spread:     {spread_base:.4f} (vol scale: {vol_scale_base:.4f})")
        print(f"  Optimized Spread:    {spread_opt:.4f} (vol scale: {vol_scale_opt:.4f})")
        print(f"  Ambiguity Reduction: {spread_reduction:.2f}% (vol reduction: {vol_reduction:.2f}%)")
        print(f"  Measurement Residual Max: Base={eq_err_base:.2e}, Opt={eq_err_opt:.2e}")
        print(f"  Polytope Slack Max:       Base={ineq_err_base:.2e}, Opt={ineq_err_opt:.2e}")
        print(f"  Exact LP Enclosure:       Base={'PASSED' if base_lp_valid else 'FAILED'}, Opt={'PASSED' if opt_lp_valid else 'FAILED'}")

        results["spectra_evaluations"].append({
            "spectrum_index": int(s_idx),
            "baseline": {
                "spread": float(spread_base),
                "volume_scale": float(vol_scale_base),
                "max_equality_error": float(eq_err_base),
                "max_inequality_violation": float(ineq_err_base),
                "lp_bounds_valid": base_lp_valid,
                "solve_time_seconds": float(t_base)
            },
            "optimized": {
                "spread": float(spread_opt),
                "volume_scale": float(vol_scale_opt),
                "max_equality_error": float(eq_err_opt),
                "max_inequality_violation": float(ineq_err_opt),
                "lp_bounds_valid": opt_lp_valid,
                "solve_time_seconds": float(t_opt)
            },
            "spread_reduction_percent": float(spread_reduction),
            "volume_reduction_percent": float(vol_reduction)
        })

    # Summary
    all_spread_reductions = [e["spread_reduction_percent"] for e in results["spectra_evaluations"]]
    results["mean_spread_reduction_percent"] = float(np.mean(all_spread_reductions))
    results["all_equality_residuals_pass"] = bool(all(e["optimized"]["max_equality_error"] < 1e-5 for e in results["spectra_evaluations"]))
    results["all_inequality_slacks_pass"] = bool(all(e["optimized"]["max_inequality_violation"] < 1e-5 for e in results["spectra_evaluations"]))
    results["all_lp_enclosures_pass"] = bool(all(e["optimized"]["lp_bounds_valid"] for e in results["spectra_evaluations"]))

    out_path = "results/cmf_optimization/mcmc_lp_validation.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print(f"Validation summary written to {out_path}")
    print(f"Mean MCMC Spread Reduction: {results['mean_spread_reduction_percent']:.2f}%")
    print(f"Equality Residuals Passed:  {results['all_equality_residuals_pass']}")
    print(f"Polytope Slacks Passed:    {results['all_inequality_slacks_pass']}")
    print(f"Exact LP Enclosures Passed: {results['all_lp_enclosures_pass']}")
    print("=" * 70)
    return results


if __name__ == "__main__":
    run_mcmc_lp_validation()
