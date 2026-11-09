import os
import sys
import json
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.metamer_solver import MetamerSolver
from src.hit_and_run import HitAndRunSampler
from src.ambiguity_metrics import AmbiguityMetrics
from src.spectral_bounds import SpectralBoundsCalculator


def run_sampling_stability():
    artifacts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "artifacts"))
    solver = MetamerSolver(artifacts_dir)
    rgb = [0.5, 0.5, 0.5]
    
    print("=" * 80)
    print("RUNNING HIT-AND-RUN SAMPLE-COUNT STABILITY & CONVERGENCE STUDY")
    print("=" * 80)
    
    fiber = solver.construct_fiber(rgb)
    if not fiber["feasible"]:
        print("Error: Target RGB is infeasible")
        return

    z0 = fiber["z0"]
    G = fiber["G"]
    h = fiber["h"]
    N = fiber["N"]

    # Compute exact LP bounds for 550 nm (green peak) as ground truth reference
    target_wave_idx = 15 # 550 nm
    exact_bound_550 = solver.bounds_calc.compute_wavelength_bound(target_wave_idx, z0, h)
    print(f"Ground-Truth LP Bounds at 550 nm: [{exact_bound_550['exact_min']:.4f}, {exact_bound_550['exact_max']:.4f}]")

    sample_counts = [100, 250, 500, 1000, 2000]
    results_by_count = []
    
    # Generate largest sample set first (reusing / downsampling for consistency)
    max_count = max(sample_counts)
    print(f"\nSampling reference chain of {max_count} points...")
    sampler = HitAndRunSampler(G, h)
    t0 = time.time()
    full_samples, diag = sampler.sample_chain(
        num_samples=max_count, burn_in=50, thinning=2, seed=42
    )
    sampling_time = time.time() - t0
    print(f"Reference chain generated in {sampling_time:.2f}s.")

    # Reference statistics at max sample count
    full_z = z0 + full_samples @ N.T
    full_spectra = solver.reconstructor.reconstruct(full_z)
    ref_stats = AmbiguityMetrics.compute_sample_statistics(
        full_samples, full_z, full_spectra
    )
    ref_centroid = np.array(ref_stats["alpha_space"]["centroid"])
    ref_cov = np.array(ref_stats["alpha_space"]["covariance_matrix"])
    ref_eigvals = np.array(ref_stats["alpha_space"]["eigenvalues"])

    for count in sample_counts:
        sub_samples = full_samples[:count]
        sub_z = z0 + sub_samples @ N.T
        sub_spectra = solver.reconstructor.reconstruct(sub_z)
        
        stats = AmbiguityMetrics.compute_sample_statistics(
            sub_samples, sub_z, sub_spectra
        )
        c_alpha = np.array(stats["alpha_space"]["centroid"])
        cov_alpha = np.array(stats["alpha_space"]["covariance_matrix"])
        eigvals_alpha = np.array(stats["alpha_space"]["eigenvalues"])

        # Convergence metrics against reference:
        centroid_drift = float(np.linalg.norm(c_alpha - ref_centroid))
        cov_diff_rel = float(np.linalg.norm(cov_alpha - ref_cov, 'fro') / np.linalg.norm(ref_cov, 'fro'))
        eigval_rel_diff = float(np.linalg.norm(eigvals_alpha - ref_eigvals) / np.linalg.norm(ref_eigvals))

        # Sample extrema at 550 nm
        s_min_550 = float(np.min(sub_spectra[:, target_wave_idx]))
        s_max_550 = float(np.max(sub_spectra[:, target_wave_idx]))
        min_gap = float(s_min_550 - exact_bound_550["exact_min"])
        max_gap = float(exact_bound_550["exact_max"] - s_max_550)

        entry = {
            "sample_count": count,
            "centroid_norm": float(np.linalg.norm(c_alpha)),
            "centroid_drift_vs_ref": centroid_drift,
            "covariance_rel_diff_vs_ref": cov_diff_rel,
            "eigenvalue_rel_diff_vs_ref": eigval_rel_diff,
            "leading_eigenvalues": eigvals_alpha[:3].tolist(),
            "spread_score": stats["spread_score"]["alpha_trace"]["value"],
            "volume_scale_estimate": stats["volume_scores"]["ellipsoid_scale_a_det"]["value"],
            "sample_550nm_min": s_min_550,
            "sample_550nm_max": s_max_550,
            "exact_550nm_min": exact_bound_550["exact_min"],
            "exact_550nm_max": exact_bound_550["exact_max"],
            "undercoverage_550nm_min": min_gap,
            "undercoverage_550nm_max": max_gap
        }
        results_by_count.append(entry)

        print(f"\nSample Count = {count}:")
        print(f"  Centroid drift vs ref: {centroid_drift:.4e}")
        print(f"  Covariance relative error: {cov_diff_rel:.4f}")
        print(f"  Leading eigenvalues: {[round(v, 4) for v in eigvals_alpha[:3]]}")
        print(f"  Spread score: {stats['spread_score']['alpha_trace']['value']:.4f}")
        print(f"  Volume scale: {stats['volume_scores']['ellipsoid_scale_a_det']['value']:.4f}")
        print(f"  550nm Sample Range: [{s_min_550:.4f}, {s_max_550:.4f}] (Gap to exact: min={min_gap:.4f}, max={max_gap:.4f})")

    out_file = os.path.join(artifacts_dir, "sampling_stability_results.json")
    with open(out_file, "w") as fp:
        json.dump({
            "target_rgb": rgb,
            "ground_truth_lp_550nm": exact_bound_550,
            "stability_study": results_by_count
        }, fp, indent=2)
    print(f"\nSaved stability results to: {out_file}")


if __name__ == '__main__':
    run_sampling_stability()
