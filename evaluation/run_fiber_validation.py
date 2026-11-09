import os
import sys
import json
import csv
import time
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.metamer_solver import MetamerSolver
from src.validation import Validator
from src.hit_and_run import HitAndRunSampler
from src.spectral_bounds import SpectralBoundsCalculator
from src.ambiguity_metrics import AmbiguityMetrics


def run_evaluation():
    artifacts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "artifacts"))
    solver = MetamerSolver(artifacts_dir)
    validator = Validator(artifacts_dir=artifacts_dir)

    # 4 distinct evaluation targets
    targets = [
        {
            "id": "target_1_known_feasible",
            "name": "Known Feasible Neutral",
            "rgb": [0.5, 0.5, 0.5],
            "expected_feasible": True
        },
        {
            "id": "target_2_interior_feasible",
            "name": "Interior Feasible Chromatic",
            "rgb": [0.3, 0.35, 0.4],
            "expected_feasible": True
        },
        {
            "id": "target_3_near_boundary",
            "name": "Near-Boundary Low Luminance",
            "rgb": [0.18, 0.18, 0.18],
            "expected_feasible": True
        },
        {
            "id": "target_4_outside_gamut",
            "name": "Outside-Gamut Saturated Green",
            "rgb": [0.05, 0.95, 0.05],
            "expected_feasible": False
        }
    ]

    all_results = []
    summary_rows = []

    print("=" * 80)
    print("RUNNING REPRODUCIBLE SPECTRAL FIBER RESEARCH EVALUATION PIPELINE")
    print("=" * 80)

    for target in targets:
        rgb = target["rgb"]
        t_id = target["id"]
        t_name = target["name"]
        print(f"\nEvaluating {t_name} (RGB = {rgb})...")
        
        t0 = time.time()
        fiber = solver.construct_fiber(rgb)
        feasibility_time = time.time() - t0

        record = {
            "target_id": t_id,
            "target_name": t_name,
            "rgb": rgb,
            "feasible": fiber["feasible"],
            "feasibility_time_sec": float(feasibility_time)
        }

        if not fiber["feasible"]:
            print(f"  Result: Outside prior-induced gamut (cleanly rejected in {feasibility_time:.2f}s).")
            record["solver_status"] = "Infeasible (no point satisfies equality and polytope constraints)"
            record["fiber_dimension"] = None
            record["constraint_rank"] = None
            record["anchor_residual"] = None
            record["max_polytope_violation"] = None
            record["valid_samples"] = 0
            record["mean_rgb_error"] = None
            record["exact_bounds"] = None
            record["ambiguity_metrics"] = None
            record["runtime_sec"] = float(feasibility_time)
            
            summary_rows.append({
                "Target": t_name,
                "RGB": str(rgb),
                "Feasible": False,
                "Rank": "-",
                "Fiber Dim": "-",
                "Anchor Res": "-",
                "Max Viol": "-",
                "Valid Pts": 0,
                "RGB RMS Err": "-",
                "Spread Score": "-",
                "Vol Scale": "-",
                "Runtime (s)": f"{feasibility_time:.2f}"
            })
            all_results.append(record)
            continue

        # Feasible target analysis
        z0 = fiber["z0"]
        G = fiber["G"]
        h = fiber["h"]
        N = fiber["N"]
        d_f = fiber["fiber_dimension"]
        rank = fiber["constraint_rank"]

        # 1. Anchor residual
        c_target = np.array(rgb)
        b_eq = c_target - solver.camera_offset
        anchor_residual = float(np.linalg.norm(solver.A_camera @ z0 - b_eq))
        anchor_ineq_viol = float(np.max(solver.A_red @ z0 - solver.b_red))

        # 2. Multi-chain Hit-and-Run sampling
        t_sample_start = time.time()
        sampler = HitAndRunSampler(G, h)
        chains_alpha, diags = sampler.sample_multiple_chains(
            num_chains=2,
            num_samples_per_chain=500,
            burn_in=50,
            thinning=2,
            base_seed=42
        )
        sampling_time = time.time() - t_sample_start
        
        # Combine chains for aggregated statistics
        combined_alpha = np.vstack(chains_alpha)
        combined_z = z0 + combined_alpha @ N.T
        combined_spectra = solver.reconstructor.reconstruct(combined_z)

        # 3. Point validation
        val = validator.validate_points(rgb, combined_z)
        
        # 4. Multi-chain agreement
        chains_z = [z0 + ch @ N.T for ch in chains_alpha]
        chain_agreement = Validator.compare_chains(chains_z)

        # 5. Exact LP spectral bounds across key wavelengths (400 to 700 nm)
        t_bounds_start = time.time()
        key_wave_indices = [0, 5, 10, 15, 20, 25, 30] # 400, 450, 500, 550, 600, 650, 700 nm
        exact_bounds = solver.bounds_calc.compute_all_bounds(z0, h, wave_indices=key_wave_indices)
        bounds_comparisons = SpectralBoundsCalculator.compare_with_samples(exact_bounds, combined_spectra)
        bounds_time = time.time() - t_bounds_start

        # 6. Ambiguity metrics
        metrics = AmbiguityMetrics.compute_sample_statistics(
            alpha_samples=combined_alpha,
            z_samples=combined_z,
            spectra_samples=combined_spectra
        )

        total_time = time.time() - t0

        print(f"  Result: Feasible! Fiber Dim = {d_f}, Rank = {rank}")
        print(f"  Anchor Residual = {anchor_residual:.4e}, Max Viol = {val['max_violation']:.4e}")
        alpha_spread = metrics.get('spread_score', {}).get('alpha_trace', {}).get('value', 0.0)
        vol_scale = metrics.get('volume_scores', {}).get('ellipsoid_scale_a_det', {}).get('value', 0.0)
        print(f"  Alpha Spread Score = {alpha_spread:.4f}")
        print(f"  Volume Scale Est = {vol_scale:.4f}")
        print(f"  Subspace Chain Agreement Distance = {chain_agreement['mean_subspace_distance']:.4f}")
        print(f"  Total Runtime: {total_time:.2f}s (Sampling: {sampling_time:.2f}s, Bounds: {bounds_time:.2f}s)")

        record.update({
            "solver_status": "Optimal feasible fiber constructed",
            "fiber_dimension": d_f,
            "constraint_rank": rank,
            "anchor_residual": anchor_residual,
            "anchor_inequality_violation": anchor_ineq_viol,
            "max_polytope_violation": val["max_violation"],
            "valid_samples": val["valid_points"],
            "total_samples": val["total_points"],
            "percent_valid": val["percent_valid"],
            "mean_rgb_error": val["mean_error"],
            "rms_rgb_error": val["rms_error"],
            "max_rgb_error": val["max_error"],
            "chain_agreement": chain_agreement,
            "ambiguity_metrics": metrics,
            "exact_bounds_comparisons": bounds_comparisons,
            "runtime_breakdown": {
                "feasibility_sec": float(feasibility_time),
                "sampling_sec": float(sampling_time),
                "bounds_sec": float(bounds_time),
                "total_sec": float(total_time)
            }
        })

        summary_rows.append({
            "Target": t_name,
            "RGB": str(rgb),
            "Feasible": True,
            "Rank": rank,
            "Fiber Dim": d_f,
            "Anchor Res": f"{anchor_residual:.2e}",
            "Max Viol": f"{val['max_violation']:.2e}",
            "Valid Pts": f"{val['valid_points']}/{val['total_points']}",
            "RGB RMS Err": f"{val['rms_error']:.2e}",
            "Spread Score": f"{alpha_spread:.4f}",
            "Vol Scale": f"{vol_scale:.4f}",
            "Runtime (s)": f"{total_time:.2f}"
        })
        all_results.append(record)

    # Save JSON results
    json_path = os.path.join(artifacts_dir, "fiber_validation_results.json")
    with open(json_path, "w") as fp:
        json.dump(all_results, fp, indent=2)
    print(f"\nSaved structured JSON results to: {json_path}")

    # Save CSV summary
    csv_path = os.path.join(artifacts_dir, "fiber_validation_results.csv")
    fieldnames = list(summary_rows[0].keys())
    with open(csv_path, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Saved summary CSV results to: {csv_path}")

    print("\n" + "=" * 80)
    print("RESEARCH EVALUATION SUMMARY TABLE")
    print("=" * 80)
    col_fmt = "{:<28} {:<15} {:<10} {:<6} {:<10} {:<12} {:<12} {:<10} {:<10}"
    print(col_fmt.format("Target", "RGB", "Feasible", "Dim", "Anchor Res", "RGB RMS Err", "Spread", "Vol Scale", "Time (s)"))
    print("-" * 115)
    for r in summary_rows:
        print(col_fmt.format(
            r["Target"][:26],
            r["RGB"],
            str(r["Feasible"]),
            str(r["Fiber Dim"]),
            r["Anchor Res"],
            r["RGB RMS Err"],
            r["Spread Score"],
            r["Vol Scale"],
            r["Runtime (s)"]
        ))
    print("=" * 80)


if __name__ == '__main__':
    run_evaluation()
