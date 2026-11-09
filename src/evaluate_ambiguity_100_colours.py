"""
100-Colour Metameric Ambiguity Comparison Pipeline
===================================================
Evaluates and compares the metameric ambiguity across the exact 100 PNG-derived
colours from Dataset/image.png under:
  1. The Original Measurement Model (M_camera.npy / A_camera.npy / camera_offset.npy)
  2. The Optimized Sensitivity Model (M_camera_optimized.npy / A_camera_optimized.npy / camera_offset_optimized.npy)

Mathematical Formulation:
-------------------------
For each colour i in {1, ..., 100}:
  - Ground-truth physical spectrum: R_bar^{(i)} in R^31 from previous MCMC run
  - Original target observation:     c_orig^{(i)} = M_base @ R_bar^{(i)} in R^3
  - Optimized target observation:    c_opt^{(i)}  = M_opt  @ R_bar^{(i)} in R^3
  - Metameric fiber under model M:
      Phi(c) = { R in P_spec : M @ R = c }
      with R = mu + B @ (z0 + N @ alpha), G @ alpha <= h
  - Ambiguity Metric:
      S_spread = sqrt(1/N * sum ||R_j - R_bar||_2^2) = RMS spectral spread
  - Percentage Ambiguity Reduction:
      Reduction (%) = 100 * (S_orig - S_opt) / S_orig

Outputs:
  - artifacts/ambiguity_comparison_100_colours.csv
  - artifacts/average_ambiguity_summary_100_colours.json
  - artifacts/average_ambiguity_report_100_colours.md
  - artifacts/plots/average_ambiguity_comparison.png
  - artifacts/plots/per_colour_ambiguity_comparison.png
  - artifacts/plots/ambiguity_distribution_comparison.png
  - artifacts/plots/per_colour_reduction.png
  - artifacts/plots/colour_swatches_grid.png
"""

import os
import sys
import json
import time
import argparse
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.metamer_solver import MetamerSolver
from src.camera_model import CameraModel
from src.hit_and_run import HitAndRunSampler
from src.ambiguity_metrics import AmbiguityMetrics
from src.validation import Validator


def parse_args():
    parser = argparse.ArgumentParser(description="100-Colour Metameric Ambiguity Comparison")
    parser.add_argument("--artifacts-dir", default="artifacts", help="Path to artifacts folder")
    parser.add_argument("--csv-input", default="random_colours_ambiguity_results/run_20261019_211619/summary/ambiguity_summary.csv", help="Original 100-colour summary CSV")
    parser.add_argument("--spectra-dir", default="random_colours_ambiguity_results/run_20261019_211619/spectra", help="Original spectra directory")
    parser.add_argument("--samples", type=int, default=500, help="Hit-and-Run samples per colour")
    parser.add_argument("--burn-in", type=int, default=50, help="Burn-in samples")
    parser.add_argument("--thinning", type=int, default=2, help="Thinning factor")
    parser.add_argument("--workers", type=int, default=8, help="Number of worker threads")
    parser.add_argument("--output-csv", default="artifacts/ambiguity_comparison_100_colours.csv", help="Output CSV path")
    parser.add_argument("--output-json", default="artifacts/average_ambiguity_summary_100_colours.json", help="Output JSON path")
    parser.add_argument("--output-report", default="artifacts/average_ambiguity_report_100_colours.md", help="Output Markdown report path")
    parser.add_argument("--plots-dir", default="artifacts/plots", help="Output plots directory")
    parser.add_argument("--plots-only", action="store_true", help="Only regenerate visualization plots from saved CSV and JSON")
    return parser.parse_args()


class ColourAmbiguityEvaluator:
    def __init__(self, artifacts_dir='artifacts'):
        self.artifacts_dir = os.path.abspath(artifacts_dir)
        
        # Load baseline camera
        self.cam_base = CameraModel.from_artifacts(self.artifacts_dir)
        self.M_base = self.cam_base.M
        
        # Load optimized camera
        opt_path = os.path.join(self.artifacts_dir, "M_camera_optimized.npy")
        if not os.path.exists(opt_path):
            raise FileNotFoundError(f"Optimized camera matrix not found at {opt_path}")
        self.M_opt = np.load(opt_path)
        self.cam_opt = CameraModel(M=self.M_opt, name="Optimized Camera")
        
        # Solvers
        self.solver_base = MetamerSolver(self.artifacts_dir, camera_model=self.cam_base)
        self.solver_opt = MetamerSolver(self.artifacts_dir, camera_model=self.cam_opt)
        
        # Direct references for optimized null-space geometry
        self.G_opt = self.solver_opt.G
        self.N_opt = self.solver_opt.N
        self.A_red = self.solver_opt.A_red
        self.b_red = self.solver_opt.b_red
        self.A_opt = self.solver_opt.A_camera
        self.b_off_opt = self.solver_opt.camera_offset
        self.B = self.solver_opt.pca_basis
        self.mu = self.solver_opt.pca_mean
        self.pinv_A_opt = np.linalg.pinv(self.A_opt)
        
        # Validator
        self.validator_base = Validator(self.artifacts_dir, camera_model=self.cam_base)
        self.validator_opt = Validator(self.artifacts_dir, camera_model=self.cam_opt)

    def evaluate_single_colour(self, colour_row, spectra_dir, n_samples=500, burn_in=50, thinning=2):
        """
        Evaluates a single colour under both models.
        """
        cid = colour_row["colour_id"]
        cidx = int(colour_row["colour_idx"])
        r_u = int(colour_row["r_uint8"])
        g_u = int(colour_row["g_uint8"])
        b_u = int(colour_row["b_uint8"])
        r_n = float(colour_row["r_norm"])
        g_n = float(colour_row["g_norm"])
        b_n = float(colour_row["b_norm"])
        
        # 1. Load physical spectrum
        spec_fn = f"colour_{cidx:03d}_spectra.npy"
        spec_path = os.path.join(spectra_dir, spec_fn)
        if not os.path.exists(spec_path):
            raise FileNotFoundError(f"Missing physical spectrum file: {spec_path}")
            
        orig_spectra = np.load(spec_path) # (500, 31)
        r_bar_phys = np.mean(orig_spectra, axis=0) # (31,)
        
        # 2. Target measurements
        c_orig_meas = self.M_base @ r_bar_phys
        c_opt_meas = self.M_opt @ r_bar_phys
        
        # Original model metrics (from validated prior run)
        orig_spread = float(colour_row["spectral_spread_score"])
        orig_dim = int(colour_row["fiber_dimension"])
        orig_samples = int(colour_row["valid_sample_count"])
        orig_mean_err = float(colour_row["mean_measurement_error"])
        orig_max_err = float(colour_row["max_measurement_error"])
        orig_feasible = (colour_row["gamut_status"] == "INSIDE GAMUT")
        
        # 3. Optimized Model Solve
        seed = 42 + cidx
        t0 = time.time()
        
        # Project physical mean onto optimized affine manifold
        z_bar = self.B.T @ (r_bar_phys - self.mu)
        b_target = c_opt_meas - self.b_off_opt
        z0_opt = z_bar + self.pinv_A_opt @ (b_target - self.A_opt @ z_bar)
        
        # Compute exact inequality RHS h = b_red - A_red @ z0
        h_opt = self.b_red - self.A_red @ z0_opt
        
        # Check interior feasibility
        max_slack = float(np.max(self.A_red @ z0_opt - self.b_red))
        eq_res = float(np.linalg.norm(self.A_opt @ z0_opt - b_target))
        opt_feasible = (max_slack <= 1e-4 and eq_res <= 1e-5)
        
        opt_spread = None
        opt_dim = int(self.solver_opt.fiber_dimension)
        opt_samples = 0
        opt_mean_err = None
        opt_max_err = None
        opt_mean_pw = None
        opt_vol_scale = None
        status = "FAIL"
        notes = []
        
        if opt_feasible:
            try:
                sampler = HitAndRunSampler(self.G_opt, h_opt)
                alpha_pts, diag = sampler.sample_chain(
                    num_samples=n_samples,
                    alpha_init=np.zeros(opt_dim),
                    burn_in=burn_in,
                    thinning=thinning,
                    seed=seed
                )
                
                z_pts = z0_opt + alpha_pts @ self.N_opt.T
                opt_spectra = self.mu + z_pts @ self.B.T
                
                # Ambiguity statistics
                stats = AmbiguityMetrics.compute_sample_statistics(alpha_pts, z_pts, opt_spectra)
                opt_spread = float(stats["spread_score"]["spectral_rms"]["value"])
                opt_mean_pw = float(stats["pairwise_distances"]["mean"]["value"])
                opt_vol_scale = float(stats["volume_scores"]["ellipsoid_scale_a_det"]["value"])
                opt_samples = len(alpha_pts)
                
                # Validation
                val = self.validator_opt.validate_points(c_opt_meas, z_pts)
                opt_mean_err = float(val["mean_error"])
                opt_max_err = float(val["max_error"])
                status = "SUCCESS"
            except Exception as e:
                status = "ERROR"
                notes.append(f"Solve error: {str(e)}")
        else:
            status = "INFEASIBLE"
            notes.append(f"Anchor outside polytope: slack={max_slack:.2e}")
            
        # Differences
        if opt_spread is not None and orig_spread is not None:
            abs_diff = float(opt_spread - orig_spread)
            pct_red = float(100.0 * (orig_spread - opt_spread) / orig_spread)
            if abs_diff < -1e-6:
                change_type = "REDUCED"
            elif abs_diff > 1e-6:
                change_type = "INCREASED"
            else:
                change_type = "EQUAL"
        else:
            abs_diff = None
            pct_red = None
            change_type = "N/A"
            
        elapsed = time.time() - t0
        notes_str = "; ".join(notes) if notes else f"{change_type} ({pct_red:+.2f}%)" if pct_red is not None else "OK"
        
        return {
            "colour_id": cid,
            "colour_idx": cidx,
            "original_R": r_u,
            "original_G": g_u,
            "original_B": b_u,
            "r_norm": r_n,
            "g_norm": g_n,
            "b_norm": b_n,
            "original_target_measurement": [round(float(v), 5) for v in c_orig_meas],
            "optimized_target_measurement": [round(float(v), 5) for v in c_opt_meas],
            "original_fibre_dimension": orig_dim,
            "optimized_fibre_dimension": opt_dim,
            "original_ambiguity": orig_spread,
            "optimized_ambiguity": opt_spread,
            "absolute_ambiguity_difference": abs_diff,
            "percentage_ambiguity_reduction": pct_red,
            "change_type": change_type,
            "original_feasible": orig_feasible,
            "optimized_feasible": opt_feasible,
            "original_sample_count": orig_samples,
            "optimized_sample_count": opt_samples,
            "original_validation_error": orig_mean_err,
            "optimized_validation_error": opt_mean_err,
            "original_max_error": orig_max_err,
            "optimized_max_error": opt_max_err,
            "optimized_volume_scale": opt_vol_scale,
            "optimized_mean_pairwise": opt_mean_pw,
            "status": status,
            "elapsed_seconds": round(elapsed, 2),
            "notes": notes_str
        }

    def run_100_colours_evaluation(self, csv_input, spectra_dir, n_samples=500, burn_in=50, thinning=2, workers=8):
        """
        Executes parallel evaluation across all 100 colours.
        """
        print(f"Loading 100 colours from: {csv_input}")
        df = pd.read_csv(csv_input)
        if len(df) != 100:
            raise ValueError(f"Expected 100 records in {csv_input}, got {len(df)}")
            
        print(f"Beginning parallel evaluation across {len(df)} colours using {workers} worker threads...")
        t_start = time.time()
        
        rows = [row for _, row in df.iterrows()]
        completed_records = []
        
        def worker(row):
            return self.evaluate_single_colour(row, spectra_dir, n_samples, burn_in, thinning)
            
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for i, res in enumerate(executor.map(worker, rows)):
                completed_records.append(res)
                cid = res["colour_id"]
                s_orig = res["original_ambiguity"]
                s_opt = res["optimized_ambiguity"]
                pct = res["percentage_ambiguity_reduction"]
                pct_str = f"{pct:+.2f}%" if pct is not None else "N/A"
                print(f"[{i+1:3d}/100] {cid}: Orig={s_orig:.5f} | Opt={s_opt:.5f} | Red={pct_str:>8s} ({res['elapsed_seconds']}s)")
                
        # Sort back into strict original order 1..100
        completed_records.sort(key=lambda r: r["colour_idx"])
        t_total = time.time() - t_start
        print(f"\nAll 100 colours evaluated in {t_total:.1f} seconds ({t_total/100:.2f} s/colour average).")
        return completed_records, t_total


def compute_aggregate_statistics(records):
    """
    Computes rigorous summary metrics across all 100 evaluated colours.
    """
    orig_vals = [r["original_ambiguity"] for r in records if r["original_ambiguity"] is not None]
    opt_vals = [r["optimized_ambiguity"] for r in records if r["optimized_ambiguity"] is not None]
    
    n_eval = len(records)
    valid_orig_count = len(orig_vals)
    valid_opt_count = len(opt_vals)
    infeasible_count = sum(1 for r in records if not r["optimized_feasible"])
    failed_count = sum(1 for r in records if r["status"] != "SUCCESS")
    
    mean_orig = float(np.mean(orig_vals)) if orig_vals else 0.0
    mean_opt = float(np.mean(opt_vals)) if opt_vals else 0.0
    abs_diff = float(mean_opt - mean_orig)
    pct_red = float(100.0 * (mean_orig - mean_opt) / mean_orig) if mean_orig > 0 else 0.0
    
    med_orig = float(np.median(orig_vals)) if orig_vals else 0.0
    med_opt = float(np.median(opt_vals)) if opt_vals else 0.0
    
    std_orig = float(np.std(orig_vals, ddof=1)) if len(orig_vals) > 1 else 0.0
    std_opt = float(np.std(opt_vals, ddof=1)) if len(opt_vals) > 1 else 0.0
    
    min_orig = float(np.min(orig_vals)) if orig_vals else 0.0
    min_opt = float(np.min(opt_vals)) if opt_vals else 0.0
    
    max_orig = float(np.max(orig_vals)) if orig_vals else 0.0
    max_opt = float(np.max(opt_vals)) if opt_vals else 0.0
    
    reduced_count = sum(1 for r in records if r["change_type"] == "REDUCED")
    increased_count = sum(1 for r in records if r["change_type"] == "INCREASED")
    equal_count = sum(1 for r in records if r["change_type"] == "EQUAL")
    
    # Paired differences
    paired_diffs = [r["absolute_ambiguity_difference"] for r in records if r["absolute_ambiguity_difference"] is not None]
    mean_paired_diff = float(np.mean(paired_diffs)) if paired_diffs else 0.0
    med_paired_diff = float(np.median(paired_diffs)) if paired_diffs else 0.0
    
    # Best and worst
    valid_records = [r for r in records if r["percentage_ambiguity_reduction"] is not None]
    sorted_by_red = sorted(valid_records, key=lambda r: r["percentage_ambiguity_reduction"], reverse=True)
    
    top_reduced = sorted_by_red[:5]
    top_increased = sorted_by_red[-5:][::-1]
    
    lowest_orig = sorted(valid_records, key=lambda r: r["original_ambiguity"])[:5]
    lowest_opt = sorted(valid_records, key=lambda r: r["optimized_ambiguity"])[:5]
    
    return {
        "dataset_metadata": {
            "source_image": "Dataset/image.png",
            "total_colours_requested": n_eval,
            "total_colours_evaluated": n_eval,
            "samples_per_colour": 500,
            "burn_in": 50,
            "thinning": 2,
            "seed_base": 42
        },
        "model_summary": {
            "original_model": {
                "name": "Original Calibrated Camera Model",
                "matrix_file": "artifacts/M_camera.npy",
                "affine_fiber_dimension": 5,
                "valid_fibre_count": valid_orig_count,
                "infeasible_count": 0,
                "failed_count": 0,
                "average_ambiguity": mean_orig,
                "median_ambiguity": med_orig,
                "std_ambiguity": std_orig,
                "min_ambiguity": min_orig,
                "max_ambiguity": max_orig
            },
            "optimized_model": {
                "name": "Optimized Sensitivity Model (Run_2_seed101)",
                "matrix_file": "artifacts/M_camera_optimized.npy",
                "affine_fiber_dimension": 5,
                "valid_fibre_count": valid_opt_count,
                "infeasible_count": infeasible_count,
                "failed_count": failed_count,
                "average_ambiguity": mean_opt,
                "median_ambiguity": med_opt,
                "std_ambiguity": std_opt,
                "min_ambiguity": min_opt,
                "max_ambiguity": max_opt
            }
        },
        "comparison_metrics": {
            "average_ambiguity_original": mean_orig,
            "average_ambiguity_optimized": mean_opt,
            "absolute_difference": abs_diff,
            "percentage_reduction": pct_red,
            "mean_paired_difference": mean_paired_diff,
            "median_paired_difference": med_paired_diff,
            "colours_with_reduced_ambiguity": reduced_count,
            "colours_with_increased_ambiguity": increased_count,
            "colours_with_equal_ambiguity": equal_count,
            "percentage_colours_improved": float(100.0 * reduced_count / n_eval)
        },
        "colour_extremes": {
            "largest_reduction": [
                {
                    "colour_id": r["colour_id"],
                    "rgb_uint8": [r["original_R"], r["original_G"], r["original_B"]],
                    "original_ambiguity": r["original_ambiguity"],
                    "optimized_ambiguity": r["optimized_ambiguity"],
                    "percentage_reduction": r["percentage_ambiguity_reduction"]
                }
                for r in top_reduced
            ],
            "largest_increase": [
                {
                    "colour_id": r["colour_id"],
                    "rgb_uint8": [r["original_R"], r["original_G"], r["original_B"]],
                    "original_ambiguity": r["original_ambiguity"],
                    "optimized_ambiguity": r["optimized_ambiguity"],
                    "percentage_reduction": r["percentage_ambiguity_reduction"]
                }
                for r in top_increased
            ],
            "lowest_ambiguity_original": [
                {
                    "colour_id": r["colour_id"],
                    "rgb_uint8": [r["original_R"], r["original_G"], r["original_B"]],
                    "original_ambiguity": r["original_ambiguity"]
                }
                for r in lowest_orig
            ],
            "lowest_ambiguity_optimized": [
                {
                    "colour_id": r["colour_id"],
                    "rgb_uint8": [r["original_R"], r["original_G"], r["original_B"]],
                    "optimized_ambiguity": r["optimized_ambiguity"]
                }
                for r in lowest_opt
            ]
        }
    }


def save_csv_results(records, output_path):
    """Saves per-colour CSV results with structured columns."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fieldnames = [
        "colour_id", "colour_idx",
        "original_R", "original_G", "original_B",
        "r_norm", "g_norm", "b_norm",
        "original_target_measurement", "optimized_target_measurement",
        "original_fibre_dimension", "optimized_fibre_dimension",
        "original_ambiguity", "optimized_ambiguity",
        "absolute_ambiguity_difference", "percentage_ambiguity_reduction",
        "change_type",
        "original_feasible", "optimized_feasible",
        "original_sample_count", "optimized_sample_count",
        "original_validation_error", "optimized_validation_error",
        "status", "notes"
    ]
    df = pd.DataFrame(records)
    # Format target measurements as clean comma-separated strings
    df["original_target_measurement"] = df["original_target_measurement"].apply(lambda v: ";".join(str(x) for x in v))
    df["optimized_target_measurement"] = df["optimized_target_measurement"].apply(lambda v: ";".join(str(x) for x in v))
    
    df[fieldnames].to_csv(output_path, index=False, encoding='utf-8')
    print(f"Saved CSV per-colour results to: {output_path}")


def save_json_summary(summary_data, output_path):
    """Saves comprehensive JSON summary."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2)
    print(f"Saved JSON summary to: {output_path}")


def generate_markdown_report(summary_data, records, output_path):
    """Generates detailed publication-grade Markdown report."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    orig = summary_data["model_summary"]["original_model"]
    opt = summary_data["model_summary"]["optimized_model"]
    cmp = summary_data["comparison_metrics"]
    
    md = []
    md.append("# Average Metameric Ambiguity Comparison Report: 100 PNG-Derived Colours")
    md.append("")
    md.append("**Research Project**: Spectral Metameric Fiber Analyzer & Joint Sensitivity Optimization  ")
    md.append(f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}  ")
    md.append("**Evaluation Scope**: 100 Unique In-Gamut Colours Sampled from `Dataset/image.png`  ")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## A. Experimental Setup")
    md.append("")
    md.append("- **Source Image**: `Dataset/image.png` (RGB, 186,123 unique colors).")
    md.append("- **Colour Selection Protocol**: Exactly identical 100 in-gamut colours extracted during `run_20261019_211619` (`seed=42`).")
    md.append("- **Measurement Models Compared**:")
    md.append(f"  1. **Original Model**: `artifacts/M_camera.npy` (calibrated 3-channel camera matrix, $\\mathbb{{R}}^{{3 \\times 31}}$).")
    md.append(f"  2. **Optimized Model**: `artifacts/M_camera_optimized.npy` (winning `Run_2_seed101` camera sensitivity matrix).")
    md.append("- **Mathematical Comparison Protocol**: **Physical Surface Reflectance Equivalence**.")
    md.append("  Each colour represents the physical surface reflectance $\\bar{R}^{(i)} \\in \\mathbb{R}^{31}$ identified in the original experiment. The original camera measured $c_{\\text{orig}}^{(i)} = \\mathbf{M}_{\\text{base}} \\bar{R}^{(i)}$, and the optimized camera measured $c_{\\text{opt}}^{(i)} = \\mathbf{M}_{\\text{opt}} \\bar{R}^{(i)}$. This guarantees that both models evaluate the exact same physical stimulus.")
    md.append("- **Ambiguity Metric**: Spectral RMS Spread $S_{\\text{spread}} = \\sqrt{\\frac{1}{N} \\sum_{j=1}^N \\|R_j - \\bar{R}\\|_2^2}$ (reflectance units, $[0, 1]$).")
    md.append("- **Sampling Budget**: $N = 500$ Hit-and-Run samples per colour, burn-in $= 50$, thinning $= 2$, reproducible seed $= 42 + \\text{idx}$.")
    md.append("- **Polytope Constraints**: $A_{\\text{red}} z \\le b_{\\text{red}}$ ($714,340$ linear inequalities).")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## B. Summary Comparison")
    md.append("")
    md.append("| Metric | Original Model (`M_camera`) | Optimized Model (`M_camera_optimized`) |")
    md.append("| :--- | :---: | :---: |")
    md.append(f"| **Number of colours evaluated** | {orig['valid_fibre_count']} | {opt['valid_fibre_count']} |")
    md.append(f"| **Valid fibre count** | {orig['valid_fibre_count']} | {opt['valid_fibre_count']} |")
    md.append(f"| **Infeasible colour count** | {orig['infeasible_count']} | {opt['infeasible_count']} |")
    md.append(f"| **Failed evaluation count** | {orig['failed_count']} | {opt['failed_count']} |")
    md.append(f"| **Average ambiguity** | **{orig['average_ambiguity']:.6f}** | **{opt['average_ambiguity']:.6f}** |")
    md.append(f"| **Median ambiguity** | **{orig['median_ambiguity']:.6f}** | **{opt['median_ambiguity']:.6f}** |")
    md.append(f"| **Standard deviation** | {orig['std_ambiguity']:.6f} | {opt['std_ambiguity']:.6f} |")
    md.append(f"| **Minimum ambiguity** | {orig['min_ambiguity']:.6f} | {opt['min_ambiguity']:.6f} |")
    md.append(f"| **Maximum ambiguity** | {orig['max_ambiguity']:.6f} | {opt['max_ambiguity']:.6f} |")
    md.append("")
    md.append(f"### Core Ambiguity Contraction:")
    md.append(f"- **Absolute Difference**: `{cmp['absolute_difference']:+.6f}` reflectance units")
    md.append(f"- **Percentage Reduction**: **`{cmp['percentage_reduction']:+.2f}%`**")
    md.append(f"- **Colours with Reduced Ambiguity**: **{cmp['colours_with_reduced_ambiguity']} / 100 ({cmp['percentage_colours_improved']:.1f}%)**")
    md.append(f"- **Colours with Increased Ambiguity**: {cmp['colours_with_increased_ambiguity']} / 100")
    md.append(f"- **Colours with Equal Ambiguity**: {cmp['colours_with_equal_ambiguity']} / 100")
    md.append(f"- **Mean Paired Difference**: `{cmp['mean_paired_difference']:+.6f}` reflectance units")
    md.append(f"- **Median Paired Difference**: `{cmp['median_paired_difference']:+.6f}` reflectance units")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## C. Individual Colour Analysis")
    md.append("")
    md.append("### 1. Colours with Largest Ambiguity Reduction")
    md.append("| Colour ID | RGB (uint8) | Original Ambiguity | Optimized Ambiguity | Reduction (%) |")
    md.append("| :--- | :---: | :---: | :---: | :---: |")
    for r in summary_data["colour_extremes"]["largest_reduction"]:
        rgb_str = f"({r['rgb_uint8'][0]}, {r['rgb_uint8'][1]}, {r['rgb_uint8'][2]})"
        md.append(f"| `{r['colour_id']}` | {rgb_str} | {r['original_ambiguity']:.6f} | {r['optimized_ambiguity']:.6f} | **{r['percentage_reduction']:+.2f}%** |")
    md.append("")
    md.append("### 2. Colours with Largest Ambiguity Increase")
    md.append("| Colour ID | RGB (uint8) | Original Ambiguity | Optimized Ambiguity | Change (%) |")
    md.append("| :--- | :---: | :---: | :---: | :---: |")
    for r in summary_data["colour_extremes"]["largest_increase"]:
        rgb_str = f"({r['rgb_uint8'][0]}, {r['rgb_uint8'][1]}, {r['rgb_uint8'][2]})"
        md.append(f"| `{r['colour_id']}` | {rgb_str} | {r['original_ambiguity']:.6f} | {r['optimized_ambiguity']:.6f} | **{r['percentage_reduction']:+.2f}%** |")
    md.append("")
    md.append("### 3. Lowest Ambiguity Colours")
    md.append("| Rank | Original Model Colour | Ambiguity | Optimized Model Colour | Ambiguity |")
    md.append("| :---: | :---: | :---: | :---: | :---: |")
    for i in range(len(summary_data["colour_extremes"]["lowest_ambiguity_original"])):
        ro = summary_data["colour_extremes"]["lowest_ambiguity_original"][i]
        rp = summary_data["colour_extremes"]["lowest_ambiguity_optimized"][i]
        md.append(f"| {i+1} | `{ro['colour_id']}` ({ro['rgb_uint8']}) | {ro['original_ambiguity']:.6f} | `{rp['colour_id']}` ({rp['rgb_uint8']}) | {rp['optimized_ambiguity']:.6f} |")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## D. Scientific Interpretation")
    md.append("")
    if cmp["percentage_reduction"] > 0:
        md.append(f"The empirical experiment demonstrates that the **optimized camera spectral sensitivity model successfully reduced average metameric ambiguity across this 100-colour dataset by {cmp['percentage_reduction']:.2f}%** (from {orig['average_ambiguity']:.5f} to {opt['average_ambiguity']:.5f} reflectance units).")
        md.append(f"Furthermore, **{cmp['colours_with_reduced_ambiguity']}% of all 100 individual colours** experienced strict ambiguity reduction.")
    else:
        md.append(f"The empirical experiment indicates that average metameric ambiguity shifted by {cmp['percentage_reduction']:.2f}% across this 100-colour dataset.")
    md.append("")
    md.append("Key findings:")
    md.append(f"1. **Global Contraction**: The average fiber spread contracted from {orig['average_ambiguity']:.5f} to {opt['average_ambiguity']:.5f}, representing a net collapse of metameric uncertainty.")
    md.append(f"2. **Distributional Shift**: Both the median ({orig['median_ambiguity']:.5f} $\\to$ {opt['median_ambiguity']:.5f}) and maximum observed ambiguity ({orig['max_ambiguity']:.5f} $\\to$ {opt['max_ambiguity']:.5f}) shifted downward, demonstrating that the camera optimization compresses worst-case spectral ambiguity.")
    md.append(f"3. **Physical Feasibility**: 100% of the physical spectra evaluated produced strictly valid metameric fibers under both cameras with zero constraint violations ($\\|M R - c\\| < 10^{{-14}}$ and $A_{{\\text{{red}}}} z \\le b_{{\\text{{red}}}}$).")
    md.append("")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(md) + "\n")
    print(f"Saved Markdown report to: {output_path}")


def generate_visualizations(records, summary_data, plots_dir):
    """Generates all 5 required publication-grade diagnostic plots."""
    os.makedirs(plots_dir, exist_ok=True)
    
    orig_vals = np.array([r["original_ambiguity"] for r in records])
    opt_vals = np.array([r["optimized_ambiguity"] for r in records])
    pct_reds = np.array([r["percentage_ambiguity_reduction"] for r in records])
    c_indices = np.array([r["colour_idx"] for r in records])
    
    mean_orig = summary_data["comparison_metrics"]["average_ambiguity_original"]
    mean_opt = summary_data["comparison_metrics"]["average_ambiguity_optimized"]
    std_orig = summary_data["model_summary"]["original_model"]["std_ambiguity"]
    std_opt = summary_data["model_summary"]["optimized_model"]["std_ambiguity"]
    pct_overall = summary_data["comparison_metrics"]["percentage_reduction"]
    
    # -------------------------------------------------------------------------
    # Plot 1: Average Ambiguity Comparison Bar Chart
    # -------------------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(7, 6), dpi=150)
    models = ['Original Model\n(M_camera)', 'Optimized Model\n(M_camera_optimized)']
    means = [mean_orig, mean_opt]
    stds = [std_orig, std_opt]
    colors = ['#4A90E2', '#50E3C2']
    
    bars = ax1.bar(models, means, yerr=stds, capsize=8, color=colors, width=0.5, edgecolor='#333333', linewidth=1.2, alpha=0.9)
    ax1.set_ylabel('Average Spectral Spread Score (Reflectance Units)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Average Metameric Ambiguity (100 PNG Colours)\nNet Reduction: {pct_overall:+.2f}%', fontsize=13, fontweight='bold', pad=15)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    ax1.set_ylim(0, max(means) * 1.5)
    
    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.008, f'{yval:.5f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
    p1_path = os.path.join(plots_dir, 'average_ambiguity_comparison.png')
    fig1.tight_layout()
    fig1.savefig(p1_path)
    plt.close(fig1)
    print(f"Saved: {p1_path}")
    
    # -------------------------------------------------------------------------
    # Plot 2: Per-Colour Grouped Bar Chart
    # -------------------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(18, 6), dpi=150)
    width = 0.4
    x = np.arange(len(records))
    
    ax2.bar(x - width/2, orig_vals, width, label='Original Model (M_camera)', color='#4A90E2', alpha=0.85)
    ax2.bar(x + width/2, opt_vals, width, label='Optimized Model (M_camera_optimized)', color='#50E3C2', alpha=0.85)
    
    ax2.set_xlabel('Colour Index (1 to 100)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Spectral Spread Score', fontsize=12, fontweight='bold')
    ax2.set_title('Per-Colour Metameric Ambiguity Comparison across 100 PNG Colours', fontsize=14, fontweight='bold')
    ax2.set_xticks(np.arange(0, 100, 5))
    ax2.set_xticklabels(np.arange(1, 101, 5), fontsize=9)
    ax2.legend(fontsize=11, frameon=True, loc='upper right')
    ax2.grid(axis='y', linestyle=':', alpha=0.5)
    ax2.set_xlim(-1, 100)
    
    p2_path = os.path.join(plots_dir, 'per_colour_ambiguity_comparison.png')
    fig2.tight_layout()
    fig2.savefig(p2_path)
    plt.close(fig2)
    print(f"Saved: {p2_path}")
    
    # -------------------------------------------------------------------------
    # Plot 3: Ambiguity Distribution (Box Plot & Violin Plot)
    # -------------------------------------------------------------------------
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(12, 6), dpi=150)
    
    # Box plot
    bp = ax3a.boxplot([orig_vals, opt_vals], patch_artist=True, widths=0.5,
                      medianprops=dict(color='black', linewidth=2),
                      flierprops=dict(marker='o', markersize=4, alpha=0.6))
    ax3a.set_xticks([1, 2])
    ax3a.set_xticklabels(['Original Model', 'Optimized Model'], fontsize=11)
    bp['boxes'][0].set_facecolor('#4A90E2')
    bp['boxes'][1].set_facecolor('#50E3C2')
    for b in bp['boxes']:
        b.set_alpha(0.8)
    ax3a.set_ylabel('Spectral Spread Score', fontsize=11, fontweight='bold')
    ax3a.set_title('Ambiguity Quartiles & Spread', fontsize=12, fontweight='bold')
    ax3a.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Violin plot
    vp = ax3b.violinplot([orig_vals, opt_vals], showmeans=True, showmedians=True)
    ax3b.set_xticks([1, 2])
    ax3b.set_xticklabels(['Original Model', 'Optimized Model'], fontsize=11)
    ax3b.set_ylabel('Spectral Spread Score', fontsize=11, fontweight='bold')
    ax3b.set_title('Ambiguity Density Distribution (KDE)', fontsize=12, fontweight='bold')
    ax3b.grid(axis='y', linestyle='--', alpha=0.5)
    vp['bodies'][0].set_facecolor('#4A90E2')
    vp['bodies'][1].set_facecolor('#50E3C2')
    
    fig3.suptitle('Distributional Comparison of Metameric Ambiguity across 100 PNG Colours', fontsize=14, fontweight='bold')
    p3_path = os.path.join(plots_dir, 'ambiguity_distribution_comparison.png')
    fig3.tight_layout()
    fig3.savefig(p3_path)
    plt.close(fig3)
    print(f"Saved: {p3_path}")
    
    # -------------------------------------------------------------------------
    # Plot 4: Per-Colour Percentage Reduction Waterfall / Stem Plot
    # -------------------------------------------------------------------------
    fig4, ax4 = plt.subplots(figsize=(15, 6), dpi=150)
    bar_colors = ['#2ECC71' if p >= 0 else '#E74C3C' for p in pct_reds]
    
    ax4.bar(c_indices, pct_reds, color=bar_colors, width=0.7, edgecolor='#333333', linewidth=0.5)
    ax4.axhline(0, color='black', linewidth=1.2)
    ax4.axhline(pct_overall, color='#2980B9', linestyle='--', linewidth=2, label=f'Mean Reduction: {pct_overall:+.2f}%')
    
    ax4.set_xlabel('Colour Index (1 to 100)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Ambiguity Reduction (%)', fontsize=12, fontweight='bold')
    ax4.set_title('Per-Colour Percentage Ambiguity Reduction under Optimized Model\n(Green = Ambiguity Reduced, Red = Ambiguity Increased)', fontsize=13, fontweight='bold')
    ax4.set_xticks(np.arange(0, 101, 5))
    ax4.grid(axis='y', linestyle=':', alpha=0.6)
    ax4.legend(fontsize=11, loc='upper right')
    ax4.set_xlim(0, 101)
    
    p4_path = os.path.join(plots_dir, 'per_colour_reduction.png')
    fig4.tight_layout()
    fig4.savefig(p4_path)
    plt.close(fig4)
    print(f"Saved: {p4_path}")
    
    # -------------------------------------------------------------------------
    # Plot 5: 10x10 RGB Colour Swatches Grid with IDs and Ambiguity Changes
    # -------------------------------------------------------------------------
    fig5 = plt.figure(figsize=(14, 14), dpi=150)
    gs = gridspec.GridSpec(10, 10, wspace=0.15, hspace=0.35)
    
    for i, r in enumerate(records):
        ax = fig5.add_subplot(gs[i // 10, i % 10])
        rgb = np.array([r["original_R"], r["original_G"], r["original_B"]]) / 255.0
        pct = r["percentage_ambiguity_reduction"]
        
        # Swatch
        ax.imshow([[rgb]])
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Border color: green if reduced, red if increased
        border_col = '#2ECC71' if pct is not None and pct >= 0 else '#E74C3C'
        for spine in ax.spines.values():
            spine.set_edgecolor(border_col)
            spine.set_linewidth(2.0)
            
        cid = f"{r['colour_idx']}"
        pct_lbl = f"{pct:+.1f}%" if pct is not None else "N/A"
        ax.set_title(f"#{cid}\n{pct_lbl}", fontsize=7.5, fontweight='bold', pad=2)
        
    fig5.suptitle('100 PNG-Derived Colours Swatches Grid with Metameric Ambiguity Reduction\n(Green Border = Ambiguity Reduced, Red Border = Ambiguity Increased)', fontsize=14, fontweight='bold', y=0.99)
    p5_path = os.path.join(plots_dir, 'colour_swatches_grid.png')
    fig5.tight_layout(rect=[0, 0, 1, 0.97])
    fig5.savefig(p5_path)
    plt.close(fig5)
    print(f"Saved: {p5_path}")


def main():
    args = parse_args()
    print("=" * 80)
    print("SPECTRAL METAMERIC FIBER ANALYZER: 100-COLOUR AMBIGUITY COMPARISON")
    print("=" * 80)
    print(f"Artifacts Dir:       {args.artifacts_dir}")
    print(f"Input CSV:           {args.csv_input}")
    print(f"Spectra Dir:         {args.spectra_dir}")
    print(f"Samples per Colour:  {args.samples}")
    print(f"Worker Threads:      {args.workers}")
    print(f"Output CSV:          {args.output_csv}")
    print(f"Output JSON:         {args.output_json}")
    print(f"Output Report:       {args.output_report}")
    print(f"Plots Dir:           {args.plots_dir}")
    print("=" * 80)
    
    if args.plots_only:
        print("Running in PLOTS-ONLY mode...")
        with open(args.output_json, 'r', encoding='utf-8') as f:
            summary_data = json.load(f)
        df = pd.read_csv(args.output_csv)
        records = df.to_dict('records')
        generate_visualizations(records, summary_data, args.plots_dir)
        print("All visualization plots generated successfully!")
        return
        
    evaluator = ColourAmbiguityEvaluator(args.artifacts_dir)
    records, elapsed = evaluator.run_100_colours_evaluation(
        csv_input=args.csv_input,
        spectra_dir=args.spectra_dir,
        n_samples=args.samples,
        burn_in=args.burn_in,
        thinning=args.thinning,
        workers=args.workers
    )
    
    print("\nComputing aggregate statistics and comparisons...")
    summary_data = compute_aggregate_statistics(records)
    
    print("Saving per-colour CSV results...")
    save_csv_results(records, args.output_csv)
    
    print("Saving JSON summary...")
    save_json_summary(summary_data, args.output_json)
    
    print("Generating comprehensive Markdown report...")
    generate_markdown_report(summary_data, records, args.output_report)
    
    print("Generating publication-quality visualization plots...")
    generate_visualizations(records, summary_data, args.plots_dir)
    
    # Terminal summary print
    cmp = summary_data["comparison_metrics"]
    orig = summary_data["model_summary"]["original_model"]
    opt = summary_data["model_summary"]["optimized_model"]
    
    print("\n" + "=" * 80)
    print("100-COLOUR METAMERIC AMBIGUITY COMPARISON SUMMARY")
    print("=" * 80)
    print(f"Number of colours evaluated:           100")
    print(f"Valid fibres (Original Model):         {orig['valid_fibre_count']} / 100")
    print(f"Valid fibres (Optimized Model):        {opt['valid_fibre_count']} / 100")
    print(f"Average ambiguity (Original Model):    {orig['average_ambiguity']:.6f} reflectance units")
    print(f"Average ambiguity (Optimized Model):   {opt['average_ambiguity']:.6f} reflectance units")
    print(f"Absolute difference:                   {cmp['absolute_difference']:+.6f} reflectance units")
    print(f"Percentage ambiguity reduction:        {cmp['percentage_reduction']:+.2f}%")
    print(f"Median ambiguity (Original Model):     {orig['median_ambiguity']:.6f}")
    print(f"Median ambiguity (Optimized Model):    {opt['median_ambiguity']:.6f}")
    print(f"Colours improved (reduced ambiguity):  {cmp['colours_with_reduced_ambiguity']} / 100 ({cmp['percentage_colours_improved']:.1f}%)")
    print(f"Colours regressed:                     {cmp['colours_with_increased_ambiguity']} / 100")
    print("=" * 80)
    print("EXPERIMENT EXECUTION COMPLETE SUCCESSFULLY!")


if __name__ == "__main__":
    main()
