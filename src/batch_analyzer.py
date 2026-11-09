import os
import sys
import time
import json
import zipfile
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from PIL import Image

from src.metamer_solver import MetamerSolver
from src.validation import Validator
from src.dashboard_renderer import render_color_dashboard
from src.report_generator import (
    generate_csv_summary,
    generate_json_summary,
    generate_txt_summary,
    generate_docx_summary,
    generate_comparison_visualizations
)

class BatchAnalyzer:
    """
    Executes large-scale batch analysis of unique in-gamut RGB colours selected
    from a source image. Generates 1920x1080 dashboard images, summary tables,
    scientific DOCX/TXT reports, and comparison plots.
    """
    def __init__(self, artifacts_dir='artifacts', base_output_dir='random_colours_ambiguity_results'):
        self.artifacts_dir = os.path.abspath(artifacts_dir)
        self.base_output_dir = os.path.abspath(base_output_dir)
        self.solver = MetamerSolver(self.artifacts_dir)
        self.validator = Validator(self.artifacts_dir)
        
        # State tracking for UI integration
        self.current_run = None
        self.lock = threading.Lock()
        
    def run_analysis(
        self,
        source_image_path='Dataset/image.png',
        target_colours=100,
        n_samples=500,
        burn_in=50,
        thinning=2,
        seed=42,
        max_workers=4,
        progress_cb=None
    ):
        """
        Executes complete 100-colour analysis workflow:
        1. Extract unique RGB colours from source image.
        2. Screen for in-gamut feasibility.
        3. Solve metameric fiber, sample MCMC, validate, and render dashboard.
        4. Generate comprehensive reports and comparison visualizations.
        5. Package into ZIP archive.
        """
        t_start = time.time()
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"run_{timestamp_str}"
        run_dir = os.path.join(self.base_output_dir, run_id)
        
        # Create directories
        img_dir = os.path.join(run_dir, "individual_images")
        sum_dir = os.path.join(run_dir, "summary")
        cmp_dir = os.path.join(run_dir, "comparisons")
        spc_dir = os.path.join(run_dir, "spectra")
        bnd_dir = os.path.join(run_dir, "spectral_bounds")
        val_dir = os.path.join(run_dir, "validation")
        
        for d in [run_dir, img_dir, sum_dir, cmp_dir, spc_dir, bnd_dir, val_dir]:
            os.makedirs(d, exist_ok=True)
            
        # Initialize run status
        with self.lock:
            self.current_run = {
                "run_id": run_id,
                "run_dir": run_dir,
                "status": "running",
                "source_image": source_image_path,
                "target_colours": target_colours,
                "n_samples": n_samples,
                "seed": seed,
                "total_unique_found": 0,
                "candidates_tested": 0,
                "selected_in_gamut": 0,
                "completed_count": 0,
                "successful_solves": 0,
                "validated_count": 0,
                "total_valid_samples": 0,
                "percent_complete": 0.0,
                "current_action": "Extracting unique colours from source image...",
                "latest_colour": None,
                "error": None,
                "start_time": t_start
            }

        # Step 1: Load and extract unique colours
        img_abs = os.path.abspath(source_image_path)
        if not os.path.exists(img_abs):
            err_msg = f"Source image not found: {img_abs}"
            with self.lock:
                self.current_run["status"] = "error"
                self.current_run["error"] = err_msg
            raise FileNotFoundError(err_msg)
            
        pil_img = Image.open(img_abs).convert('RGB')
        pixels = np.array(pil_img).reshape(-1, 3)
        unique_colours = np.unique(pixels, axis=0)
        tot_unique = len(unique_colours)
        
        with self.lock:
            self.current_run["total_unique_found"] = tot_unique
            self.current_run["current_action"] = f"Found {tot_unique} unique colours. Shuffling with seed {seed}..."

        # Reproducible random permutation
        rng = np.random.RandomState(seed)
        shuffled_indices = rng.permutation(tot_unique)
        candidates = unique_colours[shuffled_indices]

        # Step 2: Screen and process colours until target_colours is reached
        completed_records = []
        candidates_checked = 0
        in_gamut_found = 0
        
        def _solve_and_render_colour(idx, c_u):
            """Processes a single colour: solve, validate, render, and record."""
            c_n = c_u.astype(np.float64) / 255.0
            t0 = time.time()
            
            # Solve metameric fiber
            res = self.solver.solve(
                rgb=c_n,
                num_points=n_samples,
                burn_in=burn_in,
                thinning=thinning,
                seed=seed + idx,
                compute_bounds=True,
                bound_wavelengths=[15] # 550nm green peak
            )
            
            if not res["feasible"]:
                return None
                
            # Validate points
            z_pts = res["z_points"]
            val_res = self.validator.validate_points(c_n, z_pts)
            
            # Render dashboard image
            img_filename = f"colour_{idx:03d}_analysis.png"
            img_path = os.path.join(img_dir, img_filename)
            render_color_dashboard(
                colour_idx=idx,
                rgb_uint8=c_u,
                rgb_norm=c_n,
                result=res,
                val_result=val_res,
                output_path=img_path,
                dpi=100
            )
            
            # Save spectra and bounds
            spc_path = os.path.join(spc_dir, f"colour_{idx:03d}_spectra.npy")
            recon_spectra = self.solver.reconstructor.reconstruct(z_pts)
            np.save(spc_path, recon_spectra)
            
            bnd_path = os.path.join(bnd_dir, f"colour_{idx:03d}_bounds.json")
            with open(bnd_path, 'w', encoding='utf-8') as f:
                json.dump(res.get("spectral_envelope", []), f, indent=2)
                
            val_path = os.path.join(val_dir, f"colour_{idx:03d}_val.json")
            with open(val_path, 'w', encoding='utf-8') as f:
                json.dump(val_res, f, indent=2)
                
            # Extract summary record
            m = res.get('ambiguity_metrics', {})
            spread = m.get('spread_score', {})
            pw = m.get('pairwise_distances', {})
            vol = m.get('volume_scores', {})
            c_idx = m.get('composite_index', {})
            
            record = {
                "colour_id": f"colour_{idx:03d}",
                "colour_idx": idx,
                "r_uint8": int(c_u[0]),
                "g_uint8": int(c_u[1]),
                "b_uint8": int(c_u[2]),
                "r_norm": float(c_n[0]),
                "g_norm": float(c_n[1]),
                "b_norm": float(c_n[2]),
                "gamut_status": "INSIDE GAMUT",
                "fiber_dimension": res.get("fiber_dimension", 5),
                "constraint_rank": res.get("constraint_rank", 3),
                "spectral_spread_score": float(spread.get('spectral_rms', {}).get('value', 0.0)),
                "coordinate_spread": float(spread.get('alpha_trace', {}).get('value', 0.0)),
                "mean_pairwise_distance": float(pw.get('mean', {}).get('value', 0.0)),
                "median_pairwise_distance": float(pw.get('median', {}).get('value', 0.0)),
                "p95_pairwise_distance": float(pw.get('p95', {}).get('value', 0.0)),
                "max_pairwise_distance": float(pw.get('max_observed', {}).get('value', 0.0)),
                "volume_scale_a_det": float(vol.get('ellipsoid_scale_a_det', {}).get('value', 0.0)),
                "bounding_box_volume": float(vol.get('bounding_box_volume', {}).get('value', 0.0)),
                "composite_ambiguity_index": float(c_idx.get('value', 0.0)),
                "valid_sample_count": val_res.get("valid_points", len(z_pts)),
                "mean_measurement_error": float(val_res.get("mean_error", 0.0)),
                "max_measurement_error": float(val_res.get("max_error", 0.0)),
                "rms_measurement_error": float(val_res.get("rms_error", 0.0)),
                "anchor_equality_residual": float(val_res.get("constraint_residual", 0.0)),
                "max_polytope_slack_violation": float(val_res.get("max_violation", 0.0)),
                "validation_status": val_res.get("overall_status", "PASS"),
                "image_filename": img_filename,
                "process_time_sec": round(time.time() - t0, 2)
            }
            return record

        # We first screen candidates to find exactly target_colours in-gamut RGBs
        with self.lock:
            self.current_run["current_action"] = f"Screening unique candidate colours for in-gamut feasibility..."

        selected_candidates = []
        cand_ptr = 0
        batch_check_size = 24
        
        def _check_feas(c_u):
            c_n = c_u.astype(np.float64) / 255.0
            feas, _ = self.solver.gamut_checker.check_feasibility(c_n)
            return c_u, feas

        while len(selected_candidates) < target_colours and cand_ptr < len(candidates):
            batch_candidates = candidates[cand_ptr : cand_ptr + batch_check_size]
            cand_ptr += len(batch_candidates)
            candidates_checked += len(batch_candidates)
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                check_results = list(executor.map(_check_feas, batch_candidates))
                
            for c_u, feas in check_results:
                if feas:
                    selected_candidates.append(c_u)
                    if len(selected_candidates) == target_colours:
                        break
                        
            with self.lock:
                self.current_run["candidates_tested"] = candidates_checked
                self.current_run["selected_in_gamut"] = len(selected_candidates)
                self.current_run["current_action"] = f"Screened {candidates_checked} candidates; found {len(selected_candidates)}/{target_colours} in-gamut..."

        actual_selected_count = len(selected_candidates)
        print(f"Selection complete: {actual_selected_count} unique in-gamut colours found from {candidates_checked} candidates.")

        # Step 3: Run full analysis and render dashboards for all selected colours
        with self.lock:
            self.current_run["current_action"] = f"Processing {actual_selected_count} colours in parallel (Hit-and-Run N={n_samples}, LP bounds, rendering)..."

        # Worker tasks
        items_to_process = [(idx + 1, c_u) for idx, c_u in enumerate(selected_candidates)]
        
        # We can execute using ThreadPoolExecutor with max_workers
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_solve_and_render_colour, idx, c_u): (idx, c_u) for idx, c_u in items_to_process}
            for fut in as_completed(futures):
                idx, c_u = futures[fut]
                try:
                    record = fut.result()
                    if record is not None:
                        completed_records.append(record)
                        # Sort by colour_idx so order is preserved
                        completed_records.sort(key=lambda r: r["colour_idx"])
                        
                        # Incremental CSV saving
                        csv_path = os.path.join(sum_dir, "ambiguity_summary.csv")
                        generate_csv_summary(completed_records, csv_path)
                        
                        tot_valid = sum(r.get("valid_sample_count", 0) for r in completed_records)
                        tot_pass = sum(1 for r in completed_records if r.get("validation_status") == "PASS")
                        
                        with self.lock:
                            self.current_run["completed_count"] = len(completed_records)
                            self.current_run["successful_solves"] = len(completed_records)
                            self.current_run["validated_count"] = tot_pass
                            self.current_run["total_valid_samples"] = tot_valid
                            self.current_run["percent_complete"] = round((len(completed_records) / actual_selected_count) * 100.0, 1)
                            self.current_run["latest_colour"] = {
                                "colour_id": record["colour_id"],
                                "rgb_uint8": [record["r_uint8"], record["g_uint8"], record["b_uint8"]],
                                "s_spread": record["spectral_spread_score"],
                                "composite_index": record["composite_ambiguity_index"],
                                "image_filename": record["image_filename"]
                            }
                            self.current_run["current_action"] = f"Completed {len(completed_records)}/{actual_selected_count} ({self.current_run['percent_complete']}%) — Latest: {record['colour_id']}"
                            
                        if progress_cb:
                            progress_cb(self.current_run)
                except Exception as e:
                    print(f"Error processing colour #{idx}: {e}")

        # Ensure order
        completed_records.sort(key=lambda r: r["colour_idx"])
        elapsed = time.time() - t_start

        # Step 4: Generate reports and comparison visualizations
        with self.lock:
            self.current_run["current_action"] = "Generating comparison visualizations, reports, and summary packages..."

        metadata = {
            "source_image": source_image_path,
            "run_id": run_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "seed": seed,
            "total_unique_colors": tot_unique,
            "candidates_tested": candidates_checked,
            "total_in_gamut_found": actual_selected_count,
            "requested_samples_per_color": n_samples,
            "burn_in": burn_in,
            "thinning": thinning,
            "elapsed_seconds": round(elapsed, 2)
        }

        # 1. Comparison plots
        generate_comparison_visualizations(completed_records, cmp_dir)
        
        # 1b. CIE 1931 xy Chromaticity Diagram & Ambiguity Heatmap
        try:
            from src.chromaticity_analyzer import ChromaticityAnalyzer
            chrom_analyzer = ChromaticityAnalyzer(self.artifacts_dir)
            chrom_png = os.path.join(cmp_dir, "cie_chromaticity_ambiguity_heatmap.png")
            chrom_analyzer.render_chromaticity_diagram(completed_records, chrom_png)
            chrom_json = os.path.join(cmp_dir, "chromaticity_ambiguity_data.json")
            chrom_analyzer.export_summary_dataset(completed_records, chrom_json)
        except Exception as e:
            print(f"Warning: Failed to generate chromaticity diagram: {e}")

        # 2. CSV summary
        csv_path = os.path.join(sum_dir, "ambiguity_summary.csv")
        generate_csv_summary(completed_records, csv_path)
        
        # 3. JSON summary
        json_path = os.path.join(sum_dir, "ambiguity_summary.json")
        generate_json_summary(metadata, completed_records, json_path)
        
        # 4. TXT summary
        txt_path = os.path.join(sum_dir, "ambiguity_summary.txt")
        generate_txt_summary(metadata, completed_records, txt_path)
        
        # 5. DOCX summary
        docx_path = os.path.join(sum_dir, "ambiguity_summary.docx")
        generate_docx_summary(metadata, completed_records, docx_path, comparisons_dir=cmp_dir)
        
        # 6. Run metadata JSON
        meta_path = os.path.join(run_dir, "run_metadata.json")
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump({
                "metadata": metadata,
                "total_records": len(completed_records),
                "status": "completed"
            }, f, indent=2)

        # Step 5: Create complete ZIP archive
        zip_path = os.path.join(run_dir, "complete_results.zip")
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(run_dir):
                for file in files:
                    if file == "complete_results.zip":
                        continue
                    full_p = os.path.join(root, file)
                    rel_p = os.path.relpath(full_p, run_dir)
                    zipf.write(full_p, rel_p)
        print(f"Created complete ZIP archive: {zip_path}")

        with self.lock:
            self.current_run["status"] = "completed"
            self.current_run["current_action"] = "Analysis complete! All 100 images and reports generated."
            self.current_run["zip_path"] = zip_path

        return {
            "run_id": run_id,
            "run_dir": run_dir,
            "total_completed": len(completed_records),
            "zip_path": zip_path,
            "metadata": metadata
        }

    def get_status(self):
        with self.lock:
            if self.current_run is None:
                return {"status": "idle"}
            return dict(self.current_run)
