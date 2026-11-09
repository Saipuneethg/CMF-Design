"""
parse_camera_database.py
========================
Parses the 28-camera spectral sensitivity database text file:
- Validates 33 raw wavelength samples (400 - 720 nm)
- Trims indices 0-30 to retain exactly 31 samples (400 - 700 nm)
- Exports individual camera CSV files
- Exports combined database CSV file
- Exports (28, 3, 31) NumPy array of sensitivity matrices
- Exports detailed metadata and validation inventory
- Exports/copies references for original and optimized models
"""

import os
import sys
import json
import shutil
import numpy as np
import pandas as pd

def parse_database(source_txt_path, base_out_dir="camera comparisions"):
    print("=" * 80)
    print("PARSING AND VALIDATING CAMERA SENSITIVITY DATABASE")
    print("=" * 80)
    print(f"Source file:      {source_txt_path}")
    print(f"Output directory: {base_out_dir}")
    
    sens_dir = os.path.join(base_out_dir, "sensitivity_functions")
    meta_dir = os.path.join(base_out_dir, "metadata")
    orig_dir = os.path.join(base_out_dir, "original_model")
    opt_dir = os.path.join(base_out_dir, "optimized_model")
    logs_dir = os.path.join(base_out_dir, "logs")
    
    for d in [sens_dir, meta_dir, orig_dir, opt_dir, logs_dir]:
        os.makedirs(d, exist_ok=True)
        
    with open(source_txt_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
        
    if len(lines) % 4 != 0:
        raise ValueError(f"Expected multiple of 4 lines (Name, R, G, B), got {len(lines)} lines.")
        
    num_cameras = len(lines) // 4
    print(f"Detected {num_cameras} camera entries in source file.")
    
    raw_wavelengths = np.linspace(400, 720, 33)
    target_wavelengths = np.linspace(400, 700, 31)
    
    camera_matrices = []
    camera_inventory = []
    combined_rows = []
    
    parsing_log = []
    parsing_log.append(f"Source file: {source_txt_path}")
    parsing_log.append(f"Total lines parsed: {len(lines)}")
    parsing_log.append(f"Cameras detected: {num_cameras}")
    parsing_log.append(f"Raw wavelength grid: 400 - 720 nm (33 samples, 10 nm step)")
    parsing_log.append(f"Target wavelength grid: 400 - 700 nm (31 samples, 10 nm step)")
    parsing_log.append("-" * 60)
    
    for idx in range(num_cameras):
        line_idx = idx * 4
        cam_name = lines[line_idx].strip()
        
        try:
            r_raw = [float(x) for x in lines[line_idx + 1].split()]
            g_raw = [float(x) for x in lines[line_idx + 2].split()]
            b_raw = [float(x) for x in lines[line_idx + 3].split()]
        except Exception as e:
            raise ValueError(f"Numeric parse error for camera '{cam_name}': {e}")
            
        # Validation checks
        if len(r_raw) != 33 or len(g_raw) != 33 or len(b_raw) != 33:
            raise ValueError(
                f"Camera '{cam_name}' does not have 33 raw samples per channel: "
                f"R={len(r_raw)}, G={len(g_raw)}, B={len(b_raw)}"
            )
            
        r_raw = np.array(r_raw, dtype=np.float64)
        g_raw = np.array(g_raw, dtype=np.float64)
        b_raw = np.array(b_raw, dtype=np.float64)
        
        # Check non-negativity and finiteness
        if not (np.all(np.isfinite(r_raw)) and np.all(np.isfinite(g_raw)) and np.all(np.isfinite(b_raw))):
            raise ValueError(f"Camera '{cam_name}' contains non-finite values.")
        if np.any(r_raw < 0) or np.any(g_raw < 0) or np.any(b_raw < 0):
            raise ValueError(f"Camera '{cam_name}' contains negative sensitivity values.")
            
        # Trim to 400 - 700 nm (first 31 bands, exclude 710 and 720 nm)
        r_31 = r_raw[:31]
        g_31 = g_raw[:31]
        b_31 = b_raw[:31]
        
        M_cam = np.vstack([r_31, g_31, b_31])  # (3, 31)
        camera_matrices.append(M_cam)
        
        # Clean filename slug
        clean_slug = cam_name.replace(" ", "_").replace("-", "_").lower()
        csv_filename = f"{clean_slug}_sensitivity.csv"
        csv_path = os.path.join(sens_dir, csv_filename)
        
        # Individual CSV
        df_cam = pd.DataFrame({
            "wavelength_nm": target_wavelengths.astype(int),
            "red": r_31,
            "green": g_31,
            "blue": b_31
        })
        df_cam.to_csv(csv_path, index=False)
        
        # Combined rows
        for w_idx, wl in enumerate(target_wavelengths):
            combined_rows.append({
                "camera_name": cam_name,
                "camera_id": idx + 1,
                "wavelength_nm": int(wl),
                "red": r_31[w_idx],
                "green": g_31[w_idx],
                "blue": b_31[w_idx]
            })
            
        # Record inventory metadata
        cam_info = {
            "camera_id": idx + 1,
            "camera_name": cam_name,
            "slug": clean_slug,
            "csv_file": csv_filename,
            "source_samples_per_channel": 33,
            "retained_samples_per_channel": 31,
            "wavelength_min_nm": 400,
            "wavelength_max_nm": 700,
            "wavelength_step_nm": 10,
            "channel_order": ["red", "green", "blue"],
            "normalization_status": "Independently peak-normalized in source dataset",
            "red_max": float(np.max(r_31)),
            "red_peak_wavelength_nm": int(target_wavelengths[np.argmax(r_31)]),
            "green_max": float(np.max(g_31)),
            "green_peak_wavelength_nm": int(target_wavelengths[np.argmax(g_31)]),
            "blue_max": float(np.max(b_31)),
            "blue_peak_wavelength_nm": int(target_wavelengths[np.argmax(b_31)]),
            "validation_passed": True,
            "parsing_warnings": []
        }
        camera_inventory.append(cam_info)
        parsing_log.append(f"[{idx+1:02d}/28] {cam_name:30s} -> Saved {csv_filename} (31 bands, peaks: R={cam_info['red_peak_wavelength_nm']}nm, G={cam_info['green_peak_wavelength_nm']}nm, B={cam_info['blue_peak_wavelength_nm']}nm)")

    # Save combined CSV
    comb_df = pd.DataFrame(combined_rows)
    comb_csv_path = os.path.join(sens_dir, "all_cameras_sensitivities.csv")
    comb_df.to_csv(comb_csv_path, index=False)
    print(f"Saved combined sensitivities CSV: {comb_csv_path} ({len(comb_df)} rows)")
    
    # Save NumPy array: shape (28, 3, 31)
    matrices_arr = np.array(camera_matrices, dtype=np.float64)
    npy_path = os.path.join(sens_dir, "camera_sensitivity_matrices.npy")
    np.save(npy_path, matrices_arr)
    print(f"Saved sensitivity matrices array: {npy_path} (shape: {matrices_arr.shape})")
    
    # Save inventory metadata JSON
    meta_json_path = os.path.join(meta_dir, "camera_database_inventory.json")
    with open(meta_json_path, 'w', encoding='utf-8') as f:
        json.dump({
            "database_title": "28 Measured Camera Spectral Sensitivity Functions Database",
            "number_of_cameras": num_cameras,
            "wavelength_grid_nm": target_wavelengths.astype(int).tolist(),
            "channel_ordering": ["red", "green", "blue"],
            "array_shape": list(matrices_arr.shape),
            "array_dimensions": ["camera_index", "channel (0=R, 1=G, 2=B)", "wavelength (0=400nm .. 30=700nm)"],
            "cameras": camera_inventory
        }, f, indent=2)
    print(f"Saved metadata inventory: {meta_json_path}")
    
    # Export/Copy Original and Optimized Model references
    orig_src = "artifacts/M_camera.npy"
    orig_dst = os.path.join(orig_dir, "M_camera.npy")
    if os.path.exists(orig_src):
        shutil.copy2(orig_src, orig_dst)
        M_orig = np.load(orig_dst)
        orig_meta = {
            "model_name": "Original Calibrated Camera Model",
            "matrix_shape": list(M_orig.shape),
            "source_file": "artifacts/M_camera.npy",
            "wavelength_min_nm": 400,
            "wavelength_max_nm": 700,
            "wavelength_step_nm": 10,
            "channel_ordering": ["red", "green", "blue"],
            "channel_throughputs": [float(np.sum(M_orig[k])) for k in range(3)],
            "mean_throughput": float(np.mean(np.sum(M_orig, axis=1)))
        }
        with open(os.path.join(orig_dir, "model_metadata.json"), 'w', encoding='utf-8') as f:
            json.dump(orig_meta, f, indent=2)
        print(f"Saved original model reference to: {orig_dst}")
        
    opt_src = "artifacts/M_camera_optimized.npy"
    opt_dst = os.path.join(opt_dir, "M_camera_optimized.npy")
    if os.path.exists(opt_src):
        shutil.copy2(opt_src, opt_dst)
        M_opt = np.load(opt_dst)
        opt_meta = {
            "model_name": "Optimized Sensitivity Model (Run_2_seed101)",
            "matrix_shape": list(M_opt.shape),
            "source_file": "artifacts/M_camera_optimized.npy",
            "wavelength_min_nm": 400,
            "wavelength_max_nm": 700,
            "wavelength_step_nm": 10,
            "channel_ordering": ["red", "green", "blue"],
            "channel_throughputs": [float(np.sum(M_opt[k])) for k in range(3)],
            "mean_throughput": float(np.mean(np.sum(M_opt, axis=1)))
        }
        with open(os.path.join(opt_dir, "model_metadata.json"), 'w', encoding='utf-8') as f:
            json.dump(opt_meta, f, indent=2)
        print(f"Saved optimized model reference to: {opt_dst}")
        
    # Write processing log
    log_path = os.path.join(logs_dir, "database_parsing.log")
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(parsing_log) + "\n")
    print(f"Saved parsing log: {log_path}")
    print("=" * 80)
    print("CAMERA SENSITIVITY DATABASE PARSING COMPLETED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    src_file = "camera comparisions/source/camera_sensitivities_database.txt"
    parse_database(src_file)
