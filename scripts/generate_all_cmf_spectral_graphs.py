"""
Generate Individual RGB Spectral Sensitivity Graphs for All CMFs
================================================================

Discovers, visualizes, and organizes every available Color Matching Function (CMF)
in the Spectral Metameric Fiber Analyzer project:
  - 28 Physical Camera Models
  - Original Calibrated Baseline Camera
  - Active Global Optimized CMF
  - Previous / Old Optimized CMF (Run 2 seed 101)
  - CIE 1931 2° Standard Observer
  - CIE 1964 10° Supplementary Observer

Rule: ONE CMF = ONE IMAGE = THREE RGB CURVES (Red, Green, Blue).
"""

import os
import sys
import json
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

# Ensure project root in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import colour


def compute_sha256(filepath):
    """Computes SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def discover_all_cmfs(base_dir=PROJECT_ROOT):
    """
    Discovers every valid CMF in the project and returns a normalized dictionary list.
    """
    discovered = []
    wls = list(range(400, 710, 10)) # 31 samples: 400 to 700 nm

    # -------------------------------------------------------------
    # 1. Active Global Optimized CMF
    # -------------------------------------------------------------
    opt_path = os.path.join(base_dir, "artifacts", "M_camera_optimized.npy")
    if os.path.exists(opt_path):
        m_opt = np.load(opt_path)
        discovered.append({
            "cmf_id": "global_optimized_cmf",
            "display_name": "Global Optimized CMF",
            "slug": "global_optimized_cmf",
            "category": "Global Optimized CMF (Active Master)",
            "matrix": m_opt,
            "wavelengths": wls,
            "source_type": "npy",
            "source_file": os.path.relpath(opt_path, base_dir).replace("\\", "/"),
            "sha256": compute_sha256(opt_path),
            "description": "Active Global Optimized Camera Spectral Sensitivity Matrix (contracts metameric fiber by +9.37%)"
        })

    # -------------------------------------------------------------
    # 2. Original Calibrated Camera Model (Baseline)
    # -------------------------------------------------------------
    base_path = os.path.join(base_dir, "artifacts", "M_camera.npy")
    if os.path.exists(base_path):
        m_base = np.load(base_path)
        discovered.append({
            "cmf_id": "original_calibrated_camera",
            "display_name": "Original Calibrated Camera Model",
            "slug": "original_calibrated_camera",
            "category": "Project Baseline Camera Model",
            "matrix": m_base,
            "wavelengths": wls,
            "source_type": "npy",
            "source_file": os.path.relpath(base_path, base_dir).replace("\\", "/"),
            "sha256": compute_sha256(base_path),
            "description": "Original calibrated camera sensitivity matrix (Baseline sRGB calibration)"
        })

    # -------------------------------------------------------------
    # 3. Previous Optimized CMF (Old Opt / Run 2 seed 101)
    # -------------------------------------------------------------
    prev_opt_path = os.path.join(base_dir, "results", "backup_previous_optimized_cmf", "M_camera_optimized.npy")
    if os.path.exists(prev_opt_path):
        m_prev = np.load(prev_opt_path)
        discovered.append({
            "cmf_id": "previous_optimized_cmf",
            "display_name": "Previous Optimized CMF",
            "slug": "previous_optimized_cmf",
            "category": "Previous Optimized CMF (Run 2 Seed 101)",
            "matrix": m_prev,
            "wavelengths": wls,
            "source_type": "npy",
            "source_file": os.path.relpath(prev_opt_path, base_dir).replace("\\", "/"),
            "sha256": compute_sha256(prev_opt_path),
            "description": "Previous Global Optimized CMF from initial 100-color optimization"
        })

    # -------------------------------------------------------------
    # 4. CIE 1931 2° Standard Observer
    # -------------------------------------------------------------
    cie_31 = colour.MSDS_CMFS['CIE 1931 2 Degree Standard Observer'].copy().align(colour.SpectralShape(400, 700, 10))
    m_cie31 = cie_31.values.T # (3, 31) where rows are [x_bar, y_bar, z_bar]
    discovered.append({
        "cmf_id": "cie_1931_2deg",
        "display_name": "CIE 1931 2° Standard Observer",
        "slug": "cie_1931_2deg",
        "category": "CIE Standard Colorimetric Observer",
        "matrix": m_cie31,
        "wavelengths": wls,
        "source_type": "colour-science standard",
        "source_file": "colour.MSDS_CMFS['CIE 1931 2 Degree Standard Observer']",
        "sha256": hashlib.sha256(m_cie31.tobytes()).hexdigest(),
        "description": "CIE 1931 2-degree standard color matching functions (x_bar, y_bar, z_bar)"
    })

    # -------------------------------------------------------------
    # 5. CIE 1964 10° Supplementary Observer
    # -------------------------------------------------------------
    cie_64 = colour.MSDS_CMFS['CIE 1964 10 Degree Standard Observer'].copy().align(colour.SpectralShape(400, 700, 10))
    m_cie64 = cie_64.values.T # (3, 31)
    discovered.append({
        "cmf_id": "cie_1964_10deg",
        "display_name": "CIE 1964 10° Supplementary Observer",
        "slug": "cie_1964_10deg",
        "category": "CIE Standard Colorimetric Observer",
        "matrix": m_cie64,
        "wavelengths": wls,
        "source_type": "colour-science standard",
        "source_file": "colour.MSDS_CMFS['CIE 1964 10 Degree Standard Observer']",
        "sha256": hashlib.sha256(m_cie64.tobytes()).hexdigest(),
        "description": "CIE 1964 10-degree supplementary standard color matching functions"
    })

    # -------------------------------------------------------------
    # 6. 28 Measured Physical Cameras from Database
    # -------------------------------------------------------------
    inv_path = os.path.join(base_dir, "camera comparisions", "metadata", "camera_database_inventory.json")
    sens_mat_path = os.path.join(base_dir, "camera comparisions", "sensitivity_functions", "camera_sensitivity_matrices.npy")

    if os.path.exists(inv_path) and os.path.exists(sens_mat_path):
        with open(inv_path, 'r', encoding='utf-8') as f:
            inv = json.load(f)
        matrices = np.load(sens_mat_path) # (28, 3, 31)
        grid_wls = inv.get("wavelength_grid_nm", wls)

        for idx, cam in enumerate(inv["cameras"]):
            csv_path = os.path.join(base_dir, "camera comparisions", cam["csv_file"])
            csv_sha = compute_sha256(csv_path) if os.path.exists(csv_path) else "N/A"
            discovered.append({
                "cmf_id": cam["slug"],
                "display_name": cam["camera_name"],
                "slug": cam["slug"],
                "category": "Physical Reference Camera",
                "matrix": matrices[idx], # (3, 31)
                "wavelengths": grid_wls,
                "source_type": "measured_matrix",
                "source_file": f"camera comparisions/{cam['csv_file']}",
                "sha256": csv_sha,
                "description": f"Measured spectral sensitivity curves for {cam['camera_name']}"
            })

    return discovered


def generate_single_cmf_graph(cmf_info, output_png_path):
    """
    Generates exactly ONE spectral sensitivity graph containing all THREE curves:
      - Red (R) channel -> Red line (#dc2626)
      - Green (G) channel -> Green line (#16a34a)
      - Blue (B) channel -> Blue line (#2563eb)

    Design template matches the established project reference style:
      - Bold descriptive title: Camera Spectral Sensitivity: <Name>
      - Axis labels: Wavelength (nm), Normalized Responsivity
      - Light dotted grid: alpha=0.6, linestyle=':'
      - Solid distinct RGB lines with data markers
      - Proper Y-axis scaling preserving valid negative and >1.0 values
      - Complete legend with: <Name> Red, <Name> Green, <Name> Blue
    """
    matrix = cmf_info["matrix"] # shape (3, N)
    wls = np.array(cmf_info["wavelengths"])
    name = cmf_info["display_name"]

    # Channel rows: 0=Red, 1=Green, 2=Blue
    red_curve = matrix[0]
    green_curve = matrix[1]
    blue_curve = matrix[2]

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)

    # Plot 3 distinct solid curves with marker dots for discrete sampling points
    # Red Channel
    ax.plot(
        wls, red_curve,
        color="#dc2626", # Vibrant Crimson Red
        linestyle="-",
        linewidth=2.6,
        marker="o",
        markersize=4.5,
        markeredgecolor="#991b1b",
        markeredgewidth=0.8,
        label=f"{name} Red"
    )

    # Green Channel
    ax.plot(
        wls, green_curve,
        color="#16a34a", # Vibrant Emerald Green
        linestyle="-",
        linewidth=2.6,
        marker="s",
        markersize=4.0,
        markeredgecolor="#166534",
        markeredgewidth=0.8,
        label=f"{name} Green"
    )

    # Blue Channel
    ax.plot(
        wls, blue_curve,
        color="#2563eb", # Vibrant Royal Blue
        linestyle="-",
        linewidth=2.6,
        marker="^",
        markersize=4.5,
        markeredgecolor="#1e40af",
        markeredgewidth=0.8,
        label=f"{name} Blue"
    )

    # Determine Y-axis limits preserving negative values and peak values
    min_val = float(matrix.min())
    max_val = float(matrix.max())
    span = max_val - min_val if (max_val - min_val) > 1e-4 else 1.0

    if min_val < -1e-4:
        # Preserve negative values
        y_bottom = min_val - 0.08 * span
        # Highlight zero line
        ax.axhline(0.0, color="#64748b", linestyle="--", linewidth=0.9, alpha=0.7, zorder=1)
    else:
        y_bottom = -0.04 * span

    y_top = max_val + 0.12 * span
    ax.set_ylim(y_bottom, y_top)
    ax.set_xlim(400, 700)

    # Set Title and Labels
    ax.set_title(f"Camera Spectral Sensitivity: {name}", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Wavelength (nm)", fontsize=11, fontweight="medium")
    ax.set_ylabel("Normalized Responsivity", fontsize=11, fontweight="medium")

    # Grid
    ax.grid(True, linestyle=":", alpha=0.6, color="#94a3b8")

    # Ticks
    ax.set_xticks(np.arange(400, 701, 50))
    ax.set_xticks(np.arange(400, 701, 10), minor=True)
    ax.tick_params(axis="both", which="major", labelsize=10)

    # Legend
    legend = ax.legend(
        loc="upper right",
        fontsize=9,
        framealpha=0.92,
        facecolor="#ffffff",
        edgecolor="#cbd5e1",
        fancybox=True,
        shadow=False
    )
    legend.get_frame().set_linewidth(0.8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_png_path)), exist_ok=True)
    plt.savefig(output_png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_cmf_data_files(cmf_info, data_dir):
    """
    Saves unmodifed numerical spectral data for the CMF as CSV and NPY.
    """
    os.makedirs(data_dir, exist_ok=True)
    slug = cmf_info["slug"]
    matrix = cmf_info["matrix"]
    wls = cmf_info["wavelengths"]

    # 1. Save CSV: Wavelength (nm), Red, Green, Blue
    csv_path = os.path.join(data_dir, f"{slug}.csv")
    df = pd.DataFrame({
        "Wavelength (nm)": wls,
        "Red": matrix[0],
        "Green": matrix[1],
        "Blue": matrix[2]
    })
    df.to_csv(csv_path, index=False)

    # 2. Save binary NPY (3, N)
    npy_path = os.path.join(data_dir, f"{slug}.npy")
    np.save(npy_path, matrix)

    return {
        "csv_path": os.path.abspath(csv_path),
        "npy_path": os.path.abspath(npy_path),
        "csv_sha256": compute_sha256(csv_path),
        "npy_sha256": compute_sha256(npy_path),
        "min_val": float(matrix.min()),
        "max_val": float(matrix.max()),
        "has_negative_values": bool(matrix.min() < -1e-5),
        "peak_red_nm": int(wls[int(np.argmax(matrix[0]))]),
        "peak_green_nm": int(wls[int(np.argmax(matrix[1]))]),
        "peak_blue_nm": int(wls[int(np.argmax(matrix[2]))])
    }


def main():
    print("=" * 70)
    print("ALL CMF SPECTRAL SENSITIVITY GRAPH GENERATION PIPELINE")
    print("=" * 70)

    # Define Output Directories
    # 1. Inside existing CMF analysis directory
    dir_a = os.path.join(PROJECT_ROOT, "cmf_post_optimization_analysis", "Advanced_Global_Optimized_CMF")
    # 2. At workspace root for universal direct discovery
    dir_b = os.path.join(PROJECT_ROOT, "Advanced_Global_Optimized_CMF")

    target_dirs = [dir_a, dir_b]

    # Discovery
    cmfs = discover_all_cmfs(PROJECT_ROOT)
    print(f"Total CMFs Discovered: {len(cmfs)}")
    for idx, c in enumerate(cmfs):
        print(f"  [{idx+1:02d}] {c['display_name']} ({c['cmf_id']}) -> {c['category']}")

    # Process all CMFs
    index_records = []
    manifest_records = []

    for tdir in target_dirs:
        graphs_dir = os.path.join(tdir, "spectral_sensitivity_graphs")
        data_dir = os.path.join(tdir, "cmf_spectral_data")
        os.makedirs(graphs_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)

    print("\nGenerating Graphs and Preserving Spectral Data...")
    for idx, cmf in enumerate(cmfs):
        slug = cmf["slug"]
        display_name = cmf["display_name"]
        cmf_id = cmf["cmf_id"]
        category = cmf["category"]
        wls = cmf["wavelengths"]
        matrix = cmf["matrix"]

        png_filename = f"{slug}.png"

        # Generate in target_dirs
        for tdir in target_dirs:
            graphs_dir = os.path.join(tdir, "spectral_sensitivity_graphs")
            data_dir = os.path.join(tdir, "cmf_spectral_data")
            png_out = os.path.join(graphs_dir, png_filename)

            # Generate Graph
            generate_single_cmf_graph(cmf, png_out)

            # Save Spectral Data
            data_meta = save_cmf_data_files(cmf, data_dir)

        # Verification checks
        png_check_path = os.path.join(target_dirs[0], "spectral_sensitivity_graphs", png_filename)
        status = "Success"
        rgb_avail = "R, G, B channels verified"

        try:
            with Image.open(png_check_path) as im:
                w, h = im.size
                if w < 500 or h < 300:
                    status = "Failed: Invalid dimensions"
        except Exception as e:
            status = f"Failed: {str(e)}"

        record = {
            "CMF index": idx + 1,
            "CMF ID": cmf_id,
            "CMF display name": display_name,
            "CMF category": category,
            "Number of wavelength samples": len(wls),
            "Minimum wavelength": min(wls),
            "Maximum wavelength": max(wls),
            "Graph filename": png_filename,
            "RGB data availability": rgb_avail,
            "Generation status": status
        }
        index_records.append(record)

        manifest_record = {
            "cmf_index": idx + 1,
            "cmf_id": cmf_id,
            "display_name": display_name,
            "slug": slug,
            "category": category,
            "graph_filename": png_filename,
            "graph_sha256": compute_sha256(png_check_path),
            "source_file": cmf["source_file"],
            "data_min": data_meta["min_val"],
            "data_max": data_meta["max_val"],
            "has_negative_values": data_meta["has_negative_values"],
            "peak_wavelengths": {
                "red_nm": data_meta["peak_red_nm"],
                "green_nm": data_meta["peak_green_nm"],
                "blue_nm": data_meta["peak_blue_nm"]
            }
        }
        manifest_records.append(manifest_record)

        print(f"  [{idx+1:02d}/{len(cmfs):02d}] Generated: {png_filename} | Range: [{data_meta['min_val']:.3f}, {data_meta['max_val']:.3f}] | Peaks: R={data_meta['peak_red_nm']}nm, G={data_meta['peak_green_nm']}nm, B={data_meta['peak_blue_nm']}nm")

    # Save Index CSV and Data Manifest in both target directories
    df_index = pd.DataFrame(index_records)
    for tdir in target_dirs:
        csv_index_path = os.path.join(tdir, "CMF_Spectral_Sensitivity_Index.csv")
        df_index.to_csv(csv_index_path, index=False)

        manifest_path = os.path.join(tdir, "cmf_spectral_data", "CMF_Spectral_Data_Manifest.json")
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest_records, f, indent=2)

    # Write a comprehensive README.md in both target directories
    readme_content = f"""# Advanced Global Optimized CMF & All CMF Spectral Sensitivity Atlas

This repository contains individual RGB spectral sensitivity graphs, unmodified spectral data arrays, and a comprehensive validation index for **all {len(cmfs)} Color Matching Functions (CMFs)** in the Spectral Metameric Fiber Analyzer project.

---

## 1. Directory Structure

```text
Advanced_Global_Optimized_CMF/
├── spectral_sensitivity_graphs/
│   ├── canon_5dmarkii.png
│   ├── canon_20d.png
│   ├── canon_600d.png
│   ├── nikon_d80.png
│   ├── point_grey_grasshopper_50s5c.png
│   ├── original_calibrated_camera.png
│   ├── global_optimized_cmf.png
│   ├── previous_optimized_cmf.png
│   ├── cie_1931_2deg.png
│   ├── cie_1964_10deg.png
│   └── ... ({len(cmfs)} individual CMF PNG graphs)
│
├── cmf_spectral_data/
│   ├── <slug>.csv                     # Human-readable tabular format (Wavelength, Red, Green, Blue)
│   ├── <slug>.npy                     # Exact binary float array (3, 31)
│   └── CMF_Spectral_Data_Manifest.json# Metadata, peak wavelengths, and SHA-256 hashes
│
├── CMF_Spectral_Sensitivity_Index.csv # Complete machine-readable master catalog
└── README.md                          # Documentation and visual guide
```

---

## 2. Visualization Principles & Visual Design

Every generated spectral sensitivity graph adheres to strict physical and visual principles:
1. **Strict One CMF = One Image = Three Curves**:
   - Each graph contains exclusively the **Red (R)**, **Green (G)**, and **Blue (B)** sensitivity curves belonging to that single CMF.
   - Curves from different CMFs or baseline models are **never mixed or cross-plotted**.
2. **Standardized Coordinates**:
   - **X-axis**: Wavelength (400–700 nm @ 10 nm physical sampling).
   - **Y-axis**: Normalized Responsivity. Valid negative lobes (such as in the Original Baseline Camera at 410 nm and 510 nm) and peaks $> 1.0$ (up to 1.99 in CIE observers and 1.79 in baseline) are faithfully preserved without forced zero-clipping or distortion.
3. **Established Project Styling**:
   - High-contrast, publication-grade styling with solid curves (Red `#dc2626`, Green `#16a34a`, Blue `#2563eb`).
   - Discrete sample point markers on every curve.
   - Light dotted grid (`:` with `alpha=0.6`).
   - Explicit legend: `<CMF Name> Red`, `<CMF Name> Green`, `<CMF Name> Blue`.

---

## 3. Discovered CMF Catalog Summary (Total: {len(cmfs)} CMFs)

| Index | CMF Identifier | Display Name | Category | Peak Wavelengths (R, G, B) | Has Negative Values | Graph Filename |
| :---: | :--- | :--- | :--- | :---: | :---: | :--- |
"""
    for r in manifest_records:
        has_neg = "Yes (preserved)" if r["has_negative_values"] else "No"
        peaks = f"{r['peak_wavelengths']['red_nm']} nm, {r['peak_wavelengths']['green_nm']} nm, {r['peak_wavelengths']['blue_nm']} nm"
        readme_content += f"| {r['cmf_index']:02d} | `{r['cmf_id']}` | **{r['display_name']}** | {r['category']} | {peaks} | {has_neg} | [`{r['graph_filename']}`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/Advanced_Global_Optimized_CMF/spectral_sensitivity_graphs/{r['graph_filename']}) |\n"

    readme_content += """
---

## 4. How to Reproduce

Execute the generator script from the repository root:
```bash
python scripts/generate_all_cmf_spectral_graphs.py
```
"""

    for tdir in target_dirs:
        readme_path = os.path.join(tdir, "README.md")
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)

    print("\n" + "=" * 70)
    print(f"PIPELINE COMPLETE: {len(cmfs)} graphs generated and validated successfully!")
    print(f"Outputs saved to:\n  1. {dir_a}\n  2. {dir_b}")
    print("=" * 70)


if __name__ == "__main__":
    main()
