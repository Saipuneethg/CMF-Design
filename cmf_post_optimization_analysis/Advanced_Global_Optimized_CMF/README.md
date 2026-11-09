# Advanced Global Optimized CMF & All CMF Spectral Sensitivity Atlas

This repository contains individual RGB spectral sensitivity graphs, unmodified spectral data arrays, and a comprehensive validation index for **all 33 Color Matching Functions (CMFs)** in the Spectral Metameric Fiber Analyzer project.

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
│   └── ... (33 individual CMF PNG graphs)
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

## 3. Discovered CMF Catalog Summary (Total: 33 CMFs)

| Index | CMF Identifier | Display Name | Category | Peak Wavelengths (R, G, B) | Has Negative Values | Graph Filename |
| :---: | :--- | :--- | :--- | :---: | :---: | :--- |
| 01 | `global_optimized_cmf` | **Global Optimized CMF** | Global Optimized CMF (Active Master) | 600 nm, 540 nm, 460 nm | No | [`global_optimized_cmf.png`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/Advanced_Global_Optimized_CMF/spectral_sensitivity_graphs/global_optimized_cmf.png) |
| 02 | `original_calibrated_camera` | **Original Calibrated Camera Model** | Project Baseline Camera Model | 600 nm, 560 nm, 450 nm | Yes (preserved) | [`original_calibrated_camera.png`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/Advanced_Global_Optimized_CMF/spectral_sensitivity_graphs/original_calibrated_camera.png) |
| 03 | `previous_optimized_cmf` | **Previous Optimized CMF** | Previous Optimized CMF (Run 2 Seed 101) | 600 nm, 540 nm, 460 nm | No | [`previous_optimized_cmf.png`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/Advanced_Global_Optimized_CMF/spectral_sensitivity_graphs/previous_optimized_cmf.png) |
| 04 | `cie_1931_2deg` | **CIE 1931 2° Standard Observer** | CIE Standard Colorimetric Observer | 600 nm, 560 nm, 450 nm | No | [`cie_1931_2deg.png`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/Advanced_Global_Optimized_CMF/spectral_sensitivity_graphs/cie_1931_2deg.png) |
| 05 | `cie_1964_10deg` | **CIE 1964 10° Supplementary Observer** | CIE Standard Colorimetric Observer | 600 nm, 560 nm, 450 nm | No | [`cie_1964_10deg.png`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/Advanced_Global_Optimized_CMF/spectral_sensitivity_graphs/cie_1964_10deg.png) |
| 06 | `canon_1dmarkiii` | **Canon 1DMarkIII** | Physical Reference Camera | 590 nm, 530 nm, 460 nm | No | [`canon_1dmarkiii.png`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/Advanced_Global_Optimized_CMF/spectral_sensitivity_graphs/canon_1dmarkiii.png) |
| 07 | `canon_20d` | **Canon 20D** | Physical Reference Camera | 600 nm, 530 nm, 470 nm | No | [`canon_20d.png`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/Advanced_Global_Optimized_CMF/spectral_sensitivity_graphs/canon_20d.png) |
| 08 | `canon_300d` | **Canon 300D** | Physical Reference Camera | 590 nm, 530 nm, 470 nm | No | [`canon_300d.png`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/Advanced_Global_Optimized_CMF/spectral_sensitivity_graphs/canon_300d.png) |
| 09 | `canon_40d` | **Canon 40D** | Physical Reference Camera | 600 nm, 530 nm, 460 nm | No | [`canon_40d.png`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/Advanced_Global_Optimized_CMF/spectral_sensitivity_graphs/canon_40d.png) |
| 10 | `canon_500d` | **Canon 500D** | Physical Reference Camera | 580 nm, 530 nm, 460 nm | No | [`canon_500d.png`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/Advanced_Global_Optimized_CMF/spectral_sensitivity_graphs/canon_500d.png) |
| 11 | `canon_50d` | **Canon 50D** | Physical Reference Camera | 590 nm, 530 nm, 460 nm | No | [`canon_50d.png`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/Advanced_Global_Optimized_CMF/spectral_sensitivity_graphs/canon_50d.png) |
| 12 | `canon_5dmarkii` | **Canon 5DMarkII** | Physical Reference Camera | 600 nm, 510 nm, 460 nm | No | [`canon_5dmarkii.png`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/Advanced_Global_Optimized_CMF/spectral_sensitivity_graphs/canon_5dmarkii.png) |
| 13 | `canon_600d` | **Canon 600D** | Physical Reference Camera | 590 nm, 530 nm, 460 nm | No | [`canon_600d.png`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/Advanced_Global_Optimized_CMF/spectral_sensitivity_graphs/canon_600d.png) |
| 14 | `canon_60d` | **Canon 60D** | Physical Reference Camera | 590 nm, 530 nm, 460 nm | No | [`canon_60d.png`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/Advanced_Global_Optimized_CMF/spectral_sensitivity_graphs/canon_60d.png) |
| 15 | `hasselblad_h2` | **Hasselblad H2** | Physical Reference Camera | 610 nm, 530 nm, 480 nm | No | [`hasselblad_h2.png`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/Advanced_Global_Optimized_CMF/spectral_sensitivity_graphs/hasselblad_h2.png) |
| 16 | `nikon_d3x` | **Nikon D3X** | Physical Reference Camera | 600 nm, 540 nm, 470 nm | No | [`nikon_d3x.png`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/Advanced_Global_Optimized_CMF/spectral_sensitivity_graphs/nikon_d3x.png) |
| 17 | `nikon_d200` | **Nikon D200** | Physical Reference Camera | 590 nm, 530 nm, 450 nm | No | [`nikon_d200.png`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/Advanced_Global_Optimized_CMF/spectral_sensitivity_graphs/nikon_d200.png) |
| 18 | `nikon_d3` | **Nikon D3** | Physical Reference Camera | 590 nm, 530 nm, 460 nm | No | [`nikon_d3.png`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/Advanced_Global_Optimized_CMF/spectral_sensitivity_graphs/nikon_d3.png) |
| 19 | `nikon_d300s` | **Nikon D300s** | Physical Reference Camera | 590 nm, 530 nm, 460 nm | No | [`nikon_d300s.png`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/Advanced_Global_Optimized_CMF/spectral_sensitivity_graphs/nikon_d300s.png) |
| 20 | `nikon_d40` | **Nikon D40** | Physical Reference Camera | 590 nm, 530 nm, 460 nm | No | [`nikon_d40.png`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/Advanced_Global_Optimized_CMF/spectral_sensitivity_graphs/nikon_d40.png) |
| 21 | `nikon_d50` | **Nikon D50** | Physical Reference Camera | 590 nm, 530 nm, 460 nm | No | [`nikon_d50.png`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/Advanced_Global_Optimized_CMF/spectral_sensitivity_graphs/nikon_d50.png) |
| 22 | `nikon_d5100` | **Nikon D5100** | Physical Reference Camera | 590 nm, 530 nm, 450 nm | No | [`nikon_d5100.png`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/Advanced_Global_Optimized_CMF/spectral_sensitivity_graphs/nikon_d5100.png) |
| 23 | `nikon_d700` | **Nikon D700** | Physical Reference Camera | 590 nm, 530 nm, 460 nm | No | [`nikon_d700.png`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/Advanced_Global_Optimized_CMF/spectral_sensitivity_graphs/nikon_d700.png) |
| 24 | `nikon_d80` | **Nikon D80** | Physical Reference Camera | 590 nm, 530 nm, 470 nm | No | [`nikon_d80.png`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/Advanced_Global_Optimized_CMF/spectral_sensitivity_graphs/nikon_d80.png) |
| 25 | `nikon_d90` | **Nikon D90** | Physical Reference Camera | 590 nm, 530 nm, 460 nm | No | [`nikon_d90.png`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/Advanced_Global_Optimized_CMF/spectral_sensitivity_graphs/nikon_d90.png) |
| 26 | `nokia_n900` | **Nokia N900** | Physical Reference Camera | 600 nm, 530 nm, 460 nm | No | [`nokia_n900.png`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/Advanced_Global_Optimized_CMF/spectral_sensitivity_graphs/nokia_n900.png) |
| 27 | `olympus_e_pl2` | **Olympus E-PL2** | Physical Reference Camera | 610 nm, 530 nm, 470 nm | No | [`olympus_e_pl2.png`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/Advanced_Global_Optimized_CMF/spectral_sensitivity_graphs/olympus_e_pl2.png) |
| 28 | `pentax_k_5` | **Pentax K-5** | Physical Reference Camera | 590 nm, 530 nm, 470 nm | No | [`pentax_k_5.png`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/Advanced_Global_Optimized_CMF/spectral_sensitivity_graphs/pentax_k_5.png) |
| 29 | `pentax_q` | **Pentax Q** | Physical Reference Camera | 610 nm, 520 nm, 470 nm | No | [`pentax_q.png`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/Advanced_Global_Optimized_CMF/spectral_sensitivity_graphs/pentax_q.png) |
| 30 | `point_grey_grasshopper_50s5c` | **Point Grey Grasshopper 50S5C** | Physical Reference Camera | 620 nm, 550 nm, 480 nm | No | [`point_grey_grasshopper_50s5c.png`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/Advanced_Global_Optimized_CMF/spectral_sensitivity_graphs/point_grey_grasshopper_50s5c.png) |
| 31 | `point_grey_grasshopper2_14s5c` | **Point Grey Grasshopper2 14S5C** | Physical Reference Camera | 600 nm, 530 nm, 470 nm | No | [`point_grey_grasshopper2_14s5c.png`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/Advanced_Global_Optimized_CMF/spectral_sensitivity_graphs/point_grey_grasshopper2_14s5c.png) |
| 32 | `phase_one` | **Phase One** | Physical Reference Camera | 590 nm, 530 nm, 470 nm | No | [`phase_one.png`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/Advanced_Global_Optimized_CMF/spectral_sensitivity_graphs/phase_one.png) |
| 33 | `sony_nex_5n` | **SONY NEX-5N** | Physical Reference Camera | 590 nm, 530 nm, 450 nm | No | [`sony_nex_5n.png`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/Advanced_Global_Optimized_CMF/spectral_sensitivity_graphs/sony_nex_5n.png) |

---

## 4. How to Reproduce

Execute the generator script from the repository root:
```bash
python scripts/generate_all_cmf_spectral_graphs.py
```
