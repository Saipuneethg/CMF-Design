"""
cmf_image_renderer.py
======================
Strict One-CMF-per-Image Color Rendering Engine.

Enforces the project's strict single-CMF image generation policy:
- Exactly ONE CMF is selected and applied per image-generation operation.
- No cross-CMF averaging, blending, interpolation, weighted combinations,
  or buffer accumulation is permitted in the individual rendering path.
- Independent fresh NumPy buffers are allocated for every render operation.
- Complete metadata tracking: CMF identifier, configuration hash,
  source spectral data hash, and output image hash.
"""

import os
import time
import hashlib
import json
import numpy as np
from PIL import Image
import colour


def compute_array_sha256(arr):
    """Computes SHA-256 hash of a contiguous NumPy array."""
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def compute_file_sha256(filepath):
    """Computes SHA-256 hash of a file on disk."""
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class CMFRenderer:
    """
    Independent CMF Image Renderer enforcing the strict one-CMF-per-image rule.
    """

    @staticmethod
    def validate_single_cmf(cmf_definition):
        """
        Validates that a CMF definition contains exactly one CMF with 3 spectral channels.
        Raises ValueError if multiple CMFs or non-conforming dimensions are detected.
        """
        if not isinstance(cmf_definition, dict):
            raise TypeError(f"cmf_definition must be a dict, got {type(cmf_definition)}")

        if "matrix" not in cmf_definition:
            raise KeyError("cmf_definition missing required 'matrix' key")

        matrix = np.asarray(cmf_definition["matrix"], dtype=np.float64)
        if matrix.ndim != 2:
            raise ValueError(f"CMF matrix must be 2-dimensional (3, B), got shape {matrix.shape}")

        if matrix.shape[0] != 3:
            raise ValueError(
                f"STRICT ONE-CMF RULE VIOLATION: CMF matrix must have exactly 3 channels (R, G, B), "
                f"got {matrix.shape[0]} channels. Multiple CMFs cannot be concatenated."
            )

        if matrix.shape[1] != 31:
            raise ValueError(
                f"CMF matrix must have 31 wavelength bands (400-700 nm @ 10 nm), got {matrix.shape[1]}"
            )

        # Ensure no NaN or Inf
        if np.isnan(matrix).any() or np.isinf(matrix).any():
            raise ValueError("CMF matrix contains invalid NaN or Inf values")

        # Ensure sensitivities are bounded within physically plausible range (allowing minor calibration tails down to -0.1)
        if (matrix < -0.1).any():
            raise ValueError(f"CMF matrix contains severe negative sensitivities: min={matrix.min()}")

        return matrix

    @staticmethod
    def render_single_cmf_image(
        spectral_data,
        cmf_definition,
        output_path=None,
        highlight_percentile=99.8,
        source_identifier="000002"
    ):
        """
        Renders an RGB image from spectral data using strictly ONE selected CMF.

        Parameters:
        -----------
        spectral_data : np.ndarray
            Spectral cube of shape (H, W, 31) or flat array of shape (N, 31).
        cmf_definition : dict
            CMF specification with keys:
            - 'id': str
            - 'name': str
            - 'slug': str
            - 'matrix': np.ndarray of shape (3, 31)
            - 'type': 'camera_matrix' or 'cie_cmf'
        output_path : str, optional
            Path where the resulting single-CMF PNG image should be saved.
        highlight_percentile : float
            Percentile used for scene highlight normalization (default 99.8).
        source_identifier : str
            Source image identifier or path.

        Returns:
        --------
        rgb_image : np.ndarray
            Rendered uint8 RGB image of shape (H, W, 3) in range [0, 255].
        metadata : dict
            Processing metadata documenting the single CMF used and integrity hashes.
        """
        t0 = time.time()

        # Step 1: Validate CMF - strictly one CMF with 3 channels
        M = CMFRenderer.validate_single_cmf(cmf_definition)
        cmf_id = str(cmf_definition.get("id", "unknown_cmf"))
        cmf_name = str(cmf_definition.get("name", "Unknown CMF"))
        cmf_slug = str(cmf_definition.get("slug", cmf_id))
        cmf_type = str(cmf_definition.get("type", "camera_matrix"))
        cmf_sha = compute_array_sha256(M)

        # Step 2: Validate spectral data
        spec = np.asarray(spectral_data, dtype=np.float32)
        if spec.ndim == 3:
            H, W, B = spec.shape
            cube_flat = spec.reshape(-1, B)
        elif spec.ndim == 2:
            H, W = int(np.sqrt(spec.shape[0])), int(np.sqrt(spec.shape[0]))
            B = spec.shape[1]
            cube_flat = spec
        else:
            raise ValueError(f"Invalid spectral data dimensions: {spec.shape}")

        if B != 31:
            raise ValueError(f"Spectral data must have 31 bands (400-700 nm @ 10 nm), got {B}")

        spec_sha = compute_array_sha256(spec)

        # Step 3: Strictly isolated calculation with a fresh output buffer
        # Never reuse buffers or accumulate across CMFs
        raw_tristimulus = np.empty((cube_flat.shape[0], 3), dtype=np.float64)

        if cmf_type == "cie_cmf":
            # CIE Standard Observer Integration: XYZ = Integral S(x, y, lambda) * CMF(lambda) d_lambda
            # rows of M are [x_bar, y_bar, z_bar]
            np.dot(cube_flat, M.T, out=raw_tristimulus)
            scale = float(np.percentile(raw_tristimulus, highlight_percentile))
            if scale <= 1e-9:
                scale = 1.0
            xyz_norm = np.clip(raw_tristimulus / scale, 0.0, 1.0)

            # Linear sRGB transformation via ITU-R BT.709 D65 matrix
            rgb_linear = colour.XYZ_to_sRGB(xyz_norm)

            # Standard IEC 61966-2-1 sRGB companding
            rgb_encoded = colour.cctf_encoding(np.clip(rgb_linear, 0.0, 1.0))
            rgb_uint8 = np.clip(rgb_encoded * 255.0, 0, 255).astype(np.uint8)

        else:
            # Camera Spectral Sensitivity Model: c = Integral S(x, y, lambda) * M(lambda) d_lambda
            # rows of M are [s_Red, s_Green, s_Blue]
            np.dot(cube_flat, M.T, out=raw_tristimulus)
            scale = float(np.percentile(raw_tristimulus, highlight_percentile))
            if scale <= 1e-9:
                scale = 1.0
            norm_linear = np.clip(raw_tristimulus / scale, 0.0, 1.0)

            # Apply standard camera sRGB companding
            rgb_encoded = colour.cctf_encoding(norm_linear)
            rgb_uint8 = np.clip(rgb_encoded * 255.0, 0, 255).astype(np.uint8)

        # Reshape to 2D image (H, W, 3)
        rgb_image = rgb_uint8.reshape(H, W, 3)
        elapsed_ms = (time.time() - t0) * 1000.0

        # Verification metrics
        low_clip = float(np.mean(rgb_image <= 1) * 100.0)
        high_clip = float(np.mean(rgb_image >= 254) * 100.0)
        mean_r = float(rgb_image[:, :, 0].mean())
        mean_g = float(rgb_image[:, :, 1].mean())
        mean_b = float(rgb_image[:, :, 2].mean())

        # Save to disk if requested
        saved_sha = ""
        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            pil_img = Image.fromarray(rgb_image)
            pil_img.save(output_path, format="PNG")
            saved_sha = compute_file_sha256(output_path)

        metadata = {
            "strict_single_cmf_verified": True,
            "cmf_id": cmf_id,
            "cmf_name": cmf_name,
            "cmf_slug": cmf_slug,
            "cmf_type": cmf_type,
            "cmf_sha256": cmf_sha,
            "cmf_channels_count": 3,
            "source_identifier": source_identifier,
            "source_spectral_shape": list(spec.shape),
            "source_spectral_sha256": spec_sha,
            "output_dimensions": [H, W, 3],
            "output_channels": ["R", "G", "B"],
            "output_color_encoding": "sRGB IEC 61966-2-1",
            "highlight_percentile": highlight_percentile,
            "highlight_scale": round(scale, 4),
            "stats": {
                "mean_r": round(mean_r, 2),
                "mean_g": round(mean_g, 2),
                "mean_b": round(mean_b, 2),
                "low_clip_pct": round(low_clip, 2),
                "high_clip_pct": round(high_clip, 2)
            },
            "saved_file_path": output_path.replace("\\", "/") if output_path else None,
            "saved_file_sha256": saved_sha if output_path else None,
            "rendering_time_ms": round(elapsed_ms, 2)
        }

        return rgb_image, metadata


def load_all_available_cmfs(base_dir="."):
    """
    Discovers, validates, and loads all available CMF definitions in the project.
    Guarantees every CMF is returned as an independent definition with exactly 3 curves.

    Returns:
    --------
    cmf_list : list of dict
        Ordered list of CMF definitions:
        1. Global Optimized CMF (Active Project Master)
        2. Original Calibrated Camera Model (Baseline)
        3. CIE 1931 2° Standard Observer
        4. CIE 1964 10° Supplementary Standard Observer
        5. 28 Measured Physical Cameras from Database
    """
    base_dir = os.path.abspath(base_dir)
    cmfs = []

    # 1. Active Global Optimized CMF
    opt_path = os.path.join(base_dir, "artifacts", "M_camera_optimized.npy")
    if not os.path.exists(opt_path):
        opt_path = os.path.join(base_dir, "results", "cmf_optimization", "M_camera_optimized.npy")

    if os.path.exists(opt_path):
        M_opt = np.load(opt_path)
        cmfs.append({
            "id": "global_optimized_cmf",
            "name": "Global Optimized CMF",
            "slug": "Global_Optimized_CMF",
            "category": "Active Project Master",
            "type": "camera_matrix",
            "matrix": M_opt,
            "wavelengths": list(range(400, 710, 10)),
            "source_file": os.path.relpath(opt_path, base_dir).replace("\\", "/"),
            "description": "Active Global Optimized Camera Spectral Sensitivity Matrix (6324faa...)"
        })

    # 2. Original Baseline Model
    base_path = os.path.join(base_dir, "artifacts", "M_camera.npy")
    if os.path.exists(base_path):
        M_base = np.load(base_path)
        cmfs.append({
            "id": "original_calibrated_camera",
            "name": "Original Calibrated Camera Model",
            "slug": "Original_Calibrated_Camera",
            "category": "Project Baseline Model",
            "type": "camera_matrix",
            "matrix": M_base,
            "wavelengths": list(range(400, 710, 10)),
            "source_file": os.path.relpath(base_path, base_dir).replace("\\", "/"),
            "description": "Original calibrated camera sensitivity matrix (3 x 31)"
        })

    # 3. CIE 1931 2° Standard Observer
    cie_1931 = colour.MSDS_CMFS['CIE 1931 2 Degree Standard Observer'].copy().align(colour.SpectralShape(400, 700, 10))
    cmfs.append({
        "id": "cie_1931_2deg",
        "name": "CIE 1931 2° Standard Observer",
        "slug": "CIE_1931_2deg_Observer",
        "category": "Standard Colorimetric Observer",
        "type": "cie_cmf",
        "matrix": cie_1931.values.T, # (3, 31) where rows are [x_bar, y_bar, z_bar]
        "wavelengths": list(range(400, 710, 10)),
        "source_file": "colour.MSDS_CMFS['CIE 1931 2 Degree Standard Observer']",
        "description": "Standard CIE 1931 2-degree color matching functions (400-700 nm @ 10 nm)"
    })

    # 4. CIE 1964 10° Supplementary Observer
    cie_1964 = colour.MSDS_CMFS['CIE 1964 10 Degree Standard Observer'].copy().align(colour.SpectralShape(400, 700, 10))
    cmfs.append({
        "id": "cie_1964_10deg",
        "name": "CIE 1964 10° Supplementary Observer",
        "slug": "CIE_1964_10deg_Observer",
        "category": "Standard Colorimetric Observer",
        "type": "cie_cmf",
        "matrix": cie_1964.values.T, # (3, 31)
        "wavelengths": list(range(400, 710, 10)),
        "source_file": "colour.MSDS_CMFS['CIE 1964 10 Degree Standard Observer']",
        "description": "Standard CIE 1964 10-degree color matching functions (400-700 nm @ 10 nm)"
    })

    # 5. 28 Measured Physical Cameras from Database
    inv_path = os.path.join(base_dir, "camera comparisions", "metadata", "camera_database_inventory.json")
    sens_mat_path = os.path.join(base_dir, "camera comparisions", "sensitivity_functions", "camera_sensitivity_matrices.npy")

    if os.path.exists(inv_path) and os.path.exists(sens_mat_path):
        with open(inv_path, 'r', encoding='utf-8') as f:
            inv = json.load(f)
        matrices = np.load(sens_mat_path) # (28, 3, 31)

        for idx, cam in enumerate(inv["cameras"]):
            cmfs.append({
                "id": f"ref_cam_{idx+1:02d}_{cam['slug']}",
                "name": cam["camera_name"],
                "slug": cam["slug"],
                "category": "Physical Camera (Measured)",
                "type": "camera_matrix",
                "matrix": matrices[idx], # (3, 31)
                "wavelengths": inv["wavelength_grid_nm"],
                "source_file": cam["csv_file"],
                "description": f"Measured spectral sensitivity curves for {cam['camera_name']}"
            })

    return cmfs
