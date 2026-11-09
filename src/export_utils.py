import io
import csv
import json
import numpy as np


def convert_numpy_to_python(obj):
    """Recursively converts NumPy numbers and arrays to Python native types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, (np.floating, float)):
        return None if (np.isnan(obj) or np.isinf(obj)) else float(obj)
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_python(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_to_python(v) for v in obj)
    return obj


def export_as_json(data):
    """Formats full calculation results as formatted JSON string."""
    clean_data = convert_numpy_to_python(data)
    return json.dumps(clean_data, indent=2)


def export_as_csv(data):
    """
    Flattens calculation metrics, dimensions, ambiguity scores, and validation
    results into a structured CSV with explicit units and categories.
    """
    clean_data = convert_numpy_to_python(data)
    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow(["Category", "Metric", "Value", "Unit", "Type / Kind", "Description / Formula"])

    # 1. Target and Feasibility
    rgb = clean_data.get("rgb", [None, None, None])
    writer.writerow(["Target", "R", rgb[0], "fraction [0-1]", "exact", "Target Red channel"])
    writer.writerow(["Target", "G", rgb[1], "fraction [0-1]", "exact", "Target Green channel"])
    writer.writerow(["Target", "B", rgb[2], "fraction [0-1]", "exact", "Target Blue channel"])
    writer.writerow(["Target", "Gamut Feasible", clean_data.get("feasible", False), "boolean", "exact", "Prior-induced gamut feasibility"])

    # 2. Dimensions
    dims = clean_data.get("dimension_analysis", {})
    for key, info in dims.items():
        if isinstance(info, dict) and "value" in info:
            writer.writerow([
                "Dimensions",
                info.get("label", key),
                info.get("value"),
                info.get("unit", ""),
                info.get("kind", "exact"),
                info.get("description", "")
            ])

    # 3. Ambiguity Metrics
    amb = clean_data.get("ambiguity_metrics", {})
    spread = amb.get("spread_score", {})
    for key, info in spread.items():
        if isinstance(info, dict):
            writer.writerow([
                "Ambiguity: Spread",
                info.get("label", key),
                f"{info.get('value', 0):.6f}",
                info.get("unit", ""),
                info.get("kind", "sample_estimate"),
                info.get("formula", "")
            ])

    pairwise = amb.get("pairwise_distances", {})
    for key in ["mean", "median", "p95", "max_observed"]:
        info = pairwise.get(key, {})
        if isinstance(info, dict):
            writer.writerow([
                "Ambiguity: Pairwise",
                info.get("label", key),
                f"{info.get('value', 0):.6f}",
                info.get("unit", ""),
                info.get("kind", "sample_estimate"),
                f"Evaluated pairs: {pairwise.get('pairs_evaluated')}"
            ])

    vols = amb.get("volume_scores", {})
    for key, info in vols.items():
        if isinstance(info, dict):
            writer.writerow([
                "Ambiguity: Volume Scale",
                info.get("label", key),
                f"{info.get('value', 0):.6e}",
                info.get("unit", ""),
                info.get("kind", "sample_estimate"),
                info.get("formula", "")
            ])

    comp = amb.get("composite_index", {})
    if comp:
        writer.writerow([
            "Ambiguity: Composite",
            comp.get("label", "Composite Index"),
            f"{comp.get('value', 0):.4f}",
            comp.get("unit", ""),
            comp.get("kind", "constructed_index"),
            comp.get("formula", "")
        ])

    # 4. Validation
    val = clean_data.get("validation", {})
    writer.writerow(["Validation", "Mean RGB Error", f"{val.get('mean_error', 0):.6e}", "RGB units", "measured", "Mean measurement reconstruction error"])
    writer.writerow(["Validation", "Max RGB Error", f"{val.get('max_error', 0):.6e}", "RGB units", "measured", "Max measurement reconstruction error"])
    writer.writerow(["Validation", "RMS RGB Error", f"{val.get('rms_error', 0):.6e}", "RGB units", "measured", "RMS measurement reconstruction error"])
    writer.writerow(["Validation", "Constraint Residual", f"{val.get('constraint_residual', 0):.6e}", "RGB units", "measured", "Anchor constraint residual"])
    writer.writerow(["Validation", "Max Polytope Violation", f"{val.get('max_violation', 0):.6e}", "constraint units", "measured", "Maximum slack violation"])
    writer.writerow(["Validation", "Percent Valid Samples", f"{val.get('percent_valid', 0):.2f}", "%", "measured", "Retained samples within tolerance"])

    # 5. Spectral Bounds
    env = clean_data.get("spectral_envelope", [])
    for item in env:
        wl = item.get("wavelength_nm")
        s_min = item.get("sample_min")
        s_max = item.get("sample_max")
        ex_min = item.get("exact_min")
        ex_max = item.get("exact_max")
        writer.writerow(["Spectral Bounds", f"{wl} nm Sample Min", f"{s_min:.4f}" if s_min is not None else "", "reflectance (0-1)", "empirical sample extrema", "Sample minimum"])
        writer.writerow(["Spectral Bounds", f"{wl} nm Sample Max", f"{s_max:.4f}" if s_max is not None else "", "reflectance (0-1)", "empirical sample extrema", "Sample maximum"])
        if ex_min is not None:
            writer.writerow(["Spectral Bounds", f"{wl} nm Exact LP Min", f"{ex_min:.4f}", "reflectance (0-1)", "exact optimization (HiGHS LP)", "Linear program minimum"])
            writer.writerow(["Spectral Bounds", f"{wl} nm Exact LP Max", f"{ex_max:.4f}", "reflectance (0-1)", "exact optimization (HiGHS LP)", "Linear program maximum"])

    return output.getvalue()


def export_as_markdown(data):
    """Generates a structured research Markdown report."""
    clean_data = convert_numpy_to_python(data)
    rgb = clean_data.get("rgb", [0.5, 0.5, 0.5])
    dims = clean_data.get("dimension_analysis", {})
    amb = clean_data.get("ambiguity_metrics", {})
    val = clean_data.get("validation", {})
    env = clean_data.get("spectral_envelope", [])

    lines = [
        "# Spectral Metameric Fiber Analysis Report",
        "",
        f"**Target RGB:** `[{rgb[0]}, {rgb[1]}, {rgb[2]}]`  ",
        f"**Feasibility:** `{'✓ INSIDE GAMUT' if clean_data.get('feasible') else '✗ OUTSIDE GAMUT'}`  ",
        f"**Samples Retained (N):** `{clean_data.get('n_samples', 'N/A')}` (burn-in: `{clean_data.get('burn_in')}`, thinning: `{clean_data.get('thinning')}`)  ",
        "",
        "## 1. Mathematical Fiber Dimensions",
        "",
        "| Metric | Value | Unit | Kind | Description |",
        "|---|---|---|---|---|"
    ]

    for key, info in dims.items():
        if isinstance(info, dict) and "value" in info:
            lines.append(f"| {info.get('label', key)} | **{info.get('value')}** | {info.get('unit')} | {info.get('kind')} | {info.get('description')} |")

    lines.extend([
        "",
        "## 2. Spectral Ambiguity Quantification",
        "",
        "| Ambiguity Measure | Value | Unit | Formula / Interpretation |",
        "|---|---|---|---|"
    ])

    spread = amb.get("spread_score", {})
    for key, info in spread.items():
        if isinstance(info, dict):
            lines.append(f"| {info.get('label', key)} | **{info.get('value', 0):.5f}** | {info.get('unit')} | `{info.get('formula')}` |")

    pairwise = amb.get("pairwise_distances", {})
    for key in ["mean", "median", "p95", "max_observed"]:
        info = pairwise.get(key, {})
        if isinstance(info, dict):
            lines.append(f"| {info.get('label', key)} | **{info.get('value', 0):.5f}** | {info.get('unit')} | Sample estimate ({pairwise.get('pairs_evaluated')} pairs) |")

    vols = amb.get("volume_scores", {})
    for key, info in vols.items():
        if isinstance(info, dict):
            lines.append(f"| {info.get('label', key)} | **{info.get('value', 0):.5e}** | {info.get('unit')} | `{info.get('formula')}` |")

    comp = amb.get("composite_index", {})
    if comp:
        lines.append(f"| {comp.get('label')} | **{comp.get('value', 0):.4f}** | {comp.get('unit')} | `{comp.get('formula')}` |")

    lines.extend([
        "",
        "## 3. Numerical Validation & Quality Diagnostics",
        "",
        "| Validation Check | Measured Value | Threshold | Status | Units |",
        "|---|---|---|---|---|"
    ])

    checks = val.get("checks", {})
    for key, chk in checks.items():
        lines.append(f"| {chk.get('label', key)} | `{chk.get('measured', 0):.4e}` | `{chk.get('threshold')}` | **{chk.get('status')}** | {chk.get('unit')} |")

    lines.append(f"| Polytope Valid Rate | `{val.get('percent_valid', 100):.1f}%` ({val.get('valid_points')}/{val.get('total_points')}) | `100.0%` | **{val.get('overall_status', 'PASS')}** | % |")

    lines.extend([
        "",
        "## 4. Key Wavelength Bounds Summary",
        "",
        "| Wavelength | Sample Min | Sample Max | Exact LP Min | Exact LP Max | Spectral Span |",
        "|---|---|---|---|---|---|"
    ])

    for item in env[:10]: # First 10 wavelengths
        wl = item.get("wavelength_nm")
        s_min = f"{item.get('sample_min'):.4f}" if item.get('sample_min') is not None else "-"
        s_max = f"{item.get('sample_max'):.4f}" if item.get('sample_max') is not None else "-"
        ex_min = f"{item.get('exact_min'):.4f}" if item.get('exact_min') is not None else "-"
        ex_max = f"{item.get('exact_max'):.4f}" if item.get('exact_max') is not None else "-"
        span = f"{item.get('sample_span'):.4f}" if item.get('sample_span') is not None else "-"
        lines.append(f"| {wl} nm | {s_min} | {s_max} | {ex_min} | {ex_max} | {span} |")

    lines.extend([
        "",
        "---",
        "*Note: The 3D visualization rendering is a projection of the higher-dimensional metameric fiber and is provided for visual exploration only. All metrics are computed in full dimensions without projection loss.*"
    ])

    return "\n".join(lines)
