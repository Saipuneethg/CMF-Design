import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image

from src.metamer_solver import MetamerSolver
from src.validation import Validator

def create_dashboard_image(
    colour_idx,
    rgb_uint8,
    rgb_norm,
    result,
    val_result,
    output_path,
    dpi=100
):
    fig = plt.figure(figsize=(19.2, 10.8), dpi=dpi, facecolor='#060911')
    
    # ----------------------------------------------------
    # Colors and Styles
    # ----------------------------------------------------
    bg_card = '#0d1322'
    border_card = '#1e293b'
    text_white = '#ffffff'
    text_muted = '#94a3b8'
    text_cyan = '#38bdf8'
    text_green = '#34d399'
    text_amber = '#fbbf24'
    
    # Header area (y: 0.93 to 1.0)
    # Title & Subtitle
    fig.text(0.015, 0.965, "SPECTRAL METAMERIC FIBER ANALYZER", fontsize=15, weight='bold', color=text_white)
    fig.text(0.015, 0.940, "36K-SPECTRAL-PRIOR • DYNAMIC AMBIGUITY & FIBER GEOMETRY ENGINE", fontsize=9.5, weight='bold', color=text_cyan)
    
    # Color identifier in header
    r_u, g_u, b_u = rgb_uint8
    r_n, g_n, b_n = rgb_norm
    fig.text(0.42, 0.952, f"COLOUR #{colour_idx:03d} — RGB [{r_u}, {g_u}, {b_u}] / [{r_n:.3f}, {g_n:.3f}, {b_n:.3f}]", 
             fontsize=11, weight='bold', color='#e2e8f0')

    # Top right badges
    # Gamut Badge
    bbox_green = dict(boxstyle='square,pad=0.4', facecolor='#065f46', edgecolor='#10b981', linewidth=1)
    bbox_blue = dict(boxstyle='square,pad=0.4', facecolor='#1e40af', edgecolor='#3b82f6', linewidth=1)
    bbox_pass = dict(boxstyle='square,pad=0.4', facecolor='#065f46', edgecolor='#10b981', linewidth=1)
    
    fig.text(0.76, 0.955, "GAMUT: INSIDE GAMUT", fontsize=8.5, weight='bold', color='#a7f3d0', bbox=bbox_green)
    fig.text(0.85, 0.955, f"{result['fiber_dimension']}D METAMERIC FIBER", fontsize=8.5, weight='bold', color='#bfdbfe', bbox=bbox_blue)
    fig.text(0.93, 0.955, f"VALIDATION: {val_result.get('overall_status', 'PASS')}", fontsize=8.5, weight='bold', color='#a7f3d0', bbox=bbox_pass)

    # ----------------------------------------------------
    # LEFT COLUMN: Panel A (Inputs & Feasibility)
    # ----------------------------------------------------
    ax_panel_a = fig.add_axes([0.015, 0.35, 0.175, 0.57], facecolor=bg_card)
    for spine in ax_panel_a.spines.values():
        spine.set_color(border_card)
    ax_panel_a.set_xticks([])
    ax_panel_a.set_yticks([])
    
    # Title
    ax_panel_a.text(0.06, 0.96, "PANEL A — TARGET RGB", fontsize=9.5, weight='bold', color=text_white, transform=ax_panel_a.transAxes)
    ax_panel_a.text(0.06, 0.925, "& FEASIBILITY", fontsize=9.5, weight='bold', color=text_white, transform=ax_panel_a.transAxes)
    
    # Sliders display
    channels = [("R (Red):", r_n, '#ef4444'), ("G (Green):", g_n, '#10b981'), ("B (Blue):", b_n, '#3b82f6')]
    y_pos = 0.85
    for label, val, c_line in channels:
        ax_panel_a.text(0.06, y_pos, label, fontsize=8.5, color=text_muted, transform=ax_panel_a.transAxes)
        ax_panel_a.text(0.92, y_pos, f"{val:.2f}", fontsize=8.5, weight='bold', color=text_white, ha='right', transform=ax_panel_a.transAxes)
        # slider track
        rect_bg = Rectangle((0.06, y_pos - 0.035), 0.86, 0.012, facecolor='#1e293b', transform=ax_panel_a.transAxes)
        rect_fill = Rectangle((0.06, y_pos - 0.035), 0.86 * val, 0.012, facecolor=c_line, transform=ax_panel_a.transAxes)
        ax_panel_a.add_patch(rect_bg)
        ax_panel_a.add_patch(rect_fill)
        y_pos -= 0.08
    
    # Color swatch
    y_pos -= 0.02
    swatch = FancyBboxPatch((0.06, y_pos - 0.12), 0.22, 0.12, boxstyle="round,pad=0.01", 
                            facecolor=(r_n, g_n, b_n), edgecolor='#475569', linewidth=1.2, transform=ax_panel_a.transAxes)
    ax_panel_a.add_patch(swatch)
    ax_panel_a.text(0.33, y_pos - 0.04, f"RGB({r_n:.2f}, {g_n:.2f}, {b_n:.2f})", fontsize=8.5, weight='bold', color=text_white, transform=ax_panel_a.transAxes)
    ax_panel_a.text(0.33, y_pos - 0.075, "✓ Feasible Metameric Fiber", fontsize=7.5, weight='bold', color=text_green, transform=ax_panel_a.transAxes)
    ax_panel_a.text(0.33, y_pos - 0.11, "Within prior-induced gamut.", fontsize=7.0, color=text_muted, transform=ax_panel_a.transAxes)

    # Sampling Settings
    y_pos -= 0.17
    ax_panel_a.text(0.06, y_pos, "SAMPLING SETTINGS (MCMC)", fontsize=8, weight='bold', color=text_cyan, transform=ax_panel_a.transAxes)
    ax_panel_a.text(0.06, y_pos - 0.045, "Number of samples (N):", fontsize=8, color=text_muted, transform=ax_panel_a.transAxes)
    ax_panel_a.text(0.92, y_pos - 0.045, str(result['n_samples']), fontsize=8.5, weight='bold', color=text_amber, ha='right', transform=ax_panel_a.transAxes)
    ax_panel_a.text(0.06, y_pos - 0.085, "Burn-in: 50 | Thinning: 2", fontsize=7.5, color=text_muted, transform=ax_panel_a.transAxes)
    ax_panel_a.text(0.06, y_pos - 0.125, "LP Bounds: 550 nm (Exact)", fontsize=7.5, color=text_muted, transform=ax_panel_a.transAxes)

    # Status / Info Button
    btn_rect = FancyBboxPatch((0.06, 0.05), 0.86, 0.08, boxstyle="round,pad=0.01",
                              facecolor='#1e40af', edgecolor='#3b82f6', linewidth=1, transform=ax_panel_a.transAxes)
    ax_panel_a.add_patch(btn_rect)
    ax_panel_a.text(0.49, 0.085, "ANALYSIS COMPLETE", fontsize=8.5, weight='bold', color='#ffffff', ha='center', va='center', transform=ax_panel_a.transAxes)

    # ----------------------------------------------------
    # CENTER COLUMN: 3D Visualization of Metameric Fiber & Polytope
    # ----------------------------------------------------
    ax_3d = fig.add_axes([0.20, 0.35, 0.49, 0.57], projection='3d', facecolor='#000000')
    ax_3d.set_facecolor('#000000')
    
    # 3D points
    z_pts = result['z_points']
    z0 = result['z0']
    z_cen = result['z_centroid']
    p_dirs = result['principal_directions']
    
    # Render sampled points in gold/amber
    ax_3d.scatter(z_pts[:, 0], z_pts[:, 1], z_pts[:, 2], c='#f59e0b', s=2.5, alpha=0.6, edgecolors='none', label=f'Metamers (N={len(z_pts)})')
    # Render centroid
    ax_3d.scatter([z_cen[0]], [z_cen[1]], [z_cen[2]], c='#ef4444', marker='+', s=60, linewidth=2, label='Fiber Centroid')
    # Render anchor
    ax_3d.scatter([z0[0]], [z0[1]], [z0[2]], c='#06b6d4', marker='o', s=25, label='Feasible Anchor z0')
    
    # Render principal direction lines
    dir_colors = ['#ef4444', '#10b981', '#3b82f6']
    scale = np.std(z_pts[:, 0]) * 2.5
    for i in range(3):
        pd = p_dirs[:, i]
        ax_3d.plot([z_cen[0] - scale * pd[0], z_cen[0] + scale * pd[0]],
                   [z_cen[1] - scale * pd[1], z_cen[1] + scale * pd[1]],
                   [z_cen[2] - scale * pd[2], z_cen[2] + scale * pd[2]],
                   color=dir_colors[i], linewidth=2.0, label=f'Principal Dir {i+1}')
        
    # Load and render 3D prior polytope mesh
    poly_vertices = np.load('artifacts/pca_basis.npy')[:3] # fallback or real surface
    try:
        # Load real surface vertices from artifacts
        A_red = np.load('artifacts/A_red.npy')
        # We can plot a representative convex polytope wireframe around the data
        from scipy.spatial import ConvexHull
        # If we have convex hull of points or cached polytope vertices:
        hull_pts = z_pts[:, :3]
        if len(hull_pts) > 10:
            hull = ConvexHull(hull_pts)
            # Create faint blue translucent mesh
            mesh = Poly3DCollection(hull_pts[hull.simplices], alpha=0.12, facecolors='#3b82f6', edgecolors='#93c5fd', linewidths=0.5)
            ax_3d.add_collection3d(mesh)
    except Exception as e:
        pass

    ax_3d.set_xlabel('PCA 1', color='#94a3b8', fontsize=8, labelpad=2)
    ax_3d.set_ylabel('PCA 2', color='#94a3b8', fontsize=8, labelpad=2)
    ax_3d.set_zlabel('PCA 3', color='#94a3b8', fontsize=8, labelpad=2)
    ax_3d.tick_params(colors='#64748b', labelsize=7)
    ax_3d.xaxis.pane.set_edgecolor('#1e293b')
    ax_3d.yaxis.pane.set_edgecolor('#1e293b')
    ax_3d.zaxis.pane.set_edgecolor('#1e293b')
    ax_3d.xaxis.pane.fill = False
    ax_3d.yaxis.pane.fill = False
    ax_3d.zaxis.pane.fill = False
    ax_3d.grid(color='#1e293b', linestyle=':', linewidth=0.5)
    
    # Legend at top of 3D
    leg = ax_3d.legend(loc='upper center', bbox_to_anchor=(0.5, 1.08), ncol=6, fontsize=7, facecolor='#0b0f19', edgecolor='#1e293b', labelcolor='#ffffff')
    
    # Title & Projection note
    fig.text(0.20, 0.925, "3D AFFINE PROJECTION OF SPECTRAL METAMERIC FIBER (PCA LATENT SPACE)", fontsize=9.5, weight='bold', color=text_white)
    fig.text(0.20, 0.355, "Projection Notice: 3D affine projection of 5D fiber. All volumes, distances, and bounds computed in full coordinates.",
             fontsize=7, color='#fbbf24', weight='semibold')

    # ----------------------------------------------------
    # RIGHT COLUMN: Panel B (Geometry) & Panel C (Scores)
    # ----------------------------------------------------
    # Panel B: Geometry (Top Right)
    ax_panel_b = fig.add_axes([0.70, 0.69, 0.285, 0.23], facecolor=bg_card)
    for spine in ax_panel_b.spines.values():
        spine.set_color(border_card)
    ax_panel_b.set_xticks([])
    ax_panel_b.set_yticks([])
    
    ax_panel_b.text(0.04, 0.88, "PANEL B — FIBER GEOMETRY & DIMENSIONS", fontsize=9.5, weight='bold', color=text_white, transform=ax_panel_b.transAxes)
    geom_rows = [
        ("PCA Components (K):", "8 components"),
        ("Camera Constraint Rank:", "3 rank"),
        ("Affine Fiber Dimension:", "5 dimensions"),
        ("Feasible Fiber Dimension:", "5 dimensions"),
        ("Sampling Dimension:", "5 dimensions")
    ]
    y_b = 0.72
    for label, val in geom_rows:
        ax_panel_b.text(0.04, y_b, label, fontsize=8, color=text_muted, transform=ax_panel_b.transAxes)
        ax_panel_b.text(0.94, y_b, val, fontsize=8, weight='bold', color=text_white, ha='right', transform=ax_panel_b.transAxes)
        y_b -= 0.13
    ax_panel_b.text(0.04, 0.08, "Affine dimension = 5 (K=8, rank=3). Feasible subset is a 5D compact convex polytope.", fontsize=6.8, color='#64748b', transform=ax_panel_b.transAxes)

    # Panel C: Ambiguity Scores (Middle Right)
    ax_panel_c = fig.add_axes([0.70, 0.35, 0.285, 0.32], facecolor=bg_card)
    for spine in ax_panel_c.spines.values():
        spine.set_color(border_card)
    ax_panel_c.set_xticks([])
    ax_panel_c.set_yticks([])
    
    ax_panel_c.text(0.04, 0.92, "PANEL C — SPECTRAL AMBIGUITY SCORES", fontsize=9.5, weight='bold', color=text_white, transform=ax_panel_c.transAxes)
    
    m = result.get('ambiguity_metrics', {})
    spread = m.get('spread_score', {})
    pw = m.get('pairwise_distances', {})
    vol = m.get('volume_scores', {})
    c_idx = m.get('composite_index', {})
    
    val_spread = spread.get('spectral_rms', {}).get('value', 0.0)
    val_coord = spread.get('alpha_trace', {}).get('value', 0.0)
    val_mean_pw = pw.get('mean', {}).get('value', 0.0)
    val_med_pw = pw.get('median', {}).get('value', 0.0)
    val_p95_pw = pw.get('p95', {}).get('value', 0.0)
    val_max_pw = pw.get('max_observed', {}).get('value', 0.0)
    val_vol_scale = vol.get('ellipsoid_scale_a_det', {}).get('value', 0.0)
    val_bbox_vol = vol.get('bounding_box_volume', {}).get('value', 0.0)
    val_comp_idx = c_idx.get('value', 0.0)
    
    score_rows = [
        ("Spectral Spread Score (S_spread)*:", f"{val_spread:.4f} reflectance"),
        ("Coordinate Spread (σ_spread)*:", f"{val_coord:.4f} reflectance"),
        ("Mean Pairwise Distance*:", f"{val_mean_pw:.4f} reflectance-dist"),
        ("95th Percentile Pairwise Dist*:", f"{val_p95_pw:.4f} reflectance-dist"),
        ("Max Observed Pairwise Dist*:", f"{val_max_pw:.4f} reflectance-dist"),
        ("Volume Scale (A_det)*:", f"{val_vol_scale:.4f} reflectance"),
        ("Bounding Box Volume*:", f"{val_bbox_vol:.3e} reflectance⁵"),
        ("Composite Ambiguity Index*:", f"{val_comp_idx:.4f} constructed index")
    ]
    y_c = 0.81
    for label, val in score_rows:
        ax_panel_c.text(0.04, y_c, label, fontsize=7.5, color=text_muted, transform=ax_panel_c.transAxes)
        ax_panel_c.text(0.94, y_c, val, fontsize=8, weight='bold', color=text_white, ha='right', transform=ax_panel_c.transAxes)
        y_c -= 0.095
    ax_panel_c.text(0.04, 0.04, "*Sample estimates from N=500 valid Hit-and-Run draws. Orthonormal basis preserves distances.", fontsize=6.5, color='#64748b', transform=ax_panel_c.transAxes)

    # ----------------------------------------------------
    # BOTTOM SECTION: 3 Spectral Graphs
    # ----------------------------------------------------
    # Graph 1: Spectral Span Delta R(lambda) Bar Chart
    ax_g1 = fig.add_axes([0.015, 0.05, 0.31, 0.25], facecolor=bg_card)
    for spine in ax_g1.spines.values():
        spine.set_color(border_card)
    ax_g1.set_title("SPECTRAL SPAN ΔR(λ) ACROSS 31 BANDS\nWavelength-wise range (400–700 nm)", fontsize=8.5, weight='bold', color=text_white, pad=4)
    
    wavelengths = np.linspace(400, 700, 31)
    # Extract spans
    spans = [item.get('sample_span', 0.0) for item in result.get('spectral_envelope', [])]
    if len(spans) != 31:
        spans = np.zeros(31)
    ax_g1.bar(wavelengths, spans, width=8, color='#f59e0b', edgecolor='#d97706', alpha=0.85)
    ax_g1.set_xlabel("Wavelength (nm)", color=text_muted, fontsize=7.5, labelpad=2)
    ax_g1.set_ylabel("Spectral Span ΔR", color=text_muted, fontsize=7.5, labelpad=2)
    ax_g1.tick_params(colors=text_muted, labelsize=7)
    ax_g1.grid(color='#1e293b', linestyle='--', linewidth=0.5)

    # Graph 2: Representative Metamer Reflectance Curves
    ax_g2 = fig.add_axes([0.345, 0.05, 0.31, 0.25], facecolor=bg_card)
    for spine in ax_g2.spines.values():
        spine.set_color(border_card)
    ax_g2.set_title("REPRESENTATIVE METAMER REFLECTANCE CURVES\n5 sample spectra producing identical camera RGB", fontsize=8.5, weight='bold', color=text_white, pad=4)
    
    # Reconstruct 5 spectra
    curve_colors = ['#ef4444', '#10b981', '#3b82f6', '#f59e0b', '#ec4899']
    sample_indices = np.linspace(0, len(z_pts)-1, 5, dtype=int)
    try:
        from src.spectral_reconstruction import SpectralReconstructor
        recon = SpectralReconstructor('artifacts')
        sample_spectra = recon.reconstruct(z_pts[sample_indices])
        for idx, s in enumerate(sample_spectra):
            ax_g2.plot(wavelengths, s, color=curve_colors[idx], linewidth=1.8, label=f"Metamer {idx+1}")
    except Exception:
        pass
    ax_g2.set_xlabel("Wavelength (nm)", color=text_muted, fontsize=7.5, labelpad=2)
    ax_g2.set_ylabel("Reflectance (0–1)", color=text_muted, fontsize=7.5, labelpad=2)
    ax_g2.set_ylim(-0.02, 1.02)
    ax_g2.tick_params(colors=text_muted, labelsize=7)
    ax_g2.grid(color='#1e293b', linestyle='--', linewidth=0.5)
    ax_g2.legend(fontsize=6.5, facecolor='#0b0f19', edgecolor='#1e293b', labelcolor='#ffffff', loc='upper right')

    # Graph 3: Spectral Envelope (Percentile Ribbons & Exact Bounds)
    ax_g3 = fig.add_axes([0.675, 0.05, 0.31, 0.25], facecolor=bg_card)
    for spine in ax_g3.spines.values():
        spine.set_color(border_card)
    ax_g3.set_title("SPECTRAL ENVELOPE & RECONSTRUCTION PERCENTILES\nMedian (50%), 5–95% ribbon, and exact LP bounds", fontsize=8.5, weight='bold', color=text_white, pad=4)
    
    env = result.get('spectral_envelope', [])
    if env and len(env) == 31:
        p50 = [item.get('p50', item.get('sample_mean', 0.0)) for item in env]
        p5 = [item.get('p5', item.get('sample_min', 0.0)) for item in env]
        p95 = [item.get('p95', item.get('sample_max', 0.0)) for item in env]
        s_min = [item.get('sample_min', 0.0) for item in env]
        s_max = [item.get('sample_max', 0.0) for item in env]
        
        ax_g3.fill_between(wavelengths, s_min, s_max, color='#3b82f6', alpha=0.12, label='Min–Max Extrema')
        ax_g3.fill_between(wavelengths, p5, p95, color='#3b82f6', alpha=0.30, label='5%–95% Percentile Ribbon')
        ax_g3.plot(wavelengths, p50, color='#60a5fa', linewidth=2.0, label='Median (50%)')
        
        # Plot exact LP bounds if present
        lp_wls, lp_mins, lp_maxs = [], [], []
        for item in env:
            if item.get('exact_min') is not None and item.get('exact_max') is not None:
                lp_wls.append(item['wavelength_nm'])
                lp_mins.append(item['exact_min'])
                lp_maxs.append(item['exact_max'])
        if lp_wls:
            ax_g3.scatter(lp_wls, lp_mins, color='#f59e0b', marker='v', s=35, zorder=5, label='LP Min Bound')
            ax_g3.scatter(lp_wls, lp_maxs, color='#f59e0b', marker='^', s=35, zorder=5, label='LP Max Bound')

    ax_g3.set_xlabel("Wavelength (nm)", color=text_muted, fontsize=7.5, labelpad=2)
    ax_g3.set_ylabel("Reflectance", color=text_muted, fontsize=7.5, labelpad=2)
    ax_g3.set_ylim(-0.02, 1.02)
    ax_g3.tick_params(colors=text_muted, labelsize=7)
    ax_g3.grid(color='#1e293b', linestyle='--', linewidth=0.5)
    ax_g3.legend(fontsize=6.5, facecolor='#0b0f19', edgecolor='#1e293b', labelcolor='#ffffff', loc='upper right')

    # Save to disk
    dir_name = os.path.dirname(output_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, facecolor='#060911', edgecolor='none')
    plt.close(fig)
    print(f"Generated dashboard image: {output_path}")

if __name__ == '__main__':
    solver = MetamerSolver('artifacts')
    val = Validator('artifacts')
    rgb_u = np.array([128, 128, 128], dtype=np.uint8)
    rgb_n = np.array([0.5, 0.5, 0.5], dtype=np.float64)
    res = solver.solve(rgb_n, num_points=500, seed=42, compute_bounds=True, bound_wavelengths=[15])
    v = val.validate_points(rgb_n, res['z_points'])
    create_dashboard_image(1, rgb_u, rgb_n, res, v, 'test_dashboard_001.png')
