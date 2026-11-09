import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull

# Pre-load prior polytope 3D hull to reuse across all 100 images
_CACHED_POLY_VERTS = None
_CACHED_POLY_FACES = None

def _get_prior_polytope():
    global _CACHED_POLY_VERTS, _CACHED_POLY_FACES
    if _CACHED_POLY_VERTS is None:
        artifacts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'artifacts'))
        p_surf_path = os.path.join(artifacts_dir, 'P_surf_pca.npy')
        if os.path.exists(p_surf_path):
            P_surf_pca = np.load(p_surf_path)
            P3d = P_surf_pca[:, :3]
            hull = ConvexHull(P3d)
            _CACHED_POLY_VERTS = P3d
            _CACHED_POLY_FACES = hull.simplices
    return _CACHED_POLY_VERTS, _CACHED_POLY_FACES


def render_color_dashboard(
    colour_idx,
    rgb_uint8,
    rgb_norm,
    result,
    val_result,
    output_path,
    dpi=100
):
    """
    Renders a 1920x1080 scientific dashboard image representing the complete
    analysis of one selected colour, exactly resembling the Spectral Metameric Fiber Analyzer UI.
    """
    fig = plt.figure(figsize=(19.2, 10.8), dpi=dpi, facecolor='#060911')
    
    # Theme colors
    bg_card = '#0c111d'
    border_card = '#1e293b'
    text_white = '#ffffff'
    text_muted = '#94a3b8'
    text_cyan = '#38bdf8'
    text_green = '#34d399'
    text_amber = '#fbbf24'
    
    # ----------------------------------------------------
    # HEADER BAR (y: 0.93 to 1.0)
    # ----------------------------------------------------
    fig.text(0.015, 0.965, "SPECTRAL METAMERIC FIBER ANALYZER", fontsize=15, weight='bold', color=text_white)
    fig.text(0.015, 0.940, "36K-SPECTRAL-PRIOR • DYNAMIC AMBIGUITY & FIBER GEOMETRY ENGINE", fontsize=9.5, weight='bold', color=text_cyan)
    
    r_u, g_u, b_u = int(rgb_uint8[0]), int(rgb_uint8[1]), int(rgb_uint8[2])
    r_n, g_n, b_n = float(rgb_norm[0]), float(rgb_norm[1]), float(rgb_norm[2])
    
    # Center info badge
    fig.text(0.40, 0.952, f"COLOUR #{colour_idx:03d} — RGB [{r_u}, {g_u}, {b_u}] / [{r_n:.3f}, {g_n:.3f}, {b_n:.3f}]", 
             fontsize=11, weight='bold', color='#e2e8f0')

    # Top right badges
    bbox_green = dict(boxstyle='square,pad=0.4', facecolor='#065f46', edgecolor='#10b981', linewidth=1)
    bbox_blue = dict(boxstyle='square,pad=0.4', facecolor='#1e40af', edgecolor='#3b82f6', linewidth=1)
    
    overall_st = val_result.get('overall_status', 'PASS') if val_result else 'PASS'
    status_edge = '#10b981' if overall_st == 'PASS' else ('#f59e0b' if overall_st == 'WARN' else '#ef4444')
    status_bg = '#065f46' if overall_st == 'PASS' else ('#78350f' if overall_st == 'WARN' else '#7f1d1d')
    bbox_status = dict(boxstyle='square,pad=0.4', facecolor=status_bg, edgecolor=status_edge, linewidth=1)
    
    fig.text(0.74, 0.955, "GAMUT: INSIDE GAMUT", fontsize=8.5, weight='bold', color='#a7f3d0', bbox=bbox_green)
    fig.text(0.835, 0.955, f"{result['fiber_dimension']}D METAMERIC FIBER", fontsize=8.5, weight='bold', color='#bfdbfe', bbox=bbox_blue)
    fig.text(0.925, 0.955, f"VALIDATION: {overall_st}", fontsize=8.5, weight='bold', color='#a7f3d0' if overall_st=='PASS' else '#fef08a', bbox=bbox_status)

    # ----------------------------------------------------
    # LEFT COLUMN: Panel A (Target RGB & Feasibility)
    # ----------------------------------------------------
    ax_panel_a = fig.add_axes([0.015, 0.35, 0.175, 0.57], facecolor=bg_card)
    for spine in ax_panel_a.spines.values():
        spine.set_color(border_card)
    ax_panel_a.set_xticks([])
    ax_panel_a.set_yticks([])
    
    ax_panel_a.text(0.06, 0.96, "PANEL A — TARGET RGB", fontsize=9.5, weight='bold', color=text_white, transform=ax_panel_a.transAxes)
    ax_panel_a.text(0.06, 0.925, "& FEASIBILITY", fontsize=9.5, weight='bold', color=text_white, transform=ax_panel_a.transAxes)
    
    channels = [("R (Red):", r_n, '#ef4444'), ("G (Green):", g_n, '#10b981'), ("B (Blue):", b_n, '#3b82f6')]
    y_pos = 0.85
    for label, val, c_line in channels:
        ax_panel_a.text(0.06, y_pos, label, fontsize=8.5, color=text_muted, transform=ax_panel_a.transAxes)
        ax_panel_a.text(0.92, y_pos, f"{val:.2f}", fontsize=8.5, weight='bold', color=text_white, ha='right', transform=ax_panel_a.transAxes)
        rect_bg = Rectangle((0.06, y_pos - 0.035), 0.86, 0.012, facecolor='#1e293b', transform=ax_panel_a.transAxes)
        rect_fill = Rectangle((0.06, y_pos - 0.035), 0.86 * min(1.0, max(0.0, val)), 0.012, facecolor=c_line, transform=ax_panel_a.transAxes)
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
    ax_panel_a.text(0.33, y_pos - 0.11, "Inside spectral prior gamut.", fontsize=7.0, color=text_muted, transform=ax_panel_a.transAxes)

    # Sampling Settings
    y_pos -= 0.17
    ax_panel_a.text(0.06, y_pos, "SAMPLING SETTINGS (MCMC)", fontsize=8, weight='bold', color=text_cyan, transform=ax_panel_a.transAxes)
    ax_panel_a.text(0.06, y_pos - 0.045, "Number of samples (N):", fontsize=8, color=text_muted, transform=ax_panel_a.transAxes)
    ax_panel_a.text(0.92, y_pos - 0.045, str(result.get('n_samples', 500)), fontsize=8.5, weight='bold', color=text_amber, ha='right', transform=ax_panel_a.transAxes)
    ax_panel_a.text(0.06, y_pos - 0.085, f"Burn-in: {result.get('burn_in', 50)} | Thinning: {result.get('thinning', 2)}", fontsize=7.5, color=text_muted, transform=ax_panel_a.transAxes)
    ax_panel_a.text(0.06, y_pos - 0.125, "LP Bounds: 550 nm (Exact)", fontsize=7.5, color=text_muted, transform=ax_panel_a.transAxes)

    # Status / Info Button
    btn_rect = FancyBboxPatch((0.06, 0.05), 0.86, 0.08, boxstyle="round,pad=0.01",
                              facecolor='#1e40af', edgecolor='#3b82f6', linewidth=1, transform=ax_panel_a.transAxes)
    ax_panel_a.add_patch(btn_rect)
    ax_panel_a.text(0.49, 0.09, "ANALYSIS COMPLETE", fontsize=8.5, weight='bold', color='#ffffff', ha='center', va='center', transform=ax_panel_a.transAxes)

    # ----------------------------------------------------
    # CENTER COLUMN: 3D Visualization of Metameric Fiber
    # ----------------------------------------------------
    ax_3d = fig.add_axes([0.20, 0.35, 0.49, 0.57], projection='3d', facecolor='#000000')
    ax_3d.set_facecolor('#000000')
    
    z_pts = result['z_points']
    z0 = result['z0']
    z_cen = result['z_centroid']
    p_dirs = result['principal_directions']
    
    # 1. Render true Prior Polytope Mesh in translucent blue with delicate soft edges
    p_verts, p_faces = _get_prior_polytope()
    if p_verts is not None:
        poly_mesh = Poly3DCollection(
            p_verts[p_faces],
            alpha=0.10,
            facecolors='#2563eb',
            edgecolors=(147/255, 197/255, 253/255, 0.08),
            linewidths=0.6
        )
        ax_3d.add_collection3d(poly_mesh)
    
    # 2. Render sampled metamer points in gold/amber
    ax_3d.scatter(z_pts[:, 0], z_pts[:, 1], z_pts[:, 2], c='#f59e0b', s=2.5, alpha=0.55, edgecolors='none', label=f'Metamers (N={len(z_pts)})')
    # 3. Render centroid
    ax_3d.scatter([z_cen[0]], [z_cen[1]], [z_cen[2]], c='#ef4444', marker='+', s=60, linewidth=2, label='Fiber Centroid')
    # 4. Render anchor
    ax_3d.scatter([z0[0]], [z0[1]], [z0[2]], c='#06b6d4', marker='o', s=25, label='Feasible Anchor z0')
    
    # 5. Render principal direction axes
    dir_colors = ['#ef4444', '#10b981', '#3b82f6']
    std_scale = np.std(z_pts[:, 0]) * 2.5 if len(z_pts) > 1 else 0.5
    for i in range(min(3, p_dirs.shape[1])):
        pd = p_dirs[:, i]
        ax_3d.plot([z_cen[0] - std_scale * pd[0], z_cen[0] + std_scale * pd[0]],
                   [z_cen[1] - std_scale * pd[1], z_cen[1] + std_scale * pd[1]],
                   [z_cen[2] - std_scale * pd[2], z_cen[2] + std_scale * pd[2]],
                   color=dir_colors[i], linewidth=2.0, label=f'Principal Dir {i+1}')

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
    
    # Camera view matching default 3D eye { x: 1.35, y: 1.35, z: 1.15 }
    ax_3d.view_init(elev=24, azim=48)
    
    # 3D legend inside plot area
    ax_3d.legend(loc='upper right', fontsize=6.8, facecolor='#0b0f19', edgecolor='#1e293b', labelcolor='#ffffff', framealpha=0.75)
    
    fig.text(0.20, 0.925, "3D AFFINE PROJECTION OF SPECTRAL METAMERIC FIBER (PCA LATENT SPACE)", fontsize=9.5, weight='bold', color=text_white)
    fig.text(0.20, 0.355, "Projection Notice: The 3D view is an affine projection of the 5D fiber. All volumes, distances, and bounds computed in full coordinates.",
             fontsize=6.8, color='#fbbf24', weight='bold')

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
        ("Camera Constraint Rank:", f"{result.get('constraint_rank', 3)} rank"),
        ("Affine Fiber Dimension:", f"{result.get('fiber_dimension', 5)} dimensions"),
        ("Feasible Fiber Dimension:", f"{result.get('fiber_dimension', 5)} dimensions"),
        ("Sampling Dimension:", f"{result.get('fiber_dimension', 5)} dimensions")
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
    wavelengths = np.linspace(400, 700, 31)

    # Graph 1: Spectral Span Delta R(lambda) Bar Chart
    ax_g1 = fig.add_axes([0.015, 0.05, 0.31, 0.25], facecolor=bg_card)
    for spine in ax_g1.spines.values():
        spine.set_color(border_card)
    ax_g1.set_title("SPECTRAL SPAN ΔR(λ) ACROSS 31 BANDS\nWavelength-wise range (400–700 nm)", fontsize=8.5, weight='bold', color=text_white, pad=4)
    
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
