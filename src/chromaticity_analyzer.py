import os
import json
import csv
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyBboxPatch
from matplotlib.path import Path
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
import colour

class ChromaticityAnalyzer:
    """
    Computes scientifically valid CIE 1931 xy chromaticity coordinates for
    spectral metameric fiber observations, generates continuous ambiguity heatmaps,
    and produces publication-quality visualizations and interactive dataset payloads.
    """
    def __init__(self, run_dir=None):
        if run_dir is None:
            # Default to the completed run
            base_res = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'random_colours_ambiguity_results'))
            runs = [r for r in os.listdir(base_res) if r.startswith('run_')]
            if runs:
                run_dir = os.path.join(base_res, sorted(runs)[-1])
            else:
                raise FileNotFoundError("No run directory found in random_colours_ambiguity_results")
        
        self.run_dir = os.path.abspath(run_dir)
        self.csv_path = os.path.join(self.run_dir, 'summary', 'ambiguity_summary.csv')
        self.meta_path = os.path.join(self.run_dir, 'run_metadata.json')
        
        # Load data
        self.df = pd.read_csv(self.csv_path)
        self._prepare_chromaticity_data()
        self._init_spectral_locus()

    def _prepare_chromaticity_data(self):
        """
        Converts all 100 RGB records into rigorous CIE 1931 xy coordinates
        via standard inverse sRGB gamma decoding and sRGB-to-XYZ transformation.
        """
        xy_coords = []
        hex_colors = []
        
        for _, row in self.df.iterrows():
            ru, gu, bu = int(row['r_uint8']), int(row['g_uint8']), int(row['b_uint8'])
            rgb_norm = np.array([ru, gu, bu], dtype=float) / 255.0
            
            # 1. sRGB gamma decoding and conversion to CIE XYZ
            xyz = colour.sRGB_to_XYZ(rgb_norm)
            sum_xyz = float(np.sum(xyz))
            
            # 2. Convert to xy chromaticity coordinates (handle near-black safely)
            if sum_xyz > 1e-9:
                xy = colour.XYZ_to_xy(xyz)
            else:
                # D65 neutral white point fallback for black
                xy = np.array([0.3127, 0.3290])
                
            xy_coords.append(xy)
            hex_colors.append(f"#{ru:02x}{gu:02x}{bu:02x}")
            
        xy_arr = np.array(xy_coords)
        self.df['cie_x'] = xy_arr[:, 0]
        self.df['cie_y'] = xy_arr[:, 1]
        self.df['hex_color'] = hex_colors

    def _init_spectral_locus(self):
        """
        Generates official CIE 1931 2-degree Standard Observer spectral locus,
        purple line boundary, and Planckian blackbody locus from first principles.
        """
        cmfs = colour.MSDS_CMFS['CIE 1931 2 Degree Standard Observer']
        
        # Continuous spectral locus from 380 nm to 700 nm
        self.locus_wavelengths = np.arange(380, 701, 1)
        xyz_locus = cmfs[self.locus_wavelengths]
        self.locus_xy = colour.XYZ_to_xy(xyz_locus)
        
        # Key annotated wavelengths matching the reference diagram
        self.annotated_wls = [380, 460, 470, 480, 490, 500, 520, 540, 560, 580, 600, 620, 700]
        self.annotated_coords = {}
        for wl in self.annotated_wls:
            self.annotated_coords[wl] = colour.XYZ_to_xy(cmfs[wl])
            
        # Purple boundary line connecting 380 nm and 700 nm
        p380 = self.annotated_coords[380]
        p700 = self.annotated_coords[700]
        self.purple_line = np.array([p380, p700])
        
        # Closed polygon of the visible chromaticity gamut
        self.chromaticity_polygon = np.vstack([self.locus_xy, [p380]])
        self.chromaticity_path = Path(self.chromaticity_polygon)
        
        # Planckian locus (blackbody curve) from 1500 K to 15000 K
        wls_m = np.arange(380, 781, 5) * 1e-9
        c1 = 3.741771e-16
        c2 = 1.438777e-2
        cmf_vals = cmfs[np.arange(380, 781, 5)]
        
        self.planckian_temps = [1500, 2000, 2500, 3000, 4000, 6000, 10000]
        planck_dense_temps = np.linspace(1400, 20000, 150)
        
        dense_xy = []
        for T in planck_dense_temps:
            bb = c1 / (wls_m**5 * (np.exp(c2 / (wls_m * T)) - 1.0))
            X = np.sum(bb[:, None] * cmf_vals[:, 0:1])
            Y = np.sum(bb[:, None] * cmf_vals[:, 1:2])
            Z = np.sum(bb[:, None] * cmf_vals[:, 2:3])
            dense_xy.append([X / (X+Y+Z), Y / (X+Y+Z)])
        self.planckian_curve = np.array(dense_xy)
        
        # Explicit temp points for tick marks
        self.planckian_ticks = {}
        for T in self.planckian_temps:
            bb = c1 / (wls_m**5 * (np.exp(c2 / (wls_m * T)) - 1.0))
            X = np.sum(bb[:, None] * cmf_vals[:, 0:1])
            Y = np.sum(bb[:, None] * cmf_vals[:, 1:2])
            Z = np.sum(bb[:, None] * cmf_vals[:, 2:3])
            self.planckian_ticks[T] = (X / (X+Y+Z), Y / (X+Y+Z))

        # sRGB Gamut triangle primaries and D65 white point
        self.srgb_triangle = np.array([
            [0.640, 0.330], # Red
            [0.300, 0.600], # Green
            [0.150, 0.060], # Blue
            [0.640, 0.330]  # Close
        ])
        self.white_point_d65 = np.array([0.3127, 0.3290])

    def get_supported_metrics(self):
        return {
            'composite_ambiguity_index': {
                'name': 'Composite Ambiguity Index',
                'unit': 'constructed index',
                'description': 'Overall composite metameric ambiguity score relative to prior dispersion'
            },
            'spectral_spread_score': {
                'name': 'Spectral Spread Score (S_spread)',
                'unit': 'reflectance',
                'description': 'Root-mean-square spectral reflectance dispersion around metamer mean'
            },
            'coordinate_spread': {
                'name': 'Coordinate Spread (σ_spread)',
                'unit': 'reflectance',
                'description': 'Reduced 5D alpha-space trace dispersion'
            },
            'mean_pairwise_distance': {
                'name': 'Mean Pairwise Distance',
                'unit': 'reflectance-dist',
                'description': 'Mean Euclidean distance between distinct sampled metamers'
            },
            'p95_pairwise_distance': {
                'name': '95th Percentile Pairwise Dist',
                'unit': 'reflectance-dist',
                'description': '95th percentile of metamer pairwise distance distribution'
            },
            'max_pairwise_distance': {
                'name': 'Max Observed Pairwise Dist',
                'unit': 'reflectance-dist',
                'description': 'Empirical diameter lower-bound for the metameric fiber'
            },
            'volume_scale_a_det': {
                'name': 'Volume Scale (A_det)',
                'unit': 'reflectance',
                'description': 'Ellipsoidal volume scale computed from 5D covariance determinant'
            }
        }

    def compute_summary_statistics(self, metric_key='composite_ambiguity_index'):
        """Computes statistical summary for the selected metric across all 100 observations."""
        s = self.df[metric_key].dropna()
        sorted_df = self.df.sort_values(by=metric_key)
        
        min_row = sorted_df.iloc[0]
        max_row = sorted_df.iloc[-1]
        
        return {
            'metric_key': metric_key,
            'metric_name': self.get_supported_metrics()[metric_key]['name'],
            'metric_unit': self.get_supported_metrics()[metric_key]['unit'],
            'total_colours': len(self.df),
            'valid_chromaticity_count': int((~self.df['cie_x'].isna()).sum()),
            'in_gamut_count': int((self.df['gamut_status'] == 'INSIDE GAMUT').sum()),
            'mean': float(s.mean()),
            'median': float(s.median()),
            'min': float(s.min()),
            'max': float(s.max()),
            'std': float(s.std()),
            'highest_colour': {
                'colour_id': str(max_row['colour_id']),
                'r_uint8': int(max_row['r_uint8']),
                'g_uint8': int(max_row['g_uint8']),
                'b_uint8': int(max_row['b_uint8']),
                'hex': str(max_row['hex_color']),
                'value': float(max_row[metric_key]),
                'cie_x': float(max_row['cie_x']),
                'cie_y': float(max_row['cie_y'])
            },
            'lowest_colour': {
                'colour_id': str(min_row['colour_id']),
                'r_uint8': int(min_row['r_uint8']),
                'g_uint8': int(min_row['g_uint8']),
                'b_uint8': int(min_row['b_uint8']),
                'hex': str(min_row['hex_color']),
                'value': float(min_row[metric_key]),
                'cie_x': float(min_row['cie_x']),
                'cie_y': float(min_row['cie_y'])
            },
            'mean_spectral_spread': float(self.df['spectral_spread_score'].mean()),
            'mean_pairwise_distance': float(self.df['mean_pairwise_distance'].mean()),
            'missing_or_invalid_count': int(self.df[metric_key].isna().sum())
        }

    def render_chromaticity_diagram(
        self,
        output_path,
        metric_key='composite_ambiguity_index',
        show_rgb_markers=True,
        show_labels=True,
        show_heatmap=True,
        dpi=150
    ):
        """
        Renders a publication-quality landscape CIE 1931 xy chromaticity diagram
        with a metameric ambiguity heatmap overlaid for the 100 observations.
        Matches the layout, spectral locus, wavelength annotations, grid, and blackbody locus.
        """
        metric_info = self.get_supported_metrics()[metric_key]
        metric_name = metric_info['name']
        metric_unit = metric_info['unit']
        
        xy_pts = self.df[['cie_x', 'cie_y']].values
        vals = self.df[metric_key].values
        
        # Setup Figure
        fig, ax = plt.subplots(figsize=(16, 12), dpi=dpi, facecolor='#060911')
        ax.set_facecolor('#0b0f19')
        
        # 1. Coordinate Grid & Axes
        ax.set_xlim(-0.02, 0.82)
        ax.set_ylim(-0.02, 0.92)
        ax.set_xticks(np.arange(0.0, 0.81, 0.1))
        ax.set_yticks(np.arange(0.0, 0.91, 0.1))
        ax.set_xlabel('CIE x Chromaticity Coordinate', color='#cbd5e1', fontsize=12, weight='bold', labelpad=8)
        ax.set_ylabel('CIE y Chromaticity Coordinate', color='#cbd5e1', fontsize=12, weight='bold', labelpad=8)
        ax.tick_params(colors='#94a3b8', labelsize=10, width=1.2)
        for spine in ax.spines.values():
            spine.set_color('#334155')
            spine.set_linewidth(1.2)
        
        # Major and minor grid lines
        ax.grid(color='#1e293b', linestyle='-', linewidth=0.8, alpha=0.9)
        ax.minorticks_on()
        ax.grid(which='minor', color='#1e293b', linestyle=':', linewidth=0.5, alpha=0.5)

        # 2. Draw Spectral Locus & Purple Boundary Line
        ax.plot(self.locus_xy[:, 0], self.locus_xy[:, 1], color='#38bdf8', linewidth=2.2, label='Spectral Locus (380–700 nm)')
        ax.plot(self.purple_line[:, 0], self.purple_line[:, 1], color='#ec4899', linewidth=2.0, linestyle='--', label='Line of Purples')

        # Outer spectral locus background shading (subtle gray envelope)
        poly_patch = Polygon(self.chromaticity_polygon, facecolor='#0f172a', edgecolor='none', alpha=0.6, zorder=1)
        ax.add_patch(poly_patch)

        # 3. Annotated Wavelength Ticks and Labels along Spectral Locus
        for wl, (wx, wy) in self.annotated_coords.items():
            # Small outward tick marker
            ax.plot([wx], [wy], marker='o', markersize=3.5, color='#38bdf8', zorder=4)
            # Label positioning offset slightly away from locus curvature
            dx = 0.015 if wx > 0.3 else -0.025
            dy = 0.015 if wy > 0.4 else -0.02
            if wl in [520, 540]:
                dx, dy = 0.005, 0.015
            elif wl in [460, 470, 480]:
                dx, dy = -0.035, -0.005
            elif wl == 380:
                dx, dy = 0.015, -0.015
            elif wl == 700:
                dx, dy = 0.015, 0.005
                
            ax.text(wx + dx, wy + dy, str(wl), color='#60a5fa', fontsize=9, weight='bold',
                    ha='center', va='center', zorder=5)

        # 4. Planckian (Blackbody) Locus & Temperature Labels Tc(K)
        ax.plot(self.planckian_curve[:, 0], self.planckian_curve[:, 1], color='#f59e0b', linewidth=1.8, linestyle='-', label='Planckian Locus Tc(K)', zorder=3)
        ax.text(0.38, 0.45, r'T$_c$(K)', color='#f59e0b', fontsize=11, weight='bold', zorder=5)
        
        for T, (px, py) in self.planckian_ticks.items():
            ax.plot([px], [py], marker='|', markersize=8, color='#f59e0b', mew=1.5, zorder=4)
            if T in [1500, 2000, 2500, 3000, 4000, 6000, 10000]:
                ax.text(px + 0.005, py - 0.022, f'{T}', color='#fbbf24', fontsize=7.5, weight='semibold', zorder=5)

        # 5. sRGB Gamut Triangle & D65 White Point
        ax.plot(self.srgb_triangle[:, 0], self.srgb_triangle[:, 1], color='#94a3b8', linewidth=1.2, linestyle=':', label='sRGB Gamut Triangle', zorder=3)
        ax.scatter([self.white_point_d65[0]], [self.white_point_d65[1]], color='#ffffff', marker='x', s=45, linewidth=2, label='D65 White Point (0.3127, 0.3290)', zorder=6)

        # 6. Ambiguity Heatmap (Interpolation with scientific distance masking)
        cbar = None
        if show_heatmap:
            # High-resolution grid over [0, 0.8] x [0, 0.9]
            gx, gy = np.mgrid[0.05:0.75:250j, 0.05:0.85:250j]
            grid_points = np.c_[gx.ravel(), gy.ravel()]
            
            # Distance masking: only interpolate within support of observations (max distance = 0.065)
            tree = cKDTree(xy_pts)
            dists, _ = tree.query(grid_points)
            dists = dists.reshape(gx.shape)
            
            # Cubic interpolation with linear fallback
            gz = griddata(xy_pts, vals, (gx, gy), method='cubic')
            gz_lin = griddata(xy_pts, vals, (gx, gy), method='linear')
            gz = np.where(np.isnan(gz), gz_lin, gz)
            
            # Clip inside spectral locus and observations support radius
            in_locus = self.chromaticity_path.contains_points(grid_points).reshape(gx.shape)
            mask = (dists <= 0.065) & in_locus
            gz_masked = np.where(mask, gz, np.nan)
            
            # Color map matching scientific specification (dark blue -> cyan -> green -> yellow -> red)
            cmap = plt.cm.turbo
            vmin = float(np.min(vals))
            vmax = float(np.max(vals))
            
            # Contour fill overlay
            cf = ax.contourf(gx, gy, gz_masked, levels=28, cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.72, zorder=2)
            # Subtle contour lines
            ax.contour(gx, gy, gz_masked, levels=12, colors='#1e293b', linewidths=0.5, alpha=0.4, zorder=2)
            
            # Vertical Color Bar on Right Side
            cbar = fig.colorbar(cf, ax=ax, fraction=0.032, pad=0.025)
            cbar.set_label(f"{metric_name} ({metric_unit})", color='#ffffff', fontsize=11, weight='bold', labelpad=12)
            cbar.ax.yaxis.set_tick_params(color='#ffffff', labelsize=9.5)
            plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#ffffff')
            cbar.outline.set_edgecolor('#334155')

        # 7. Plot All 100 Selected RGB Observations
        if show_rgb_markers:
            # Outer ring colored by ambiguity score for visual alignment with heatmap
            scatter_halo = ax.scatter(
                xy_pts[:, 0], xy_pts[:, 1],
                c=vals, cmap=plt.cm.turbo, vmin=vals.min(), vmax=vals.max(),
                s=110, edgecolors='#0f172a', linewidth=1.5, zorder=7,
                label='100 Selected In-Gamut Colours'
            )
            
            # Inner circle displaying the actual original RGB color
            for i, row in self.df.iterrows():
                px, py = row['cie_x'], row['cie_y']
                rgb_frac = np.array([row['r_uint8'], row['g_uint8'], row['b_uint8']]) / 255.0
                ax.plot(px, py, marker='o', markersize=5.2, color=rgb_frac, zorder=8)
                
            # Optional colour identifiers for key extremes or selective samples
            if show_labels:
                # Label lowest, highest, and every 10th colour to keep diagram clean
                extreme_indices = [
                    self.df[metric_key].idxmin(),
                    self.df[metric_key].idxmax()
                ]
                sample_indices = set(range(0, len(self.df), 10)).union(set(extreme_indices))
                
                for idx in sample_indices:
                    row = self.df.iloc[idx]
                    cid = f"#{row['colour_idx']:03d}"
                    px, py = row['cie_x'], row['cie_y']
                    ax.text(px + 0.008, py + 0.008, cid, color='#f8fafc', fontsize=7.5, weight='bold',
                            bbox=dict(boxstyle='round,pad=0.15', facecolor='#090d16', edgecolor='#334155', alpha=0.85, linewidth=0.8),
                            zorder=9)

        # 8. Title, Subtitle, and Scientific Disclaimers
        fig.text(0.08, 0.965, "CIE 1931 xy CHROMATICITY DIAGRAM — METAMERIC AMBIGUITY HEATMAP",
                 fontsize=15, weight='bold', color='#ffffff')
        fig.text(0.08, 0.942, "36K-SPECTRAL-PRIOR • DISTRIBUTION OF 100 UNIQUE IN-GAMUT OBSERVATIONS (Dataset/image.png)",
                 fontsize=10, weight='semibold', color='#38bdf8')
        
        # Bottom Scientific Notice
        notice_text = (
            "Scientific Notice: Chromaticity coordinates (x, y) computed via standard sRGB D65 inverse gamma and XYZ transformation. "
            "Heatmap shows continuous local interpolation of the Composite Ambiguity Index estimated from 100 actual in-gamut observations "
            "(50,000 retained Hit-and-Run metamers). Region masked to observation support. Marker centers show actual RGB colour."
        )
        fig.text(0.08, 0.02, notice_text, fontsize=8, color='#94a3b8', style='italic')

        # Legend
        leg = ax.legend(loc='upper right', fontsize=8.5, facecolor='#090d16', edgecolor='#334155', labelcolor='#ffffff', framealpha=0.9)

        # Save Image
        dir_name = os.path.dirname(output_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
            
        plt.savefig(output_path, dpi=dpi, facecolor='#060911', edgecolor='none', bbox_inches='tight')
        plt.close(fig)
        print(f"Generated CIE xy Chromaticity Ambiguity Diagram: {output_path}")

    def export_summary_dataset(self, output_json_path):
        """
        Exports full structured dataset for interactive Plotly.js web application:
        coordinates, all metrics, locus path, Planckian curve, and aggregate statistics.
        """
        dir_name = os.path.dirname(output_json_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        records = []
        for _, r in self.df.iterrows():
            records.append({
                'colour_id': str(r['colour_id']),
                'colour_idx': int(r['colour_idx']),
                'r_uint8': int(r['r_uint8']),
                'g_uint8': int(r['g_uint8']),
                'b_uint8': int(r['b_uint8']),
                'hex_color': str(r['hex_color']),
                'cie_x': round(float(r['cie_x']), 4),
                'cie_y': round(float(r['cie_y']), 4),
                'gamut_status': str(r['gamut_status']),
                'fiber_dimension': int(r['fiber_dimension']),
                'constraint_rank': int(r['constraint_rank']),
                'spectral_spread_score': round(float(r['spectral_spread_score']), 4),
                'coordinate_spread': round(float(r['coordinate_spread']), 4),
                'mean_pairwise_distance': round(float(r['mean_pairwise_distance']), 4),
                'median_pairwise_distance': round(float(r['median_pairwise_distance']), 4),
                'p95_pairwise_distance': round(float(r['p95_pairwise_distance']), 4),
                'max_pairwise_distance': round(float(r['max_pairwise_distance']), 4),
                'volume_scale_a_det': round(float(r['volume_scale_a_det']), 4),
                'bounding_box_volume': float(r['bounding_box_volume']),
                'composite_ambiguity_index': round(float(r['composite_ambiguity_index']), 4),
                'valid_sample_count': int(r['valid_sample_count']),
                'validation_status': str(r['validation_status'])
            })

        locus_data = [
            {'wavelength': int(wl), 'x': round(float(self.locus_xy[i, 0]), 4), 'y': round(float(self.locus_xy[i, 1]), 4)}
            for i, wl in enumerate(self.locus_wavelengths)
        ]

        annotated_locus = [
            {'wavelength': wl, 'x': round(float(self.annotated_coords[wl][0]), 4), 'y': round(float(self.annotated_coords[wl][1]), 4)}
            for wl in self.annotated_wls
        ]

        planckian_data = [
            {'x': round(float(p[0]), 4), 'y': round(float(p[1]), 4)}
            for p in self.planckian_curve
        ]

        planckian_labels = [
            {'temp_k': T, 'x': round(float(self.planckian_ticks[T][0]), 4), 'y': round(float(self.planckian_ticks[T][1]), 4)}
            for T in self.planckian_temps
        ]

        srgb_triangle_data = [
            {'x': float(pt[0]), 'y': float(pt[1])} for pt in self.srgb_triangle
        ]

        # Compute summary statistics for all metrics
        stats_by_metric = {}
        for m_key in self.get_supported_metrics():
            stats_by_metric[m_key] = self.compute_summary_statistics(m_key)

        payload = {
            'run_id': os.path.basename(self.run_dir),
            'total_observations': len(records),
            'records': records,
            'spectral_locus': locus_data,
            'annotated_wavelengths': annotated_locus,
            'purple_boundary': [
                {'x': round(float(self.purple_line[0, 0]), 4), 'y': round(float(self.purple_line[0, 1]), 4)},
                {'x': round(float(self.purple_line[1, 0]), 4), 'y': round(float(self.purple_line[1, 1]), 4)}
            ],
            'planckian_locus': planckian_data,
            'planckian_labels': planckian_labels,
            'srgb_gamut_triangle': srgb_triangle_data,
            'white_point_d65': {'x': 0.3127, 'y': 0.3290},
            'metrics_metadata': self.get_supported_metrics(),
            'statistics': stats_by_metric
        }

        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2)
        print(f"Exported interactive chromaticity JSON: {output_json_path}")
        return payload
