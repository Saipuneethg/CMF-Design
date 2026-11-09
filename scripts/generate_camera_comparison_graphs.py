"""
generate_camera_comparison_graphs.py
====================================
Generates publication-quality graphs representing all 30 camera models' ambiguity,
styled in the exact visual design system of results/cmf_optimization/ambiguity_comparison.png.

Outputs:
1. camera comparisions/plots/camera_models_ambiguity_comparison.png
2. results/cmf_optimization/all_camera_models_ambiguity_comparison.png
3. camera comparisions/plots/all_camera_models_distribution_boxplots.png
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def main():
    base_dir = "camera comparisions"
    csv_cam_sum = os.path.join(base_dir, "ambiguity_results", "camera_ambiguity_summary.csv")
    csv_per_col = os.path.join(base_dir, "ambiguity_results", "ambiguity_per_colour.csv")
    json_path = os.path.join(base_dir, "ambiguity_results", "ambiguity_summary.json")
    
    if not os.path.exists(csv_cam_sum) or not os.path.exists(csv_per_col):
        raise FileNotFoundError("Ambiguity results CSVs not found in camera comparisions/ambiguity_results/")
        
    df_cam = pd.read_csv(csv_cam_sum)
    df_col = pd.read_csv(csv_per_col)
    
    # Sort strictly by average ambiguity ascending (Rank 1 to 30)
    df_cam = df_cam.sort_values(by="average_ambiguity", ascending=True).reset_index(drop=True)
    df_cam["rank"] = range(1, len(df_cam) + 1)
    
    orig_row = df_cam[df_cam["model_type"] == "original_model"].iloc[0]
    opt_row = df_cam[df_cam["model_type"] == "optimized_model"].iloc[0]
    orig_mean = orig_row["average_ambiguity"]
    opt_mean = opt_row["average_ambiguity"]
    
    print("=" * 80)
    print("GENERATING CAMERA MODELS AMBIGUITY COMPARISON GRAPHS")
    print("=" * 80)
    print(f"Total camera models: {len(df_cam)}")
    print(f"Original Model Ambiguity:  {orig_mean:.6f} (Rank #{orig_row['rank']})")
    print(f"Optimized Model Ambiguity: {opt_mean:.6f} (Rank #{opt_row['rank']})")
    print("=" * 80)
    
    # =========================================================================
    # MASTER FIGURE 1: 2-Panel ambiguity_comparison style (Ranked Bar Chart + Distribution Boxplot)
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 12), dpi=200)
    
    # Color palette
    color_opt = '#10b981'   # Emerald Green
    color_orig = '#f43f5e'  # Rose Red
    color_db = '#38bdf8'    # Sky Blue
    
    # -------------------------------------------------------------------------
    # Panel 1 (Left): Ranked Mean Ambiguity Bar Chart (30 Camera Models)
    # -------------------------------------------------------------------------
    # For horizontal bars, display from rank 30 at bottom to rank 1 at top
    df_plot = df_cam.iloc[::-1].reset_index(drop=True)
    
    bar_colors = []
    edge_colors = []
    for _, r in df_plot.iterrows():
        if r["model_type"] == "optimized_model":
            bar_colors.append(color_opt)
            edge_colors.append('#047857')
        elif r["model_type"] == "original_model":
            bar_colors.append(color_orig)
            edge_colors.append('#be123c')
        else:
            bar_colors.append(color_db)
            edge_colors.append('#0284c7')
            
    y_pos = np.arange(len(df_plot))
    means = df_plot["average_ambiguity"].values
    stds = df_plot["std_ambiguity"].values
    names = df_plot["camera_name"].values
    ranks = df_plot["rank"].values
    
    # Clean names for display
    display_names = []
    for name, rk in zip(names, ranks):
        prefix = f"#{rk:02d} "
        if "Optimized Model" in name:
            display_names.append(f"{prefix}Optimized Model (Run_2_seed101) ★")
        elif "Original Model" in name:
            display_names.append(f"{prefix}Original Model (M_camera)")
        else:
            display_names.append(f"{prefix}{name}")
            
    bars = ax1.barh(y_pos, means, xerr=stds / np.sqrt(100), capsize=3, color=bar_colors, edgecolor=edge_colors, height=0.72, alpha=0.88, ecolor='#475569', lw=1.1)
    
    # Reference vertical lines
    ax1.axvline(opt_mean, color='#047857', linestyle='-', lw=2.0, alpha=0.9, label=f'Optimized Model Mean ({opt_mean:.5f})')
    ax1.axvline(orig_mean, color='#be123c', linestyle='--', lw=2.0, alpha=0.9, label=f'Original Model Mean ({orig_mean:.5f})')
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(display_names, fontsize=8.5, fontweight='500')
    ax1.set_xlabel('Average Metameric Ambiguity (Spectral RMS Spread)', fontsize=11, fontweight='bold')
    ax1.set_title('30-Camera Metameric Ambiguity Ranking (100 PNG Colours)\nLower Spread = Superior Spectral Constraint', fontsize=12, fontweight='bold', pad=12)
    ax1.set_xlim(0.065, 0.108)
    ax1.grid(True, axis='x', linestyle='--', alpha=0.35)
    
    # Add value annotations on each bar
    for idx, (b, m, rk) in enumerate(zip(bars, means, ranks)):
        diff_pct = 100.0 * (orig_mean - m) / orig_mean
        if rk == 1:
            lbl = f" {m:.5f} (+{diff_pct:.1f}% vs Orig) WINNER"
            weight = 'bold'
            txt_color = '#047857'
        elif "Original" in names[idx]:
            lbl = f" {m:.5f} (Baseline)"
            weight = 'bold'
            txt_color = '#be123c'
        else:
            sign = "+" if diff_pct >= 0 else ""
            lbl = f" {m:.5f} ({sign}{diff_pct:.1f}%)"
            weight = 'normal'
            txt_color = '#1e293b'
            
        ax1.text(m + 0.0008, b.get_y() + b.get_height()/2.0, lbl, va='center', fontsize=7.8, fontweight=weight, color=txt_color)
        
    ax1.legend(loc='lower right', fontsize=9.5, framealpha=0.95)
    
    # -------------------------------------------------------------------------
    # Panel 2 (Right): 30-Camera Distribution Boxplot (IQR, Medians, Outliers)
    # -------------------------------------------------------------------------
    # Gather per-colour data for each camera in the same rank order (30 down to 1)
    box_data = []
    for name in names:
        sub = df_col[df_col["camera_name"] == name]["ambiguity_value"].dropna().values
        box_data.append(sub)
        
    bp = ax2.boxplot(box_data, vert=False, positions=y_pos, patch_artist=True, widths=0.6,
                     medianprops=dict(color='#0f172a', lw=2.2),
                     flierprops=dict(marker='o', markersize=3, alpha=0.4, markerfacecolor='#64748b', markeredgecolor='none'),
                     whiskerprops=dict(color='#475569', lw=1.1),
                     capprops=dict(color='#475569', lw=1.1))
                     
    for patch, col, ecol in zip(bp['boxes'], bar_colors, edge_colors):
        patch.set_facecolor(col)
        patch.set_edgecolor(ecol)
        patch.set_alpha(0.75)
        patch.set_linewidth(1.1)
        
    ax2.axvline(opt_row["median_ambiguity"], color='#047857', linestyle='-', lw=2.0, alpha=0.9, label=f'Optimized Median ({opt_row["median_ambiguity"]:.5f})')
    ax2.axvline(orig_row["median_ambiguity"], color='#be123c', linestyle='--', lw=2.0, alpha=0.9, label=f'Original Median ({orig_row["median_ambiguity"]:.5f})')
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([]) # Share y-ticks with Panel 1
    ax2.set_xlabel('Metameric Ambiguity Distribution across 100 Colours', fontsize=11, fontweight='bold')
    ax2.set_title('Ambiguity Quartiles & Dispersion across 100 Natural Colours\n(Solid Line = Median | Box = 25th–75th Percentile IQR)', fontsize=12, fontweight='bold', pad=12)
    ax2.set_xlim(0.0, 0.27)
    ax2.grid(True, axis='x', linestyle='--', alpha=0.35)
    ax2.legend(loc='lower right', fontsize=9.5, framealpha=0.95)
    
    # Legend for camera types
    patch_opt = mpatches.Patch(facecolor=color_opt, edgecolor='#047857', label='Optimized Model (Rank #1)')
    patch_orig = mpatches.Patch(facecolor=color_orig, edgecolor='#be123c', label='Original Model (Rank #15)')
    patch_db = mpatches.Patch(facecolor=color_db, edgecolor='#0284c7', label='28 Measured Physical Cameras')
    fig.legend(handles=[patch_opt, patch_orig, patch_db], loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=3, fontsize=10.5, frameon=True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save destinations
    out_plot1 = os.path.join(base_dir, "plots", "camera_models_ambiguity_comparison.png")
    out_results = os.path.join("results", "cmf_optimization", "all_camera_models_ambiguity_comparison.png")
    out_amb_res = os.path.join(base_dir, "ambiguity_results", "camera_models_ambiguity_comparison.png")
    
    os.makedirs(os.path.dirname(out_results), exist_ok=True)
    fig.savefig(out_plot1, dpi=200)
    fig.savefig(out_results, dpi=200)
    fig.savefig(out_amb_res, dpi=200)
    plt.close(fig)
    print(f"Saved master graph to:")
    print(f"  -> {out_plot1}")
    print(f"  -> {out_results}")
    print(f"  -> {out_amb_res}")
    
    # =========================================================================
    # MASTER FIGURE 2: Vertical Boxplot & Trajectory Comparison (Identical to results/cmf_optimization/ambiguity_comparison.png layout)
    # =========================================================================
    fig2, (v_ax1, v_ax2) = plt.subplots(1, 2, figsize=(16, 6.5), dpi=200)
    
    # Left subplot: Vertical Boxplot of Top 10 Models + Original Model
    top_and_orig = df_cam.head(10).copy()
    if not (top_and_orig["model_type"] == "original_model").any():
        top_and_orig = pd.concat([top_and_orig, pd.DataFrame([orig_row])]).reset_index(drop=True)
        
    v_box_data = []
    v_labels = []
    v_colors = []
    v_edges = []
    for _, r in top_and_orig.iterrows():
        cname = r["camera_name"]
        sub = df_col[df_col["camera_name"] == cname]["ambiguity_value"].dropna().values
        v_box_data.append(sub)
        short_name = cname.replace(" (measured_database)", "").replace(" (M_camera)", "").replace(" (Run_2_seed101)", "")
        v_labels.append(f"#{r['rank']} {short_name}")
        if r["model_type"] == "optimized_model":
            v_colors.append(color_opt)
            v_edges.append('#047857')
        elif r["model_type"] == "original_model":
            v_colors.append(color_orig)
            v_edges.append('#be123c')
        else:
            v_colors.append(color_db)
            v_edges.append('#0284c7')
            
    bp2 = v_ax1.boxplot(v_box_data, patch_artist=True, widths=0.55,
                        medianprops=dict(color='#0f172a', lw=2.0),
                        flierprops=dict(marker='o', markersize=3.5, alpha=0.5, markerfacecolor='#64748b'))
    for patch, col, ecol in zip(bp2['boxes'], v_colors, v_edges):
        patch.set_facecolor(col)
        patch.set_edgecolor(ecol)
        patch.set_alpha(0.8)
        patch.set_linewidth(1.1)
        
    v_ax1.set_xticks(range(1, len(v_labels) + 1))
    v_ax1.set_xticklabels(v_labels, rotation=35, ha='right', fontsize=9.5, fontweight='500')
    v_ax1.set_ylabel("Spectral RMS Spread (reflectance units)", fontsize=11, fontweight='bold')
    v_ax1.set_title("Metameric Fiber Spread: Leading Camera Models\n(Emerald = Optimized, Rose = Original, Blue = Database)", fontsize=12, fontweight='bold')
    v_ax1.grid(True, axis='y', linestyle='--', alpha=0.35)
    
    # Right subplot: Ambiguity Reduction Waterfall / Delta relative to Original Model
    # Sort models by percentage reduction descending
    df_delta = df_cam.sort_values(by="pct_reduction_from_original", ascending=False).reset_index(drop=True)
    d_x = np.arange(len(df_delta))
    d_vals = df_delta["pct_reduction_from_original"].values
    d_colors = ['#10b981' if v > 0 else '#f43f5e' for v in d_vals]
    
    v_ax2.bar(d_x, d_vals, color=d_colors, edgecolor='#334155', width=0.72, lw=0.8, alpha=0.85)
    v_ax2.axhline(0.0, color='#0f172a', linestyle='-', lw=1.2)
    opt_delta_val = opt_row["pct_reduction_from_original"]
    v_ax2.axhline(opt_delta_val, color='#047857', linestyle='--', lw=1.8, label=f'Optimized Model Reduction: +{opt_delta_val:.2f}%')
    
    # Label key points
    v_ax2.set_xlabel("Camera Models (Ranked by Reduction vs Original)", fontsize=11, fontweight='bold')
    v_ax2.set_ylabel("Ambiguity Reduction (%) vs Original Model", fontsize=11, fontweight='bold')
    v_ax2.set_title("Percentage Ambiguity Reduction vs Original Baseline across All 30 Models\n(Green = Outperformed Original, Red = Higher Ambiguity)", fontsize=12, fontweight='bold')
    v_ax2.grid(True, axis='y', linestyle='--', alpha=0.35)
    v_ax2.set_xticks(d_x[::3])
    v_ax2.set_xticklabels([f"#{r}" for r in df_delta["rank"].values[::3]], fontsize=9)
    v_ax2.legend(loc='lower left', fontsize=10)
    
    plt.tight_layout()
    out_plot2 = os.path.join(base_dir, "plots", "camera_models_ambiguity_reduction_analysis.png")
    out_results2 = os.path.join("results", "cmf_optimization", "camera_models_ambiguity_reduction_analysis.png")
    fig2.savefig(out_plot2, dpi=200)
    fig2.savefig(out_results2, dpi=200)
    plt.close(fig2)
    print(f"Saved secondary analysis graph to:")
    print(f"  -> {out_plot2}")
    print(f"  -> {out_results2}")
    print("=" * 80)
    print("ALL GRAPHS GENERATED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    main()
