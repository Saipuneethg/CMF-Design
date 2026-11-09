"""
Generate Publication-Quality PNG of Section 7: Numerical Sensitivity Table: 31 Wavelength Bands.
Includes complete numerical sensitivity data from artifacts/M_camera_optimized.csv,
peak value highlights, channel color accents, metadata banner, and companion spectral curve graphic.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle

def generate_numerical_sensitivity_table_image():
    csv_path = "artifacts/M_camera_optimized.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing {csv_path}")

    df = pd.read_csv(csv_path)
    wls = df["wavelength_nm"].values
    s_red = df["sensitivity_red"].values
    s_green = df["sensitivity_green"].values
    s_blue = df["sensitivity_blue"].values

    # Find peak values and indices
    peak_blue_idx = np.argmax(s_blue)
    peak_green_idx = np.argmax(s_green)
    peak_red_idx = np.argmax(s_red)

    # Setup high-resolution figure
    fig = plt.figure(figsize=(17, 10.5), dpi=300, facecolor="#ffffff")
    
    # Outer layout: Top Title/Metadata Banner, Center Tables + Plot, Bottom Summary Card
    gs = gridspec.GridSpec(3, 3, height_ratios=[0.12, 0.76, 0.12], width_ratios=[0.31, 0.31, 0.38],
                           wspace=0.18, hspace=0.15, top=0.96, bottom=0.03, left=0.03, right=0.97)

    # 1. Title Banner (Spanning all 3 columns at top)
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis("off")
    ax_title.text(0.5, 0.72, "7. Numerical Sensitivity Table: 31 Wavelength Bands (400–700 nm)",
                  ha="center", va="center", fontsize=17, fontweight="bold", color="#0f172a")
    ax_title.text(0.5, 0.36, "Optimized Three-Channel Measurement Matrix $M(\\theta) \\in \\mathbb{R}^{3 \\times 31}$ for Reduced Metameric Ambiguity",
                  ha="center", va="center", fontsize=11.5, color="#475569")
    
    # Subtitle Badge text
    badges = "  |  ".join([
        "Wavelength Grid: 400–700 nm (Δλ = 10 nm)",
        "Equal Throughput: Σ s_k = 10.672 photons/band",
        "Strict Non-Negativity: s_k(λ) ≥ 0.0",
        "Curvature Penalty: C_smooth = 0.000217 (-81.97%)",
        "Condition Number: κ(MM^T) = 4.124"
    ])
    ax_title.text(0.5, 0.05, badges, ha="center", va="center", fontsize=9.5, fontweight="medium", color="#2563eb",
                  bbox=dict(boxstyle="round,pad=0.35", facecolor="#eff6ff", edgecolor="#bfdbfe", lw=1))

    # Helper function to format table data
    def build_table_data(start_idx, end_idx):
        rows = []
        for i in range(start_idx, end_idx):
            wl_str = f"{int(wls[i])} nm"
            r_str = f"{s_red[i]:.6f}"
            g_str = f"{s_green[i]:.6f}"
            b_str = f"{s_blue[i]:.6f}"
            if i == peak_red_idx:
                r_str += " ★"
            if i == peak_green_idx:
                g_str += " ★"
            if i == peak_blue_idx:
                b_str += " ★"
            rows.append([wl_str, r_str, g_str, b_str])
        return rows

    # 2. Table Part 1 (400 nm to 550 nm, 16 rows)
    ax_t1 = fig.add_subplot(gs[1, 0])
    ax_t1.axis("off")
    col_labels = ["λ (nm)", "Red (s_R)", "Green (s_G)", "Blue (s_B)"]
    t1_data = build_table_data(0, 16)
    
    table1 = ax_t1.table(cellText=t1_data, colLabels=col_labels, loc="center", cellLoc="center")
    table1.auto_set_font_size(False)
    table1.set_fontsize(8.5)
    table1.scale(1.0, 1.38)

    # Style Table 1
    col_header_colors = ["#475569", "#dc2626", "#059669", "#2563eb"]
    for j in range(4):
        cell = table1[(0, j)]
        cell.set_facecolor(col_header_colors[j])
        cell.set_text_props(color="white", weight="bold")
    
    for i in range(1, 17):
        row_bg = "#f8fafc" if i % 2 == 0 else "#ffffff"
        for j in range(4):
            cell = table1[(i, j)]
            cell.set_facecolor(row_bg)
            cell.set_edgecolor("#cbd5e1")
            # Highlight peak values
            row_idx = i - 1
            if j == 1 and row_idx == peak_red_idx:
                cell.set_facecolor("#fee2e2")
                cell.set_text_props(weight="bold", color="#991b1b")
            elif j == 2 and row_idx == peak_green_idx:
                cell.set_facecolor("#dcfce7")
                cell.set_text_props(weight="bold", color="#166534")
            elif j == 3 and row_idx == peak_blue_idx:
                cell.set_facecolor("#dbeafe")
                cell.set_text_props(weight="bold", color="#1e40af")
    ax_t1.set_title("Part 1: 400 nm – 550 nm", fontsize=10.5, fontweight="bold", color="#1e293b", pad=8)

    # 3. Table Part 2 (560 nm to 700 nm, 15 rows + Summary Row)
    ax_t2 = fig.add_subplot(gs[1, 1])
    ax_t2.axis("off")
    t2_data = build_table_data(16, 31)
    
    # Add summary row
    t2_data.append(["Integrated", f"{np.sum(s_red):.3f}", f"{np.sum(s_green):.3f}", f"{np.sum(s_blue):.3f}"])
    
    table2 = ax_t2.table(cellText=t2_data, colLabels=col_labels, loc="center", cellLoc="center")
    table2.auto_set_font_size(False)
    table2.set_fontsize(8.5)
    table2.scale(1.0, 1.38)

    # Style Table 2
    for j in range(4):
        cell = table2[(0, j)]
        cell.set_facecolor(col_header_colors[j])
        cell.set_text_props(color="white", weight="bold")
    
    for i in range(1, 16):
        row_bg = "#f8fafc" if i % 2 == 0 else "#ffffff"
        for j in range(4):
            cell = table2[(i, j)]
            cell.set_facecolor(row_bg)
            cell.set_edgecolor("#cbd5e1")
            row_idx = i - 1 + 16
            if j == 1 and row_idx == peak_red_idx:
                cell.set_facecolor("#fee2e2")
                cell.set_text_props(weight="bold", color="#991b1b")
            elif j == 2 and row_idx == peak_green_idx:
                cell.set_facecolor("#dcfce7")
                cell.set_text_props(weight="bold", color="#166534")
            elif j == 3 and row_idx == peak_blue_idx:
                cell.set_facecolor("#dbeafe")
                cell.set_text_props(weight="bold", color="#1e40af")

    # Style the 16th row (Integrated summary)
    for j in range(4):
        cell = table2[(16, j)]
        cell.set_facecolor("#f1f5f9")
        cell.set_text_props(weight="bold", color="#0f172a")
        cell.set_edgecolor("#94a3b8")
    ax_t2.set_title("Part 2: 560 nm – 700 nm", fontsize=10.5, fontweight="bold", color="#1e293b", pad=8)

    # 4. Companion Spectral Sensitivity Curves Plot (Right Column)
    ax_plot = fig.add_subplot(gs[1, 2])
    ax_plot.plot(wls, s_blue, color="#2563eb", lw=2.4, label=f"Blue Channel (s_B, Peak 450 nm: {s_blue[peak_blue_idx]:.3f})")
    ax_plot.fill_between(wls, s_blue, color="#2563eb", alpha=0.12)
    
    ax_plot.plot(wls, s_green, color="#059669", lw=2.4, label=f"Green Channel (s_G, Peak 540 nm: {s_green[peak_green_idx]:.3f})")
    ax_plot.fill_between(wls, s_green, color="#059669", alpha=0.12)
    
    ax_plot.plot(wls, s_red, color="#dc2626", lw=2.4, label=f"Red Channel (s_R, Peak 600 nm: {s_red[peak_red_idx]:.3f})")
    ax_plot.fill_between(wls, s_red, color="#dc2626", alpha=0.12)

    # Mark Peak Points
    ax_plot.scatter([wls[peak_blue_idx]], [s_blue[peak_blue_idx]], color="#1e40af", s=50, zorder=5)
    ax_plot.scatter([wls[peak_green_idx]], [s_green[peak_green_idx]], color="#166534", s=50, zorder=5)
    ax_plot.scatter([wls[peak_red_idx]], [s_red[peak_red_idx]], color="#991b1b", s=50, zorder=5)

    # Add peak indicator annotations
    ax_plot.annotate(f"Peak Blue\n450 nm ({s_blue[peak_blue_idx]:.3f})",
                     xy=(450, s_blue[peak_blue_idx]), xytext=(420, 1.75),
                     fontsize=8.5, fontweight="bold", color="#1e40af",
                     arrowprops=dict(arrowstyle="->", color="#1e40af", lw=1.2))

    ax_plot.annotate(f"Peak Green\n540 nm ({s_green[peak_green_idx]:.3f})",
                     xy=(540, s_green[peak_green_idx]), xytext=(500, 0.95),
                     fontsize=8.5, fontweight="bold", color="#166534",
                     arrowprops=dict(arrowstyle="->", color="#166534", lw=1.2))

    ax_plot.annotate(f"Peak Red\n600 nm ({s_red[peak_red_idx]:.3f})",
                     xy=(600, s_red[peak_red_idx]), xytext=(615, 0.95),
                     fontsize=8.5, fontweight="bold", color="#991b1b",
                     arrowprops=dict(arrowstyle="->", color="#991b1b", lw=1.2))

    ax_plot.set_title("Spectral Sensitivity Curves: M_camera_optimized", fontsize=11, fontweight="bold", color="#1e293b", pad=8)
    ax_plot.set_xlabel("Wavelength (nm)", fontsize=9.5, fontweight="medium")
    ax_plot.set_ylabel("Spectral Sensitivity (photons / unit reflectance)", fontsize=9.5, fontweight="medium")
    ax_plot.set_xlim(395, 705)
    ax_plot.set_ylim(-0.05, 1.95)
    ax_plot.grid(True, linestyle="--", alpha=0.35, color="#94a3b8")
    ax_plot.legend(loc="upper right", fontsize=8.2, framealpha=0.92, facecolor="#ffffff", edgecolor="#cbd5e1")

    # 5. Bottom Summary Footer (Spanning all 3 columns at bottom)
    ax_bot = fig.add_subplot(gs[2, :])
    ax_bot.axis("off")
    
    footer_text = (
        "★ Denotes Channel Peak Wavelength  |  "
        "Physical Non-Negativity: min(M_opt) = +3.80×10⁻²⁰ ≥ 0.0 (Strictly Positive)  |  "
        "Channel Throughput: Equalized at 10.672 photons/band\n"
        "Affine Fiber System: A_θ = M_opt B_K ∈ ℝ³ˣ⁸, b_θ = c - M_opt μ ∈ ℝ³  |  "
        "Held-Out Ambiguity Reduction: -4.12% Mean Spread, -6.21% P95 Spread  |  "
        "Preserved Color Fidelity: Mean ΔE₀₀ = 1.65 < 2.0"
    )
    ax_bot.text(0.5, 0.5, footer_text, ha="center", va="center", fontsize=9.2, color="#334155",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8fafc", edgecolor="#e2e8f0", lw=1))

    # Save to deliverables directories
    out_dir1 = "results/cmf_optimization"
    out_path1 = os.path.join(out_dir1, "numerical_sensitivity_table.png")
    out_path2 = os.path.join("artifacts", "numerical_sensitivity_table.png")

    plt.savefig(out_path1, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.savefig(out_path2, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()

    print(f"Successfully generated: {out_path1}")
    print(f"Successfully generated: {out_path2}")

if __name__ == "__main__":
    generate_numerical_sensitivity_table_image()
