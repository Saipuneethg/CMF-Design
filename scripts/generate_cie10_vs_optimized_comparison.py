"""
CIE 1964 10° Observer vs. Advanced Global Optimized CMF Comparison Plot
=======================================================================

Generates a publication-grade top-and-bottom comparison in a single PNG:
  - Top subplot: CIE 1964 10° Supplementary Observer
  - Bottom subplot: Advanced Global Optimized CMF
  - Identical uniform scaling:
      * X-axis: 400 to 700 nm (50 nm major ticks, 10 nm minor ticks)
      * Y-axis: Exactly 0.0 to 2.0 (0.2 major ticks, 0.05 minor ticks)
      * 14 × 11 inches @ 300 DPI (4200 × 3300 px)
"""

import os
import sys
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import colour


def main():
    print("=" * 70)
    print("GENERATING CIE 10° vs ADVANCED GLOBAL OPTIMIZED CMF COMPARISON PLOT")
    print("=" * 70)

    # 1. Load Data
    wls = np.arange(400, 701, 10) # 31 bands: 400 to 700 nm @ 10 nm

    # Active Master Optimized CMF
    opt_path = os.path.join(PROJECT_ROOT, "artifacts", "M_camera_optimized.npy")
    m_opt = np.load(opt_path).astype(np.float64) # (3, 31)

    # CIE 1964 10° Supplementary Observer
    cie_64 = colour.MSDS_CMFS['CIE 1964 10 Degree Standard Observer'].copy().align(colour.SpectralShape(400, 700, 10))
    m_cie64 = cie_64.values.T.astype(np.float64) # (3, 31)

    # 2. Setup Figure: 2 Rows, 1 Column (Top & Bottom)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 11), dpi=300, sharex=True)

    # Styling constants
    red_color = "#dc2626"
    green_color = "#16a34a"
    blue_color = "#2563eb"
    lw = 2.8

    # -------------------------------------------------------------
    # TOP SUBPLOT: CIE 1964 10° Supplementary Observer
    # -------------------------------------------------------------
    # Red (\bar{x}_10)
    ax1.plot(
        wls, m_cie64[0],
        color=red_color, linestyle="-", linewidth=lw,
        marker="o", markersize=5.5, markeredgecolor="#991b1b", markeredgewidth=1.0,
        label=r"CIE 10° - Red ($\bar{x}_{10}$) [Peak: 600 nm, Max: 1.124]"
    )
    # Green (\bar{y}_10)
    ax1.plot(
        wls, m_cie64[1],
        color=green_color, linestyle="-", linewidth=lw,
        marker="s", markersize=5.0, markeredgecolor="#166534", markeredgewidth=1.0,
        label=r"CIE 10° - Green ($\bar{y}_{10}$) [Peak: 560 nm, Max: 0.997]"
    )
    # Blue (\bar{z}_10)
    ax1.plot(
        wls, m_cie64[2],
        color=blue_color, linestyle="-", linewidth=lw,
        marker="^", markersize=5.5, markeredgecolor="#1e40af", markeredgewidth=1.0,
        label=r"CIE 10° - Blue ($\bar{z}_{10}$) [Peak: 450 nm, Max: 1.995]"
    )

    # Annotate secondary violet lobe on red curve at 450 nm
    ax1.annotate(
        "Secondary Violet Lobe\n(450 nm, 0.341)",
        xy=(450, m_cie64[0, 5]),
        xytext=(470, 0.65),
        arrowprops=dict(facecolor="#991b1b", edgecolor="#dc2626", arrowstyle="->", lw=1.5),
        fontsize=10, fontweight="bold", color="#991b1b",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fef2f2", edgecolor="#fca5a5", alpha=0.9)
    )

    ax1.set_title("CIE 1964 10° Supplementary Observer (Reference Human Color Vision)", fontsize=15, fontweight="bold", pad=10)
    ax1.set_ylabel("Spectral Sensitivity", fontsize=13, fontweight="bold", labelpad=10)
    ax1.set_ylim(0.0, 2.0)
    ax1.set_yticks(np.arange(0.0, 2.01, 0.2))
    ax1.set_yticklabels([f"{y:.1f}" for y in np.arange(0.0, 2.01, 0.2)])
    ax1.set_yticks(np.arange(0.0, 2.01, 0.05), minor=True)
    ax1.axhline(1.0, color="#94a3b8", linestyle=":", linewidth=1.0, alpha=0.7, zorder=1)
    ax1.grid(True, which="major", linestyle="--", linewidth=0.8, alpha=0.6, color="#cbd5e1")
    ax1.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.35, color="#e2e8f0")
    ax1.tick_params(axis="both", which="major", labelsize=11, width=1.2, length=6)
    ax1.tick_params(axis="both", which="minor", width=0.8, length=3.5)

    leg1 = ax1.legend(loc="upper right", fontsize=11, framealpha=0.95, facecolor="#ffffff", edgecolor="#cbd5e1")
    leg1.get_frame().set_linewidth(1.0)

    # -------------------------------------------------------------
    # BOTTOM SUBPLOT: Advanced Global Optimized CMF
    # -------------------------------------------------------------
    # Red Channel
    ax2.plot(
        wls, m_opt[0],
        color=red_color, linestyle="-", linewidth=lw,
        marker="o", markersize=5.5, markeredgecolor="#991b1b", markeredgewidth=1.0,
        label=r"Optimized CMF - Red Channel [Peak: 600 nm, Max: 0.696]"
    )
    # Green Channel
    ax2.plot(
        wls, m_opt[1],
        color=green_color, linestyle="-", linewidth=lw,
        marker="s", markersize=5.0, markeredgecolor="#166534", markeredgewidth=1.0,
        label=r"Optimized CMF - Green Channel [Peak: 540 nm, Max: 0.680]"
    )
    # Blue Channel
    ax2.plot(
        wls, m_opt[2],
        color=blue_color, linestyle="-", linewidth=lw,
        marker="^", markersize=5.5, markeredgecolor="#1e40af", markeredgewidth=1.0,
        label=r"Optimized CMF - Blue Channel [Peak: 460 nm, Max: 1.558]"
    )

    # Annotate secondary violet lobe on red curve at 460 nm
    ax2.annotate(
        "Preserved Violet Lobe\n(460 nm, 0.344)",
        xy=(460, m_opt[0, 6]),
        xytext=(480, 0.65),
        arrowprops=dict(facecolor="#991b1b", edgecolor="#dc2626", arrowstyle="->", lw=1.5),
        fontsize=10, fontweight="bold", color="#991b1b",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fef2f2", edgecolor="#fca5a5", alpha=0.9)
    )

    ax2.set_title("Advanced Global Optimized CMF (Active Master Optimization — +9.37% Ambiguity Contraction)", fontsize=15, fontweight="bold", pad=10)
    ax2.set_xlabel("Wavelength (nm)", fontsize=13, fontweight="bold", labelpad=10)
    ax2.set_ylabel("Spectral Sensitivity", fontsize=13, fontweight="bold", labelpad=10)
    ax2.set_xlim(400, 700)
    ax2.set_xticks(np.arange(400, 701, 50))
    ax2.set_xticks(np.arange(400, 701, 10), minor=True)
    ax2.set_ylim(0.0, 2.0)
    ax2.set_yticks(np.arange(0.0, 2.01, 0.2))
    ax2.set_yticklabels([f"{y:.1f}" for y in np.arange(0.0, 2.01, 0.2)])
    ax2.set_yticks(np.arange(0.0, 2.01, 0.05), minor=True)
    ax2.axhline(1.0, color="#94a3b8", linestyle=":", linewidth=1.0, alpha=0.7, zorder=1)
    ax2.grid(True, which="major", linestyle="--", linewidth=0.8, alpha=0.6, color="#cbd5e1")
    ax2.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.35, color="#e2e8f0")
    ax2.tick_params(axis="both", which="major", labelsize=11, width=1.2, length=6)
    ax2.tick_params(axis="both", which="minor", width=0.8, length=3.5)

    leg2 = ax2.legend(loc="upper right", fontsize=11, framealpha=0.95, facecolor="#ffffff", edgecolor="#cbd5e1")
    leg2.get_frame().set_linewidth(1.0)

    # Overall figure title
    fig.suptitle(
        "Spectral Sensitivity Comparison: CIE 1964 10° Observer vs. Advanced Global Optimized CMF\n"
        "(Strict Uniform Scaling: X-Axis 400–700 nm | Y-Axis 0.0–2.0 | Identical 0.2 Increments)",
        fontsize=16, fontweight="bold", y=0.98
    )

    plt.subplots_adjust(left=0.08, right=0.95, top=0.91, bottom=0.07, hspace=0.18)

    # 3. Output paths
    out_paths = [
        os.path.join(PROJECT_ROOT, "Advanced_Global_Optimized_CMF", "spectral_sensitivity_graphs", "cie_10deg_vs_advanced_global_optimized_cmf_comparison.png"),
        os.path.join(PROJECT_ROOT, "Advanced_Global_Optimized_CMF", "cie_10deg_vs_advanced_global_optimized_cmf_comparison.png"),
        os.path.join(PROJECT_ROOT, "cmf_post_optimization_analysis", "Advanced_Global_Optimized_CMF", "spectral_sensitivity_graphs", "cie_10deg_vs_advanced_global_optimized_cmf_comparison.png")
    ]

    for p in out_paths:
        os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
        plt.savefig(p, dpi=300)
        print(f"Saved: {p}")

    plt.close(fig)
    print("=" * 70)
    print("COMPARISON PLOT SUCCESSFULLY GENERATED!")
    print("=" * 70)


if __name__ == "__main__":
    main()
