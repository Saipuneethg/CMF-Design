# Visual Appearance and Perceptual Color Analysis: All 32 CMFs

**Experiment**: Post-Optimization Global CMF Evaluation  
**Reference Asset**: [`comparison_images/0002_all_32_cmfs_comparison.png`](file:///c:/Users/saipu/OneDrive/Desktop/fiber%20polytopes-2/cmf_post_optimization_analysis/comparison_images/0002_all_32_cmfs_comparison.png)

---

## 1. Visual Comparison Overview

The 33-panel unified comparison poster provides direct visual inspection between the original ground-truth image and all 32 CMF renderings under identical display scaling:

1. **Original Reference Image (Left Green Panel)**:
   - Natural scene displaying balanced chromaticity, neutral illumination, and wide dynamic range.
2. **Global Optimized CMF (Gold Border Card)**:
   - Preserves overall scene tonal structure, contrast, and edge sharpness.
   - Shows a subtle warm/amber tint compared to the sRGB reference, stemming from its dual-lobe red sensitivity filter centered at 603 nm and 460 nm.
   - Exhibits superior color balance compared to industrial cameras (e.g., Point Grey) which show severe blue/magenta tinting.
3. **CIE 1931 2° and CIE 1964 10° Standard Observers**:
   - Deliver neutral chromatic balance but show lower contrast due to the unadapted D65 linear XYZ-to-sRGB mapping.
4. **Physical DSLR Cameras (Canon, Nikon, Pentax, Sony)**:
   - Display very consistent midtone contrast, reflecting their consumer-tuned Bayer filter dyes.
   - Pentax K-5, Canon 5D Mark II, and Nikon D80 produce the closest visual brightness match to the sRGB input.
5. **Industrial Machine Vision Cameras (Point Grey Grasshopper, Nokia N900, Hasselblad H2)**:
   - Exhibit noticeable color shifts. Point Grey Grasshopper shifts warm tones toward yellow-green, and deep blues toward violet, visibly degrading natural skin and foliage tones.

---

## 2. Specific Artifact and Difference Observations

### A. Highlight and Shadow Fidelity
- The 99.8th percentile highlight normalization successfully preserves specular highlights across all 32 renderings without blown-out flat white patches.
- Deep shadow regions remain clean and free from negative value clipping across all physical and optimized models.

### B. Hue and Saturation Shifts
- **Blue Channel**: Cameras with narrow short-wavelength filters (e.g. Point Grey) exhibit desaturated cyan skies and over-saturated deep blues. The Global Optimized CMF exhibits rich, smooth blue transitions.
- **Green Foliage**: The Global Optimized CMF maintains vibrant greens with high structural clarity ($\text{SSIM} = 0.7967$).

---

## 3. Visual vs Numerical Metric Correlation
- **SSIM vs Visual Quality**: High SSIM ($> 0.81$) correlates strongly with preservation of texture and contour detail.
- **MAE vs $\Delta E_{00}$ Discrepancies**: In several cameras (e.g. Olympus E-PL2), MAE is relatively low ($\sim 44$), but perceptual colour error $\Delta E_{00}$ is high ($> 20$), because small numerical shifts in blue/violet produce disproportionately large perceptual hue errors to human eyes. This confirms the necessity of evaluating CIEDE2000 alongside Euclidean RGB metrics.
