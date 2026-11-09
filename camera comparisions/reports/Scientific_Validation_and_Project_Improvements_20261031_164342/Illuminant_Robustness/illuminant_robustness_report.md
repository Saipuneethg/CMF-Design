# Illuminant Robustness Investigation Report

## 1. Experimental Configuration
- **Standard Illuminants Evaluated:**
  - **CIE D50:** Mid-day daylight / graphic arts standard (Correlated Color Temperature ~5000 K).
  - **CIE D65:** Average daylight / international colorimetry standard (Correlated Color Temperature ~6500 K).
  - **CIE Illuminant A:** Incandescent tungsten filament (Correlated Color Temperature ~2856 K).
- **Wavelength Alignment:** 400 nm to 700 nm at 10 nm increments (31 spectral bands), normalized to $Y = 100$ at 560 nm.
- **Evaluation Set:** 100 diverse real-world material reflectances evaluated under identical conditions.

## 2. Verified Authoritative Results
All ambiguity reductions are calculated strictly using the ratio of aggregate means:
$$\text{Reduction (\%)} = \frac{\text{Mean Ambiguity}_{\text{Baseline}} - \text{Mean Ambiguity}_{\text{Optimized}}}{\text{Mean Ambiguity}_{\text{Baseline}}} \times 100$$

| Illuminant | Baseline Mean | Optimized Mean | Ambiguity Reduction | 95\% Conf. Interval | Paired t-stat | P-Value | Cohen's d | Win Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CIE D50** | 0.277179 | 0.250881 | **+9.49\%** | [0.0220, 0.0306] | 11.91 | $8.01 \times 10^{-21}$ | 1.191 | 92.0\% |
| **CIE D65** | 0.273236 | 0.252316 | **+7.66\%** | [0.0171, 0.0248] | 10.67 | $3.79 \times 10^{-18}$ | 1.067 | 87.0\% |
| **CIE Illuminant A** | 0.281225 | 0.242258 | **+13.86\%** | [0.0341, 0.0438] | 15.66 | $1.55 \times 10^{-28}$ | 1.566 | 97.0\% |

## 3. Physical & Mathematical Analysis of Illuminant A Advantage
- Under CIE Illuminant A (incandescent tungsten), short-wavelength blue radiant power is heavily depleted (~14% of peak at 400 nm compared to daylight).
- Standard commercial RGB cameras feature narrow, non-overlapping blue filters that suffer catastrophic signal-to-noise loss and nullspace expansion in the blue region under tungsten.
- The **Optimized CSSF** features a broader blue spectral window with calibrated spectral overlap with the green channel, retaining superior spectral constraint authority even under severe blue photon starvation.
- Consequently, the ambiguity reduction jumps to **+13.86%**, with a 97.0% per-material win rate and an exceptional effect size of $d = 1.566$.
