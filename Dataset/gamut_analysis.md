# Gamut Analysis

## Dataset Information
The dataset comprises hyperspectral images stored as `.npy` files. These files were processed iteratively to extract their spectral content, encompassing every spatial pixel across all valid files.

## Wavelength Information
The spectral range utilized is 400 nm to 700 nm in 10 nm increments. This provides 31 spectral bands. These are the verified original bands of the spectral data.

## CIE XYZ Conversion
The spectral data for each valid pixel was integrated using the standard CIE 1931 2° observer color matching functions. The integral was computed numerically across the 31 available wavelengths. 

## CIE xy Conversion
The XYZ values were converted to chromaticity coordinates `(x, y)` using standard normalization: `x = X / (X+Y+Z)` and `y = Y / (X+Y+Z)`. Pixels with zero energy or invalid (NaN/Inf) values were correctly filtered.

## sRGB Gamut
The standard sRGB triangle is defined by the primary coordinates:
- Red: (0.640, 0.330)
- Green: (0.300, 0.600)
- Blue: (0.150, 0.060)

## Dataset Gamut
The dataset gamut boundary was determined by calculating the 2D convex hull of all valid `(x, y)` coordinates from all spectral images in the dataset. This mathematically rigorous method ensures the true bounds of the dataset's spectral diversity are identified.

## Gamut Area
The area of both the sRGB standard triangle and the dataset's convex hull were calculated mathematically using polygon geometry in the CIE xy space.

## sRGB Coverage
This represents the percentage of the standard sRGB gamut that can be reproduced by the colors found in the hyperspectral dataset.

## Dataset Gamut Outside sRGB
This represents the proportion of the dataset's colors that fall outside the bounds of the standard sRGB triangle (i.e., highly saturated or wide-gamut colors).

## Numerical Results

| Metric                     | Value |
| -------------------------- | ----: |
| Number of spectral files   | 1200 |
| Total pixels               | 19660800 |
| Valid pixels               | 19660403 |
| Wavelength range           | 400-700 nm |
| Number of bands            | 31 |
| sRGB gamut area            | 0.11205 |
| Dataset gamut area         | 0.33073 |
| Intersection area          | 0.11205 |
| sRGB covered by dataset    | 100.00% |
| Dataset gamut inside sRGB  | 33.88% |
| Dataset gamut outside sRGB | 66.12% |

### Dataset Gamut Bounds
* Min x: 0.0082
* Max x: 0.7355
* Min y: 0.0048
* Max y: 0.8338

## Conclusion
The analysis demonstrates the specific chromaticity range of the dataset. Comparing the dataset's gamut to the sRGB standard reveals the extent of spectral colors that are inherently out-of-gamut for standard displays, highlighting the value of the full hyperspectral data.
