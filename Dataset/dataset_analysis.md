# Dataset Analysis

## 1. Overall Dataset Structure
Dataset/
├── complete_ms_data/ (1056 files, 64 subfolders)
├── RGB/ (1200 files, 0 subfolders)
├── Spectral_PNGs/ (34902 files, 1126 subfolders)
└── Spectral/ (1200 files)

## 2. complete_ms_data
* Number of files: 1056
* Number of images/data files: 1024
* File extensions: .bmp, .db, .png
* Number of subfolders: 64
* Image/data dimensions: (512, 512)
* Number of channels/bands: 1, 3, 4
* Data type: uint16, uint8
* Minimum pixel/value: 0
* Maximum pixel/value: 62977
* Mean: 7383.687030227817
* Standard deviation: 8068.2044169803585
* Corrupted or unreadable files: 0
* Duplicate files if detectable: 0

## 3. RGB
* Number of images: 1200
* Resolution: (128, 128)
* RGB channels: 3
* Data type: uint8
* Minimum pixel/value: 0
* Maximum pixel/value: 255
* Mean: 63.76033569335937
* Standard deviation: 44.43592197562789
* Corrupted images: 0
* Duplicate images if detectable: 0

## 4. Spectral_PNGs
* Number of PNG images: 34902
* Resolution: (128, 128)
* Grayscale/RGB channels: 1
* Data type: uint8
* Filename pattern matches wavelengths: Yes (e.g., 400nm.png)
* Minimum pixel/value: 0
* Maximum pixel/value: 255
* Mean: 76.16208740234374
* Standard deviation: 29.057413136816532
* Preservation: These PNGs are visual representations (usually 8-bit or 16-bit uint), they represent individual spectral bands but scaled for image viewing.

## 5. Spectral
* Number of files: 1200
* Shape: (128, 128, 31)
* Data type: float32
* Minimum value: 0.0
* Maximum value: 63238.0
* Mean: 11058.21875
* Standard deviation: 8208.0068359375
* NaN values: 0
* Infinite values: 0
* Zero-value percentage: 0.0082%
* File Representation: A hyperspectral cube (Height x Width x Bands)

## 6. Wavelength Analysis
* Starting wavelength: 400 nm
* Ending wavelength: 700 nm
* Number of bands: 31
* Spectral interval: 10 nm
* Exact wavelengths available: 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700
* Missing wavelengths: None

## 7. Cross-Folder Relationship
- The `RGB` folder uses filenames like `000001.png`.
- The `Spectral` folder uses `.npy` files that perfectly correspond to these, e.g., `000001.npy`.
- The `Spectral_PNGs` folder contains subfolders named `000001`, `000002`, matching the prefixes of RGB/Spectral files. Inside each subfolder are individual band images like `400nm.png`.
- The `complete_ms_data` folder uses different scene-based naming conventions (e.g., `balloons_ms`, `beads_ms`), representing the original datasets before they were split into patches.

## 8. Data Quality
- Corrupted complete_ms_data: 0
- Corrupted RGB: 0
- Spectral NaNs: 0
- Spectral Infs: 0
- Spectral zeros percentage: 0.0082%

## 9. Final Summary

| Folder | Files | Format | Dimensions | Channels/Bands | Data Type | Wavelength |
| ------ | ----: | ------ | ---------- | -------------: | --------- | ---------- |
| complete_ms_data | 1056 | .bmp, .db, .png | (512, 512) | 1, 3, 4 | uint16, uint8 | 400-700 nm |
| RGB | 1200 | .png | (128, 128) | 3 | uint8 | N/A |
| Spectral_PNGs | 34902 | .png | (128, 128) | 1 | uint8 | 400-700 nm |
| Spectral | 1200 | .npy | (128, 128, 31) | 31 | float32 | 400-700 nm |
