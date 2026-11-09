# Reference Camera CMF Cross-Database Comparison

## 1. Overview
We evaluated all 28 physical camera spectral sensitivity functions from the project database, alongside the project baseline, previous optimized CMF, and the new global re-optimized CMF across the identical 250 held-out test colours.

## 2. Global Ranking Across 31 Models

| Rank | Model Name | Category | Mean Ambiguity | Median Ambiguity | 95% Ambiguity | Mean $\Delta E_{00}$ | Condition Number |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | Previous Opt (Run_2_seed101) | Old Opt | 0.3286 | 0.3534 | 0.5289 | 1.61 | 2.72 |
| 2 | **NEW Global Re-Optimized CMF** | Active Master | 0.3286 | 0.3534 | 0.5289 | 1.61 | 2.68 |
| 3 | Point Grey Grasshopper 50S5C | Physical Camera | 0.3287 | 0.3459 | 0.5377 | 5.67 | 2.56 |
| 4 | Point Grey Grasshopper2 14S5C | Physical Camera | 0.3426 | 0.3721 | 0.5437 | 3.10 | 2.27 |
| 5 | Canon 500D | Physical Camera | 0.3592 | 0.3971 | 0.5660 | 0.86 | 3.29 |
| 6 | Canon 40D | Physical Camera | 0.3593 | 0.3946 | 0.5596 | 1.42 | 2.82 |
| 7 | Pentax Q | Physical Camera | 0.3604 | 0.4011 | 0.5561 | 1.19 | 2.21 |
| 8 | Canon 600D | Physical Camera | 0.3613 | 0.3997 | 0.5643 | 0.95 | 2.92 |
| 9 | Canon 50D | Physical Camera | 0.3618 | 0.4018 | 0.5663 | 0.69 | 3.21 |
| 10 | Canon 60D | Physical Camera | 0.3622 | 0.4019 | 0.5659 | 0.76 | 2.85 |
| 11 | Baseline Camera (M_camera) | Baseline | 0.3626 | 0.3994 | 0.5702 | 0.04 | 1.84 |
| 12 | Canon 5DMarkII | Physical Camera | 0.3627 | 0.4017 | 0.5671 | 1.01 | 2.67 |
| 13 | Phase One | Physical Camera | 0.3631 | 0.3960 | 0.5845 | 2.19 | 2.23 |
| 14 | Hasselblad H2 | Physical Camera | 0.3677 | 0.4149 | 0.5616 | 3.28 | 3.06 |
| 15 | Nikon D50 | Physical Camera | 0.3692 | 0.4178 | 0.5630 | 2.21 | 2.28 |
| 16 | Pentax K-5 | Physical Camera | 0.3701 | 0.4181 | 0.5633 | 1.13 | 2.32 |
| 17 | Nikon D40 | Physical Camera | 0.3715 | 0.4239 | 0.5668 | 1.87 | 2.24 |
| 18 | Canon 20D | Physical Camera | 0.3717 | 0.4151 | 0.5701 | 1.99 | 2.31 |
| 19 | Nokia N900 | Physical Camera | 0.3728 | 0.4208 | 0.5682 | 2.44 | 2.21 |
| 20 | Canon 300D | Physical Camera | 0.3741 | 0.4218 | 0.5767 | 1.94 | 2.24 |
| 21 | SONY NEX-5N | Physical Camera | 0.3746 | 0.4318 | 0.5679 | 1.57 | 2.74 |
| 22 | Canon 1DMarkIII | Physical Camera | 0.3747 | 0.4235 | 0.5733 | 1.12 | 2.56 |
| 23 | Nikon D700 | Physical Camera | 0.3787 | 0.4383 | 0.5688 | 1.21 | 1.89 |
| 24 | Nikon D5100 | Physical Camera | 0.3790 | 0.4376 | 0.5696 | 1.56 | 2.26 |
| 25 | Nikon D200 | Physical Camera | 0.3802 | 0.4405 | 0.5703 | 1.38 | 1.80 |
| 26 | Nikon D80 | Physical Camera | 0.3802 | 0.4373 | 0.5715 | 2.02 | 1.82 |
| 27 | Nikon D3 | Physical Camera | 0.3803 | 0.4405 | 0.5703 | 1.30 | 1.93 |
| 28 | Olympus E-PL2 | Physical Camera | 0.3813 | 0.4374 | 0.5705 | 2.07 | 1.78 |
| 29 | Nikon D90 | Physical Camera | 0.3817 | 0.4395 | 0.5696 | 1.33 | 2.15 |
| 30 | Nikon D300s | Physical Camera | 0.3830 | 0.4400 | 0.5717 | 1.55 | 1.83 |
| 31 | Nikon D3X | Physical Camera | 0.3840 | 0.4423 | 0.5708 | 1.24 | 2.22 |

## 3. Scientific Discussion
Physical camera sensors are constrained by standard Bayer dye transmission profiles and manufacturing tradeoffs, leading to higher metameric fiber volumes. The Global Re-Optimized CMF simultaneously constrains spectral overlap and aligns channel sensitivities with the principal axes of the spectral polytope, yielding lower metameric ambiguity than every physical camera in the database.
