import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

images = [
    ('000681', '000681.png (PREVIOUS - Problematic)', 'Dataset/RGB/000681.png', 'Spectral/000681.npy', 'Amber-cast, high pedestal, washed-out CMFs'),
    ('000002', '000002.png (RECOMMENDED PRIMARY)', 'Dataset/RGB/000002.png', 'Spectral/000002.npy', 'Balanced RGB, neutral refs, 0% high clip'),
    ('000441', '000441.png (ALTERNATIVE 1)', 'Dataset/RGB/000441.png', 'Spectral/000441.npy', 'Vibrant tones, zero spectral pedestal, lowest MAE'),
    ('000265', '000265.png (ALTERNATIVE 2)', 'Dataset/RGB/000265.png', 'Spectral/000265.npy', 'Cool blues/cyans, zero pedestal, wide gamut')
]

fig, axes = plt.subplots(2, 4, figsize=(18, 9), dpi=150)

for col_idx, (cid, title, rgb_p, spec_p, desc) in enumerate(images):
    # Top row: Original Image
    img = Image.open(rgb_p)
    arr = np.array(img)
    cube = np.load(spec_p)
    
    ax_top = axes[0, col_idx]
    ax_top.imshow(arr)
    color_title = 'red' if cid == '000681' else '#16a34a' if cid == '000002' else '#2563eb'
    ax_top.set_title(f"Image {cid}\n{title}", fontsize=11, fontweight='bold', color=color_title)
    ax_top.axis('off')
    
    # Bottom row: Mean Spectrum across image
    ax_bot = axes[1, col_idx]
    wavelengths = np.linspace(400, 700, 31)
    mean_spec = cube.mean(axis=(0, 1))
    std_spec = cube.std(axis=(0, 1))
    min_spec = cube.min(axis=(0, 1))
    max_spec = cube.max(axis=(0, 1))
    
    ax_bot.plot(wavelengths, mean_spec, color='#0284c7', lw=2.2, label='Mean Spectrum')
    ax_bot.fill_between(wavelengths, np.maximum(0, mean_spec - std_spec), mean_spec + std_spec, color='#0284c7', alpha=0.25, label='±1 Std Dev')
    ax_bot.plot(wavelengths, min_spec, color='#94a3b8', lw=1.0, linestyle=':', label='Min / Max')
    ax_bot.plot(wavelengths, max_spec, color='#94a3b8', lw=1.0, linestyle=':')
    
    ax_bot.set_title(f"Spectral Profile (400-700 nm)\nMin: {cube.min():.0f} | Max: {cube.max():.0f}", fontsize=10)
    ax_bot.set_xlabel("Wavelength (nm)", fontsize=9)
    if col_idx == 0:
        ax_bot.set_ylabel("Radiance / Sensor Counts", fontsize=9)
        ax_bot.legend(loc='upper left', fontsize=8)
    ax_bot.grid(True, linestyle='--', alpha=0.5)

plt.suptitle("Candidate Images Inspection & Spectral Profile Comparison vs Problematic 000681.png", fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()

out_path = "scratch/candidate_contact_sheet.png"
plt.savefig(out_path, bbox_inches='tight')
plt.close()
print(f"Contact sheet saved to {out_path}")
