import os
import glob
import numpy as np
from PIL import Image
import colour
import json

# Align CIE 1931 2 degree observer
cmfs = colour.MSDS_CMFS['CIE 1931 2 Degree Standard Observer'].copy().align(colour.SpectralShape(400, 700, 10))
cmf_vals = cmfs.values # (31, 3)

candidates = []

print("Scanning 1200 dataset images...")
for idx in range(1, 1201):
    if idx == 681:
        continue # Strictly exclude 000681
        
    rgb_path = f"Dataset/RGB/{idx:06d}.png"
    spec_path = f"Spectral/{idx:06d}.npy"
    
    if not (os.path.exists(rgb_path) and os.path.exists(spec_path)):
        continue
        
    try:
        # Load RGB
        img = Image.open(rgb_path)
        rgb = np.array(img, dtype=np.float32) # (128, 128, 3)
        
        # Load Spectral
        cube = np.load(spec_path) # (128, 128, 31)
        
        # Verify shape & data validity
        if cube.shape != (128, 128, 31):
            continue
        if np.isnan(cube).any() or np.isinf(cube).any():
            continue
            
        # Original RGB stats
        rgb_mean = float(rgb.mean())
        rgb_min = float(rgb.min())
        rgb_max = float(rgb.max())
        rgb_std = float(rgb.std())
        clip_low = float(np.mean(rgb <= 2) * 100)
        clip_high = float(np.mean(rgb >= 253) * 100)
        
        # Spectral stats
        spec_min = float(cube.min())
        spec_max = float(cube.max())
        spec_mean = float(cube.mean())
        spec_std = float(cube.std())
        
        # Spectral dynamic range / baseline ratio
        pedestal_ratio = spec_min / (spec_max + 1e-6)
        
        # CMF direct rendering to sRGB
        # Compute XYZ
        xyz = np.dot(cube, cmf_vals) # (128, 128, 3)
        xyz_max = xyz.max()
        if xyz_max <= 1e-6:
            continue
            
        # Normalization to [0, 1] relative to peak or 99.5th percentile
        xyz_norm = xyz / np.percentile(xyz, 99.8)
        xyz_norm = np.clip(xyz_norm, 0, 1)
        rgb_lin = colour.XYZ_to_sRGB(xyz_norm)
        rgb_rendered = np.clip(colour.cctf_encoding(np.clip(rgb_lin, 0, 1)), 0, 1) * 255.0
        
        # Metrics between original RGB and rendered sRGB
        mae = float(np.mean(np.abs(rgb - rgb_rendered)))
        orig_chroma = float(np.mean(np.std(rgb, axis=-1)))
        
        candidates.append({
            'idx': idx,
            'filename': f"{idx:06d}.png",
            'rgb_path': rgb_path,
            'spec_path': spec_path,
            'rgb_mean': round(rgb_mean, 2),
            'rgb_min': round(rgb_min, 2),
            'rgb_max': round(rgb_max, 2),
            'rgb_std': round(rgb_std, 2),
            'clip_low': round(clip_low, 2),
            'clip_high': round(clip_high, 2),
            'spec_min': round(spec_min, 2),
            'spec_max': round(spec_max, 2),
            'spec_mean': round(spec_mean, 2),
            'pedestal_ratio': round(pedestal_ratio, 4),
            'orig_chroma': round(orig_chroma, 2),
            'render_mae': round(mae, 2)
        })
    except Exception as e:
        continue

print(f"Successfully evaluated {len(candidates)} candidate images.")

# Sort candidates by criteria:
# 1. Good dynamic range: rgb_mean between 60 and 160, rgb_std > 30
# 2. Low clipping: clip_low < 5%, clip_high < 5%
# 3. Good spectral dynamic range: pedestal_ratio < 0.1 (low baseline offset)
# 4. Good color richness: orig_chroma > 15
# 5. Low render_mae (close rendering match to original)

filtered = [
    c for c in candidates
    if 50 <= c['rgb_mean'] <= 170
    and c['rgb_std'] >= 35
    and c['clip_low'] <= 5.0
    and c['clip_high'] <= 5.0
    and c['orig_chroma'] >= 15
    and c['pedestal_ratio'] <= 0.05
]

print(f"Candidates meeting strict criteria: {len(filtered)}")

# Sort by render_mae
filtered.sort(key=lambda x: x['render_mae'])

print("\n--- TOP 10 CANDIDATES BY RENDERING FIDELITY AND BALANCE ---")
for c in filtered[:10]:
    print(f"Image {c['idx']:06d}: RGB Mean={c['rgb_mean']}, Std={c['rgb_std']}, Chroma={c['orig_chroma']}, LowClip={c['clip_low']}%, HighClip={c['clip_high']}%, SpecMin={c['spec_min']}, SpecMax={c['spec_max']}, PedestalRatio={c['pedestal_ratio']}, RenderMAE={c['render_mae']}")

with open('scratch/candidate_evaluation.json', 'w') as f:
    json.dump({'all_count': len(candidates), 'filtered': filtered[:20]}, f, indent=2)
