import numpy as np
from PIL import Image
import json

with open('scratch/candidate_evaluation.json') as f:
    data = json.load(f)

print(f"Total candidates in evaluation: {len(data['filtered'])}")

rows = []
for item in data['filtered']:
    idx = item['idx']
    cid = f"{idx:06d}"
    im = np.array(Image.open(item['rgb_path']), dtype=float)
    diff_rg = np.abs(im[:,:,0] - im[:,:,1])
    diff_gb = np.abs(im[:,:,1] - im[:,:,2])
    diff_rb = np.abs(im[:,:,0] - im[:,:,2])
    neutral_mask = (diff_rg < 25) & (diff_gb < 25) & (diff_rb < 25) & (im.mean(axis=-1) > 40)
    neutral_pct = float(np.mean(neutral_mask) * 100)
    
    rows.append({
        'cid': cid,
        'item': item,
        'neutral_pct': neutral_pct
    })

print("\n--- CANDIDATES WITH GOOD NEUTRAL CONTENT & RICH COLORS ---")
for r in rows:
    item = r['item']
    cid = r['cid']
    npct = r['neutral_pct']
    print(f"ID: {cid} | RGB mean={item['rgb_mean']:5.1f} | Std={item['rgb_std']:4.1f} | NeutralPct={npct:5.1f}% | LowClip={item['clip_low']:4.1f}% | HighClip={item['clip_high']:4.1f}% | SpecMin={item['spec_min']:5.1f} | SpecMax={item['spec_max']:6.1f} | RenderMAE={item['render_mae']:5.1f}")
